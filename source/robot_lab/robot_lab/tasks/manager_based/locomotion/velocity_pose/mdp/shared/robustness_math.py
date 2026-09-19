# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent kernels for the RoboDuet locomotion port.

The reset cohorts and bounded joint-space disturbance follow RoboDuet
v3-stage2 (97397ad), adapted to Isaac Lab's policy/physics clocks.
"""

import math
import torch


def ramp(iteration: float, start: float, end: float) -> float:
    if end <= start:
        raise ValueError("Curriculum end must be greater than start")
    return min(1.0, max(0.0, (iteration - start) / (end - start)))


def fixed_hard_mask(num_envs: int, fraction: float, seed: int, device) -> torch.Tensor:
    """A fixed, reproducible cohort without consuming the simulation RNG."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.zeros(num_envs, dtype=torch.bool)
    mask[torch.randperm(num_envs, generator=generator)[: round(num_envs * fraction)]] = True
    return mask.to(device)


def ground_height(hits: torch.Tensor, fallback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of finite hits per environment; return validity for diagnostics."""
    valid = torch.isfinite(hits)
    count = valid.sum(-1)
    mean = torch.where(valid, hits, 0.0).sum(-1) / count.clamp_min(1)
    return torch.where(count > 0, mean, fallback), count > 0


def contact_slip(velocity_w: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    """Contact foot speed relative to static ground, in world XY (m/s)."""
    return (velocity_w[..., :2].norm(dim=-1) * contact).sum(-1)


def gravity_wrench(mass: torch.Tensor, offset_w: torch.Tensor, gravity: float = 9.81):
    force = torch.zeros_like(offset_w)
    force[..., 2] = -gravity * mass
    return force, torch.cross(offset_w, force, dim=-1)


class BoundedArmMotion:
    """Random, sinusoidal, hold, reversal and forward-extension motion cases.

    Targets are integrated with velocity/acceleration limits and anticipatory
    braking at joint limits. Reset starts at the *actual reset joint position*.
    The generator never teleports simulator joints. Limits apply to targets;
    actual joint motion is measured independently by the environment.
    """

    def __init__(
        self,
        lower,
        upper,
        dt: float,
        max_velocity=2.5,
        max_acceleration=8.0,
        fixed_mode=-1,
        reversal_fraction=0.0,
        accel_resample_time_s=0.01,
        zero_accel_probability=0.0,
        zero_velocity_probability=0.0,
        full_extension_fraction=0.0,
        full_extension_joint_pos=None,
        full_extension_hold_s=3.0,
        full_extension_max_velocity=1.5,
        full_extension_max_acceleration=3.0,
    ):
        if dt <= 0 or max_velocity <= 0 or max_acceleration <= 0:
            raise ValueError("Arm dt and limits must be positive")
        if fixed_mode not in (-1, 0, 1, 2):
            raise ValueError("Arm modes are 0=random, 1=structured, 2=hold; -1=mixed")
        if not torch.all(upper > lower):
            raise ValueError("Arm workspace has empty joint intervals")
        if not 0 <= reversal_fraction <= 1:
            raise ValueError("Arm reversal fraction must be in [0, 1]")
        if accel_resample_time_s <= 0:
            raise ValueError("Arm acceleration resample time must be positive")
        if not 0 <= zero_accel_probability <= 1 or not 0 <= zero_velocity_probability <= 1:
            raise ValueError("Arm zero probabilities must be in [0, 1]")
        if not 0 <= full_extension_fraction <= 1:
            raise ValueError("Arm full-extension fraction must be in [0, 1]")
        if any(not math.isfinite(x) or x <= 0 for x in (
            full_extension_hold_s, full_extension_max_velocity, full_extension_max_acceleration
        )):
            raise ValueError("Full-extension hold time and motion limits must be finite and positive")
        self.lower, self.upper = lower, upper
        self.dt, self.max_velocity, self.max_acceleration = dt, max_velocity, max_acceleration
        self.fixed_mode = fixed_mode
        self.reversal_fraction = reversal_fraction
        self.accel_resample_steps = max(1, int(accel_resample_time_s / dt))
        self.zero_accel_probability = zero_accel_probability
        self.zero_velocity_probability = zero_velocity_probability
        self.step_count = 0
        self.q = (lower + upper) * 0.5
        self.v = torch.zeros_like(lower)
        self.random_accel = torch.zeros_like(lower)
        self.center = self.q.clone()
        self.amplitude = torch.zeros_like(lower)
        self.phase = torch.zeros_like(lower)
        self.frequency = torch.zeros_like(lower)
        self.hold = self.q.clone()
        self.reversal_age = torch.zeros(lower.shape[0], device=lower.device)
        self.reversal_period = torch.ones_like(self.reversal_age)
        self.mode = torch.zeros(lower.shape[0], dtype=torch.long, device=lower.device)
        self.full_extension_fraction = full_extension_fraction
        self.extension_target = self.q.clone()
        if full_extension_fraction > 0:
            if full_extension_joint_pos is None:
                raise ValueError("Full-extension cases require a six-joint target")
            target = torch.as_tensor(full_extension_joint_pos, dtype=lower.dtype, device=lower.device)
            if target.shape != (lower.shape[1],) or not torch.isfinite(target).all():
                raise ValueError("Full-extension target must contain one finite angle per arm joint")
            if ((target < lower) | (target > upper)).any():
                raise ValueError("Full-extension target exceeds the arm workspace; use arm_workspace_fraction=1")
            self.extension_target[:] = target
        self.extension_rest = self.q.clone()
        self.extension_returning = torch.zeros_like(self.mode, dtype=torch.bool)
        self.extension_hold_age = torch.zeros_like(self.reversal_age)
        self.extension_hold_s = full_extension_hold_s
        self.extension_max_velocity = min(max_velocity, full_extension_max_velocity)
        self.extension_max_acceleration = min(max_acceleration, full_extension_max_acceleration)
        # Latch the goal sent this step before the phase and training clock advance.
        self.commanded_extension_goal = self.q.clone()
        self.commanded_extension_outbound = torch.zeros_like(self.mode, dtype=torch.bool)

    def extension_goal(self, intensity: float) -> torch.Tensor:
        """Return the curriculum-scaled target used by the extension controller."""
        intensity = min(1.0, max(0.0, intensity))
        goal = self.extension_rest + intensity * (self.extension_target - self.extension_rest)
        return torch.where(self.extension_returning[:, None], self.extension_rest, goal)

    def reset(self, env_ids, joint_pos):
        self.q[env_ids] = joint_pos.clamp(self.lower[env_ids], self.upper[env_ids])
        self.v[env_ids] = 0
        self.random_accel[env_ids] = 0
        u = torch.rand(len(env_ids), device=self.q.device)
        self.mode[env_ids] = (u >= 0.7).long() + (u >= 0.9).long()
        if self.fixed_mode >= 0:
            self.mode[env_ids] = self.fixed_mode
        elif self.reversal_fraction > 0:
            # Reserve a cohort for coherent, multi-axis target reversals.
            self.mode[env_ids] = torch.where(u < self.reversal_fraction, 3, self.mode[env_ids])
        if self.full_extension_fraction > 0:
            selected = torch.rand(len(env_ids), device=self.q.device) < self.full_extension_fraction
            self.mode[env_ids] = torch.where(selected, 4, self.mode[env_ids])
        self.extension_rest[env_ids] = self.q[env_ids]
        self.extension_returning[env_ids] = False
        self.extension_hold_age[env_ids] = 0.0
        self.commanded_extension_goal[env_ids] = self.q[env_ids]
        self.commanded_extension_outbound[env_ids] = False
        lower, upper = self.lower[env_ids], self.upper[env_ids]
        radius = (upper - lower) * 0.5
        self.center[env_ids] = (upper + lower) * 0.5
        self.amplitude[env_ids] = radius * (0.2 + 0.6 * torch.rand_like(lower))
        self.phase[env_ids] = 2 * math.pi * torch.rand_like(lower)
        self.frequency[env_ids] = 0.12 + 0.33 * torch.rand_like(lower)
        self.hold[env_ids] = lower + torch.rand_like(lower) * (upper - lower)
        if self.reversal_fraction > 0:
            self.reversal_age[env_ids] = 0
            self.reversal_period[env_ids] = 0.6 + 0.6 * torch.rand(len(env_ids), device=self.q.device)

    def step(self, intensity: float, actual_joint_pos=None):
        intensity = min(1.0, max(0.0, intensity))
        self.step_count += 1
        extension = self.mode == 4
        # Acceleration remains available for braking if the curriculum is reduced.
        acceleration = torch.where(extension[:, None], self.extension_max_acceleration, self.max_acceleration)
        a_dt = acceleration * self.dt
        self.phase.add_(2 * math.pi * self.frequency * self.dt * intensity)
        if self.step_count % self.accel_resample_steps == 0:
            self.random_accel = (2 * torch.rand_like(self.v) - 1) * self.max_acceleration * intensity
        step_accel = self.random_accel
        if self.zero_accel_probability > 0:
            zero = torch.rand(self.v.shape[0], 1, device=self.v.device) < self.zero_accel_probability
            step_accel = torch.where(zero, torch.zeros_like(step_accel), step_accel)
        random_v = self.v + step_accel * self.dt
        structured = self.center + self.amplitude * torch.sin(self.phase)
        target = torch.where((self.mode == 1)[:, None], structured, self.hold)
        if self.reversal_fraction > 0:
            self.reversal_age.add_(self.dt * intensity)
            direction = torch.where((self.reversal_age / self.reversal_period).floor().long() % 2 == 0, 1.0, -1.0)
            # Target changes abruptly; q/v remain limited by the same integrator.
            axis_sign = torch.where(torch.sin(self.phase) >= 0, 1.0, -1.0)
            reversal = self.center + direction[:, None] * axis_sign * self.amplitude
            target = torch.where((self.mode == 3)[:, None], reversal, target)
        extension_goal = self.extension_goal(intensity)
        self.commanded_extension_goal.copy_(extension_goal)
        self.commanded_extension_outbound = extension & ~self.extension_returning & (intensity > 0)
        target = torch.where(extension[:, None], extension_goal, target)
        # Count a hold only after the target AND measured arm have arrived.
        arrived = ((self.q - extension_goal).abs().amax(-1) < 0.02) & (self.v.abs().amax(-1) < 0.05)
        if actual_joint_pos is not None:
            arrived &= (actual_joint_pos - extension_goal).abs().amax(-1) < 0.15
        holding = extension & arrived & (intensity > 0)
        self.extension_hold_age = torch.where(holding, self.extension_hold_age + self.dt, 0.0)
        switch = extension & (self.extension_hold_age >= self.extension_hold_s - 1e-6)
        self.extension_returning ^= switch
        self.extension_hold_age[switch] = 0.0
        desired = torch.where((self.mode == 0)[:, None], random_v, 6.0 * (target - self.q))
        v_max = torch.where(extension[:, None], self.extension_max_velocity, self.max_velocity) * intensity
        desired = desired.clamp(-v_max, v_max)
        stop = None
        if self.zero_velocity_probability > 0:
            stop = torch.rand(self.v.shape[0], 1, device=self.v.device) < self.zero_velocity_probability
            stop &= ~extension[:, None]
        positive = torch.sqrt(a_dt**2 + 2 * acceleration * (self.upper - self.q).clamp_min(0)) - a_dt
        negative = torch.sqrt(a_dt**2 + 2 * acceleration * (self.q - self.lower).clamp_min(0)) - a_dt
        desired = torch.minimum(torch.maximum(desired, -negative), positive)
        self.v.add_((desired - self.v).clamp(-a_dt, a_dt))
        if stop is not None:
            self.v = torch.where(stop, torch.zeros_like(self.v), self.v)
        new_q = (self.q + self.v * self.dt).clamp(self.lower, self.upper)
        self.v = (new_q - self.q) / self.dt
        self.q = new_q
        return self.q


class ExtensionArrivalTracker:
    """One arrival per outbound phase; zero intensity and return are excluded."""

    def __init__(self, num_envs, device, dt):
        self.dt = dt
        self.elapsed = torch.zeros(num_envs, device=device)
        self.recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        self.elapsed[env_ids] = 0
        self.recorded[env_ids] = False

    def update(self, outbound, settled):
        self.elapsed = torch.where(outbound, self.elapsed + self.dt, 0.0)
        self.recorded &= outbound
        arrival = outbound & settled & ~self.recorded
        self.recorded |= arrival
        return arrival, torch.where(arrival, self.elapsed, 0.0)


class RecoveryTracker:
    """Event counts with hysteresis, a stable hold, and explicit censoring.

    Enter at tilt>=30deg, height<=0.23m, or roll/pitch speed>=1.5rad/s.
    Recover at tilt<=15deg, height>=0.26m and speed<=0.8rad/s for 0.3s.
    A 3s unresolved event is counted once and cannot rearm until stable.
    Groups distinguish reset-age<=2s from disturbances later in the episode.
    """

    fields = (
        "attempts",
        "recovered",
        "failed",
        "unresolved",
        "censored",
        "recovery_time_sum_s",
        "recovery_time_square_sum_s2",
    )

    def __init__(self, num_envs, device, dt):
        self.dt = dt
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ready = torch.ones_like(self.active)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.stable_age = torch.zeros_like(self.elapsed)
        self.origin = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.counts = torch.zeros(2, len(self.fields), device=device)

    def _add(self, mask, column, values=None):
        for group in (0, 1):
            selected = mask & (self.origin == group)
            self.counts[group, column] += selected.sum() if values is None else (selected * values).sum()

    def update(self, tilt, height, angular_speed, episode_age, failed):
        stable = (tilt <= math.radians(15)) & (height >= 0.26) & (angular_speed <= 0.8)
        self.stable_age = torch.where(stable, self.stable_age + self.dt, 0.0)
        held = self.stable_age >= 0.3 - 1e-6
        self.ready |= held & ~self.active
        danger = (tilt >= math.radians(30)) | (height <= 0.23) | (angular_speed >= 1.5)
        started = danger & self.ready & ~self.active
        self.origin[started] = (episode_age[started] > 2.0).long()
        self._add(started, 0)
        self.active |= started
        self.ready[started] = False
        self.elapsed[started] = 0
        self.elapsed += self.active * self.dt
        recovered = self.active & held & ~failed
        terminal = self.active & failed
        unresolved = self.active & (self.elapsed >= 3.0 - 1e-6) & ~recovered & ~terminal
        self._add(recovered, 1)
        self._add(terminal, 2)
        self._add(unresolved, 3)
        self._add(recovered, 5, self.elapsed)
        self._add(recovered, 6, self.elapsed.square())
        self.active &= ~(recovered | terminal | unresolved)
        self.ready |= recovered

    def reset(self, env_ids, failed=None):
        selected = torch.zeros_like(self.active)
        selected[env_ids] = self.active[env_ids]
        terminal = torch.zeros_like(self.active)
        if failed is not None:
            terminal[env_ids] = failed
        self._add(selected & terminal, 2)
        self._add(selected & ~terminal, 4)
        self.active[env_ids] = False
        self.ready[env_ids] = True
        self.elapsed[env_ids] = 0
        self.stable_age[env_ids] = 0

    def flush(self):
        logs = {}
        for group, name in ((0, "reset"), (1, "rollout")):
            for column, field in enumerate(self.fields):
                logs[f"Recovery/{name}/{field}"] = self.counts[group, column].clone()
            logs[f"Recovery/{name}/active"] = (self.active & (self.origin == group)).sum().float()
        self.counts.zero_()
        return logs
