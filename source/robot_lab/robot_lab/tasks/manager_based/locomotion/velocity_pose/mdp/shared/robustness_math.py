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
    """Three tensorized modes: random acceleration, sinusoidal target, pose hold.

    Targets are integrated with velocity/acceleration limits and anticipatory
    braking at joint limits. Reset starts at the *actual reset joint position*.
    The generator never teleports simulator joints. Limits apply to targets;
    actual joint motion is measured independently by the environment.
    """

    def __init__(self, lower, upper, dt: float, max_velocity=2.5, max_acceleration=8.0, fixed_mode=-1):
        if dt <= 0 or max_velocity <= 0 or max_acceleration <= 0:
            raise ValueError("Arm dt and limits must be positive")
        if fixed_mode not in (-1, 0, 1, 2):
            raise ValueError("Arm modes are 0=random, 1=structured, 2=hold; -1=mixed")
        if not torch.all(upper > lower):
            raise ValueError("Arm workspace has empty joint intervals")
        self.lower, self.upper = lower, upper
        self.dt, self.max_velocity, self.max_acceleration = dt, max_velocity, max_acceleration
        self.fixed_mode = fixed_mode
        self.q = (lower + upper) * 0.5
        self.v = torch.zeros_like(lower)
        self.center = self.q.clone()
        self.amplitude = torch.zeros_like(lower)
        self.phase = torch.zeros_like(lower)
        self.frequency = torch.zeros_like(lower)
        self.hold = self.q.clone()
        self.mode = torch.zeros(lower.shape[0], dtype=torch.long, device=lower.device)

    def reset(self, env_ids, joint_pos):
        self.q[env_ids] = joint_pos.clamp(self.lower[env_ids], self.upper[env_ids])
        self.v[env_ids] = 0
        u = torch.rand(len(env_ids), device=self.q.device)
        self.mode[env_ids] = (u >= 0.7).long() + (u >= 0.9).long()
        if self.fixed_mode >= 0:
            self.mode[env_ids] = self.fixed_mode
        lower, upper = self.lower[env_ids], self.upper[env_ids]
        radius = (upper - lower) * 0.5
        self.center[env_ids] = (upper + lower) * 0.5
        self.amplitude[env_ids] = radius * (0.2 + 0.6 * torch.rand_like(lower))
        self.phase[env_ids] = 2 * math.pi * torch.rand_like(lower)
        self.frequency[env_ids] = 0.12 + 0.33 * torch.rand_like(lower)
        self.hold[env_ids] = lower + torch.rand_like(lower) * (upper - lower)

    def step(self, intensity: float):
        intensity = min(1.0, max(0.0, intensity))
        # Acceleration remains available for braking if the curriculum is reduced.
        a_dt = self.max_acceleration * self.dt
        self.phase.add_(2 * math.pi * self.frequency * self.dt * intensity)
        random_v = self.v + (2 * torch.rand_like(self.v) - 1) * a_dt * intensity
        structured = self.center + self.amplitude * torch.sin(self.phase)
        target = torch.where((self.mode == 1)[:, None], structured, self.hold)
        desired = torch.where((self.mode == 0)[:, None], random_v, 6.0 * (target - self.q))
        v_max = self.max_velocity * intensity
        desired = desired.clamp(-v_max, v_max)
        positive = torch.sqrt(a_dt**2 + 2 * self.max_acceleration * (self.upper - self.q).clamp_min(0)) - a_dt
        negative = torch.sqrt(a_dt**2 + 2 * self.max_acceleration * (self.q - self.lower).clamp_min(0)) - a_dt
        desired = torch.minimum(torch.maximum(desired, -negative), positive)
        self.v.add_((desired - self.v).clamp(-a_dt, a_dt))
        new_q = (self.q + self.v * self.dt).clamp(self.lower, self.upper)
        self.v = (new_q - self.q) / self.dt
        self.q = new_q
        return self.q
