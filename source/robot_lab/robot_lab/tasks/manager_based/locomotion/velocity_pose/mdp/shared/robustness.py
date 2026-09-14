# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab integration of reset, disturbance and measurement curricula."""

import json
import math
import time
import torch
from pathlib import Path

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat

from .robustness_math import contact_slip, fixed_hard_mask, gravity_wrench, ground_height, ramp


def prepare_robustness_cfg(cfg):
    """Apply DR mode after CLI overrides, before assets/actuators are created."""
    settings = cfg.robustness
    if settings.domain_rand not in ("none", "benchmark", "sim2real"):
        raise ValueError("robustness.domain_rand must be none, benchmark or sim2real")
    if settings.steps_per_iteration <= 0 or settings.metrics_interval <= 0:
        raise ValueError("Curriculum and metrics periods must be positive")
    for name in ("reset", "arm", "push", "pose"):
        ramp(0, getattr(settings, name + "_start"), getattr(settings, name + "_end"))
    if not 0 < settings.arm_workspace_fraction <= 1:
        raise ValueError("arm_workspace_fraction must be in (0, 1]")
    if settings.domain_rand != "sim2real":
        cfg.observations.policy.enable_corruption = False
        cfg.events.randomize_actuator_gains = None
        cfg.events.randomize_arm_gains = None
        for actuator in cfg.scene.robot.actuators.values():
            if hasattr(actuator, "min_delay"):
                actuator.min_delay = actuator.max_delay = 0
    if settings.domain_rand == "none":
        for name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_others",
            "randomize_com_positions",
            "randomize_push_robot",
        ):
            setattr(cfg.events, name, None)


def terrain_height(env):
    fallback = env.scene.env_origins[:, 2]
    sensor = env.scene.sensors.get("height_scanner_base")
    if sensor is None:
        return fallback, torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return ground_height(sensor.data.ray_hits_w[..., 2], fallback)


class LocomotionRuntime:
    metric_names = (
        "vx_error_mps",
        "vy_error_mps",
        "yaw_rate_error_radps",
        "height_error_m",
        "tilt_error_rad",
        "contact_slip_mps",
        "arm_velocity_radps",
        "arm_acceleration_radps2",
        "arm_torque_saturation",
        "ee_base_vz_mps",
        "invalid_height_scan",
    )
    group_names = tuple(f"{cohort}/{age}" for cohort in ("all", "easy", "hard") for age in ("all", "early", "late"))

    def __init__(self, env):
        self.env, self.cfg = env, env.cfg.robustness
        self.robot = env.scene["robot"]
        self.hard = fixed_hard_mask(env.num_envs, self.cfg.hard_fraction, self.cfg.cohort_seed, env.device)
        self.arm_ids = self.robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        feet = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
        self.feet_ids = self.robot.find_bodies(feet, preserve_order=True)[0]
        self.contact_ids = env.scene["contact_forces"].find_bodies(feet, preserve_order=True)[0]
        self.ee_id = self.robot.find_bodies("link6")[0][0]
        self.step_offset = 0
        self.payload_mass = torch.zeros(env.num_envs, device=env.device)
        self.payload_offset = torch.zeros(env.num_envs, 3, device=env.device)
        self.failure_age = torch.zeros(env.num_envs, device=env.device)
        self.age = torch.zeros_like(self.failure_age)
        self.previous_arm_velocity = torch.zeros(env.num_envs, 6, device=env.device)
        self.velocity_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_metrics = {}
        self.sums = torch.zeros(9, len(self.metric_names), device=env.device)
        self.squares = torch.zeros_like(self.sums)
        self.value_counts = torch.zeros_like(self.sums)
        self.counts = torch.zeros(9, device=env.device)
        self.episodes = torch.zeros(3, 2, device=env.device)  # completed, failed
        self.forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self.torques = torch.zeros_like(self.forces)

    @property
    def iteration(self):
        if self.cfg.iteration_override >= 0:
            return self.cfg.iteration_override
        return (self.env.common_step_counter + self.step_offset) / self.cfg.steps_per_iteration

    def progress(self, name):
        return ramp(self.iteration, getattr(self.cfg, name + "_start"), getattr(self.cfg, name + "_end"))

    def update_command_ranges(self):
        command = self.env.command_manager.get_term("base_velocity_pose")
        progress = self.progress("pose")
        h = command.cfg.default_height
        command.cfg.ranges.height = tuple(h + (x - h) * progress for x in self.cfg.height_range)
        command.cfg.ranges.roll = tuple(x * progress for x in self.cfg.roll_range)
        command.cfg.ranges.pitch = tuple(x * progress for x in self.cfg.pitch_range)

    def reset(self, env_ids):
        self.failure_age[env_ids] = 0
        self.age[env_ids] = 0
        self.velocity_valid[env_ids] = False
        self.payload_mass[env_ids] = torch.rand(len(env_ids), device=self.env.device) * self.cfg.payload_max_kg
        limits = torch.tensor(self.cfg.payload_offset_m, device=self.env.device)
        self.payload_offset[env_ids] = (2 * torch.rand(len(env_ids), 3, device=self.env.device) - 1) * limits

    def apply_payload(self):
        # Refresh every physics step: the local EE offset rotates with the actual link.
        enabled = self.cfg.domain_rand != "none"
        mass = self.payload_mass * (self.progress("arm") if enabled else 0.0)
        offset = quat_apply(self.robot.data.body_quat_w[:, self.ee_id], self.payload_offset)
        # PhysX applies a wrench about the link COM, while payload offsets are
        # specified from the EE link origin. Account for their difference.
        offset += self.robot.data.body_pos_w[:, self.ee_id] - self.robot.data.body_com_pos_w[:, self.ee_id]
        force, torque = gravity_wrench(mass, offset, -self.env.cfg.sim.gravity[2])
        self.forces[:, 0], self.torques[:, 0] = force, torque
        self.robot.set_external_force_and_torque(self.forces, self.torques, body_ids=[self.ee_id], is_global=True)

    def check_finite(self, **values):
        # One device synchronization for the whole batch, instead of one per field.
        finite = torch.stack(
            [torch.isfinite(value).reshape(self.env.num_envs, -1).all(-1) for value in values.values()], dim=-1
        )
        if not finite.all():
            for column, name in enumerate(values):
                if finite[:, column].all():
                    continue
                ids = (~finite[:, column]).nonzero(as_tuple=False).flatten()
                log_dir = Path(getattr(self.env.cfg, "log_dir", None) or "outputs/locomotion_diagnostics")
                log_dir.mkdir(parents=True, exist_ok=True)
                path = log_dir / f"numerical_fault_{time.time_ns()}.pt"
                torch.save(
                    {
                        "field": name,
                        "env_ids": ids.cpu(),
                        "step": self.env.common_step_counter,
                        "iteration": self.iteration,
                        "values": {key: val[ids].detach().cpu() for key, val in values.items()},
                    },
                    path,
                )
                raise FloatingPointError(f"Non-finite {name}; stopped before PPO receives invalid data. Dump: {path}")

    def measure_and_check_failure(self):
        """Called by termination manager BEFORE rewards, command resampling and auto-reset."""
        env, data = self.env, self.robot.data
        sensor = env.scene["contact_forces"]
        self.check_finite(
            root=data.root_state_w,
            joint_pos=data.joint_pos,
            joint_vel=data.joint_vel,
            contact=sensor.data.net_forces_w,
            torque=data.applied_torque,
        )
        cmd = env.command_manager.get_command("base_velocity_pose")
        height, valid_scan = terrain_height(env)
        current_height = data.root_pos_w[:, 2] - height
        lin_vel = quat_apply_inverse(yaw_quat(data.root_quat_w), data.root_lin_vel_w)
        target_quat = quat_from_euler_xyz(cmd[:, 4], cmd[:, 5], torch.zeros_like(cmd[:, 4]))
        target_gravity = quat_apply_inverse(target_quat, data.GRAVITY_VEC_W)
        tilt = (target_gravity * data.projected_gravity_b).sum(-1).clamp(-1, 1).acos()
        contact = sensor.data.net_forces_w_history[:, :, self.contact_ids].norm(dim=-1).amax(1) > 1.0
        slip = contact_slip(data.body_lin_vel_w[:, self.feet_ids], contact) / contact.sum(-1).clamp_min(1)
        arm_v = data.joint_vel[:, self.arm_ids]
        arm_a = (arm_v - self.previous_arm_velocity) / env.step_dt
        arm_a *= self.velocity_valid[:, None]
        limits = data.joint_effort_limits[:, self.arm_ids].clamp_min(1e-6)
        saturated = (data.applied_torque[:, self.arm_ids].abs() >= 0.98 * limits).float().mean(-1)
        # Velocity induced at the actual EE point by base translation and rotation.
        ee_offset = data.body_pos_w[:, self.ee_id] - data.root_pos_w
        ee_base_velocity = data.root_link_lin_vel_w + torch.cross(data.root_ang_vel_w, ee_offset, dim=-1)
        errors = torch.stack(
            (
                lin_vel[:, 0] - cmd[:, 0],
                lin_vel[:, 1] - cmd[:, 1],
                data.root_ang_vel_w[:, 2] - cmd[:, 2],
                current_height - cmd[:, 3],
                tilt,
                slip,
                arm_v.square().mean(-1).sqrt(),
                arm_a.square().mean(-1).sqrt(),
                saturated,
                ee_base_velocity[:, 2],
                (~valid_scan).float(),
            ),
            dim=-1,
        )
        self.age += env.step_dt
        cohorts = torch.stack((torch.ones_like(self.hard), ~self.hard, self.hard))
        ages = torch.stack((torch.ones_like(self.hard), self.age <= 2.0, self.age > 2.0))
        masks = (cohorts[:, None] & ages[None]).reshape(9, env.num_envs).float()
        self.sums += masks @ errors.abs()
        self.squares += masks @ errors.square()
        self.counts += masks.sum(-1)
        sample_valid = torch.ones_like(errors)
        sample_valid[:, 7] = self.velocity_valid.float()
        self.value_counts += masks @ sample_valid
        self.previous_arm_velocity.copy_(arm_v)
        self.velocity_valid[:] = True
        fallen = (-data.projected_gravity_b[:, 2] < math.cos(math.radians(self.cfg.failure_tilt_degrees))) | (
            current_height < self.cfg.failure_height_m
        )
        self.failure_age = torch.where(fallen, self.failure_age + env.step_dt, 0.0)
        return (self.failure_age >= self.cfg.recovery_grace_s) & self.cfg.terminate_unrecovered

    def record_completed(self, env_ids):
        # Initial explicit reset is not a completed episode; PPO randomized episode
        # lengths do not affect the independently tracked early/late sample age.
        valid = self.age[env_ids] > 0
        hard = self.hard[env_ids]
        masks = torch.stack((valid, valid & ~hard, valid & hard)).float()
        self.episodes[:, 0] += masks.sum(-1)
        self.episodes[:, 1] += masks @ self.env.reset_terminated[env_ids].float()

    def flush_metrics(self):
        counts = self.value_counts.clamp_min(1)
        means, rms = self.sums / counts, (self.squares / counts).sqrt()
        logs = {}
        for index, group in enumerate(self.group_names):
            prefix = "Robustness/" + group + "/"
            logs[prefix + "samples"] = self.counts[index].clone()
            logs[prefix + "arm_acceleration_samples"] = self.value_counts[index, 7].clone()
            for column, name in enumerate(self.metric_names):
                logs[prefix + name + "_mae"] = means[index, column]
                if column < 5:
                    logs[prefix + name + "_rmse"] = rms[index, column]
        for index, group in enumerate(("all", "easy", "hard")):
            logs[f"Robustness/{group}/completed"] = self.episodes[index, 0].clone()
            logs[f"Robustness/{group}/failure_fraction"] = self.episodes[index, 1] / self.episodes[index, 0].clamp_min(
                1
            )
        logs["Curriculum/locomotion_iteration"] = self.iteration
        for name in ("reset", "arm", "push", "pose"):
            logs["Curriculum/" + name + "_intensity"] = self.progress(name)
        logs["Robustness/payload_mean_kg"] = (
            self.payload_mass.mean() * self.progress("arm") * (self.cfg.domain_rand != "none")
        )
        self.sums.zero_()
        self.squares.zero_()
        self.counts.zero_()
        self.value_counts.zero_()
        self.episodes.zero_()
        self.previous_metrics = logs
        # Preserve actual sample counts and window boundaries independently of
        # the training logger's averaging of episode-info dictionaries.
        directory = Path(getattr(self.env.cfg, "log_dir", None) or "outputs/locomotion_diagnostics")
        directory.mkdir(parents=True, exist_ok=True)
        scalars = torch.stack([torch.as_tensor(value, device=self.env.device) for value in logs.values()])
        record = dict(zip(logs, scalars.detach().cpu().tolist()))
        record["policy_steps"] = self.env.common_step_counter + self.step_offset
        with (directory / "locomotion_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return logs


def runtime_for(env):
    if not hasattr(env, "_locomotion"):
        env._locomotion = LocomotionRuntime(env)
    return env._locomotion


def reset_locomotion_root(env, env_ids):
    runtime = runtime_for(env)
    runtime.update_command_ranges()
    cfg, robot = runtime.cfg, runtime.robot
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, :2] += (torch.rand(len(env_ids), 2, device=env.device) - 0.5) * 0.2
    state[:, 2] += torch.rand(len(env_ids), device=env.device) * 0.03
    easy = math.radians(cfg.easy_reset_degrees)
    hard = easy + (math.radians(cfg.hard_reset_degrees) - easy) * runtime.progress("reset")
    angle_limit = torch.where(runtime.hard[env_ids], hard, easy)
    rp = (2 * torch.rand(len(env_ids), 2, device=env.device) - 1) * angle_limit[:, None]
    yaw = (2 * torch.rand(len(env_ids), device=env.device) - 1) * math.pi
    state[:, 3:7] = quat_from_euler_xyz(rp[:, 0], rp[:, 1], yaw)
    state[:, 7:13] = (2 * torch.rand(len(env_ids), 6, device=env.device) - 1) * 0.1
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)
    runtime.reset(env_ids)


def reset_locomotion_joints(env, env_ids):
    """Small leg noise and curriculum-scaled arm poses, only at episode reset."""
    runtime = runtime_for(env)
    robot = runtime.robot
    action = env.action_manager.get_term("joint_pos")
    positions = robot.data.default_joint_pos[env_ids].clone()
    leg_ids, arm_ids = action._joint_ids, runtime.arm_ids
    positions[:, leg_ids] += (2 * torch.rand(len(env_ids), len(leg_ids), device=env.device) - 1) * 0.05
    arm = positions[:, arm_ids] + (2 * torch.rand(len(env_ids), 6, device=env.device) - 1) * 0.25 * runtime.progress(
        "arm"
    )
    positions[:, arm_ids] = arm.clamp(action._arm_controller.lower[env_ids], action._arm_controller.upper[env_ids])
    limits = robot.data.soft_joint_pos_limits[env_ids]
    positions = positions.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=env_ids)


def push_locomotion(env, env_ids):
    runtime = runtime_for(env)
    if runtime.cfg.domain_rand == "none":
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    scale = runtime.progress("push")
    delta = 2 * torch.rand(len(env_ids), 6, device=env.device) - 1
    delta[:, :2] *= runtime.cfg.push_max_xy * scale
    delta[:, 2] = 0
    delta[:, 3:] *= runtime.cfg.push_max_angular * scale
    velocity = runtime.robot.data.root_vel_w[env_ids] + delta
    runtime.robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def locomotion_diagnostics(env):
    return runtime_for(env).measure_and_check_failure()


def bounded_standing_yaw_penalty(env, command_name, sensor_cfg):
    """Bounded instantaneous yaw drift penalty, without hidden integral state."""
    command = env.command_manager.get_command(command_name)
    standing = command[:, :3].norm(dim=-1) < 0.1
    contact = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    yaw_rate = env.scene["robot"].data.root_ang_vel_w[:, 2]
    return torch.tanh((yaw_rate / 0.3).square()) * standing * (contact.sum(-1) >= 2)


def track_commanded_tilt(env, command_name, std):
    """Yaw-independent tilt tracking using gravity vectors, including coupled roll/pitch."""
    command = env.command_manager.get_command(command_name)
    data = env.scene["robot"].data
    target = quat_from_euler_xyz(command[:, 4], command[:, 5], torch.zeros_like(command[:, 4]))
    gravity = quat_apply_inverse(target, data.GRAVITY_VEC_W)
    angle = (gravity * data.projected_gravity_b).sum(-1).clamp(-1, 1).acos()
    return torch.exp(-angle.square() / std**2)
