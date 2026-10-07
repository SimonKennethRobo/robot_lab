# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab integration of reset, disturbance and measurement curricula."""

import json
import math
import time
from pathlib import Path

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat

from .arm_kinematics import ArmKinematics
from .robustness_math import (
    ExtensionArrivalTracker,
    contact_slip,
    fixed_hard_mask,
    gravity_wrench,
    ground_height,
    ramp,
)


def prepare_robustness_cfg(cfg):  # noqa: C901
    """Apply DR mode after CLI overrides, before assets/actuators are created."""
    settings = cfg.robustness
    cfg.commands.base_velocity_pose.include_pose_yaw = False
    cfg.observations.policy.height_error = None
    cfg.observations.critic.height_error = None
    if settings.domain_rand not in ("none", "benchmark", "sim2real"):
        raise ValueError("robustness.domain_rand must be none, benchmark or sim2real")
    if settings.steps_per_iteration <= 0 or settings.metrics_interval <= 0:
        raise ValueError("Curriculum and metrics periods must be positive")
    for name in ("reset", "arm", "push", "pose"):
        ramp(0, getattr(settings, name + "_start"), getattr(settings, name + "_end"))
    if not 0 < settings.arm_workspace_fraction <= 1:
        raise ValueError("arm_workspace_fraction must be in (0, 1]")
    if settings.arm_accel_resample_time_s <= 0:
        raise ValueError("arm_accel_resample_time_s must be positive")
    if not 0 <= settings.arm_zero_accel_probability <= 1 or not 0 <= settings.arm_zero_velocity_probability <= 1:
        raise ValueError("arm zero probabilities must be in [0, 1]")
    if settings.arm_init_joint_noise < 0:
        raise ValueError("arm_init_joint_noise must be nonnegative")
    if not 0 <= settings.arm_full_extension_standing_fraction <= 1:
        raise ValueError("arm_full_extension_standing_fraction must be in [0, 1]")
    if not 0 <= settings.nearfall_reset_fraction <= settings.hard_fraction:
        raise ValueError("Near-fall resets must be a subset of the hard cohort")
    if (
        not 0
        < settings.nearfall_reset_min_degrees
        <= settings.nearfall_reset_max_degrees
        < settings.failure_tilt_degrees
    ):
        raise ValueError("Near-fall reset angles must be positive and below failure tilt")
    gait_weights = (
        settings.gait_timing_variance_cost_weight,
        settings.gait_swing_height_cost_weight,
        settings.gait_joint_velocity_mirror_cost_weight,
        settings.gait_contact_sync_reward_weight,
        settings.standing_contact_reward_weight,
        settings.leg_velocity_balance_cost_weight,
    )
    if any(weight < 0 for weight in gait_weights):
        raise ValueError("Gait symmetry cost/reward weights must be nonnegative")
    foot_names = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
    if settings.standing_contact_reward_weight > 0:
        cfg.rewards.standing_all_feet_contact = RewTerm(
            func=standing_all_feet_contact_reward,
            weight=settings.standing_contact_reward_weight,
            params={
                "command_name": "base_velocity_pose",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True),
            },
        )
    if settings.leg_velocity_balance_cost_weight > 0:
        cfg.rewards.leg_velocity_balance = RewTerm(
            func=leg_velocity_balance_cost,
            weight=-settings.leg_velocity_balance_cost_weight,
            params={
                "command_name": "base_velocity_pose",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")
                    ],
                    preserve_order=True,
                ),
            },
        )
    if settings.gait_joint_velocity_mirror_cost_weight > 0:
        cfg.rewards.joint_velocity_mirror = RewTerm(
            func=diagonal_leg_joint_velocity_mirror_cost,
            weight=-settings.gait_joint_velocity_mirror_cost_weight,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
            },
        )
    cfg.rewards.feet_gait.weight = settings.gait_contact_sync_reward_weight
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
            "randomize_arm_link_mass",
            "randomize_com_positions",
            "randomize_arm_com_positions",
            "randomize_push_robot",
        ):
            setattr(cfg.events, name, None)


def diagonal_leg_joint_velocity_mirror_cost(
    env, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]], command_name: str = "base_velocity_pose"
):
    """Penalize unequal speed magnitudes within the two diagonal trot pairs."""
    asset = env.scene[asset_cfg.name]
    cache_name = "_diagonal_leg_joint_velocity_pairs"
    if not hasattr(env, cache_name):
        setattr(
            env,
            cache_name,
            [[asset.find_joints(pattern)[0] for pattern in pair] for pair in mirror_joints],
        )
    cost = torch.zeros(env.num_envs, device=env.device)
    for first_ids, second_ids in getattr(env, cache_name):
        first_speed = torch.abs(asset.data.joint_vel.torch[:, first_ids])
        second_speed = torch.abs(asset.data.joint_vel.torch[:, second_ids])
        cost += torch.mean(torch.square(first_speed - second_speed), dim=-1)
    cost /= max(len(mirror_joints), 1)
    # Compare only the in-phase diagonal partners, not swing versus stance legs.
    # Do not constrain load-compensating leg motions while standing with the arm out.
    cost *= env.command_manager.get_command(command_name)[:, :3].norm(dim=-1) > 0.1
    cost *= torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return cost


def standing_all_feet_contact_reward(env, command_name: str, sensor_cfg: SceneEntityCfg):
    """Strong preference for four-foot support, without prescribing equal loads."""
    command = env.command_manager.get_command(command_name)
    sensor = env.scene[sensor_cfg.name]
    contact = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    standing = command[:, :3].norm(dim=-1) < 0.1
    upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0.0, 0.7) / 0.7
    return contact.float().mean(dim=-1).pow(3) * standing * upright


def leg_velocity_balance_cost(env, command_name: str, asset_cfg: SceneEntityCfg):
    """Penalize one leg running faster than the median leg during translation."""
    asset = env.scene[asset_cfg.name]
    velocities = asset.data.joint_vel.torch[:, asset_cfg.joint_ids].reshape(env.num_envs, 4, 3)
    speed = velocities.square().mean(-1).sqrt()
    median = speed.median(dim=-1, keepdim=True).values
    imbalance = torch.tanh((speed - median).abs()).square().mean(-1)
    command = env.command_manager.get_command(command_name)
    moving = command[:, :2].norm(dim=-1) > 0.1
    upright = torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], 0.0, 0.7) / 0.7
    return imbalance * moving * upright


def terrain_height(env):
    fallback = env.scene.env_origins[:, 2]
    sensor = env.scene.sensors.get("height_scanner_base")
    if sensor is None:
        return fallback, torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return ground_height(sensor.data.ray_hits_w.torch[..., 2], fallback)


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
        "foot_width_m",
        "front_foot_width_m",
        "rear_foot_width_m",
        "swing_height_fl_m",
        "swing_height_fr_m",
        "swing_height_rl_m",
        "swing_height_rr_m",
        "swing_speed_fl_mps",
        "swing_speed_fr_mps",
        "swing_speed_rl_mps",
        "swing_speed_rr_mps",
        "air_time_fl_s",
        "air_time_fr_s",
        "air_time_rl_s",
        "air_time_rr_s",
        "arm_full_extension_case",
        "arm_full_extension_at_target",
        "arm_extension_target_error_rad",
        "arm_extension_torque_saturation",
        "arm_extension_time_to_target_s",
        "arm_extension_ee_forward_error_m",
        "arm_extension_command_tracking_error_rad",
        "standing_four_feet_contact_fraction",
        "standing_contact_count",
        "joint_speed_fl_radps",
        "joint_speed_fr_radps",
        "joint_speed_rl_radps",
        "joint_speed_rr_radps",
    )
    group_names = tuple(f"{cohort}/{age}" for cohort in ("all", "easy", "hard") for age in ("all", "early", "late"))

    def __init__(self, env):
        self.env, self.cfg = env, env.cfg.robustness
        self.robot = env.scene["robot"]
        self.hard = fixed_hard_mask(env.num_envs, self.cfg.hard_fraction, self.cfg.cohort_seed, env.device)
        self.nearfall = fixed_hard_mask(
            env.num_envs, self.cfg.nearfall_reset_fraction, self.cfg.cohort_seed, env.device
        )
        self.window_maxima = {}
        self.arm_ids = self.robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        self.leg_ids = self.robot.find_joints(
            [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")],
            preserve_order=True,
        )[0]
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
        self.extension_arrival = ExtensionArrivalTracker(env.num_envs, env.device, env.step_dt)
        self.arm_kinematics = ArmKinematics(
            self.robot.cfg.spawn.usd_path, [self.robot.joint_names[i] for i in self.arm_ids], env.device
        )
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
        self.extension_arrival.reset(env_ids)
        self.payload_mass[env_ids] = torch.rand(len(env_ids), device=self.env.device) * self.cfg.payload_max_kg
        limits = torch.tensor(self.cfg.payload_offset_m, device=self.env.device)
        self.payload_offset[env_ids] = (2 * torch.rand(len(env_ids), 3, device=self.env.device) - 1) * limits

    def apply_payload(self):
        # Refresh every physics step: the local EE offset rotates with the actual link.
        enabled = self.cfg.domain_rand != "none"
        mass = self.payload_mass * (self.progress("arm") if enabled else 0.0)
        offset = quat_apply(self.robot.data.body_quat_w.torch[:, self.ee_id], self.payload_offset)
        # PhysX applies a wrench about the link COM, while payload offsets are
        # specified from the EE link origin. Account for their difference.
        offset += self.robot.data.body_pos_w.torch[:, self.ee_id] - self.robot.data.body_com_pos_w.torch[:, self.ee_id]
        force, torque = gravity_wrench(mass, offset, -self.env.cfg.sim.gravity[2])
        self.forces[:, 0], self.torques[:, 0] = force, torque
        self.robot.permanent_wrench_composer.reset()
        self.robot.permanent_wrench_composer.add_forces_and_torques(
            self.forces, self.torques, body_ids=[self.ee_id], is_global=True
        )

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
        sensor_data = sensor.data
        applied_effort = self.robot.actuators.applied_effort.torch
        invalid = getattr(env, "_guarded_invalid_worlds", None)
        if invalid is not None:
            from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terminations import invalid_physics_state

            from ...newton_recovery import FiniteDiagnosticData, zero_invalid_rows

            invalid.copy_(invalid_physics_state(env))
            env._guarded_any = bool(invalid.any())
            if env._guarded_any:
                # The existing finite check below still checks every healthy row.
                data = FiniteDiagnosticData(data, invalid)
                sensor_data = FiniteDiagnosticData(sensor_data, invalid)
                applied_effort = zero_invalid_rows(applied_effort, invalid)
        self.check_finite(
            root=data.root_state_w.torch,
            joint_pos=data.joint_pos.torch,
            joint_vel=data.joint_vel.torch,
            contact=sensor_data.net_normal_forces_w.torch,
            torque=applied_effort,
        )
        cmd = env.command_manager.get_command("base_velocity_pose")
        height, valid_scan = terrain_height(env)
        current_height = data.root_pos_w.torch[:, 2] - height
        lin_vel = quat_apply_inverse(yaw_quat(data.root_quat_w.torch), data.root_lin_vel_w.torch)
        target_quat = quat_from_euler_xyz(cmd[:, 4], cmd[:, 5], torch.zeros_like(cmd[:, 4]))
        target_gravity = quat_apply_inverse(
            target_quat, torch.nn.functional.normalize(data.GRAVITY_VEC_W.torch, dim=-1)
        )
        tilt = (target_gravity * data.projected_gravity_b.torch).sum(-1).clamp(-1, 1).acos()
        contact = sensor_data.net_normal_forces_w_history.torch[:, :, self.contact_ids].norm(dim=-1).amax(1) > 1.0
        slip = contact_slip(data.body_lin_vel_w.torch[:, self.feet_ids], contact) / contact.sum(-1).clamp_min(1)
        arm_v = data.joint_vel.torch[:, self.arm_ids]
        arm_a = (arm_v - self.previous_arm_velocity) / env.step_dt
        arm_a *= self.velocity_valid[:, None]
        limits = data.joint_effort_limits.torch[:, self.arm_ids].clamp_min(1e-6)
        saturated = (applied_effort[:, self.arm_ids].abs() >= 0.98 * limits).float().mean(-1)
        # Velocity induced at the actual EE point by base translation and rotation.
        ee_offset = data.body_pos_w.torch[:, self.ee_id] - data.root_pos_w.torch
        ee_base_velocity = data.root_link_lin_vel_w.torch + torch.cross(data.root_ang_vel_w.torch, ee_offset, dim=-1)
        feet_in_yaw = quat_apply_inverse(
            yaw_quat(data.root_quat_w.torch).unsqueeze(1).expand(-1, len(self.feet_ids), -1),
            data.body_pos_w.torch[:, self.feet_ids] - data.root_pos_w.torch.unsqueeze(1),
        )
        front_foot_width = (feet_in_yaw[:, 0, 1] - feet_in_yaw[:, 1, 1]).abs()
        rear_foot_width = (feet_in_yaw[:, 2, 1] - feet_in_yaw[:, 3, 1]).abs()
        foot_width = 0.5 * (front_foot_width + rear_foot_width)
        swing = sensor_data.current_air_time.torch[:, self.contact_ids] > 0.0
        swing_height = (data.body_pos_w.torch[:, self.feet_ids, 2] - height.unsqueeze(-1)) * swing
        swing_speed = torch.linalg.norm(data.body_lin_vel_w.torch[:, self.feet_ids, :2], dim=-1) * swing
        air_time = sensor_data.current_air_time.torch[:, self.contact_ids]
        errors = torch.stack(
            (
                lin_vel[:, 0] - cmd[:, 0],
                lin_vel[:, 1] - cmd[:, 1],
                data.root_ang_vel_w.torch[:, 2] - cmd[:, 2],
                current_height - cmd[:, 3],
                tilt,
                slip,
                arm_v.square().mean(-1).sqrt(),
                arm_a.square().mean(-1).sqrt(),
                saturated,
                ee_base_velocity[:, 2],
                (~valid_scan).float(),
                foot_width,
                front_foot_width,
                rear_foot_width,
            ),
            dim=-1,
        )
        arm_controller = env.action_manager.get_term("joint_pos")._arm_controller
        extension_case = arm_controller.mode == 4
        extension_goal = arm_controller.commanded_extension_goal
        outbound = arm_controller.commanded_extension_outbound
        extension_error = (data.joint_pos.torch[:, self.arm_ids] - extension_goal).abs().amax(-1)
        extension_reached = outbound & (extension_error < 0.15)
        settled = extension_reached & ((arm_controller.q - extension_goal).abs().amax(-1) < 0.02)
        settled &= arm_controller.v.abs().amax(-1) < 0.05
        extension_arrival, extension_time_to_target = self.extension_arrival.update(outbound, settled)
        ee_body = quat_apply_inverse(data.root_quat_w.torch, ee_offset)
        ee_goal = self.arm_kinematics.position(extension_goal)
        ee_error = ee_body[:, 0] - ee_goal[:, 0]
        tracking_error = (data.joint_pos.torch[:, self.arm_ids] - arm_controller.q).abs().amax(-1)
        errors = torch.cat(
            (
                errors,
                swing_height,
                swing_speed,
                air_time,
                extension_case[:, None].float(),
                extension_reached[:, None].float(),
                extension_error[:, None],
                saturated[:, None],
                extension_time_to_target[:, None],
                ee_error[:, None],
                tracking_error[:, None],
            ),
            dim=-1,
        )
        standing = cmd[:, :3].norm(dim=-1) < 0.1
        current_contact = sensor_data.net_normal_forces_w.torch[:, self.contact_ids].norm(dim=-1) > 1.0
        leg_speed = data.joint_vel.torch[:, self.leg_ids].reshape(env.num_envs, 4, 3).square().mean(-1).sqrt()
        errors = torch.cat(
            (
                errors,
                current_contact.all(-1, keepdim=True).float(),
                current_contact.sum(-1, keepdim=True).float(),
                leg_speed,
            ),
            dim=-1,
        )
        if invalid is not None:
            errors = zero_invalid_rows(errors, invalid)
        self.age += env.step_dt
        cohorts = torch.stack((torch.ones_like(self.hard), ~self.hard, self.hard))
        ages = torch.stack((torch.ones_like(self.hard), self.age <= 2.0, self.age > 2.0))
        masks = (cohorts[:, None] & ages[None]).reshape(9, env.num_envs).float()
        if invalid is not None:
            masks[:, invalid] = 0.0
        self.sums += masks @ errors.abs()
        self.squares += masks @ errors.square()
        self.counts += masks.sum(-1)
        sample_valid = torch.ones_like(errors)
        sample_valid[:, 7] = self.velocity_valid.float()
        sample_valid[:, 14:26] = swing.repeat(1, 3)
        # Case fraction uses all samples; outbound diagnostics exclude return/rest.
        sample_valid[:, 27:30] = outbound[:, None]
        sample_valid[:, 30:31] = extension_arrival[:, None]
        sample_valid[:, 31:33] = outbound[:, None]
        sample_valid[:, 33:35] = standing[:, None]
        sample_valid[:, 35:39] = (~standing)[:, None]
        # Numerators and denominators must use the same conditional samples.
        self.sums -= masks @ (errors.abs() * (1 - sample_valid))
        self.squares -= masks @ (errors.square() * (1 - sample_valid))
        self.value_counts += masks @ sample_valid
        self.previous_arm_velocity.copy_(arm_v)
        self.velocity_valid[:] = True
        fallen = (-data.projected_gravity_b.torch[:, 2] < math.cos(math.radians(self.cfg.failure_tilt_degrees))) | (
            current_height < self.cfg.failure_height_m
        )
        self.failure_age = torch.where(fallen, self.failure_age + env.step_dt, 0.0)
        failed = (self.failure_age >= self.cfg.recovery_grace_s) & self.cfg.terminate_unrecovered
        if invalid is not None:
            failed |= invalid
        extension_max = torch.where(outbound, extension_error, 0.0).max()
        key = "arm_extension_joint_error_max_rad"
        self.window_maxima[key] = torch.maximum(
            self.window_maxima.get(key, torch.zeros_like(extension_max)), extension_max
        )
        return failed

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
        logs.update({"Numerics/" + key: value.clone() for key, value in self.window_maxima.items()})
        self.window_maxima.clear()
        for index, group in enumerate(self.group_names):
            prefix = "Robustness/" + group + "/"
            logs[prefix + "samples"] = self.counts[index].clone()
            logs[prefix + "arm_acceleration_samples"] = self.value_counts[index, 7].clone()
            logs[prefix + "arm_extension_outbound_samples"] = self.value_counts[index, 27].clone()
            logs[prefix + "arm_extension_arrivals"] = self.value_counts[index, 30].clone()
            for column, name in enumerate(self.metric_names):
                logs[prefix + name + "_mae"] = means[index, column]
                if column < 5 or name.endswith("foot_width_m"):
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
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids or slice(None)]
    runtime = runtime_for(env)
    runtime.update_command_ranges()
    cfg, robot = runtime.cfg, runtime.robot
    if hasattr(env, "_guarded_invalid_worlds") and robot.is_fixed_base:
        # Newton's fixed-base root setter updates its exposed root cache while
        # the constrained base body stays put. Do not randomize a fixed root.
        runtime.reset(env_ids)
        return
    state = robot.data.default_root_state.torch[env_ids].clone()
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
    if cfg.nearfall_reset_fraction > 0:
        selected = runtime.nearfall[env_ids]
        n = int(selected.sum())
        principal_axis = torch.randint(2, (n,), device=env.device)
        direction = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
        amplitude = math.radians(cfg.nearfall_reset_min_degrees) + torch.rand(n, device=env.device) * math.radians(
            cfg.nearfall_reset_max_degrees - cfg.nearfall_reset_min_degrees
        )
        near_rp = (torch.rand(n, 2, device=env.device) * 2 - 1) * math.radians(8)
        near_rp[torch.arange(n, device=env.device), principal_axis] = amplitude * direction
        state[selected, 3:7] = quat_from_euler_xyz(near_rp[:, 0], near_rp[:, 1], yaw[selected])
        omega_local = torch.zeros(n, 3, device=env.device)
        speed = cfg.nearfall_reset_angular_speed * (0.5 + 0.5 * torch.rand(n, device=env.device))
        omega_local[torch.arange(n, device=env.device), principal_axis] = direction * speed
        state[selected, 10:13] = quat_apply(yaw_quat(state[selected, 3:7]), omega_local)
    robot.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=state[:, 7:13], env_ids=env_ids)
    runtime.reset(env_ids)


def reset_locomotion_joints(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids or slice(None)]
    """Small leg noise and curriculum-scaled arm poses, only at episode reset."""
    runtime = runtime_for(env)
    robot = runtime.robot
    action = env.action_manager.get_term("joint_pos")
    positions = robot.data.default_joint_pos.torch[env_ids].clone()
    leg_ids, arm_ids = action._joint_ids, runtime.arm_ids
    positions[:, leg_ids] += (2 * torch.rand(len(env_ids), len(leg_ids), device=env.device) - 1) * 0.05
    arm = positions[:, arm_ids] + (
        (2 * torch.rand(len(env_ids), 6, device=env.device) - 1)
        * runtime.cfg.arm_init_joint_noise
        * runtime.progress("arm")
    )
    positions[:, arm_ids] = arm.clamp(action._arm_controller.lower[env_ids], action._arm_controller.upper[env_ids])
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    positions = positions.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim_index(position=positions, velocity=torch.zeros_like(positions), env_ids=env_ids)


def push_locomotion(env, env_ids):
    runtime = runtime_for(env)
    if runtime.cfg.domain_rand == "none":
        return
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids or slice(None)]
    scale = runtime.progress("push")
    delta = 2 * torch.rand(len(env_ids), 6, device=env.device) - 1
    delta[:, :2] *= runtime.cfg.push_max_xy * scale
    delta[:, 2] = 0
    delta[:, 3:] *= runtime.cfg.push_max_angular * scale
    velocity = runtime.robot.data.root_vel_w.torch[env_ids] + delta
    runtime.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def locomotion_diagnostics(env):
    return runtime_for(env).measure_and_check_failure()


def bounded_standing_yaw_penalty(env, command_name, sensor_cfg):
    """Bounded instantaneous yaw drift penalty, without hidden integral state."""
    command = env.command_manager.get_command(command_name)
    standing = command[:, :3].norm(dim=-1) < 0.1
    contact = env.scene[sensor_cfg.name].data.net_normal_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    yaw_rate = env.scene["robot"].data.root_ang_vel_w.torch[:, 2]
    return torch.tanh((yaw_rate / 0.3).square()) * standing * (contact.sum(-1) >= 2)


def track_commanded_tilt(env, command_name, std):
    """Yaw-independent tilt tracking using gravity vectors, including coupled roll/pitch."""
    command = env.command_manager.get_command(command_name)
    data = env.scene["robot"].data
    target = quat_from_euler_xyz(command[:, 4], command[:, 5], torch.zeros_like(command[:, 4]))
    gravity = quat_apply_inverse(target, torch.nn.functional.normalize(data.GRAVITY_VEC_W.torch, dim=-1))
    angle = (gravity * data.projected_gravity_b.torch).sum(-1).clamp(-1, 1).acos()
    return torch.exp(-angle.square() / std**2)
