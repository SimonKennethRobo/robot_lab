# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab integration of reset, disturbance and measurement curricula."""

import json
import math
import time
import torch
from pathlib import Path

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import (
    feet_air_time_variance_penalty,
    feet_distance_y_exp,
    feet_height_body,
)

from .robustness_math import RecoveryTracker, contact_slip, fixed_hard_mask, gravity_wrench, ground_height, ramp


def prepare_robustness_cfg(cfg):  # noqa: C901
    """Apply DR mode after CLI overrides, before assets/actuators are created."""
    from ...recovery_recipes import apply_recovery_recipe
    from ...observation_contract import LAYOUTS

    settings = cfg.robustness
    apply_recovery_recipe(settings)
    layouts = tuple(LAYOUTS)
    if settings.observation_layout not in layouts:
        raise ValueError(f"robustness.observation_layout must be one of {layouts}")
    include_yaw, include_height = LAYOUTS[settings.observation_layout]
    cfg.commands.base_velocity_pose.include_pose_yaw = include_yaw
    if include_height:
        term = ObsTerm(func=terrain_relative_height_error, clip=(-0.2, 0.2))
        cfg.observations.policy.height_error = term
        cfg.observations.critic.height_error = ObsTerm(func=terrain_relative_height_error, clip=(-0.2, 0.2))
    else:
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
    if settings.nearfall_reset_angular_speed < 0 or not 0 < settings.recovery_action_rate_scale <= 1:
        raise ValueError("Invalid recovery angular-speed or action-rate scale")
    if settings.recovery_upright_weight < 0:
        raise ValueError("Recovery upright cost weight must be nonnegative")
    if settings.stance_width_target_m < 0 or settings.stance_width_reward_weight < 0:
        raise ValueError("Stance-width target and reward weight must be nonnegative")
    if settings.stance_width_reward_weight > 0 and (
        settings.stance_width_target_m <= 0 or settings.stance_width_std_m <= 0
    ):
        raise ValueError("Enabled stance-width reward requires positive target and std")
    if settings.rear_stance_width_target_m < 0 or settings.rear_stance_width_reward_weight < 0:
        raise ValueError("Rear stance-width target and reward weight must be nonnegative")
    if settings.rear_stance_width_reward_weight > 0 and (
        settings.rear_stance_width_target_m <= 0 or settings.rear_stance_width_std_m <= 0
    ):
        raise ValueError("Enabled rear stance-width reward requires positive target and std")
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
    if settings.gait_swing_height_cost_weight > 0 and settings.gait_swing_height_body_target_m >= 0:
        raise ValueError("Enabled swing-height cost requires a negative body-frame foot-height target")
    if settings.recovery_action_rate_scale < 1:
        cfg.rewards.action_rate_l2.func = recovery_action_rate_l2
    if settings.recovery_upright_weight > 0:
        cfg.rewards.recovery_upright = RewTerm(func=recovery_upright_cost, weight=-settings.recovery_upright_weight)
    if settings.stance_width_reward_weight > 0:
        cfg.rewards.feet_distance_y_exp = RewTerm(
            func=feet_distance_y_exp,
            weight=settings.stance_width_reward_weight,
            params={
                "stance_width": settings.stance_width_target_m,
                "std": settings.stance_width_std_m,
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=[f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")], preserve_order=True
                ),
            },
        )
    if settings.rear_stance_width_reward_weight > 0:
        cfg.rewards.rear_feet_distance_y_exp = RewTerm(
            func=feet_distance_y_exp,
            weight=settings.rear_stance_width_reward_weight,
            params={
                "stance_width": settings.rear_stance_width_target_m,
                "std": settings.rear_stance_width_std_m,
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=[f"{leg}_foot" for leg in ("RL", "RR")], preserve_order=True
                ),
            },
        )
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
                        f"{leg}_{joint}_joint"
                        for leg in ("FL", "FR", "RL", "RR")
                        for joint in ("hip", "thigh", "calf")
                    ],
                    preserve_order=True,
                ),
            },
        )
    if settings.gait_timing_variance_cost_weight > 0:
        cfg.rewards.feet_air_time_variance = RewTerm(
            func=feet_air_time_variance_penalty,
            weight=-settings.gait_timing_variance_cost_weight,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)},
        )
    if settings.gait_swing_height_cost_weight > 0:
        cfg.rewards.feet_height_body = RewTerm(
            func=feet_height_body,
            weight=-settings.gait_swing_height_cost_weight,
            params={
                "command_name": "base_velocity_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_names, preserve_order=True),
                "target_height": settings.gait_swing_height_body_target_m,
                "tanh_mult": 2.0,
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
    """Penalize unequal speed magnitudes within the two diagonal trot pairs.

    Unlike the generic action-mirror term, this indexes the articulation's joint
    velocity tensor and is therefore valid when the 12D leg action is attached to
    an 18-joint quadruped-manipulator articulation.
    """
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
        first_speed = torch.abs(asset.data.joint_vel[:, first_ids])
        second_speed = torch.abs(asset.data.joint_vel[:, second_ids])
        cost += torch.mean(torch.square(first_speed - second_speed), dim=-1)
    cost /= max(len(mirror_joints), 1)
    # Compare only the in-phase diagonal partners, not swing versus stance legs.
    # Do not constrain load-compensating leg motions while standing with the arm out.
    cost *= env.command_manager.get_command(command_name)[:, :3].norm(dim=-1) > 0.1
    cost *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return cost


def standing_all_feet_contact_reward(env, command_name: str, sensor_cfg: SceneEntityCfg):
    """Strong preference for four-foot support, without prescribing equal loads.

    Cubing the contact fraction gives 1 for four feet and 0.422 for three.
    Current forces (not contact history) prevent brief taps earning full support.
    Height/tilt commands remain allowed; only locomotion commands gate standing.
    """
    command = env.command_manager.get_command(command_name)
    sensor = env.scene[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    standing = command[:, :3].norm(dim=-1) < 0.1
    upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return contact.float().mean(dim=-1).pow(3) * standing * upright


def leg_velocity_balance_cost(env, command_name: str, asset_cfg: SceneEntityCfg):
    """Penalize one leg running faster than the median leg during translation."""
    asset = env.scene[asset_cfg.name]
    velocities = asset.data.joint_vel[:, asset_cfg.joint_ids].reshape(env.num_envs, 4, 3)
    speed = velocities.square().mean(-1).sqrt()
    median = speed.median(dim=-1, keepdim=True).values
    imbalance = torch.tanh((speed - median).abs()).square().mean(-1)
    command = env.command_manager.get_command(command_name)
    moving = command[:, :2].norm(dim=-1) > 0.1
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return imbalance * moving * upright


def terrain_height(env):
    fallback = env.scene.env_origins[:, 2]
    sensor = env.scene.sensors.get("height_scanner_base")
    if sensor is None:
        return fallback, torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return ground_height(sensor.data.ray_hits_w[..., 2], fallback)


def terrain_relative_height_error(env):
    """Commanded minus measured base height using the reward's ground reference."""
    ground, _ = terrain_height(env)
    command = env.command_manager.get_command("base_velocity_pose")[:, 3]
    measured = env.scene["robot"].data.root_pos_w[:, 2] - ground
    return (command - measured).unsqueeze(-1)


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
        self.recovery = RecoveryTracker(env.num_envs, env.device, env.step_dt) if self.cfg.recovery_metrics else None
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
        if self.recovery is not None:
            self.recovery.reset(env_ids)
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
        feet_in_yaw = quat_apply_inverse(
            yaw_quat(data.root_quat_w).unsqueeze(1).expand(-1, len(self.feet_ids), -1),
            data.body_pos_w[:, self.feet_ids] - data.root_pos_w.unsqueeze(1),
        )
        front_foot_width = (feet_in_yaw[:, 0, 1] - feet_in_yaw[:, 1, 1]).abs()
        rear_foot_width = (feet_in_yaw[:, 2, 1] - feet_in_yaw[:, 3, 1]).abs()
        foot_width = 0.5 * (front_foot_width + rear_foot_width)
        swing = sensor.data.current_air_time[:, self.contact_ids] > 0.0
        swing_height = (data.body_pos_w[:, self.feet_ids, 2] - height.unsqueeze(-1)) * swing
        swing_speed = torch.linalg.norm(data.body_lin_vel_w[:, self.feet_ids, :2], dim=-1) * swing
        air_time = sensor.data.current_air_time[:, self.contact_ids]
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
                foot_width,
                front_foot_width,
                rear_foot_width,
            ),
            dim=-1,
        )
        arm_controller = env.action_manager.get_term("joint_pos")._arm_controller
        extension_case = arm_controller.mode == 4
        extension_reached = extension_case & (
            (data.joint_pos[:, self.arm_ids] - arm_controller.extension_target).abs().amax(-1) < 0.15
        )
        errors = torch.cat((errors, swing_height, swing_speed, air_time,
                            extension_case[:, None].float(), extension_reached[:, None].float()), dim=-1)
        standing = cmd[:, :3].norm(dim=-1) < 0.1
        current_contact = sensor.data.net_forces_w[:, self.contact_ids].norm(dim=-1) > 1.0
        leg_speed = data.joint_vel[:, self.leg_ids].reshape(env.num_envs, 4, 3).square().mean(-1).sqrt()
        errors = torch.cat((errors, current_contact.all(-1, keepdim=True).float(),
                            current_contact.sum(-1, keepdim=True).float(), leg_speed), dim=-1)
        self.age += env.step_dt
        cohorts = torch.stack((torch.ones_like(self.hard), ~self.hard, self.hard))
        ages = torch.stack((torch.ones_like(self.hard), self.age <= 2.0, self.age > 2.0))
        masks = (cohorts[:, None] & ages[None]).reshape(9, env.num_envs).float()
        self.sums += masks @ errors.abs()
        self.squares += masks @ errors.square()
        self.counts += masks.sum(-1)
        sample_valid = torch.ones_like(errors)
        sample_valid[:, 7] = self.velocity_valid.float()
        sample_valid[:, 14:26] = swing.repeat(1, 3)
        sample_valid[:, 26:28] = extension_case[:, None]
        sample_valid[:, 28:30] = standing[:, None]
        sample_valid[:, 30:34] = (~standing)[:, None]
        # Numerators and denominators must use the same conditional samples.
        self.sums -= masks @ (errors.abs() * (1 - sample_valid))
        self.squares -= masks @ (errors.square() * (1 - sample_valid))
        self.value_counts += masks @ sample_valid
        self.previous_arm_velocity.copy_(arm_v)
        self.velocity_valid[:] = True
        fallen = (-data.projected_gravity_b[:, 2] < math.cos(math.radians(self.cfg.failure_tilt_degrees))) | (
            current_height < self.cfg.failure_height_m
        )
        self.failure_age = torch.where(fallen, self.failure_age + env.step_dt, 0.0)
        failed = (self.failure_age >= self.cfg.recovery_grace_s) & self.cfg.terminate_unrecovered
        if self.recovery is not None:
            absolute_tilt = (-data.projected_gravity_b[:, 2]).clamp(-1, 1).acos()
            self.recovery.update(
                absolute_tilt, current_height, data.root_ang_vel_b[:, :2].norm(dim=-1), self.age, failed
            )
            values = {
                "joint_velocity_abs_max_radps": data.joint_vel.abs().max(),
                "arm_velocity_abs_max_radps": arm_v.abs().max(),
                "arm_acceleration_abs_max_radps2": arm_a.abs().max(),
                "applied_torque_abs_max_nm": data.applied_torque.abs().max(),
                "root_angular_velocity_max_radps": data.root_ang_vel_w.norm(dim=-1).max(),
                "executed_action_abs_max": env.action_manager.action.abs().max(),
                "action_rate_l2_max": (
                    (env.action_manager.action - env.action_manager.prev_action).square().sum(-1).max()
                ),
            }
            for name, value in values.items():
                self.window_maxima[name] = torch.maximum(self.window_maxima.get(name, torch.zeros_like(value)), value)
        return failed

    def record_completed(self, env_ids):
        if self.recovery is not None:
            self.recovery.reset(env_ids, self.env.reset_terminated[env_ids])
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
        if self.recovery is not None:
            logs.update(self.recovery.flush())
            logs.update({"Numerics/" + key: value.clone() for key, value in self.window_maxima.items()})
            self.window_maxima.clear()
        for index, group in enumerate(self.group_names):
            prefix = "Robustness/" + group + "/"
            logs[prefix + "samples"] = self.counts[index].clone()
            logs[prefix + "arm_acceleration_samples"] = self.value_counts[index, 7].clone()
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
    arm = positions[:, arm_ids] + (
        (2 * torch.rand(len(env_ids), 6, device=env.device) - 1)
        * runtime.cfg.arm_init_joint_noise
        * runtime.progress("arm")
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


def recovery_action_rate_l2(env):
    """Allow faster corrective leg actions as absolute tilt rises from 20 to 50deg."""
    robot = env.scene["robot"]
    tilt = (-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1).acos()
    danger = ((tilt - math.radians(20)) / math.radians(30)).clamp(0, 1)
    scale = 1 - (1 - env.cfg.robustness.recovery_action_rate_scale) * danger
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1) * scale


def recovery_upright_cost(env):
    """A broad, bounded tilt cost; no positive bonus for inducing a recovery event."""
    return (1 + env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0, 2)


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
