# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for VelocityPose task with command-aware penalties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


# Command-aware tracking rewards


def track_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking height command with exponential growth - more sensitive to small errors"""
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    target_height = command[:, 3]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w.torch[..., 2]
        from ..shared.robustness_math import ground_height

        ground, _ = ground_height(ray_hits, env.scene.env_origins[:, 2])
        current_height = asset.data.root_pos_w.torch[:, 2] - ground
    else:
        current_height = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]

    height_error_abs = torch.abs(target_height - current_height)

    # Use exponential growth reward: smaller error -> exponentially higher reward
    # reward = exp(-|error|/std) where std controls sensitivity
    # When error=0: reward=1.0, When error=std: reward≈0.37
    reward = torch.exp(-height_error_abs / std)

    # Only give reward when robot is upright (avoid rewarding when fallen)
    # projected_gravity_b[:, 2] is close to -1 when upright
    # Use clamp to limit to [0, 0.7] range, normalized to [0, 1]
    gz = env.scene["robot"].data.projected_gravity_b.torch[:, 2]
    upright_factor = torch.clamp(-gz, 0, 0.7) / 0.7
    reward *= upright_factor

    return reward


def track_orientation_exp_without_yaw(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quaternion-based roll/pitch tracking; yaw ignored."""
    from isaaclab.utils.math import quat_conjugate, quat_from_euler_xyz, quat_mul

    asset: RigidObject = env.scene[asset_cfg.name]

    # NOTE: We completely ignore yaw (index 6) to decouple from localization
    command = env.command_manager.get_command(command_name)
    target_roll = command[:, 4]  # (num_envs,) - Roll in Point Frame B
    target_pitch = command[:, 5]  # (num_envs,) - Pitch in Point Frame B
    # target_yaw is NOT used - hardcoded to 0

    # Quaternion order: [x, y, z, w]
    # YAW IS HARDCODED TO ZERO - this decouples from localization
    target_quat = quat_from_euler_xyz(
        roll=target_roll,
        pitch=target_pitch,
        yaw=torch.zeros_like(target_roll),  # Always 0 - no yaw tracking
    )  # (num_envs, 4)

    current_quat_w = asset.data.root_quat_w.torch  # (num_envs, 4) in [x, y, z, w] format

    # This represents the rotation from World Frame A to Point Frame B
    # For quaternion [x, y, z, w], yaw angle can be extracted by projecting to z-axis:
    # yaw = atan2(2*(w*z + x*y), w^2 + x^2 - y^2 - z^2)
    # This is more numerically stable than the 1 - 2*(y^2 + z^2) form
    w, x, y, z = current_quat_w[:, 3], current_quat_w[:, 0], current_quat_w[:, 1], current_quat_w[:, 2]
    yaw_angle = torch.atan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)

    # This quaternion is automatically normalized since cos²(θ/2) + sin²(θ/2) = 1
    half_yaw = yaw_angle / 2
    current_yaw_quat = torch.stack(
        [
            torch.zeros_like(half_yaw),  # x
            torch.zeros_like(half_yaw),  # y
            torch.sin(half_yaw),  # z
            torch.cos(half_yaw),  # w
        ],
        dim=1,
    )  # (num_envs, 4) - guaranteed to be normalized

    # current_quat_yaw_aligned = yaw_quat^(-1) * current_quat_w
    current_yaw_quat_inv = quat_conjugate(current_yaw_quat)
    current_quat_yaw_aligned = quat_mul(current_yaw_quat_inv, current_quat_w)  # (num_envs, 4)

    # IMPROVED METHOD: Only consider roll and pitch components (x, y) in yaw-aligned frame
    # This completely decouples from yaw by ignoring quaternion z-component
    #
    # In yaw-aligned frame quaternion [x, y, z, w]:
    #   x component ~ roll error
    #   y component ~ pitch error
    #   z component ~ yaw error (IGNORED)
    #
    # Instead of computing full quaternion error, we directly measure roll/pitch deviation
    # using only the x and y components of the yaw-aligned quaternion.

    # Method: Use small angle approximation for roll and pitch
    # For small rotations, quaternion components approximate half-angles:
    #   x ≈ sin(roll/2) ≈ roll/2 (for small roll)
    #   y ≈ sin(pitch/2) ≈ pitch/2 (for small pitch)
    #
    # However, for larger angles we need the full formula:
    #   roll = 2 * atan2(x, w)  (ignoring pitch/yaw coupling)
    #   pitch = 2 * atan2(y, w)  (ignoring roll/yaw coupling)

    # Current orientation in yaw-aligned frame
    current_x = current_quat_yaw_aligned[:, 0]  # roll component
    current_y = current_quat_yaw_aligned[:, 1]  # pitch component
    # current_z is ignored (yaw component)

    # Target orientation in yaw-aligned frame
    target_x = target_quat[:, 0]  # roll component (from target_roll)
    target_y = target_quat[:, 1]  # pitch component (from target_pitch)
    # target_z = 0 (yaw is zero)

    # Using the fact that for rotations around x-axis (roll): x = sin(roll/2)
    # and for rotations around y-axis (pitch): y = sin(pitch/2)

    # For better accuracy, compute the angle error from x and y components only
    # Error metric: ||[x_err, y_err]||^2 where x_err = x_current - x_target
    roll_error_component = current_x - target_x
    pitch_error_component = current_y - target_y

    # This is proportional to the actual angular error for small angles
    # For larger angles, it still provides a good approximation
    component_error_norm = torch.sqrt(roll_error_component**2 + pitch_error_component**2)

    # Since x ≈ sin(roll/2), the error in x is approximately error_x ≈ cos(roll/2) * (Δroll/2)
    # For simplicity and numerical stability, we use: angle_error ≈ 2 * arcsin(component_error_norm)
    angle_error = 2.0 * torch.arcsin(
        torch.clamp(component_error_norm, 0.0, 1.0)
    )  # (num_envs,)    # Use exponential growth reward: smaller error -> exponentially higher reward
    # reward = exp(-|error|/std) where std controls sensitivity
    reward = torch.exp(-angle_error / std)

    # Only give reward when robot is upright
    projected_gravity = asset.data.projected_gravity_b.torch
    upright_factor = torch.clamp(-projected_gravity[:, 2], 0, 0.7) / 0.7
    reward *= upright_factor

    return reward


# Command-aware conditional penalties


def lin_vel_z_penalty_conditional(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Conditional penalty on vertical velocity - only penalize when height command is close to default"""
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    height_cmd = command[:, 3]

    default_height = env.command_manager.get_term("base_velocity_pose").default_height

    height_cmd_diff = torch.abs(height_cmd - default_height)

    # Only penalize vertical velocity when height command is close to default value
    should_penalize = height_cmd_diff < height_threshold

    penalty = torch.square(asset.data.root_lin_vel_b.torch[:, 2])

    # Conditionally apply penalty: keep original value when should penalize, otherwise set to zero
    penalty = torch.where(should_penalize, penalty, torch.zeros_like(penalty))

    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7

    return penalty


def ang_vel_xy_penalty_conditional(
    env: ManagerBasedRLEnv,
    command_name: str,
    angle_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Conditional penalty on roll/pitch angular velocity - only penalize when target orientation is close to zero"""
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    target_roll = command[:, 4]  # Target roll angle (rad)
    target_pitch = command[:, 5]  # Target pitch angle (rad)

    target_orientation_norm = torch.sqrt(target_roll**2 + target_pitch**2)

    # Only penalize angular velocity when target orientation is close to zero (i.e., target is flat)
    should_penalize = target_orientation_norm < angle_threshold

    penalty = torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1)

    # Conditionally apply penalty: penalize only when target is flat
    penalty = torch.where(should_penalize, penalty, torch.zeros_like(penalty))

    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7

    return penalty


def stand_still_full_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stand still penalty considering full command (including height and orientation)"""
    _unused_asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    velocity_cmd = command[:, :3]  # [vx, vy, ωz]
    height_cmd = command[:, 3]
    pose_cmd = command[:, 4:7]  # [roll, pitch, yaw]

    default_height = env.command_manager.get_term("base_velocity_pose").default_height

    # Determine if each dimension is "still"
    velocity_small = torch.norm(velocity_cmd, dim=1) < velocity_threshold
    height_default = torch.abs(height_cmd - default_height) < height_threshold
    pose_zero = torch.norm(pose_cmd, dim=1) < angle_threshold

    # Only consider "truly static" when all dimensions are still
    is_truly_static = velocity_small & height_default & pose_zero

    penalty = mdp.joint_deviation_l1(env, asset_cfg)

    # Only apply penalty when truly static
    penalty = torch.where(is_truly_static, penalty, torch.zeros_like(penalty))

    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7

    return penalty


def joint_pos_penalty_full_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    velocity_cmd_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
) -> torch.Tensor:
    """Joint position penalty considering full command"""
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    velocity_cmd = command[:, :3]
    height_cmd = command[:, 3]
    pose_cmd = command[:, 4:7]

    default_height = env.command_manager.get_term("base_velocity_pose").default_height

    # Determine if in motion
    velocity_cmd_norm = torch.norm(velocity_cmd, dim=1)
    body_vel = torch.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    height_cmd_diff = torch.abs(height_cmd - default_height)
    pose_cmd_norm = torch.norm(pose_cmd, dim=1)

    # Motion condition: velocity command large OR actual velocity large OR height change OR orientation change
    is_moving = (
        (velocity_cmd_norm > velocity_cmd_threshold)
        | (body_vel > velocity_threshold)
        | (height_cmd_diff > height_threshold)
        | (pose_cmd_norm > angle_threshold)
    )

    base_penalty = torch.norm(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids],
        dim=1,
    )

    # Adjust penalty strength based on motion state
    # Moving: 1.0× penalty, Standing still: 5.0× penalty
    penalty = torch.where(is_moving, base_penalty, stand_still_scale * base_penalty)

    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7

    return penalty


# Helper functions


# Updated existing reward functions to use full command awareness


def accumulated_ang_vel_penalty_when_standing(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity_pose",
    velocity_threshold: float = 0.1,
    angle_std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize accumulated angular velocity when standing with EXPONENTIAL penalty (deployable to real robot)."""
    asset: Articulation = env.scene[asset_cfg.name]

    # CRITICAL: Check curriculum stage - MUST be >= 1 to activate
    # Changed from >= 2 to >= 1 to prevent self-spinning from the beginning
    current_stage = getattr(env, "_curriculum_stage", 0)
    if current_stage < 1:
        return torch.zeros(env.num_envs, device=env.device)

    command = env.command_manager.get_command(command_name)

    contact_sensor = env.scene.sensors.get("contact_forces", None)
    if contact_sensor is not None:
        # net_normal_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
        # We only need the most recent forces: [:, 0, :, :]
        net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch[
            :, 0, :, :
        ]  # (num_envs, num_bodies, 3)
        force_threshold = 1.0  # N
        foot_contact = torch.norm(net_contact_forces, dim=-1) > force_threshold  # (num_envs, num_bodies)
        # Robot is grounded if at least 2 feet are in contact
        num_feet_contact = foot_contact.sum(dim=-1)  # (num_envs,)
        is_grounded = num_feet_contact >= 2  # (num_envs,)
    else:
        # Fallback: assume always grounded if no contact sensor
        is_grounded = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    # CRITICAL: Must check angular velocity command (ωz) as well!
    # Only penalize self-spinning when robot is commanded to stand still (no rotation)
    lin_vel_cmd = command[:, :2]  # (num_envs, 2) - [vx, vy]
    ang_vel_cmd = command[:, 2]  # (num_envs,) - [ωz]
    lin_vel_cmd_norm = torch.norm(lin_vel_cmd, dim=1)  # (num_envs,)
    ang_vel_cmd_abs = torch.abs(ang_vel_cmd)  # (num_envs,)

    # Determine if robot should be standing still
    # Condition: linear velocity command ≈ 0 AND angular velocity command ≈ 0 AND robot is grounded
    is_standing_command = (lin_vel_cmd_norm < velocity_threshold) & (ang_vel_cmd_abs < velocity_threshold)
    is_standing = is_standing_command & is_grounded  # (num_envs,)

    ang_vel_z = asset.data.root_ang_vel_b.torch[:, 2]  # (num_envs,)

    if not hasattr(env, "_accumulated_ang_vel_local"):
        env._accumulated_ang_vel_local = torch.zeros(env.num_envs, device=env.device)
        env._standing_time_local = torch.zeros(env.num_envs, device=env.device)

    # CRITICAL: Detect episode reset (when episode just started)
    # episode_length_buf tracks how many steps have elapsed in current episode
    # When it's 0 or 1, the episode just started/reset
    just_reset = env.episode_length_buf <= 1  # (num_envs,) bool tensor

    # Time step
    dt = env.step_dt

    # CRITICAL: Update accumulated angular velocity with THREE reset conditions:
    # 1. Episode just reset (just_reset=True) -> Reset to zero
    # 2. Command is non-zero (not standing) -> Reset to zero
    # 3. Standing command is active -> Continue accumulating
    env._accumulated_ang_vel_local = torch.where(
        just_reset,
        torch.zeros_like(env._accumulated_ang_vel_local),  # Reset when episode resets
        torch.where(
            is_standing,
            env._accumulated_ang_vel_local + ang_vel_z * dt,  # Accumulate during standing
            torch.zeros_like(env._accumulated_ang_vel_local),  # Reset when moving
        ),
    )

    env._standing_time_local = torch.where(
        just_reset,
        torch.zeros_like(env._standing_time_local),  # Reset when episode resets
        torch.where(
            is_standing,
            env._standing_time_local + dt,  # Continue timing during standing
            torch.zeros_like(env._standing_time_local),  # Reset when moving
        ),
    )

    # Exponential penalty with angle_std control on accumulated angular velocity
    # penalty = -(exp(|accumulated_angle| / angle_std) - 1)
    # angle_std controls sensitivity: smaller angle_std → stricter penalty
    #
    # CRITICAL: Clamp the exponent to prevent numerical overflow AND limit max penalty
    # Target: penalty should not exceed -200 in magnitude
    # Math: -(exp(x) - 1) ≥ -200  =>  exp(x) ≤ 201  =>  x ≤ ln(201) ≈ 5.3
    # We use max_exponent=5.3 to ensure: exp(5.3) ≈ 200, penalty ≈ -199
    abs_accumulated = torch.abs(env._accumulated_ang_vel_local)
    exponent = abs_accumulated / angle_std
    max_exponent = 5.3  # ln(201) ≈ 5.3, ensures penalty magnitude ≤ 200
    exponent_clamped = torch.clamp(exponent, max=max_exponent)
    penalty = -(torch.exp(exponent_clamped) - 1.0)  # (num_envs,), range: [0, -199]

    # Only apply penalty when:
    # 1. Currently standing
    # 2. Have been standing for at least 0.5 seconds (avoid transient effects)
    penalty = torch.where((is_standing) & (env._standing_time_local > 0.5), penalty, torch.zeros_like(penalty))

    return penalty


# ARX5 Arm Stability and Anti-Flip Rewards (Stage 1)
