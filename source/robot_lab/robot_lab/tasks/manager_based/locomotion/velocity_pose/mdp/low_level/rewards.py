# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for VelocityPose task with command-aware penalties."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Command-aware tracking rewards
##


def track_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking height command with exponential growth - more sensitive to small errors
    
    Uses exponential growth reward function: reward = exp(-|error|/std)
    Gives high reward when robot base height is close to commanded height.
    
    Args:
        env: Environment instance
        std: Tolerance parameter in meters (smaller = more sensitive)
              e.g., std=0.05m means ±5cm tolerance
        command_name: Command name in command manager (should be "base_velocity_pose")
        sensor_cfg: Height sensor config for getting terrain height. If None, uses world z coordinate
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward values with shape (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get target height command (4th dimension of command)
    command = env.command_manager.get_command(command_name)
    target_height = command[:, 3]
    
    # Calculate current height (considering terrain)
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        from ..shared.robustness_math import ground_height
        ground, _ = ground_height(ray_hits, env.scene.env_origins[:, 2])
        current_height = asset.data.root_pos_w[:, 2] - ground
    else:
        current_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    # Calculate absolute height error (more sensitive to small errors)
    height_error_abs = torch.abs(target_height - current_height)
    
    # Use exponential growth reward: smaller error -> exponentially higher reward
    # reward = exp(-|error|/std) where std controls sensitivity
    # When error=0: reward=1.0, When error=std: reward≈0.37
    reward = torch.exp(-height_error_abs / std)
    
    # Only give reward when robot is upright (avoid rewarding when fallen)
    # projected_gravity_b[:, 2] is close to -1 when upright
    # Use clamp to limit to [0, 0.7] range, normalized to [0, 1]
    gz = env.scene["robot"].data.projected_gravity_b[:, 2]
    upright_factor = torch.clamp(-gz, 0, 0.7) / 0.7
    reward *= upright_factor
    
    # Disable reward during Stage 1 (base training phase, 0-20,000 iterations)
    # MODIFIED: Stage 1 is now skipped, so this check is no longer needed
    # Rewards are always enabled starting from Stage 2
    # if hasattr(env, "_curriculum_stage") and env._curriculum_stage == 1:
    #     reward = torch.zeros_like(reward)
    
    # Debug: Print statistics every 100 steps to catch the issue early (DISABLED for performance)
    # if not hasattr(env, "_height_debug_counter"):
    #     env._height_debug_counter = 0
    # env._height_debug_counter += 1
    # if env._height_debug_counter % 100 == 0:
    #     stage_info = f" [Stage {env._curriculum_stage}]" if hasattr(env, "_curriculum_stage") else ""
    #     print(f"\n[DEBUG] Height Tracking Reward Statistics (Step {env._height_debug_counter}){stage_info}:")
    #     print(f"  Current height:               mean={current_height.mean().item():.4f}, min={current_height.min().item():.4f}, max={current_height.max().item():.4f}")
    #     print(f"  Target height:                mean={target_height.mean().item():.4f}, min={target_height.min().item():.4f}, max={target_height.max().item():.4f}")
    #     print(f"  Height error (abs):           mean={height_error_abs.mean().item():.4f}, max={height_error_abs.max().item():.4f}")
    #     print(f"  projected_gravity[:, 2] (gz): mean={gz.mean().item():.6f}, min={gz.min().item():.6f}, max={gz.max().item():.6f}")
    #     print(f"  Upright factor:               mean={upright_factor.mean().item():.6f}, min={upright_factor.min().item():.6f}, max={upright_factor.max().item():.6f}")
    #     print(f"  Raw reward (before upright):  mean={torch.exp(-height_error_abs / std).mean().item():.6f}")
    #     print(f"  Final reward (after upright): mean={reward.mean().item():.9f}, min={reward.min().item():.9f}, max={reward.max().item():.9f}")
    
    return reward


def track_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking orientation command using quaternion-based error calculation in yaw-aligned frame.
    
    Coordinate System Framework:
    - World Frame A: Fixed global reference (never changes)
    - Robot Point Frame B (Yaw-Aligned): Z-axis vertical, XY rotates with motion direction
    - Robot Base Frame C (Body): Fully follows base orientation (roll, pitch, yaw)
    
    This function tracks the orientation of Base Frame C relative to Point Frame B.
    The command specifies [roll, pitch, yaw] angles that define the desired orientation
    of Base Frame C when Point Frame B is used as the reference.
    
    Args:
        env: Environment instance
        std: Tolerance parameter in radians (smaller = more sensitive)
              e.g., std=0.15rad ≈ 8.6° tolerance
        command_name: Command name in command manager (must be 7D command)
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward values with shape (num_envs,)
        
    Design Intent:
        - Use quaternion math in yaw-aligned frame (Point Frame B)
        - Supports full 3D orientation control (roll, pitch, yaw)
        - More accurate than angle decomposition (no coupling errors)
        - Exponential reward function for sensitivity to small errors
        
    Reward Characteristics:
        - Angle error = 0.00rad (0°): reward = 1.00 (perfect)
        - Angle error = 0.05rad (2.9°): reward ≈ 0.72 (good)
        - Angle error = 0.15rad (8.6°, =std): reward ≈ 0.37 (acceptable)
        - Angle error = 0.30rad (17.2°): reward ≈ 0.14 (poor)
        
    Implementation:
        1. Extract target [roll, pitch, yaw] from 7D command (indices 4, 5, 6)
        2. Convert to target quaternion in Point Frame B
        3. Get current base quaternion in World Frame A
        4. Extract motion direction (yaw_motion) from World Frame A
        5. Project current quaternion to Point Frame B by removing motion direction
        6. Compute quaternion error between target and current (both in Point Frame B)
        7. Extract rotation angle from error quaternion
        8. Apply exponential reward based on angle error
    """
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_conjugate
    
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get target orientation command (indices 4, 5, 6 of 7D command: [vx, vy, ωz, h, roll, pitch, yaw])
    command = env.command_manager.get_command(command_name)
    target_roll = command[:, 4]   # (num_envs,) - Roll in Point Frame B
    target_pitch = command[:, 5]  # (num_envs,) - Pitch in Point Frame B  
    target_yaw = command[:, 6]    # (num_envs,) - Yaw in Point Frame B
    
    # Convert target [roll, pitch, yaw] to quaternion in Point Frame B
    # Quaternion order: [w, x, y, z]
    target_quat = quat_from_euler_xyz(
        roll=target_roll,
        pitch=target_pitch,
        yaw=target_yaw  # Now we include yaw in pose control
    )  # (num_envs, 4)
    
    # Get current base quaternion in World Frame A
    current_quat_w = asset.data.root_quat_w  # (num_envs, 4) in [w, x, y, z] format
    
    # Extract motion direction (yaw_motion) from World Frame A
    # This represents the rotation from World Frame A to Point Frame B
    # For quaternion [w, x, y, z], yaw angle can be extracted by projecting to z-axis:
    # yaw = atan2(2*(w*z + x*y), w^2 + x^2 - y^2 - z^2)
    # This is more numerically stable than the 1 - 2*(y^2 + z^2) form
    w, x, y, z = current_quat_w[:, 0], current_quat_w[:, 1], current_quat_w[:, 2], current_quat_w[:, 3]
    yaw_angle = torch.atan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)
    
    # Create pure yaw quaternion: q_yaw = [cos(yaw/2), 0, 0, sin(yaw/2)]
    # This quaternion is automatically normalized since cos²(θ/2) + sin²(θ/2) = 1
    half_yaw = yaw_angle / 2
    current_yaw_quat = torch.stack([
        torch.cos(half_yaw),  # w
        torch.zeros_like(half_yaw),  # x
        torch.zeros_like(half_yaw),  # y
        torch.sin(half_yaw)   # z
    ], dim=1)  # (num_envs, 4) - guaranteed to be normalized
    
    # Remove yaw from current quaternion to get yaw-aligned orientation
    # current_quat_yaw_aligned = yaw_quat^(-1) * current_quat_w
    current_yaw_quat_inv = quat_conjugate(current_yaw_quat)
    current_quat_yaw_aligned = quat_mul(current_yaw_quat_inv, current_quat_w)  # (num_envs, 4)
    
    # Compute quaternion error in yaw-aligned frame: q_error = q_current^(-1) * q_target
    # This gives us the rotation needed to go from current to target orientation
    # The magnitude of rotation angle is the same regardless of multiplication order,
    # but this order is semantically clearer for future directional error analysis
    current_quat_inv = quat_conjugate(current_quat_yaw_aligned)
    quat_error = quat_mul(current_quat_inv, target_quat)  # (num_envs, 4)
    
    # Extract rotation angle from error quaternion
    # For quaternion q = [w, x, y, z], rotation angle θ = 2 * arccos(|w|)
    # Clamp w to [-1, 1] to avoid numerical issues with arccos
    quat_w = torch.clamp(quat_error[:, 0], -1.0, 1.0)
    angle_error = 2.0 * torch.acos(torch.abs(quat_w))  # (num_envs,)
    
    # Use exponential growth reward: smaller error -> exponentially higher reward
    # reward = exp(-|error|/std) where std controls sensitivity
    reward = torch.exp(-angle_error / std)
    
    # Only give reward when robot is upright
    projected_gravity = asset.data.projected_gravity_b
    upright_factor = torch.clamp(-projected_gravity[:, 2], 0, 0.7) / 0.7
    reward *= upright_factor
    
    # Disable reward during Stage 1 (base training phase, 0-20,000 iterations)
    # MODIFIED: Stage 1 is now skipped, so this check is no longer needed
    # Rewards are always enabled starting from Stage 2
    # if hasattr(env, "_curriculum_stage") and env._curriculum_stage == 1:
    #     reward = torch.zeros_like(reward)
    
    # Debug: Print statistics every 100 steps (DISABLED for performance)
    # if not hasattr(env, "_orient_debug_counter"):
    #     env._orient_debug_counter = 0
    # env._orient_debug_counter += 1
    # if env._orient_debug_counter % 100 == 0:
    #     gz = projected_gravity[:, 2]
    #     # Verify yaw quaternion normalization
    #     yaw_quat_norm = torch.norm(current_yaw_quat, dim=1)
    #     current_quat_w_norm = torch.norm(current_quat_w, dim=1)
    #     
    #     stage_info = f" [Stage {env._curriculum_stage}]" if hasattr(env, "_curriculum_stage") else ""
    #     print(f"\n[DEBUG] Orientation Tracking Reward Statistics (Step {env._orient_debug_counter}){stage_info}:")
    #     print(f"  Target roll (deg):             mean={torch.rad2deg(target_roll).mean().item():.2f}, max={torch.rad2deg(target_roll.abs()).max().item():.2f}")
    #     print(f"  Target pitch (deg):            mean={torch.rad2deg(target_pitch).mean().item():.2f}, max={torch.rad2deg(target_pitch.abs()).max().item():.2f}")
    #     print(f"  Target yaw (deg):              mean={torch.rad2deg(target_yaw).mean().item():.2f}, max={torch.rad2deg(target_yaw.abs()).max().item():.2f}")
    #     print(f"  Target quat [w,x,y,z]:         mean=[{target_quat[:, 0].mean():.3f}, {target_quat[:, 1].mean():.3f}, {target_quat[:, 2].mean():.3f}, {target_quat[:, 3].mean():.3f}]")
    #     print(f"  Current quat (world):          mean=[{current_quat_w[:, 0].mean():.3f}, {current_quat_w[:, 1].mean():.3f}, {current_quat_w[:, 2].mean():.3f}, {current_quat_w[:, 3].mean():.3f}]")
    #     print(f"  Current quat norm:             mean={current_quat_w_norm.mean():.6f}, min={current_quat_w_norm.min():.6f}, max={current_quat_w_norm.max():.6f}")
    #     print(f"  Current yaw quat:              mean=[{current_yaw_quat[:, 0].mean():.3f}, {current_yaw_quat[:, 1].mean():.3f}, {current_yaw_quat[:, 2].mean():.3f}, {current_yaw_quat[:, 3].mean():.3f}]")
    #     print(f"  Current yaw quat norm:         mean={yaw_quat_norm.mean():.6f}, min={yaw_quat_norm.min():.6f}, max={yaw_quat_norm.max():.6f}")
    #     print(f"  Current quat (yaw-aligned):    mean=[{current_quat_yaw_aligned[:, 0].mean():.3f}, {current_quat_yaw_aligned[:, 1].mean():.3f}, {current_quat_yaw_aligned[:, 2].mean():.3f}, {current_quat_yaw_aligned[:, 3].mean():.3f}]")
    #     print(f"  Error quat [w,x,y,z]:          mean=[{quat_error[:, 0].mean():.3f}, {quat_error[:, 1].mean():.3f}, {quat_error[:, 2].mean():.3f}, {quat_error[:, 3].mean():.3f}]")
    #     print(f"  Error quat w component:        mean={quat_error[:, 0].mean():.6f}, min={quat_error[:, 0].min():.6f}, max={quat_error[:, 0].max():.6f}")
    #     print(f"  Quaternion error angle (deg):  mean={torch.rad2deg(angle_error).mean().item():.2f}, max={torch.rad2deg(angle_error).max().item():.2f}")
    #     print(f"  projected_gravity[:, 2] (gz):  mean={gz.mean().item():.6f}, min={gz.min().item():.6f}, max={gz.max().item():.6f}")
    #     print(f"  Upright factor (-gz clamped):  mean={upright_factor.mean().item():.6f}, min={upright_factor.min().item():.6f}, max={upright_factor.max().item():.6f}")
    #     print(f"  Raw reward (before upright):   mean={torch.exp(-angle_error / std).mean().item():.6f}")
    #     print(f"  Final reward (after upright):  mean={reward.mean().item():.9f}, min={reward.min().item():.9f}, max={reward.max().item():.9f}")
    
    return reward


def track_orientation_exp_without_yaw(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking orientation command (roll and pitch only, yaw ignored) using quaternion-based error calculation.
    
    This function is identical to track_orientation_exp but completely ignores yaw component.
    This is designed to avoid coupling with localization systems, as yaw cannot be determined
    from IMU alone (only roll and pitch can be calculated from gravity projection).
    
    Coordinate System Framework:
    - World Frame A: Fixed global reference (never changes)
    - Robot Point Frame B (Yaw-Aligned): Z-axis vertical, XY rotates with motion direction
    - Robot Base Frame C (Body): Fully follows base orientation (roll, pitch, yaw)
    
    This function tracks ONLY roll and pitch of Base Frame C relative to Point Frame B.
    Yaw is completely ignored in both command and error calculation.
    
    Args:
        env: Environment instance
        std: Tolerance parameter in radians (smaller = more sensitive)
              e.g., std=0.15rad ≈ 8.6° tolerance
        command_name: Command name in command manager (must be 7D command)
        asset_cfg: Robot asset configuration
        
    Returns:
        Reward values with shape (num_envs,)
        
    """
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_conjugate
    
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get target orientation command (indices 4, 5 of 7D command: [vx, vy, ωz, h, roll, pitch, yaw])
    # NOTE: We completely ignore yaw (index 6) to decouple from localization
    command = env.command_manager.get_command(command_name)
    target_roll = command[:, 4]   # (num_envs,) - Roll in Point Frame B
    target_pitch = command[:, 5]  # (num_envs,) - Pitch in Point Frame B
    # target_yaw is NOT used - hardcoded to 0
    
    # Convert target [roll, pitch, 0] to quaternion in Point Frame B
    # Quaternion order: [w, x, y, z]
    # YAW IS HARDCODED TO ZERO - this decouples from localization
    target_quat = quat_from_euler_xyz(
        roll=target_roll,
        pitch=target_pitch,
        yaw=torch.zeros_like(target_roll)  # Always 0 - no yaw tracking
    )  # (num_envs, 4)
    
    # Get current base quaternion in World Frame A
    current_quat_w = asset.data.root_quat_w  # (num_envs, 4) in [w, x, y, z] format
    
    # Extract motion direction (yaw_motion) from World Frame A
    # This represents the rotation from World Frame A to Point Frame B
    # For quaternion [w, x, y, z], yaw angle can be extracted by projecting to z-axis:
    # yaw = atan2(2*(w*z + x*y), w^2 + x^2 - y^2 - z^2)
    # This is more numerically stable than the 1 - 2*(y^2 + z^2) form
    w, x, y, z = current_quat_w[:, 0], current_quat_w[:, 1], current_quat_w[:, 2], current_quat_w[:, 3]
    yaw_angle = torch.atan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)
    
    # Create pure yaw quaternion: q_yaw = [cos(yaw/2), 0, 0, sin(yaw/2)]
    # This quaternion is automatically normalized since cos²(θ/2) + sin²(θ/2) = 1
    half_yaw = yaw_angle / 2
    current_yaw_quat = torch.stack([
        torch.cos(half_yaw),  # w
        torch.zeros_like(half_yaw),  # x
        torch.zeros_like(half_yaw),  # y
        torch.sin(half_yaw)   # z
    ], dim=1)  # (num_envs, 4) - guaranteed to be normalized
    
    # Remove yaw from current quaternion to get yaw-aligned orientation
    # current_quat_yaw_aligned = yaw_quat^(-1) * current_quat_w
    current_yaw_quat_inv = quat_conjugate(current_yaw_quat)
    current_quat_yaw_aligned = quat_mul(current_yaw_quat_inv, current_quat_w)  # (num_envs, 4)
    
    # IMPROVED METHOD: Only consider roll and pitch components (x, y) in yaw-aligned frame
    # This completely decouples from yaw by ignoring quaternion z-component
    #
    # In yaw-aligned frame quaternion [w, x, y, z]:
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

    # Extract roll and pitch from yaw-aligned quaternions
    # Current orientation in yaw-aligned frame
    current_x = current_quat_yaw_aligned[:, 1]  # roll component
    current_y = current_quat_yaw_aligned[:, 2]  # pitch component
    # current_z is ignored (yaw component)

    # Target orientation in yaw-aligned frame
    target_x = target_quat[:, 1]  # roll component (from target_roll)
    target_y = target_quat[:, 2]  # pitch component (from target_pitch)
    # target_z = 0 (yaw is zero)

    # Compute roll and pitch errors directly from quaternion components
    # Using the fact that for rotations around x-axis (roll): x = sin(roll/2)
    # and for rotations around y-axis (pitch): y = sin(pitch/2)

    # For better accuracy, compute the angle error from x and y components only
    # Error metric: ||[x_err, y_err]||^2 where x_err = x_current - x_target
    roll_error_component = current_x - target_x
    pitch_error_component = current_y - target_y

    # Compute angular error from components (using L2 norm of sine of half-angles)
    # This is proportional to the actual angular error for small angles
    # For larger angles, it still provides a good approximation
    component_error_norm = torch.sqrt(roll_error_component**2 + pitch_error_component**2)

    # Convert component error to approximate angle error
    # Since x ≈ sin(roll/2), the error in x is approximately error_x ≈ cos(roll/2) * (Δroll/2)
    # For simplicity and numerical stability, we use: angle_error ≈ 2 * arcsin(component_error_norm)
    # Clamp to avoid numerical issues with arcsin (domain is [-1, 1])
    angle_error = 2.0 * torch.arcsin(torch.clamp(component_error_norm, 0.0, 1.0))  # (num_envs,)    # Use exponential growth reward: smaller error -> exponentially higher reward
    # reward = exp(-|error|/std) where std controls sensitivity
    reward = torch.exp(-angle_error / std)
    
    # Only give reward when robot is upright
    projected_gravity = asset.data.projected_gravity_b
    upright_factor = torch.clamp(-projected_gravity[:, 2], 0, 0.7) / 0.7
    reward *= upright_factor
    
    # Debug: Print statistics every 100 steps (DISABLED for performance)
    # if not hasattr(env, "_orient_noyaw_debug_counter"):
    #     env._orient_noyaw_debug_counter = 0
    # env._orient_noyaw_debug_counter += 1
    # if env._orient_noyaw_debug_counter % 100 == 0:
    #     gz = projected_gravity[:, 2]
    #     # Verify yaw quaternion normalization
    #     yaw_quat_norm = torch.norm(current_yaw_quat, dim=1)
    #     current_quat_w_norm = torch.norm(current_quat_w, dim=1)
    #     
    #     stage_info = f" [Stage {env._curriculum_stage}]" if hasattr(env, "_curriculum_stage") else ""
    #     print(f"\n[DEBUG] Orientation Tracking (NO YAW) Reward Statistics (Step {env._orient_noyaw_debug_counter}){stage_info}:")
    #     print(f"  Target roll (deg):             mean={torch.rad2deg(target_roll).mean().item():.2f}, max={torch.rad2deg(target_roll.abs()).max().item():.2f}")
    #     print(f"  Target pitch (deg):            mean={torch.rad2deg(target_pitch).mean().item():.2f}, max={torch.rad2deg(target_pitch.abs()).max().item():.2f}")
    #     print("  Target yaw:                    IGNORED (no yaw tracking)")
    #     print(f"  Target quat [w,x,y,z]:         mean=[{target_quat[:, 0].mean():.3f}, {target_quat[:, 1].mean():.3f}, {target_quat[:, 2].mean():.3f}, {target_quat[:, 3].mean():.3f}]")
    #     print(f"  Current quat (world):          mean=[{current_quat_w[:, 0].mean():.3f}, {current_quat_w[:, 1].mean():.3f}, {current_quat_w[:, 2].mean():.3f}, {current_quat_w[:, 3].mean():.3f}]")
    #     print(f"  Current quat norm:             mean={current_quat_w_norm.mean():.6f}, min={current_quat_w_norm.min():.6f}, max={current_quat_w_norm.max():.6f}")
    #     print(f"  Current yaw quat:              mean=[{current_yaw_quat[:, 0].mean():.3f}, {current_yaw_quat[:, 1].mean():.3f}, {current_yaw_quat[:, 2].mean():.3f}, {current_yaw_quat[:, 3].mean():.3f}]")
    #     print(f"  Current yaw quat norm:         mean={yaw_quat_norm.mean():.6f}, min={yaw_quat_norm.min():.6f}, max={yaw_quat_norm.max():.6f}")
    #     print(f"  Current quat (yaw-aligned):    mean=[{current_quat_yaw_aligned[:, 0].mean():.3f}, {current_quat_yaw_aligned[:, 1].mean():.3f}, {current_quat_yaw_aligned[:, 2].mean():.3f}, {current_quat_yaw_aligned[:, 3].mean():.3f}]")
    #     print(f"  Roll error component (x):      mean={roll_error_component.mean().item():.6f}, max={roll_error_component.abs().max().item():.6f}")
    #     print(f"  Pitch error component (y):     mean={pitch_error_component.mean().item():.6f}, max={pitch_error_component.abs().max().item():.6f}")
    #     print(f"  Component error norm:          mean={component_error_norm.mean().item():.6f}, max={component_error_norm.max().item():.6f}")
    #     print(f"  Angular error (deg):           mean={torch.rad2deg(angle_error).mean().item():.2f}, max={torch.rad2deg(angle_error).max().item():.2f}")
    #     print(f"  projected_gravity[:, 2] (gz):  mean={gz.mean().item():.6f}, min={gz.min().item():.6f}, max={gz.max().item():.6f}")
    #     print(f"  Upright factor (-gz clamped):  mean={upright_factor.mean().item():.6f}, min={upright_factor.min().item():.6f}, max={upright_factor.max().item():.6f}")
    #     print(f"  Raw reward (before upright):   mean={torch.exp(-angle_error / std).mean().item():.6f}")
    #     print(f"  Final reward (after upright):  mean={reward.mean().item():.9f}, min={reward.min().item():.9f}, max={reward.max().item():.9f}")
    
    return reward


##
# Command-aware conditional penalties
##


def lin_vel_z_penalty_conditional(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Conditional penalty on vertical velocity - only penalize when height command is close to default
    
    This function replaces the original lin_vel_z_l2 by checking if the height command has changed
    to decide whether to penalize vertical velocity.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        height_threshold: Height command change threshold (meters), below this value means "no height command change"
        asset_cfg: Robot asset configuration
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Allow vertical velocity during height adjustments (avoid conflicts)
        - Still penalize unnecessary vertical motion (e.g., jumping)
        - Implement intelligent switching through conditional logic
        
    Logic:
        IF |height_cmd - default_height| < threshold:
            Penalize vertical velocity (no height adjustment needed)
        ELSE:
            No penalty (height adjustment in progress)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get height command
    command = env.command_manager.get_command(command_name)
    height_cmd = command[:, 3]
    
    # Get default height (from command manager configuration)
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    
    # Calculate difference between height command and default value
    height_cmd_diff = torch.abs(height_cmd - default_height)
    
    # Only penalize vertical velocity when height command is close to default value
    should_penalize = height_cmd_diff < height_threshold
    
    # Calculate squared vertical velocity (z direction in body frame)
    penalty = torch.square(asset.data.root_lin_vel_b[:, 2])
    
    # Conditionally apply penalty: keep original value when should penalize, otherwise set to zero
    penalty = torch.where(should_penalize, penalty, torch.zeros_like(penalty))
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


def ang_vel_xy_penalty_conditional(
    env: ManagerBasedRLEnv,
    command_name: str,
    angle_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Conditional penalty on roll/pitch angular velocity - only penalize when target orientation is close to zero
    
    This function replaces the original ang_vel_xy_l2 by checking if the target orientation (command) 
    is close to zero to decide whether to penalize angular velocity.
    
    Note: roll and pitch are orientation angles (not angular velocities). This function penalizes 
    the angular velocities (ω_x, ω_y) only when the commanded orientation angles are near zero.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        angle_threshold: Target orientation threshold (radians), below this value means "target is flat"
                        e.g., 0.05 rad ≈ 2.86°
        asset_cfg: Robot asset configuration
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Allow angular velocity when target orientation requires tilting (avoid conflicts)
        - Penalize unnecessary angular velocity when target is to remain flat (maintain stability)
        - Consider both roll and pitch target angles
        
    Logic:
        IF sqrt(target_roll^2 + target_pitch^2) < threshold:
            Penalize angular velocities ω_x and ω_y (target is flat, shouldn't be rotating)
        ELSE:
            No penalty (target requires tilting, rotation is necessary)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get target roll and pitch angles (orientation commands, not angular velocities)
    command = env.command_manager.get_command(command_name)
    target_roll = command[:, 4]   # Target roll angle (rad)
    target_pitch = command[:, 5]  # Target pitch angle (rad)
    
    # Calculate L2 norm of target orientation
    target_orientation_norm = torch.sqrt(target_roll**2 + target_pitch**2)
    
    # Only penalize angular velocity when target orientation is close to zero (i.e., target is flat)
    should_penalize = target_orientation_norm < angle_threshold
    
    # Calculate sum of squared roll and pitch angular velocities (ω_x and ω_y in body frame)
    penalty = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    
    # Conditionally apply penalty: penalize only when target is flat
    penalty = torch.where(should_penalize, penalty, torch.zeros_like(penalty))
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


def stand_still_full_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stand still penalty considering full command (including height and orientation)
    
    This function replaces the original stand_still by checking all 6D commands to determine
    if the robot is truly "standing still".
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        velocity_threshold: Velocity command threshold (m/s or rad/s)
        height_threshold: Height command change threshold (m)
        angle_threshold: Orientation command threshold (rad)
        asset_cfg: Robot asset configuration
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Only penalize joint deviation when truly standing still (all velocity, height, orientation commands are zero)
        - Allow "standing still but adjusting posture" scenarios
        - Maintain original safety constraint strength
        
    Logic:
        Truly static = (velocity command small) AND (height close to default) AND (orientation close to zero)
        IF truly static:
            Penalize joint deviation from default position
        ELSE:
            No penalty (allow adjustments)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get full 7D command
    command = env.command_manager.get_command(command_name)
    velocity_cmd = command[:, :3]  # [vx, vy, ωz]
    height_cmd = command[:, 3]
    pose_cmd = command[:, 4:7]  # [roll, pitch, yaw]
    
    # Get default height
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    
    # Determine if each dimension is "still"
    velocity_small = torch.norm(velocity_cmd, dim=1) < velocity_threshold
    height_default = torch.abs(height_cmd - default_height) < height_threshold
    pose_zero = torch.norm(pose_cmd, dim=1) < angle_threshold
    
    # Only consider "truly static" when all dimensions are still
    is_truly_static = velocity_small & height_default & pose_zero
    
    # Calculate L1 penalty for joint deviation from default position
    penalty = mdp.joint_deviation_l1(env, asset_cfg)
    
    # Only apply penalty when truly static
    penalty = torch.where(is_truly_static, penalty, torch.zeros_like(penalty))
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
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
    """Joint position penalty considering full command
    
    This function replaces the original joint_pos_penalty, using full 6D command to determine motion state.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        asset_cfg: Robot asset configuration (must include joint_ids)
        stand_still_scale: Penalty scale factor when standing still
        velocity_threshold: Actual velocity threshold (m/s)
        velocity_cmd_threshold: Velocity command threshold (m/s or rad/s)
        height_threshold: Height command change threshold (m)
        angle_threshold: Orientation command threshold (rad)
        
    Returns:
        Penalty values with shape (num_envs,)
        
    Design Intent:
        - Light penalty on joint deviation during motion (maintain flexibility)
        - Heavy penalty on joint deviation when standing still (return to default posture)
        - Consider all 6D commands to determine "motion" state
        
    Logic:
        Motion state = (velocity command large) OR (actual velocity large) OR (height change) OR (orientation change)
        IF moving:
            penalty = 1.0 × base_penalty
        ELSE:
            penalty = 5.0 × base_penalty
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get full 7D command
    command = env.command_manager.get_command(command_name)
    velocity_cmd = command[:, :3]
    height_cmd = command[:, 3]
    pose_cmd = command[:, 4:7]
    
    # Get default height
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    
    # Determine if in motion
    velocity_cmd_norm = torch.norm(velocity_cmd, dim=1)
    body_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    height_cmd_diff = torch.abs(height_cmd - default_height)
    pose_cmd_norm = torch.norm(pose_cmd, dim=1)
    
    # Motion condition: velocity command large OR actual velocity large OR height change OR orientation change
    is_moving = (velocity_cmd_norm > velocity_cmd_threshold) | \
                (body_vel > velocity_threshold) | \
                (height_cmd_diff > height_threshold) | \
                (pose_cmd_norm > angle_threshold)
    
    # Calculate base penalty: L2 norm of joint position deviation from default
    base_penalty = torch.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1
    )
    
    # Adjust penalty strength based on motion state
    # Moving: 1.0× penalty, Standing still: 5.0× penalty
    penalty = torch.where(is_moving, base_penalty, stand_still_scale * base_penalty)
    
    # Only penalize when robot is upright
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


##
# Helper functions
##


def is_moving_full_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
) -> torch.Tensor:
    """Determine if robot is moving (considering full 6D command)
    
    This is a helper function used in other reward functions to determine motion state.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        velocity_threshold: Velocity command threshold
        height_threshold: Height command change threshold
        angle_threshold: Orientation command threshold
        
    Returns:
        Boolean tensor with shape (num_envs,), True indicates moving
        
    Usage:
        Can be used in other reward functions to determine whether to apply certain penalties:
        
        ```python
        is_moving = is_moving_full_cmd(env, "base_velocity_pose")
        penalty = torch.where(is_moving, small_penalty, large_penalty)
        ```
    """
    command = env.command_manager.get_command(command_name)
    
    # Calculate velocity command norm
    velocity_cmd_norm = torch.norm(command[:, :3], dim=1)
    
    # Calculate height command difference
    default_height = env.command_manager.get_term("base_velocity_pose").default_height
    height_cmd_diff = torch.abs(command[:, 3] - default_height)
    
    # Calculate orientation command norm (roll, pitch, yaw)
    pose_cmd_norm = torch.norm(command[:, 4:7], dim=1)
    
    # Consider moving if any dimension has command change
    return (velocity_cmd_norm > velocity_threshold) | \
           (height_cmd_diff > height_threshold) | \
           (pose_cmd_norm > angle_threshold)


##
# Updated existing reward functions to use full command awareness
##


def feet_contact_without_cmd_full(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.1,
    height_threshold: float = 0.02,
    angle_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize feet not in contact with ground when standing still (using full command judgment)
    
    This is an improved version of the original feet_contact_without_cmd, considering height and orientation commands.
    
    Args:
        env: Environment instance
        command_name: Command name in command manager
        sensor_cfg: Contact sensor configuration
        velocity_threshold: Velocity command threshold
        height_threshold: Height command change threshold
        angle_threshold: Orientation command threshold
        
    Returns:
        Penalty values with shape (num_envs,)
    """
    # Use full command to determine if moving
    is_moving = is_moving_full_cmd(env, command_name, velocity_threshold, height_threshold, angle_threshold)
    
    # Get contact sensor data
    contact_sensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0
    
    # Calculate number of feet not in contact
    num_feet = sensor_cfg.body_ids.__len__()
    not_contact_count = num_feet - is_contact.sum(dim=-1)
    
    # Only penalize when standing still
    penalty = torch.where(is_moving, torch.zeros_like(not_contact_count, dtype=torch.float), 
                         not_contact_count.float())
    
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return penalty


def accumulated_ang_vel_penalty_when_standing(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity_pose",
    velocity_threshold: float = 0.1,
    angle_std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize accumulated angular velocity when standing with EXPONENTIAL penalty (deployable to real robot).
    
    This function tracks the INTEGRAL of angular velocity during standing, which represents
    accumulated rotation. Unlike heading_deviation_penalty, this does NOT require world-frame
    yaw angle, making it fully deployable to real robots with only IMU.
    
    ACTIVATION CONDITIONS (ALL must be satisfied):
        1. Curriculum stage >= 1 (enabled from Stage 1 to prevent self-spinning from start)
        2. Robot is grounded (at least 2 feet in contact with ground)
        3. Linear velocity command ≈ 0 (vx, vy < velocity_threshold)
        4. Angular velocity command ≈ 0 (ωz < velocity_threshold) - CRITICAL!
        5. Standing duration > 0.5 seconds (avoid transient effects)
    
    Design Rationale:
    - Real robot IMU provides: angular velocity (gyroscope) 
    - Real robot IMU does NOT provide: absolute yaw angle 
    - Accumulated angular velocity = integral of ωz over time
    - This captures the same "self-spinning" behavior without needing global coordinates
    - Stage-based activation ensures basic locomotion skills are learned first
    - Ground contact check prevents penalizing aerial rotations
    
    Mathematical Formulation:
        accumulated_angle = ∫ωz·dt (during standing)
        penalty = -(exp(|accumulated_angle| / angle_std) - 1)
        
    Exponential Growth Property with angle_std control:
        - angle_std controls the sensitivity of the exponential penalty
        - Smaller angle_std → faster exponential growth → stricter control
        - Larger angle_std → slower exponential growth → gentler control
        - For 10° target: recommend angle_std = 0.15~0.25 (8.6°~14.3°)
    
    Args:
        env: Environment instance
        command_name: Name of the velocity-pose command
        velocity_threshold: Threshold for linear velocity to be considered "standing" (m/s)
                          - Only checks vx and vy, ignores vz
                          - Recommended: 0.05 m/s
        angle_std: Standard deviation controlling penalty sensitivity (rad)
                   - For 10° (0.175 rad) target, use angle_std=0.15-0.20
                   - Smaller angle_std = stricter, larger angle_std = gentler
        asset_cfg: Robot asset configuration
        
    Returns:
        Penalty tensor (num_envs,), negative values when accumulated rotation is large
        Zero penalty if curriculum stage < 2
        
    Penalty Characteristics with angle_std=0.175 (10°) - RECOMMENDED for 10° target:
        - Accumulated ωz = 0 rad (0°):       penalty = 0
        - Accumulated ωz = 0.1 rad (5.7°):   penalty ≈ -0.76  (moderate)
        - Accumulated ωz = 0.175 rad (10°):  penalty ≈ -1.72  (strong) ← matches angle_std (10°)
        - Accumulated ωz = 0.35 rad (20°):   penalty ≈ -6.39  (very strong) ← 2x target
        - Accumulated ωz = 0.5 rad (28.6°):  penalty ≈ -14.15 (extreme!)
        
    Advantages:
        - Fully deployable (only uses IMU angular velocity)
        - No need for world-frame reference
        - Works in both simulation and real robot
        - EXPONENTIAL penalty makes large deviations extremely costly
        - angle_std parameter allows fine-tuning penalty sensitivity
        - Curriculum-aware: activates only when robot has basic skills
        - Ground-contact aware: only penalizes when robot is grounded
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # CRITICAL: Check curriculum stage - MUST be >= 1 to activate
    # Changed from >= 2 to >= 1 to prevent self-spinning from the beginning
    current_stage = getattr(env, "_curriculum_stage", 0)
    if current_stage < 1:
        # Return zero penalty immediately if Stage < 1
        return torch.zeros(env.num_envs, device=env.device)
    
    # Get velocity command
    command = env.command_manager.get_command(command_name)
    
    # Check if robot is grounded (all feet touching ground)
    # Get contact sensor data
    contact_sensor = env.scene.sensors.get("contact_forces", None)
    if contact_sensor is not None:
        # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
        # We only need the most recent forces: [:, 0, :, :]
        net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]  # (num_envs, num_bodies, 3)
        # Check if any foot has contact (force magnitude > threshold)
        force_threshold = 1.0  # N
        foot_contact = torch.norm(net_contact_forces, dim=-1) > force_threshold  # (num_envs, num_bodies)
        # Robot is grounded if at least 2 feet are in contact
        num_feet_contact = foot_contact.sum(dim=-1)  # (num_envs,)
        is_grounded = num_feet_contact >= 2  # (num_envs,)
    else:
        # Fallback: assume always grounded if no contact sensor
        is_grounded = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    
    # Check BOTH linear velocity command AND angular velocity command
    # CRITICAL: Must check angular velocity command (ωz) as well!
    # Only penalize self-spinning when robot is commanded to stand still (no rotation)
    lin_vel_cmd = command[:, :2]  # (num_envs, 2) - [vx, vy]
    ang_vel_cmd = command[:, 2]    # (num_envs,) - [ωz]
    lin_vel_cmd_norm = torch.norm(lin_vel_cmd, dim=1)  # (num_envs,)
    ang_vel_cmd_abs = torch.abs(ang_vel_cmd)  # (num_envs,)
    
    # Determine if robot should be standing still
    # Condition: linear velocity command ≈ 0 AND angular velocity command ≈ 0 AND robot is grounded
    is_standing_command = (lin_vel_cmd_norm < velocity_threshold) & (ang_vel_cmd_abs < velocity_threshold)
    is_standing = is_standing_command & is_grounded  # (num_envs,)
    
    # Get current angular velocity (body frame, IMU can measure this!)
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]  # (num_envs,)
    
    # Initialize accumulators if not exist
    if not hasattr(env, "_accumulated_ang_vel_local"):
        env._accumulated_ang_vel_local = torch.zeros(env.num_envs, device=env.device)
        env._standing_time_local = torch.zeros(env.num_envs, device=env.device)
    
    # CRITICAL: Detect episode reset (when episode just started)
    # Reset accumulators for environments that just reset
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
            torch.zeros_like(env._accumulated_ang_vel_local)   # Reset when moving
        )
    )
    
    # Track standing time (reset when not standing OR when episode resets)
    env._standing_time_local = torch.where(
        just_reset,
        torch.zeros_like(env._standing_time_local),  # Reset when episode resets
        torch.where(
            is_standing,
            env._standing_time_local + dt,  # Continue timing during standing
            torch.zeros_like(env._standing_time_local)  # Reset when moving
        )
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
    penalty = torch.where(
        (is_standing) & (env._standing_time_local > 0.5),
        penalty,
        torch.zeros_like(penalty)
    )
    
    # Debug: Print statistics every 100 steps (DISABLED for performance)
    # if not hasattr(env, "_accumulated_ang_vel_penalty_debug_counter"):
    #     env._accumulated_ang_vel_penalty_debug_counter = 0
    # env._accumulated_ang_vel_penalty_debug_counter += 1
    # if env._accumulated_ang_vel_penalty_debug_counter % 100 == 0:
    #     ... (all debug print statements disabled)
    
    return penalty
    
    return penalty

##
# ARX5 Arm Stability and Anti-Flip Rewards (Stage 1)
##


def combined_com_stability_reward(
    env: ManagerBasedRLEnv,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dog_mass: float = 15.0,
    arm_mass: float = 3.0,
    target_com_offset: tuple = (0.0, 0.0, 0.0),
    std: float = 0.10,
    arm_body_names: list[str] = ["link1", "link2", "link3", "link4", "link5", "link6"],
) -> torch.Tensor:
    """Reward for keeping combined CoM (dog + arm) close to target position.
    
    The combined center of mass should ideally be above the dog's base to maintain stability.
    
    Args:
        env: Environment instance.
        dog_cfg: Dog asset configuration.
        dog_mass: Total mass of the dog (kg).
        arm_mass: Total mass of the arm (kg).
        target_com_offset: Target CoM offset from base (x, y, z) in meters.
        std: Standard deviation for exponential reward (m).
        arm_body_names: Names of arm link bodies.
    
    Returns:
        Reward values with shape (num_envs,).
    """
    asset: Articulation = env.scene[dog_cfg.name]
    
    # Approximate dog CoM as base position
    dog_com = asset.data.root_pos_w
    
    # Calculate arm CoM as average of all link positions
    try:
        arm_body_indices = [asset.find_bodies(name)[0][0] for name in arm_body_names]
        arm_link_positions = torch.stack([
            asset.data.body_pos_w[:, idx, :] for idx in arm_body_indices
        ], dim=1)  # (num_envs, num_links, 3)
        arm_com = arm_link_positions.mean(dim=1)  # (num_envs, 3)
    except:
        # Fallback if arm bodies not found
        arm_com = dog_com
    
    # Calculate combined CoM
    total_mass = dog_mass + arm_mass
    combined_com = (dog_mass * dog_com + arm_mass * arm_com) / total_mass
    
    # Calculate offset from base
    com_offset = combined_com - dog_com
    
    # Target offset
    target = torch.tensor(target_com_offset, device=env.device, dtype=torch.float32)
    
    # Calculate error
    error = torch.norm(com_offset - target, dim=-1)
    
    # Exponential reward
    reward = torch.exp(-error / std)
    
    return reward


def base_stability_reward(
    env: ManagerBasedRLEnv,
    lin_acc_std: float = 3.0,
    ang_acc_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward smooth base motion by penalizing large accelerations.
    
    Args:
        env: Environment instance.
        lin_acc_std: Standard deviation for linear acceleration (m/s²).
        ang_acc_std: Standard deviation for angular acceleration (rad/s²).
        asset_cfg: Asset configuration.
    
    Returns:
        Reward values with shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get linear acceleration (approximate from velocity change)
    if hasattr(asset.data, 'root_lin_acc_w'):
        lin_acc = asset.data.root_lin_acc_w
    else:
        # Fallback: approximate from velocity
        lin_acc = torch.zeros(env.num_envs, 3, device=env.device)
    
    # Get angular velocity change as proxy for angular acceleration
    ang_vel = asset.data.root_ang_vel_w
    
    # Calculate magnitudes
    lin_acc_mag = torch.norm(lin_acc, dim=-1)
    ang_vel_mag = torch.norm(ang_vel, dim=-1)
    
    # Exponential reward for smooth motion
    lin_reward = torch.exp(-lin_acc_mag / lin_acc_std)
    ang_reward = torch.exp(-ang_vel_mag / ang_acc_std)
    
    return (lin_reward + ang_reward) / 2.0


def feet_contact_force_balance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_distribution: list[float] = [0.25, 0.25, 0.25, 0.25],
) -> torch.Tensor:
    """Reward balanced contact force distribution across all feet.
    
    Args:
        env: Environment instance.
        sensor_cfg: Contact sensor configuration.
        target_distribution: Target force distribution for each foot (should sum to 1.0).
    
    Returns:
        Reward values with shape (num_envs,).
    """
    contact_sensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_magnitudes = torch.norm(contact_forces, dim=-1).max(dim=1)[0]  # (num_envs, num_feet)
    
    # Calculate force distribution
    total_force = force_magnitudes.sum(dim=-1, keepdim=True) + 1e-6  # Avoid division by zero
    force_distribution = force_magnitudes / total_force  # (num_envs, num_feet)
    
    # Target distribution
    target = torch.tensor(target_distribution, device=env.device, dtype=torch.float32)
    
    # Calculate error
    error = torch.norm(force_distribution - target, dim=-1)
    
    # Reward (lower error = higher reward)
    reward = torch.exp(-error / 0.2)  # std=0.2 for distribution error
    
    return reward


def anti_flip_orientation_reward(
    env: ManagerBasedRLEnv,
    roll_threshold: float = 0.785,   # 45°
    pitch_threshold: float = 0.524,  # 30°
    penalize_flip_attempt: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strong penalty for approaching flip/roll-over orientations.
    
    This is a critical safety reward to prevent the robot from flipping over,
    which could damage the arm.
    
    Args:
        env: Environment instance.
        roll_threshold: Roll angle threshold (rad).
        pitch_threshold: Pitch angle threshold (rad).
        penalize_flip_attempt: Whether to strongly penalize approaching the threshold.
        asset_cfg: Asset configuration.
    
    Returns:
        Reward values with shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Extract roll and pitch from quaternion
    quat = asset.data.root_quat_w
    # Use isaaclab's math utilities
    from isaaclab.utils.math import euler_xyz_from_quat
    roll, pitch, _ = euler_xyz_from_quat(quat)
    
    # Calculate absolute errors
    roll_error = torch.abs(roll)
    pitch_error = torch.abs(pitch)
    
    # Normal range: linear reward
    roll_reward = torch.where(
        roll_error < roll_threshold * 0.5,
        1.0 - roll_error / (roll_threshold * 0.5),
        torch.zeros_like(roll_error)
    )
    
    pitch_reward = torch.where(
        pitch_error < pitch_threshold * 0.5,
        1.0 - pitch_error / (pitch_threshold * 0.5),
        torch.zeros_like(pitch_error)
    )
    
    # Approaching flip: strong penalty
    if penalize_flip_attempt:
        roll_penalty = torch.where(
            roll_error > roll_threshold * 0.7,
            -(roll_error / roll_threshold - 0.7) * 10.0,
            torch.zeros_like(roll_error)
        )
        
        pitch_penalty = torch.where(
            pitch_error > pitch_threshold * 0.7,
            -(pitch_error / pitch_threshold - 0.7) * 10.0,
            torch.zeros_like(pitch_error)
        )
    else:
        roll_penalty = 0.0
        pitch_penalty = 0.0
    
    # Combine rewards
    reward = (roll_reward + pitch_reward) / 2.0 + (roll_penalty + pitch_penalty)
    
    # Only activate when robot is upright
    upright_factor = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    reward = reward * upright_factor
    
    return reward


def upright_bonus_reward(
    env: ManagerBasedRLEnv,
    target_gravity_z: float = -1.0,
    tolerance: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus reward for maintaining upright orientation.
    
    Args:
        env: Environment instance.
        target_gravity_z: Target z-component of projected gravity in body frame.
        tolerance: Tolerance for perfect upright posture.
        asset_cfg: Asset configuration.
    
    Returns:
        Reward values with shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get gravity projection in body frame
    gravity_z = asset.data.projected_gravity_b[:, 2]
    
    # Calculate error
    error = torch.abs(gravity_z - target_gravity_z)
    
    # Exponential reward
    reward = torch.exp(-error / tolerance)
    
    return reward


def arm_joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
    soft_margin: float = 0.1,
) -> torch.Tensor:
    """Penalty for arm joints approaching limits.
    
    Args:
        env: Environment instance.
        asset_cfg: Asset configuration for arm joints.
        soft_margin: Soft margin as fraction of range (0.1 = 10%).
    
    Returns:
        Penalty values with shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.resolve_joint_indices(asset_cfg.joint_names, asset)
    
    # Get joint positions and limits
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits[:, joint_ids, :]
    
    # Calculate normalized position in range [-1, 1]
    joint_range = joint_limits[:, :, 1] - joint_limits[:, :, 0]
    joint_center = (joint_limits[:, :, 1] + joint_limits[:, :, 0]) / 2.0
    normalized_pos = (joint_pos - joint_center) / (joint_range / 2.0)
    
    # Penalty when outside soft margin
    soft_threshold = 1.0 - soft_margin
    penalty = torch.relu(torch.abs(normalized_pos) - soft_threshold)
    
    return penalty.sum(dim=-1)


def arm_joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
) -> torch.Tensor:
    """L2 penalty on arm joint accelerations for smooth motion.
    
    Args:
        env: Environment instance.
        asset_cfg: Asset configuration for arm joints.
    
    Returns:
        Penalty values with shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.resolve_joint_indices(asset_cfg.joint_names, asset)
    
    # Get joint accelerations
    joint_acc = asset.data.joint_acc[:, joint_ids]
    
    return torch.sum(torch.square(joint_acc), dim=-1)
