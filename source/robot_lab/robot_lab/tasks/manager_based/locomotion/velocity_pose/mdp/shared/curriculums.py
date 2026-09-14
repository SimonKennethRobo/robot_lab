# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Curriculum learning functions for VelocityPose task with stage-based progression.

This module implements a fixed-iteration stage-based curriculum:

- Stage 1 (0-20,000 iterations): Base training - Height and pose commands fixed at default (roll=0°, pitch=0°, yaw=0°, height=0.33m)
- Stage 2 (20,000-25,000 iterations): Small range - Height and pose commands with limited variation (±3cm height, ±8° roll, pitch=0°, yaw=0°)
- Stage 3 (25,000-30,000 iterations): Medium range for height and pose commands (±10cm height, ±35° roll, ±20° pitch, yaw=0°)
- Stage 4 (30,000-35,000 iterations): Large range for height and pose commands (±12.5cm height, ±45° roll, ±25° pitch, yaw=0°)
  * Stage 4 ranges based on real robot rosbag analysis (2026-01-28): real robot achieved roll [-40.73°, +39.05°], pitch [-23.29°, +24.91°]
- Stage 5 (35,000+ iterations): Same as Stage 4, but with 1.5x arm motion amplitude for extreme disturbance training

NOTE: Yaw is ALWAYS fixed at 0° to decouple from localization systems. Only roll and pitch are tracked,
which can be determined from IMU gravity projection alone without external localization.

The curriculum automatically tracks total iterations across training sessions,
so --resume will correctly continue from the accumulated iteration count.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _update_reward_parameters(env: ManagerBasedRLEnv, stage: int):
    
    # Try to get pose tracking reward terms (may not exist in all environments)
    try:
        height_reward_cfg = env.reward_manager.get_term_cfg("track_height_exp")
        orient_reward_cfg = env.reward_manager.get_term_cfg("track_orientation_exp")
    except (AttributeError, KeyError, ValueError):
        height_reward_cfg = None
        orient_reward_cfg = None
    
    # Try to get yaw angular velocity tracking reward (prevents self-spinning)
    try:
        ang_vel_z_tracking_cfg = env.reward_manager.get_term_cfg("track_ang_vel_z_exp")
    except (AttributeError, KeyError, ValueError):
        ang_vel_z_tracking_cfg = None
    
    # Try to get locomotion penalty terms
    try:
        lin_vel_z_l2_cfg = env.reward_manager.get_term_cfg("lin_vel_z_l2")
    except (AttributeError, KeyError):
        lin_vel_z_l2_cfg = None
    
    try:
        ang_vel_xy_l2_cfg = env.reward_manager.get_term_cfg("ang_vel_xy_l2")
    except (AttributeError, KeyError):
        ang_vel_xy_l2_cfg = None
    
    # Try to get upward reward (encourages keeping base upright)
    try:
        upward_cfg = env.reward_manager.get_term_cfg("upward")
    except (AttributeError, KeyError):
        upward_cfg = None
    
    # Try to get accumulated angular velocity penalty (anti-spinning when standing)
    try:
        accumulated_ang_vel_cfg = env.reward_manager.get_term_cfg("accumulated_ang_vel_standing")
    except (AttributeError, KeyError):
        accumulated_ang_vel_cfg = None
    
    # Stage 1: Disable pose tracking rewards, enable locomotion penalties and upward reward
    if stage == 1:
        if height_reward_cfg:
            height_reward_cfg.weight = 0.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_height_exp"] = 0.0
        
        if orient_reward_cfg:
            orient_reward_cfg.weight = 0.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_orientation_exp"] = 0.0
        
        # Yaw control: relaxed tolerance (allow some drift during basic locomotion learning)
        if ang_vel_z_tracking_cfg:
            ang_vel_z_tracking_cfg.params["std"] = math.sqrt(0.25)  
        
        # Enable locomotion penalties in Stage 1 only
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = -2.0  
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0  
        
        # Enable upward reward in Stage 1 to maintain stability during basic locomotion learning
        if upward_cfg:
            upward_cfg.weight = 1.0
        
        # Accumulated angular velocity penalty: ENABLED in Stage 1 to prevent self-spinning from start
        if accumulated_ang_vel_cfg:
            accumulated_ang_vel_cfg.weight = 0.5 
            accumulated_ang_vel_cfg.params["angle_std"] = math.radians(20)  

    # Stage 2: Enable pose tracking with relaxed tolerance, disable locomotion penalties and upward
    elif stage == 2:
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.25)  
            height_reward_cfg.weight = 4.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_height_exp"] = 4.0
        
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.50)  
            orient_reward_cfg.weight = 4.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_orientation_exp"] = 4.0
        
        if ang_vel_z_tracking_cfg:
            ang_vel_z_tracking_cfg.params["std"] = math.sqrt(0.10)  
        
        # Disable locomotion penalties from Stage 2 onwards
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = 0.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0
        
        # CRITICAL: Disable upward reward from Stage 2 onwards (conflicts with pose tracking)
        if upward_cfg:
            upward_cfg.weight = 0.0
        
        if accumulated_ang_vel_cfg:
            accumulated_ang_vel_cfg.weight = 0.5
            accumulated_ang_vel_cfg.params["angle_std"] = math.radians(20)

    # Stage 3: Strict tracking with high weight, upward remains disabled
    elif stage == 3:
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.05)  
            height_reward_cfg.weight = 12.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_height_exp"] = 12.0
        
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.10)  
            orient_reward_cfg.weight = 12.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_orientation_exp"] = 12.0
        
        if ang_vel_z_tracking_cfg:
            ang_vel_z_tracking_cfg.params["std"] = math.sqrt(0.025)  
        
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = 0.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0
        
        # Keep upward disabled
        if upward_cfg:
            upward_cfg.weight = 0.0
        
        if accumulated_ang_vel_cfg:
            accumulated_ang_vel_cfg.weight = 1.0
            accumulated_ang_vel_cfg.params["angle_std"] = math.radians(15)

    # Stage 4: Very strict tracking with very high weight, upward remains disabled
    elif stage == 4:
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.05)  
            height_reward_cfg.weight = 16.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_height_exp"] = 16.0
        
        # Orientation tracking: very strict tolerance, very high weight
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.10) 
            orient_reward_cfg.weight = 16.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_orientation_exp"] = 16.0
        
        # Yaw control: very strict tolerance (near-zero yaw drift tolerance)
        if ang_vel_z_tracking_cfg:
            ang_vel_z_tracking_cfg.params["std"] = math.sqrt(0.015)  
        
        # Keep locomotion penalties disabled
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = 0.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0
        
        # Keep upward disabled
        if upward_cfg:
            upward_cfg.weight = 0.0
        
        # Anti-spin when standing: moderate control (5° tolerance)
        # NOTE: Function returns negative values, so weight should be POSITIVE
        if accumulated_ang_vel_cfg:
            accumulated_ang_vel_cfg.weight = 1.5
            accumulated_ang_vel_cfg.params["angle_std"] = math.radians(5)

    # Stage 5: Same as Stage 4, but with 1.5x arm motion amplitude (extreme disturbance)
    elif stage == 5:
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.05)  
            height_reward_cfg.weight = 16.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_height_exp"] = 16.0
        
        # Orientation tracking: very strict tolerance, very high weight
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.10) 
            orient_reward_cfg.weight = 16.0
            if hasattr(env.reward_manager, '_term_weights'):
                env.reward_manager._term_weights["track_orientation_exp"] = 16.0
        
        # Yaw control: very strict tolerance (near-zero yaw drift tolerance)
        if ang_vel_z_tracking_cfg:
            ang_vel_z_tracking_cfg.params["std"] = math.sqrt(0.015)  
        
        # Keep locomotion penalties disabled
        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = 0.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0
        
        # Keep upward disabled
        if upward_cfg:
            upward_cfg.weight = 0.0
        
        # Anti-spin when standing: moderate control (5° tolerance)
        # NOTE: Function returns negative values, so weight should be POSITIVE
        if accumulated_ang_vel_cfg:
            accumulated_ang_vel_cfg.weight = 1.5
            accumulated_ang_vel_cfg.params["angle_std"] = math.radians(5)


def _print_reward_parameters(env: ManagerBasedRLEnv):
    """Print current reward parameters for debugging. (DISABLED for performance)"""
    pass
    # import math
    
    # print("  Reward Parameters:")
    
    # # Print pose tracking rewards
    # try:
    #     height_reward_cfg = env.reward_manager.get_term_cfg("track_height_exp")
    #     height_std = height_reward_cfg.params.get("std", 0.0)
    #     height_weight = height_reward_cfg.weight
    #     print(f"    track_height_exp:       weight={height_weight:.1f}, std={height_std:.3f} ({height_std:.2f}m)")
    # except (AttributeError, KeyError, ValueError):
    #     print("    track_height_exp:       Not configured")
    
    # try:
    #     orient_reward_cfg = env.reward_manager.get_term_cfg("track_orientation_exp")
    #     orient_std = orient_reward_cfg.params.get("std", 0.0)
    #     orient_weight = orient_reward_cfg.weight
    #     print(f"    track_orientation_exp:  weight={orient_weight:.1f}, std={orient_std:.3f} ({math.degrees(orient_std):.1f}°)")
    # except (AttributeError, KeyError, ValueError):
    #     print("    track_orientation_exp:  Not configured")
    
    # # Print yaw angular velocity tracking (prevents self-spinning)
    # try:
    #     ang_vel_z_cfg = env.reward_manager.get_term_cfg("track_ang_vel_z_exp")
    #     ang_vel_z_std = ang_vel_z_cfg.params.get("std", 0.0)
    #     ang_vel_z_weight = ang_vel_z_cfg.weight
    #     print(f"    track_ang_vel_z_exp:    weight={ang_vel_z_weight:.1f}, std={ang_vel_z_std:.3f} ({ang_vel_z_std:.2f} rad/s)")
    # except (AttributeError, KeyError):
    #     print("    track_ang_vel_z_exp:    Not configured")
    
    # # Print locomotion penalty rewards
    # try:
    #     lin_vel_z_cfg = env.reward_manager.get_term_cfg("lin_vel_z_l2")
    #     lin_vel_z_weight = lin_vel_z_cfg.weight
    #     print(f"    lin_vel_z_l2:           weight={lin_vel_z_weight:.2f}")
    # except (AttributeError, KeyError):
    #     print("    lin_vel_z_l2:           Not configured")
    
    # try:
    #     ang_vel_xy_cfg = env.reward_manager.get_term_cfg("ang_vel_xy_l2")
    #     ang_vel_xy_weight = ang_vel_xy_cfg.weight
    #     print(f"    ang_vel_xy_l2:          weight={ang_vel_xy_weight:.2f}")
    # except (AttributeError, KeyError):
    #     print("    ang_vel_xy_l2:          Not configured")
    
    # # Print upward reward (stability vs pose tracking trade-off)
    # try:
    #     upward_cfg = env.reward_manager.get_term_cfg("upward")
    #     upward_weight = upward_cfg.weight
    #     print(f"    upward:                 weight={upward_weight:.2f}")
    # except (AttributeError, KeyError):
    #     print("    upward:                 Not configured")
    
    # # Print anti-spin penalty when standing
    # try:
    #     accumulated_ang_vel_cfg = env.reward_manager.get_term_cfg("accumulated_ang_vel_standing")
    #     accumulated_weight = accumulated_ang_vel_cfg.weight
    #     angle_std = accumulated_ang_vel_cfg.params.get("angle_std", 0.0)
    #     print(f"    accumulated_ang_vel_standing: weight={accumulated_weight:.1f}, angle_std={angle_std:.3f} ({math.degrees(angle_std):.1f}°)")
    # except (AttributeError, KeyError):
    #     print("    accumulated_ang_vel_standing: Not configured")


def terrain_levels_velocity_pose(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str
) -> torch.Tensor:
    """Curriculum that updates the terrain levels based on velocity and pose tracking performance.
    
    This adapts the original terrain curriculum to also consider height and orientation tracking.
    """
    # Get the current terrain levels
    terrain_levels = env.terrain_levels.float()
    
    # Get tracking rewards
    vel_tracking_reward = env.reward_manager._episode_sums.get("track_lin_vel_xy_exp", 0.0)
    ang_tracking_reward = env.reward_manager._episode_sums.get("track_ang_vel_z_exp", 0.0)
    height_tracking_reward = env.reward_manager._episode_sums.get("track_height_exp", 0.0)
    pose_tracking_reward = env.reward_manager._episode_sums.get("track_orientation_exp", 0.0)
    
    # Combine all tracking rewards (weighted average)
    total_tracking = (
        vel_tracking_reward * 0.4 + 
        ang_tracking_reward * 0.2 + 
        height_tracking_reward * 0.2 + 
        pose_tracking_reward * 0.2
    )
    
    # Normalize by episode length and get weight
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    normalized_reward = total_tracking[env_ids] / env.max_episode_length_s
    
    # Increase terrain level if tracking is good (>80% of max reward)
    increase_level = normalized_reward > 0.8 * (
        reward_term_cfg.weight * 0.4 +  # vel weight portion
        2.0 * 0.2 +  # height weight (from config)
        1.0 * 0.2    # pose weight (from config)
    )
    
    # Decrease terrain level if robot dies (episode ended but not timeout)
    decrease_level = ~env.termination_manager.get_term("time_out").time_out[env_ids]
    
    # Update terrain levels
    terrain_levels[env_ids] += increase_level.float()
    terrain_levels[env_ids] -= decrease_level.float()
    
    # Clamp terrain levels to valid range
    terrain_levels[env_ids] = torch.clamp(terrain_levels[env_ids], min=0)
    
    return terrain_levels[env_ids]


def command_curriculum_height_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity_pose",
) -> torch.Tensor:

    # Get command manager ranges
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    default_height = command_term.default_height
    
    # Calculate total iterations from environment step counter

    # Try to get iteration from various sources (in priority order):
    if hasattr(env, '_curriculum_manual_iteration'):
        # Manually injected by a wrapper (most reliable)
        total_iterations = env._curriculum_manual_iteration  
        iteration_source = "manual_injection"
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_rsl_rl_runner'):
        # Get iteration from RSL-RL runner (injected by train.py)
        runner = env.unwrapped._rsl_rl_runner  
        if hasattr(runner, 'current_learning_iteration'):
            total_iterations = runner.current_learning_iteration
            iteration_source = "rsl_rl_runner"
        else:
            # Fallback to calculation
            total_steps = env.common_step_counter
            steps_per_iteration = 24
            total_iterations = total_steps // steps_per_iteration
            iteration_source = "rsl_rl_fallback"
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_current_iteration'):
        # Check if the environment tracks current iteration
        total_iterations = env.unwrapped._current_iteration  # type: ignore
        iteration_source = "env_tracked_iteration"
    elif hasattr(env.unwrapped, 'episode_length_buf'):
        # Calculate from episode counter (for RSL-RL)
        total_steps = env.common_step_counter
        steps_per_iteration = 24  # RSL-RL default: 24 steps per env per iteration
        total_iterations = total_steps // steps_per_iteration
        iteration_source = "rsl_rl_calculation"
    else:
        # Last resort fallback
        total_steps = env.common_step_counter  
        steps_per_iteration = 24  # Assume RSL-RL default
        total_iterations = total_steps // steps_per_iteration
        iteration_source = "fallback_calculation"
        
        if hasattr(env, '_curriculum_stage'):
            expected_stage_for_iter = (
                1 if total_iterations < 20000 else  
                2 if total_iterations < 25000 else  
                3 if total_iterations < 30000 else
                4 if total_iterations < 35000 else
                5
            )
            if env._curriculum_stage != expected_stage_for_iter:  
                if not hasattr(env, '_curriculum_resume_warning_shown'):
                    # (DISABLED for performance - only warns once on resume)
                    # print(f"\n{'!'*80}")
                    # print("[WARNING] Curriculum iteration mismatch detected!")
                    # print(f"  Calculated iteration from steps: {total_iterations:,}")
                    # print(f"  Current stage: {env._curriculum_stage}")  # type: ignore
                    # print(f"  Expected stage for this iteration: {expected_stage_for_iter}")
                    # print("  This likely means training was resumed with --resume.")
                    # print("  Curriculum will maintain current stage until next boundary.")
                    # print(f"{'!'*80}\n")
                    env._curriculum_resume_warning_shown = True  # type: ignore
    
    # DEBUG: Print to verify iteration counting (DISABLED for performance)
    # if not hasattr(env, "_curriculum_debug_counter"):
    #     env._curriculum_debug_counter = 0  # type: ignore
    #     print(f"\n{'='*80}")
    #     print("[Curriculum] Initialization - 4-Stage Curriculum Enabled")
    #     print("  Stage 1: 0-20,000 iterations (Fixed pose at default)")
    #     print("  Stage 2: 20,000-25,000 iterations (±3cm, ±8° roll)")
    #     print("  Stage 3: 25,000-30,000 iterations (±10cm, ±20° roll, ±12° pitch/yaw)")
    #     print("  Stage 4: 30,000+ iterations (±15cm, ±30° roll, ±15° pitch/yaw)")
    #     print(f"  num_envs: {env.num_envs}")
    #     print(f"  Iteration source: {iteration_source}")
    #     print(f"  Current total_iterations: {total_iterations:,}")
    #     print(f"  common_step_counter: {env.common_step_counter:,}")
    #     print(f"{'='*80}\n")
        
    # env._curriculum_debug_counter += 1 
    # if env._curriculum_debug_counter % 1000 == 0:  
    #     print(f"\n[DEBUG Curriculum] Call {env._curriculum_debug_counter}:") 
    #     print(f"  Iteration source: {iteration_source}")
    #     print(f"  common_step_counter: {env.common_step_counter:,}")
    #     print(f"  total_iterations: {total_iterations:,}")
    #     print(f"  Current _curriculum_stage: {getattr(env, '_curriculum_stage', 'NOT SET')}")
    
    
    # This check must happen BEFORE stage calculation to ensure it persists across resets
    # Only consider it inference mode if EXPLICITLY marked or if runner is None (not just missing)
    is_inference_mode = (
        hasattr(env.unwrapped, '_is_inference_mode') or  # Explicitly marked by play.py
        (hasattr(env.unwrapped, '_rsl_rl_runner') and env.unwrapped._rsl_rl_runner is None)  # type: ignore
    )
    
    # Check if a specific inference stage was requested (from play.py --curriculum_stage argument)
    inference_stage = getattr(env.unwrapped, '_inference_curriculum_stage', 4)  # Default to Stage 4 if not set
    
    # FORCE Stage 1 at iteration 0 to avoid incorrect initialization
    if total_iterations == 0 and not is_inference_mode:
        target_stage = 1
        height_range = (default_height, default_height)
        roll_range = (0.0, 0.0)
        pitch_range = (0.0, 0.0)
        yaw_range = (0.0, 0.0)
        arm_motion_scale = 0.0
        arm_frequency_range = (0.3, 0.8)
        arm_amplitude_range = (0.3, 0.6)
    elif is_inference_mode:
        # Use the requested inference stage (from --curriculum_stage argument)
        target_stage = inference_stage

        # Set ranges based on the requested stage
        if target_stage == 1:
            height_range = (default_height, default_height)
            roll_range = (0.0, 0.0)
            pitch_range = (0.0, 0.0)
            yaw_range = (0.0, 0.0)
            arm_motion_scale = 0.0
            arm_frequency_range = (0.3, 0.8)
            arm_amplitude_range = (0.3, 0.6)
        elif target_stage == 2:
            height_range = (0.30, 0.36)
            roll_range = (-0.14, 0.14)
            pitch_range = (0.0, 0.0)
            yaw_range = (0.0, 0.0)
            arm_motion_scale = 1.2
            arm_frequency_range = (0.5, 1.2)
            arm_amplitude_range = (0.4, 0.8)
        elif target_stage == 3:
            height_range = (0.23, 0.43)
            roll_range = (-0.611, 0.611)
            pitch_range = (-0.349, 0.349)
            yaw_range = (0.0, 0.0)
            arm_motion_scale = 1.8
            arm_frequency_range = (0.8, 1.8)
            arm_amplitude_range = (0.5, 1.0)
        elif target_stage == 4:
            height_range = (0.18, 0.43)
            roll_range = (-0.524, 0.524)
            pitch_range = (-0.262, 0.262)
            yaw_range = (0.0, 0.0)
            arm_motion_scale = 2.5
            arm_frequency_range = (1.0, 2.5)
            arm_amplitude_range = (0.6, 1.2)
        else:  # Stage 5
            height_range = (0.18, 0.43)
            roll_range = (-0.524, 0.524)
            pitch_range = (-0.262, 0.262)
            yaw_range = (0.0, 0.0)
            arm_motion_scale = 3.75
            arm_frequency_range = (1.0, 2.5)
            arm_amplitude_range = (0.9, 1.8)

        # Print message only once per session
        if not hasattr(env, "_curriculum_inference_message_shown"):
            env._curriculum_inference_message_shown = True  # type: ignore
            print(f"\n{'='*80}")
            print("[Curriculum] INFERENCE MODE DETECTED")
            print(f"  Using Stage {target_stage} as requested")
            print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
            print(f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
            print(f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
            print(f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
            print(f"  Arm Motion Scale: {arm_motion_scale:.2f}")
            print(f"  Arm Frequency Range: [{arm_frequency_range[0]:.1f}, {arm_frequency_range[1]:.1f}] Hz")
            print(f"  Arm Amplitude Range: [{arm_amplitude_range[0]:.1f}, {arm_amplitude_range[1]:.1f}] rad")
            print(f"{'='*80}\n")
    elif total_iterations < 20000:  # Stage 1: Base training (0-20k iterations)
        target_stage = 1
        height_range = (default_height, default_height)  
        roll_range = (0.0, 0.0) 
        pitch_range = (0.0, 0.0)  
        yaw_range = (0.0, 0.0)
        # Arm motion parameters - Stage 1: No motion
        arm_motion_scale = 0.0
        arm_frequency_range = (0.3, 0.8)
        arm_amplitude_range = (0.3, 0.6)
    elif total_iterations < 25000:  # Stage 2: Height variation (20k-25k iterations)
        target_stage = 2
        height_range = (0.30, 0.36) 
        roll_range = (-0.14, 0.14)  
        pitch_range = (0.0, 0.0)  
        yaw_range = (0.0, 0.0)
        # Arm motion parameters - Stage 2: Moderate motion
        arm_motion_scale = 1.2
        arm_frequency_range = (0.5, 1.2)
        arm_amplitude_range = (0.4, 0.8)
    elif total_iterations < 30000:  # Stage 3: Full pose control (25k-30k iterations)
        target_stage = 3
        height_range = (0.23, 0.43)  
        roll_range = (-0.611, 0.611)  
        pitch_range = (-0.349, 0.349)  
        yaw_range = (0.0, 0.0)
        # Arm motion parameters - Stage 3: Active motion
        arm_motion_scale = 1.8
        arm_frequency_range = (0.8, 1.8)
        arm_amplitude_range = (0.5, 1.0)
    elif total_iterations < 35000:  # Stage 4: Maximum range (30k-35k iterations)
        target_stage = 4
        height_range = (0.18, 0.43)  
        roll_range = (-0.785, 0.785)  
        pitch_range = (-0.436, 0.436)  
        yaw_range = (0.0, 0.0)
        # Arm motion parameters - Stage 4: High motion
        arm_motion_scale = 2.5
        arm_frequency_range = (1.0, 2.5)
        arm_amplitude_range = (0.6, 1.2)
    else:  # Stage 5: Same as Stage 4 but with 1.5x arm motion (35k+ iterations)
        target_stage = 5
        height_range = (0.18, 0.43)  
        roll_range = (-0.785, 0.785)  
        pitch_range = (-0.436, 0.436)  
        yaw_range = (0.0, 0.0)
        # Arm motion parameters - Stage 5: Extreme motion (1.5x Stage 4)
        arm_motion_scale = 3.75
        arm_frequency_range = (1.0, 2.5)
        arm_amplitude_range = (0.9, 1.8) 
    
    # Initialize curriculum state on first call, using the target_stage we just determined
    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = target_stage
        env._curriculum_last_update = 0

        # IMPORTANT: Set initial command ranges based on starting stage
        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        ranges.yaw = yaw_range

        # Store arm motion parameters in env for arm controller to access
        env._arm_motion_scale = arm_motion_scale
        env._arm_frequency_range = arm_frequency_range
        env._arm_amplitude_range = arm_amplitude_range

        # Set initial reward parameters based on starting stage
        _update_reward_parameters(env, target_stage)

        print(f"\n{'='*80}")
        print(f"[Curriculum] Initialized at iteration {total_iterations}")
        print(f"  Starting Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
        print(f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
        print(f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
        print(f"  Arm Motion Scale: {arm_motion_scale:.2f}")
        print(f"  Arm Frequency Range: [{arm_frequency_range[0]:.1f}, {arm_frequency_range[1]:.1f}] Hz")
        print(f"  Arm Amplitude Range: [{arm_amplitude_range[0]:.1f}, {arm_amplitude_range[1]:.1f}] rad")
        _print_reward_parameters(env)
        print(f"{'='*80}\n")

    # Update ranges if stage changed
    if target_stage != env._curriculum_stage:
        env._curriculum_stage = target_stage
        env._curriculum_last_update = total_iterations

        # Update command ranges
        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        ranges.yaw = yaw_range

        # Update arm motion parameters
        env._arm_motion_scale = arm_motion_scale
        env._arm_frequency_range = arm_frequency_range
        env._arm_amplitude_range = arm_amplitude_range

        # Update reward parameters for new stage
        _update_reward_parameters(env, target_stage)

        # Print stage transition message
        print(f"\n{'='*80}")
        print(f"[Curriculum] Stage Transition at Iteration {total_iterations}")
        print(f"{'='*80}")
        print(f"  New Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
        print(f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
        print(f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
        print(f"  Arm Motion Scale: {arm_motion_scale:.2f}")
        print(f"  Arm Frequency Range: [{arm_frequency_range[0]:.1f}, {arm_frequency_range[1]:.1f}] Hz")
        print(f"  Arm Amplitude Range: [{arm_amplitude_range[0]:.1f}, {arm_amplitude_range[1]:.1f}] rad")
        _print_reward_parameters(env)
        print(f"{'='*80}\n")
    
    # Log current stage every 100 iterations with detailed command ranges (DISABLED for performance)
    # if total_iterations > 0 and total_iterations % 100 == 0 and total_iterations != env._curriculum_last_update:  
    #     iterations_in_stage = total_iterations - (
    #         0 if target_stage == 1 else       
    #         20000 if target_stage == 2 else    
    #         25000 if target_stage == 3 else   
    #         30000                             
    #     )
    #     stage_total = (
    #         20000 if target_stage == 1 else
    #         25000 if target_stage == 2 else
    #         30000 if target_stage == 3 else
    #         35000
    #     )
    #     progress = min(100.0, (iterations_in_stage / stage_total) * 100) if stage_total > 0 else 0.0
    #     
    #     print(f"\n{'='*80}")
    #     print(f"[Curriculum Progress] Iteration {total_iterations}")
    #     print(f"{'='*80}")
    #     print(f"  Current Stage: {target_stage}")
    #     print(f"  Stage Progress: {iterations_in_stage}/{stage_total} iterations ({progress:.1f}%)")
    #     print("  Command Ranges:")
    #     print(f"    Height: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
    #     print(f"    Roll:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = [{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°")
    #     print(f"    Pitch:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = [{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°")
    #     print(f"    Yaw:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = [{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°")
    #     print(f"{'='*80}\n")
    
    # Return current stage as scalar for logging (take mean of all envs, which are all the same)
    return torch.tensor(float(target_stage), device=env.device)


def arm_randomization_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    threshold_iteration: int = 5000,
) -> torch.Tensor:
    """Enable arm randomization after a threshold number of iterations.

    This curriculum function enables arm trajectory motion after the policy
    has learned basic locomotion skills.

    Args:
        env: The environment instance.
        env_ids: Environment indices (not used, but required by curriculum interface).
        threshold_iteration: Iteration threshold to enable arm randomization.

    Returns:
        Tensor with current arm randomization state (0=disabled, 1=enabled).
    """
    # Get current iteration
    if hasattr(env, '_curriculum_manual_iteration'):
        total_iterations = env._curriculum_manual_iteration
    elif hasattr(env.unwrapped, '_rsl_rl_runner'):
        runner = env.unwrapped._rsl_rl_runner
        if hasattr(runner, 'current_learning_iteration'):
            total_iterations = runner.current_learning_iteration
        else:
            total_steps = env.common_step_counter
            steps_per_iteration = 24
            total_iterations = total_steps // steps_per_iteration
    else:
        total_steps = env.common_step_counter
        steps_per_iteration = 24
        total_iterations = total_steps // steps_per_iteration

    # Check if we should enable arm randomization
    should_enable = total_iterations >= threshold_iteration

    # Update flag if it changed
    if not hasattr(env, '_arm_randomization_enabled'):
        env._arm_randomization_enabled = False

    if should_enable and not env._arm_randomization_enabled:
        env._arm_randomization_enabled = True
        print(f"\n{'='*80}")
        print(f"[Arm Curriculum] Enabled at Iteration {total_iterations}")
        print(f"  Arm trajectory controller is now active")
        print(f"  Policy must learn to compensate for arm disturbances")
        print(f"{'='*80}\n")
    elif not should_enable and not hasattr(env, '_arm_curriculum_init'):
        env._arm_curriculum_init = True
        print(f"\n{'='*80}")
        print(f"[Arm Curriculum] Initialized")
        print(f"  Stage 1 (0-{threshold_iteration} iter): Arm stays still")
        print(f"  Stage 2 ({threshold_iteration}+ iter): Arm trajectory enabled")
        print(f"  Current iteration: {total_iterations}")
        print(f"{'='*80}\n")

    # Return current state
    return torch.tensor(float(env._arm_randomization_enabled), device=env.device)


# Legacy function names for backward compatibility (if needed)
def command_levels_height(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity_pose",
) -> torch.Tensor:
    """Legacy wrapper for height curriculum. Use command_curriculum_height_pose instead."""
    return command_curriculum_height_pose(env, env_ids, command_name)


def command_levels_orientation(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity_pose",
) -> torch.Tensor:
    """Legacy wrapper for orientation curriculum. Use command_curriculum_height_pose instead."""
    return command_curriculum_height_pose(env, env_ids, command_name)
