# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# fmt: off
"""Curriculum learning functions for VelocityPose task with stage-based progression."""
# fmt: on

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Preserve v2 curriculum branches during API migration
def _update_reward_parameters(env: ManagerBasedRLEnv, stage: int):  # noqa: C901
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
            if hasattr(env.reward_manager, "_term_weights"):
                env.reward_manager._term_weights["track_height_exp"] = 0.0

        if orient_reward_cfg:
            orient_reward_cfg.weight = 0.0
            if hasattr(env.reward_manager, "_term_weights"):
                env.reward_manager._term_weights["track_orientation_exp"] = 0.0

        # Yaw control: relaxed tolerance (allow some drift during basic locomotion learning)
        if ang_vel_z_tracking_cfg:
            ang_vel_z_tracking_cfg.params["std"] = math.sqrt(0.25)

        if lin_vel_z_l2_cfg:
            lin_vel_z_l2_cfg.weight = -2.0
        if ang_vel_xy_l2_cfg:
            ang_vel_xy_l2_cfg.weight = 0.0

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
            if hasattr(env.reward_manager, "_term_weights"):
                env.reward_manager._term_weights["track_height_exp"] = 4.0

        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.50)
            orient_reward_cfg.weight = 4.0
            if hasattr(env.reward_manager, "_term_weights"):
                env.reward_manager._term_weights["track_orientation_exp"] = 4.0

        if ang_vel_z_tracking_cfg:
            ang_vel_z_tracking_cfg.params["std"] = math.sqrt(0.10)

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
            if hasattr(env.reward_manager, "_term_weights"):
                env.reward_manager._term_weights["track_height_exp"] = 12.0

        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.10)
            orient_reward_cfg.weight = 12.0
            if hasattr(env.reward_manager, "_term_weights"):
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
    elif stage == 4 or stage == 5:
        if height_reward_cfg:
            height_reward_cfg.params["std"] = math.sqrt(0.05)
            height_reward_cfg.weight = 16.0
            if hasattr(env.reward_manager, "_term_weights"):
                env.reward_manager._term_weights["track_height_exp"] = 16.0

        # Orientation tracking: very strict tolerance, very high weight
        if orient_reward_cfg:
            orient_reward_cfg.params["std"] = math.sqrt(0.10)
            orient_reward_cfg.weight = 16.0
            if hasattr(env.reward_manager, "_term_weights"):
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


def command_curriculum_height_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity_pose",
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    default_height = command_term.default_height

    # Try to get iteration from various sources (in priority order):
    if hasattr(env, "_curriculum_manual_iteration"):
        # Manually injected by a wrapper (most reliable)
        total_iterations = env._curriculum_manual_iteration
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_rsl_rl_runner"):
        runner = env.unwrapped._rsl_rl_runner
        if hasattr(runner, "current_learning_iteration"):
            total_iterations = runner.current_learning_iteration
        else:
            # Fallback to calculation
            total_steps = env.common_step_counter
            steps_per_iteration = 24
            total_iterations = total_steps // steps_per_iteration
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_current_iteration"):
        total_iterations = env.unwrapped._current_iteration  # type: ignore
    elif hasattr(env.unwrapped, "episode_length_buf"):
        total_steps = env.common_step_counter
        steps_per_iteration = 24  # RSL-RL default: 24 steps per env per iteration
        total_iterations = total_steps // steps_per_iteration
    else:
        # Last resort fallback
        total_steps = env.common_step_counter
        steps_per_iteration = 24  # Assume RSL-RL default
        total_iterations = total_steps // steps_per_iteration

        if hasattr(env, "_curriculum_stage"):
            expected_stage_for_iter = (
                1
                if total_iterations < 20000
                else 2
                if total_iterations < 25000
                else 3
                if total_iterations < 30000
                else 4
                if total_iterations < 35000
                else 5
            )
            if env._curriculum_stage != expected_stage_for_iter:
                if not hasattr(env, "_curriculum_resume_warning_shown"):
                    env._curriculum_resume_warning_shown = True  # type: ignore

    # This check must happen BEFORE stage calculation to ensure it persists across resets
    # Only consider it inference mode if EXPLICITLY marked or if runner is None (not just missing)
    is_inference_mode = (
        hasattr(env.unwrapped, "_is_inference_mode")  # Explicitly marked by play.py
        or (hasattr(env.unwrapped, "_rsl_rl_runner") and env.unwrapped._rsl_rl_runner is None)  # type: ignore
    )

    inference_stage = getattr(env.unwrapped, "_inference_curriculum_stage", 4)  # Default to Stage 4 if not set

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
            print(f"\n{'=' * 80}")
            print("[Curriculum] INFERENCE MODE DETECTED")
            print(f"  Using Stage {target_stage} as requested")
            print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
            print(
                f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = "
                f"[{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°"
            )
            print(
                f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = "
                f"[{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°"
            )
            print(
                f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = "
                f"[{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°"
            )
            print(f"  Arm Motion Scale: {arm_motion_scale:.2f}")
            print(f"  Arm Frequency Range: [{arm_frequency_range[0]:.1f}, {arm_frequency_range[1]:.1f}] Hz")
            print(f"  Arm Amplitude Range: [{arm_amplitude_range[0]:.1f}, {arm_amplitude_range[1]:.1f}] rad")
            print(f"{'=' * 80}\n")
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

    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = target_stage
        env._curriculum_last_update = 0

        # IMPORTANT: Set initial command ranges based on starting stage
        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        ranges.yaw = yaw_range

        env._arm_motion_scale = arm_motion_scale
        env._arm_frequency_range = arm_frequency_range
        env._arm_amplitude_range = arm_amplitude_range

        _update_reward_parameters(env, target_stage)

        print(f"\n{'=' * 80}")
        print(f"[Curriculum] Initialized at iteration {total_iterations}")
        print(f"  Starting Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(
            f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = "
            f"[{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°"
        )
        print(
            f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = "
            f"[{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°"
        )
        print(
            f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = "
            f"[{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°"
        )
        print(f"  Arm Motion Scale: {arm_motion_scale:.2f}")
        print(f"  Arm Frequency Range: [{arm_frequency_range[0]:.1f}, {arm_frequency_range[1]:.1f}] Hz")
        print(f"  Arm Amplitude Range: [{arm_amplitude_range[0]:.1f}, {arm_amplitude_range[1]:.1f}] rad")
        print(f"{'=' * 80}\n")

    if target_stage != env._curriculum_stage:
        env._curriculum_stage = target_stage
        env._curriculum_last_update = total_iterations

        ranges.height = height_range
        ranges.roll = roll_range
        ranges.pitch = pitch_range
        ranges.yaw = yaw_range

        env._arm_motion_scale = arm_motion_scale
        env._arm_frequency_range = arm_frequency_range
        env._arm_amplitude_range = arm_amplitude_range

        _update_reward_parameters(env, target_stage)

        # Print stage transition message
        print(f"\n{'=' * 80}")
        print(f"[Curriculum] Stage Transition at Iteration {total_iterations}")
        print(f"{'=' * 80}")
        print(f"  New Stage: {target_stage}")
        print(f"  Height Range: [{height_range[0]:.3f}, {height_range[1]:.3f}] m")
        print(
            f"  Roll Range:   [{roll_range[0]:.3f}, {roll_range[1]:.3f}] rad = "
            f"[{math.degrees(roll_range[0]):.1f}, {math.degrees(roll_range[1]):.1f}]°"
        )
        print(
            f"  Pitch Range:  [{pitch_range[0]:.3f}, {pitch_range[1]:.3f}] rad = "
            f"[{math.degrees(pitch_range[0]):.1f}, {math.degrees(pitch_range[1]):.1f}]°"
        )
        print(
            f"  Yaw Range:    [{yaw_range[0]:.3f}, {yaw_range[1]:.3f}] rad = "
            f"[{math.degrees(yaw_range[0]):.1f}, {math.degrees(yaw_range[1]):.1f}]°"
        )
        print(f"  Arm Motion Scale: {arm_motion_scale:.2f}")
        print(f"  Arm Frequency Range: [{arm_frequency_range[0]:.1f}, {arm_frequency_range[1]:.1f}] Hz")
        print(f"  Arm Amplitude Range: [{arm_amplitude_range[0]:.1f}, {arm_amplitude_range[1]:.1f}] rad")
        print(f"{'=' * 80}\n")

    return torch.tensor(float(target_stage), device=env.device)


# Legacy function names for backward compatibility (if needed)
