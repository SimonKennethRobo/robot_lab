# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Moderate first-batch locomotion recipe; all schedules use PPO iterations."""

from isaaclab.utils import configclass


@configclass
class RobustnessCfg:
    observation_layout: str = "go2_x5_locomotion_v2_64"
    domain_rand: str = "sim2real"  # none / benchmark / sim2real
    steps_per_iteration: int = 24
    iteration_override: int = -1  # fixed evaluation level; -1 follows training clock
    hard_fraction: float = 0.2
    cohort_seed: int = 1234
    easy_reset_degrees: float = 10.0
    hard_reset_degrees: float = 45.0
    reset_start: int = 2000
    reset_end: int = 8000
    arm_start: int = 2000
    arm_end: int = 16000
    arm_max_velocity: float = 5.0
    arm_max_acceleration: float = 10.0
    arm_workspace_fraction: float = 1.0
    arm_accel_resample_time_s: float = 0.01
    arm_zero_accel_probability: float = 0.3
    arm_zero_velocity_probability: float = 0.005
    arm_init_joint_noise: float = 1.0
    arm_mode: int = 0  # WBC stage-1 fallback uses random acceleration; structured / hold = 1 / 2
    arm_reversal_fraction: float = 0.0  # optional fourth mode in the mixed generator
    push_start: int = 1000
    push_end: int = 8000
    push_max_xy: float = 0.6
    push_max_angular: float = 0.6
    payload_max_kg: float = 1.5
    payload_offset_m: tuple[float, float, float] = (0.10, 0.05, 0.05)
    pose_start: int = 1000
    pose_end: int = 6000
    height_range: tuple[float, float] = (0.28, 0.38)
    roll_range: tuple[float, float] = (-0.262, 0.262)
    pitch_range: tuple[float, float] = (-0.175, 0.175)
    recovery_grace_s: float = 2.0
    terminate_unrecovered: bool = True
    failure_tilt_degrees: float = 75.0
    failure_height_m: float = 0.12
    recovery_metrics: bool = False
    recovery_recipe: str = "none"  # none / control / arm / recover
    nearfall_reset_fraction: float = 0.0  # fixed subset of the hard cohort
    nearfall_reset_min_degrees: float = 30.0
    nearfall_reset_max_degrees: float = 50.0
    nearfall_reset_angular_speed: float = 1.5
    recovery_action_rate_scale: float = 1.0
    recovery_upright_weight: float = 0.0
    stance_width_target_m: float = 0.0  # 0 disables the gait-width reward
    stance_width_std_m: float = 0.10
    stance_width_reward_weight: float = 0.0
    rear_stance_width_target_m: float = 0.0  # 0 disables the rear-pair-specific reward
    rear_stance_width_std_m: float = 0.10
    rear_stance_width_reward_weight: float = 0.0
    gait_timing_variance_cost_weight: float = 0.0
    gait_swing_height_cost_weight: float = 0.0
    gait_swing_height_body_target_m: float = -0.27
    gait_joint_velocity_mirror_cost_weight: float = 0.0
    gait_contact_sync_reward_weight: float = 0.5
    metrics_interval: int = 24
