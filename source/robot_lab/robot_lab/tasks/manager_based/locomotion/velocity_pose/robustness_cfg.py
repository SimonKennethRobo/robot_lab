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
    arm_start: int = 1000
    arm_end: int = 8000
    arm_max_velocity: float = 2.5
    arm_max_acceleration: float = 8.0
    arm_workspace_fraction: float = 0.7
    arm_mode: int = -1  # random / structured / hold = 0 / 1 / 2
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
    metrics_interval: int = 24
