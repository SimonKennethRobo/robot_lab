# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Run Isaac Lab RSL-RL with VelocityPose task hooks."""

import sys

import warp as wp

wp.config.enable_backward = False

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity_pose.rsl_rl import run

if __name__ == "__main__":
    run(sys.argv[1:], play=False)
