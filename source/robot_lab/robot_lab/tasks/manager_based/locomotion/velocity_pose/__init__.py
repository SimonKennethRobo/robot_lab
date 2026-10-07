# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environments with velocity and pose tracking commands."""

# Register Gym environments for all robots

from .config.quadruped.unitree_go2 import *  # noqa: F401, F403
from .config.quadruped.unitree_go2_x5 import *  # noqa: F401, F403

# Loading the runtime environment before launch_simulation imports USD before
# Kit selects its USD build. Keep Gym's environment entry point lazy.
from isaaclab.utils.module import lazy_export

lazy_export()
