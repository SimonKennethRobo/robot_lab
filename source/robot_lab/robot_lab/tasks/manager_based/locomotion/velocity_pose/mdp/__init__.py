# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Locomotion MDP terms; experimental IK modules load only when requested."""

from robot_lab.tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .low_level import *  # noqa: F401, F403
from .shared import *  # noqa: F401, F403


def __getattr__(name):
    if name in {"DogArmIKCompositeAction", "ARX5IKController", "create_ik_arm_controller", "extract_yaw_quat"}:
        from . import high_level

        return getattr(high_level, name)
    raise AttributeError(name)
