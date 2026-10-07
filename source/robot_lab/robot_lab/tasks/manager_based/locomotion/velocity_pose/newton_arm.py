# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Newton X5 arm integration, preserving the authored PhysX configuration."""


def configure_newton_x5(cfg):
    from isaaclab_tasks.utils import preset

    from robot_lab.assets.delayed_implicit import implicit_with_same_limits
    from robot_lab.tasks.manager_based.locomotion.velocity.newton_support import configure_newton_from_profile

    configure_newton_from_profile(cfg, "unitree_go2_x5", njmax=512, nconmax=256)
    original = cfg.scene.robot.actuators["arm"]
    cfg.scene.robot.actuators["arm"] = preset(default=original, newton_mjwarp=implicit_with_same_limits(original))
