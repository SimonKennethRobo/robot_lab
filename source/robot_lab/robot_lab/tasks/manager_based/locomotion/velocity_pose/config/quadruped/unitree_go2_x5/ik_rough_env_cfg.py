# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Rough task variant tracking reachable exogenous arm positions via native IK."""

from dataclasses import fields

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.high_level.composite_actions_ik import DogArmIKActionCfg

from .rough_env_cfg import UnitreeGo2X5VelocityPoseRoughEnvCfg


@configclass
class UnitreeGo2X5VelocityPoseIKRoughEnvCfg(UnitreeGo2X5VelocityPoseRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        parent_action = self.actions.joint_pos
        self.actions.joint_pos = DogArmIKActionCfg(
            **{
                field.name: getattr(parent_action, field.name)
                for field in fields(parent_action)
                if field.name != "class_type"
            }
        )

        if self.__class__.__name__ == "UnitreeGo2X5VelocityPoseIKRoughEnvCfg":
            from robot_lab.tasks.manager_based.locomotion.velocity_pose.newton_arm import configure_newton_x5

            explicit = self.scene.robot.actuators["arm"].copy()
            configure_newton_x5(self)
            from isaaclab_tasks.utils import preset

            from robot_lab.assets.delayed_implicit import implicit_with_same_limits
            from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.high_level.arm_gravity import (
                GravityLimitedImplicitActuatorCfg,
            )

            implicit = implicit_with_same_limits(explicit)
            compensated = GravityLimitedImplicitActuatorCfg(
                **{
                    field.name: getattr(implicit, field.name)
                    for field in fields(implicit)
                    if field.name != "class_type"
                }
            )
            self.scene.robot.actuators["arm"] = preset(default=explicit, newton_mjwarp=compensated)
