# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat terrain configuration for Unitree GO2 + ARX5."""

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2X5VelocityPoseRoughEnvCfg


@configclass
class UnitreeGo2X5VelocityPoseFlatEnvCfg(UnitreeGo2X5VelocityPoseRoughEnvCfg):
    """Configuration for flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        # Change terrain to flat plane (critical for flat terrain!)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Flat terrain - no height scanner needed
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        if hasattr(self.rewards, "track_height_exp") and self.rewards.track_height_exp is not None:
            self.rewards.track_height_exp.params["sensor_cfg"] = None
            print(f"[Config] track_height_exp weight: {self.rewards.track_height_exp.weight}")

        if hasattr(self.rewards, "track_orientation_exp") and self.rewards.track_orientation_exp is not None:
            print(f"[Config] track_orientation_exp weight: {self.rewards.track_orientation_exp.weight}")

        if hasattr(self.rewards, "base_height_l2") and self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["sensor_cfg"] = None
            self.rewards.base_height_l2.params["asset_cfg"].body_names = self.base_link_name
            self.rewards.base_height_l2.weight = 0.0

        if hasattr(self, "curriculum") and hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "UnitreeGo2X5VelocityPoseFlatEnvCfg":
            from robot_lab.tasks.manager_based.locomotion.velocity_pose.newton_arm import configure_newton_x5

            configure_newton_x5(self)
