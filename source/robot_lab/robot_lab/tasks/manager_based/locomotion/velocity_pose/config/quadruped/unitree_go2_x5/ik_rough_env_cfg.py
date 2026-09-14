# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reserved task ID for the unfinished experimental IK controller."""

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2X5VelocityPoseRoughEnvCfg


@configclass
class UnitreeGo2X5VelocityPoseIKRoughEnvCfg(UnitreeGo2X5VelocityPoseRoughEnvCfg):
    def __post_init__(self):
        raise NotImplementedError(
            "The experimental IK task uses a placeholder Jacobian and is not a supported training recipe. "
            "Use the VelocityPose Flat, Mild or Rough Go2-X5 task with bounded arm disturbances."
        )
