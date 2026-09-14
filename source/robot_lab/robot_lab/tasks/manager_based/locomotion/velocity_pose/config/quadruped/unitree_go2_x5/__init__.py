# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0",
    entry_point="robot_lab.tasks.manager_based.locomotion.velocity_pose:VelocityPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mild_env_cfg:UnitreeGo2X5VelocityPoseMildEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2X5VelocityPoseMildPPORunnerCfg",
    },
)

##
# Register Gym environments for Unitree GO2 + ARX5
##

gym.register(
    id="RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0",
    entry_point="robot_lab.tasks.manager_based.locomotion.velocity_pose:VelocityPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2X5VelocityPoseFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2X5VelocityPoseFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-VelocityPose-Rough-Unitree-Go2-X5-v0",
    entry_point="robot_lab.tasks.manager_based.locomotion.velocity_pose:VelocityPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2X5VelocityPoseRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2X5VelocityPoseRoughPPORunnerCfg",
    },
)

##
# Register IK-based environment (Stage 2 Training)
##

gym.register(
    id="RobotLab-Isaac-VelocityPose-IK-Unitree-Go2-X5-v0",
    entry_point="robot_lab.tasks.manager_based.locomotion.velocity_pose:VelocityPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rough_env_cfg:UnitreeGo2X5VelocityPoseIKRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2X5VelocityPoseRoughPPORunnerCfg",
    },
)
