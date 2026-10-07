# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO configuration for Unitree GO2 + ARX5 (Stage 1)."""

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity_pose.config.quadruped.unitree_go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2VelocityPoseRoughPPORunnerCfg,
)


@configclass
class UnitreeGo2X5VelocityPoseRoughPPORunnerCfg(UnitreeGo2VelocityPoseRoughPPORunnerCfg):
    """PPO runner configuration for GO2+X5 Stage 1."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_x5_velocity_pose_rough"
        self.logger = "wandb"

        # 63D proprioception -> 512 -> 256 -> 128 -> 12 leg actions.
        self.actor.class_name = "MLPModel"

        # Initial budget: 8k curriculum + 12k at full intensity. Extend from
        # checkpoints only if tracking/failure metrics justify more training.
        self.max_iterations = 20000
        self.actor.distribution_cfg.std_type = "log"
        self.clip_actions = 3.0

        # Learning rate schedule
        self.actor.distribution_cfg.init_std = 1.0
        self.algorithm.learning_rate = 5e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.desired_kl = 0.01

        # PPO-specific
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4


@configclass
class UnitreeGo2X5VelocityPoseFlatPPORunnerCfg(UnitreeGo2X5VelocityPoseRoughPPORunnerCfg):
    """PPO runner configuration for flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_x5_velocity_pose_flat"

        # Flat terrain can be slightly easier
        self.algorithm.learning_rate = 5e-4


@configclass
class UnitreeGo2X5VelocityPoseMildPPORunnerCfg(UnitreeGo2X5VelocityPoseRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_x5_velocity_pose_mild"
        # Must match the Mild env action clip (see mild_env_cfg.py).
        self.clip_actions = 10.0
