# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

# Import base velocity environment configuration
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ActionsCfg,
    ObservationsCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)

import robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp as mdp


@configclass
class VelocityPoseCommandsCfg:
    """Command specifications for the MDP with velocity and pose control."""

    base_velocity_pose = mdp.UniformVelocityPoseCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        default_height=0.35,  
        ranges=mdp.UniformVelocityPoseCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
            
            height=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0), 
        ),
    )


@configclass
class LocomotionVelocityPoseRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the locomotion velocity and pose tracking environment.
    
    This environment extends velocity tracking with height and orientation control.
    For curriculum learning, height, roll, and pitch commands start at default values.
    """

    # Override commands with velocity_pose commands
    commands: VelocityPoseCommandsCfg = VelocityPoseCommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # Update observations to include the new command dimensions
        if self.observations.policy.velocity_commands is not None:
            self.observations.policy.velocity_commands.params["command_name"] = "base_velocity_pose"
        if self.observations.critic.velocity_commands is not None:
            self.observations.critic.velocity_commands.params["command_name"] = "base_velocity_pose"
        
        if hasattr(self.rewards, "track_lin_vel_xy_exp") and self.rewards.track_lin_vel_xy_exp is not None:
            self.rewards.track_lin_vel_xy_exp.params["command_name"] = "base_velocity_pose"
            from robot_lab.tasks.manager_based.locomotion.velocity.mdp import rewards as velocity_rewards
            self.rewards.track_lin_vel_xy_exp.func = velocity_rewards.track_lin_vel_xy_yaw_frame_exp
        if hasattr(self.rewards, "track_ang_vel_z_exp") and self.rewards.track_ang_vel_z_exp is not None:
            self.rewards.track_ang_vel_z_exp.params["command_name"] = "base_velocity_pose"
            from robot_lab.tasks.manager_based.locomotion.velocity.mdp import rewards as velocity_rewards
            self.rewards.track_ang_vel_z_exp.func = velocity_rewards.track_ang_vel_z_world_exp
        
        command_aware_rewards = [
            "stand_still",
            "joint_pos_penalty",
            "wheel_vel_penalty",
            "feet_air_time",
            "feet_gait",
            "feet_contact",
            "feet_contact_without_cmd",
            "feet_height",
            "feet_height_body",
        ]
        
        for reward_name in command_aware_rewards:
            if hasattr(self.rewards, reward_name):
                reward_term = getattr(self.rewards, reward_name)
                if reward_term is not None:
                    reward_term.params["command_name"] = "base_velocity_pose"
