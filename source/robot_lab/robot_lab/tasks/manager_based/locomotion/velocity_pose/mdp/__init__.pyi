"""Local terms take precedence over stock velocity fallbacks."""

from .low_level.composite_actions import (
    DogArmCompositeAction,
)
from .low_level.rewards import (
    track_height_exp,
    track_orientation_exp_without_yaw,
    lin_vel_z_penalty_conditional,
    ang_vel_xy_penalty_conditional,
    stand_still_full_cmd,
    joint_pos_penalty_full_cmd,
    accumulated_ang_vel_penalty_when_standing,
)
from .shared.commands import (
    UniformVelocityPoseCommand,
    UniformVelocityPoseCommandCfg,
)
from .shared.curriculums import command_curriculum_height_pose
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.events import (
    randomize_rigid_body_inertia,
    randomize_com_positions,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.utils import is_env_assigned_to_terrain
from .shared.observations import arm_joint_pos_rel, arm_joint_vel_rel, joint_pos_rel_without_wheel, phase
from .shared.visualizers import (
    VelocityPoseCommandVisualizer,
)
from .high_level import DogArmIKCompositeAction, extract_yaw_quat

from robot_lab.tasks.manager_based.locomotion.velocity.mdp import *
