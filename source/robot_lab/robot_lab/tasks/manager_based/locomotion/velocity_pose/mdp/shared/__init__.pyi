"""Explicit lazy exports for migrated MDP terms."""

from .commands import (
    UniformVelocityPoseCommand,
    UniformVelocityPoseCommandCfg,
)
from .curriculums import command_curriculum_height_pose
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.events import (
    randomize_rigid_body_inertia,
    randomize_com_positions,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.utils import is_env_assigned_to_terrain
from .observations import arm_joint_pos_rel, arm_joint_vel_rel, joint_pos_rel_without_wheel, phase
from .visualizers import (
    VelocityPoseCommandVisualizer,
)
