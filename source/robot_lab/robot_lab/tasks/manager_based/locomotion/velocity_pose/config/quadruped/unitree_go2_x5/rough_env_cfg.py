# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""GO2 + X5 locomotion: stable rewards and continuous robustness curricula."""

import math

import isaaclab.envs.mdp as lab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robot_lab.assets.go2_x5 import GO2_X5_CFG
from robot_lab.tasks.manager_based.locomotion.velocity_pose import mdp
from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.low_level.composite_actions import DogArmActionCfg
from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.shared import robustness
from robot_lab.tasks.manager_based.locomotion.velocity_pose.robustness_cfg import RobustnessCfg
from robot_lab.tasks.manager_based.locomotion.velocity_pose.velocity_pose_env_cfg import (
    LocomotionVelocityPoseRoughEnvCfg,
)

DOG_JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
ARM_BODY_NAMES = [f"link{i}" for i in range(1, 9)]
# The Lab 2.3 X5 contact reporter exposes only these 19 Go2 bodies. Restrict the
# sensor itself so every contact reward/diagnostic uses that same physical scope.
LEGACY_CONTACT_BODY_NAMES = [
    "base",
    *[f"{leg}_{part}" for leg in ("FL", "FR") for part in ("hip", "thigh", "calf", "foot")],
    "Head_upper",
    "Head_lower",
    *[f"{leg}_{part}" for leg in ("RL", "RR") for part in ("hip", "thigh", "calf", "foot")],
]


def leg_entity():
    return SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES, preserve_order=True)


@configclass
class GO2X5ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """63D proprioception; leg observations/actions use FR, FL, RR, RL order."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Enabled for explicit height-error contracts in prepare_robustness_cfg.
        # Keeping the field here fixes its concatenation position at index 9.
        height_error = None
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity_pose"})
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": leg_entity()}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": leg_entity()}, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        arm_joint_pos = ObsTerm(
            func=mdp.arm_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        arm_joint_vel = ObsTerm(
            func=mdp.arm_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True)},
            scale=0.1,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class UnitreeGo2X5VelocityPoseRoughEnvCfg(LocomotionVelocityPoseRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    dog_joint_names = DOG_JOINT_NAMES
    arm_joint_names = ARM_JOINT_NAMES
    all_joint_names = DOG_JOINT_NAMES + ARM_JOINT_NAMES
    robustness: RobustnessCfg = RobustnessCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = GO2X5ObservationsCfg.PolicyCfg()
        self.observations.critic = GO2X5ObservationsCfg.PolicyCfg()
        self.observations.critic.enable_corruption = False
        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/(" + "|".join(LEGACY_CONTACT_BODY_NAMES) + ")"
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.36)
        for name in ("legs", "arm"):
            self.scene.robot.actuators[name].max_delay = 2  # physics ticks: 0--10 ms
        self.scene.height_scanner = None
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/base"
        command = self.commands.base_velocity_pose
        command.include_pose_yaw = False
        command.debug_vis = False
        command.default_height = 0.33
        command.ranges.height = (0.33, 0.33)
        command.ranges.roll = command.ranges.pitch = (0.0, 0.0)
        command.ranges.lin_vel_x = (-1.0, 1.0)
        command.ranges.lin_vel_y = (-0.6, 0.6)
        command.ranges.ang_vel_z = (-1.0, 1.0)
        command.heading_command = False
        command.ranges.heading = None
        command.rel_standing_envs = 0.15
        self.actions.joint_pos = DogArmActionCfg(
            asset_name="robot",
            joint_names=DOG_JOINT_NAMES,
            preserve_order=True,
            scale={".*_hip_joint": 0.125, ".*_(thigh|calf)_joint": 0.25},
            clip={".*": (-3.0, 3.0)},
            use_default_offset=True,
        )

        events = self.events
        events.randomize_apply_external_force_torque = None
        events.randomize_rigid_body_material.params.update(
            static_friction_range=(0.5, 1.25),
            dynamic_friction_range=(0.4, 1.0),
            restitution_range=(0.0, 0.1),
            make_consistent=True,
        )
        events.randomize_rigid_body_mass_base.params.update(
            asset_cfg=SceneEntityCfg("robot", body_names="base"), mass_distribution_params=(-1.0, 1.0)
        )
        events.randomize_rigid_body_mass_others.params.update(
            asset_cfg=SceneEntityCfg("robot", body_names="^(?!base$|link[1-8]$).*"),
            mass_distribution_params=(0.85, 1.15),
        )
        events.randomize_arm_link_mass = EventTerm(
            func=lab_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ARM_BODY_NAMES),
                "mass_distribution_params": (0.1, 2.0),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        events.randomize_com_positions.params.update(
            asset_cfg=SceneEntityCfg("robot", body_names="base"),
            com_range={axis: (-0.02, 0.02) for axis in ("x", "y", "z")},
        )
        events.randomize_arm_com_positions = EventTerm(
            func=lab_mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ARM_BODY_NAMES),
                "com_range": {axis: (-0.1, 0.1) for axis in ("x", "y", "z")},
            },
        )
        events.randomize_actuator_gains.params.update(
            asset_cfg=leg_entity(), stiffness_distribution_params=(0.8, 1.2), damping_distribution_params=(0.8, 1.2)
        )
        events.randomize_arm_gains = EventTerm(
            func=lab_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
                "stiffness_distribution_params": (0.5, 1.5),
                "damping_distribution_params": (0.2, 2.0),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        events.randomize_reset_base = EventTerm(func=robustness.reset_locomotion_root, mode="reset")
        events.randomize_reset_joints = EventTerm(func=robustness.reset_locomotion_joints, mode="reset")
        events.randomize_push_robot = EventTerm(
            func=robustness.push_locomotion, mode="interval", interval_range_s=(4.0, 8.0)
        )

        rewards = self.rewards
        for name in (
            "is_terminated",
            "flat_orientation_l2",
            "base_height_l2",
            "body_lin_acc_l2",
            "joint_vel_l2",
            "joint_vel_limits",
            "stand_still",
            "joint_mirror",
            "feet_air_time_variance",
            "feet_contact",
            "feet_stumble",
            "feet_height",
            "feet_height_body",
            "upward",
        ):
            setattr(rewards, name, None)
        rewards.lin_vel_z_l2.weight = -0.5
        rewards.ang_vel_xy_l2.weight = -0.05
        for name, weight in (
            ("joint_torques_l2", -2.5e-5),
            ("joint_acc_l2", -2.5e-7),
            ("joint_pos_limits", -5.0),
            ("joint_power", -2e-5),
        ):
            term = getattr(rewards, name)
            term.weight = weight
            term.params["asset_cfg"] = leg_entity()
        rewards.joint_pos_penalty = RewTerm(
            func=mdp.joint_pos_penalty_full_cmd,
            weight=-0.25,
            params={
                "command_name": "base_velocity_pose",
                "asset_cfg": leg_entity(),
                "stand_still_scale": 1.0,
                "velocity_threshold": 0.5,
                "velocity_cmd_threshold": 0.1,
                "height_threshold": 0.02,
                "angle_threshold": 0.05,
            },
        )
        rewards.action_rate_l2.weight = -0.1
        rewards.undesired_contacts.weight = -1.0
        rewards.undesired_contacts.params["sensor_cfg"].body_names = ["^(?!.*_foot$).*"]
        rewards.contact_forces.weight = -1.5e-4
        rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        rewards.track_lin_vel_xy_exp.weight = rewards.track_ang_vel_z_exp.weight = 6.0
        rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)
        rewards.track_height_exp = RewTerm(
            func=mdp.track_height_exp,
            weight=2.0,
            params={
                "command_name": "base_velocity_pose",
                "std": 0.06,
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            },
        )
        rewards.track_orientation_exp = RewTerm(
            func=robustness.track_commanded_tilt,
            weight=2.0,
            params={"command_name": "base_velocity_pose", "std": 0.25},
        )
        rewards.accumulated_ang_vel_standing = RewTerm(
            func=robustness.bounded_standing_yaw_penalty,
            weight=-0.5,
            params={
                "command_name": "base_velocity_pose",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
            },
        )
        rewards.feet_air_time.weight = 0.1
        rewards.feet_air_time.params["threshold"] = 0.3
        rewards.feet_contact_without_cmd.weight = 0.1
        rewards.feet_slide.weight = -0.1
        for name in ("feet_air_time", "feet_contact_without_cmd", "feet_slide"):
            getattr(rewards, name).params["sensor_cfg"].body_names = [self.foot_link_name]
        rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        for key in ("sensor_cfg", "asset_cfg"):
            rewards.feet_slide.params[key].body_names = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
            rewards.feet_slide.params[key].preserve_order = True
        rewards.feet_gait.weight = 0.5
        rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.disable_zero_weight_rewards()

        self.terminations.illegal_contact = None
        self.terminations.locomotion_failure = DoneTerm(func=robustness.locomotion_diagnostics)
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_curriculum_height_pose = None

        if self.__class__.__name__ == "UnitreeGo2X5VelocityPoseRoughEnvCfg":
            from robot_lab.tasks.manager_based.locomotion.velocity_pose.newton_arm import configure_newton_x5

            configure_newton_x5(self)
