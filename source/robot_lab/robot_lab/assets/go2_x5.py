# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

GO2_X5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2_x5_description/usd/go2_x5/go2_x5.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper_joint_left": 0.0,
            "gripper_joint_right": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[r"(FL|FR|RL|RR)_(hip|thigh|calf)_joint"],
            joint_effort_limit=1.0e9,  # Preserve the 2.3 explicit-actuator solver clamp.
            actuator_effort_limit=33.5,
            actuator_velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            min_delay=0,
            max_delay=5,
        ),
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            joint_effort_limit=1.0e9,  # Preserve the 2.3 explicit-actuator solver clamp.
            actuator_effort_limit=20.0,
            actuator_velocity_limit=10.0,
            stiffness={
                "joint1": 50.0,
                "joint2": 50.0,
                "joint3": 80.0,
                "joint4": 30.0,
                "joint5": 20.0,
                "joint6": 20.0,
            },
            damping={
                "joint1": 5.0,
                "joint2": 10.0,
                "joint3": 10.0,
                "joint4": 2.5,
                "joint5": 2.0,
                "joint6": 1.0,
            },
            friction=0.01,
            min_delay=0,
            max_delay=5,
        ),
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=["gripper_joint_(left|right)"],
            joint_effort_limit=1.0e9,  # Preserve the 2.3 explicit-actuator solver clamp.
            actuator_effort_limit=12.0,
            actuator_velocity_limit=5.0,
            stiffness=25,
            damping=0.5,
        ),
    },
)
