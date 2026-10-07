# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observation functions for VelocityPose task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations import (
    joint_pos_rel_without_wheel,
    phase,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ARX5 Arm Observations (Stage 1)


def arm_joint_pos_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
) -> torch.Tensor:
    """Arm joint positions relative to default position."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    joint_pos_default = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]

    pos_rel = joint_pos - joint_pos_default

    # NUMERICAL STABILITY FIX: Clip extreme positions to prevent observation explosion
    # ARX5 joints have typical range ±π, extreme cases might reach ±2π
    # Clip at ±3π for safety without affecting normal motion
    pos_rel = torch.clamp(pos_rel, -3.0 * 3.14159, 3.0 * 3.14159)

    return pos_rel


def arm_joint_vel_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
) -> torch.Tensor:
    """Arm joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]

    # NUMERICAL STABILITY FIX: Clip extreme velocities to prevent observation explosion
    # Without this, new arm motion modes (fishing/grasping/swinging/probing) can generate
    # extreme velocities that cause critic network to output huge values, leading to
    # value function loss explosion (e.g., 9.23×10²⁸ at iteration 26430)
    # Typical arm velocities: ±5 rad/s, extreme: ±20 rad/s, so clip at ±50 rad/s for safety
    joint_vel = torch.clamp(joint_vel, -50.0, 50.0)

    return joint_vel


__all__ = ["arm_joint_pos_rel", "arm_joint_vel_rel", "joint_pos_rel_without_wheel", "phase"]
