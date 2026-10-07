# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityPoseCommand(velocity_mdp.UniformThresholdVelocityCommand):
    """Command generator that extends velocity commands with height and full pose (roll, pitch, yaw) control."""

    cfg: UniformVelocityPoseCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        # Height command: (num_envs, 1) - target height for root link
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        # GO2-X5 uses [roll, pitch]; the third channel is legacy compatibility.
        pose_dim = 3 if cfg.include_pose_yaw else 2
        self.pose_command = torch.zeros(self.num_envs, pose_dim, device=self.device)

        self.default_height = self.cfg.default_height

    @property
    def command(self) -> torch.Tensor:
        """Desired [vx, vy, wz, height, roll, pitch], plus yaw only for legacy layouts."""
        return torch.cat([self.vel_command_b, self.height_command, self.pose_command], dim=1)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        """Sample absolute height and pose ranges, including nonzero constants."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        super()._resample_command(env_ids)
        height_range = self.cfg.ranges.height
        # Preserve the legacy (0, 0) sentinel for plain GO2 configurations.
        if height_range == (0.0, 0.0):
            height_range = (self.default_height, self.default_height)
        self.height_command[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(*height_range)
        pose_names = ("roll", "pitch", "yaw") if self.cfg.include_pose_yaw else ("roll", "pitch")
        for index, name in enumerate(pose_names):
            bounds = getattr(self.cfg.ranges, name)
            self.pose_command[env_ids, index] = torch.empty(len(env_ids), device=self.device).uniform_(*bounds)

    def _update_command(self):
        super()._update_command()
        standing = getattr(self._env, "_arm_extension_standing", None)
        if standing is not None:
            self.vel_command_b[standing] = 0.0
            self.height_command[standing] = self.default_height
            self.pose_command[standing] = 0.0

    def _update_metrics(self):
        """Use the same yaw-aligned XY and world-Z frames as tracking rewards."""
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        velocity = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w.torch), self.robot.data.root_lin_vel_w.torch)
        self.metrics["error_vel_xy"] += (self.vel_command_b[:, :2] - velocity[:, :2]).norm(dim=-1) / steps
        self.metrics["error_vel_yaw"] += (
            self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_w.torch[:, 2]
        ).abs() / steps


@configclass
class UniformVelocityPoseCommandCfg(velocity_mdp.UniformThresholdVelocityCommandCfg):
    """Configuration for the uniform velocity and pose command generator."""

    class_type: type = UniformVelocityPoseCommand
    # New GO2-X5 layouts disable this; older tasks and explicit legacy layouts keep it.
    include_pose_yaw: bool = True

    # Default height for the robot base (will be overridden per robot)
    default_height: float = 0.35
    """Default height for robot root link in meters."""

    @configclass
    class Ranges(velocity_mdp.UniformThresholdVelocityCommandCfg.Ranges):
        """Ranges for the velocity and pose commands."""

        # Height command range (relative to default height)
        height: tuple[float, float] = (0.0, 0.0)
        """Range for height command in meters. Set to (0.0, 0.0) for curriculum learning."""

        # Roll command range (Base Frame C relative to Point Frame B)
        roll: tuple[float, float] = (0.0, 0.0)
        """Range for roll command in radians. Set to (0.0, 0.0) for curriculum learning."""

        # Pitch command range (Base Frame C relative to Point Frame B)
        pitch: tuple[float, float] = (0.0, 0.0)
        """Range for pitch command in radians. Set to (0.0, 0.0) for curriculum learning."""

        # Yaw command range (Base Frame C relative to Point Frame B)
        yaw: tuple[float, float] = (0.0, 0.0)
        """Range for yaw command in radians. Set to (0.0, 0.0) for curriculum learning."""

    ranges: Ranges = Ranges()
