# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for VelocityPose commands."""

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VelocityPoseCommandVisualizer:
    """Visualizer for VelocityPose 6D commands: [vx, vy, w_z, height, roll, pitch]"""

    def __init__(self, env: "ManagerBasedEnv", num_envs: int):
        """Initialize visualizer"""
        self.env = env
        self.num_envs = num_envs
        self.device = env.device

        self._create_markers()

        print(f"[VelocityPoseVisualizer] Initialized for {num_envs} robots")

    def _create_markers(self):
        """Create visualization markers"""
        for kind, scale, color, brightness in (
            ("target", 1.0, (1.0, 1.0, 0.0), 1.0),
            ("current", 0.8, (1.0, 0.5, 0.0), 0.6),
        ):
            origin = VisualizationMarkersCfg(
                prim_path=f"/Visuals/VelocityPoseCommand/{kind}_origin",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.025 * scale, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                    )
                },
            )
            setattr(self, f"{kind}_origin_marker", VisualizationMarkers(origin))
            for index, axis in enumerate("xyz"):
                axis_color = tuple(brightness if i == index else 0.0 for i in range(3))
                cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/VelocityPoseCommand/{kind}_{axis}_axis",
                    markers={
                        "cylinder": sim_utils.CylinderCfg(
                            radius=0.01 * scale,
                            height=0.6,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=axis_color),
                        )
                    },
                )
                setattr(self, f"{kind}_{axis}_axis_marker", VisualizationMarkers(cfg))

    def update(self, commands: torch.Tensor, robot: Articulation):
        """Update visualization - display both target and current pose"""
        commands = commands[: self.num_envs]
        base_pos_w = robot.data.root_pos_w.torch[: self.num_envs]
        base_quat_w = robot.data.root_quat_w.torch[: self.num_envs]

        # Parse target commands
        from .robustness import terrain_height

        ground, _ = terrain_height(self.env)
        target_height = commands[:, 3] + ground[: self.num_envs]
        target_roll = commands[:, 4]
        target_pitch = commands[:, 5]

        # Target origin position: CoM XY + target height Z
        target_origin_pos = base_pos_w.clone()
        target_origin_pos[:, 2] = target_height

        # CRITICAL: Correct computation of target quaternion
        # Commands define roll/pitch in Point Frame B (Yaw-Aligned Frame)
        # To visualize in world frame: target_quat_world = yaw_quat * target_quat_relative

        from isaaclab.utils.math import quat_mul, yaw_quat

        current_yaw_quat = yaw_quat(base_quat_w)

        target_quat_relative = quat_from_euler_xyz(target_roll, target_pitch, torch.zeros_like(target_roll))

        target_quat_world = quat_mul(current_yaw_quat, target_quat_relative)

        target_axes = self._compute_axis_markers(target_origin_pos, target_quat_world, axis_length=0.3)

        current_origin_pos = base_pos_w.clone()

        # CRITICAL: Use robot's actual world quaternion directly
        current_quat_world = base_quat_w.clone()

        current_axes = self._compute_axis_markers(current_origin_pos, current_quat_world, axis_length=0.3)

        identity_quat = torch.zeros((self.num_envs, 4), device=self.device)
        identity_quat[:, 3] = 1.0

        # Target pose (bright colors)
        self.target_origin_marker.visualize(target_origin_pos, identity_quat)
        self.target_x_axis_marker.visualize(target_axes["x_pos"], target_axes["x_quat"])
        self.target_y_axis_marker.visualize(target_axes["y_pos"], target_axes["y_quat"])
        self.target_z_axis_marker.visualize(target_axes["z_pos"], target_axes["z_quat"])

        # Current pose (dark colors)
        self.current_origin_marker.visualize(current_origin_pos, identity_quat)
        self.current_x_axis_marker.visualize(current_axes["x_pos"], current_axes["x_quat"])
        self.current_y_axis_marker.visualize(current_axes["y_pos"], current_axes["y_quat"])
        self.current_z_axis_marker.visualize(current_axes["z_pos"], current_axes["z_quat"])

    def _compute_axis_markers(
        self, origin_pos: torch.Tensor, orientation_quat: torch.Tensor, axis_length: float
    ) -> dict:
        """Compute position and orientation for axis markers"""
        half_length = axis_length / 2.0

        # X-axis (red)
        x_direction = quat_apply(
            orientation_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        )
        x_pos = origin_pos + x_direction * half_length

        x_quat = quat_mul(
            orientation_quat,
            quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.ones(self.num_envs, device=self.device) * 1.5708,
                torch.zeros(self.num_envs, device=self.device),
            ),
        )

        # Y-axis (green)
        y_direction = quat_apply(
            orientation_quat, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3)
        )
        y_pos = origin_pos + y_direction * half_length

        y_quat = quat_mul(
            orientation_quat,
            quat_from_euler_xyz(
                torch.ones(self.num_envs, device=self.device) * 1.5708,
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            ),
        )

        # Z-axis (blue)
        z_direction = quat_apply(
            orientation_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        )
        z_pos = origin_pos + z_direction * half_length

        z_quat = orientation_quat.clone()

        return {
            "x_pos": x_pos,
            "x_quat": x_quat,
            "y_pos": y_pos,
            "y_quat": y_quat,
            "z_pos": z_pos,
            "z_quat": z_quat,
        }


__all__ = ["VelocityPoseCommandVisualizer"]
