# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for VelocityPose commands."""

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VelocityPoseCommandVisualizer:
    """Visualizer for VelocityPose 6D commands: [vx, vy, w_z, height, roll, pitch]
    
    Displays target and current pose axes above robot CoM with RGB colors.
    """
    
    def __init__(self, env: "ManagerBasedEnv", num_envs: int):
        """Initialize visualizer
        
        Args:
            env: Manager-based RL environment
            num_envs: Number of environments
        """
        self.env = env
        self.num_envs = num_envs
        self.device = env.device
        
        # Create visualization markers
        self._create_markers()
        
        print(f"[VelocityPoseVisualizer] Initialized for {num_envs} robots")
    
    def _create_markers(self):
        """Create visualization markers
        
        Two sets of coordinate axes:
        1. Target pose (bright colors) - from command
        2. Current pose (dark colors) - from gravity projection
        """
        # Axis dimensions
        axis_length = 0.6   
        axis_radius = 0.01  # 1cm radius
        sphere_radius = 0.025  # 2.5cm sphere
        
        # ============================================================
        # Target pose visualization (bright, from command)
        # ============================================================
        
        # Target origin - yellow
        target_origin_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_origin",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=sphere_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            },
        )
        self.target_origin_marker = VisualizationMarkers(target_origin_cfg)
        
        # Target X-axis - bright red
        target_x_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_x_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.target_x_axis_marker = VisualizationMarkers(target_x_axis_cfg)
        
        # Target Y-axis - bright green
        target_y_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_y_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        self.target_y_axis_marker = VisualizationMarkers(target_y_axis_cfg)
        
        # Target Z-axis - bright blue
        target_z_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_z_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        )
        self.target_z_axis_marker = VisualizationMarkers(target_z_axis_cfg)
        
        # ============================================================
        # Current pose visualization (dark, from gravity)
        # ============================================================
        
        # Current origin - orange
        current_origin_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_origin",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=sphere_radius * 0.8,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                ),
            },
        )
        self.current_origin_marker = VisualizationMarkers(current_origin_cfg)
        
        # Current X-axis - dark red
        current_x_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_x_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius * 0.8,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.0, 0.0)),
                ),
            },
        )
        self.current_x_axis_marker = VisualizationMarkers(current_x_axis_cfg)
        
        # Current Y-axis - dark green
        current_y_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_y_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius * 0.8,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.6, 0.0)),
                ),
            },
        )
        self.current_y_axis_marker = VisualizationMarkers(current_y_axis_cfg)
        
        # Current Z-axis - dark blue
        current_z_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_z_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius * 0.8,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.6)),
                ),
            },
        )
        self.current_z_axis_marker = VisualizationMarkers(current_z_axis_cfg)
    
    def update(self, commands: torch.Tensor, robot: Articulation):
        """Update visualization - display both target and current pose
        
        Args:
            commands: (N, 6) [vx, vy, w_z, height, roll, pitch]
            robot: Robot articulation
        """
        # Get current base position and orientation
        commands = commands[:self.num_envs]
        base_pos_w = robot.data.root_pos_w[:self.num_envs]
        base_quat_w = robot.data.root_quat_w[:self.num_envs]
        
        # Parse target commands
        from .robustness import terrain_height
        ground, _ = terrain_height(self.env)
        target_height = commands[:, 3] + ground[:self.num_envs]
        target_roll = commands[:, 4]
        target_pitch = commands[:, 5]
        
        # Compute current orientation from gravity projection
        current_roll, current_pitch = self._compute_orientation_from_gravity(robot)
        
        # Compute target pose visualization (in world frame)
        # Target origin position: CoM XY + target height Z
        target_origin_pos = base_pos_w.clone()
        target_origin_pos[:, 2] = target_height
        
        # CRITICAL: Correct computation of target quaternion
        # Commands define roll/pitch in Point Frame B (Yaw-Aligned Frame)
        # To visualize in world frame: target_quat_world = yaw_quat * target_quat_relative
        
        from isaaclab.utils.math import yaw_quat, quat_mul, quat_conjugate
        
        # Get robot's yaw quaternion (Point Frame B orientation in world)
        current_yaw_quat = yaw_quat(base_quat_w)
        
        # Compute relative pose from command (Base Frame C relative to Point Frame B)
        target_quat_relative = quat_from_euler_xyz(target_roll, target_pitch, torch.zeros_like(target_roll))
        
        # Combine to get target pose in world frame
        target_quat_world = quat_mul(current_yaw_quat, target_quat_relative)
        
        # Compute target coordinate axes (cylinder with one end at origin)
        target_axes = self._compute_axis_markers(target_origin_pos, target_quat_world, axis_length=0.3)
        
        # Compute current pose visualization (in world frame)
        current_origin_pos = base_pos_w.clone()
        
        # CRITICAL: Use robot's actual world quaternion directly
        current_quat_world = base_quat_w.clone()
        
        # Compute current coordinate axes (cylinder with one end at origin)
        current_axes = self._compute_axis_markers(current_origin_pos, current_quat_world, axis_length=0.3)
        
        # Update all markers
        identity_quat = torch.zeros((self.num_envs, 4), device=self.device)
        identity_quat[:, 0] = 1.0
        
        # Target pose (bright colors)
        self.target_origin_marker.visualize(target_origin_pos, identity_quat)
        self.target_x_axis_marker.visualize(target_axes['x_pos'], target_axes['x_quat'])
        self.target_y_axis_marker.visualize(target_axes['y_pos'], target_axes['y_quat'])
        self.target_z_axis_marker.visualize(target_axes['z_pos'], target_axes['z_quat'])
        
        # Current pose (dark colors)
        self.current_origin_marker.visualize(current_origin_pos, identity_quat)
        self.current_x_axis_marker.visualize(current_axes['x_pos'], current_axes['x_quat'])
        self.current_y_axis_marker.visualize(current_axes['y_pos'], current_axes['y_quat'])
        self.current_z_axis_marker.visualize(current_axes['z_pos'], current_axes['z_quat'])
    
    def _compute_orientation_from_gravity(self, robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute current base roll and pitch from gravity projection
        
        Principle:
        - Gravity in base frame = R^T * g_world
        - projected_gravity is gravity representation in base frame
        
        Recover orientation from projected_gravity:
        - roll = atan2(gy, gz)  # Left-right tilt
        - pitch = atan2(-gx, sqrt(gy^2 + gz^2))  # Forward-backward tilt
        
        Args:
            robot: Robot articulation
            
        Returns:
            roll: (N,) Current roll angle (radians)
            pitch: (N,) Current pitch angle (radians)
        """
        projected_gravity = robot.data.projected_gravity_b[:self.num_envs]
        
        gx = projected_gravity[:, 0]
        gy = projected_gravity[:, 1]
        gz = projected_gravity[:, 2]
        
        # Compute roll and pitch from gravity projection
        roll = torch.atan2(-gy, -gz)
        pitch = torch.atan2(gx, torch.sqrt(gy**2 + gz**2))
        
        return roll, pitch
    
    def _compute_axis_markers(self, origin_pos: torch.Tensor, orientation_quat: torch.Tensor, 
                             axis_length: float) -> dict:
        """Compute position and orientation for axis markers
        
        Cylinders have one end at origin, extending outward.
        Cylinder center is at: origin + axis_direction * (length/2)
        
        Args:
            origin_pos: (N, 3) Origin position
            orientation_quat: (N, 4) Orientation quaternion
            axis_length: Cylinder length
            
        Returns:
            dict with keys: x_pos, x_quat, y_pos, y_quat, z_pos, z_quat
        """
        half_length = axis_length / 2.0
        
        # X-axis (red)
        x_direction = self._rotate_vector(
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
            orientation_quat
        )
        x_pos = origin_pos + x_direction * half_length
        
        x_quat = self._quat_multiply(
            orientation_quat,
            quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.ones(self.num_envs, device=self.device) * 1.5708,
                torch.zeros(self.num_envs, device=self.device)
            )
        )
        
        # Y-axis (green)
        y_direction = self._rotate_vector(
            torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3),
            orientation_quat
        )
        y_pos = origin_pos + y_direction * half_length
        
        y_quat = self._quat_multiply(
            orientation_quat,
            quat_from_euler_xyz(
                torch.ones(self.num_envs, device=self.device) * 1.5708,
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )
        )
        
        # Z-axis (blue)
        z_direction = self._rotate_vector(
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3),
            orientation_quat
        )
        z_pos = origin_pos + z_direction * half_length
        
        z_quat = orientation_quat.clone()
        
        return {
            'x_pos': x_pos,
            'x_quat': x_quat,
            'y_pos': y_pos,
            'y_quat': y_quat,
            'z_pos': z_pos,
            'z_quat': z_quat,
        }
    
    def _rotate_vector(self, vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Rotate vector by quaternion using Isaac Lab built-in function
        
        Args:
            vec: (N, 3) Vector
            quat: (N, 4) Quaternion [w, x, y, z]
            
        Returns:
            rotated_vec: (N, 3) Rotated vector
        """
        return quat_apply(quat, vec)
    
    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication q1 * q2 using Isaac Lab built-in function
        
        Args:
            q1, q2: (N, 4) [w, x, y, z]
            
        Returns:
            result: (N, 4) q1 * q2
        """
        return quat_mul(q1, q2)
    
    def reset(self, env_ids: torch.Tensor):
        """Reset visualization for specified environments
        
        Markers handle reset automatically, no additional action needed.
        """
        pass


__all__ = ["VelocityPoseCommandVisualizer"]
