# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Thin adapter for Isaac Lab position IK at the native link6 body origin."""

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms


class ArmIKController:
    """Track root-frame position targets with native articulation Jacobians."""

    def __init__(self, robot: Articulation, num_envs: int, device: str, arm_joint_ids: Sequence[int]):
        self._robot = robot
        self._num_envs = num_envs
        self._device = device
        self._arm_joint_ids = list(arm_joint_ids)
        expected_names = [f"joint{i}" for i in range(1, 7)]
        if len(self._arm_joint_ids) != 6 or [robot.joint_names[i] for i in self._arm_joint_ids] != expected_names:
            raise ValueError(f"Expected six ordered arm joints {expected_names}, got {self._arm_joint_ids}")
        body_ids, body_names = robot.find_bodies("link6")
        if len(body_ids) != 1:
            raise ValueError(f"Expected exactly one link6 body, got {body_names}")
        self._body_id = body_ids[0]
        self._jacobian_body_id = self._body_id - 1 if robot.is_fixed_base else self._body_id
        self._jacobian_joint_ids = [index + robot.num_base_dofs for index in self._arm_joint_ids]
        self._controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="position", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
            ),
            num_envs=num_envs,
            device=device,
        )
        self.target_position_b = self.current_position_b().clone()
        self._ready = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def native_controller(self) -> DifferentialIKController:
        """Isaac Lab controller exposed for validation harnesses."""
        return self._controller

    def _current_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._robot.data
        return subtract_frame_transforms(
            data.root_pos_w.torch,
            data.root_quat_w.torch,
            data.body_pos_w.torch[:, self._body_id],
            data.body_quat_w.torch[:, self._body_id],
        )

    def current_position_b(self) -> torch.Tensor:
        """Return the measured link6 origin position relative to the root."""
        return self._current_pose_b()[0]

    def jacobian_b(self) -> torch.Tensor:
        """Return an owned native link-origin Jacobian rotated into root axes."""
        jacobian = self._robot.data.body_link_jacobian_w.torch[:, self._jacobian_body_id][
            :, :, self._jacobian_joint_ids
        ].clone()
        rotation = matrix_from_quat(quat_inv(self._robot.data.root_quat_w.torch))
        jacobian[:, :3] = torch.bmm(rotation, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(rotation, jacobian[:, 3:])
        return jacobian

    def set_target(self, position_b: torch.Tensor) -> None:
        """Copy a finite, full-batch absolute root-frame position target."""
        if position_b.shape != (self._num_envs, 3) or not torch.isfinite(position_b).all():
            raise ValueError(f"Expected finite position targets with shape ({self._num_envs}, 3)")
        self.target_position_b = position_b.to(device=self._device).clone()

    def compute(self) -> torch.Tensor:
        """Compute native IK targets, bounded by per-step delta and soft limits."""
        position_b, quaternion_b = self._current_pose_b()
        joint_pos = self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
        self._controller.set_command(self.target_position_b, ee_quat=quaternion_b)
        targets = self._controller.compute(position_b, quaternion_b, self.jacobian_b(), joint_pos)
        targets = joint_pos + (targets - joint_pos).clamp(-0.2, 0.2)
        limits = self._robot.data.soft_joint_pos_limits.torch[:, self._arm_joint_ids]
        targets = targets.clamp(limits[..., 0], limits[..., 1])
        # Follow the native IK example: hold on the first physics tick after a
        targets = torch.where(self._ready[:, None], targets, joint_pos)
        self._ready[:] = True
        return targets

    def reset(self, env_ids) -> None:
        """Reset native controller state and hold the next physics tick."""
        self._controller.reset(env_ids)
        self._ready[env_ids] = False
