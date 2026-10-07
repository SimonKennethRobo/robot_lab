# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Twelve policy leg actions with native IK tracking exogenous arm targets."""

import torch

from isaaclab.utils import configclass

from ..low_level.composite_actions import DogArmActionCfg, DogArmCompositeAction
from .arm_gravity import arm_gravity_effort
from .arm_ik_controller import ArmIKController


class DogArmIKCompositeAction(DogArmCompositeAction):
    """Convert the parent's reachable arm trajectory to link6 position targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ik_controller = ArmIKController(self._asset, env.num_envs, env.device, self._runtime.arm_ids)
        self._ee_target_override = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def ik_controller(self) -> ArmIKController:
        """Expose the native-IK adapter for smoke validation."""
        return self._ik_controller

    def set_ee_target(self, position_b: torch.Tensor) -> None:
        """Hold a copied, finite full-batch root-frame target until reset."""
        self._ik_controller.set_target(position_b)
        self._ee_target_override[:] = True

    def process_actions(self, actions):
        super().process_actions(actions)
        trajectory_target = self._runtime.arm_kinematics.position(self._arm_targets)
        self._ik_controller.set_target(
            torch.where(self._ee_target_override[:, None], self._ik_controller.target_position_b, trajectory_target)
        )

    def apply_actions(self):
        self._arm_targets = self._ik_controller.compute()
        super().apply_actions()
        if self.cfg.gravity_compensation:
            # Physics-frequency update, before the actuator's existing delay
            # buffers and PD+feedforward clipping. Legs receive no feedforward.
            self._asset.actuators.target_command.set_effort_index(
                value=arm_gravity_effort(self._asset, self._arm_joint_ids), joint_ids=self._arm_joint_ids
            )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self._env.num_envs, device=self.device)[env_ids]
        super().reset(env_ids)
        self._ee_target_override[env_ids] = False
        actual_target = self._runtime.arm_kinematics.position(self._asset.data.joint_pos.torch[:, self._arm_joint_ids])
        target = self._ik_controller.target_position_b.clone()
        target[env_ids] = actual_target[env_ids]
        self._ik_controller.set_target(target)
        self._ik_controller.reset(env_ids)


@configclass
class DogArmIKActionCfg(DogArmActionCfg):
    """Preserve the leg-action contract while selecting native arm IK."""

    class_type: type = DogArmIKCompositeAction
    gravity_compensation: bool = True
