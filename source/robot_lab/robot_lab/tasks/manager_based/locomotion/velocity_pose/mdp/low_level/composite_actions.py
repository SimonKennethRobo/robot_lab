# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Twelve policy leg targets plus a bounded, exogenous six-joint arm target."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from ..shared.robustness import runtime_for
from ..shared.robustness_math import BoundedArmMotion


class DogArmCompositeAction(JointPositionAction):
    def __init__(self, cfg, env):
        # Native processing handles scale, clipping, offset and joint order.
        super().__init__(cfg, env)
        self._runtime = runtime_for(env)
        self._arm_joint_ids = self._runtime.arm_ids
        self._gripper_ids = self._asset.find_joints("gripper_joint_.*")[0]
        settings = env.cfg.robustness
        limits = self._asset.data.soft_joint_pos_limits[:, self._arm_joint_ids]
        default = self._asset.data.default_joint_pos[:, self._arm_joint_ids]
        radius = (limits[..., 1] - limits[..., 0]) * 0.5 * settings.arm_workspace_fraction
        lower = torch.maximum(limits[..., 0], default - radius)
        upper = torch.minimum(limits[..., 1], default + radius)
        self._arm_controller = BoundedArmMotion(
            lower,
            upper,
            env.step_dt,
            settings.arm_max_velocity,
            settings.arm_max_acceleration,
            fixed_mode=settings.arm_mode,
        )
        self._arm_targets = default.clone()
        self._arm_controller.reset(torch.arange(env.num_envs, device=env.device), default)

    def process_actions(self, actions):
        self._runtime.check_finite(action=actions)
        self._runtime.update_command_ranges()
        super().process_actions(actions)
        self._arm_targets = self._arm_controller.step(self._runtime.progress("arm"))

    def apply_actions(self):
        super().apply_actions()
        self._asset.set_joint_position_target(self._arm_targets, joint_ids=self._arm_joint_ids)
        if self._gripper_ids:
            self._asset.set_joint_position_target(
                self._asset.data.default_joint_pos[:, self._gripper_ids], joint_ids=self._gripper_ids
            )
        self._runtime.apply_payload()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        self._arm_controller.reset(env_ids, self._asset.data.joint_pos[env_ids][:, self._arm_joint_ids])
        self._arm_targets[env_ids] = self._arm_controller.q[env_ids]


@configclass
class DogArmActionCfg(JointPositionActionCfg):
    class_type: type = DogArmCompositeAction
