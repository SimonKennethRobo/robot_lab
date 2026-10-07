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
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._arm_joint_ids]
        default = self._asset.data.default_joint_pos.torch[:, self._arm_joint_ids]
        if settings.arm_full_extension_fraction > 0:
            # Expand toward both soft limits; asymmetric joints must reach their
            # forward pose even when the default is near their lower limit.
            center = default.clamp(limits[..., 0], limits[..., 1])
            lower = center + settings.arm_workspace_fraction * (limits[..., 0] - center)
            upper = center + settings.arm_workspace_fraction * (limits[..., 1] - center)
        else:
            # Preserve the original random-arm workspace for explicit legacy runs.
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
            reversal_fraction=settings.arm_reversal_fraction,
            accel_resample_time_s=settings.arm_accel_resample_time_s,
            zero_accel_probability=settings.arm_zero_accel_probability,
            zero_velocity_probability=settings.arm_zero_velocity_probability,
            full_extension_fraction=settings.arm_full_extension_fraction,
            full_extension_joint_pos=settings.arm_full_extension_joint_pos,
            full_extension_hold_s=settings.arm_full_extension_hold_s,
            full_extension_max_velocity=settings.arm_full_extension_max_velocity,
            full_extension_max_acceleration=settings.arm_full_extension_max_acceleration,
        )
        env._arm_extension_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._arm_targets = default.clone()
        self._arm_controller.reset(torch.arange(env.num_envs, device=env.device), default)

    def process_actions(self, actions):
        self._runtime.check_finite(action=actions)
        self._runtime.update_command_ranges()
        super().process_actions(actions)
        self._arm_targets = self._arm_controller.step(
            self._runtime.progress("arm"), self._asset.data.joint_pos.torch[:, self._arm_joint_ids]
        )

    def apply_actions(self):
        super().apply_actions()
        self._asset.actuators.target_command.set_position_index(value=self._arm_targets, joint_ids=self._arm_joint_ids)
        if self._gripper_ids:
            self._asset.actuators.target_command.set_position_index(
                value=self._asset.data.default_joint_pos.torch[:, self._gripper_ids], joint_ids=self._gripper_ids
            )
        self._runtime.apply_payload()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self._env.num_envs, device=self.device)[env_ids or slice(None)]
        self._arm_controller.reset(env_ids, self._asset.data.joint_pos.torch[env_ids][:, self._arm_joint_ids])
        self._arm_targets[env_ids] = self._arm_controller.q[env_ids]
        if self._runtime.cfg.arm_full_extension_fraction > 0:
            self._env._arm_extension_standing[env_ids] = (self._arm_controller.mode[env_ids] == 4) & (
                torch.rand(len(env_ids), device=self.device) < self._runtime.cfg.arm_full_extension_standing_fraction
            )


@configclass
class DogArmActionCfg(JointPositionActionCfg):
    class_type: type = DogArmCompositeAction
