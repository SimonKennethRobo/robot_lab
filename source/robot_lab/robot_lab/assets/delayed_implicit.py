# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Solver-integrated PD with the same command delay as Lab DelayedPD."""

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass


class DelayedImplicitActuator(ImplicitActuator):
    """Delay setpoints, then let the backend integrate the PD joint drive."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._delay_buffers = (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer)

    def reset(self, env_ids):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        count = len(range(self._num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        lags = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (count,), dtype=torch.int, device=self._device)
        for buffer in self._delay_buffers:
            buffer.set_time_lag(lags, env_ids)
            buffer.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = DelayedImplicitActuator
    min_delay: int = 0
    max_delay: int = 0


def implicit_with_same_limits(explicit_cfg):
    """Move the existing model torque clamp into the implicit solver drive."""
    from dataclasses import fields

    values = {
        field.name: getattr(explicit_cfg, field.name)
        for field in fields(DelayedImplicitActuatorCfg)
        if field.name != "class_type"
    }
    # Implicit drives bypass explicit effort clipping. The physical 20 Nm
    # ceiling must therefore be enforced at the drive, not left at 1e9 Nm.
    values["joint_effort_limit"] = explicit_cfg.actuator_effort_limit
    return DelayedImplicitActuatorCfg(**values)
