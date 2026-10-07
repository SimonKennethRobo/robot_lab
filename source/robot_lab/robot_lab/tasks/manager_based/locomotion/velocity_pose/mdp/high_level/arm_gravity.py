# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""IK-only arm feedforward, retaining gravity and its load on the floating base."""

import warp as wp

from isaaclab.utils import configclass

from robot_lab.assets.delayed_implicit import DelayedImplicitActuator, DelayedImplicitActuatorCfg


def arm_gravity_effort(robot, joint_ids):
    """Native g(q), in public joint order, with the floating-base prefix removed."""
    values = robot.data.gravity_compensation_forces.torch
    return values[:, [i + robot.num_base_dofs for i in joint_ids]].clone()


class GravityLimitedImplicitActuator(DelayedImplicitActuator):
    """Delayed implicit drive plus effort feedforward, with a TOTAL 20 Nm bound."""

    def compute(self, control_action, joint_pos, joint_vel):
        result = super().compute(control_action, joint_pos, joint_vel)
        if not hasattr(self, "_range_view"):
            import mujoco
            from isaaclab_newton.physics import NewtonManager

            solver = NewtonManager._solver
            assert hasattr(solver, "mjw_model"), "Total feedforward clamp requires Newton MJWarp"
            model = solver.mj_model
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
            ids = []
            for name in self.joint_names:
                matches = [
                    i
                    for i, value in enumerate(names)
                    if value and (value.rsplit("/", 1)[-1] == name or value.endswith("_" + name))
                ]
                assert len(matches) == 1, (name, names)
                ids.append(matches[0])
            self._range_view = wp.to_torch(solver.mjw_model.jnt_actfrcrange)
            assert self._range_view.shape[0] == self._num_envs, self._range_view.shape
            self._range_ids = ids
            self._solver = solver
            self._dof_ids = model.jnt_dofadr[ids].tolist()
            assert all(model.jnt_actfrclimited[ids])
        ff = result.joint_efforts
        limit = self.actuator_effort_limit
        self._range_view[:, self._range_ids, 0] = -limit - ff
        self._range_view[:, self._range_ids, 1] = limit - ff
        self.last_feedforward = ff.clone()
        return result

    def actual_total_effort(self):
        """Previous solver-step PD plus the applied external feedforward."""
        data = self._solver.mjw_data
        return (wp.to_torch(data.qfrc_actuator) + wp.to_torch(data.qfrc_applied))[:, self._dof_ids]


@configclass
class GravityLimitedImplicitActuatorCfg(DelayedImplicitActuatorCfg):
    class_type: type = GravityLimitedImplicitActuator
