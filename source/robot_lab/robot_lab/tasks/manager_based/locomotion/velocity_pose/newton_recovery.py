# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Row-local recovery for explicitly guarded Newton velocity-pose worlds."""

from types import SimpleNamespace

import torch


def zero_invalid_rows(value, mask):
    """Keep healthy rows bit-exact, without mutating the source tensor."""
    return torch.where(mask.reshape((-1,) + (1,) * (value.ndim - 1)), 0.0, value)


class FiniteDiagnosticData:
    """Read-only finite placeholders for already condemned diagnostic rows."""

    def __init__(self, data, mask):
        self._data, self._mask = data, mask

    def __getattr__(self, name):
        value = getattr(self._data, name)
        tensor = value.torch
        if tensor.ndim and tensor.shape[0] == len(self._mask):
            return SimpleNamespace(torch=zero_invalid_rows(tensor, self._mask))
        return value


class GuardedReward:
    """Zero each invalid world's reward before episodic sums are updated."""

    def __init__(self, function):
        self.function = function

    def __call__(self, env, **kwargs):
        value = self.function(env, **kwargs)
        if not env._guarded_any:
            return value
        value = zero_invalid_rows(value, env._guarded_invalid_worlds)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite reward outside a guarded invalid Newton world")
        return value

    def reset(self, env_ids=None):
        if hasattr(self.function, "reset"):
            return self.function.reset(env_ids=env_ids)
        return None


def install_newton_recovery(env):
    """Leave PhysX untouched; activate only with the known Newton guard."""
    from isaaclab_newton.physics import NewtonCfg

    from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terminations import invalid_physics_state

    guard = getattr(env.cfg.terminations, "invalid_physics_state", None)
    if not isinstance(env.cfg.sim.physics, NewtonCfg) or guard is None or guard.func is not invalid_physics_state:
        return
    env._guarded_invalid_worlds = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.invalid_world_resets = 0
    env._guarded_any = False
    for term in env.reward_manager._term_cfgs:
        term.func = GuardedReward(term.func)
    native_compute = env.observation_manager.compute

    def compute_guarded_observations(*args, **kwargs):
        result = native_compute(*args, **kwargs)
        if not env._guarded_any:
            return result
        for name, value in result.items():
            result[name] = zero_invalid_rows(value, env._guarded_invalid_worlds)
        return result

    env.observation_manager.compute = compute_guarded_observations
