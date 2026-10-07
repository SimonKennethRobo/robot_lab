# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Velocity and Pose Tracking Environment for Quadruped Locomotion"""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from .mdp.shared.robustness import prepare_robustness_cfg, runtime_for
from .mdp.shared.visualizers import VelocityPoseCommandVisualizer


class VelocityPoseEnv(ManagerBasedRLEnv):
    """Environment for quadruped locomotion with velocity and pose tracking."""

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the velocity pose environment."""
        if getattr(cfg, "robustness", None) is not None:
            prepare_robustness_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        from .newton_recovery import install_newton_recovery

        install_newton_recovery(self)
        self._pose_visualizer = None
        self._visualizer_initialized = False

    def set_locomotion_iteration(self, iteration: int):
        """Restore the policy-step curriculum clock after checkpoint loading."""
        if getattr(self.cfg, "robustness", None) is not None:
            runtime = runtime_for(self)
            runtime.step_offset = iteration * runtime.cfg.steps_per_iteration - self.common_step_counter
            runtime.update_command_ranges()

    def _reset_idx(self, env_ids):
        invalid = getattr(self, "_guarded_invalid_worlds", None)
        if invalid is not None and invalid[env_ids].any():
            healthy_ids = env_ids[~invalid[env_ids]]
            invalid_ids = env_ids[invalid[env_ids]]
            if healthy_ids.numel():
                self._reset_locomotion_idx(healthy_ids)
            # An extra invalid-world reset must not consume the healthy worlds'
            # RNG stream (including later observation noise and command samples).
            with torch.random.fork_rng(devices=[torch.device(self.device)]):
                self._reset_locomotion_idx(invalid_ids)
            self.invalid_world_resets += invalid_ids.numel()
            return
        self._reset_locomotion_idx(env_ids)

    def _reset_locomotion_idx(self, env_ids):
        if hasattr(self, "_locomotion") and getattr(self, "_collecting_transition", False):
            self._locomotion.record_completed(env_ids)
        super()._reset_idx(env_ids)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment and initialize visualizer on first reset."""
        obs, info = super().reset(seed=seed, options=options)

        if not self._visualizer_initialized:
            try:
                if hasattr(self.command_manager, "_terms"):
                    for term_name, term in self.command_manager._terms.items():
                        if "velocity_pose" in term_name.lower():
                            if hasattr(term, "cfg") and hasattr(term.cfg, "debug_vis"):
                                if term.cfg.debug_vis:
                                    self._pose_visualizer = VelocityPoseCommandVisualizer(self, min(self.num_envs, 8))
                                break
            except Exception as e:
                print(f"[WARNING] Failed to initialize VelocityPose visualizer: {e}")
                import traceback

                traceback.print_exc()

            self._visualizer_initialized = True

        return obs, info

    def step(self, action: torch.Tensor) -> tuple[Any, Any, Any, Any, Any]:
        """Execute environment step with visualization update."""
        # Execute normal step
        invalid = getattr(self, "_guarded_invalid_worlds", None)
        if invalid is not None:
            invalid.zero_()
            self._guarded_any = False
        self._collecting_transition = True
        try:
            result = super().step(action)
        finally:
            self._collecting_transition = False
        if hasattr(self, "_locomotion"):
            self._locomotion.check_finite(reward=result[1], **result[0])
            if self.common_step_counter % self.cfg.robustness.metrics_interval == 0:
                metrics = self._locomotion.flush_metrics()
                # The full cohort/age breakdown lives in locomotion_metrics.jsonl.
                # Keep the per-iteration PPO console and logger compact.
                self.extras.setdefault("log", {}).update(
                    {
                        key: value
                        for key, value in metrics.items()
                        if "/all/all/" in key or key.startswith("Curriculum/") or key.endswith("failure_fraction")
                    }
                )

        if self._pose_visualizer is not None:
            command = self.command_manager.get_command("base_velocity_pose")
            robot = self.scene["robot"]
            self._pose_visualizer.update(command, robot)

        return result
