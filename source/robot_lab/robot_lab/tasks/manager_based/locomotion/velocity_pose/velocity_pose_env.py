# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Velocity and Pose Tracking Environment for Quadruped Locomotion

This environment extends velocity tracking with height and orientation control.
For GO2+ARX5, it integrates the arm trajectory controller to handle composite actions.
"""

from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from .mdp.shared.robustness import prepare_robustness_cfg, runtime_for
from .mdp.shared.visualizers import VelocityPoseCommandVisualizer


class VelocityPoseEnv(ManagerBasedRLEnv):
    """Environment for quadruped locomotion with velocity and pose tracking.

    For robots with manipulators (e.g., GO2+ARX5), the composite action space 
    is handled by DogArmCompositeAction in the ActionManager:
    - Policy outputs actions for dog joints only (12D for GO2)
    - Arm trajectory controller generates arm actions (6D for ARX5)
    - DogArmCompositeAction combines them and applies to robot (18D total)
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the velocity pose environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for the environment.
            **kwargs: Additional arguments.
        """
        if getattr(cfg, "robustness", None) is not None:
            prepare_robustness_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize pose visualizer (will be set up after first reset)
        self._pose_visualizer = None
        self._visualizer_initialized = False

    def set_locomotion_iteration(self, iteration: int):
        """Restore the policy-step curriculum clock after checkpoint loading."""
        if getattr(self.cfg, "robustness", None) is not None:
            runtime = runtime_for(self)
            runtime.step_offset = iteration * runtime.cfg.steps_per_iteration - self.common_step_counter
            runtime.update_command_ranges()

    def _reset_idx(self, env_ids):
        if hasattr(self, "_locomotion") and getattr(self, "_collecting_transition", False):
            self._locomotion.record_completed(env_ids)
        super()._reset_idx(env_ids)
        
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment and initialize visualizer on first reset.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Observations and info dict
        """
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialize visualizer on first reset (when sim is ready)
        if not self._visualizer_initialized:
            try:
                # Check if debug_vis is enabled in velocity_pose command
                if hasattr(self.command_manager, '_terms'):
                    for term_name, term in self.command_manager._terms.items():
                        if 'velocity_pose' in term_name.lower():
                            if hasattr(term, 'cfg') and hasattr(term.cfg, 'debug_vis'):
                                if term.cfg.debug_vis:
                                    self._pose_visualizer = VelocityPoseCommandVisualizer(
                                        self, min(self.num_envs, 8)
                                    )
                                    print("[VelocityPoseEnv] Pose visualizer initialized successfully")
                                else:
                                    print("[VelocityPoseEnv] debug_vis=False, visualizer disabled")
                                break
            except Exception as e:
                print(f"[WARNING] Failed to initialize VelocityPose visualizer: {e}")
                import traceback
                traceback.print_exc()
            
            self._visualizer_initialized = True
        
        return obs, info
        
    def step(self, action: torch.Tensor) -> tuple[Any, Any, Any, Any, Any]:
        """Execute environment step with visualization update.
        
        Args:
            action: Actions from policy
            
        Returns:
            Tuple of (obs, rew, terminated, truncated, info)
        """
        # Execute normal step
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
                self.extras.setdefault("log", {}).update({
                    key: value for key, value in metrics.items()
                    if "/all/all/" in key or key.startswith("Curriculum/") or key.endswith("failure_fraction")
                })
        
        # Update visualization if enabled
        if self._pose_visualizer is not None:
            command = self.command_manager.get_command("base_velocity_pose")
            robot = self.scene["robot"]
            self._pose_visualizer.update(command, robot)
        
        return result
