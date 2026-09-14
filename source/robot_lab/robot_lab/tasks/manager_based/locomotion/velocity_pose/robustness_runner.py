# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL checkpoint contract and exact policy-step curriculum resumption."""

import torch

from rsl_rl.runners import OnPolicyRunner


class LocomotionOnPolicyRunner(OnPolicyRunner):
    def _contract(self):
        env = self.env.unwrapped
        return {
            "layout": env.cfg.robustness.observation_layout,
            "policy_terms": env.observation_manager.active_terms["policy"],
            "policy_dim": list(env.observation_manager.group_obs_dim["policy"]),
            "critic_dim": list(env.observation_manager.group_obs_dim["critic"]),
            "leg_joint_order": list(env.cfg.dog_joint_names),
            "arm_joint_order": list(env.cfg.arm_joint_names),
            "steps_per_iteration": env.cfg.robustness.steps_per_iteration,
        }

    def save(self, path, infos=None):
        env = self.env.unwrapped
        metadata = dict(infos or {})
        metadata["locomotion"] = {
            "contract": self._contract(),
            "policy_steps": env.common_step_counter + env._locomotion.step_offset,
            "recipe": env.cfg.robustness.to_dict(),
        }
        super().save(path, infos=metadata)

    def load(self, path, load_optimizer=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = (checkpoint.get("infos") or {}).get("locomotion")
        if metadata is None or metadata.get("contract") != self._contract():
            raise ValueError(
                "Incompatible locomotion checkpoint: expected go2_x5_locomotion_v2_64 "
                "with matching observation/joint order and rollout length. Legacy 84D "
                "weights require retraining or an explicit migration; they cannot be resumed silently."
            )
        result = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        env = self.env.unwrapped
        steps = metadata["policy_steps"]
        env._locomotion.step_offset = steps - env.common_step_counter
        saved_override = metadata["recipe"].get("iteration_override", -1)
        if getattr(env.cfg, "inference_mode", False) and env.cfg.robustness.iteration_override < 0:
            env.cfg.robustness.iteration_override = saved_override
        # RSL-RL stores the last zero-based update index. Our next update and
        # curriculum use the completed policy-step count, avoiding resume drift.
        self.current_learning_iteration = steps // env.cfg.robustness.steps_per_iteration
        env._locomotion.update_command_ranges()
        with torch.inference_mode():
            env.reset()
        return result
