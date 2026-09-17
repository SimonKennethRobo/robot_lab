# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL checkpoint contract, warm starts, and curriculum resumption."""

import hashlib
import torch
import torch.nn.functional as F

from rsl_rl.runners import OnPolicyRunner


class HeightInsertedLinear(torch.nn.Linear):
    """A 65D linear layer that preserves the original 64D GEMM at zero height error."""

    def forward(self, input):
        legacy_input = torch.cat((input[..., :9], input[..., 10:]), dim=-1)
        legacy_weight = torch.cat((self.weight[:, :9], self.weight[:, 10:]), dim=-1)
        legacy = F.linear(legacy_input, legacy_weight, self.bias)
        return legacy + input[..., 9:10] * self.weight[:, 9]


class LocomotionOnPolicyRunner(OnPolicyRunner):
    _warm_start = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        policy = self.alg.policy
        if self.env.unwrapped.cfg.robustness.observation_layout == "go2_x5_locomotion_v3_65":
            # Preserve parameter objects already registered in PPO's optimizer.
            policy.actor[0].__class__ = HeightInsertedLinear
            policy.critic[0].__class__ = HeightInsertedLinear
        original_update_distribution = policy.update_distribution

        def checked_update_distribution(obs):
            if not torch.isfinite(obs).all():
                raise FloatingPointError("Non-finite policy observation before action distribution")
            original_update_distribution(obs)
            if not torch.isfinite(policy.action_mean).all():
                raise FloatingPointError("Non-finite actor mean")
            if not torch.isfinite(policy.action_std).all() or not (policy.action_std > 0).all():
                raise FloatingPointError("Action std must be finite and strictly positive")

        policy.update_distribution = checked_update_distribution

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

    def log(self, locs, width=80, pad=35):
        super().log(locs, width=width, pad=pad)
        runtime = self.env.unwrapped._locomotion
        if runtime.cfg.recovery_metrics:
            # Write exact interval counters, bypassing episode-info averaging.
            for key, value in runtime.previous_metrics.items():
                if key.startswith(("Recovery/", "Numerics/")):
                    self.writer.add_scalar(key, value.item(), locs["it"])
            std = self.alg.policy.action_std
            self.writer.add_scalar("Policy/std_min", std.min().item(), locs["it"])
            self.writer.add_scalar("Policy/std_max", std.max().item(), locs["it"])
            if self._warm_start:
                self.writer.add_scalar(
                    "Training/finetune_updates_completed",
                    locs["it"] - self._warm_start["source_iteration"] + 1,
                    locs["it"],
                )

    def save(self, path, infos=None):
        env = self.env.unwrapped
        metadata = dict(infos or {})
        metadata["locomotion"] = {
            "contract": self._contract(),
            "policy_steps": env.common_step_counter + env._locomotion.step_offset,
            "recipe": env.cfg.robustness.to_dict(),
        }
        if self._warm_start is not None:
            metadata["locomotion"]["warm_start"] = dict(self._warm_start)
        super().save(path, infos=metadata)

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_warm_start(self, path, map_location=None):
        """Load finite weights, migrate scalar std, and discard optimizer state."""
        checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        metadata = (checkpoint.get("infos") or {}).get("locomotion")
        target_contract = self._contract()
        source_contract = None if metadata is None else metadata.get("contract")
        migrate_height_error = (
            source_contract is not None
            and source_contract.get("layout") == "go2_x5_locomotion_v2_64"
            and target_contract.get("layout") == "go2_x5_locomotion_v3_65"
            and source_contract.get("policy_dim") == [64]
            and source_contract.get("critic_dim") == [64]
            and target_contract.get("policy_dim") == [65]
            and target_contract.get("critic_dim") == [65]
            and source_contract.get("leg_joint_order") == target_contract.get("leg_joint_order")
            and source_contract.get("arm_joint_order") == target_contract.get("arm_joint_order")
            and source_contract.get("steps_per_iteration") == target_contract.get("steps_per_iteration")
        )
        if metadata is None or (source_contract != target_contract and not migrate_height_error):
            raise ValueError("Warm-start checkpoint has an incompatible locomotion contract")
        state = dict(checkpoint.get("model_state_dict") or {})
        if not state:
            raise ValueError("Warm-start checkpoint has no model_state_dict")
        for name, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite warm-start tensor: {name}")
        target = self.alg.policy.state_dict()
        migrated = False
        floor_applied = False
        if "log_std" in target and "std" in state:
            std = state.pop("std")
            if not (std > 0).all():
                raise FloatingPointError("Warm-start scalar std must be strictly positive")
            floored = std.clamp_min(1.0e-6)
            floor_applied = not torch.equal(std, floored)
            state["log_std"] = floored.log()
            migrated = True
        if migrate_height_error:
            for name in ("actor.0.weight", "critic.0.weight"):
                old = state[name]
                new = target[name].new_zeros(target[name].shape)
                new[:, :9] = old[:, :9]
                new[:, 10:] = old[:, 9:]
                state[name] = new
        self.alg.policy.load_state_dict(state, strict=True)
        env = self.env.unwrapped
        steps = int(metadata["policy_steps"])
        env._locomotion.step_offset = steps - env.common_step_counter
        self.current_learning_iteration = int(checkpoint.get("iter", steps // env.cfg.robustness.steps_per_iteration))
        env._locomotion.update_command_ranges()
        self._warm_start = {
            "source_path": str(path),
            "source_sha256": self._sha256(path),
            "source_iteration": int(checkpoint.get("iter", -1)),
            "source_policy_steps": steps,
            "scalar_std_to_log_std": migrated,
            "std_floor_applied": floor_applied,
            "height_error_zero_column_migration": migrate_height_error,
            "source_observation_layout": source_contract["layout"],
            "target_observation_layout": target_contract["layout"],
            "optimizer_restored": False,
        }
        with torch.inference_mode():
            env.reset()
        return metadata

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
