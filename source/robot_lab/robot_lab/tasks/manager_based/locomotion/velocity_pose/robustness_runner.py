# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL checkpoint contract, warm starts, and curriculum resumption."""

import os

import torch
from rsl_rl.runners import OnPolicyRunner


class LocomotionOnPolicyRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alg.actor.register_forward_pre_hook(self._check_actor_observations)
        self.alg.actor.register_forward_hook(self._check_actor_output)
        # Observe sampled rollout actions at the wrapper boundary, before its
        # +/-3 clipping. PPO minibatch forwards are excluded from this metric.
        self._raw_action_exceedances = torch.zeros((), device=self.device, dtype=torch.int64)
        self._raw_action_count = 0
        self._invalid_resets_logged = 0
        self._invalid_window_start = 0
        native_step = self.env.step

        def step_with_action_audit(actions):
            self._raw_action_exceedances.add_((actions.abs() > 3.0).sum())
            self._raw_action_count += actions.numel()
            return native_step(actions)

        self.env.step = step_with_action_audit
        native_log = self.logger.log

        def log_with_locomotion(**values):
            native_log(**values)
            self._log_locomotion(values["it"])

        self.logger.log = log_with_locomotion

    @staticmethod
    def _check_actor_observations(module, inputs):
        for group in module.obs_groups:
            if not torch.isfinite(inputs[0][group]).all():
                raise FloatingPointError("Non-finite policy observation before action distribution")

    @staticmethod
    def _check_actor_output(module, inputs, output):
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite actor output")
        if module.distribution is not None and module.distribution._distribution is not None:
            if not torch.isfinite(module.output_std).all() or not (module.output_std > 0).all():
                raise FloatingPointError("Action std must be finite and strictly positive")

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

    def _log_locomotion(self, iteration):
        writer = self.logger.writer
        if writer is not None and self._raw_action_count:
            writer.add_scalar(
                "Audit/raw_action_abs_gt3_fraction",
                self._raw_action_exceedances.item() / self._raw_action_count,
                iteration,
            )
            writer.add_scalar("Audit/raw_action_count", self._raw_action_count, iteration)
        total_invalid = getattr(self.env.unwrapped, "invalid_world_resets", 0)
        if writer is not None:
            writer.add_scalar("Numerics/invalid_world_resets", total_invalid - self._invalid_resets_logged, iteration)
            writer.add_scalar("Numerics/invalid_world_resets_total", total_invalid, iteration)
            steps = self.env.unwrapped.common_step_counter * self.env.num_envs
            writer.add_scalar(
                "Numerics/invalid_world_resets_per_million_env_steps", total_invalid * 1e6 / max(steps, 1), iteration
            )
        self._invalid_resets_logged = total_invalid
        if (iteration + 1) % 100 == 0 and hasattr(self.env.unwrapped, "invalid_world_resets"):
            window_resets = total_invalid - self._invalid_window_start
            self._invalid_window_start = total_invalid
            window_steps = 100 * self.cfg["num_steps_per_env"] * self.env.num_envs
            rate = window_resets * 1e6 / window_steps
            if writer is not None:
                writer.add_scalar("Numerics/invalid_world_100_update_rate_per_million", rate, iteration)
                writer.add_scalar("Numerics/invalid_world_100_update_count", window_resets, iteration)
            limit = os.environ.get("ROBOT_LAB_INVALID_WORLD_WINDOW_RATE_LIMIT")
            if limit is not None and iteration + 1 > 500 and rate > float(limit):
                if writer is not None:
                    writer.flush()
                raise RuntimeError(
                    f"Newton invalid-world gate: updates {iteration - 98}-{iteration + 1}: "
                    f"{window_resets}/{window_steps} env-steps = {rate:.6f}/million > {limit}"
                )
        self._raw_action_exceedances.zero_()
        self._raw_action_count = 0

    def save(self, path, infos=None):
        env = self.env.unwrapped
        metadata = dict(infos or {})
        metadata["locomotion"] = {
            "contract": self._contract(),
            "policy_steps": env.common_step_counter + env._locomotion.step_offset,
            "recipe": env.cfg.robustness.to_dict(),
        }
        super().save(path, infos=metadata)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = (checkpoint.get("infos") or {}).get("locomotion")
        if metadata is None or metadata.get("contract") != self._contract():
            raise ValueError(
                f"Incompatible locomotion checkpoint: expected {self._contract()}. "
                "Incompatible weights cannot be resumed silently."
            )
        if "model_state_dict" in checkpoint:
            if load_cfg is None or load_cfg.get("optimizer", False):
                raise ValueError(
                    "Legacy optimizer resume is unsupported; use load_cfg with optimizer=False for inference"
                )
            actor, critic = map_legacy_model_state(
                checkpoint["model_state_dict"], self.alg.actor.state_dict(), self.alg.critic.state_dict()
            )
            migrated_checkpoint = dict(checkpoint, actor_state_dict=actor, critic_state_dict=critic)
            self.alg.load(migrated_checkpoint, load_cfg, strict=True)
            result = checkpoint["infos"]
        else:
            result = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
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


def map_legacy_model_state(state, actor_target, critic_target):
    """Strict RSL-RL 3.0.1 non-normalized MLP -> 5.x MLPModel mapping."""
    actor, critic = {}, {}
    for key, tensor in state.items():
        if key.startswith("actor."):
            destination, name = actor, "mlp." + key.removeprefix("actor.")
        elif key.startswith("critic."):
            destination, name = critic, "mlp." + key.removeprefix("critic.")
        elif key in ("std", "log_std"):
            destination, name = actor, "distribution." + key + "_param"
        else:
            raise ValueError(f"Unsupported legacy checkpoint key: {key}")
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite or non-tensor legacy checkpoint key: {key}")
        destination[name] = tensor
    for label, mapped, target in (("actor", actor, actor_target), ("critic", critic, critic_target)):
        if set(mapped) != set(target):
            raise ValueError(
                f"Strict {label} mapping keys mismatch: missing={set(target) - set(mapped)}, "
                f"extra={set(mapped) - set(target)}"
            )
        for key in mapped:
            if mapped[key].shape != target[key].shape:
                raise ValueError(f"Strict {label} mapping shape mismatch: {key}")
    return actor, critic
