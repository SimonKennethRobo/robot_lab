#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Export 63D (or legacy 64D) Go2-X5 locomotion actors into rl_sar bundles.

This exporter is intentionally independent of Isaac Sim.  It reconstructs the
RSL-RL MLP from the checkpoint actor tensors, validates the saved locomotion
contract, scripts the actor, and writes the matching rl_sar configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import yaml

EXPORT_LAYOUTS = {
    "go2_x5_locomotion_v2_64": (64, "robot_lab/velocity_pose_commands"),
    "go2_x5_locomotion_v4_63": (63, "robot_lab/velocity_pose_commands_6d"),
}
EXPECTED_TERMS = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "actions",
    "joint_pos",
    "joint_vel",
    "arm_joint_pos",
    "arm_joint_vel",
]
EXPECTED_LEG_ORDER = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
EXPECTED_ARM_ORDER = [f"joint{i}" for i in range(1, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rl-sar-root", type=Path, required=True)
    parser.add_argument("--config-name", default="robot_lab_mild_s2r_s42_8200")
    parser.add_argument(
        "--output-dir", type=Path, help="Write the bundle here; keep the reference rl_sar tree read only"
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        help="RSL-RL agent.yaml; defaults to CHECKPOINT_DIR/params/agent.yaml",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_checkpoint(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if state is None and isinstance(checkpoint.get("actor_state_dict"), dict):
        native = checkpoint["actor_state_dict"]
        if any(not key.startswith(("mlp.", "distribution.")) for key in native):
            raise ValueError("Normalized/recurrent/CNN native actors are unsupported by this exporter")
        state = {"actor." + key.removeprefix("mlp."): value for key, value in native.items() if key.startswith("mlp.")}
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")
    metadata = (checkpoint.get("infos") or {}).get("locomotion")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint has no infos.locomotion metadata")
    contract = metadata.get("contract") or {}
    if contract.get("layout") not in EXPORT_LAYOUTS:
        raise ValueError("Unsupported export layout; height-error layouts need a deployment height observation")
    input_dim, _ = EXPORT_LAYOUTS[contract["layout"]]
    expected = {
        "policy_terms": EXPECTED_TERMS,
        "policy_dim": [input_dim],
        "leg_joint_order": EXPECTED_LEG_ORDER,
        "arm_joint_order": EXPECTED_ARM_ORDER,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"incompatible checkpoint contract {key}: {contract.get(key)!r} != {value!r}")
    for key, tensor in state.items():
        if torch.is_tensor(tensor) and (tensor.is_floating_point() or tensor.is_complex()):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite model tensor: {key}")
    return checkpoint, state


def activation(name: str) -> torch.nn.Module:
    table = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh,
    }
    try:
        return table[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unsupported actor activation: {name}") from error


def build_actor(state: dict, agent_cfg: dict, input_dim: int = 63) -> torch.nn.Sequential:
    policy_cfg = agent_cfg.get("actor") or agent_cfg.get("policy") or {}
    if policy_cfg.get("obs_normalization", policy_cfg.get("actor_obs_normalization", False)):
        raise ValueError("actor observation normalization is not supported by this exporter")
    actor_weights = []
    for key, tensor in state.items():
        match = re.fullmatch(r"actor\.(\d+)\.weight", key)
        if match:
            actor_weights.append((int(match.group(1)), tensor))
    actor_weights.sort()
    if not actor_weights or actor_weights[0][1].shape[1] != input_dim or actor_weights[-1][1].shape[0] != 12:
        shapes = [tuple(tensor.shape) for _, tensor in actor_weights]
        raise ValueError(f"expected a {input_dim}D -> 12D actor, found layers {shapes}")

    modules: list[torch.nn.Module] = []
    activation_name = str(policy_cfg.get("activation", "elu"))
    for layer_number, (state_index, weight) in enumerate(actor_weights):
        bias_key = f"actor.{state_index}.bias"
        if bias_key not in state:
            raise ValueError(f"missing actor bias: {bias_key}")
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(state[bias_key])
        modules.append(layer)
        if layer_number + 1 < len(actor_weights):
            modules.append(activation(activation_name))
    actor = torch.nn.Sequential(*modules).eval()
    with torch.inference_mode():
        probe = actor(torch.zeros(1, input_dim))
    if probe.shape != (1, 12) or not torch.isfinite(probe).all():
        raise ValueError(f"invalid exported actor probe: shape={tuple(probe.shape)}")
    return actor


def resolve_action_clip(env_cfg: dict, agent_cfg: dict) -> tuple[list[float], list[float]]:
    """Resolve raw leg-action bounds and reject inconsistent saved parameters."""
    env_clip = env_cfg.get("actions", {}).get("joint_pos", {}).get("clip")
    if env_clip == "null":
        env_clip = None
    agent_clip = agent_cfg.get("clip_actions")
    env_bounds = None
    if env_clip is not None:
        env_bounds = []
        for joint in EXPECTED_LEG_ORDER:
            matches = [bounds for pattern, bounds in env_clip.items() if re.fullmatch(pattern, joint)]
            if len(matches) != 1:
                raise ValueError(f"Expected one env action clip for {joint}, found {matches}")
            lower, upper = map(float, matches[0])
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError(f"Invalid env action clip for {joint}: {matches[0]}")
            env_bounds.append((lower, upper))
    agent_bounds = None
    if agent_clip is not None:
        magnitude = float(agent_clip)
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError(f"Invalid agent clip_actions: {agent_clip}")
        agent_bounds = [(-magnitude, magnitude)] * 12
    if env_bounds is not None and agent_bounds is not None and env_bounds != agent_bounds:
        raise ValueError(f"env.yaml action clip disagrees with agent.yaml clip_actions: {env_bounds} != {agent_bounds}")
    bounds = env_bounds if env_bounds is not None else agent_bounds
    if bounds is None:
        raise ValueError("No action clip in resolved env.yaml or agent.yaml")
    return [b[0] for b in bounds], [b[1] for b in bounds]


def policy_config(config_key: str, metadata: dict, action_clip: tuple[list[float], list[float]]) -> dict:
    input_dim, command_term = EXPORT_LAYOUTS[metadata["contract"]["layout"]]
    # Policy order equals Unitree/MuJoCo sensor order: FR, FL, RR, RL, X5.
    default_dof_pos = [
        -0.1,
        0.8,
        -1.5,
        0.1,
        0.8,
        -1.5,
        -0.1,
        1.0,
        -1.5,
        0.1,
        1.0,
        -1.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    recipe = metadata.get("recipe") or {}
    height_range = [float(value) for value in recipe.get("height_range", (0.28, 0.38))]
    roll_range = [float(value) for value in recipe.get("roll_range", (-0.262, 0.262))]
    pitch_range = [float(value) for value in recipe.get("pitch_range", (-0.175, 0.175))]
    base_height_target = sum(height_range) / 2.0
    return {
        config_key: {
            "model_name": "policy.pt",
            "num_observations": input_dim,
            "observations": [
                "lin_vel",
                "ang_vel",
                "gravity_vec",
                command_term,
                "roboduet/leg_actions",
                "roboduet/leg_dof_pos",
                "roboduet/leg_dof_vel",
                "roboduet/arm_dof_pos",
                "robot_lab/arm_dof_vel",
            ],
            "observations_history": [0],
            "observations_history_priority": "time",
            "clip_obs": 1.0e6,
            "clip_actions_lower": action_clip[0],
            "clip_actions_upper": action_clip[1],
            "dt": 0.005,
            "decimation": 4,
            "num_of_dofs": 18,
            "num_leg_dofs": 12,
            "num_arm_dofs": 6,
            "wheel_indices": [],
            "lin_vel_scale": 2.0,
            "ang_vel_scale": 0.25,
            "dof_pos_scale": 1.0,
            "dof_vel_scale": 0.05,
            "arm_dof_vel_scale": 0.1,
            "base_height_target": base_height_target,
            "action_scale": [0.125, 0.25, 0.25] * 4 + [0.0] * 6,
            "rl_kp": [25.0] * 12 + [50.0, 50.0, 80.0, 30.0, 20.0, 20.0],
            "rl_kd": [0.5] * 12 + [5.0, 10.0, 10.0, 2.5, 2.0, 1.0],
            "torque_limits": [33.5] * 12 + [20.0] * 6,
            "default_dof_pos": default_dof_pos,
            "joint_mapping": list(range(18)),
            "dog_commands_extra": [],
            "use_base_state_sensor": True,
            "limit_vel_x": [-1.0, 1.0],
            "limit_vel_y": [-0.6, 0.6],
            "limit_vel_yaw": [-1.0, 1.0],
            "limit_body_height": [round(value - base_height_target, 6) for value in height_range],
            "limit_body_roll": roll_range,
            "limit_body_pitch": pitch_range,
        }
    }


def validate_runtime(rl_sar_root: Path, metadata: dict) -> None:
    """Fail before export if the target SDK cannot assemble the six-wide command."""
    _, term = EXPORT_LAYOUTS[metadata["contract"]["layout"]]
    if term.endswith("_6d"):
        sdk = rl_sar_root / "library/core/rl_sdk/rl_sdk.cpp"
        if not sdk.is_file() or term not in sdk.read_text():
            raise ValueError(f"Update/rebuild rl_sar with {term} support before exporting to {rl_sar_root}")


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    rl_sar_root = args.rl_sar_root.resolve()
    agent_path = (args.agent_config or checkpoint_path.parent / "params" / "agent.yaml").resolve()
    checkpoint, state = load_and_validate_checkpoint(checkpoint_path)
    metadata = checkpoint["infos"]["locomotion"]
    input_dim, _ = EXPORT_LAYOUTS[metadata["contract"]["layout"]]
    validate_runtime(rl_sar_root, metadata)
    with agent_path.open() as stream:
        agent_cfg = yaml.safe_load(stream)
    # BaseLoader reads Isaac Lab's Python tuple tags without constructing objects.
    with (checkpoint_path.parent / "params" / "env.yaml").open() as stream:
        env_cfg = yaml.load(stream, Loader=yaml.BaseLoader)
    action_clip = resolve_action_clip(env_cfg, agent_cfg)
    actor = build_actor(state, agent_cfg, input_dim)

    config_key = f"go2_x5/{args.config_name}"
    output_dir = args.output_dir or rl_sar_root / "policy" / "go2_x5" / args.config_name
    output_dir.mkdir(parents=True, exist_ok=args.output_dir is None)
    scripted = torch.jit.script(actor)
    scripted.save(str(output_dir / "policy.pt"))
    with (output_dir / "config.yaml").open("w") as stream:
        stream.write("# Generated by robot_lab export_rl_sar.py.\n")
        yaml.safe_dump(policy_config(config_key, metadata, action_clip), stream, sort_keys=False)

    with torch.inference_mode():
        torch.manual_seed(20260914)
        probes = torch.randn(64, input_dim)
        max_error = float((actor(probes) - torch.jit.load(str(output_dir / "policy.pt"))(probes)).abs().max())
    if max_error != 0.0:
        raise ValueError(f"TorchScript replay mismatch: max_abs_error={max_error}")

    manifest = {
        "format": "robot_lab_rl_sar_v1",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_iteration": int(checkpoint.get("iter", -1)),
        "contract": metadata["contract"]["layout"],
        "actor_input_dim": input_dim,
        "actor_output_dim": 12,
        "clip_actions_lower": action_clip[0],
        "clip_actions_upper": action_clip[1],
        "policy_joint_order": EXPECTED_LEG_ORDER,
        "arm_joint_order": EXPECTED_ARM_ORDER,
        "torchscript_replay_max_abs_error": max_error,
        "policy_sha256": sha256(output_dir / "policy.pt"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
