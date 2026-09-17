#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Export the 64D Go2-X5 locomotion actor into an rl_sar policy bundle.

This exporter is intentionally independent of Isaac Sim.  It reconstructs the
RSL-RL MLP from the checkpoint actor tensors, validates the saved locomotion
contract, scripts the actor, and writes the matching rl_sar configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import torch
import yaml
from pathlib import Path

EXPECTED_LAYOUT = "go2_x5_locomotion_v2_64"
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
        "--agent-config",
        type=Path,
        help="RSL-RL agent.yaml; defaults to CHECKPOINT_DIR/params/agent.yaml",
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Set policy/go2_x5/base.yaml config_name to the exported bundle.",
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
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")
    metadata = (checkpoint.get("infos") or {}).get("locomotion")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint has no infos.locomotion metadata")
    contract = metadata.get("contract") or {}
    expected = {
        "layout": EXPECTED_LAYOUT,
        "policy_terms": EXPECTED_TERMS,
        "policy_dim": [64],
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


def build_actor(state: dict, agent_cfg: dict) -> torch.nn.Sequential:
    policy_cfg = agent_cfg.get("policy") or {}
    if policy_cfg.get("actor_obs_normalization", False):
        raise ValueError("actor observation normalization is not supported by this exporter")
    actor_weights = []
    for key, tensor in state.items():
        match = re.fullmatch(r"actor\.(\d+)\.weight", key)
        if match:
            actor_weights.append((int(match.group(1)), tensor))
    actor_weights.sort()
    if not actor_weights or actor_weights[0][1].shape[1] != 64 or actor_weights[-1][1].shape[0] != 12:
        shapes = [tuple(tensor.shape) for _, tensor in actor_weights]
        raise ValueError(f"expected a 64D -> 12D actor, found layers {shapes}")

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
        probe = actor(torch.zeros(1, 64))
    if probe.shape != (1, 12) or not torch.isfinite(probe).all():
        raise ValueError(f"invalid exported actor probe: shape={tuple(probe.shape)}")
    return actor


def policy_config(config_key: str, metadata: dict) -> dict:
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
            "num_observations": 64,
            "observations": [
                "lin_vel",
                "ang_vel",
                "gravity_vec",
                "robot_lab/velocity_pose_commands",
                "roboduet/leg_actions",
                "roboduet/leg_dof_pos",
                "roboduet/leg_dof_vel",
                "roboduet/arm_dof_pos",
                "robot_lab/arm_dof_vel",
            ],
            "observations_history": [0],
            "observations_history_priority": "time",
            "clip_obs": 1.0e6,
            "clip_actions_lower": [-3.0] * 12,
            "clip_actions_upper": [3.0] * 12,
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


def set_default(base_yaml: Path, config_name: str) -> None:
    text = base_yaml.read_text()
    updated, replacements = re.subn(
        r'(?m)^(\s*config_name:\s*)["\']?[^"\'\n]+["\']?\s*$',
        rf'\1"{config_name}"',
        text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"could not uniquely update config_name in {base_yaml}")
    base_yaml.write_text(updated)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    rl_sar_root = args.rl_sar_root.resolve()
    agent_path = (args.agent_config or checkpoint_path.parent / "params" / "agent.yaml").resolve()
    checkpoint, state = load_and_validate_checkpoint(checkpoint_path)
    metadata = checkpoint["infos"]["locomotion"]
    with agent_path.open() as stream:
        agent_cfg = yaml.safe_load(stream)
    actor = build_actor(state, agent_cfg)

    config_key = f"go2_x5/{args.config_name}"
    output_dir = rl_sar_root / "policy" / "go2_x5" / args.config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(actor)
    scripted.save(str(output_dir / "policy.pt"))
    with (output_dir / "config.yaml").open("w") as stream:
        stream.write("# Generated by robot_lab export_rl_sar.py.\n")
        yaml.safe_dump(policy_config(config_key, metadata), stream, sort_keys=False)

    with torch.inference_mode():
        torch.manual_seed(20260914)
        probes = torch.randn(64, 64)
        max_error = float((actor(probes) - torch.jit.load(str(output_dir / "policy.pt"))(probes)).abs().max())
    if max_error != 0.0:
        raise ValueError(f"TorchScript replay mismatch: max_abs_error={max_error}")

    manifest = {
        "format": "robot_lab_rl_sar_v1",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_iteration": int(checkpoint.get("iter", -1)),
        "contract": EXPECTED_LAYOUT,
        "actor_input_dim": 64,
        "actor_output_dim": 12,
        "policy_joint_order": EXPECTED_LEG_ORDER,
        "arm_joint_order": EXPECTED_ARM_ORDER,
        "torchscript_replay_max_abs_error": max_error,
        "policy_sha256": sha256(output_dir / "policy.pt"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.set_default:
        set_default(rl_sar_root / "policy" / "go2_x5" / "base.yaml", args.config_name)
    print(output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
