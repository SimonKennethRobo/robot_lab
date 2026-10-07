# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Bounded Isaac Lab 3.0 construction/step receipt for one registered task."""

import argparse
import json
import socket
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import robot_lab.tasks  # noqa: F401

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import parse_env_cfg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--output", required=True)
parser.add_argument("--preset", default=None)
add_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
cfg = parse_env_cfg(
    args.task,
    device=args.device,
    num_envs=args.num_envs,
    overrides=[] if args.preset is None else [f"presets={args.preset}"],
)
cfg.seed = 42
cfg.log_dir = str(output)
import gymnasium as gym
import torch

with launch_simulation(cfg, args):
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        with torch.inference_mode():
            obs, _ = env.reset()

            def finite_obs(observations):
                for name, tensor in observations.items():
                    assert torch.isfinite(tensor).all(), f"nonfinite observation {name}"

            finite_obs(obs)
            for step in range(args.steps):
                action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                obs, reward, terminated, timeout, _ = env.step(action)
                finite_obs(obs)
                assert torch.isfinite(reward).all(), f"nonfinite reward step={step}"
            receipt = {
                "task": args.task,
                "backend": type(cfg.sim.physics).__name__,
                "node": socket.gethostname(),
                "num_envs": env.num_envs,
                "steps": args.steps,
                "status": "pass",
                "obs_dims": {name: list(value.shape) for name, value in obs.items()},
                "command_dim": list(env.command_manager.get_command("base_velocity_pose").shape),
                "action_dim": env.action_manager.total_action_dim,
            }
            if "Go2-X5" in args.task:
                assert obs["policy"].shape[1] == 63
                assert receipt["command_dim"][1] == 6
                assert receipt["action_dim"] == 12
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print("VELOCITY_POSE_SMOKE_PASS", json.dumps(receipt))
    finally:
        env.close()
