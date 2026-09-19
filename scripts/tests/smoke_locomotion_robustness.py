# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Bounded Isaac Lab smoke: runtime invariants plus optional short PPO updates."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=144)
parser.add_argument("--iteration", type=int, default=8000)
parser.add_argument("--domain_rand", choices=("none", "benchmark", "sim2real"), default="sim2real")
parser.add_argument("--ppo_iterations", type=int, default=0)
parser.add_argument("--observation_layout", default="go2_x5_locomotion_v4_63")
parser.add_argument("--full_extension_fraction", type=float, default=0.25)
parser.add_argument(
    "--arm_full_extension_eval",
    action="store_true",
    help="Use a fixed zero-velocity, full-extension scenario at the final arm curriculum target.",
)
parser.add_argument("--output", default="outputs/locomotion_smoke")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
app = app_launcher.app

import gymnasium as gym
import json
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity_pose.mdp.shared.robustness import push_locomotion, terrain_height
from robot_lab.tasks.manager_based.locomotion.velocity_pose.observation_contract import LAYOUTS, layout_features


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    cfg.robustness.iteration_override = 16000 if args.arm_full_extension_eval else args.iteration
    cfg.robustness.domain_rand = "none" if args.arm_full_extension_eval else args.domain_rand
    cfg.robustness.observation_layout = args.observation_layout
    cfg.robustness.arm_full_extension_fraction = 1.0 if args.arm_full_extension_eval else args.full_extension_fraction
    if args.arm_full_extension_eval:
        cfg.robustness.arm_full_extension_standing_fraction = 1.0
        command = cfg.commands.base_velocity_pose
        command.resampling_time_range = (1.0e9, 1.0e9)
        command.rel_standing_envs = 1.0
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.heading_command = False
        command.ranges.heading = None
        command.ranges.height = (command.default_height, command.default_height)
        command.ranges.roll = (0.0, 0.0)
        command.ranges.pitch = (0.0, 0.0)
    cfg.log_dir = args.output
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 1
        cfg.scene.terrain.terrain_generator.num_cols = 4
        cfg.scene.terrain.max_init_terrain_level = None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        width = len(layout_features(args.observation_layout))
        assert obs["policy"].shape == (args.num_envs, width), obs["policy"].shape
        assert obs["critic"].shape == (args.num_envs, width)
        assert env.command_manager.get_command("base_velocity_pose").shape[1] == 6 + LAYOUTS[args.observation_layout][0]
        assert env.action_manager.total_action_dim == 12
        if "Mild" in args.task:
            import numpy as np

            from pxr import Usd, UsdGeom

            ground = env.sim.stage.GetPrimAtPath("/World/ground")
            points = [
                np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                for prim in Usd.PrimRange(ground)
                if prim.IsA(UsdGeom.Mesh)
            ]
            terrain_vertices = np.concatenate(points)
            generator = cfg.scene.terrain.terrain_generator
            # The importer also adds a border box extending below the walking
            # surface. Inspect the interior height field, excluding that box.
            interior = (np.abs(terrain_vertices[:, 0]) < generator.num_rows * generator.size[0] / 2 - 0.1) & (
                np.abs(terrain_vertices[:, 1]) < generator.num_cols * generator.size[1] / 2 - 0.1
            )
            terrain_vertices = terrain_vertices[interior]
            assert terrain_vertices[:, 2].min() >= -0.040001
            assert terrain_vertices[:, 2].max() <= 0.040001
        runtime = env._locomotion
        cohort = runtime.hard.clone()
        action = env.action_manager.get_term("joint_pos")
        actual_order = [env.scene["robot"].joint_names[i] for i in action._joint_ids]
        assert actual_order == cfg.dog_joint_names
        assert len(action._arm_joint_ids) == 6
        torch.testing.assert_close(
            action._arm_controller.q, env.scene["robot"].data.joint_pos[:, action._arm_joint_ids]
        )
        assert env._pose_visualizer is None
        push_ids = torch.arange(min(2, args.num_envs), device=env.device)
        before_push = env.scene["robot"].data.root_vel_w.clone()
        push_locomotion(env, push_ids)
        delta = env.scene["robot"].data.root_vel_w - before_push
        push_scale = runtime.progress("push") * (cfg.robustness.domain_rand != "none")
        assert delta[:, :2].abs().max() <= cfg.robustness.push_max_xy * push_scale + 1e-6
        assert delta[:, 3:].abs().max() <= cfg.robustness.push_max_angular * push_scale + 1e-6
        assert (delta[:, 2] == 0).all()
        if push_scale:
            assert delta[push_ids].abs().sum() > 0
        runtime.capture_evaluation = True
        max_fk_error = 0.0
        for _ in range(args.steps):
            obs, reward, terminated, timeout, info = env.step(torch.zeros(args.num_envs, 12, device=env.device))
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            frame = runtime.last_evaluation
            fk_error = (runtime.arm_kinematics.position(frame["arm_pos"]) - frame["ee_body"]).norm(dim=-1).max()
            max_fk_error = max(max_fk_error, float(fk_error))
            assert fk_error < 0.003, f"USD FK differs from measured link6: {fk_error} m"
            if args.arm_full_extension_eval:
                assert (frame["command"][:, :3] == 0).all()
            assert (action._arm_controller.q <= action._arm_controller.upper + 1e-6).all()
            assert (action._arm_controller.q >= action._arm_controller.lower - 1e-6).all()
        assert torch.equal(cohort, runtime.hard)
        height, valid = terrain_height(env)
        assert valid.all(), "Base scanner must hit ground in every environment"
        # Auto-reset may have resampled a payload after the final physics step.
        runtime.apply_payload()
        mass = runtime.payload_mass * runtime.progress("arm") * (cfg.robustness.domain_rand != "none")
        torch.testing.assert_close(runtime.forces[:, 0, 2], -9.81 * mass)
        # Check a partial reset does not reassign cohorts or retain arm velocity.
        ids = torch.arange(min(2, args.num_envs), device=env.device)
        env._reset_idx(ids)
        assert torch.equal(cohort, runtime.hard)
        assert (action._arm_controller.v[ids] == 0).all()
        report = {
            "task": args.task,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "policy_dim": width,
            "observation_layout": args.observation_layout,
            "action_dim": 12,
            "hard_envs": int(cohort.sum()),
            "iteration": runtime.iteration,
            "domain_rand": cfg.robustness.domain_rand,
            "ground_height_min_max": [float(height.min()), float(height.max())],
            "leg_joint_order": actual_order,
            "payload_kg_max": float(mass.max()),
            "pd_gains": {
                name: [float(actuator.stiffness.min()), float(actuator.stiffness.max())]
                for name, actuator in env.scene["robot"].actuators.items()
            },
            "ppo_iterations": args.ppo_iterations,
            "push_delta_max": float(delta.abs().max()),
            "arm_full_extension_eval": args.arm_full_extension_eval,
            "ee_fk_max_error_m": max_fk_error,
        }
        if args.ppo_iterations:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

            from robot_lab.tasks.manager_based.locomotion.velocity_pose.robustness_runner import (
                LocomotionOnPolicyRunner,
            )

            agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
            agent.logger = "tensorboard"
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = LocomotionOnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(output / "ppo"), device=env.device)
            before = {key: value.detach().clone() for key, value in runner.alg.policy.state_dict().items()}
            runner.learn(num_learning_iterations=args.ppo_iterations, init_at_random_ep_len=False)
            after = runner.alg.policy.state_dict()
            assert all(torch.isfinite(value).all() for value in after.values())
            assert any(not torch.equal(before[key], value) for key, value in after.items())
            report["finite_changed_policy"] = True
            checkpoint = output / "resume_test.pt"
            expected_steps = env.common_step_counter + runtime.step_offset
            runner.save(str(checkpoint))
            original_override = runtime.cfg.iteration_override
            runtime.cfg.iteration_override = -1
            runtime.step_offset = -env.common_step_counter
            runner.load(str(checkpoint))
            assert env.common_step_counter + runtime.step_offset == expected_steps
            assert runtime.iteration == expected_steps / cfg.robustness.steps_per_iteration
            runtime.cfg.iteration_override = original_override
            report["checkpoint_clock_restored"] = True
            legacy = torch.load(checkpoint, weights_only=False)
            legacy["infos"] = None
            torch.save(legacy, output / "legacy_test.pt")
            try:
                runner.load(str(output / "legacy_test.pt"))
            except ValueError as error:
                assert "Incompatible locomotion checkpoint" in str(error)
            else:
                raise AssertionError("Unversioned checkpoint was accepted")
            report["legacy_checkpoint_rejected"] = True
        records = [json.loads(line) for line in (output / "locomotion_metrics.jsonl").read_text().splitlines()]
        if args.arm_full_extension_eval:
            extension_case_key = "Robustness/all/all/arm_full_extension_case_mae"
            extension_error_key = "Robustness/all/all/arm_extension_target_error_rad_mae"
            assert records and records[-1][extension_case_key] == 1.0
            assert records[-1][extension_error_key] >= 0.0
        for record in records:
            count = record["Robustness/all/all/samples"]
            assert count == args.num_envs * cfg.robustness.metrics_interval
            assert count == record["Robustness/easy/all/samples"] + record["Robustness/hard/all/samples"]
            assert count == record["Robustness/all/early/samples"] + record["Robustness/all/late/samples"]
        report["metric_counts_verified"] = True
        # Synthetic invalid input must fail before any simulator or PPO step.
        env.cfg.log_dir = str(output / "expected_fault")
        bad_action = torch.zeros(args.num_envs, 12, device=env.device)
        bad_action[0, 0] = float("nan")
        try:
            runtime.check_finite(action=bad_action)
        except FloatingPointError:
            report["nonfinite_action_guard"] = True
        else:
            raise AssertionError("Non-finite action guard did not raise")
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("LOCOMOTION_SMOKE_PASS", json.dumps(report))
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close()
