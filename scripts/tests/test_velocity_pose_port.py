# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Port regressions: Newton recovery, delays, clip bounds, quaternion/FK and contacts."""

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from robot_lab.assets.delayed_implicit import DelayedImplicitActuator, implicit_with_same_limits
from robot_lab.tasks.manager_based.locomotion.velocity_pose.robustness_runner import LocomotionOnPolicyRunner

from pxr import Usd, UsdPhysics

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.types import ArticulationActions

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp"
NEWTON_PATH = MDP.parent / "newton_recovery.py"
USD_PATH = ROOT / "source/robot_lab/data/Robots/unitree/go2_x5_description/usd/go2_x5/go2_x5.usd"
JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load_source("newton_recovery", NEWTON_PATH)
exporter = load_source("export_rl_sar", ROOT / "scripts/reinforcement_learning/rsl_rl/export_rl_sar.py")


def legacy_rotation_wxyz(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def usd_transform_wxyz(pos, rot):
    # Reconstruct original v2 USD GetReal/GetImaginary boundary independently.
    q = torch.tensor([rot.GetReal(), *rot.GetImaginary()], dtype=torch.float64)
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = legacy_rotation_wxyz(q)
    transform[:3, 3] = torch.tensor(tuple(pos), dtype=torch.float64)
    return transform


def reference_chain():
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {USD_PATH}")
    joints = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            children = joint.GetBody1Rel().GetTargets()
            if children:
                joints[str(children[0])] = joint
    tips = [path for path in joints if path.rsplit("/", 1)[-1] == "link6"]
    if len(tips) != 1:
        raise ValueError(f"Expected one link6: {tips}")
    chain = []
    child = tips[0]
    while child.rsplit("/", 1)[-1] != "base":
        joint = joints[child]
        parents = joint.GetBody0Rel().GetTargets()
        if not parents:
            raise ValueError(f"Broken USD chain: {joint.GetPath()}")
        name = joint.GetPrim().GetName()
        index = JOINT_NAMES.index(name) if name in JOINT_NAMES else None
        if index is None and not joint.GetPrim().IsA(UsdPhysics.FixedJoint):
            raise ValueError(f"Unmapped movable joint: {name}")
        axis = None if index is None else "XYZ".index(UsdPhysics.RevoluteJoint(joint.GetPrim()).GetAxisAttr().Get())
        before = usd_transform_wxyz(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        after = torch.linalg.inv(usd_transform_wxyz(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()))
        chain.append((index, axis, before, after))
        child = str(parents[0])
    return list(reversed(chain))


def reference_position(chain, angles):
    angles = angles.double()
    transform = torch.eye(4, dtype=torch.float64).repeat(len(angles), 1, 1)
    for index, axis, before, after in chain:
        transform = transform @ before
        if index is not None:
            direction = torch.eye(3, dtype=torch.float64)[axis]
            x, y, z = direction
            skew = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float64)
            theta = angles[:, index, None, None]
            rotation = torch.eye(4, dtype=torch.float64).repeat(len(angles), 1, 1)
            rotation[:, :3, :3] = (
                torch.eye(3, dtype=torch.float64) + theta.sin() * skew + (1 - theta.cos()) * (skew @ skew)
            )
            transform = transform @ rotation
        transform = transform @ after
    return transform[:, :3, 3]


class TestVelocityPose3Math(unittest.TestCase):
    def test_usd_fk_preserves_legacy_wxyz_rotation(self):
        module = load_source("velocity_pose_arm_kinematics_cpu", MDP / "shared/arm_kinematics.py")
        chain = reference_chain()
        movable = [index for index, _, _, _ in chain if index is not None]
        self.assertEqual(len(movable), 6)
        self.assertEqual(sorted(movable), list(range(6)))
        migrated = module.ArmKinematics(str(USD_PATH), JOINT_NAMES, "cpu")
        self.assertEqual(sum(index is not None for index, _, _, _ in migrated.chain), 6)
        generator = torch.Generator(device="cpu").manual_seed(20261005)
        angles = 2 * torch.rand(256, 6, generator=generator) - 1
        expected = reference_position(chain, angles)
        actual = migrated.position(angles).double()
        maximum_error_m = (actual - expected).norm(dim=-1).max().item()
        print(f"FK: 256 seeded samples; max position error={maximum_error_m:.12g} m; movable joints=6")
        self.assertLessEqual(maximum_error_m, 1e-6)

    def test_yaw_quaternion_preserves_legacy_rotation(self):
        module = load_source("velocity_pose_arm_controller_cpu", MDP / "high_level/arm_controller_ik.py")
        generator = torch.Generator(device="cpu").manual_seed(20261006)
        legacy = torch.randn(256, 4, generator=generator)
        legacy = legacy / legacy.norm(dim=-1, keepdim=True)
        w, x, y, z = legacy.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        expected_xyzw = torch.stack(
            (torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin(), (yaw / 2).cos()), dim=-1
        )
        actual_xyzw = module.extract_yaw_quat(legacy[:, [1, 2, 3, 0]])
        dot = (actual_xyzw * expected_xyzw).sum(-1).abs()
        torch.testing.assert_close(dot, torch.ones_like(dot), atol=1e-6, rtol=0)
        # Each matrix column is one rotated basis vector; independent standard formula.
        expected_basis = legacy_rotation_wxyz(expected_xyzw[:, [3, 0, 1, 2]])
        actual_basis = legacy_rotation_wxyz(actual_xyzw[:, [3, 0, 1, 2]])
        torch.testing.assert_close(actual_basis, expected_basis, atol=1e-6, rtol=0)
        print(
            "Yaw: 256 seeded quaternions; max rotated-basis error="
            f"{(actual_basis - expected_basis).abs().max().item():.12g}"
        )


def reward_function(name):
    path = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {"torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class TestContactSemantics(unittest.TestCase):
    def reward_env(self):
        sensor = SimpleNamespace(
            data=SimpleNamespace(
                net_normal_forces_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 10.0]] * 4])),
                net_normal_forces_w_history=SimpleNamespace(torch=torch.tensor([[[[0.0, 0.0, 10.0]] * 4]])),
                last_air_time=SimpleNamespace(torch=torch.ones(1, 4)),
            ),
            compute_first_contact=lambda dt: SimpleNamespace(torch=torch.ones(1, 4, dtype=torch.bool)),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]])),
                root_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
                body_lin_vel_w=SimpleNamespace(torch=torch.zeros(1, 4, 3)),
            )
        )

        class Scene(dict):
            pass

        scene = Scene(robot=robot)
        scene.sensors = {"contact": sensor}
        return SimpleNamespace(
            scene=scene,
            step_dt=0.02,
            command_manager=SimpleNamespace(
                get_command=lambda name: torch.tensor([[0.0, 0.0, 0.0, 0.33, 0.2, 0.0, 0.0]])
            ),
        )

    def test_reward_masks_ignore_height_and_pose(self):
        env = self.reward_env()
        sensor = SimpleNamespace(name="contact", body_ids=[0, 1, 2, 3])
        air = reward_function("feet_air_time")(env, "cmd", sensor, 0.3)
        standing = reward_function("feet_contact_without_cmd")(env, "cmd", sensor)
        torch.testing.assert_close(air, torch.zeros(1))
        torch.testing.assert_close(standing, torch.tensor([4.0]))

    def test_actual_slip_reward_does_not_subtract_base_velocity(self):
        env = self.reward_env()
        sensor = SimpleNamespace(name="contact", body_ids=[0, 1, 2, 3])
        asset = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])
        torch.testing.assert_close(reward_function("feet_slide")(env, sensor, asset), torch.zeros(1))


class RecoveryBoundaryTests(unittest.TestCase):
    def test_mask_preserves_healthy_rows_and_source(self):
        value = torch.tensor([[float("nan"), float("inf")], [-0.0, 1.125], [2.0, -7.0]])
        mask = torch.tensor([True, False, False])
        out = recovery.zero_invalid_rows(value, mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out[1:].view(torch.int32), value[1:].view(torch.int32)))
        self.assertTrue(torch.isnan(value[0, 0]))

    def test_unguarded_nonfinite_reward_still_raises(self):
        env = SimpleNamespace(_guarded_any=True, _guarded_invalid_worlds=torch.tensor([True, False]))
        reward = recovery.GuardedReward(lambda env: torch.tensor([float("nan"), float("nan")]))
        with self.assertRaises(FloatingPointError):
            reward(env)

    def test_guarded_reward_is_zero_before_manager_accumulation(self):
        env = SimpleNamespace(_guarded_any=True, _guarded_invalid_worlds=torch.tensor([True, False]))
        reward = recovery.GuardedReward(lambda env: torch.tensor([float("nan"), 1.25]))
        self.assertTrue(torch.equal(reward(env), torch.tensor([0.0, 1.25])))

    def test_no_invalid_world_returns_original_reward_object(self):
        value = torch.tensor([1.0, 2.0])
        env = SimpleNamespace(_guarded_any=False)
        self.assertIs(recovery.GuardedReward(lambda env: value)(env), value)

    def test_installed_guard_preserves_healthy_observations_and_rewards(self):
        newton = ModuleType("isaaclab_newton.physics")
        newton.NewtonCfg = type("NewtonCfg", (), {})
        termination_name = "robot_lab.tasks.manager_based.locomotion.velocity.mdp.terminations"
        termination = ModuleType(termination_name)
        termination.invalid_physics_state = lambda env: None
        original = torch.tensor([[float("nan"), float("inf")], [-0.0, 1.125]])
        reward = SimpleNamespace(func=lambda env: torch.tensor([float("nan"), 2.0]))
        env = SimpleNamespace(
            cfg=SimpleNamespace(
                sim=SimpleNamespace(physics=newton.NewtonCfg()),
                terminations=SimpleNamespace(
                    invalid_physics_state=SimpleNamespace(func=termination.invalid_physics_state)
                ),
            ),
            num_envs=2,
            device="cpu",
            reward_manager=SimpleNamespace(_term_cfgs=[reward]),
            observation_manager=SimpleNamespace(compute=lambda: {"policy": original}),
        )
        with patch.dict(sys.modules, {newton.__name__: newton, termination_name: termination}):
            recovery.install_newton_recovery(env)
        self.assertIs(env.observation_manager.compute()["policy"], original)
        env._guarded_invalid_worlds[0] = True
        env._guarded_any = True
        guarded = env.observation_manager.compute()["policy"]
        self.assertTrue(torch.isfinite(guarded).all())
        self.assertTrue(torch.equal(guarded[1].view(torch.int32), original[1].view(torch.int32)))
        self.assertTrue(torch.isnan(original[0, 0]))
        self.assertTrue(torch.equal(reward.func(env), torch.tensor([0.0, 2.0])))

    def test_invalid_reset_preserves_healthy_rng_stream_and_counts_only_invalid(self):
        # Exercise the actual reset method without importing the simulator base class.
        tree = ast.parse((NEWTON_PATH.parent / "velocity_pose_env.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        reset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[reset], type_ignores=[]), str(NEWTON_PATH), "exec"), namespace)
        calls = []
        env = SimpleNamespace(
            _guarded_invalid_worlds=torch.tensor([True, False, False]),
            device="cpu",
            invalid_world_resets=0,
        )

        def consume_rng(ids):
            calls.append(ids.clone())
            torch.rand(len(ids))

        env._reset_locomotion_idx = consume_rng
        torch.manual_seed(42)
        torch.rand(2)
        expected_next = torch.rand(8)
        torch.manual_seed(42)
        fork = torch.random.fork_rng
        # CPU has no CUDA device list; the production RNG fork otherwise runs unchanged.
        with patch.object(torch.random, "fork_rng", side_effect=lambda devices: fork(devices=[])):
            namespace["_reset_idx"](env, torch.arange(3))
        self.assertTrue(torch.equal(torch.rand(8), expected_next))
        self.assertEqual([ids.tolist() for ids in calls], [[1, 2], [0]])
        self.assertEqual(env.invalid_world_resets, 1)


class DelayedImplicitTests(unittest.TestCase):
    def test_delayed_targets_match_explicit_pd_across_partial_resets(self):
        cfg = DelayedPDActuatorCfg(
            joint_names_expr=["joint1"],
            stiffness=50.0,
            damping=5.0,
            actuator_effort_limit=20.0,
            joint_effort_limit=1e9,
            min_delay=0,
            max_delay=2,
        )
        implicit_cfg = implicit_with_same_limits(cfg)
        self.assertEqual(implicit_cfg.joint_effort_limit, 20.0)
        explicit = DelayedPDActuator(cfg, ["joint1"], slice(None), 4, "cpu")
        implicit = DelayedImplicitActuator(implicit_cfg, ["joint1"], slice(None), 4, "cpu")
        zero = torch.zeros(4, 1)
        for tick in range(10):
            if tick in (0, 5):
                ids = slice(None) if tick == 0 else torch.tensor([1, 3])
                torch.manual_seed(42 + tick)
                explicit.reset(ids)
                torch.manual_seed(42 + tick)
                implicit.reset(ids)
            target = torch.arange(4).float().reshape(4, 1) + tick

            def command():
                return ArticulationActions(
                    joint_positions=target.clone(), joint_velocities=zero.clone(), joint_efforts=zero.clone()
                )

            explicit.compute(command(), zero, zero)
            actual = implicit.compute(command(), zero, zero)
            # Explicit PD error equals the delayed target at zero state; avoid
            # the physical torque clamp by reading its computed effort.
            torch.testing.assert_close(actual.joint_positions, explicit.computed_effort / 50.0)


class ActionClipTests(unittest.TestCase):
    def test_saved_tuple_clip_reaches_export_config(self):
        env = yaml.load(
            "actions:\n  joint_pos:\n    clip:\n      .*: !!python/tuple [-10.0, 10.0]\n", Loader=yaml.BaseLoader
        )
        bounds = exporter.resolve_action_clip(env, {"clip_actions": 10.0})
        config = exporter.policy_config("test", {"contract": {"layout": "go2_x5_locomotion_v4_63"}}, bounds)["test"]
        self.assertEqual(config["clip_actions_lower"], [-10.0] * 12)
        self.assertEqual(config["clip_actions_upper"], [10.0] * 12)

    def test_disagreement_fails(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            exporter.resolve_action_clip({"actions": {"joint_pos": {"clip": {".*": [-10, 10]}}}}, {"clip_actions": 3})

    def test_single_source_and_missing_bounds(self):
        env = yaml.load("actions:\n  joint_pos:\n    clip: null\n", Loader=yaml.BaseLoader)
        self.assertEqual(exporter.resolve_action_clip(env, {"clip_actions": 3}), ([-3.0] * 12, [3.0] * 12))
        self.assertEqual(
            exporter.resolve_action_clip({"actions": {"joint_pos": {"clip": {".*": [-10, 10]}}}}, {}),
            ([-10.0] * 12, [10.0] * 12),
        )
        with self.assertRaisesRegex(ValueError, "No action clip"):
            exporter.resolve_action_clip({}, {})


class InvalidWindowTests(unittest.TestCase):
    def test_individual_window_not_cumulative_rate(self):
        env = SimpleNamespace(
            invalid_world_resets=1000,
            common_step_counter=12000,
        )
        runner = SimpleNamespace(
            env=SimpleNamespace(unwrapped=env, num_envs=1024),
            cfg={"num_steps_per_env": 24},
            logger=SimpleNamespace(writer=None),
            _invalid_resets_logged=0,
            _invalid_window_start=0,
            _raw_action_count=0,
            _raw_action_exceedances=torch.tensor(0),
        )
        with patch.dict(os.environ, ROBOT_LAB_INVALID_WORLD_WINDOW_RATE_LIMIT="10"):
            LocomotionOnPolicyRunner._log_locomotion(runner, 499)
            env.invalid_world_resets += 24
            LocomotionOnPolicyRunner._log_locomotion(runner, 599)
            env.invalid_world_resets += 25
            with self.assertRaisesRegex(RuntimeError, "updates 601-700: 25/2457600"):
                LocomotionOnPolicyRunner._log_locomotion(runner, 699)


if __name__ == "__main__":
    unittest.main()
