# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests; no simulator installation or initialization is needed."""

import ast
import importlib.util
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp/shared/robustness_math.py"
spec = importlib.util.spec_from_file_location("robustness_math", MODULE)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


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


class RobustnessTests(unittest.TestCase):
    def test_recovery_hysteresis_counts_and_no_repeated_timeout(self):
        tracker = math.RecoveryTracker(2, "cpu", 0.1)
        age = torch.tensor([1.0, 5.0])
        bad = torch.tensor([0.7, 0.7])
        good = torch.tensor([0.0, 0.0])
        height = torch.tensor([0.33, 0.33])
        no_failure = torch.tensor([False, False])
        tracker.update(bad, height, good, age, no_failure)
        first = tracker.flush()
        self.assertEqual(first["Recovery/reset/attempts"].item(), 1)
        self.assertEqual(first["Recovery/rollout/attempts"].item(), 1)
        for _ in range(3):
            tracker.update(good, height, good, age, no_failure)
        second = tracker.flush()
        self.assertEqual(second["Recovery/reset/recovered"].item(), 1)
        self.assertEqual(second["Recovery/rollout/recovered"].item(), 1)
        for _ in range(50):
            tracker.update(bad, height, good, age, no_failure)
        third = tracker.flush()
        self.assertEqual(third["Recovery/reset/attempts"].item(), 1)
        self.assertEqual(third["Recovery/reset/unresolved"].item(), 1)
        self.assertFalse(tracker.active.any())

    def test_recovery_partial_reset_censors_and_failure_is_not_success(self):
        tracker = math.RecoveryTracker(3, "cpu", 0.1)
        tracker.update(
            torch.ones(3), torch.full((3,), 0.2), torch.ones(3), torch.ones(3), torch.zeros(3, dtype=torch.bool)
        )
        tracker.reset(torch.tensor([0, 1]), torch.tensor([False, True]))
        self.assertTrue(tracker.active[2])
        data = tracker.flush()
        self.assertEqual(data["Recovery/reset/censored"].item(), 1)
        self.assertEqual(data["Recovery/reset/failed"].item(), 1)
        self.assertEqual(data["Recovery/reset/active"].item(), 1)
        tracker.reset(torch.tensor([0, 1]))
        self.assertEqual(tracker.flush()["Recovery/reset/censored"].item(), 0)

    def test_coherent_arm_reversals_remain_bounded(self):
        torch.manual_seed(4)
        arm = math.BoundedArmMotion(
            torch.full((32, 6), -0.5),
            torch.full((32, 6), 0.5),
            0.02,
            max_velocity=3.5,
            max_acceleration=16.0,
            reversal_fraction=1.0,
        )
        arm.reset(torch.arange(32), torch.zeros(32, 6))
        self.assertTrue((arm.mode == 3).all())
        saw_positive = torch.zeros(32, 6, dtype=torch.bool)
        saw_negative = torch.zeros_like(saw_positive)
        for _ in range(500):
            previous = arm.v.clone()
            arm.step(1.0)
            self.assertTrue((arm.q.abs() <= 0.500001).all())
            self.assertLessEqual(arm.v.abs().max().item(), 3.50001)
            self.assertLessEqual((arm.v - previous).abs().max().item(), 0.32001)
            saw_positive |= arm.v > 0.1
            saw_negative |= arm.v < -0.1
        self.assertTrue((saw_positive & saw_negative).all())

    def reward_env(self):
        sensor = SimpleNamespace(
            data=SimpleNamespace(
                net_forces_w=torch.tensor([[[0.0, 0.0, 10.0]] * 4]),
                net_forces_w_history=torch.tensor([[[[0.0, 0.0, 10.0]] * 4]]),
                last_air_time=torch.ones(1, 4),
            ),
            compute_first_contact=lambda dt: torch.ones(1, 4, dtype=torch.bool),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
                root_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
                body_lin_vel_w=torch.zeros(1, 4, 3),
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

    def test_fixed_cohorts_independent_of_rng_and_reset_order(self):
        torch.manual_seed(42)
        state = torch.random.get_rng_state().clone()
        mask = math.fixed_hard_mask(100, 0.2, 1234, "cpu")
        self.assertEqual(mask.sum().item(), 20)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(mask, math.fixed_hard_mask(100, 0.2, 1234, "cpu")))
        self.assertFalse(torch.equal(mask, math.fixed_hard_mask(100, 0.2, 1235, "cpu")))

    def test_ramp_endpoints(self):
        self.assertEqual([math.ramp(i, 1000, 8000) for i in (0, 1000, 4500, 8000, 9000)], [0, 0, 0.5, 1, 1])
        with self.assertRaises(ValueError):
            math.ramp(0, 1, 1)

    def test_ground_height_is_per_environment(self):
        hits = torch.tensor([[1.0, 1.2], [float("inf"), 3.0], [float("nan"), float("inf")]])
        height, valid = math.ground_height(hits, torch.tensor([0.0, 0.0, 0.4]))
        torch.testing.assert_close(height, torch.tensor([1.1, 3.0, 0.4]))
        self.assertEqual(valid.tolist(), [True, True, False])

    def test_stationary_contact_feet_have_zero_slip(self):
        velocity = torch.tensor([[[0.0, 0.0, 3.0], [1.0, 2.0, 0.0]], [[3.0, 4.0, 0.0], [7.0, 8.0, 0.0]]])
        contact = torch.tensor([[True, False], [True, False]])
        torch.testing.assert_close(math.contact_slip(velocity, contact), torch.tensor([0.0, 5.0]))

    def test_offset_load_moment(self):
        force, torque = math.gravity_wrench(torch.tensor([1.0, 0.0]), torch.tensor([[0.1, 0.2, 0.0], [1.0, 1.0, 1.0]]))
        torch.testing.assert_close(force, torch.tensor([[0.0, 0.0, -9.81], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(torque, torch.tensor([[-1.962, 0.981, 0.0], [0.0, 0.0, 0.0]]))

    def test_bounded_arm_all_modes_and_partial_reset(self):
        torch.manual_seed(3)
        for mode in (0, 1, 2):
            arm = math.BoundedArmMotion(
                torch.full((32, 6), -0.3),
                torch.full((32, 6), 0.4),
                0.02,
                max_velocity=2.5,
                max_acceleration=8,
                fixed_mode=mode,
            )
            ids = torch.arange(32)
            q0 = torch.full((32, 6), 0.39)
            arm.reset(ids, q0)
            torch.testing.assert_close(arm.step(0), q0)
            prev = arm.v.clone()
            for step in range(1200):
                q = arm.step(min(1.0, step / 100))
                self.assertTrue(torch.isfinite(q).all())
                self.assertLessEqual(q.max().item(), 0.400001)
                self.assertGreaterEqual(q.min().item(), -0.300001)
                self.assertLessEqual(arm.v.abs().max().item(), 2.50001)
                self.assertLessEqual(((arm.v - prev) / 0.02).abs().max().item(), 8.001)
                prev = arm.v.clone()
            untouched = arm.q[1:].clone()
            arm.reset(torch.tensor([0]), torch.full((1, 6), 0.15))
            torch.testing.assert_close(arm.q[1:], untouched)
            torch.testing.assert_close(arm.q[0], torch.full((6,), 0.15))
            self.assertTrue((arm.v[0] == 0).all())

    def test_wbc_arm_zero_probabilities_are_applied(self):
        arm = math.BoundedArmMotion(
            torch.full((8, 6), -1.0),
            torch.full((8, 6), 1.0),
            0.02,
            max_velocity=5.0,
            max_acceleration=10.0,
            fixed_mode=0,
            accel_resample_time_s=0.01,
            zero_accel_probability=1.0,
            zero_velocity_probability=1.0,
        )
        arm.reset(torch.arange(8), torch.zeros(8, 6))
        torch.testing.assert_close(arm.step(1.0), torch.zeros(8, 6))
        torch.testing.assert_close(arm.v, torch.zeros(8, 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
