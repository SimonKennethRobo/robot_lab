# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""No-simulator tests for user-facing robustness Hydra normalization."""

import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "reinforcement_learning/rsl_rl/robustness_cli.py"
SPEC = importlib.util.spec_from_file_location("robustness_cli", MODULE)
robustness_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(robustness_cli)


class RobustnessCliTests(unittest.TestCase):
    def test_converts_only_integer_float_overrides(self):
        args = [
            "env.robustness.hard_fraction=0",
            "env.robustness.easy_reset_degrees=-10",
            "env.robustness.arm_max_velocity=2",
            "env.robustness.hard_fraction=0.2",
            "env.robustness.cohort_seed=0",
            "agent.max_iterations=0",
        ]
        self.assertEqual(
            robustness_cli.normalize_robustness_float_overrides(args),
            [
                "env.robustness.hard_fraction=0.0",
                "env.robustness.easy_reset_degrees=-10.0",
                "env.robustness.arm_max_velocity=2.0",
                "env.robustness.hard_fraction=0.2",
                "env.robustness.cohort_seed=0",
                "agent.max_iterations=0",
            ],
        )

    def test_preserves_hydra_add_prefix(self):
        self.assertEqual(
            robustness_cli.normalize_robustness_float_overrides(["+env.robustness.payload_max_kg=0"]),
            ["+env.robustness.payload_max_kg=0.0"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
