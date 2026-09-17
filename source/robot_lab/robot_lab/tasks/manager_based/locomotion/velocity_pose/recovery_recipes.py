# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Frozen first recovery screen for the deployed 64D policy, without payload.

Each named recipe owns the fields below; use 'none' for manual overrides.
Arm speed/acceleration are simulation stress settings, not hardware limits.
"""

RECIPES = {
    "control": {},
    "arm": {"arm_max_velocity": 3.5, "arm_max_acceleration": 16.0, "arm_reversal_fraction": 0.3},
    "recover": {
        "arm_max_velocity": 3.5,
        "arm_max_acceleration": 16.0,
        "arm_reversal_fraction": 0.3,
        "nearfall_reset_fraction": 0.1,
        "nearfall_reset_min_degrees": 30.0,
        "nearfall_reset_max_degrees": 50.0,
        "nearfall_reset_angular_speed": 1.5,
        "recovery_action_rate_scale": 0.25,
        "recovery_upright_weight": 2.0,
    },
}


def apply_recovery_recipe(settings):
    name = settings.recovery_recipe
    if name == "none":
        return
    if name not in RECIPES:
        raise ValueError(f"Unknown recovery recipe: {name}")
    common = {
        "observation_layout": "go2_x5_locomotion_v2_64",
        "domain_rand": "sim2real",
        "iteration_override": 8000,
        "recovery_metrics": True,
        "payload_max_kg": 0.0,
        "hard_fraction": 0.2,
        "arm_mode": -1,
        "arm_max_velocity": 2.5,
        "arm_max_acceleration": 8.0,
        "arm_reversal_fraction": 0.0,
        "nearfall_reset_fraction": 0.0,
        "recovery_action_rate_scale": 1.0,
        "recovery_upright_weight": 0.0,
    }
    for key, value in {**common, **RECIPES[name]}.items():
        setattr(settings, key, value)
