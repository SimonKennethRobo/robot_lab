# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Normalize scalar Hydra overrides used by the GO2-X5 robustness recipe."""

import re

_ROBUSTNESS_FLOAT_FIELDS = (
    "hard_fraction",
    "easy_reset_degrees",
    "hard_reset_degrees",
    "arm_max_velocity",
    "arm_max_acceleration",
    "arm_workspace_fraction",
    "arm_reversal_fraction",
    "push_max_xy",
    "push_max_angular",
    "payload_max_kg",
    "recovery_grace_s",
    "failure_tilt_degrees",
    "failure_height_m",
    "nearfall_reset_fraction",
    "nearfall_reset_min_degrees",
    "nearfall_reset_max_degrees",
    "nearfall_reset_angular_speed",
    "recovery_action_rate_scale",
    "recovery_upright_weight",
    "stance_width_target_m",
    "stance_width_std_m",
    "stance_width_reward_weight",
    "rear_stance_width_target_m",
    "rear_stance_width_std_m",
    "rear_stance_width_reward_weight",
    "gait_timing_variance_cost_weight",
    "gait_swing_height_cost_weight",
    "gait_swing_height_body_target_m",
    "gait_joint_velocity_mirror_cost_weight",
    "gait_contact_sync_reward_weight",
)
_INTEGER_LITERAL = re.compile(
    rf"^(?P<prefix>\+{{0,2}}env\.robustness\.(?:{'|'.join(_ROBUSTNESS_FLOAT_FIELDS)}))=" r"(?P<value>[+-]?\d+)$"
)


def normalize_robustness_float_overrides(arguments: list[str]) -> list[str]:
    """Convert ``env.robustness.float_field=0`` to ``=0.0`` for Hydra.

    IsaacLab compares scalar types exactly while merging its generated config.
    Hydra parses unqualified integer literals as ``int``, despite those values
    being valid real-valued settings in this task.
    """
    normalized = []
    for argument in arguments:
        match = _INTEGER_LITERAL.fullmatch(argument)
        normalized.append(f"{match['prefix']}={match['value']}.0" if match else argument)
    return normalized
