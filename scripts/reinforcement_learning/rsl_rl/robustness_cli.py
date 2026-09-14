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
    "push_max_xy",
    "push_max_angular",
    "payload_max_kg",
    "recovery_grace_s",
    "failure_tilt_degrees",
    "failure_height_m",
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
