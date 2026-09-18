# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Versioned observation layouts and weight-only input migrations."""

LAYOUTS = {
    "go2_x5_locomotion_v2_64": (True, False),
    "go2_x5_locomotion_v3_65": (True, True),
    "go2_x5_locomotion_v4_63": (False, False),
    "go2_x5_locomotion_v5_64": (False, True),
}
POLICY_TERMS = [
    "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "actions",
    "joint_pos", "joint_vel", "arm_joint_pos", "arm_joint_vel",
]


def layout_features(layout):
    """Stable feature identities, including optional height-error and legacy yaw."""
    yaw, height = LAYOUTS[layout]
    features = [f"base_{i}" for i in range(9)]
    if height:
        features.append("height_error")
    features += ["vx", "vy", "wz", "height", "roll", "pitch"]
    if yaw:
        features.append("pose_yaw")
    return features + [f"joint_action_{i}" for i in range(48)]


def layout_terms(layout):
    terms = list(POLICY_TERMS)
    if LAYOUTS[layout][1]:
        terms.insert(3, "height_error")
    return terms


def migrate_input_weights(state, source, target):
    """Drop legacy zero-yaw columns and/or insert a zero height-error column.

    Resume requires an exact contract. Migrations discard the optimizer and
    must be requested with --warm_start.
    """
    for contract in (source, target):
        if not isinstance(contract, dict) or contract.get("layout") not in LAYOUTS:
            raise ValueError("Warm-start checkpoint has an incompatible locomotion contract")
        layout = contract["layout"]
        width = len(layout_features(layout))
        if any(contract.get(key) != [width] for key in ("policy_dim", "critic_dim")):
            raise ValueError("Observation dimensions do not match the named layout")
        if contract.get("policy_terms") != layout_terms(layout):
            raise ValueError("Observation term order does not match the named layout")
    for key in ("leg_joint_order", "arm_joint_order", "steps_per_iteration"):
        if key not in source or source[key] != target.get(key):
            raise ValueError(f"Incompatible warm-start contract field: {key}")
    old_features, new_features = layout_features(source["layout"]), layout_features(target["layout"])
    removed = set(old_features) - set(new_features)
    added = set(new_features) - set(old_features)
    if removed - {"pose_yaw"} or added - {"height_error"}:
        raise ValueError("Unsupported observation migration")
    result = dict(state)
    for name in ("actor.0.weight", "critic.0.weight"):
        old = state[name]
        if old.shape[1] != len(old_features):
            raise ValueError(f"Checkpoint input width disagrees with contract: {name}")
        if old_features != new_features:
            new = old.new_zeros((old.shape[0], len(new_features)))
            for index, feature in enumerate(new_features):
                if feature in old_features:
                    new[:, index] = old[:, old_features.index(feature)]
            result[name] = new
    return result, "pose_yaw" in removed, "height_error" in added
