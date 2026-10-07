# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ARX5 Arm IK Controller for Curriculum Stage 2 Training"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    pass


def extract_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion from full quaternion."""
    w, x, y, z = quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    zeros = torch.zeros_like(yaw)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)

    return yaw_quat
