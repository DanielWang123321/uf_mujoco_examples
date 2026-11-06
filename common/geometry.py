# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Small quaternion and vector helpers (MuJoCo ``wxyz`` quaternion convention)."""

from __future__ import annotations

import mujoco
import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def yaw_rotate_quat(yaw_rad: float, base_quat: np.ndarray) -> np.ndarray:
    """Left-multiply ``base_quat`` by a world +Z rotation of ``yaw_rad``."""
    if abs(yaw_rad) < 1e-6:
        return base_quat.copy()
    z_rot = np.array([np.cos(0.5 * yaw_rad), 0.0, 0.0, np.sin(0.5 * yaw_rad)], dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, z_rot, base_quat)
    return out


def rotate_axis_up_quat(axis: np.ndarray, tool_quat: np.ndarray) -> np.ndarray:
    """Return ``tool_quat`` pre-rotated so that world vector ``axis`` is turned to world +Z."""
    up = np.array([0.0, 0.0, 1.0])
    cross = np.cross(axis, up)
    length = float(np.linalg.norm(cross))
    if length < 1e-9:
        correction = (
            np.array([1.0, 0.0, 0.0, 0.0]) if axis[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        )
    else:
        angle = float(np.arctan2(length, np.dot(axis, up)))
        correction = np.r_[np.cos(angle / 2), cross / length * np.sin(angle / 2)]
    result = np.empty(4)
    mujoco.mju_mulQuat(result, correction, tool_quat)
    return result


def canonical_quat(quat: np.ndarray) -> np.ndarray:
    """Return the quaternion in MuJoCo's canonical ``mju_mat2Quat`` hemisphere."""
    mat = np.empty(9, dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    mujoco.mju_mat2Quat(out, mat)
    return out
