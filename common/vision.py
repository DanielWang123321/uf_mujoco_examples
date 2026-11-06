# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Eye-in-Hand vision helpers: hand-eye calibration, pixel/camera projection, workspace checks,
and camera look-down IK. Frame capture lives in ``common.camera`` and is re-exported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import yaml
from ufactory_mujoco.calibration import KinematicsCalibration
from ufactory_mujoco.ik import solve_ik
from ufactory_mujoco.safety import SafetyGateError
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

from common.camera import (  # noqa: F401  # re-exported for the examples
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_SIZE,
    CameraError,
    OrbbecCameraStream,
    capture_rgbd_frame,
)
from common.paths import DEFAULT_HANDEYE_YAML, resolve_repo_path
from common.sim2real import flange_pose, make_collision_checker

# Clear tabletop workspace in the robot base frame.
WORKSPACE_BOUNDS_M: dict[str, float] = {
    "x_min": 0.20,
    "x_max": 0.60,
    "y_min": -0.30,
    "y_max": 0.22,
}


class PerceptionError(RuntimeError):
    """Raised when camera capture, hand-eye calibration, or object segmentation fails."""


def validate_xy_in_workspace(
    xy_m: np.ndarray | tuple[float, float],
    label: str,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> None:
    """Raise ``PerceptionError`` if a base-frame XY point is outside ``bounds``."""
    x, y = float(xy_m[0]), float(xy_m[1])
    if not (bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]):
        raise PerceptionError(
            f"{label} ({x:.3f}, {y:.3f}) m is outside the workspace "
            f"X=[{bounds['x_min']:.2f}, {bounds['x_max']:.2f}], "
            f"Y=[{bounds['y_min']:.2f}, {bounds['y_max']:.2f}] m"
        )


def load_handeye_calibration(yaml_path: str | Path = DEFAULT_HANDEYE_YAML) -> dict[str, Any]:
    """Load and validate an Eye-in-Hand calibration YAML (``camera`` + ``extrinsics.X_flange_cam``)."""
    path = resolve_repo_path(yaml_path).resolve()
    if not path.is_file():
        raise PerceptionError(
            f"Hand-eye calibration file not found: {path} (see calibration/README.md)"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "camera" not in data or "extrinsics" not in data:
        raise PerceptionError(f"Invalid hand-eye calibration file: {path}")
    x_fc = np.asarray(data["extrinsics"]["X_flange_cam"], dtype=np.float64)
    if x_fc.shape != (4, 4) or not np.all(np.isfinite(x_fc)):
        raise PerceptionError(f"Invalid X_flange_cam matrix: {path}")
    return data


def deproject_pixels_to_cam(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    camera_cfg: dict[str, Any],
) -> np.ndarray:
    """Deproject distorted pixels (N, 2) with optical depth (N,) to camera-frame points (N, 3)."""
    pts_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depth_m, dtype=np.float64).reshape(-1, 1)
    k_mat = np.asarray(camera_cfg["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    undist = cv2.undistortPoints(pts_uv.reshape(-1, 1, 2), k_mat, dist).reshape(-1, 2)
    return np.column_stack([undist, np.ones(len(undist), dtype=np.float64)]) * z


def project_cam_to_pixels(points_cam: np.ndarray, camera_cfg: dict[str, Any]) -> np.ndarray:
    """Project camera-frame points (N, 3) to image pixels (N, 2)."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    k_mat = np.asarray(camera_cfg["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k_mat, dist)
    return proj.reshape(-1, 2)


def compute_flange_and_cam_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q6: np.ndarray,
    x_flange_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_base_flange, T_base_cam) 4x4 matrices for arm configuration ``q6``."""
    data.qpos[:6] = np.asarray(q6, dtype=np.float64).reshape(6)
    mujoco.mj_forward(model, data)
    t_bf = flange_pose(model, data)
    t_bc = t_bf @ np.asarray(x_flange_cam, dtype=np.float64).reshape(4, 4)
    return t_bf, t_bc


# --------------------------------------------------------------------------------------
# Camera look-down IK
# --------------------------------------------------------------------------------------


def solve_camera_lookdown_q(
    target_xy_m: tuple[float, float] | np.ndarray,
    *,
    z_cam_m: float = 0.280,
    q_seed: np.ndarray,
    scene_path: str | Path,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    ignore_bodies: tuple[str, ...] = (),
) -> np.ndarray:
    """Collision-free joints that put the camera optical center at (x, y, ``z_cam_m``), looking down.

    The gripper keeps the grip-down orientation; the TCP is offset by the hand-eye extrinsics so
    the camera (not the TCP) is centered over the target.
    """
    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = make_collision_checker(scene_path, calibration, ignore_bodies=ignore_bodies)

    c_xy = np.asarray(target_xy_m, dtype=np.float64).reshape(2)
    model = checker.model
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
    site_name = model.site(site_id).name

    # 1. Reference grip-down pose to measure the camera-minus-TCP offset.
    quat = GRIP_DOWN_QUAT.copy()
    q_full = data.qpos.copy()
    q_full[:6] = np.asarray(q_seed, dtype=np.float64).reshape(6)
    q_tmp, ok = solve_ik(
        model,
        data,
        site_name,
        np.array([0.40, 0.0, 0.25], dtype=np.float64),
        quat,
        q_init=q_full,
    )
    if not ok:
        raise SafetyGateError("IK failed for the camera look-down reference pose")

    data.qpos[:] = q_tmp
    mujoco.mj_forward(model, data)
    t_bc = flange_pose(model, data) @ x_flange_cam
    offset_cam_minus_tcp = t_bc[:3, 3] - data.site_xpos[site_id].copy()

    # 2. Solve the TCP target that places the camera over the requested point.
    target_tcp = (
        np.array([c_xy[0], c_xy[1], float(z_cam_m)], dtype=np.float64) - offset_cam_minus_tcp
    )
    q_sol, ok = solve_ik(model, data, site_name, target_tcp, quat, q_init=q_tmp)
    if not ok:
        raise SafetyGateError(
            f"IK failed for the camera look-down pose above ({c_xy[0]:.3f}, {c_xy[1]:.3f})"
        )
    q6 = np.asarray(q_sol[:6], dtype=np.float64).copy()
    if checker.in_collision(q6):
        raise SafetyGateError("Camera look-down pose is in collision")
    return q6
