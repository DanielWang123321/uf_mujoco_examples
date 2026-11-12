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
    CameraStream,
    OrbbecCameraStream,
    RealSenseCameraStream,
    capture_rgbd_frame,
    create_camera_stream,
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


def component_touches_fov_boundary(component: np.ndarray, fov_mask: np.ndarray) -> bool:
    """Reject cropped silhouettes: a partial cup can reverse the inferred bottom/rim axis."""
    interior = cv2.erode(
        fov_mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return bool(np.any(component & (interior == 0)))


def cup_overview_fov(
    height: int, width: int, camera_cfg: dict[str, Any], *, multi: bool = False
) -> np.ndarray:
    """Use the full uncropped D435 frame; Gemini's lower housing mask is specific to that mounting."""
    if camera_cfg.get("type") == "realsense":
        return np.ones((height, width), dtype=bool)
    ys, xs = np.mgrid[:height, :width]
    top, bottom, left, right = (0.06 if multi else 0.18), 0.73, 0.18, 0.92
    return (
        (ys >= int(top * height))
        & (ys < int(bottom * height))
        & (xs >= int(left * width))
        & (xs < int(right * width))
    )


def white_cup_mask(
    hsv: np.ndarray,
    camera_cfg: dict[str, Any],
    *,
    saturation: int = 75,
    value: int = 155,
) -> np.ndarray:
    """Keep warm-tinted D435 paper (including concave cavity reflections) while excluding mat/logos."""
    green = (hsv[:, :, 0] >= 65) & (hsv[:, :, 0] <= 105) & (hsv[:, :, 1] >= 25)
    if camera_cfg.get("type") == "realsense":
        neutral = (hsv[:, :, 1] < 110) & (hsv[:, :, 2] > value)
        warm_cavity = (
            (hsv[:, :, 0] >= 5)
            & (hsv[:, :, 0] <= 22)
            & (hsv[:, :, 1] < 165)
            & (hsv[:, :, 2] > max(value, 160))
        )
        return (~green) & (neutral | warm_cavity)
    return (~green) & (hsv[:, :, 1] < saturation) & (hsv[:, :, 2] > value)


def cup_grasp_yaw_offset(handeye: dict[str, Any]) -> float:
    """Use the equivalent jaw orientation that keeps a rear-mounted camera up during the flip."""
    x_fc = np.asarray(handeye["extrinsics"]["X_flange_cam"])
    return float(np.pi) if x_fc[0, 3] < 0 else 0.0


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
    dist = _opencv_distortion(camera_cfg)
    undist = cv2.undistortPoints(pts_uv.reshape(-1, 1, 2), k_mat, dist).reshape(-1, 2)
    return np.column_stack([undist, np.ones(len(undist), dtype=np.float64)]) * z


def project_cam_to_pixels(points_cam: np.ndarray, camera_cfg: dict[str, Any]) -> np.ndarray:
    """Project camera-frame points (N, 3) to image pixels (N, 2)."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    k_mat = np.asarray(camera_cfg["K"], dtype=np.float64).reshape(3, 3)
    dist = _opencv_distortion(camera_cfg)
    proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k_mat, dist)
    return proj.reshape(-1, 2)


def _opencv_distortion(camera_cfg: dict[str, Any]) -> np.ndarray:
    dist = np.asarray(camera_cfg.get("distortion", [0.0] * 5), dtype=np.float64)
    if not np.all(np.isfinite(dist)):
        raise PerceptionError("Camera distortion coefficients must be finite")
    # D435 color reports inverse Brown-Conrady with all-zero coefficients on this device.
    # Zero coefficients are a pinhole model regardless of the SDK's model label.
    if not np.any(dist):
        return np.zeros(5, dtype=np.float64)
    model = camera_cfg.get("distortion_model", camera_cfg.get("model", "opencv"))
    if model not in (
        "opencv",
        "brown_conrady",
        "brown_conrady_k6",
        "plumb_bob",
        "rational_polynomial",
    ):
        raise PerceptionError(f"Unsupported nonzero camera distortion model: {model!r}")
    if dist.shape not in ((4,), (5,), (8,), (12,), (14,)):
        raise PerceptionError("Invalid OpenCV camera distortion coefficients")
    return dist


def _infer_camera_type(camera_cfg: dict[str, Any] | None) -> str | None:
    """Infer ``realsense`` or ``orbbec`` when a calibration YAML omits ``camera.type``."""
    if not isinstance(camera_cfg, dict):
        return None
    explicit = camera_cfg.get("type")
    if explicit in ("realsense", "orbbec"):
        return str(explicit)
    name = str(camera_cfg.get("name", "")).lower()
    if any(token in name for token in ("realsense", "d435", "d405")):
        return "realsense"
    if any(token in name for token in ("orbbec", "gemini")):
        return "orbbec"
    return None


def resolve_camera_config(
    handeye: dict[str, Any],
    color_bgr: np.ndarray,
    depth_m: np.ndarray | None,
    camera_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use live color intrinsics when supplied; YAML remains valid for offline images."""
    if color_bgr is None or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise PerceptionError("Invalid BGR color image")
    if depth_m is not None and depth_m.shape != color_bgr.shape[:2]:
        raise PerceptionError("RGB and depth images have different sizes")
    cfg = dict(handeye["camera"] if camera_metadata is None else camera_metadata)
    if cfg.get("type") not in ("realsense", "orbbec"):
        inferred = _infer_camera_type(cfg) or _infer_camera_type(handeye.get("camera"))
        if inferred is not None:
            cfg["type"] = inferred
    h, w = color_bgr.shape[:2]
    if cfg.get("width", w) != w or cfg.get("height", h) != h:
        raise PerceptionError("Camera intrinsics dimensions do not match the RGB-D image")
    if camera_metadata is not None:
        missing = {"width", "height", "K", "distortion", "distortion_model"} - cfg.keys()
        if missing:
            raise PerceptionError(f"Incomplete live camera metadata: {sorted(missing)}")
    k = np.asarray(cfg.get("K"), dtype=np.float64)
    if (
        k.shape != (3, 3)
        or not np.all(np.isfinite(k))
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        raise PerceptionError("Invalid camera intrinsic matrix K")
    _opencv_distortion(cfg)
    return cfg


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


def target_cylinder_in_camera_view(
    t_base_flange: np.ndarray,
    target_xy_m: tuple[float, float] | np.ndarray,
    *,
    top_z_m: float,
    radius_m: float = 0.032,
    table_z_m: float = 0.0,
    min_cam_clearance_m: float = 0.15,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    camera_meta: dict[str, Any] | None = None,
) -> bool:
    """Return True if the upright target cylinder projects inside the camera frame with sufficient depth clearance."""
    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam
    if float(t_base_cam[2, 3]) - float(top_z_m) < float(min_cam_clearance_m):
        return False

    cam_src = dict(camera_meta) if camera_meta and "K" in camera_meta else dict(handeye["camera"])
    w = int(cam_src["width"])
    h = int(cam_src["height"])
    cam_cfg = {
        "width": w,
        "height": h,
        "K": cam_src["K"],
        "distortion": cam_src.get("distortion", [0.0] * 5),
        "distortion_model": cam_src.get("distortion_model", "pinhole"),
    }
    r_bc = t_base_cam[:3, :3]
    t_cam_base_r = r_bc.T
    t_cam_base_t = -r_bc.T @ t_base_cam[:3, 3]
    target = np.asarray(target_xy_m, dtype=np.float64).reshape(2)
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ring_bot = np.column_stack(
        [
            target[0] + radius_m * np.cos(angles),
            target[1] + radius_m * np.sin(angles),
            np.full_like(angles, float(table_z_m)),
        ]
    )
    ring_top = np.column_stack(
        [
            target[0] + radius_m * np.cos(angles),
            target[1] + radius_m * np.sin(angles),
            np.full_like(angles, float(top_z_m)),
        ]
    )
    pts_cam = np.vstack([ring_bot, ring_top]) @ t_cam_base_r.T + t_cam_base_t
    if np.any(pts_cam[:, 2] <= 0.10):
        return False
    uv = project_cam_to_pixels(pts_cam, cam_cfg)
    return bool(
        np.all(uv[:, 0] >= 4.0)
        and np.all(uv[:, 0] <= w - 5.0)
        and np.all(uv[:, 1] >= 4.0)
        and np.all(uv[:, 1] <= h - 5.0)
    )


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
