"""Eye-in-Hand RGB-D perception, look-down secondary refinement, and closed-loop verification for 0.0.3 screw_nut."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.vision import (
    CAMERA_TOOL_SRC,
    DEFAULT_HANDEYE_YAML,
    PY312_PYTHON,
    WORKSPACE_BOUNDS_M,
    OrbbecCameraStream,
    PerceptionError,
    capture_rgbd_frame,
    deproject_pixels_to_cam,
    load_handeye_calibration,
    project_cam_to_pixels,
    validate_xy_in_workspace,
)

# Physical specs from prd.md 0.0.3 and Gemini 305 scan
SCREW_TOTAL_LEN_MM = 45.0
SCREW_HEAD_LEN_MM = 14.0
SCREW_HEAD_DIAM_MM = 20.0
SCREW_THREAD_DIAM_MM = 12.0

NUT_BOT_DIAM_MM = 29.0
NUT_TOP_DIAM_MM = 20.0
NUT_BORE_DIAM_MM = 12.0
NUT_HEIGHT_MM = 14.5
NUT_FLANGE_HEIGHT_MM = 3.8
MIN_NUT_SCREW_SEPARATION_M = 0.045

__all__ = [
    "CAMERA_TOOL_SRC",
    "DEFAULT_HANDEYE_YAML",
    "PY312_PYTHON",
    "WORKSPACE_BOUNDS_M",
    "NutPoseEstimate",
    "OrbbecCameraStream",
    "PerceptionError",
    "ScrewPoseEstimate",
    "ThreadedNutVerification",
    "capture_rgbd_frame",
    "deproject_pixels_to_cam",
    "detect_nut_and_screw_overview",
    "detect_nut_near",
    "detect_screw_near",
    "load_handeye_calibration",
    "pose_estimates_from_manual",
    "project_cam_to_pixels",
    "save_detection_overlay",
    "save_post_thread_overlay",
    "validate_xy_in_workspace",
    "verify_threaded_nut",
]


@dataclass(frozen=True)
class NutPoseEstimate:
    """Estimated 3D tabletop pose of the hex flange nut in the robot base frame."""

    xy_m: tuple[float, float]
    yaw_deg: float
    table_z_m: float
    height_mm: float
    outer_diam_mm: float
    point_count: int
    source: str = "camera"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "xy_m": [float(self.xy_m[0]), float(self.xy_m[1])],
            "yaw_deg": float(self.yaw_deg),
            "table_z_m": float(self.table_z_m),
            "height_mm": float(self.height_mm),
            "outer_diam_mm": float(self.outer_diam_mm),
            "point_count": int(self.point_count),
        }


@dataclass(frozen=True)
class ScrewPoseEstimate:
    """Estimated 3D tabletop position of the upright glued M12 screw in the robot base frame."""

    xy_m: tuple[float, float]
    table_z_m: float
    height_mm: float
    point_count: int
    source: str = "camera"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "xy_m": [float(self.xy_m[0]), float(self.xy_m[1])],
            "table_z_m": float(self.table_z_m),
            "height_mm": float(self.height_mm),
            "point_count": int(self.point_count),
        }


@dataclass(frozen=True)
class ThreadedNutVerification:
    """Closed-loop visual verification result after threading the nut onto the glued screw."""

    ok: bool
    screw_xy_m: tuple[float, float]
    measured_xy_m: tuple[float, float]
    xy_error_mm: float
    nut_top_height_mm: float
    assembly_top_height_mm: float
    nut_ring_points: int
    cleared_initial_nut_site: bool = True
    reason: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "screw_xy_m": [float(self.screw_xy_m[0]), float(self.screw_xy_m[1])],
            "measured_xy_m": [float(self.measured_xy_m[0]), float(self.measured_xy_m[1])],
            "xy_error_mm": float(self.xy_error_mm),
            "nut_top_height_mm": float(self.nut_top_height_mm),
            "assembly_top_height_mm": float(self.assembly_top_height_mm),
            "nut_ring_points": int(self.nut_ring_points),
            "cleared_initial_nut_site": bool(self.cleared_initial_nut_site),
            "reason": self.reason,
        }


def pose_estimates_from_manual(
    nut_xy_m: tuple[float, float] | np.ndarray,
    nut_yaw_deg: float,
    screw_xy_m: tuple[float, float] | np.ndarray,
    *,
    table_z_m: float = 0.0,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[NutPoseEstimate, ScrewPoseEstimate]:
    """Validate and construct manual NutPoseEstimate and ScrewPoseEstimate."""
    n_xy = np.asarray(nut_xy_m, dtype=np.float64).reshape(2)
    s_xy = np.asarray(screw_xy_m, dtype=np.float64).reshape(2)
    validate_xy_in_workspace(n_xy, "手动指定的螺母中心", bounds)
    validate_xy_in_workspace(s_xy, "手动指定的固定螺丝中心", bounds)
    sep = float(np.linalg.norm(n_xy - s_xy))
    if sep < MIN_NUT_SCREW_SEPARATION_M:
        raise PerceptionError(
            f"螺母初始位置与固定螺丝间距过近 ({sep * 1000:.1f} mm < "
            f"{MIN_NUT_SCREW_SEPARATION_M * 1000:.0f} mm)，存在夹爪干涉风险"
        )
    nut_est = NutPoseEstimate(
        xy_m=(float(n_xy[0]), float(n_xy[1])),
        yaw_deg=float(nut_yaw_deg),
        table_z_m=float(table_z_m),
        height_mm=NUT_HEIGHT_MM,
        outer_diam_mm=NUT_BOT_DIAM_MM,
        point_count=0,
        source="manual",
    )
    screw_est = ScrewPoseEstimate(
        xy_m=(float(s_xy[0]), float(s_xy[1])),
        table_z_m=float(table_z_m),
        height_mm=SCREW_TOTAL_LEN_MM,
        point_count=0,
        source="manual",
    )
    return nut_est, screw_est


def _foreground_mask_against_green_mat(color_bgr: np.ndarray) -> np.ndarray:
    """Segment metallic / non-green foreground objects against the green anti-static mat."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    is_green = (hsv[:, :, 0] >= 65) & (hsv[:, :, 0] <= 105) & (hsv[:, :, 1] >= 60)
    return (~is_green) & (hsv[:, :, 2] >= 70)


def _estimate_hex_yaw_deg(xy_centered: np.ndarray) -> float:
    """Estimate hex flat yaw angle in [-30, +30) deg using 6-fold angular harmonic on outer rim points."""
    if len(xy_centered) < 12:
        return 0.0
    radii = np.hypot(xy_centered[:, 0], xy_centered[:, 1])
    # Focus on upper hex body rim (radius between 6.2 mm and 11.5 mm)
    rim_mask = (radii >= 0.0062) & (radii <= 0.0115)
    pts = xy_centered[rim_mask] if int(np.count_nonzero(rim_mask)) >= 12 else xy_centered
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    # For a regular hexagon, vertices are at yaw + 30 deg + k*60 deg (larger radius),
    # while flats are at yaw + k*60 deg. Weighting by radius aligns 6*theta with vertices.
    weights = np.hypot(pts[:, 0], pts[:, 1])
    z6 = np.sum(weights * np.exp(1j * 6.0 * angles))
    if abs(z6) < 1e-9:
        return 0.0
    vertex_angle_rad = float(np.angle(z6)) / 6.0
    flat_yaw_deg = (math.degrees(vertex_angle_rad) - 30.0) % 60.0
    if flat_yaw_deg >= 30.0:
        flat_yaw_deg -= 60.0
    return round(flat_yaw_deg, 2)


def detect_nut_near(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    expected_xy_m: tuple[float, float] | np.ndarray,
    *,
    search_radius_m: float = 0.080,
    exclude_xy_m: tuple[float, float] | np.ndarray | None = None,
    exclude_radius_m: float = 0.042,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[NutPoseEstimate | None, dict[str, Any]]:
    """Detect and refine the hex flange nut pose near expected_xy_m in look-down RGB-D view."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = handeye["camera"]
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam
    exp_xy = np.asarray(expected_xy_m, dtype=np.float64).reshape(2)
    excl_xy = (
        np.asarray(exclude_xy_m, dtype=np.float64).reshape(2)
        if exclude_xy_m is not None
        else None
    )

    if color_bgr is None or depth_m is None or color_bgr.shape[:2] != depth_m.shape[:2]:
        raise PerceptionError("输入的 RGB 与深度图尺寸无效或不匹配")

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    fov_mask = (
        (ys >= int(0.06 * h))
        & (ys < int(0.88 * h))
        & (xs >= int(0.06 * w))
        & (xs < int(0.94 * w))
    )
    valid_d = np.isfinite(depth_m) & (depth_m > 0.07) & (depth_m < 0.72) & fov_mask
    if int(np.count_nonzero(valid_d)) < 200:
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "insufficient_valid_depth",
        }

    fg_mask = _foreground_mask_against_green_mat(color_bgr)
    px_all = np.column_stack([xs[valid_d], ys[valid_d]])
    pts_cam = deproject_pixels_to_cam(px_all, depth_m[valid_d], cam_cfg)
    pts_base = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
    fg_all = fg_mask[valid_d]

    dist_to_exp = np.linalg.norm(pts_base[:, :2] - exp_xy, axis=1)
    bg_ring = (~fg_all) & (dist_to_exp >= 0.022) & (dist_to_exp <= max(0.095, search_radius_m + 0.02)) & (
        np.abs(pts_base[:, 2]) < 0.030
    )
    local_table_z = float(np.median(pts_base[bg_ring, 2])) if int(np.count_nonzero(bg_ring)) >= 60 else 0.0

    cand_mask = (
        fg_all
        & (dist_to_exp <= float(search_radius_m))
        & (pts_base[:, 2] >= local_table_z + 0.0020)
        & (pts_base[:, 2] <= local_table_z + 0.0240)
    )
    if excl_xy is not None:
        cand_mask &= np.linalg.norm(pts_base[:, :2] - excl_xy, axis=1) > float(exclude_radius_m)

    cand_pts = pts_base[cand_mask]
    if len(cand_pts) < 60:
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "nut_not_found_in_search_radius",
        }

    # Robustly cluster around the densest 2D median to reject isolated noise specks
    rough_xy = np.median(cand_pts[:, :2], axis=0)
    in_cluster = np.linalg.norm(cand_pts[:, :2] - rough_xy, axis=1) <= 0.022
    nut_pts = cand_pts[in_cluster]
    if len(nut_pts) < 55:
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "nut_cluster_too_small",
        }

    rel_z = nut_pts[:, 2] - local_table_z
    height_p98_mm = float(np.percentile(rel_z, 98.0) * 1000.0)
    if not (8.5 <= height_p98_mm <= 22.0):
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": f"nut_height_out_of_range({height_p98_mm:.1f}mm)",
        }

    # Prefer upper hex body points (z >= 5.5 mm above table) for coaxial center estimation
    upper_pts = nut_pts[rel_z >= 0.0055]
    center_xy = (
        np.median(upper_pts[:, :2], axis=0)
        if len(upper_pts) >= 25
        else np.median(nut_pts[:, :2], axis=0)
    )
    try:
        validate_xy_in_workspace(center_xy, "感知到的螺母中心", workspace_bounds)
    except PerceptionError:
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "nut_out_of_workspace",
        }

    radii = np.linalg.norm(nut_pts[:, :2] - center_xy, axis=1)
    outer_diam_mm = float(2.0 * np.percentile(radii, 95.0) * 1000.0)
    yaw_src = (upper_pts[:, :2] - center_xy) if len(upper_pts) >= 25 else (nut_pts[:, :2] - center_xy)
    yaw_deg = _estimate_hex_yaw_deg(yaw_src)

    est = NutPoseEstimate(
        xy_m=(float(center_xy[0]), float(center_xy[1])),
        yaw_deg=yaw_deg,
        table_z_m=local_table_z,
        height_mm=round(height_p98_mm, 2),
        outer_diam_mm=round(outer_diam_mm, 2),
        point_count=len(nut_pts),
        source="camera",
    )
    return est, {
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
        "nut_points_base": nut_pts,
        "reason": "ok",
    }


def detect_screw_near(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    expected_xy_m: tuple[float, float] | np.ndarray,
    *,
    search_radius_m: float = 0.080,
    exclude_xy_m: tuple[float, float] | np.ndarray | None = None,
    exclude_radius_m: float = 0.042,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[ScrewPoseEstimate | None, dict[str, Any]]:
    """Detect and refine the upright tabletop-glued M12 screw near expected_xy_m."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = handeye["camera"]
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam
    exp_xy = np.asarray(expected_xy_m, dtype=np.float64).reshape(2)
    excl_xy = (
        np.asarray(exclude_xy_m, dtype=np.float64).reshape(2)
        if exclude_xy_m is not None
        else None
    )

    if color_bgr is None or depth_m is None or color_bgr.shape[:2] != depth_m.shape[:2]:
        raise PerceptionError("输入的 RGB 与深度图尺寸无效或不匹配")

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    fov_mask = (
        (ys >= int(0.06 * h))
        & (ys < int(0.88 * h))
        & (xs >= int(0.06 * w))
        & (xs < int(0.94 * w))
    )
    valid_d = np.isfinite(depth_m) & (depth_m > 0.07) & (depth_m < 0.72) & fov_mask
    if int(np.count_nonzero(valid_d)) < 200:
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "insufficient_valid_depth",
        }

    fg_mask = _foreground_mask_against_green_mat(color_bgr)
    px_all = np.column_stack([xs[valid_d], ys[valid_d]])
    pts_cam = deproject_pixels_to_cam(px_all, depth_m[valid_d], cam_cfg)
    pts_base = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
    fg_all = fg_mask[valid_d]

    dist_to_exp = np.linalg.norm(pts_base[:, :2] - exp_xy, axis=1)
    bg_ring = (~fg_all) & (dist_to_exp >= 0.020) & (dist_to_exp <= max(0.095, search_radius_m + 0.02)) & (
        np.abs(pts_base[:, 2]) < 0.030
    )
    local_table_z = float(np.median(pts_base[bg_ring, 2])) if int(np.count_nonzero(bg_ring)) >= 60 else 0.0

    # Upright M12 screw reaches 45.0 mm height; select elevated shaft points (z > 20 mm)
    shaft_mask = (
        fg_all
        & (dist_to_exp <= float(search_radius_m))
        & (pts_base[:, 2] >= local_table_z + 0.020)
        & (pts_base[:, 2] <= local_table_z + 0.054)
    )
    if excl_xy is not None:
        shaft_mask &= np.linalg.norm(pts_base[:, :2] - excl_xy, axis=1) > float(exclude_radius_m)

    shaft_pts = pts_base[shaft_mask]
    if len(shaft_pts) < 35:
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "screw_not_found_in_search_radius",
        }

    top_xy = np.median(shaft_pts[:, :2], axis=0)
    full_screw_mask = (
        fg_all
        & (np.linalg.norm(pts_base[:, :2] - top_xy, axis=1) <= 0.018)
        & (pts_base[:, 2] >= local_table_z + 0.002)
        & (pts_base[:, 2] <= local_table_z + 0.054)
    )
    screw_pts = pts_base[full_screw_mask]
    rel_z = screw_pts[:, 2] - local_table_z
    height_p98_mm = float(np.percentile(rel_z, 98.0) * 1000.0)
    if not (34.0 <= height_p98_mm <= 52.0):
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": f"screw_height_out_of_range({height_p98_mm:.1f}mm)",
        }

    # Use upper shaft tip (z >= 25 mm) for coaxial center so any base glue/shadow doesn't bias XY
    tip_pts = screw_pts[rel_z >= 0.025]
    center_xy = np.median(tip_pts[:, :2], axis=0) if len(tip_pts) >= 20 else top_xy

    try:
        validate_xy_in_workspace(center_xy, "感知到的固定螺丝中心", workspace_bounds)
    except PerceptionError:
        return None, {
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "screw_out_of_workspace",
        }

    est = ScrewPoseEstimate(
        xy_m=(float(center_xy[0]), float(center_xy[1])),
        table_z_m=local_table_z,
        height_mm=round(height_p98_mm, 2),
        point_count=len(screw_pts),
        source="camera",
    )
    return est, {
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
        "screw_points_base": screw_pts,
        "reason": "ok",
    }


def detect_nut_and_screw_overview(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    *,
    nut_prior_xy: tuple[float, float] = (0.3538, -0.0114),
    screw_prior_xy: tuple[float, float] = (0.3560, -0.1028),
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[NutPoseEstimate, ScrewPoseEstimate, dict[str, Any]]:
    """Detect both the tabletop nut and the upright glued screw from an overview RGB-D frame."""
    screw_est, screw_dbg = detect_screw_near(
        color_bgr,
        depth_m,
        t_base_flange,
        screw_prior_xy,
        search_radius_m=0.14,
        handeye_yaml=handeye_yaml,
        workspace_bounds=workspace_bounds,
    )
    if screw_est is None:
        raise PerceptionError(f"全局视野未检测到桌面固定螺丝 (原因: {screw_dbg.get('reason')})")

    nut_est, nut_dbg = detect_nut_near(
        color_bgr,
        depth_m,
        t_base_flange,
        nut_prior_xy,
        search_radius_m=0.14,
        exclude_xy_m=screw_est.xy_m,
        exclude_radius_m=0.042,
        handeye_yaml=handeye_yaml,
        workspace_bounds=workspace_bounds,
    )
    if nut_est is None:
        raise PerceptionError(f"全局视野未检测到桌面六角法兰螺母 (原因: {nut_dbg.get('reason')})")

    debug_info = {
        "t_base_cam": nut_dbg["t_base_cam"],
        "camera_cfg": nut_dbg["camera_cfg"],
        "nut_points_base": nut_dbg.get("nut_points_base"),
        "screw_points_base": screw_dbg.get("screw_points_base"),
    }
    return nut_est, screw_est, debug_info


def verify_threaded_nut(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    screw_xy_m: tuple[float, float] | np.ndarray,
    *,
    initial_nut_xy_m: tuple[float, float] | np.ndarray | None = None,
    expected_nut_bottom_z_mm: float = SCREW_HEAD_LEN_MM,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    table_z_m: float = 0.0,
) -> tuple[ThreadedNutVerification, dict[str, Any]]:
    """Verify in closed loop that the initial nut site is cleared and the nut sits coaxially on the screw."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = handeye["camera"]
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam
    s_xy = np.asarray(screw_xy_m, dtype=np.float64).reshape(2)

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    valid_d = np.isfinite(depth_m) & (depth_m > 0.07) & (depth_m < 0.72)
    fg_mask = _foreground_mask_against_green_mat(color_bgr)

    px_all = np.column_stack([xs[valid_d], ys[valid_d]])
    pts_cam = deproject_pixels_to_cam(px_all, depth_m[valid_d], cam_cfg)
    pts_base = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
    fg_all = fg_mask[valid_d]

    dist_to_screw = np.linalg.norm(pts_base[:, :2] - s_xy, axis=1)
    bg_ring = (~fg_all) & (dist_to_screw >= 0.025) & (dist_to_screw <= 0.095) & (
        np.abs(pts_base[:, 2] - table_z_m) < 0.030
    )
    local_table_z = (
        float(np.median(pts_base[bg_ring, 2]))
        if int(np.count_nonzero(bg_ring)) >= 60
        else float(table_z_m)
    )

    # 1. Check that the initial nut pickup site is now cleared
    cleared_initial = True
    if initial_nut_xy_m is not None:
        init_xy = np.asarray(initial_nut_xy_m, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(init_xy - s_xy)) >= MIN_NUT_SCREW_SEPARATION_M:
            near_init = (
                fg_all
                & (np.linalg.norm(pts_base[:, :2] - init_xy, axis=1) <= 0.022)
                & (pts_base[:, 2] >= local_table_z + 0.004)
                & (pts_base[:, 2] <= local_table_z + 0.022)
            )
            if int(np.count_nonzero(near_init)) >= 120:
                cleared_initial = False

    # 2. Inspect assembled screw + nut at screw_xy_m
    assembly_mask = (
        fg_all
        & (dist_to_screw <= 0.022)
        & (pts_base[:, 2] >= local_table_z + 0.012)
        & (pts_base[:, 2] <= local_table_z + 0.056)
    )
    assembly_pts = pts_base[assembly_mask]
    rel_z = assembly_pts[:, 2] - local_table_z if len(assembly_pts) else np.zeros(0)

    # Points on the nut flange/hex body outside the 12 mm screw shaft (r in [6.8, 16.5] mm, z >= 13 mm)
    r_from_screw = np.linalg.norm(assembly_pts[:, :2] - s_xy, axis=1) if len(assembly_pts) else np.zeros(0)
    nut_ring_mask = (r_from_screw >= 0.0068) & (r_from_screw <= 0.0165) & (rel_z >= 0.013)
    nut_ring_pts = assembly_pts[nut_ring_mask]
    nut_ring_count = len(nut_ring_pts)

    if nut_ring_count >= 25:
        measured_xy = np.median(nut_ring_pts[:, :2], axis=0)
        nut_top_height_mm = float(np.percentile(nut_ring_pts[:, 2] - local_table_z, 95.0) * 1000.0)
    elif len(assembly_pts) >= 30:
        measured_xy = np.median(assembly_pts[:, :2], axis=0)
        nut_top_height_mm = float(np.percentile(rel_z, 80.0) * 1000.0)
    else:
        measured_xy = s_xy.copy()
        nut_top_height_mm = 0.0

    assembly_top_height_mm = (
        float(np.percentile(rel_z, 98.0) * 1000.0) if len(assembly_pts) >= 20 else 0.0
    )
    xy_error_mm = float(np.linalg.norm(measured_xy - s_xy) * 1000.0)

    expected_nut_top_mm = float(expected_nut_bottom_z_mm) + NUT_HEIGHT_MM
    reasons: list[str] = []
    if nut_ring_count < 40:
        reasons.append(f"固定螺丝处未检测到螺母法兰/六角环点云 ({nut_ring_count} < 40)")
    if xy_error_mm > 15.0:
        reasons.append(f"螺母与螺丝水平同心偏差过大 ({xy_error_mm:.1f} mm > 15.0 mm)")
    if nut_ring_count >= 40 and abs(nut_top_height_mm - expected_nut_top_mm) > 9.0:
        reasons.append(
            f"螺母顶面高度异常 ({nut_top_height_mm:.1f} mm，期望 ~{expected_nut_top_mm:.1f} mm)"
        )
    if not cleared_initial:
        reasons.append("初始抓取位置仍有螺母残留点云，疑似漏抓")

    ok = len(reasons) == 0
    ver = ThreadedNutVerification(
        ok=ok,
        screw_xy_m=(float(s_xy[0]), float(s_xy[1])),
        measured_xy_m=(float(measured_xy[0]), float(measured_xy[1])),
        xy_error_mm=round(xy_error_mm, 2),
        nut_top_height_mm=round(nut_top_height_mm, 2),
        assembly_top_height_mm=round(assembly_top_height_mm, 2),
        nut_ring_points=nut_ring_count,
        cleared_initial_nut_site=cleared_initial,
        reason="ok" if ok else "；".join(reasons),
    )
    return ver, {
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
        "assembly_points_base": assembly_pts,
    }


def save_detection_overlay(
    color_bgr: np.ndarray,
    nut_est: NutPoseEstimate | None,
    screw_est: ScrewPoseEstimate | None,
    debug_info: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render detected nut and screw poses onto the camera BGR image."""
    canvas = color_bgr.copy()
    t_base_cam = debug_info.get("t_base_cam")
    cam_cfg = debug_info.get("camera_cfg")
    if t_base_cam is not None and cam_cfg is not None:
        t_cam_base = np.linalg.inv(t_base_cam)
        if nut_est is not None:
            nut_3d = np.array(
                [[nut_est.xy_m[0], nut_est.xy_m[1], nut_est.table_z_m + 0.009]],
                dtype=np.float64,
            )
            nut_uv = project_cam_to_pixels(
                nut_3d @ t_cam_base[:3, :3].T + t_cam_base[:3, 3], cam_cfg
            ).astype(np.int32)[0]
            cv2.circle(canvas, tuple(nut_uv), 22, (0, 230, 80), 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, tuple(nut_uv), (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"Nut ({nut_est.xy_m[0]*1000:.1f},{nut_est.xy_m[1]*1000:.1f})mm yaw={nut_est.yaw_deg:.1f}deg",
                (int(nut_uv[0]) - 80, int(nut_uv[1]) - 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 100),
                2,
                cv2.LINE_AA,
            )
        if screw_est is not None:
            screw_3d = np.array(
                [[screw_est.xy_m[0], screw_est.xy_m[1], screw_est.table_z_m + 0.035]],
                dtype=np.float64,
            )
            screw_uv = project_cam_to_pixels(
                screw_3d @ t_cam_base[:3, :3].T + t_cam_base[:3, 3], cam_cfg
            ).astype(np.int32)[0]
            cv2.circle(canvas, tuple(screw_uv), 20, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                canvas, tuple(screw_uv), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                f"Screw ({screw_est.xy_m[0]*1000:.1f},{screw_est.xy_m[1]*1000:.1f})mm h={screw_est.height_mm:.1f}mm",
                (int(screw_uv[0]) - 80, int(screw_uv[1]) + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def save_post_thread_overlay(
    color_bgr: np.ndarray,
    verification: ThreadedNutVerification,
    debug_info: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render post-threading closed-loop verification result onto the camera BGR image."""
    canvas = color_bgr.copy()
    t_base_cam = debug_info.get("t_base_cam")
    cam_cfg = debug_info.get("camera_cfg")
    if t_base_cam is not None and cam_cfg is not None:
        t_cam_base = np.linalg.inv(t_base_cam)
        pts_3d = np.array(
            [
                [verification.screw_xy_m[0], verification.screw_xy_m[1], 0.025],
                [verification.measured_xy_m[0], verification.measured_xy_m[1], 0.025],
            ],
            dtype=np.float64,
        )
        uv = project_cam_to_pixels(
            pts_3d @ t_cam_base[:3, :3].T + t_cam_base[:3, 3], cam_cfg
        ).astype(np.int32)
        p_tgt, p_meas = tuple(uv[0]), tuple(uv[1])
        cv2.drawMarker(canvas, p_tgt, (255, 200, 0), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(
            canvas, p_meas, 20, (0, 255, 0) if verification.ok else (0, 0, 255), 2, cv2.LINE_AA
        )

    status_str = "PASS" if verification.ok else f"FAIL ({verification.reason})"
    summary = (
        f"Post-Thread: {status_str} | err={verification.xy_error_mm:.1f}mm "
        f"nut_top={verification.nut_top_height_mm:.1f}mm pts={verification.nut_ring_points}"
    )
    cv2.rectangle(canvas, (12, 12), (1180, 56), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        summary,
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (120, 255, 140) if verification.ok else (100, 120, 255),
        2,
        cv2.LINE_AA,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out

