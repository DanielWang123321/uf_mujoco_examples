"""Eye-in-Hand RGB-D multi-cup 6D pose estimation and closed-loop stack verification."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

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

# Cup physical dimensions from AGENTS.md
CUP_HEIGHT_M = 0.088
CUP_BOT_DIAM_M = 0.053
CUP_TOP_DIAM_M = 0.075
CUP_HALF_CONE_RAD = float(np.arctan2(0.5 * (CUP_TOP_DIAM_M - CUP_BOT_DIAM_M), CUP_HEIGHT_M))
CUP_SPAWN_Z_OFFSET_M = 0.0375
MIN_CUP_SEPARATION_M = 0.075


@dataclass(frozen=True)
class CupPoseEstimate:
    """Estimated 3D pose of a sideways paper cup in the robot base frame."""

    cup_center_xy_m: tuple[float, float]
    cup_bottom_xy_m: tuple[float, float]
    cup_top_xy_m: tuple[float, float]
    table_z_m: float
    yaw_deg: float
    mujoco_pos_m: tuple[float, float, float]
    mujoco_quat_wxyz: tuple[float, float, float, float]
    length_mm: float
    width_bot_mm: float
    width_top_mm: float
    point_count: int
    source: str = "camera"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "cup_center_xy_m": list(self.cup_center_xy_m),
            "cup_bottom_xy_m": list(self.cup_bottom_xy_m),
            "cup_top_xy_m": list(self.cup_top_xy_m),
            "table_z_m": float(self.table_z_m),
            "yaw_deg": float(self.yaw_deg),
            "mujoco_pos_m": list(self.mujoco_pos_m),
            "mujoco_quat_wxyz": list(self.mujoco_quat_wxyz),
            "length_mm": float(self.length_mm),
            "width_bot_mm": float(self.width_bot_mm),
            "width_top_mm": float(self.width_top_mm),
            "point_count": int(self.point_count),
        }


@dataclass(frozen=True)
class StackedCupsVerification:
    """Closed-loop visual verification result after stacking multiple paper cups upright."""

    ok: bool
    num_cups: int
    target_xy_m: tuple[float, float]
    measured_xy_m: tuple[float, float]
    xy_error_mm: float
    rim_height_mm: float
    point_count: int
    roi_cup_pixels: int
    cleared_initial_sites: bool = True
    reason: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "num_cups": int(self.num_cups),
            "target_xy_m": list(self.target_xy_m),
            "measured_xy_m": list(self.measured_xy_m),
            "xy_error_mm": float(self.xy_error_mm),
            "rim_height_mm": float(self.rim_height_mm),
            "point_count": int(self.point_count),
            "roi_cup_pixels": int(self.roi_cup_pixels),
            "cleared_initial_sites": bool(self.cleared_initial_sites),
            "reason": self.reason,
        }


def sideways_cup_mujoco_pose(
    bottom_xy_m: tuple[float, float] | np.ndarray,
    yaw_deg: float,
    table_z_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MuJoCo freejoint (pos_m, quat_wxyz) for a sideways paper cup."""
    bot = np.asarray(bottom_xy_m, dtype=np.float64).reshape(2)
    pos_m = np.array([bot[0], bot[1], float(table_z_m) + CUP_SPAWN_Z_OFFSET_M], dtype=np.float64)
    r_sideways = Rotation.from_euler("z", float(yaw_deg), degrees=True) * Rotation.from_euler(
        "y", 90.0, degrees=True
    )
    q_xyzw = r_sideways.as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return pos_m, quat_wxyz


__all__ = [
    "CAMERA_TOOL_SRC",
    "CUP_BOT_DIAM_M",
    "CUP_HALF_CONE_RAD",
    "CUP_HEIGHT_M",
    "CUP_SPAWN_Z_OFFSET_M",
    "CUP_TOP_DIAM_M",
    "DEFAULT_HANDEYE_YAML",
    "MIN_CUP_SEPARATION_M",
    "PY312_PYTHON",
    "WORKSPACE_BOUNDS_M",
    "CupPoseEstimate",
    "OrbbecCameraStream",
    "PerceptionError",
    "StackedCupsVerification",
    "capture_rgbd_frame",
    "deproject_pixels_to_cam",
    "detect_sideways_cups",
    "detect_single_cup_near",
    "load_handeye_calibration",
    "pose_estimate_from_manual",
    "pose_estimates_from_manual",
    "project_cam_to_pixels",
    "save_multi_detection_overlay",
    "save_post_stack_overlay",
    "sideways_cup_mujoco_pose",
    "validate_xy_in_workspace",
    "verify_stacked_cups",
]


def pose_estimate_from_manual(
    bottom_x_m: float,
    bottom_y_m: float,
    yaw_deg: float,
    *,
    table_z_m: float = 0.0,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> CupPoseEstimate:
    """Build a validated CupPoseEstimate from manually supplied (bottom_x, bottom_y, yaw_deg)."""
    bot_xy = np.array([float(bottom_x_m), float(bottom_y_m)], dtype=np.float64)
    yaw_rad = float(np.deg2rad(yaw_deg))
    axis_xy = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=np.float64)
    horiz_len = CUP_HEIGHT_M * float(np.cos(CUP_HALF_CONE_RAD))
    mid_xy = bot_xy + 0.5 * horiz_len * axis_xy
    top_xy = bot_xy + horiz_len * axis_xy
    grasp_xy = bot_xy + 0.040 * axis_xy
    validate_xy_in_workspace(grasp_xy, "纸杯抓取中心", bounds)
    pos_m, quat_wxyz = sideways_cup_mujoco_pose(bot_xy, yaw_deg, table_z_m=table_z_m)
    return CupPoseEstimate(
        cup_center_xy_m=(float(mid_xy[0]), float(mid_xy[1])),
        cup_bottom_xy_m=(float(bot_xy[0]), float(bot_xy[1])),
        cup_top_xy_m=(float(top_xy[0]), float(top_xy[1])),
        table_z_m=float(table_z_m),
        yaw_deg=float(yaw_deg),
        mujoco_pos_m=(float(pos_m[0]), float(pos_m[1]), float(pos_m[2])),
        mujoco_quat_wxyz=(
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        ),
        length_mm=CUP_HEIGHT_M * 1000.0,
        width_bot_mm=CUP_BOT_DIAM_M * 1000.0,
        width_top_mm=CUP_TOP_DIAM_M * 1000.0,
        point_count=0,
        source="manual",
    )


def pose_estimates_from_manual(
    flat_values: list[float] | tuple[float, ...],
    *,
    num_cups: int = 2,
    table_z_m: float = 0.0,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> list[CupPoseEstimate]:
    """Build and validate multiple CupPoseEstimate objects from flat [X1, Y1, YAW1, ...] values."""
    vals = [float(v) for v in flat_values]
    if len(vals) % 3 != 0 or len(vals) // 3 < num_cups:
        raise PerceptionError(
            f"--cup-poses 参数数量必须是 3 的倍数且不少于 {num_cups} 个纸杯 "
            f"(共需 {3 * num_cups} 个数值)"
        )
    estimates: list[CupPoseEstimate] = []
    for idx in range(num_cups):
        est = pose_estimate_from_manual(
            vals[3 * idx],
            vals[3 * idx + 1],
            vals[3 * idx + 2],
            table_z_m=table_z_m,
            bounds=bounds,
        )
        for prev_idx, prev in enumerate(estimates):
            dist = float(
                np.linalg.norm(
                    np.asarray(est.cup_center_xy_m) - np.asarray(prev.cup_center_xy_m)
                )
            )
            if dist < MIN_CUP_SEPARATION_M:
                raise PerceptionError(
                    f"纸杯 #{idx + 1} 与纸杯 #{prev_idx + 1} 中心间距过近 "
                    f"({dist * 1000:.1f} mm < {MIN_CUP_SEPARATION_M * 1000:.0f} mm)"
                )
        estimates.append(est)
    return estimates


def _estimate_single_component_pose(
    elevated_pts_base: np.ndarray,
    local_table_z: float,
    workspace_bounds: dict[str, float],
) -> CupPoseEstimate | None:
    """Estimate a sideways paper cup 6D pose from elevated 3D points, or return None if invalid."""
    xy = elevated_pts_base[:, :2]
    center_xy = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - center_xy, full_matrices=False)
    axis_xy = vt[0].copy()
    ortho_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float64)

    proj_along = (xy - center_xy) @ axis_xy
    proj_ortho = (xy - center_xy) @ ortho_xy
    p_min, p_max = np.percentile(proj_along, [2.0, 98.0])
    length_mm = float((p_max - p_min) * 1000.0)
    if not (42.0 <= length_mm <= 115.0):
        return None

    q35, q65 = np.percentile(proj_along, [35.0, 65.0])
    mask_neg = proj_along <= q35
    mask_pos = proj_along >= q65
    if int(np.count_nonzero(mask_neg)) < 30 or int(np.count_nonzero(mask_pos)) < 30:
        return None

    width_neg = float(
        (np.percentile(proj_ortho[mask_neg], 95.0) - np.percentile(proj_ortho[mask_neg], 5.0))
        * 1000.0
    )
    width_pos = float(
        (np.percentile(proj_ortho[mask_pos], 95.0) - np.percentile(proj_ortho[mask_pos], 5.0))
        * 1000.0
    )
    z_neg = float(np.percentile(elevated_pts_base[mask_neg, 2] - local_table_z, 90.0) * 1000.0)
    z_pos = float(np.percentile(elevated_pts_base[mask_pos, 2] - local_table_z, 90.0) * 1000.0)

    # The top rim (75 mm) is wider and higher above the table than the bottom (53 mm)
    score_pos_is_top = (width_pos - width_neg) + 0.5 * (z_pos - z_neg)
    if score_pos_is_top < 0.0:
        axis_xy = -axis_xy
        ortho_xy = -ortho_xy
        proj_along = -proj_along
        p_min, p_max = -p_max, -p_min
        width_bot_mm, width_top_mm = width_pos, width_neg
    else:
        width_bot_mm, width_top_mm = width_neg, width_pos

    if not (32.0 <= width_bot_mm <= 78.0 and 42.0 <= width_top_mm <= 95.0):
        return None

    mid_along = 0.5 * (p_min + p_max)
    mid_xy = center_xy + mid_along * axis_xy
    half_horiz_len = 0.5 * CUP_HEIGHT_M * float(np.cos(CUP_HALF_CONE_RAD))
    bot_xy = mid_xy - half_horiz_len * axis_xy
    top_xy = mid_xy + half_horiz_len * axis_xy
    grasp_xy = bot_xy + 0.040 * axis_xy

    try:
        validate_xy_in_workspace(grasp_xy, "感知到的纸杯抓取中心", workspace_bounds)
    except PerceptionError:
        return None

    yaw_deg = float(np.rad2deg(np.arctan2(axis_xy[1], axis_xy[0])))
    pos_m, quat_wxyz = sideways_cup_mujoco_pose(bot_xy, yaw_deg, table_z_m=0.0)
    return CupPoseEstimate(
        cup_center_xy_m=(float(mid_xy[0]), float(mid_xy[1])),
        cup_bottom_xy_m=(float(bot_xy[0]), float(bot_xy[1])),
        cup_top_xy_m=(float(top_xy[0]), float(top_xy[1])),
        table_z_m=0.0,
        yaw_deg=yaw_deg,
        mujoco_pos_m=(float(pos_m[0]), float(pos_m[1]), float(pos_m[2])),
        mujoco_quat_wxyz=(
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        ),
        length_mm=length_mm,
        width_bot_mm=width_bot_mm,
        width_top_mm=width_top_mm,
        point_count=len(elevated_pts_base),
        source="camera",
    )


def detect_sideways_cups(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    *,
    num_cups: int = 2,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[list[CupPoseEstimate], dict[str, Any]]:
    """Segment multiple white sideways paper cups and estimate their 6D poses in base frame."""
    if num_cups < 2:
        raise ValueError(f"多纸杯检测要求 num_cups >= 2，当前: {num_cups}")

    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = handeye["camera"]
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam

    if color_bgr is None or depth_m is None or color_bgr.shape[:2] != depth_m.shape[:2]:
        raise PerceptionError("输入的 RGB 与深度图尺寸无效或不匹配")

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    fov_mask = (
        (ys >= int(0.18 * h))
        & (ys < int(0.73 * h))
        & (xs >= int(0.18 * w))
        & (xs < int(0.92 * w))
    )
    depth_valid = np.isfinite(depth_m) & (depth_m > 0.07) & (depth_m < 0.75) & fov_mask
    if int(np.count_nonzero(depth_valid)) < 500:
        raise PerceptionError("有效深度像素不足，无法定位桌面或纸杯")

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 155) & fov_mask
    white_u8 = cv2.morphologyEx(
        white_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_u8)
    candidates: list[tuple[CupPoseEstimate, int, list[int]]] = []

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 600:
            continue
        comp_mask = (labels == label_idx) & depth_valid
        if int(np.count_nonzero(comp_mask)) < 250:
            continue

        px = np.column_stack([xs[comp_mask], ys[comp_mask]])
        pts_cam = deproject_pixels_to_cam(px, depth_m[comp_mask], cam_cfg)
        pts_base = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]

        med_xy = np.median(pts_base[:, :2], axis=0)
        if not (
            workspace_bounds["x_min"] <= med_xy[0] <= workspace_bounds["x_max"]
            and workspace_bounds["y_min"] <= med_xy[1] <= workspace_bounds["y_max"]
        ):
            continue

        comp_u8 = (labels == label_idx).astype(np.uint8) * 255
        outer_ring = cv2.dilate(comp_u8, np.ones((65, 65), dtype=np.uint8)) > 0
        inner_ring = cv2.dilate(comp_u8, np.ones((15, 15), dtype=np.uint8)) > 0
        ring_mask = outer_ring & (~inner_ring) & depth_valid
        if int(np.count_nonzero(ring_mask)) >= 100:
            ring_px = np.column_stack([xs[ring_mask], ys[ring_mask]])
            ring_cam = deproject_pixels_to_cam(ring_px, depth_m[ring_mask], cam_cfg)
            ring_base = ring_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
            local_table_z = float(np.median(ring_base[:, 2]))
        else:
            local_table_z = 0.0

        if abs(local_table_z) > 0.030:
            continue

        elevated = pts_base[
            (pts_base[:, 2] > local_table_z + 0.012) & (pts_base[:, 2] < local_table_z + 0.090)
        ]
        if len(elevated) < 400:
            continue

        est = _estimate_single_component_pose(elevated, local_table_z, workspace_bounds)
        if est is not None:
            candidates.append((est, label_idx, stats[label_idx, :4].tolist()))

    # Sort by point count descending and deduplicate nearby components
    candidates.sort(key=lambda item: item[0].point_count, reverse=True)
    selected: list[tuple[CupPoseEstimate, int, list[int]]] = []
    for cand in candidates:
        c_xy = np.asarray(cand[0].cup_center_xy_m, dtype=np.float64)
        if any(
            float(np.linalg.norm(c_xy - np.asarray(prev[0].cup_center_xy_m, dtype=np.float64)))
            < MIN_CUP_SEPARATION_M
            for prev in selected
        ):
            continue
        selected.append(cand)
        if len(selected) == num_cups:
            break

    if len(selected) < num_cups:
        raise PerceptionError(
            f"在安全工作区内仅检测到 {len(selected)} 个有效白色一次性纸杯，少于所需的 {num_cups} 个"
        )

    # Order from right (-Y) to left (+Y) so Cup #1 is placed as the base cup first
    selected.sort(key=lambda item: (item[0].cup_bottom_xy_m[1], -item[0].cup_bottom_xy_m[0]))
    estimates = [item[0] for item in selected]
    masks = [labels == item[1] for item in selected]
    bboxes = [item[2] for item in selected]

    debug_info = {
        "masks": masks,
        "bboxes": bboxes,
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
    }
    return estimates, debug_info


def detect_single_cup_near(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    expected_xy_m: tuple[float, float] | np.ndarray,
    *,
    search_radius_m: float = 0.12,
    exclude_xy_m: tuple[float, float] | np.ndarray | None = None,
    exclude_radius_m: float = 0.08,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[CupPoseEstimate | None, dict[str, Any]]:
    """Detect and refine a single sideways paper cup near expected_xy_m in vertical look-down view."""
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
        & (ys < int(0.73 * h))
        & (xs >= int(0.06 * w))
        & (xs < int(0.94 * w))
    )
    depth_valid = np.isfinite(depth_m) & (depth_m > 0.07) & (depth_m < 0.75) & fov_mask
    if int(np.count_nonzero(depth_valid)) < 300:
        return None, {
            "masks": [],
            "bboxes": [],
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "insufficient_valid_depth",
        }

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 150) & fov_mask
    white_u8 = cv2.morphologyEx(
        white_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_u8)
    best_cand: tuple[float, CupPoseEstimate, int, list[int]] | None = None

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 600:
            continue
        comp_mask = (labels == label_idx) & depth_valid
        if int(np.count_nonzero(comp_mask)) < 250:
            continue

        px = np.column_stack([xs[comp_mask], ys[comp_mask]])
        pts_cam = deproject_pixels_to_cam(px, depth_m[comp_mask], cam_cfg)
        pts_base = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]

        if excl_xy is not None:
            keep = np.linalg.norm(pts_base[:, :2] - excl_xy, axis=1) > float(exclude_radius_m)
            pts_base = pts_base[keep]
            if len(pts_base) < 250:
                continue

        near_keep = np.linalg.norm(pts_base[:, :2] - exp_xy, axis=1) <= (
            float(search_radius_m) + 0.055
        )
        pts_base = pts_base[near_keep]
        if len(pts_base) < 250:
            continue

        comp_u8 = (labels == label_idx).astype(np.uint8) * 255
        outer_ring = cv2.dilate(comp_u8, np.ones((65, 65), dtype=np.uint8)) > 0
        inner_ring = cv2.dilate(comp_u8, np.ones((15, 15), dtype=np.uint8)) > 0
        ring_mask = outer_ring & (~inner_ring) & depth_valid
        local_table_z = 0.0
        if int(np.count_nonzero(ring_mask)) >= 100:
            ring_px = np.column_stack([xs[ring_mask], ys[ring_mask]])
            ring_cam = deproject_pixels_to_cam(ring_px, depth_m[ring_mask], cam_cfg)
            ring_base = ring_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
            cand_z = float(np.median(ring_base[:, 2]))
            if abs(cand_z) <= 0.030:
                local_table_z = cand_z

        elevated = pts_base[
            (pts_base[:, 2] > local_table_z + 0.012) & (pts_base[:, 2] < local_table_z + 0.090)
        ]
        if len(elevated) < 350:
            continue

        est = _estimate_single_component_pose(elevated, local_table_z, workspace_bounds)
        if est is None:
            continue

        dist = float(np.linalg.norm(np.asarray(est.cup_center_xy_m) - exp_xy))
        if dist > float(search_radius_m):
            continue

        if best_cand is None or dist < best_cand[0]:
            best_cand = (dist, est, label_idx, stats[label_idx, :4].tolist())

    if best_cand is None:
        return None, {
            "masks": [],
            "bboxes": [],
            "t_base_cam": t_base_cam,
            "camera_cfg": cam_cfg,
            "reason": "cup_not_found_in_search_radius",
        }

    _, best_est, best_label, best_bbox = best_cand
    return best_est, {
        "masks": [labels == best_label],
        "bboxes": [best_bbox],
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
        "reason": "ok",
    }


def save_multi_detection_overlay(
    color_bgr: np.ndarray,
    estimates: list[CupPoseEstimate],
    debug_info: dict[str, Any],
    output_path: str | Path,
    *,
    target_xy_m: tuple[float, float] | np.ndarray | None = None,
) -> Path:
    """Render contours, bottom->top arrows, grasp centers, and labels for all detected cups."""
    canvas = color_bgr.copy()
    masks = debug_info.get("masks") or []
    palette = [(0, 230, 80), (255, 200, 40), (255, 110, 220)]
    for idx, mask in enumerate(masks):
        if mask is not None:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            color = palette[idx % len(palette)]
            cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)

    t_base_cam = debug_info.get("t_base_cam")
    cam_cfg = debug_info.get("camera_cfg")
    if t_base_cam is not None and cam_cfg is not None:
        t_cam_base = np.eye(4, dtype=np.float64)
        r_bc = t_base_cam[:3, :3]
        t_cam_base[:3, :3] = r_bc.T
        t_cam_base[:3, 3] = -r_bc.T @ t_base_cam[:3, 3]

        for idx, estimate in enumerate(estimates):
            z_mid = estimate.table_z_m + 0.032
            bot_3d = np.array([estimate.cup_bottom_xy_m[0], estimate.cup_bottom_xy_m[1], z_mid])
            top_3d = np.array([estimate.cup_top_xy_m[0], estimate.cup_top_xy_m[1], z_mid])
            grasp_3d = 0.545 * bot_3d + 0.455 * top_3d
            pts_base = np.vstack([bot_3d, top_3d, grasp_3d])
            pts_cam = pts_base @ t_cam_base[:3, :3].T + t_cam_base[:3, 3]
            uv = project_cam_to_pixels(pts_cam, cam_cfg).astype(np.int32)

            p_bot, p_top, p_grasp = tuple(uv[0]), tuple(uv[1]), tuple(uv[2])
            cv2.arrowedLine(canvas, p_bot, p_top, (255, 180, 0), 3, cv2.LINE_AA, tipLength=0.18)
            cv2.circle(canvas, p_bot, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, p_top, 6, (255, 120, 0), -1, cv2.LINE_AA)
            cv2.drawMarker(canvas, p_grasp, (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            role = "Base" if idx == 0 else f"Nest#{idx}"
            cv2.putText(
                canvas,
                f"Cup#{idx + 1}({role}) BOT",
                (p_bot[0] - 20, p_bot[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 100),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "TOP",
                (p_top[0] + 8, p_top[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

        if target_xy_m is not None:
            tgt = np.asarray(target_xy_m, dtype=np.float64).reshape(2)
            tgt_cam = (
                np.array([[tgt[0], tgt[1], 0.0]], dtype=np.float64) @ t_cam_base[:3, :3].T
                + t_cam_base[:3, 3]
            )
            tgt_uv = project_cam_to_pixels(tgt_cam, cam_cfg).astype(np.int32)[0]
            cv2.circle(canvas, tuple(tgt_uv), 24, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                canvas, tuple(tgt_uv), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                f"Stack Target ({tgt[0]*1000:.0f},{tgt[1]*1000:.0f})mm",
                (int(tgt_uv[0]) - 85, int(tgt_uv[1]) + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

    box_h = 24 + 28 * len(estimates)
    cv2.rectangle(canvas, (12, 12), (1140, 12 + box_h), (25, 25, 25), -1)
    for idx, est in enumerate(estimates):
        line = (
            f"Cup #{idx + 1}: bot=({est.cup_bottom_xy_m[0]*1000:.1f}, "
            f"{est.cup_bottom_xy_m[1]*1000:.1f})mm yaw={est.yaw_deg:.1f}deg "
            f"L={est.length_mm:.1f}mm W={est.width_bot_mm:.1f}->{est.width_top_mm:.1f}mm "
            f"pts={est.point_count}"
        )
        cv2.putText(
            canvas,
            line,
            (22, 38 + 26 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (240, 245, 255),
            2,
            cv2.LINE_AA,
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def verify_stacked_cups(
    color_bgr: np.ndarray,
    depth_m: np.ndarray | None,
    t_base_flange: np.ndarray,
    target_xy_m: tuple[float, float] | np.ndarray,
    *,
    num_cups: int = 2,
    initial_cups: list[CupPoseEstimate] | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    table_z_m: float = 0.0,
) -> tuple[StackedCupsVerification, dict[str, Any]]:
    """Verify via Eye-in-Hand RGB-D that multiple cups stand stacked at target_xy_m."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = handeye["camera"]
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam
    t_cam_base = np.eye(4, dtype=np.float64)
    r_bc = t_base_cam[:3, :3]
    t_cam_base[:3, :3] = r_bc.T
    t_cam_base[:3, 3] = -r_bc.T @ t_base_cam[:3, 3]

    target = np.asarray(target_xy_m, dtype=np.float64).reshape(2)
    h, w = color_bgr.shape[:2]

    # Build expected 2D image ROI by projecting the 3D bounding cylinder of the cup stack
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    r_search = 0.068
    expected_top_z = float(table_z_m) + CUP_HEIGHT_M + 0.025 * max(1, num_cups - 1) + 0.015
    ring_bot = np.column_stack(
        [
            target[0] + r_search * np.cos(angles),
            target[1] + r_search * np.sin(angles),
            np.full_like(angles, float(table_z_m)),
        ]
    )
    ring_top = np.column_stack(
        [
            target[0] + r_search * np.cos(angles),
            target[1] + r_search * np.sin(angles),
            np.full_like(angles, expected_top_z),
        ]
    )
    cyl_pts_cam = np.vstack([ring_bot, ring_top]) @ t_cam_base[:3, :3].T + t_cam_base[:3, 3]
    cyl_uv = project_cam_to_pixels(cyl_pts_cam, cam_cfg).astype(np.int32)
    hull = cv2.convexHull(cyl_uv)

    ys, xs = np.mgrid[0:h, 0:w]
    dome_mask = (
        ((xs - 0.50 * w) / (0.35 * w)) ** 2 + ((ys - 1.11 * h) / (0.27 * h)) ** 2
    ) <= 1.0
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, hull, 255)
    roi_bool = (roi_mask > 0) & (~dome_mask)
    roi_bool[int(0.95 * h) :, :] = False
    if depth_m is not None and depth_m.shape == (h, w):
        near_gripper = (depth_m > 0.04) & (depth_m < 0.22)
        near_gripper_dilated = (
            cv2.dilate(near_gripper.astype(np.uint8) * 255, np.ones((15, 15), dtype=np.uint8)) > 0
        )
        roi_bool &= ~near_gripper_dilated

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    cup_mask = (hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 145) & roi_bool
    cup_u8 = cv2.morphologyEx(
        cup_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    cup_mask = cup_u8 > 0
    roi_cup_pixels = int(np.count_nonzero(cup_mask))

    point_count = 0
    rim_height_mm = 0.0
    measured_xy = target.copy()

    if depth_m is not None and depth_m.shape == (h, w):
        valid_3d = cup_mask & np.isfinite(depth_m) & (depth_m >= 0.24) & (depth_m <= 0.75)
        if int(np.count_nonzero(valid_3d)) >= 50:
            px = np.column_stack([xs[valid_3d], ys[valid_3d]])
            pts_cam = deproject_pixels_to_cam(px, depth_m[valid_3d], cam_cfg)
            pts_base = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
            in_cyl = (
                (np.linalg.norm(pts_base[:, :2] - target, axis=1) <= 0.085)
                & (pts_base[:, 2] >= table_z_m + 0.015)
                & (pts_base[:, 2] <= expected_top_z + 0.025)
            )
            cup_pts = pts_base[in_cyl]
            point_count = len(cup_pts)
            if point_count >= 50:
                rim_height_mm = float(np.percentile(cup_pts[:, 2] - table_z_m, 95.0) * 1000.0)

    # Estimate horizontal stack center from 2D cup contour centroid intersected with z = table_z + 0.072m
    ref_z = float(table_z_m) + 0.072
    if roi_cup_pixels >= 600:
        u_med = float(np.median(xs[cup_mask]))
        v_med = float(np.median(ys[cup_mask]))
        ray_cam = deproject_pixels_to_cam(
            np.array([[u_med, v_med]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            cam_cfg,
        )[0]
        ray_dir_base = r_bc @ ray_cam
        cam_pos_base = t_base_cam[:3, 3]
        if abs(ray_dir_base[2]) > 1e-6:
            t_hit = (ref_z - cam_pos_base[2]) / ray_dir_base[2]
            if t_hit > 0:
                hit_pt = cam_pos_base + t_hit * ray_dir_base
                measured_xy = hit_pt[:2].copy()
                if point_count < 50:
                    rim_height_mm = max(rim_height_mm, 88.0 + 9.5 * max(1, num_cups - 1))

    # Optionally check that initial sideways pickup sites are now clear of white cups
    cleared_initial = True
    if initial_cups and depth_m is not None and depth_m.shape == (h, w):
        fov_valid = (
            (hsv[:, :, 1] < 75)
            & (hsv[:, :, 2] > 155)
            & np.isfinite(depth_m)
            & (depth_m >= 0.25)
            & (depth_m <= 0.75)
        )
        fov_valid[int(0.75 * h) :, :] = False
        if int(np.count_nonzero(fov_valid)) >= 100:
            px_all = np.column_stack([xs[fov_valid], ys[fov_valid]])
            pts_all = (
                deproject_pixels_to_cam(px_all, depth_m[fov_valid], cam_cfg) @ t_base_cam[:3, :3].T
                + t_base_cam[:3, 3]
            )
            for init_cup in initial_cups:
                init_mid = np.asarray(init_cup.cup_center_xy_m, dtype=np.float64)
                if float(np.linalg.norm(init_mid - target)) < 0.11:
                    continue
                near_init = (
                    (np.linalg.norm(pts_all[:, :2] - init_mid, axis=1) <= 0.040)
                    & (pts_all[:, 2] >= table_z_m + 0.020)
                    & (pts_all[:, 2] <= table_z_m + 0.085)
                )
                if int(np.count_nonzero(near_init)) > 2500:
                    cleared_initial = False

    xy_error_mm = float(np.linalg.norm(measured_xy - target) * 1000.0)
    reasons: list[str] = []
    if roi_cup_pixels < 1200:
        reasons.append(f"目标堆叠区纸杯像素不足 ({roi_cup_pixels} < 1200)")
    if xy_error_mm > 45.0:
        reasons.append(f"堆叠水平位置偏差过大 ({xy_error_mm:.1f} mm > 45.0 mm)")
    if point_count >= 150 and rim_height_mm < 50.0:
        reasons.append(f"堆叠顶层杯口高度偏低 ({rim_height_mm:.1f} mm < 50.0 mm)，疑似倾倒")
    if not cleared_initial:
        reasons.append("初始抓取位置仍有侧卧纸杯残留点云，疑似漏抓")

    ok = len(reasons) == 0
    verification = StackedCupsVerification(
        ok=ok,
        num_cups=num_cups,
        target_xy_m=(float(target[0]), float(target[1])),
        measured_xy_m=(float(measured_xy[0]), float(measured_xy[1])),
        xy_error_mm=xy_error_mm,
        rim_height_mm=rim_height_mm,
        point_count=point_count,
        roi_cup_pixels=roi_cup_pixels,
        cleared_initial_sites=cleared_initial,
        reason="ok" if ok else "；".join(reasons),
    )
    debug_info = {
        "cup_mask": cup_mask,
        "hull": hull,
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
    }
    return verification, debug_info


def save_post_stack_overlay(
    color_bgr: np.ndarray,
    verification: StackedCupsVerification,
    debug_info: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render post-stacking closed-loop verification result onto BGR image."""
    canvas = color_bgr.copy()
    cup_mask = debug_info.get("cup_mask")
    if cup_mask is not None:
        contours, _ = cv2.findContours(
            cup_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = (0, 230, 80) if verification.ok else (0, 60, 255)
        cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)

    hull = debug_info.get("hull")
    if hull is not None:
        cv2.polylines(canvas, [hull], True, (255, 200, 0), 1, cv2.LINE_AA)

    t_base_cam = debug_info.get("t_base_cam")
    cam_cfg = debug_info.get("camera_cfg")
    if t_base_cam is not None and cam_cfg is not None:
        t_cam_base = np.eye(4, dtype=np.float64)
        r_bc = t_base_cam[:3, :3]
        t_cam_base[:3, :3] = r_bc.T
        t_cam_base[:3, 3] = -r_bc.T @ t_base_cam[:3, 3]
        pts_3d = np.array(
            [
                [verification.target_xy_m[0], verification.target_xy_m[1], 0.072],
                [verification.measured_xy_m[0], verification.measured_xy_m[1], 0.072],
            ],
            dtype=np.float64,
        )
        pts_cam = pts_3d @ t_cam_base[:3, :3].T + t_cam_base[:3, 3]
        uv = project_cam_to_pixels(pts_cam, cam_cfg).astype(np.int32)
        p_tgt, p_meas = tuple(uv[0]), tuple(uv[1])
        cv2.drawMarker(canvas, p_tgt, (255, 200, 0), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(
            canvas, p_meas, 6, (0, 255, 0) if verification.ok else (0, 0, 255), -1, cv2.LINE_AA
        )

    status_str = "PASS" if verification.ok else f"FAIL ({verification.reason})"
    summary = (
        f"Post-Stack ({verification.num_cups} cups): {status_str} | "
        f"target=({verification.target_xy_m[0]*1000:.0f},{verification.target_xy_m[1]*1000:.0f})mm "
        f"err={verification.xy_error_mm:.1f}mm rim_z={verification.rim_height_mm:.1f}mm "
        f"px={verification.roi_cup_pixels}"
    )
    cv2.rectangle(canvas, (12, 12), (1220, 56), (25, 25, 25), -1)
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

