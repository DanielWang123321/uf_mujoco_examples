# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Shared sideways-cup component geometry and near-view detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.objects.paper_cup import (
    CUP_HALF_CONE_RAD,
    CUP_HEIGHT_M,
    GRASP_AXIS_OFFSET_M,
    CupPoseEstimate,
    sideways_cup_mujoco_pose,
)
from common.paths import DEFAULT_HANDEYE_YAML
from common.vision import (
    WORKSPACE_BOUNDS_M,
    PerceptionError,
    component_touches_fov_boundary,
    deproject_pixels_to_cam,
    load_handeye_calibration,
    resolve_camera_config,
    validate_xy_in_workspace,
    white_cup_mask,
)

# Tangent-plane height z = R_mid + R_mid^2 / (H - R_mid) (~35 mm) for a sideways 53->75 mm paper cup.
SIDEWAYS_CUP_SILHOUETTE_Z_M = 0.035


def project_component_silhouette_xy(
    comp_mask: np.ndarray,
    cam_cfg: dict[str, Any],
    t_base_cam: np.ndarray,
    local_table_z: float,
    *,
    exclude_xy_m: np.ndarray | None = None,
    exclude_radius_m: float = 0.08,
    expected_xy_m: np.ndarray | None = None,
    max_radius_m: float | None = None,
) -> np.ndarray | None:
    """Project a hole-filled 2D cup mask onto the horizontal cup-equator tangent plane in base XY."""
    comp_u8 = np.asarray(comp_mask, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled_u8 = np.zeros_like(comp_u8)
    cv2.drawContours(filled_u8, contours, -1, 255, thickness=cv2.FILLED)
    ys, xs = np.nonzero(filled_u8)
    if len(xs) < 250:
        return None

    px = np.column_stack([xs, ys])
    rays_cam = deproject_pixels_to_cam(px, np.ones(len(xs), dtype=np.float64), cam_cfg)
    dirs_base = rays_cam @ t_base_cam[:3, :3].T
    valid_dir = dirs_base[:, 2] < -1e-4
    if int(np.count_nonzero(valid_dir)) < 250:
        return None
    dirs_base = dirs_base[valid_dir]

    cam_orig = t_base_cam[:3, 3]
    z_plane = float(local_table_z) + SIDEWAYS_CUP_SILHOUETTE_Z_M
    t_hit = (z_plane - float(cam_orig[2])) / dirs_base[:, 2]
    valid_t = t_hit > 0.05
    if int(np.count_nonzero(valid_t)) < 250:
        return None

    xy = cam_orig[None, :2] + t_hit[valid_t, None] * dirs_base[valid_t, :2]
    if exclude_xy_m is not None:
        xy = xy[np.linalg.norm(xy - exclude_xy_m, axis=1) > float(exclude_radius_m)]
    if expected_xy_m is not None and max_radius_m is not None:
        xy = xy[np.linalg.norm(xy - expected_xy_m, axis=1) <= float(max_radius_m)]
    return xy if len(xy) >= 250 else None


def _fit_sideways_cup_xy(
    xy: np.ndarray,
    *,
    percentile_clip: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    """Fit primary axis, bottom->rim orientation, and symmetry center from planar XY samples."""
    if len(xy) < 60:
        return None
    center_xy = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - center_xy, full_matrices=False)
    axis_xy = vt[0].copy()
    ortho_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float64)

    proj_along = (xy - center_xy) @ axis_xy
    proj_ortho = (xy - center_xy) @ ortho_xy
    p_min, p_max = np.percentile(proj_along, [percentile_clip, 100.0 - percentile_clip])
    o_min, o_max = np.percentile(proj_ortho, [percentile_clip, 100.0 - percentile_clip])
    span = float(p_max - p_min)
    if span <= 1e-4:
        return None
    length_mm = span * 1000.0

    # Measure orthogonal widths in spatial end bands [5%..35%] from each end so hollow-cavity
    # depth dropouts or interior points at the open mouth cannot skew point-count percentiles.
    mask_neg = (proj_along >= p_min + 0.05 * span) & (proj_along <= p_min + 0.35 * span)
    mask_pos = (proj_along <= p_max - 0.05 * span) & (proj_along >= p_max - 0.35 * span)
    if int(np.count_nonzero(mask_neg)) < 25 or int(np.count_nonzero(mask_pos)) < 25:
        mask_neg = proj_along <= p_min + 0.35 * span
        mask_pos = proj_along >= p_max - 0.35 * span
    if int(np.count_nonzero(mask_neg)) < 20 or int(np.count_nonzero(mask_pos)) < 20:
        return None

    width_neg = float(
        (np.percentile(proj_ortho[mask_neg], 98.0) - np.percentile(proj_ortho[mask_neg], 2.0))
        * 1000.0
    )
    width_pos = float(
        (np.percentile(proj_ortho[mask_pos], 98.0) - np.percentile(proj_ortho[mask_pos], 2.0))
        * 1000.0
    )

    # Disambiguate closed bottom (53 mm) vs open rim (75 mm) strictly by conical taper width;
    # hollow open-mouth views observe the low interior wall in Z, so Z is non-monotonic.
    if width_pos < width_neg:
        axis_xy = -axis_xy
        ortho_xy = -ortho_xy
        p_min, p_max = -p_max, -p_min
        o_min, o_max = -o_max, -o_min
        width_bot_mm, width_top_mm = width_pos, width_neg
    else:
        width_bot_mm, width_top_mm = width_neg, width_pos

    mid_xy = (
        center_xy + 0.5 * float(p_min + p_max) * axis_xy + 0.5 * float(o_min + o_max) * ortho_xy
    )
    return mid_xy, axis_xy, length_mm, width_bot_mm, width_top_mm


def _valid_cup_dimensions(length_mm: float, width_bot_mm: float, width_top_mm: float) -> bool:
    return (
        42.0 <= length_mm <= 115.0 and 32.0 <= width_bot_mm <= 78.0 and 42.0 <= width_top_mm <= 95.0
    )


def estimate_sideways_cup_component(
    elevated_pts_base: np.ndarray,
    local_table_z: float,
    workspace_bounds: dict[str, float],
    *,
    silhouette_xy: np.ndarray | None = None,
    strict_dimensions: bool = True,
    raise_on_workspace: bool = False,
) -> CupPoseEstimate | None:
    """Estimate a sideways paper cup 6D pose from 2D silhouette + elevated 3D points."""
    del local_table_z
    fit: tuple[np.ndarray, np.ndarray, float, float, float] | None = None
    med_3d_xy = np.median(elevated_pts_base[:, :2], axis=0)

    if silhouette_xy is not None:
        cand = _fit_sideways_cup_xy(silhouette_xy, percentile_clip=1.0)
        if (
            cand is not None
            and _valid_cup_dimensions(cand[2], cand[3], cand[4])
            and float(np.linalg.norm(cand[0] - med_3d_xy)) <= 0.035
        ):
            fit = cand

    if fit is None:
        cand_3d = _fit_sideways_cup_xy(elevated_pts_base[:, :2], percentile_clip=2.0)
        if cand_3d is None:
            return None
        if strict_dimensions and not _valid_cup_dimensions(cand_3d[2], cand_3d[3], cand_3d[4]):
            return None
        fit = cand_3d

    mid_xy, axis_xy, length_mm, width_bot_mm, width_top_mm = fit
    half_horiz_len = 0.5 * CUP_HEIGHT_M * float(np.cos(CUP_HALF_CONE_RAD))
    bot_xy = mid_xy - half_horiz_len * axis_xy
    top_xy = mid_xy + half_horiz_len * axis_xy
    grasp_xy = bot_xy + GRASP_AXIS_OFFSET_M * axis_xy

    if raise_on_workspace:
        validate_xy_in_workspace(grasp_xy, "Detected cup grasp center", workspace_bounds)
    else:
        try:
            validate_xy_in_workspace(grasp_xy, "Detected cup grasp center", workspace_bounds)
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
    camera_metadata: dict[str, Any] | None = None,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[CupPoseEstimate | None, dict[str, Any]]:
    """Detect and refine a single sideways paper cup near expected_xy_m in vertical look-down view."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = resolve_camera_config(handeye, color_bgr, depth_m, camera_metadata)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam
    exp_xy = np.asarray(expected_xy_m, dtype=np.float64).reshape(2)
    excl_xy = (
        np.asarray(exclude_xy_m, dtype=np.float64).reshape(2) if exclude_xy_m is not None else None
    )

    if color_bgr is None or depth_m is None or color_bgr.shape[:2] != depth_m.shape[:2]:
        raise PerceptionError("RGB and depth images are invalid or have different sizes")

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    fov_mask = (
        np.ones((h, w), dtype=bool)
        if cam_cfg.get("type") == "realsense"
        else (
            (ys >= int(0.06 * h))
            & (ys < int(0.88 * h))
            & (xs >= int(0.06 * w))
            & (xs < int(0.94 * w))
        )
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
    white_mask = white_cup_mask(hsv, cam_cfg, saturation=80, value=150) & fov_mask
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
        if component_touches_fov_boundary(labels == label_idx, fov_mask):
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

        silhouette_xy = project_component_silhouette_xy(
            labels == label_idx,
            cam_cfg,
            t_base_cam,
            local_table_z,
            exclude_xy_m=excl_xy,
            exclude_radius_m=float(exclude_radius_m),
            expected_xy_m=exp_xy,
            max_radius_m=float(search_radius_m) + 0.055,
        )
        est = estimate_sideways_cup_component(
            elevated,
            local_table_z,
            workspace_bounds,
            silhouette_xy=silhouette_xy,
        )
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
