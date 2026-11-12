# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Eye-in-Hand RGB-D perception: sideways paper cup pose and upright-placement verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.objects import paper_cup
from common.objects.cup_perception import (
    detect_single_cup_near,
    estimate_sideways_cup_component,
    project_component_silhouette_xy,
)
from common.objects.paper_cup import (
    CUP_HEIGHT_M,
    CupPoseEstimate,
)
from common.paths import DEFAULT_HANDEYE_YAML
from common.vision import (
    PerceptionError,
    capture_rgbd_frame,
    component_touches_fov_boundary,
    cup_overview_fov,
    deproject_pixels_to_cam,
    load_handeye_calibration,
    project_cam_to_pixels,
    resolve_camera_config,
    validate_xy_in_workspace,
    white_cup_mask,
)

__all__ = [
    "WORKSPACE_BOUNDS_M",
    "CupPoseEstimate",
    "PerceptionError",
    "UprightCupVerification",
    "capture_rgbd_frame",
    "detect_sideways_cup",
    "detect_single_cup_near",
    "pose_estimate_from_manual",
    "save_detection_overlay",
    "save_post_place_overlay",
    "validate_xy_in_workspace",
    "verify_upright_cup",
]

# Clear workspace for this task (narrower than the shared default: excludes y > +0.12 m).
WORKSPACE_BOUNDS_M: dict[str, float] = {
    "x_min": 0.20,
    "x_max": 0.60,
    "y_min": -0.26,
    "y_max": 0.12,
}


@dataclass(frozen=True)
class UprightCupVerification:
    """Result of the camera check after placing the cup upright."""

    ok: bool
    target_xy_m: tuple[float, float]
    measured_xy_m: tuple[float, float]
    xy_error_mm: float
    rim_height_mm: float
    point_count: int
    roi_cup_pixels: int
    reason: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "target_xy_m": list(self.target_xy_m),
            "measured_xy_m": list(self.measured_xy_m),
            "xy_error_mm": float(self.xy_error_mm),
            "rim_height_mm": float(self.rim_height_mm),
            "point_count": int(self.point_count),
            "roi_cup_pixels": int(self.roi_cup_pixels),
            "reason": self.reason,
        }


def pose_estimate_from_manual(
    bottom_x_m: float,
    bottom_y_m: float,
    yaw_deg: float,
    *,
    table_z_m: float = 0.0,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> CupPoseEstimate:
    """Validated ``CupPoseEstimate`` from a manual (bottom_x, bottom_y, yaw_deg)."""
    return paper_cup.pose_estimate_from_manual(
        bottom_x_m, bottom_y_m, yaw_deg, bounds=bounds, table_z_m=table_z_m
    )


def detect_sideways_cup(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    *,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    camera_metadata: dict[str, Any] | None = None,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[CupPoseEstimate, dict[str, Any]]:
    """Segment the white sideways paper cup and estimate its 6D pose in the robot base frame."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = resolve_camera_config(handeye, color_bgr, depth_m, camera_metadata)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam

    if color_bgr is None or depth_m is None or color_bgr.shape[:2] != depth_m.shape[:2]:
        raise PerceptionError("RGB and depth images are invalid or have different sizes")

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    # Exclude bottom gripper housing/fingers visible in the lower FOV
    fov_mask = cup_overview_fov(h, w, cam_cfg)
    depth_valid = np.isfinite(depth_m) & (depth_m > 0.15) & (depth_m < 0.75) & fov_mask
    if int(np.count_nonzero(depth_valid)) < 500:
        raise PerceptionError("Not enough valid depth pixels to locate the table or cup")

    # White cup candidate mask in HSV
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white_mask = white_cup_mask(hsv, cam_cfg) & fov_mask
    white_u8 = cv2.morphologyEx(
        white_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_u8)
    best_label = -1
    best_pts_base: np.ndarray | None = None
    best_table_z = 0.0
    clipped_cup = False

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 600:
            continue
        if component_touches_fov_boundary(labels == label_idx, fov_mask):
            clipped_cup = True
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

        # Estimate local table Z around this component
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

        elevated = pts_base[
            (pts_base[:, 2] > local_table_z + 0.012) & (pts_base[:, 2] < local_table_z + 0.090)
        ]
        if len(elevated) < 200:
            continue
        if best_pts_base is None or len(elevated) > len(best_pts_base):
            best_label = label_idx
            best_pts_base = elevated
            best_table_z = local_table_z

    if best_pts_base is None or best_label < 0:
        if clipped_cup:
            raise PerceptionError(
                "Cup is clipped by the detection field of view; raise the observation pose or reposition the cup"
            )
        raise PerceptionError("No white paper cup detected inside the workspace")

    if abs(best_table_z) > 0.030:
        raise PerceptionError(
            f"Unexpected local table height ({best_table_z * 1000:.1f} mm); check the hand-eye calibration or camera pose"
        )

    silhouette_xy = project_component_silhouette_xy(
        labels == best_label,
        cam_cfg,
        t_base_cam,
        best_table_z,
    )
    estimate = estimate_sideways_cup_component(
        best_pts_base,
        best_table_z,
        workspace_bounds,
        silhouette_xy=silhouette_xy,
        strict_dimensions=False,
        raise_on_workspace=True,
    )
    if estimate is None:
        raise PerceptionError("No white paper cup detected inside the workspace")

    debug_info = {
        "mask": labels == best_label,
        "bbox": stats[best_label, :4].tolist(),
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
    }
    return estimate, debug_info


def save_detection_overlay(
    color_bgr: np.ndarray,
    estimate: CupPoseEstimate,
    debug_info: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render cup contour, bottom->top axis arrow, grasp center, and 3D metrics onto BGR image."""
    canvas = color_bgr.copy()
    mask = debug_info.get("mask")
    if mask is not None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (0, 220, 80), 2, cv2.LINE_AA)

    t_base_cam = debug_info.get("t_base_cam")
    cam_cfg = debug_info.get("camera_cfg")
    if t_base_cam is not None and cam_cfg is not None:
        t_cam_base = np.eye(4, dtype=np.float64)
        r_bc = t_base_cam[:3, :3]
        t_cam_base[:3, :3] = r_bc.T
        t_cam_base[:3, 3] = -r_bc.T @ t_base_cam[:3, 3]

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
        cv2.putText(
            canvas,
            "BOT(53mm)",
            (p_bot[0] + 8, p_bot[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "TOP(75mm)",
            (p_top[0] + 8, p_top[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    summary = (
        f"Cup bot=({estimate.cup_bottom_xy_m[0] * 1000:.1f}, {estimate.cup_bottom_xy_m[1] * 1000:.1f})mm "
        f"yaw={estimate.yaw_deg:.1f}deg table_z={estimate.table_z_m * 1000:.1f}mm "
        f"L={estimate.length_mm:.1f}mm W={estimate.width_bot_mm:.1f}->{estimate.width_top_mm:.1f}mm"
    )
    cv2.rectangle(canvas, (12, 12), (1060, 56), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        summary,
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 245, 255),
        2,
        cv2.LINE_AA,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def verify_upright_cup(
    color_bgr: np.ndarray,
    depth_m: np.ndarray | None,
    t_base_flange: np.ndarray,
    target_xy_m: tuple[float, float] | np.ndarray,
    *,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    camera_metadata: dict[str, Any] | None = None,
    table_z_m: float = 0.0,
) -> tuple[UprightCupVerification, dict[str, Any]]:
    """Verify via Eye-in-Hand RGB-D that the paper cup stands upright at target_xy_m."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = resolve_camera_config(handeye, color_bgr, depth_m, camera_metadata)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam
    t_cam_base = np.eye(4, dtype=np.float64)
    r_bc = t_base_cam[:3, :3]
    t_cam_base[:3, :3] = r_bc.T
    t_cam_base[:3, 3] = -r_bc.T @ t_base_cam[:3, 3]

    target = np.asarray(target_xy_m, dtype=np.float64).reshape(2)
    h, w = color_bgr.shape[:2]

    # Build expected 2D image ROI by projecting the 3D bounding cylinder around target_xy
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    r_search = 0.065
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
            np.full_like(angles, float(table_z_m) + CUP_HEIGHT_M + 0.010),
        ]
    )
    cyl_pts_cam = np.vstack([ring_bot, ring_top]) @ t_cam_base[:3, :3].T + t_cam_base[:3, 3]
    cyl_uv = project_cam_to_pixels(cyl_pts_cam, cam_cfg).astype(np.int32)
    hull = cv2.convexHull(cyl_uv)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, hull, 255)
    # Exclude lower gripper housing and fingers (depth < 0.24m, plus Gemini's bottom image band)
    roi_bool = roi_mask > 0
    if cam_cfg.get("type") != "realsense":
        roi_bool[int(0.80 * h) :, :] = False
    if depth_m is not None and depth_m.shape == (h, w):
        near_gripper = (depth_m > 0.05) & (depth_m < 0.24)
        near_gripper_dilated = (
            cv2.dilate(near_gripper.astype(np.uint8) * 255, np.ones((15, 15), dtype=np.uint8)) > 0
        )
        roi_bool &= ~near_gripper_dilated

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    cup_mask = (
        white_cup_mask(hsv, cam_cfg, value=145)
        if cam_cfg.get("type") == "realsense"
        else (hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 145)
    ) & roi_bool
    cup_u8 = cv2.morphologyEx(
        cup_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    cup_mask = cup_u8 > 0
    roi_cup_pixels = int(np.count_nonzero(cup_mask))

    ys, xs = np.mgrid[0:h, 0:w]
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
                & (pts_base[:, 2] <= table_z_m + 0.115)
            )
            cup_pts = pts_base[in_cyl]
            point_count = len(cup_pts)
            if point_count >= 50:
                rim_height_mm = float(np.percentile(cup_pts[:, 2] - table_z_m, 95.0) * 1000.0)

    # Estimate horizontal cup center from 2D cup contour centroid intersected with z = table_z + 0.065m
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
            t_hit = ((table_z_m + 0.065) - cam_pos_base[2]) / ray_dir_base[2]
            if t_hit > 0:
                hit_pt = cam_pos_base + t_hit * ray_dir_base
                measured_xy = hit_pt[:2].copy()
                if point_count < 50:
                    rim_height_mm = max(rim_height_mm, 80.0)

    xy_error_mm = float(np.linalg.norm(measured_xy - target) * 1000.0)
    reasons: list[str] = []
    if roi_cup_pixels < 1200:
        reasons.append(f"Too few cup pixels in the target ROI ({roi_cup_pixels} < 1200)")
    if xy_error_mm > 45.0:
        reasons.append(f"Placement XY error too large ({xy_error_mm:.1f} mm > 45.0 mm)")
    if point_count >= 150 and rim_height_mm < 45.0:
        reasons.append(
            f"Rim height too low ({rim_height_mm:.1f} mm < 45.0 mm); cup may have tipped over"
        )

    ok = len(reasons) == 0
    verification = UprightCupVerification(
        ok=ok,
        target_xy_m=(float(target[0]), float(target[1])),
        measured_xy_m=(float(measured_xy[0]), float(measured_xy[1])),
        xy_error_mm=xy_error_mm,
        rim_height_mm=rim_height_mm,
        point_count=point_count,
        roi_cup_pixels=roi_cup_pixels,
        reason="ok" if ok else "; ".join(reasons),
    )
    debug_info = {
        "cup_mask": cup_mask,
        "hull": hull,
        "t_base_cam": t_base_cam,
        "camera_cfg": cam_cfg,
    }
    return verification, debug_info


def save_post_place_overlay(
    color_bgr: np.ndarray,
    verification: UprightCupVerification,
    debug_info: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render post-placement closed-loop verification result onto BGR image."""
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
                [verification.target_xy_m[0], verification.target_xy_m[1], 0.065],
                [verification.measured_xy_m[0], verification.measured_xy_m[1], 0.065],
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
        f"Post-Place Verify: {status_str} | "
        f"target=({verification.target_xy_m[0] * 1000:.0f},{verification.target_xy_m[1] * 1000:.0f})mm "
        f"err={verification.xy_error_mm:.1f}mm rim_z={verification.rim_height_mm:.1f}mm "
        f"px={verification.roi_cup_pixels}"
    )
    cv2.rectangle(canvas, (12, 12), (1180, 56), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        summary,
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (120, 255, 140) if verification.ok else (100, 120, 255),
        2,
        cv2.LINE_AA,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out
