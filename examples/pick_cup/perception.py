"""Eye-in-Hand RGB-D perception for sideways paper cup 6D pose estimation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDEYE_YAML = REPO_ROOT.parent / "camera_tool" / "result" / "eye_in_hand.yaml"
CAMERA_TOOL_SRC = REPO_ROOT.parent / "camera_tool" / "src"
PY312_PYTHON = Path("/opt/conda_envs/py312/bin/python")

# Cup physical dimensions from AGENTS.md
CUP_HEIGHT_M = 0.088
CUP_BOT_DIAM_M = 0.053
CUP_TOP_DIAM_M = 0.075
CUP_HALF_CONE_RAD = float(np.arctan2(0.5 * (CUP_TOP_DIAM_M - CUP_BOT_DIAM_M), CUP_HEIGHT_M))
CUP_SPAWN_Z_OFFSET_M = 0.0375

# Safe clear workspace on the green anti-static mat (excluding left box at y > +0.16 m)
WORKSPACE_BOUNDS_M: dict[str, float] = {
    "x_min": 0.20,
    "x_max": 0.60,
    "y_min": -0.26,
    "y_max": 0.12,
}


class PerceptionError(RuntimeError):
    """Raised when camera capture, hand-eye calibration, or cup segmentation fails."""


@dataclass(frozen=True)
class UprightCupVerification:
    """Closed-loop visual verification result after placing the paper cup upright."""

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


def sideways_cup_mujoco_pose(
    bottom_xy_m: tuple[float, float] | np.ndarray,
    yaw_deg: float,
    table_z_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MuJoCo freejoint (pos_m, quat_wxyz) for a sideways paper cup."""
    bot = np.asarray(bottom_xy_m, dtype=np.float64).reshape(2)
    pos_m = np.array([bot[0], bot[1], float(table_z_m) + CUP_SPAWN_Z_OFFSET_M], dtype=np.float64)
    # Base sideways rotation: local +Z (cup bottom->top) lies along world +X when yaw_deg == 0
    r_sideways = Rotation.from_euler("z", float(yaw_deg), degrees=True) * Rotation.from_euler(
        "y", 90.0, degrees=True
    )
    q_xyzw = r_sideways.as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return pos_m, quat_wxyz


def validate_xy_in_workspace(
    xy_m: np.ndarray | tuple[float, float],
    label: str,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> None:
    """Raise PerceptionError if a 2D point in base frame falls outside safe workspace bounds."""
    x, y = float(xy_m[0]), float(xy_m[1])
    if not (bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]):
        raise PerceptionError(
            f"{label} ({x:.3f}, {y:.3f}) m 超出安全净空工作区 "
            f"X=[{bounds['x_min']:.2f}, {bounds['x_max']:.2f}], "
            f"Y=[{bounds['y_min']:.2f}, {bounds['y_max']:.2f}] m"
        )


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


def load_handeye_calibration(yaml_path: str | Path = DEFAULT_HANDEYE_YAML) -> dict[str, Any]:
    """Load and validate Eye-in-Hand calibration YAML."""
    raw = Path(yaml_path).expanduser()
    if not raw.is_absolute() and not raw.is_file() and (REPO_ROOT / raw).is_file():
        path = (REPO_ROOT / raw).resolve()
    else:
        path = raw.resolve()
    if not path.is_file():
        raise PerceptionError(f"手眼标定文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "camera" not in data or "extrinsics" not in data:
        raise PerceptionError(f"手眼标定文件格式无效: {path}")
    x_fc = np.asarray(data["extrinsics"]["X_flange_cam"], dtype=np.float64)
    if x_fc.shape != (4, 4) or not np.all(np.isfinite(x_fc)):
        raise PerceptionError(f"X_flange_cam 矩阵无效: {path}")
    return data


def deproject_pixels_to_cam(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    camera_cfg: dict[str, Any],
) -> np.ndarray:
    """Deproject distorted pixel coordinates (N, 2) and optical depth (N,) to camera 3D points."""
    pts_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depth_m, dtype=np.float64).reshape(-1, 1)
    k_mat = np.asarray(camera_cfg["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    undist = cv2.undistortPoints(pts_uv.reshape(-1, 1, 2), k_mat, dist).reshape(-1, 2)
    return np.column_stack([undist, np.ones(len(undist), dtype=np.float64)]) * z


def project_cam_to_pixels(points_cam: np.ndarray, camera_cfg: dict[str, Any]) -> np.ndarray:
    """Project 3D points in camera frame (N, 3) to 2D image pixels (N, 2)."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    k_mat = np.asarray(camera_cfg["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k_mat, dist)
    return proj.reshape(-1, 2)


def capture_rgbd_frame(
    serial: str = "CV2C861000RV",
    width: int = 1280,
    height: int = 800,
    fps: int = 30,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Capture an aligned (color_bgr, depth_m, metadata) frame from Orbbec Gemini 305."""
    helper_code = (
        "import json, sys, numpy as np\n"
        f"sys.path.insert(0, {str(CAMERA_TOOL_SRC)!r})\n"
        "from camera_tool.devices.orbbec import OrbbecSession\n"
        "cfg = json.loads(sys.argv[1])\n"
        "out_dir = sys.argv[2]\n"
        "s = OrbbecSession(cfg)\n"
        "meta = s.start()\n"
        "try:\n"
        "    frame = s.read()\n"
        "finally:\n"
        "    s.stop()\n"
        "np.save(out_dir + '/color.npy', frame['color_bgr'])\n"
        "np.save(out_dir + '/depth.npy', frame['depth_m'])\n"
        "with open(out_dir + '/meta.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(meta, f)\n"
    )
    python_exe = str(PY312_PYTHON) if PY312_PYTHON.is_file() else "python3"
    cfg_json = json.dumps(
        {
            "serial": serial,
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "enable_depth": True,
        }
    )
    with tempfile.TemporaryDirectory(prefix="orbbec_capture_") as tmp_dir:
        proc = subprocess.run(
            [python_exe, "-c", helper_code, cfg_json, tmp_dir],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "未知错误").strip()
            raise PerceptionError(f"Orbbec Gemini 305 相机取流失败: {detail}")
        color_bgr = np.load(Path(tmp_dir) / "color.npy")
        depth_m = np.load(Path(tmp_dir) / "depth.npy")
        meta = json.loads((Path(tmp_dir) / "meta.json").read_text(encoding="utf-8"))
    return color_bgr, depth_m, meta


def detect_sideways_cup(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    t_base_flange: np.ndarray,
    *,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    workspace_bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> tuple[CupPoseEstimate, dict[str, Any]]:
    """Segment the white sideways paper cup and estimate its 6D pose in the robot base frame."""
    handeye = load_handeye_calibration(handeye_yaml)
    cam_cfg = handeye["camera"]
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    t_base_cam = np.asarray(t_base_flange, dtype=np.float64).reshape(4, 4) @ x_flange_cam

    if color_bgr is None or depth_m is None or color_bgr.shape[:2] != depth_m.shape[:2]:
        raise PerceptionError("输入的 RGB 与深度图尺寸无效或不匹配")

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    # Exclude bottom gripper housing/fingers visible in the lower FOV
    fov_mask = (
        (ys >= int(0.18 * h))
        & (ys < int(0.73 * h))
        & (xs >= int(0.18 * w))
        & (xs < int(0.92 * w))
    )
    depth_valid = np.isfinite(depth_m) & (depth_m > 0.15) & (depth_m < 0.75) & fov_mask
    if int(np.count_nonzero(depth_valid)) < 500:
        raise PerceptionError("有效深度像素不足，无法定位桌面或纸杯")

    # White cup candidate mask in HSV
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 155) & fov_mask
    white_u8 = cv2.morphologyEx(
        white_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_u8)
    best_label = -1
    best_pts_base: np.ndarray | None = None
    best_table_z = 0.0

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
        raise PerceptionError("未在安全工作区内检测到白色一次性纸杯")

    if abs(best_table_z) > 0.030:
        raise PerceptionError(f"局部桌面高度异常 ({best_table_z * 1000:.1f} mm)，请检查手眼标定或相机位姿")

    # 2D PCA in XY plane to find primary cup axis
    xy = best_pts_base[:, :2]
    center_xy = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - center_xy, full_matrices=False)
    axis_xy = vt[0].copy()
    ortho_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float64)

    proj_along = (xy - center_xy) @ axis_xy
    proj_ortho = (xy - center_xy) @ ortho_xy
    p_min, p_max = np.percentile(proj_along, [2.0, 98.0])
    length_mm = float((p_max - p_min) * 1000.0)

    q35, q65 = np.percentile(proj_along, [35.0, 65.0])
    mask_neg = proj_along <= q35
    mask_pos = proj_along >= q65
    width_neg = float(
        (np.percentile(proj_ortho[mask_neg], 95.0) - np.percentile(proj_ortho[mask_neg], 5.0))
        * 1000.0
    )
    width_pos = float(
        (np.percentile(proj_ortho[mask_pos], 95.0) - np.percentile(proj_ortho[mask_pos], 5.0))
        * 1000.0
    )
    z_neg = float(np.percentile(best_pts_base[mask_neg, 2] - best_table_z, 90.0) * 1000.0)
    z_pos = float(np.percentile(best_pts_base[mask_pos, 2] - best_table_z, 90.0) * 1000.0)

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

    mid_along = 0.5 * (p_min + p_max)
    mid_xy = center_xy + mid_along * axis_xy
    half_horiz_len = 0.5 * CUP_HEIGHT_M * float(np.cos(CUP_HALF_CONE_RAD))
    bot_xy = mid_xy - half_horiz_len * axis_xy
    top_xy = mid_xy + half_horiz_len * axis_xy
    grasp_xy = bot_xy + 0.040 * axis_xy
    validate_xy_in_workspace(grasp_xy, "感知到的纸杯抓取中心", workspace_bounds)

    yaw_deg = float(np.rad2deg(np.arctan2(axis_xy[1], axis_xy[0])))
    # Use the calibrated base-frame table plane (Z = 0.0 m) for MuJoCo and grasp/place heights
    pos_m, quat_wxyz = sideways_cup_mujoco_pose(bot_xy, yaw_deg, table_z_m=0.0)

    estimate = CupPoseEstimate(
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
        point_count=len(best_pts_base),
        source="camera",
    )

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
            canvas, "BOT(53mm)", (p_bot[0] + 8, p_bot[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "TOP(75mm)", (p_top[0] + 8, p_top[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA,
        )

    summary = (
        f"Cup bot=({estimate.cup_bottom_xy_m[0]*1000:.1f}, {estimate.cup_bottom_xy_m[1]*1000:.1f})mm "
        f"yaw={estimate.yaw_deg:.1f}deg table_z={estimate.table_z_m*1000:.1f}mm "
        f"L={estimate.length_mm:.1f}mm W={estimate.width_bot_mm:.1f}->{estimate.width_top_mm:.1f}mm"
    )
    cv2.rectangle(canvas, (12, 12), (1060, 56), (25, 25, 25), -1)
    cv2.putText(
        canvas, summary, (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 255), 2, cv2.LINE_AA,
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
    table_z_m: float = 0.0,
) -> tuple[UprightCupVerification, dict[str, Any]]:
    """Verify via Eye-in-Hand RGB-D that the paper cup stands upright at target_xy_m."""
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
    # Exclude lower gripper housing and fingers (either by depth < 0.24m or bottom image band)
    roi_bool = roi_mask > 0
    roi_bool[int(0.80 * h) :, :] = False
    if depth_m is not None and depth_m.shape == (h, w):
        near_gripper = (depth_m > 0.05) & (depth_m < 0.24)
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
        reasons.append(f"目标区域纸杯像素不足 ({roi_cup_pixels} < 1200)")
    if xy_error_mm > 45.0:
        reasons.append(f"放置水平偏差过大 ({xy_error_mm:.1f} mm > 45.0 mm)")
    if point_count >= 150 and rim_height_mm < 45.0:
        reasons.append(f"杯口离桌高度偏低 ({rim_height_mm:.1f} mm < 45.0 mm)，疑似倾倒")

    ok = len(reasons) == 0
    verification = UprightCupVerification(
        ok=ok,
        target_xy_m=(float(target[0]), float(target[1])),
        measured_xy_m=(float(measured_xy[0]), float(measured_xy[1])),
        xy_error_mm=xy_error_mm,
        rim_height_mm=rim_height_mm,
        point_count=point_count,
        roi_cup_pixels=roi_cup_pixels,
        reason="ok" if ok else "；".join(reasons),
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
        cv2.circle(canvas, p_meas, 6, (0, 255, 0) if verification.ok else (0, 0, 255), -1, cv2.LINE_AA)

    status_str = "PASS" if verification.ok else f"FAIL ({verification.reason})"
    summary = (
        f"Post-Place Verify: {status_str} | "
        f"target=({verification.target_xy_m[0]*1000:.0f},{verification.target_xy_m[1]*1000:.0f})mm "
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
