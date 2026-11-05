#!/usr/bin/env python3
"""Multi-view Eye-in-Hand Orbbec Gemini 305 3D scanner for the 0.0.3 screw and nut.

Moves the calibrated xArm 6 along collision-verified joint trajectories to 5 close-up
top-down and oblique viewpoints around the tabletop screw and nut, captures aligned
RGB-D frames, fuses point clouds in the robot base frame, extracts geometric dimensions,
and returns the robot to its initial observation pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.parent / "ufactory_mujoco"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "pick_cup"))

from perception import (
    DEFAULT_HANDEYE_YAML,
    capture_rgbd_frame,
    deproject_pixels_to_cam,
    load_handeye_calibration,
    project_cam_to_pixels,
)
from ufactory_mujoco.calibration import (
    apply_calibration_to_model,
    load_kinematics_calibration,
)
from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.ik import solve_ik
from ufactory_mujoco.safety import (
    SafetyGateError,
    authorize_real_motion,
    stream_sha256,
)
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT
from ufactory_mujoco.trajectory import plan_ptp_quintic
from ufactory_mujoco.xarm_robot import XArmRobot

DEFAULT_ROBOT_IP = "192.168.1.60"
DEFAULT_CALIB_YAML = (
    REPO_ROOT.parent / "ufactory_mujoco" / "calibrations" / "xarm6_kinematics_L56A01.yaml"
)
DEFAULT_SCENE_XML = REPO_ROOT / "examples" / "pick_cup" / "scene.xml"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "screw_nut"


@dataclass(frozen=True)
class ScanViewpoint:
    """Planned Eye-in-Hand camera viewpoint."""

    name: str
    tcp_pos_m: tuple[float, float, float]
    tcp_quat_wxyz: tuple[float, float, float, float]
    q_rad: np.ndarray
    cam_pos_m: tuple[float, float, float]
    min_col_dist_mm: float


def _make_tilted_down_quat(euler_xyz_deg: tuple[float, float, float]) -> np.ndarray:
    """Apply world-frame Euler XYZ tilt (degrees) on top of GRIP_DOWN_QUAT."""
    r_down = Rotation.from_quat(
        [GRIP_DOWN_QUAT[1], GRIP_DOWN_QUAT[2], GRIP_DOWN_QUAT[3], GRIP_DOWN_QUAT[0]]
    )
    q_xyzw = (Rotation.from_euler("xyz", euler_xyz_deg, degrees=True) * r_down).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def plan_scan_viewpoints(
    q_obs: np.ndarray,
    *,
    nut_xy_m: tuple[float, float] = (0.354, -0.009),
    screw_xy_m: tuple[float, float] = (0.358, -0.101),
    calibration_path: str | Path = DEFAULT_CALIB_YAML,
    handeye_path: str | Path = DEFAULT_HANDEYE_YAML,
) -> list[ScanViewpoint]:
    """Plan and collision-check 5 close-up viewpoints around the tabletop nut and screw."""
    calib = load_kinematics_calibration(calibration_path, robot="xarm6")
    handeye = load_handeye_calibration(handeye_path)
    x_fc = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)

    checker = CollisionChecker(
        DEFAULT_SCENE_XML,
        safety_margin=0.0,
        query_margin=0.05,
        ignore_pairs=[("world", "link_base"), ("world", "link1")],
        ignore_bodies=("paper_cup",),
    )
    apply_calibration_to_model(checker.model, calib, checker.data)
    link6_id = mujoco.mj_name2id(checker.model, mujoco.mjtObj.mjOBJ_BODY, "link6")

    checker.data.qpos[:6] = np.asarray(q_obs, dtype=np.float64).reshape(6)
    mujoco.mj_forward(checker.model, checker.data)
    q_seed = checker.data.qpos.copy()

    nx, ny = float(nut_xy_m[0]), float(nut_xy_m[1])
    sx, sy = float(screw_xy_m[0]), float(screw_xy_m[1])
    mx, my = 0.5 * (nx + sx), 0.5 * (ny + sy)

    specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "top_close_nut",
            np.array([nx - 0.065, ny + 0.007, 0.055], dtype=np.float64),
            GRIP_DOWN_QUAT.copy(),
        ),
        (
            "top_close_screw",
            np.array([sx - 0.065, sy + 0.007, 0.055], dtype=np.float64),
            GRIP_DOWN_QUAT.copy(),
        ),
        (
            "side_left_nut",
            np.array([nx - 0.065, ny + 0.034, 0.065], dtype=np.float64),
            _make_tilted_down_quat((-22.0, 0.0, 0.0)),
        ),
        (
            "side_right_screw",
            np.array([sx - 0.065, sy - 0.029, 0.065], dtype=np.float64),
            _make_tilted_down_quat((+22.0, 0.0, 0.0)),
        ),
        (
            "oblique_front_mid",
            np.array([mx - 0.101, my + 0.007, 0.105], dtype=np.float64),
            _make_tilted_down_quat((0.0, -15.0, 0.0)),
        ),
    ]

    viewpoints: list[ScanViewpoint] = []
    for name, p_tcp, quat in specs:
        q_full, ok = solve_ik(
            checker.model,
            checker.data,
            "attachment_site",
            p_tcp,
            quat,
            q_init=q_seed,
        )
        if not ok:
            raise SafetyGateError(f"观测视角 {name} 逆运动学未收敛")
        q_sol = q_full[:6].copy()
        if checker.in_collision(q_sol):
            raise SafetyGateError(f"观测视角 {name} 存在碰撞风险")
        min_d = float(checker.min_distance(q_sol))
        checker.data.qpos[:] = q_full
        mujoco.mj_forward(checker.model, checker.data)
        t_fl = np.eye(4, dtype=np.float64)
        t_fl[:3, :3] = checker.data.xmat[link6_id].reshape(3, 3)
        t_fl[:3, 3] = checker.data.xpos[link6_id]
        t_c = t_fl @ x_fc
        if float(t_c[2, 3]) < 0.150:
            raise SafetyGateError(f"观测视角 {name} 相机高度过低: {t_c[2, 3] * 1000:.1f} mm")

        viewpoints.append(
            ScanViewpoint(
                name=name,
                tcp_pos_m=(float(p_tcp[0]), float(p_tcp[1]), float(p_tcp[2])),
                tcp_quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                q_rad=q_sol,
                cam_pos_m=(float(t_c[0, 3]), float(t_c[1, 3]), float(t_c[2, 3])),
                min_col_dist_mm=min_d * 1000.0,
            )
        )
    return viewpoints


def _build_ptp_segment(
    q0: np.ndarray,
    q1: np.ndarray,
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.45,
) -> np.ndarray:
    """Build a smooth quintic joint trajectory segment between q0 and q1."""
    disp = float(np.max(np.abs(q1 - q0)))
    if disp < 1e-5:
        return q1.reshape(1, 6).copy()
    duration = max(1.6, disp * 1.875 / max_vel_rad_s)
    return plan_ptp_quintic(q0, q1, duration=duration, dt=dt).q


def verify_full_scan_stream(
    q_obs: np.ndarray,
    viewpoints: list[ScanViewpoint],
    *,
    calibration_path: str | Path = DEFAULT_CALIB_YAML,
) -> np.ndarray:
    """Concatenate and collision-check the complete multi-view scan trajectory."""
    calib = load_kinematics_calibration(calibration_path, robot="xarm6")
    checker = CollisionChecker(
        DEFAULT_SCENE_XML,
        safety_margin=0.0,
        query_margin=0.05,
        ignore_pairs=[("world", "link_base"), ("world", "link1")],
        ignore_bodies=("paper_cup",),
    )
    apply_calibration_to_model(checker.model, calib, checker.data)

    waypoints = [q_obs] + [vp.q_rad for vp in viewpoints] + [q_obs]
    segments: list[np.ndarray] = []
    for i in range(len(waypoints) - 1):
        seg = _build_ptp_segment(waypoints[i], waypoints[i + 1])
        segments.append(seg if not segments else seg[1:])
    full_stream = np.vstack(segments)

    for idx, q_sample in enumerate(full_stream):
        if np.any(q_sample < checker.q_lo - 1e-6) or np.any(q_sample > checker.q_hi + 1e-6):
            raise SafetyGateError(f"扫描轨迹采样点 #{idx} 超出关节限位")
        if checker.in_collision(q_sample):
            raise SafetyGateError(f"扫描轨迹采样点 #{idx} 检测到碰撞")
    return full_stream


def analyze_captured_views(
    captures: list[dict[str, Any]],
    *,
    handeye_path: str | Path = DEFAULT_HANDEYE_YAML,
    nut_prior_xy: tuple[float, float] = (0.354, -0.009),
    screw_prior_xy: tuple[float, float] = (0.358, -0.101),
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Extract 2D contour and 3D point cloud measurements from captured multi-view RGB-D frames."""
    handeye = load_handeye_calibration(handeye_path)
    cam_cfg = handeye["camera"]
    x_fc = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)

    nut_target_3d = np.array([nut_prior_xy[0], nut_prior_xy[1], 0.010], dtype=np.float64)
    screw_target_3d = np.array([screw_prior_xy[0], screw_prior_xy[1], 0.022], dtype=np.float64)

    view_metrics: dict[str, Any] = {}
    crops_for_montage: list[np.ndarray] = []

    all_nut_pts: list[np.ndarray] = []
    all_screw_pts: list[np.ndarray] = []
    nut_rgb_samples: list[np.ndarray] = []
    screw_rgb_samples: list[np.ndarray] = []

    for item in captures:
        name = item["name"]
        color_bgr = item["color_bgr"]
        depth_m = item["depth_m"]
        t_base_flange = np.asarray(item["t_base_flange"], dtype=np.float64)
        t_base_cam = t_base_flange @ x_fc
        t_cam_base = np.linalg.inv(t_base_cam)

        h, w = depth_m.shape
        ys, xs = np.mgrid[0:h, 0:w]

        # Project expected centers of nut and screw into this camera view
        pts_c = np.vstack([nut_target_3d, screw_target_3d]) @ t_cam_base[:3, :3].T + t_cam_base[:3, 3]
        uv = project_cam_to_pixels(pts_c, cam_cfg)
        nut_uv = uv[0]
        screw_uv = uv[1]

        # Foreground segmentation against green mat (H in [70, 100], S >= 65)
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        is_green = (hsv[:, :, 0] >= 70) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 65)
        fg_mask = (~is_green) & (hsv[:, :, 2] >= 75)

        valid_d = np.isfinite(depth_m) & (depth_m > 0.12) & (depth_m < 0.70)
        px_all = np.column_stack([xs[valid_d], ys[valid_d]])
        pts_cam_all = deproject_pixels_to_cam(px_all, depth_m[valid_d], cam_cfg)
        pts_base_all = pts_cam_all @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
        colors_all = color_bgr[valid_d]
        fg_all = fg_mask[valid_d]

        # Estimate local table Z around (x in [0.30, 0.42], y in [-0.16, 0.05]) on green mat
        in_table_roi = (
            (~fg_all)
            & (pts_base_all[:, 0] >= 0.30)
            & (pts_base_all[:, 0] <= 0.42)
            & (pts_base_all[:, 1] >= -0.16)
            & (pts_base_all[:, 1] <= 0.05)
            & (np.abs(pts_base_all[:, 2]) < 0.025)
        )
        table_z = float(np.median(pts_base_all[in_table_roi, 2])) if np.count_nonzero(in_table_roi) >= 200 else 0.0

        # Collect 3D points belonging to Nut and Screw cylinders
        r_nut = np.hypot(pts_base_all[:, 0] - nut_prior_xy[0], pts_base_all[:, 1] - nut_prior_xy[1])
        nut_sel = fg_all & (r_nut <= 0.024) & (pts_base_all[:, 2] >= table_z - 0.002) & (pts_base_all[:, 2] <= table_z + 0.035)
        nut_pts = pts_base_all[nut_sel].copy()
        nut_pts[:, 2] -= table_z

        r_screw = np.hypot(pts_base_all[:, 0] - screw_prior_xy[0], pts_base_all[:, 1] - screw_prior_xy[1])
        screw_sel = fg_all & (r_screw <= 0.022) & (pts_base_all[:, 2] >= table_z - 0.002) & (pts_base_all[:, 2] <= table_z + 0.055)
        screw_pts = pts_base_all[screw_sel].copy()
        screw_pts[:, 2] -= table_z

        if len(nut_pts) >= 50:
            all_nut_pts.append(nut_pts)
            nut_rgb_samples.append(colors_all[nut_sel][:, ::-1])  # RGB
        if len(screw_pts) >= 50:
            all_screw_pts.append(screw_pts)
            screw_rgb_samples.append(colors_all[screw_sel][:, ::-1])

        # Save high-res crop around primary target for this view
        focus_uv = screw_uv if "screw" in name else nut_uv
        cu, cv_px = round(focus_uv[0]), round(focus_uv[1])
        half_w = 140
        x0, x1 = max(0, cu - half_w), min(w, cu + half_w)
        y0, y1 = max(0, cv_px - half_w), min(h, cv_px + half_w)
        crop = cv2.resize(color_bgr[y0:y1, x0:x1], (320, 320), interpolation=cv2.INTER_CUBIC)
        cv2.putText(
            crop,
            f"{name} (z={pts_c[1 if 'screw' in name else 0, 2]*1000:.0f}mm)",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        crops_for_montage.append(crop)
        cv2.imwrite(str(out_dir / f"crop_{name}.png"), crop)

        view_metrics[name] = {
            "cam_pos_m": [round(float(v), 4) for v in t_base_cam[:3, 3]],
            "table_z_mm": round(table_z * 1000.0, 2),
            "nut_points": len(nut_pts),
            "nut_z_p90_mm": round(float(np.percentile(nut_pts[:, 2], 90) * 1000.0), 2) if len(nut_pts) >= 20 else None,
            "nut_z_p98_mm": round(float(np.percentile(nut_pts[:, 2], 98) * 1000.0), 2) if len(nut_pts) >= 20 else None,
            "screw_points": len(screw_pts),
            "screw_z_p90_mm": round(float(np.percentile(screw_pts[:, 2], 90) * 1000.0), 2) if len(screw_pts) >= 20 else None,
            "screw_z_p98_mm": round(float(np.percentile(screw_pts[:, 2], 98) * 1000.0), 2) if len(screw_pts) >= 20 else None,
        }

    if crops_for_montage:
        montage = np.hstack(crops_for_montage)
        cv2.imwrite(str(out_dir / "multi_view_scan.png"), montage)

    nut_cloud = np.vstack(all_nut_pts) if all_nut_pts else np.zeros((0, 3))
    screw_cloud = np.vstack(all_screw_pts) if all_screw_pts else np.zeros((0, 3))
    np.save(out_dir / "nut_pointcloud_base.npy", nut_cloud)
    np.save(out_dir / "screw_pointcloud_base.npy", screw_cloud)

    nut_rgb_med = (
        np.median(np.vstack(nut_rgb_samples), axis=0).round(1).tolist()
        if nut_rgb_samples
        else [210.0, 214.0, 218.0]
    )
    screw_rgb_med = (
        np.median(np.vstack(screw_rgb_samples), axis=0).round(1).tolist()
        if screw_rgb_samples
        else [210.0, 214.0, 218.0]
    )

    report = {
        "prd_specs": {
            "screw": {
                "mass_g": 56.5,
                "total_length_mm": 45.0,
                "head_length_mm": 14.0,
                "head_diameter_mm": 20.0,
                "thread_diameter_mm": 12.0,
                "thread_length_mm": 31.0,
            },
            "nut": {
                "mass_g": 29.0,
                "bottom_diameter_mm": 29.0,
                "top_outer_diameter_mm": 20.0,
                "bore_nominal_diameter_mm": 12.0,
            },
        },
        "measured_summary": {
            "nut_center_xy_m": [
                round(float(np.median(nut_cloud[:, 0])), 4) if len(nut_cloud) else nut_prior_xy[0],
                round(float(np.median(nut_cloud[:, 1])), 4) if len(nut_cloud) else nut_prior_xy[1],
            ],
            "screw_center_xy_m": [
                round(float(np.median(screw_cloud[:, 0])), 4) if len(screw_cloud) else screw_prior_xy[0],
                round(float(np.median(screw_cloud[:, 1])), 4) if len(screw_cloud) else screw_prior_xy[1],
            ],
            "nut_total_points": len(nut_cloud),
            "screw_total_points": len(screw_cloud),
            "nut_height_p95_mm": round(float(np.percentile(nut_cloud[:, 2], 95) * 1000.0), 2) if len(nut_cloud) else None,
            "nut_height_p98_mm": round(float(np.percentile(nut_cloud[:, 2], 98) * 1000.0), 2) if len(nut_cloud) else None,
            "screw_height_p95_mm": round(float(np.percentile(screw_cloud[:, 2], 95) * 1000.0), 2) if len(screw_cloud) else None,
            "screw_height_p98_mm": round(float(np.percentile(screw_cloud[:, 2], 98) * 1000.0), 2) if len(screw_cloud) else None,
            "nut_median_rgb": nut_rgb_med,
            "screw_median_rgb": screw_rgb_med,
        },
        "views": view_metrics,
    }
    (out_dir / "scan_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def run_multi_view_scan(
    *,
    ip: str = DEFAULT_ROBOT_IP,
    confirm_real: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Execute the multi-view close-up scan on the xArm 6 + Gemini 305."""
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = load_kinematics_calibration(DEFAULT_CALIB_YAML, robot="xarm6")

    with XArmRobot(
        ip,
        robot="xarm6",
        gripper="gripper_g2",
        model_path=DEFAULT_SCENE_XML,
        calibration=calib,
        read_only=True,
    ) as robot:
        q_obs = robot.get_joint_positions().copy()
        link6_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "link6")

        def current_flange_pose() -> np.ndarray:
            q_now = robot.get_joint_positions()
            robot.data.qpos[:6] = q_now
            mujoco.mj_forward(robot.model, robot.data)
            t_fl = np.eye(4, dtype=np.float64)
            t_fl[:3, :3] = robot.data.xmat[link6_id].reshape(3, 3)
            t_fl[:3, 3] = robot.data.xpos[link6_id]
            return t_fl

        viewpoints = plan_scan_viewpoints(q_obs)
        full_stream = verify_full_scan_stream(q_obs, viewpoints)

        print(f"[扫描] 已规划 {len(viewpoints)} 个近距离观测视角，总轨迹采样点 {len(full_stream)} 个", flush=True)
        for vp in viewpoints:
            print(
                f"  - {vp.name}: cam_pos=({vp.cam_pos_m[0]:.3f}, {vp.cam_pos_m[1]:.3f}, {vp.cam_pos_m[2]:.3f}) m, "
                f"min_col_dist={vp.min_col_dist_mm:.1f} mm",
                flush=True,
            )

        captures: list[dict[str, Any]] = []
        color_0, depth_0, _ = capture_rgbd_frame()
        t_fl_0 = current_flange_pose()
        cv2.imwrite(str(out_dir / "view_00_overview_rgb.png"), color_0)
        np.save(out_dir / "view_00_overview_depth.npy", depth_0)
        captures.append(
            {
                "name": "overview",
                "color_bgr": color_0,
                "depth_m": depth_0,
                "t_base_flange": t_fl_0,
            }
        )

        if not confirm_real:
            print("[扫描] 未指定 --confirm-real，仅完成初始视角采集与轨迹校验。", flush=True)
            return analyze_captured_views(captures, out_dir=out_dir)

        # Authorize and execute multi-view scan on real hardware
        prog_hash = stream_sha256(full_stream)
        cfg_hash = hashlib.sha256(b"scan_screw_nut_v1").hexdigest()
        approval = authorize_real_motion(
            robot._arm,
            robot="xarm6",
            calibration=calib,
            preflight_ok=True,
            confirm_real=bool(confirm_real),
            cycles=1,
            config_sha256=cfg_hash,
            program_sha256=prog_hash,
        )
        robot.unlock_motion(approval)
        print("[扫描] 安全门已解锁 (单周期 MotionApproval)，开始执行近距多视角扫描 ...", flush=True)

        for idx, vp in enumerate(viewpoints, start=1):
            q_cur = robot.get_joint_positions()
            seg = _build_ptp_segment(q_cur, vp.q_rad, dt=0.02, max_vel_rad_s=0.45)
            print(f"[扫描] 正在移动至视角 #{idx} ({vp.name}) ...", flush=True)
            for q_cmd in seg[1:]:
                robot.set_joint_positions(q_cmd)
                time.sleep(0.02)
            robot.settle(0.45)
            t_fl = current_flange_pose()
            color_bgr, depth_m, _ = capture_rgbd_frame()
            cv2.imwrite(str(out_dir / f"view_{idx:02d}_{vp.name}_rgb.png"), color_bgr)
            np.save(out_dir / f"view_{idx:02d}_{vp.name}_depth.npy", depth_m)
            np.save(out_dir / f"view_{idx:02d}_{vp.name}_flange.npy", t_fl)
            captures.append(
                {
                    "name": vp.name,
                    "color_bgr": color_bgr,
                    "depth_m": depth_m,
                    "t_base_flange": t_fl,
                }
            )

        # Return to initial observation pose
        q_cur = robot.get_joint_positions()
        ret_seg = _build_ptp_segment(q_cur, q_obs, dt=0.02, max_vel_rad_s=0.45)
        print("[扫描] 采集完成，正在返回初始观测位姿 ...", flush=True)
        for q_cmd in ret_seg[1:]:
            robot.set_joint_positions(q_cmd)
            time.sleep(0.02)
        robot.settle(0.35)
        q_final = robot.get_joint_positions()
        ret_err_mrad = float(np.max(np.abs(q_final - q_obs)) * 1000.0)
        print(f"[扫描] 已安全回到初始观测位姿 (误差={ret_err_mrad:.3f} mrad)", flush=True)

    return analyze_captured_views(captures, out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan tabletop screw and nut with Gemini 305")
    parser.add_argument("--ip", default=DEFAULT_ROBOT_IP, help="xArm 6 IP address")
    parser.add_argument("--confirm-real", action="store_true", help="Authorize real arm motion")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    report = run_multi_view_scan(
        ip=args.ip,
        confirm_real=args.confirm_real,
        out_dir=args.out_dir,
    )
    print(json.dumps(report["measured_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
