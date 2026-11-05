"""Sim-to-Real digital twin preflight, trajectory smoothing, look-down refinement, and closed-loop execution for 0.0.3 screw_nut."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ufactory_mujoco"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perception import (
    DEFAULT_HANDEYE_YAML,
    NutPoseEstimate,
    OrbbecCameraStream,
    ScrewPoseEstimate,
    capture_rgbd_frame,
    detect_nut_and_screw_overview,
    detect_nut_near,
    detect_screw_near,
    load_handeye_calibration,
    pose_estimates_from_manual,
    save_detection_overlay,
    save_post_thread_overlay,
    verify_threaded_nut,
)
from ufactory_mujoco.calibration import (
    KinematicsCalibration,
    apply_calibration_to_model,
    load_kinematics_calibration,
)
from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.safety import (
    SafetyGateError,
    authorize_real_motion,
    enforce_single_cycle,
    run_sdk_simulation,
    stream_sha256,
)
from ufactory_mujoco.xarm_robot import XArmRobot

from common.motion import (
    interpolate_waypoint_chain as _interpolate_waypoint_chain,
)
from common.motion import (
    read_real_gripper_gap_mm as _read_real_gripper_gap_mm,
)
from common.motion import (
    solve_cartesian_segment as _solve_cartesian_segment,
)
from common.motion import (
    stream_waypoints_on_real as _stream_waypoints_on_real,
)
from common.vision import solve_camera_lookdown_q as _solve_camera_lookdown_q_common

if TYPE_CHECKING:
    from screw_nut import ScrewNutWaypoints

DEFAULT_ROBOT_IP = "192.168.1.60"
DEFAULT_CALIBRATION_YAML = (
    Path(__file__).resolve().parents[3]
    / "ufactory_mujoco"
    / "calibrations"
    / "xarm6_kinematics_L56A01.yaml"
)
MIN_CAMERA_TABLE_CLEARANCE_M = 0.088
CAM_HOVER_Z_M = 0.250
MIN_VALID_NUT_GRASP_GAP_MM = 12.0
MAX_HEX_FLAT_GAP_MM = 21.2
MIN_VALID_NUT_OPEN_GAP_MM = 42.0
REAL_PRE_GRASP_GAP_MM = 50.0
REAL_REGRIP_GAP_MM = 50.0
REAL_GRASP_GAP_MM = 16.5
REAL_GRASP_SPEED_PCT = 20.0
REAL_GRASP_FORCE_PCT = 25.0
REAL_OPEN_SPEED_PCT = 60.0
REAL_OPEN_FORCE_PCT = 50.0
Z_ADAPT_MIN_STEP_M = 0.0035
Z_ADAPT_MAX_STEP_M = 0.0050
MAX_Z_ADAPT_RETRIES_PER_STROKE = 3
MAX_ADAPTIVE_EXTRA_STROKES = 3
DEFAULT_EXEC_LOG_PATH = Path("build/screw_nut/real_execution.log")

__all__ = [
    "CAM_HOVER_Z_M",
    "DEFAULT_CALIBRATION_YAML",
    "DEFAULT_EXEC_LOG_PATH",
    "DEFAULT_ROBOT_IP",
    "MAX_ADAPTIVE_EXTRA_STROKES",
    "MAX_HEX_FLAT_GAP_MM",
    "MAX_Z_ADAPT_RETRIES_PER_STROKE",
    "MIN_CAMERA_TABLE_CLEARANCE_M",
    "MIN_VALID_NUT_GRASP_GAP_MM",
    "MIN_VALID_NUT_OPEN_GAP_MM",
    "REAL_GRASP_FORCE_PCT",
    "REAL_GRASP_GAP_MM",
    "REAL_GRASP_SPEED_PCT",
    "REAL_OPEN_FORCE_PCT",
    "REAL_OPEN_SPEED_PCT",
    "REAL_PRE_GRASP_GAP_MM",
    "REAL_REGRIP_GAP_MM",
    "Z_ADAPT_MAX_STEP_M",
    "Z_ADAPT_MIN_STEP_M",
    "ScrewNutTwinReport",
    "_build_screw_nut_shadow_stream",
    "_get_waypoint_tcp_z",
    "_interpolate_waypoint_chain",
    "_read_real_gripper_gap_mm",
    "_shift_waypoints_cartesian_z",
    "_solve_cartesian_segment",
    "_stream_waypoints_on_real",
    "make_screw_nut_collision_checker",
    "replan_screw_nut_from_refined",
    "run_digital_twin_preflight",
    "run_sim2real_pipeline",
    "solve_camera_lookdown_q",
]


@dataclass(frozen=True)
class ScrewNutTwinReport:
    """Verification report produced by the MuJoCo nut-threading digital twin rollout."""

    ok: bool
    nut_pose: NutPoseEstimate
    screw_pose: ScrewPoseEstimate
    thread_turns: float
    metrics: dict[str, Any]
    min_camera_z_mm: float
    min_collision_dist_mm: float
    max_joint_step_rad: float
    waypoint_count: int
    shadow_stream_rad: np.ndarray
    waypoints: ScrewNutWaypoints
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "nut_pose": self.nut_pose.as_dict(),
            "screw_pose": self.screw_pose.as_dict(),
            "thread_turns": float(self.thread_turns),
            "metrics": self.metrics,
            "min_camera_z_mm": float(self.min_camera_z_mm),
            "min_collision_dist_mm": float(self.min_collision_dist_mm),
            "max_joint_step_rad": float(self.max_joint_step_rad),
            "waypoint_count": int(self.waypoint_count),
            "shadow_samples": len(self.shadow_stream_rad),
            "failures": list(self.failures),
        }


def make_screw_nut_collision_checker(
    scene_path: str | Path,
    calibration: KinematicsCalibration | None = None,
) -> CollisionChecker:
    """Create a CollisionChecker that checks self-collision and arm/gripper vs table collision."""
    import screw_nut as demo

    checker = CollisionChecker(
        scene_path,
        safety_margin=0.0,
        query_margin=0.01,
        ignore_pairs=[("world", "link_base"), ("world", "link1")],
        ignore_bodies=demo.ALL_TASK_BODIES,
    )
    if calibration is not None:
        apply_calibration_to_model(checker.model, calibration, checker.data)
    return checker


def solve_camera_lookdown_q(
    target_xy_m: tuple[float, float] | np.ndarray,
    *,
    z_cam_m: float = CAM_HOVER_Z_M,
    q_seed: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
) -> np.ndarray:
    """Solve collision-free joint angles that place the Eye-in-Hand camera directly over target_xy_m."""
    import screw_nut as demo

    return _solve_camera_lookdown_q_common(
        target_xy_m,
        z_cam_m=z_cam_m,
        q_seed=q_seed,
        scene_path=demo.SCENE,
        calibration=calibration,
        handeye_yaml=handeye_yaml,
        ignore_bodies=demo.ALL_TASK_BODIES,
    )


def _get_waypoint_tcp_z(checker: CollisionChecker, q_wp: np.ndarray) -> float:
    """Return world TCP Z (m) of a 6-DOF joint waypoint using the calibrated MuJoCo model."""
    model = checker.model
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
    data.qpos[:6] = np.asarray(q_wp, dtype=np.float64).reshape(6)
    mujoco.mj_kinematics(model, data)
    return float(data.site_xpos[site_id][2])


def _shift_waypoints_cartesian_z(
    checker: CollisionChecker,
    waypoints: list[np.ndarray],
    dz_m: float,
    *,
    min_tcp_z_m: float | None = None,
    max_tcp_z_m: float | None = None,
) -> list[np.ndarray]:
    """Shift a 6-DOF joint waypoint list vertically by dz_m while preserving XY, tilt, and J6."""
    if abs(dz_m) < 1e-6 and min_tcp_z_m is None and max_tcp_z_m is None:
        return [np.asarray(q, dtype=np.float64).reshape(6).copy() for q in waypoints]
    from ufactory_mujoco.ik import solve_ik

    model = checker.model
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
    site_name = model.site(site_id).name

    shifted: list[np.ndarray] = []
    q_seed = data.qpos.copy()
    for q_wp in waypoints:
        q6 = np.asarray(q_wp, dtype=np.float64).reshape(6)
        data.qpos[:6] = q6
        mujoco.mj_kinematics(model, data)
        pos0 = data.site_xpos[site_id].copy()
        quat0 = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat0, data.site_xmat[site_id])
        target_z = float(pos0[2] + dz_m)
        if min_tcp_z_m is not None:
            target_z = max(float(min_tcp_z_m), target_z)
        if max_tcp_z_m is not None:
            target_z = min(float(max_tcp_z_m), target_z)
        if abs(target_z - float(pos0[2])) < 1e-6:
            shifted.append(q6.copy())
            continue
        target_pos = np.array([pos0[0], pos0[1], target_z], dtype=np.float64)
        q_seed[:6] = q6
        q_sol, ok = solve_ik(model, data, site_name, target_pos, quat0, q_init=q_seed)
        if not ok:
            raise SafetyGateError(f"Z轴自适应航点逆解未收敛: target_pos={target_pos}")
        q_new = np.asarray(q_sol[:6], dtype=np.float64).copy()
        q_new[5] = float(q6[5])
        if np.any(q_new < checker.q_lo - 1e-6) or np.any(q_new > checker.q_hi + 1e-6):
            raise SafetyGateError("Z轴自适应航点超出关节限位")
        if checker.in_collision(q_new):
            raise SafetyGateError("Z轴自适应航点检测到碰撞")
        shifted.append(q_new)
    return shifted


def _build_screw_nut_shadow_stream(
    q_init: np.ndarray,
    waypoints: ScrewNutWaypoints,
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.65,
) -> np.ndarray:
    """Concatenate all nut-threading phases into a C2-smooth 50 Hz joint trajectory (N, 6)."""
    q_cur = np.asarray(q_init, dtype=np.float64).reshape(-1).copy()
    chunks: list[np.ndarray] = [q_cur.reshape(1, -1)]

    def append_segment(wps: list[np.ndarray], *, soft_landing: bool) -> None:
        nonlocal q_cur
        if not wps:
            return
        seg = _interpolate_waypoint_chain(
            q_cur,
            wps,
            dt=dt,
            max_vel_rad_s=max_vel_rad_s,
            soft_landing=soft_landing,
        )
        if len(seg) > 1:
            chunks.append(seg[1:])
            q_cur = seg[-1].copy()

    # Phase 1: Approach and soft-landing descent to grasp nut
    append_segment(waypoints.approach_waypoints, soft_landing=True)
    # Phase 2: Lift, align over screw, and soft-landing pilot capture onto top thread ridge
    append_segment(waypoints.lift_align_waypoints, soft_landing=True)
    # Phase 3: Multi-stroke helical tightening and J6 rewind strokes
    for idx, stroke_wps in enumerate(waypoints.stroke_waypoints):
        append_segment(stroke_wps, soft_landing=True)
        if idx < len(waypoints.rewind_waypoints):
            append_segment(waypoints.rewind_waypoints[idx], soft_landing=True)
    # Phase 4: Vertical retreat and return to observation pose
    append_segment(waypoints.retreat_waypoints, soft_landing=False)

    return np.vstack(chunks)


def run_digital_twin_preflight(
    nut_pose: NutPoseEstimate,
    screw_pose: ScrewPoseEstimate,
    *,
    thread_turns: float = 10.0,
    q_init: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    headless: bool = False,
) -> ScrewNutTwinReport:
    """Execute the full nut pick, pilot capture, and multi-stroke threading in the calibrated MuJoCo twin."""
    import screw_nut as demo

    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = make_screw_nut_collision_checker(demo.SCENE, calibration=calibration)

    q_start = np.asarray(q_init, dtype=np.float64).reshape(6).copy()
    recorded_wps = demo.ScrewNutWaypoints()
    failures: list[str] = []
    min_cam_z_m = float("inf")
    min_col_dist_m = float("inf")
    metrics: dict[str, Any] = {}
    table_z_ref = float(nut_pose.table_z_m)

    init_nut = demo.InitialNutPose(
        xy_m=nut_pose.xy_m,
        yaw_deg=nut_pose.yaw_deg,
        table_z_m=table_z_ref,
    )
    screw_xy = np.asarray(screw_pose.xy_m, dtype=np.float64).reshape(2)

    with demo.ScrewNutRobot(viewer=not headless, calibration=calibration) as sim_robot:
        link6_id = mujoco.mj_name2id(sim_robot.model, mujoco.mjtObj.mjOBJ_BODY, "link6")

        def update_camera_clearance() -> None:
            nonlocal min_cam_z_m
            t_flange = np.eye(4, dtype=np.float64)
            t_flange[:3, :3] = sim_robot.data.xmat[link6_id].reshape(3, 3)
            t_flange[:3, 3] = sim_robot.data.xpos[link6_id]
            t_cam = t_flange @ x_flange_cam
            min_cam_z_m = min(min_cam_z_m, float(t_cam[2, 3]) - table_z_ref)

        orig_step = sim_robot._step

        def monitored_step() -> None:
            orig_step()
            update_camera_clearance()

        sim_robot._step = monitored_step  # type: ignore[method-assign]

        try:
            metrics = demo.run_screw_nut(
                sim_robot,
                nut_pose=init_nut,
                screw_xy=screw_xy,
                thread_turns=thread_turns,
                q_init=q_start,
                recorded_waypoints=recorded_wps,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"拧螺母数字孪生仿真预演失败: {exc}")

    if not failures:
        recorded_wps.retreat_waypoints.append(q_start.copy())

    shadow_stream = _build_screw_nut_shadow_stream(q_start, recorded_wps)
    max_step_rad = (
        float(np.max(np.abs(np.diff(shadow_stream, axis=0))))
        if len(shadow_stream) > 1
        else 0.0
    )

    fk_model = checker.model
    fk_data = mujoco.MjData(fk_model)
    link6_fk_id = mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    for idx, q_sample in enumerate(shadow_stream):
        if np.any(q_sample < checker.q_lo - 1e-6) or np.any(q_sample > checker.q_hi + 1e-6):
            failures.append(f"轨迹采样点 #{idx} 超出关节限位")
            break
        dist = float(checker.min_distance(q_sample))
        min_col_dist_m = min(min_col_dist_m, dist)
        if checker.in_collision(q_sample):
            failures.append(f"轨迹采样点 #{idx} 检测到碰撞 (min_dist={dist * 1000:.2f} mm)")
            break
        fk_data.qpos[:6] = q_sample
        mujoco.mj_forward(fk_model, fk_data)
        t_flange = np.eye(4, dtype=np.float64)
        t_flange[:3, :3] = fk_data.xmat[link6_fk_id].reshape(3, 3)
        t_flange[:3, 3] = fk_data.xpos[link6_fk_id]
        t_cam = t_flange @ x_flange_cam
        cam_clear = float(t_cam[2, 3]) - table_z_ref
        min_cam_z_m = min(min_cam_z_m, cam_clear)

    if min_cam_z_m < MIN_CAMERA_TABLE_CLEARANCE_M:
        failures.append(
            f"腕部相机最低离桌高度 {min_cam_z_m * 1000:.1f} mm 低于安全阈值 "
            f"{MIN_CAMERA_TABLE_CLEARANCE_M * 1000:.1f} mm"
        )
    if max_step_rad > 0.15:
        failures.append(f"相邻轨迹步关节角跨度过大 ({max_step_rad:.4f} rad > 0.15 rad)")

    total_wps = (
        len(recorded_wps.approach_waypoints)
        + len(recorded_wps.lift_align_waypoints)
        + len(recorded_wps.thread_waypoints)
        + len(recorded_wps.regrip_waypoints)
        + len(recorded_wps.retreat_waypoints)
    )
    return ScrewNutTwinReport(
        ok=len(failures) == 0,
        nut_pose=nut_pose,
        screw_pose=screw_pose,
        thread_turns=float(thread_turns),
        metrics=metrics,
        min_camera_z_mm=min_cam_z_m * 1000.0,
        min_collision_dist_mm=min_col_dist_m * 1000.0,
        max_joint_step_rad=max_step_rad,
        waypoint_count=total_wps,
        shadow_stream_rad=shadow_stream,
        waypoints=recorded_wps,
        failures=tuple(failures),
    )


def replan_screw_nut_from_refined(
    refined_nut: NutPoseEstimate,
    refined_screw: ScrewPoseEstimate,
    *,
    thread_turns: float,
    q_start: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
) -> ScrewNutTwinReport:
    """Recompute and collision-verify the nut-threading trajectory after look-down refinement."""
    report = run_digital_twin_preflight(
        refined_nut,
        refined_screw,
        thread_turns=thread_turns,
        q_init=q_start,
        calibration=calibration,
        handeye_yaml=handeye_yaml,
        headless=True,
    )
    if not report.ok:
        reason = "；".join(report.failures)
        raise SafetyGateError(f"正视精测后在线重规划未通过安全校验: {reason}")
    return report


def run_sim2real_pipeline(
    *,
    mode: str,
    thread_turns: float = 10.0,
    ip: str = DEFAULT_ROBOT_IP,
    calibration_path: str | Path = DEFAULT_CALIBRATION_YAML,
    handeye_path: str | Path = DEFAULT_HANDEYE_YAML,
    nut_pose_override: tuple[float, float, float] | None = None,
    screw_xy_override: tuple[float, float] | None = None,
    headless: bool = False,
    confirm_real: bool = False,
    cycles: int = 1,
    post_verify: bool = True,
    save_vis_path: str | Path | None = None,
    save_after_vis_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the end-to-end 0.0.3 nut-threading Sim-to-Real pipeline (dry-run, sdk-sim, or real)."""
    import screw_nut as demo

    if mode not in ("dry-run", "sdk-sim", "real"):
        raise ValueError(f"不支持的硬件运行模式: {mode!r}")
    enforce_single_cycle(cycles, mode)

    calib = load_kinematics_calibration(calibration_path, robot="xarm6")

    with XArmRobot(
        ip,
        robot="xarm6",
        gripper="gripper_g2",
        model_path=demo.SCENE,
        calibration=calib,
        read_only=True,
    ) as robot:
        q_obs = robot.get_joint_positions()
        link6_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        t_base_flange = np.eye(4, dtype=np.float64)
        t_base_flange[:3, :3] = robot.data.xmat[link6_id].reshape(3, 3)
        t_base_flange[:3, 3] = robot.data.xpos[link6_id]

        color_bgr: np.ndarray | None = None
        debug_info: dict[str, Any] = {}
        cam_stream: OrbbecCameraStream | None = None
        try:
            if nut_pose_override is not None and screw_xy_override is not None:
                nut_est, screw_est = pose_estimates_from_manual(
                    (nut_pose_override[0], nut_pose_override[1]),
                    nut_pose_override[2],
                    screw_xy_override,
                )
                print(
                    f"[感知] 手动指定螺母=({nut_est.xy_m[0]:.4f}, {nut_est.xy_m[1]:.4f}) m "
                    f"(yaw={nut_est.yaw_deg:.1f}°), 固定螺丝=({screw_est.xy_m[0]:.4f}, "
                    f"{screw_est.xy_m[1]:.4f}) m",
                    flush=True,
                )
            else:
                if mode == "real":
                    cam_stream = OrbbecCameraStream()
                    cam_stream.start()
                print(
                    "[感知] 正在从 Orbbec Gemini 305 采集全局 RGB-D 帧并识别螺母与固定螺丝 ...",
                    flush=True,
                )
                color_bgr, depth_m, _ = capture_rgbd_frame(stream=cam_stream)
                nut_est, screw_est, debug_info = detect_nut_and_screw_overview(
                    color_bgr,
                    depth_m,
                    t_base_flange,
                    handeye_yaml=handeye_path,
                )
                print(
                    f"[感知] 全局识别成功: 螺母=({nut_est.xy_m[0]:.4f}, {nut_est.xy_m[1]:.4f}) m "
                    f"(yaw={nut_est.yaw_deg:.1f}°, {nut_est.point_count} 点), "
                    f"固定螺丝=({screw_est.xy_m[0]:.4f}, {screw_est.xy_m[1]:.4f}) m "
                    f"(高度={screw_est.height_mm:.1f} mm, {screw_est.point_count} 点)",
                    flush=True,
                )

            if color_bgr is not None and save_vis_path is not None:
                vis_file = save_detection_overlay(
                    color_bgr, nut_est, screw_est, debug_info, save_vis_path
                )
                print(f"[感知] 已保存螺母与固定螺丝检测叠加图: {vis_file}", flush=True)

            sim_mode_label = "无头模式" if headless else "3D 可视化窗口"
            print(
                f"[规划] 开始 MuJoCo 数字孪生拧螺母预演 ({sim_mode_label}, "
                f"目标旋紧 {thread_turns:.2f} 圈) ...",
                flush=True,
            )

            twin_report = run_digital_twin_preflight(
                nut_est,
                screw_est,
                thread_turns=thread_turns,
                q_init=q_obs,
                calibration=calib,
                handeye_yaml=handeye_path,
                headless=headless,
            )
            if not twin_report.ok:
                reason_str = "；".join(twin_report.failures)
                raise SafetyGateError(f"数字孪生安全预检未通过，已拦截硬件执行：{reason_str}")

            print(
                f"[预演] 数字孪生验证通过: 旋紧 {twin_report.metrics['stroke_count']} 把 "
                f"({twin_report.metrics['rotated_deg']:.1f}°), "
                f"落座高度={twin_report.metrics['nut_z_mm']:.2f} mm, "
                f"同心误差={twin_report.metrics['xy_error_mm']:.2f} mm, "
                f"相机最低离桌={twin_report.min_camera_z_mm:.1f} mm, "
                f"最小防撞间距={twin_report.min_collision_dist_mm:.1f} mm",
                flush=True,
            )

            prog_hash = stream_sha256(twin_report.shadow_stream_rad)
            cfg_hash = hashlib.sha256(
                json.dumps(
                    {
                        "nut_pose": nut_est.as_dict(),
                        "screw_pose": screw_est.as_dict(),
                        "thread_turns": float(thread_turns),
                        "calibration_sha256": calib.sha256,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

            result_payload: dict[str, Any] = {
                "mode": mode,
                "robot": "xarm6",
                "gripper": "gripper_g2",
                "thread_turns": float(thread_turns),
                "serial_number": robot.serial_number,
                "calibration_sha256": calib.sha256,
                "config_sha256": cfg_hash,
                "program_sha256": prog_hash,
                "digital_twin": twin_report.as_dict(),
                "ok": True,
                "success": True,
                "completed": True,
            }

            if mode == "dry-run":
                _write_json_report(report_path, result_payload)
                return result_payload

            checker = make_screw_nut_collision_checker(demo.SCENE, calibration=calib)
            if mode == "sdk-sim":
                print("[SDK-Sim] 正在控制器内部仿真重放拧螺母轨迹（物理关节保持静止）...", flush=True)
                evidence = run_sdk_simulation(
                    robot._arm,
                    robot="xarm6",
                    calibration=calib,
                    shadow_joint_stream_rad=twin_report.shadow_stream_rad,
                    checker=checker,
                    preflight_ok=twin_report.ok,
                    cycles=cycles,
                    config_sha256=cfg_hash,
                    program_sha256=prog_hash,
                    step_dt_s=0.015,
                )
                ok = bool(twin_report.ok and evidence.passed)
                result_payload["sdk_sim"] = evidence.as_dict()
                result_payload["ok"] = ok
                result_payload["success"] = ok
                _write_json_report(report_path, result_payload)
                if not ok:
                    raise SafetyGateError(f"sdk-sim 校验未通过: {evidence.failures}")
                print(
                    f"[SDK-Sim] 验证通过: 采样点={evidence.samples}, "
                    f"最大关节误差={evidence.max_joint_error_rad * 1000:.3f} mrad, "
                    f"物理关节漂移={evidence.physical_joint_drift_rad * 1000:.4f} mrad",
                    flush=True,
                )
                return result_payload

            # Step 3: Real hardware execution with look-down refinement and closed-loop gates
            exec_log_path = DEFAULT_EXEC_LOG_PATH
            exec_log_path.parent.mkdir(parents=True, exist_ok=True)
            exec_log_path.write_text("", encoding="utf-8")

            def _log_real(msg: str) -> None:
                print(msg, flush=True)
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with exec_log_path.open("a", encoding="utf-8") as fp:
                    fp.write(f"[{ts}] {msg}\n")

            approval = authorize_real_motion(
                robot._arm,
                robot="xarm6",
                calibration=calib,
                preflight_ok=twin_report.ok,
                confirm_real=bool(confirm_real),
                cycles=cycles,
                config_sha256=cfg_hash,
                program_sha256=prog_hash,
            )
            robot.unlock_motion(approval)
            result_payload["confirm_real"] = True
            result_payload["completed"] = False
            result_payload["execution_log"] = str(exec_log_path)
            _log_real("[Real] 安全门已解锁 (单周期 MotionApproval)，开始真机闭环拧螺母执行 ...")

            robot.set_gripper_gap(
                REAL_PRE_GRASP_GAP_MM / 1000.0,
                speed_percent=REAL_OPEN_SPEED_PCT,
                force_percent=REAL_OPEN_FORCE_PCT,
            )
            init_open_gap_mm = _read_real_gripper_gap_mm(robot)
            _log_real(
                f"  [0/6] 预张开 Gripper G2 目标={REAL_PRE_GRASP_GAP_MM:.1f} mm, "
                f"实测开度={init_open_gap_mm:.1f} mm"
            )

            use_closed_loop_cam = nut_pose_override is None or screw_xy_override is None
            active_wps = twin_report.waypoints
            refined_poses_log: dict[str, Any] = {}

            if use_closed_loop_cam:
                # 1. Move camera directly over the nut for secondary vertical look-down measurement
                q_cam_nut = solve_camera_lookdown_q(
                    nut_est.xy_m,
                    z_cam_m=CAM_HOVER_Z_M,
                    q_seed=q_obs,
                    calibration=calib,
                    handeye_yaml=handeye_path,
                )
                _log_real(
                    f"  [1/6] 移动相机至螺母正上方观测位 (z_cam={CAM_HOVER_Z_M * 1000:.0f} mm) 进行抓取前精测 ..."
                )
                _stream_waypoints_on_real(robot, [q_cam_nut], soft_landing=True)
                robot.get_joint_positions()
                t_bf_nut = np.eye(4, dtype=np.float64)
                t_bf_nut[:3, :3] = robot.data.xmat[link6_id].reshape(3, 3)
                t_bf_nut[:3, 3] = robot.data.xpos[link6_id]
                nut_bgr, nut_depth, _ = capture_rgbd_frame(stream=cam_stream)
                refined_nut, nut_dbg = detect_nut_near(
                    nut_bgr,
                    nut_depth,
                    t_bf_nut,
                    nut_est.xy_m,
                    search_radius_m=0.080,
                    exclude_xy_m=screw_est.xy_m,
                    handeye_yaml=handeye_path,
                )
                if refined_nut is None:
                    _log_real(
                        f"  [闭环拦截] 螺母正上方复测未检测到螺母 (原因: {nut_dbg.get('reason')})，"
                        "放弃抓取并安全退回初始位姿！"
                    )
                    _stream_waypoints_on_real(robot, [q_obs], soft_landing=True)
                    q_end_abort = robot.get_joint_positions()
                    result_payload.update(
                        confirm_real=True,
                        aborted=True,
                        aborted_missing_nut=True,
                        abort_reason=str(nut_dbg.get("reason", "missing_nut")),
                        final_joint_error_rad=float(np.max(np.abs(q_end_abort - q_obs))),
                        ok=False,
                        success=False,
                        completed=False,
                    )
                    _write_json_report(report_path, result_payload)
                    return result_payload

                # 2. Move camera directly over the fixed screw for secondary vertical look-down measurement
                q_cam_screw = solve_camera_lookdown_q(
                    screw_est.xy_m,
                    z_cam_m=CAM_HOVER_Z_M,
                    q_seed=q_cam_nut,
                    calibration=calib,
                    handeye_yaml=handeye_path,
                )
                _log_real(
                    f"  [2/6] 移动相机至固定螺丝正上方观测位 (z_cam={CAM_HOVER_Z_M * 1000:.0f} mm) 进行套入前精测 ..."
                )
                _stream_waypoints_on_real(robot, [q_cam_screw], soft_landing=True)
                q_hover_screw = robot.get_joint_positions()
                t_bf_screw = np.eye(4, dtype=np.float64)
                t_bf_screw[:3, :3] = robot.data.xmat[link6_id].reshape(3, 3)
                t_bf_screw[:3, 3] = robot.data.xpos[link6_id]
                screw_bgr, screw_depth, _ = capture_rgbd_frame(stream=cam_stream)
                refined_screw, screw_dbg = detect_screw_near(
                    screw_bgr,
                    screw_depth,
                    t_bf_screw,
                    screw_est.xy_m,
                    search_radius_m=0.080,
                    exclude_xy_m=refined_nut.xy_m,
                    handeye_yaml=handeye_path,
                )
                if refined_screw is None:
                    _log_real(
                        f"  [闭环拦截] 固定螺丝正上方复测未检测到立柱螺丝 (原因: {screw_dbg.get('reason')})，"
                        "放弃执行并安全退回初始位姿！"
                    )
                    _stream_waypoints_on_real(robot, [q_obs], soft_landing=True)
                    q_end_abort = robot.get_joint_positions()
                    result_payload.update(
                        confirm_real=True,
                        aborted=True,
                        aborted_missing_screw=True,
                        abort_reason=str(screw_dbg.get("reason", "missing_screw")),
                        final_joint_error_rad=float(np.max(np.abs(q_end_abort - q_obs))),
                        ok=False,
                        success=False,
                        completed=False,
                    )
                    _write_json_report(report_path, result_payload)
                    return result_payload

                nut_dxy_mm = float(
                    np.linalg.norm(np.asarray(refined_nut.xy_m) - np.asarray(nut_est.xy_m)) * 1000.0
                )
                screw_dxy_mm = float(
                    np.linalg.norm(np.asarray(refined_screw.xy_m) - np.asarray(screw_est.xy_m))
                    * 1000.0
                )
                _log_real(
                    f"  [精测完成] 螺母精修=({refined_nut.xy_m[0]:.4f}, {refined_nut.xy_m[1]:.4f}) m "
                    f"(Δxy={nut_dxy_mm:.2f} mm, yaw={refined_nut.yaw_deg:.1f}°), "
                    f"螺丝精修=({refined_screw.xy_m[0]:.4f}, {refined_screw.xy_m[1]:.4f}) m "
                    f"(Δxy={screw_dxy_mm:.2f} mm, h={refined_screw.height_mm:.1f} mm)"
                )
                refined_poses_log = {
                    "nut_delta_xy_mm": round(nut_dxy_mm, 2),
                    "screw_delta_xy_mm": round(screw_dxy_mm, 2),
                    "refined_nut": refined_nut.as_dict(),
                    "refined_screw": refined_screw.as_dict(),
                }
                nut_est = refined_nut
                screw_est = refined_screw
                replanned_twin = replan_screw_nut_from_refined(
                    nut_est,
                    screw_est,
                    thread_turns=thread_turns,
                    q_start=q_hover_screw,
                    calibration=calib,
                    handeye_yaml=handeye_path,
                )
                active_wps = replanned_twin.waypoints

            _log_real("  [3/6] 软着陆平滑下探至螺母抓取位并闭合 Gripper G2 ...")
            _stream_waypoints_on_real(robot, active_wps.approach_waypoints, soft_landing=True)
            robot.set_gripper_gap(
                REAL_GRASP_GAP_MM / 1000.0,
                speed_percent=REAL_GRASP_SPEED_PCT,
                force_percent=REAL_GRASP_FORCE_PCT,
            )
            time.sleep(0.35)
            meas_gap_mm = _read_real_gripper_gap_mm(robot)
            measured_gaps_mm = [float(meas_gap_mm)]
            measured_open_gaps_mm: list[float] = [float(init_open_gap_mm)]
            stroke_logs: list[dict[str, Any]] = []
            result_payload["measured_grasp_gaps_mm"] = measured_gaps_mm
            result_payload["measured_open_gaps_mm"] = measured_open_gaps_mm
            result_payload["stroke_logs"] = stroke_logs
            result_payload["refined_poses"] = refined_poses_log
            _log_real(
                f"  [抓取校验] 初次夹持目标={REAL_GRASP_GAP_MM:.1f} mm, 实测开度={meas_gap_mm:.1f} mm"
            )
            if meas_gap_mm < MIN_VALID_NUT_GRASP_GAP_MM:
                _log_real(
                    f"  [闭环拦截] 实测夹持开度过小 ({meas_gap_mm:.1f} mm < "
                    f"{MIN_VALID_NUT_GRASP_GAP_MM:.1f} mm，疑似空抓)，立即松爪并安全退回初始位姿！"
                )
                robot.set_gripper_gap(
                    REAL_PRE_GRASP_GAP_MM / 1000.0,
                    speed_percent=REAL_OPEN_SPEED_PCT,
                    force_percent=REAL_OPEN_FORCE_PCT,
                )
                time.sleep(0.25)
                _stream_waypoints_on_real(
                    robot,
                    [active_wps.approach_waypoints[0], q_obs],
                    soft_landing=True,
                )
                q_end_abort = robot.get_joint_positions()
                result_payload.update(
                    confirm_real=True,
                    aborted=True,
                    aborted_empty_grasp=True,
                    abort_reason="empty_grasp_nut",
                    measured_grasp_gaps_mm=measured_gaps_mm,
                    measured_open_gaps_mm=measured_open_gaps_mm,
                    refined_poses=refined_poses_log,
                    final_joint_error_rad=float(np.max(np.abs(q_end_abort - q_obs))),
                    ok=False,
                    success=False,
                    completed=False,
                )
                _write_json_report(report_path, result_payload)
                return result_payload

            _log_real("  [4/6] 抬起螺母、平移对准固定螺丝并软着陆套入 M12 螺杆顶端 ...")
            _stream_waypoints_on_real(robot, active_wps.lift_align_waypoints, soft_landing=True)

            stroke_queue: list[list[np.ndarray]] = [
                list(wps) for wps in active_wps.stroke_waypoints
            ]
            rewind_queue: list[list[np.ndarray]] = [
                list(wps) for wps in active_wps.rewind_waypoints
            ]
            nominal_num_strokes = len(stroke_queue)
            table_z_m = float(screw_est.table_z_m)
            min_tcp_z_m = table_z_m + demo.THREAD_END_TCP_Z_M
            max_tcp_z_m = table_z_m + demo.CAPTURE_SCREW_TCP_Z_M + 0.0020
            z_adapt_offset_m = 0.0
            z_adapt_count = 0
            extra_strokes_used = 0
            z_adapt_events: list[dict[str, Any]] = []

            # Template 1.5-turn stroke and rewind for adaptive continuation strokes
            template_stroke_wps = (
                list(active_wps.stroke_waypoints[-2])
                if len(active_wps.stroke_waypoints) >= 2
                else list(active_wps.stroke_waypoints[-1])
            )
            template_rewind_wps = (
                list(active_wps.rewind_waypoints[-1]) if active_wps.rewind_waypoints else []
            )
            template_stroke_start_z = _get_waypoint_tcp_z(checker, template_stroke_wps[0])
            template_stroke_end_z = _get_waypoint_tcp_z(checker, template_stroke_wps[-1])
            template_rewind_z = (
                _get_waypoint_tcp_z(checker, template_rewind_wps[-1])
                if template_rewind_wps
                else template_stroke_end_z
            )

            _log_real(
                f"  [5/6] 执行 {nominal_num_strokes} 把螺旋软着陆旋紧与倒手回转 "
                f"(松爪目标={REAL_REGRIP_GAP_MM:.1f} mm, 最小放行开度={MIN_VALID_NUT_OPEN_GAP_MM:.1f} mm, "
                f"六角体闭环开度上限={MAX_HEX_FLAT_GAP_MM:.1f} mm, 夹持力={REAL_GRASP_FORCE_PCT:.1f}%) ..."
            )
            s_idx = 0
            while s_idx < len(stroke_queue):
                base_stroke_wps = stroke_queue[s_idx]
                cur_stroke_wps = _shift_waypoints_cartesian_z(
                    checker,
                    base_stroke_wps,
                    z_adapt_offset_m,
                    min_tcp_z_m=min_tcp_z_m,
                    max_tcp_z_m=max_tcp_z_m,
                )
                q_before_stroke = robot.get_joint_positions()
                tcp_z_start_mm = (_get_waypoint_tcp_z(checker, q_before_stroke) - table_z_m) * 1000.0
                _stream_waypoints_on_real(robot, cur_stroke_wps, soft_landing=True)
                q_after_stroke = robot.get_joint_positions()
                tcp_z_end_mm = (_get_waypoint_tcp_z(checker, q_after_stroke) - table_z_m) * 1000.0
                j6_start_deg = float(np.degrees(q_before_stroke[5]))
                j6_end_deg = float(np.degrees(q_after_stroke[5]))
                stroke_entry: dict[str, Any] = {
                    "stroke_index": s_idx + 1,
                    "is_adaptive_extra": bool(s_idx >= nominal_num_strokes),
                    "j6_start_deg": round(j6_start_deg, 2),
                    "j6_end_deg": round(j6_end_deg, 2),
                    "tcp_z_start_mm": round(tcp_z_start_mm, 2),
                    "tcp_z_end_mm": round(tcp_z_end_mm, 2),
                    "z_adapt_offset_mm": round(z_adapt_offset_m * 1000.0, 2),
                }

                # Check if closed-loop Z backoff left TCP Z above the bottom target after the final queued stroke
                if (
                    s_idx == len(stroke_queue) - 1
                    and z_adapt_count > 0
                    and tcp_z_end_mm > (demo.THREAD_END_TCP_Z_M * 1000.0 + 1.2)
                    and extra_strokes_used < MAX_ADAPTIVE_EXTRA_STROKES
                    and template_rewind_wps
                ):
                    extra_strokes_used += 1
                    base_end_z = _get_waypoint_tcp_z(checker, base_stroke_wps[-1])
                    extra_rewind_base = _shift_waypoints_cartesian_z(
                        checker,
                        template_rewind_wps,
                        base_end_z - template_rewind_z,
                    )
                    extra_stroke_base = _shift_waypoints_cartesian_z(
                        checker,
                        template_stroke_wps,
                        base_end_z - template_stroke_start_z,
                    )
                    rewind_queue.append(extra_rewind_base)
                    stroke_queue.append(extra_stroke_base)
                    _log_real(
                        f"    [自适应补拧 #{extra_strokes_used}] 第 {s_idx + 1} 把结束时 "
                        f"TCP Z={tcp_z_end_mm:.2f} mm > 落座目标 {demo.THREAD_END_TCP_Z_M * 1000.0:.2f} mm "
                        f"(累计 Z 补偿 +{z_adapt_offset_m * 1000.0:.1f} mm)，自动追加 1.5 圈补拧行程 ..."
                    )

                if s_idx < len(rewind_queue):
                    robot.set_gripper_gap(
                        REAL_REGRIP_GAP_MM / 1000.0,
                        speed_percent=REAL_OPEN_SPEED_PCT,
                        force_percent=REAL_OPEN_FORCE_PCT,
                    )
                    time.sleep(0.30)
                    open_gap_mm = _read_real_gripper_gap_mm(robot)
                    if open_gap_mm < MIN_VALID_NUT_OPEN_GAP_MM:
                        _log_real(
                            f"    [松爪重试] 第 {s_idx + 1} 把初次松爪实测开度={open_gap_mm:.1f} mm "
                            f"< {MIN_VALID_NUT_OPEN_GAP_MM:.1f} mm，正在加大力矩二次张开 ..."
                        )
                        robot.set_gripper_gap(
                            REAL_REGRIP_GAP_MM / 1000.0,
                            speed_percent=75.0,
                            force_percent=75.0,
                        )
                        time.sleep(0.35)
                        open_gap_mm = _read_real_gripper_gap_mm(robot)
                    measured_open_gaps_mm.append(float(open_gap_mm))
                    stroke_entry["open_gap_mm"] = round(float(open_gap_mm), 2)
                    if open_gap_mm < MIN_VALID_NUT_OPEN_GAP_MM:
                        stroke_logs.append(stroke_entry)
                        raise SafetyGateError(
                            f"第 {s_idx + 1} 把倒手松爪开度不足 ({open_gap_mm:.1f} mm < "
                            f"{MIN_VALID_NUT_OPEN_GAP_MM:.1f} mm)，已拦截回转防止将螺母反向带松"
                        )
                    cur_rewind_wps = _shift_waypoints_cartesian_z(
                        checker,
                        rewind_queue[s_idx],
                        z_adapt_offset_m,
                        min_tcp_z_m=min_tcp_z_m,
                        max_tcp_z_m=max_tcp_z_m,
                    )
                    _stream_waypoints_on_real(
                        robot, cur_rewind_wps, soft_landing=True
                    )
                    q_after_rewind = robot.get_joint_positions()
                    stroke_entry["rewind_j6_deg"] = round(float(np.degrees(q_after_rewind[5])), 2)
                    robot.set_gripper_gap(
                        REAL_GRASP_GAP_MM / 1000.0,
                        speed_percent=50.0,
                        force_percent=REAL_GRASP_FORCE_PCT,
                    )
                    time.sleep(0.30)
                    regrip_gap = _read_real_gripper_gap_mm(robot)
                    initial_regrip_gap = float(regrip_gap)
                    stroke_entry["initial_close_gap_mm"] = round(initial_regrip_gap, 2)

                    # Tactile Z-axis closed-loop adaptation: if regrip_gap > MAX_HEX_FLAT_GAP_MM (21.2 mm),
                    # the fingertips dropped below the 3.8 mm hex-to-flange boundary and hit the 29 mm flange.
                    adapt_retries = 0
                    while (
                        regrip_gap > MAX_HEX_FLAT_GAP_MM
                        and adapt_retries < MAX_Z_ADAPT_RETRIES_PER_STROKE
                    ):
                        adapt_retries += 1
                        z_adapt_count += 1
                        dz_up = float(
                            np.clip(
                                0.0035 + 0.0004 * (regrip_gap - MAX_HEX_FLAT_GAP_MM),
                                Z_ADAPT_MIN_STEP_M,
                                Z_ADAPT_MAX_STEP_M,
                            )
                        )
                        z_adapt_offset_m += dz_up
                        q_lift_target = _shift_waypoints_cartesian_z(
                            checker,
                            [rewind_queue[s_idx][-1]],
                            z_adapt_offset_m,
                            min_tcp_z_m=min_tcp_z_m,
                            max_tcp_z_m=max_tcp_z_m,
                        )[0]
                        lifted_tcp_z_mm = (
                            _get_waypoint_tcp_z(checker, q_lift_target) - table_z_m
                        ) * 1000.0
                        _log_real(
                            f"    [Z轴闭环自适应 #{z_adapt_count}] 第 {s_idx + 1} 把重夹实测开度="
                            f"{regrip_gap:.1f} mm > {MAX_HEX_FLAT_GAP_MM:.1f} mm (触碰 29 mm 底法兰，"
                            f"TCP Z 超前螺母下降)，松爪并上抬 TCP Z +{dz_up * 1000.0:.1f} mm 至 "
                            f"{lifted_tcp_z_mm:.2f} mm (累计补偿 +{z_adapt_offset_m * 1000.0:.1f} mm) 重新夹持上层六角体 ..."
                        )
                        robot.set_gripper_gap(
                            REAL_REGRIP_GAP_MM / 1000.0,
                            speed_percent=REAL_OPEN_SPEED_PCT,
                            force_percent=REAL_OPEN_FORCE_PCT,
                        )
                        time.sleep(0.30)
                        _stream_waypoints_on_real(robot, [q_lift_target], soft_landing=True)
                        robot.set_gripper_gap(
                            REAL_GRASP_GAP_MM / 1000.0,
                            speed_percent=50.0,
                            force_percent=REAL_GRASP_FORCE_PCT,
                        )
                        time.sleep(0.30)
                        regrip_gap = _read_real_gripper_gap_mm(robot)
                        z_adapt_events.append(
                            {
                                "stroke_index": s_idx + 1,
                                "attempt": adapt_retries,
                                "initial_gap_mm": round(initial_regrip_gap, 2),
                                "dz_up_mm": round(dz_up * 1000.0, 2),
                                "cumulative_z_offset_mm": round(z_adapt_offset_m * 1000.0, 2),
                                "lifted_tcp_z_mm": round(lifted_tcp_z_mm, 2),
                                "regrip_gap_after_mm": round(float(regrip_gap), 2),
                            }
                        )

                    measured_gaps_mm.append(float(regrip_gap))
                    stroke_entry["close_gap_mm"] = round(float(regrip_gap), 2)
                    stroke_entry["z_adapt_retries"] = adapt_retries
                    stroke_logs.append(stroke_entry)
                    _log_real(
                        f"    第 {s_idx + 1}/{len(stroke_queue)} 把: "
                        f"J6 [{j6_start_deg:.1f}° -> {j6_end_deg:.1f}°], "
                        f"TCP Z [{tcp_z_start_mm:.2f} -> {tcp_z_end_mm:.2f} mm], "
                        f"松爪开度={open_gap_mm:.1f} mm, 回转至 J6={stroke_entry['rewind_j6_deg']:.1f}°, "
                        f"重夹开度={regrip_gap:.1f} mm (初测={initial_regrip_gap:.1f} mm, "
                        f"累计Z补偿=+{z_adapt_offset_m * 1000.0:.1f} mm)"
                    )
                    if regrip_gap < MIN_VALID_NUT_GRASP_GAP_MM:
                        raise SafetyGateError(
                            f"第 {s_idx + 1} 把倒手重夹后开度过小 ({regrip_gap:.1f} mm)，疑似脱手"
                        )
                else:
                    stroke_logs.append(stroke_entry)
                    _log_real(
                        f"    第 {s_idx + 1}/{len(stroke_queue)} 把 (末把收尾): "
                        f"J6 [{j6_start_deg:.1f}° -> {j6_end_deg:.1f}°], "
                        f"TCP Z [{tcp_z_start_mm:.2f} -> {tcp_z_end_mm:.2f} mm]"
                    )
                s_idx += 1

            _log_real("  [6/6] 释放螺母并平滑撤离回初始观测位姿 ...")
            robot.set_gripper_gap(
                REAL_PRE_GRASP_GAP_MM / 1000.0,
                speed_percent=REAL_OPEN_SPEED_PCT,
                force_percent=REAL_OPEN_FORCE_PCT,
            )
            time.sleep(0.25)
            final_open_gap_mm = _read_real_gripper_gap_mm(robot)
            measured_open_gaps_mm.append(float(final_open_gap_mm))
            retreat_chain = list(active_wps.retreat_waypoints)
            if not retreat_chain or float(np.max(np.abs(retreat_chain[-1] - q_obs))) > 1e-4:
                retreat_chain.append(q_obs.copy())
            _stream_waypoints_on_real(robot, retreat_chain, soft_landing=False)

            q_final = robot.get_joint_positions()
            final_err_rad = float(np.max(np.abs(q_final - q_obs)))

            post_ok = True
            if post_verify:
                q_cam_after = solve_camera_lookdown_q(
                    screw_est.xy_m,
                    z_cam_m=CAM_HOVER_Z_M,
                    q_seed=q_obs,
                    calibration=calib,
                    handeye_yaml=handeye_path,
                )
                _log_real("[闭环确认] 正在移动相机至固定螺丝正上方拍照复核螺母旋紧状态 ...")
                _stream_waypoints_on_real(robot, [q_cam_after], soft_landing=True)
                robot.get_joint_positions()
                t_bf_after = np.eye(4, dtype=np.float64)
                t_bf_after[:3, :3] = robot.data.xmat[link6_id].reshape(3, 3)
                t_bf_after[:3, 3] = robot.data.xpos[link6_id]
                after_bgr, after_depth, _ = capture_rgbd_frame(stream=cam_stream)
                expected_bottom_mm = float(
                    twin_report.metrics.get("expected_nut_z_mm", demo.SCREW_HEAD_LEN_M * 1000.0)
                )
                post_check, after_dbg = verify_threaded_nut(
                    after_bgr,
                    after_depth,
                    t_bf_after,
                    screw_est.xy_m,
                    initial_nut_xy_m=nut_est.xy_m,
                    expected_nut_bottom_z_mm=expected_bottom_mm,
                    handeye_yaml=handeye_path,
                    table_z_m=screw_est.table_z_m,
                )
                if save_after_vis_path is not None:
                    after_file = save_post_thread_overlay(
                        after_bgr, post_check, after_dbg, save_after_vis_path
                    )
                    _log_real(f"[闭环确认] 已保存拧螺母后验证图像: {after_file}")
                _stream_waypoints_on_real(robot, [q_obs], soft_landing=False)
                q_final = robot.get_joint_positions()
                final_err_rad = float(np.max(np.abs(q_final - q_obs)))
                result_payload["post_thread_verification"] = post_check.as_dict()
                post_ok = bool(post_check.ok)
                status_tag = "通过" if post_ok else f"未通过 ({post_check.reason})"
                _log_real(
                    f"[闭环确认] 视觉复核{status_tag}: 同心偏差={post_check.xy_error_mm:.1f} mm, "
                    f"螺母顶高={post_check.nut_top_height_mm:.1f} mm, "
                    f"螺母环点云={post_check.nut_ring_points}"
                )

            overall_ok = bool(final_err_rad < 0.08 and post_ok)
            result_payload.update(
                confirm_real=True,
                measured_grasp_gaps_mm=measured_gaps_mm,
                measured_open_gaps_mm=measured_open_gaps_mm,
                stroke_logs=stroke_logs,
                z_adapt_count=z_adapt_count,
                z_adapt_total_mm=round(z_adapt_offset_m * 1000.0, 2),
                adaptive_extra_strokes=extra_strokes_used,
                z_adapt_events=z_adapt_events,
                refined_poses=refined_poses_log,
                final_joint_error_rad=final_err_rad,
                ok=overall_ok,
                success=overall_ok,
                completed=True,
            )
            _write_json_report(report_path, result_payload)
            if not post_ok:
                raise SafetyGateError(
                    f"拧螺母后相机闭环确认未通过: {result_payload['post_thread_verification']['reason']}"
                )
            return result_payload
        finally:
            if mode == "real" and report_path:
                _write_json_report(report_path, result_payload)
            if cam_stream is not None:
                cam_stream.close()


def _write_json_report(report_path: str | Path | None, payload: dict[str, Any]) -> None:
    if not report_path:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
