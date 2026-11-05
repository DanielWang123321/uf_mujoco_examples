"""Sim-to-Real digital twin preflight, trajectory verification, and hardware execution for pick_cup."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ufactory_mujoco"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perception import (
    DEFAULT_HANDEYE_YAML,
    WORKSPACE_BOUNDS_M,
    CupPoseEstimate,
    capture_rgbd_frame,
    detect_sideways_cup,
    load_handeye_calibration,
    pose_estimate_from_manual,
    save_detection_overlay,
    save_post_place_overlay,
    validate_xy_in_workspace,
    verify_upright_cup,
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
from ufactory_mujoco.trajectory import plan_ptp_quintic
from ufactory_mujoco.xarm_robot import XArmRobot

DEFAULT_ROBOT_IP = "192.168.1.60"
DEFAULT_CALIBRATION_YAML = (
    Path(__file__).resolve().parents[3]
    / "ufactory_mujoco"
    / "calibrations"
    / "xarm6_kinematics_L56A01.yaml"
)
DEFAULT_REAL_TARGET_XY = np.array([0.38, -0.18], dtype=np.float64)
FALLBACK_REAL_TARGET_XY = np.array([0.38, 0.04], dtype=np.float64)
MIN_PICK_PLACE_SEPARATION_M = 0.110
MIN_CAMERA_TABLE_CLEARANCE_M = 0.095
REAL_GRASP_GAP_MM = 52.0
REAL_GRASP_SPEED_PCT = 20.0
REAL_GRASP_FORCE_PCT = 15.0
REAL_RELEASE_GAP_MM = 84.0


@dataclass
class TwinPhaseWaypoints:
    """Recorded joint waypoints for each manipulation stage from the MuJoCo digital twin."""

    q_init: np.ndarray
    approach_waypoints: list[np.ndarray] = field(default_factory=list)
    lift_flip_place_waypoints: list[np.ndarray] = field(default_factory=list)
    retreat_waypoints: list[np.ndarray] = field(default_factory=list)
    grasp_gap_mm: float = REAL_GRASP_GAP_MM


@dataclass(frozen=True)
class DigitalTwinReport:
    """Comprehensive verification report produced by the headless MuJoCo digital twin rollout."""

    ok: bool
    cup_pose: CupPoseEstimate
    target_xy_m: tuple[float, float]
    metrics: dict[str, float]
    min_camera_z_mm: float
    min_collision_dist_mm: float
    max_joint_step_rad: float
    waypoint_count: int
    shadow_stream_rad: np.ndarray
    phases: TwinPhaseWaypoints
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "cup_pose": self.cup_pose.as_dict(),
            "target_xy_m": list(self.target_xy_m),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "min_camera_z_mm": float(self.min_camera_z_mm),
            "min_collision_dist_mm": float(self.min_collision_dist_mm),
            "max_joint_step_rad": float(self.max_joint_step_rad),
            "waypoint_count": int(self.waypoint_count),
            "shadow_samples": len(self.shadow_stream_rad),
            "failures": list(self.failures),
        }


def make_cup_collision_checker(
    scene_path: str | Path,
    calibration: KinematicsCalibration | None = None,
) -> CollisionChecker:
    """Create a CollisionChecker that checks self-collision and arm/gripper vs table collision."""
    checker = CollisionChecker(
        scene_path,
        safety_margin=0.0,
        query_margin=0.01,
        ignore_pairs=[("world", "link_base"), ("world", "link1")],
        ignore_bodies=("paper_cup",),
    )
    if calibration is not None:
        apply_calibration_to_model(checker.model, calibration, checker.data)
    return checker


def select_safe_target_xy(
    cup_pose: CupPoseEstimate,
    requested_target_xy: tuple[float, float] | np.ndarray | None = None,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> np.ndarray:
    """Choose and validate a safe upright placement target XY separated from the initial cup."""
    cup_mid = np.asarray(cup_pose.cup_center_xy_m, dtype=np.float64)
    cup_bot = np.asarray(cup_pose.cup_bottom_xy_m, dtype=np.float64)
    if requested_target_xy is not None:
        target = np.asarray(requested_target_xy, dtype=np.float64).reshape(2)
        validate_xy_in_workspace(target, "指定的目标放置位置", bounds)
        if (
            np.linalg.norm(target - cup_mid) < MIN_PICK_PLACE_SEPARATION_M
            or np.linalg.norm(target - cup_bot) < MIN_PICK_PLACE_SEPARATION_M
        ):
            raise SafetyGateError(
                f"目标放置位置 ({target[0]:.3f}, {target[1]:.3f}) m 与纸杯初始位置距离过近 "
                f"(< {MIN_PICK_PLACE_SEPARATION_M * 1000:.0f} mm)，存在碰撞风险"
            )
        return target

    for candidate in (DEFAULT_REAL_TARGET_XY, FALLBACK_REAL_TARGET_XY):
        if (
            np.linalg.norm(candidate - cup_mid) >= MIN_PICK_PLACE_SEPARATION_M
            and np.linalg.norm(candidate - cup_bot) >= MIN_PICK_PLACE_SEPARATION_M
        ):
            validate_xy_in_workspace(candidate, "默认目标放置位置", bounds)
            return candidate.copy()
    raise SafetyGateError("默认候选放置点均与当前纸杯位置重叠，请使用 --target-xy 指定净空目标点")


def _interpolate_waypoint_chain(
    q_start: np.ndarray,
    waypoints: list[np.ndarray],
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.65,
) -> np.ndarray:
    """Convert a phase's waypoint list into a 50 Hz velocity-bounded joint stream (N, 6)."""
    chain: list[np.ndarray] = [np.asarray(q_start, dtype=np.float64).reshape(-1).copy()]
    for q_wp in waypoints:
        q_arr = np.asarray(q_wp, dtype=np.float64).reshape(-1)
        if float(np.max(np.abs(q_arr - chain[-1]))) > 1e-5:
            chain.append(q_arr.copy())

    if len(chain) == 1:
        return chain[0].reshape(1, -1)

    chunks: list[np.ndarray] = []
    for i in range(len(chain) - 1):
        q0, q1 = chain[i], chain[i + 1]
        disp = float(np.max(np.abs(q1 - q0)))
        if disp > 0.30:
            # Point-to-point deploy or return move: smooth quintic profile
            duration = max(1.5, disp * 1.875 / max_vel_rad_s)
            traj = plan_ptp_quintic(q0, q1, duration=duration, dt=dt)
            seg = traj.q
        else:
            # Consecutive fine Cartesian IK waypoints
            n_sub = max(1, int(np.ceil(disp / (max_vel_rad_s * dt))))
            alphas = np.linspace(0.0, 1.0, n_sub + 1, dtype=np.float64)
            seg = q0[None, :] + alphas[:, None] * (q1 - q0)[None, :]
        chunks.append(seg if not chunks else seg[1:])
    return np.vstack(chunks)


def _build_dense_shadow_stream(
    phases: TwinPhaseWaypoints,
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.65,
) -> np.ndarray:
    """Concatenate recorded digital twin phases into a continuous trajectory array (N, 6)."""
    s1 = _interpolate_waypoint_chain(
        phases.q_init, phases.approach_waypoints, dt=dt, max_vel_rad_s=max_vel_rad_s
    )
    s2 = _interpolate_waypoint_chain(
        s1[-1], phases.lift_flip_place_waypoints, dt=dt, max_vel_rad_s=max_vel_rad_s
    )
    s3 = _interpolate_waypoint_chain(
        s2[-1], phases.retreat_waypoints, dt=dt, max_vel_rad_s=max_vel_rad_s
    )
    return np.vstack([s1, s2[1:], s3[1:]])


def run_digital_twin_preflight(
    cup_pose: CupPoseEstimate,
    target_xy_m: np.ndarray,
    *,
    q_init: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    headless: bool = False,
) -> DigitalTwinReport:
    """Execute the full pick, flip, and upright placement in the calibrated MuJoCo digital twin."""
    import pick_cup as demo

    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = make_cup_collision_checker(demo.SCENE, calibration=calibration)

    q_start = np.asarray(q_init, dtype=np.float64).reshape(6).copy()
    phases = TwinPhaseWaypoints(q_init=q_start)
    failures: list[str] = []
    min_cam_z_m = float("inf")
    min_col_dist_m = float("inf")
    metrics: dict[str, float] = {}

    with demo.CupRobot(viewer=not headless, calibration=calibration) as sim_robot:
        link6_id = mujoco.mj_name2id(sim_robot.model, mujoco.mjtObj.mjOBJ_BODY, "link6")

        def update_camera_clearance() -> None:
            nonlocal min_cam_z_m
            t_flange = np.eye(4, dtype=np.float64)
            t_flange[:3, :3] = sim_robot.data.xmat[link6_id].reshape(3, 3)
            t_flange[:3, 3] = sim_robot.data.xpos[link6_id]
            t_cam = t_flange @ x_flange_cam
            min_cam_z_m = min(min_cam_z_m, float(t_cam[2, 3]) - cup_pose.table_z_m)

        orig_step = sim_robot._step

        def monitored_step() -> None:
            orig_step()
            update_camera_clearance()

        sim_robot._step = monitored_step  # type: ignore[method-assign]

        try:
            metrics = demo.run_pick_cup(
                sim_robot,
                cup_pose=cup_pose,
                target_xy=target_xy_m,
                q_init=q_start,
                recorded_phases=phases,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"数字孪生仿真预演失败: {exc}")

    # Append final return-to-observation waypoint at the end of retreat_waypoints
    if not failures:
        phases.retreat_waypoints.append(q_start.copy())

    shadow_stream = _build_dense_shadow_stream(phases)
    max_step_rad = (
        float(np.max(np.abs(np.diff(shadow_stream, axis=0))))
        if len(shadow_stream) > 1
        else 0.0
    )

    # Check every sample in shadow_stream for joint limits, collision, and camera height
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
        cam_clear = float(t_cam[2, 3]) - cup_pose.table_z_m
        min_cam_z_m = min(min_cam_z_m, cam_clear)

    if min_cam_z_m < MIN_CAMERA_TABLE_CLEARANCE_M:
        failures.append(
            f"腕部相机最低离桌高度 {min_cam_z_m * 1000:.1f} mm 低于安全阈值 "
            f"{MIN_CAMERA_TABLE_CLEARANCE_M * 1000:.1f} mm"
        )
    if max_step_rad > 0.15:
        failures.append(f"相邻轨迹步关节角跨度过大 ({max_step_rad:.4f} rad > 0.15 rad)")

    total_wps = (
        len(phases.approach_waypoints)
        + len(phases.lift_flip_place_waypoints)
        + len(phases.retreat_waypoints)
    )
    return DigitalTwinReport(
        ok=len(failures) == 0,
        cup_pose=cup_pose,
        target_xy_m=(float(target_xy_m[0]), float(target_xy_m[1])),
        metrics=metrics,
        min_camera_z_mm=min_cam_z_m * 1000.0,
        min_collision_dist_mm=min_col_dist_m * 1000.0,
        max_joint_step_rad=max_step_rad,
        waypoint_count=total_wps,
        shadow_stream_rad=shadow_stream,
        phases=phases,
        failures=tuple(failures),
    )


def _stream_waypoints_on_real(
    robot: XArmRobot,
    waypoints: list[np.ndarray],
    *,
    max_vel_rad_s: float = 0.65,
    dt: float = 0.02,
) -> None:
    """Stream a sequence of joint waypoints smoothly in servo_j mode on the real xArm6."""
    if not waypoints:
        return
    q_cur = robot.get_joint_positions()
    stream = _interpolate_waypoint_chain(
        q_cur, waypoints, dt=dt, max_vel_rad_s=max_vel_rad_s
    )
    for q_cmd in stream[1:]:
        robot.set_joint_positions(q_cmd)
        time.sleep(dt)
    robot.settle(0.25)
    robot.get_joint_positions()


def run_sim2real_pipeline(
    *,
    mode: str,
    ip: str = DEFAULT_ROBOT_IP,
    calibration_path: str | Path = DEFAULT_CALIBRATION_YAML,
    handeye_path: str | Path = DEFAULT_HANDEYE_YAML,
    cup_pose_override: tuple[float, float, float] | None = None,
    target_xy_override: tuple[float, float] | None = None,
    headless: bool = False,
    confirm_real: bool = False,
    cycles: int = 1,
    post_verify: bool = True,
    save_vis_path: str | Path | None = None,
    save_after_vis_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the end-to-end Sim-to-Real pipeline (dry-run, sdk-sim, or real)."""
    import pick_cup as demo

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

        # Step 1: Acquire cup 3D pose (from Eye-in-Hand Gemini 305 or manual override)
        if cup_pose_override is not None:
            cup_pose = pose_estimate_from_manual(
                cup_pose_override[0], cup_pose_override[1], cup_pose_override[2]
            )
            print(
                f"[感知] 使用手动指定纸杯位姿: 杯底=({cup_pose.cup_bottom_xy_m[0]:.3f}, "
                f"{cup_pose.cup_bottom_xy_m[1]:.3f}) m, 偏航角={cup_pose.yaw_deg:.1f}°",
                flush=True,
            )
        else:
            print("[感知] 正在从 Orbbec Gemini 305 采集对齐 RGB-D 帧并估计纸杯 6D 位姿 ...", flush=True)
            color_bgr, depth_m, _ = capture_rgbd_frame()
            cup_pose, debug_info = detect_sideways_cup(
                color_bgr,
                depth_m,
                t_base_flange,
                handeye_yaml=handeye_path,
            )
            if save_vis_path is not None:
                vis_file = save_detection_overlay(color_bgr, cup_pose, debug_info, save_vis_path)
                print(f"[感知] 已保存视觉检测叠加图: {vis_file}", flush=True)
            print(
                f"[感知] 识别成功: 杯底=({cup_pose.cup_bottom_xy_m[0]:.3f}, "
                f"{cup_pose.cup_bottom_xy_m[1]:.3f}) m, 杯口=({cup_pose.cup_top_xy_m[0]:.3f}, "
                f"{cup_pose.cup_top_xy_m[1]:.3f}) m, 偏航角={cup_pose.yaw_deg:.1f}°, "
                f"局部桌面 Z={cup_pose.table_z_m * 1000:.1f} mm ({cup_pose.point_count} 点)",
                flush=True,
            )

        target_xy = select_safe_target_xy(cup_pose, target_xy_override)
        sim_mode_label = "无头模式" if headless else "3D 可视化窗口"
        print(
            f"[规划] 目标竖放位置: ({target_xy[0]:.3f}, {target_xy[1]:.3f}) m，"
            f"开始 MuJoCo 数字孪生预演 ({sim_mode_label}) ...",
            flush=True,
        )

        # Step 2: Run full physics rollout in MuJoCo digital twin (visualized by default)
        twin_report = run_digital_twin_preflight(
            cup_pose,
            target_xy,
            q_init=q_obs,
            calibration=calib,
            handeye_yaml=handeye_path,
            headless=headless,
        )
        if not twin_report.ok:
            reason_str = "；".join(twin_report.failures)
            raise SafetyGateError(f"数字孪生安全预检未通过，已拦截硬件执行：{reason_str}")

        print(
            f"[预演] 数字孪生验证通过: 模拟放置误差={twin_report.metrics['error_mm']:.2f} mm, "
            f"倾角={twin_report.metrics['tilt_deg']:.2f}°, "
            f"相机最低离桌={twin_report.min_camera_z_mm:.1f} mm, "
            f"最小防撞间距={twin_report.min_collision_dist_mm:.1f} mm",
            flush=True,
        )

        prog_hash = stream_sha256(twin_report.shadow_stream_rad)
        cfg_hash = hashlib.sha256(
            json.dumps(
                {
                    "cup_pose": cup_pose.as_dict(),
                    "target_xy": target_xy.tolist(),
                    "calibration_sha256": calib.sha256,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        result_payload: dict[str, Any] = {
            "mode": mode,
            "robot": "xarm6",
            "gripper": "gripper_g2",
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

        checker = make_cup_collision_checker(demo.SCENE, calibration=calib)
        if mode == "sdk-sim":
            print("[SDK-Sim] 正在控制器内部仿真重放轨迹（物理关节保持静止）...", flush=True)
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

        # Step 3: Real hardware execution (requires --confirm-real and MotionApproval)
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
        print("[Real] 安全门已解锁 (单周期 MotionApproval)，开始真机执行 ...", flush=True)

        print("  1/5 张开 Gripper G2 并移动至侧卧纸杯抓取位姿 ...", flush=True)
        robot.set_gripper_gap(
            REAL_RELEASE_GAP_MM / 1000.0,
            speed_percent=50.0,
            force_percent=50.0,
        )
        _stream_waypoints_on_real(robot, twin_report.phases.approach_waypoints)

        grasp_gap_mm = float(np.clip(twin_report.phases.grasp_gap_mm, 50.0, 55.0))
        print(
            f"  2/5 柔顺闭合 Gripper G2 (目标间隙={grasp_gap_mm:.1f} mm, "
            f"速度={REAL_GRASP_SPEED_PCT:.0f}%, 力控={REAL_GRASP_FORCE_PCT:.0f}%) ...",
            flush=True,
        )
        robot.set_gripper_gap(
            grasp_gap_mm / 1000.0,
            speed_percent=REAL_GRASP_SPEED_PCT,
            force_percent=REAL_GRASP_FORCE_PCT,
        )
        time.sleep(0.5)
        measured_grasp_gap_mm = robot.get_gripper_gap() * 1000.0

        print("  3/5 抬起纸杯、空中翻转 90° 并搬运至竖放目标位置 ...", flush=True)
        _stream_waypoints_on_real(robot, twin_report.phases.lift_flip_place_waypoints)
        time.sleep(0.3)

        print("  4/5 缓速张开 Gripper G2 释放竖直纸杯 ...", flush=True)
        robot.set_gripper_gap(
            REAL_RELEASE_GAP_MM / 1000.0,
            speed_percent=15.0,
            force_percent=40.0,
        )
        time.sleep(0.5)

        print("  5/5 沿夹爪轴向安全退刀并回到观测位姿 ...", flush=True)
        _stream_waypoints_on_real(robot, twin_report.phases.retreat_waypoints)

        q_final = robot.get_joint_positions()
        final_err_rad = float(np.max(np.abs(q_final - q_obs)))
        t_base_flange_after = np.eye(4, dtype=np.float64)
        t_base_flange_after[:3, :3] = robot.data.xmat[link6_id].reshape(3, 3)
        t_base_flange_after[:3, 3] = robot.data.xpos[link6_id]

        post_ok = True
        if post_verify:
            print("[闭环确认] 正在从 Orbbec Gemini 305 拍照确认竖直纸杯放置状态 ...", flush=True)
            after_bgr, after_depth, _ = capture_rgbd_frame()
            post_check, after_debug = verify_upright_cup(
                after_bgr,
                after_depth,
                t_base_flange_after,
                target_xy,
                handeye_yaml=handeye_path,
                table_z_m=cup_pose.table_z_m,
            )
            if save_after_vis_path is not None:
                after_file = save_post_place_overlay(
                    after_bgr, post_check, after_debug, save_after_vis_path
                )
                print(f"[闭环确认] 已保存放置后验证图像: {after_file}", flush=True)
            result_payload["post_place_verification"] = post_check.as_dict()
            post_ok = bool(post_check.ok)
            status_tag = "通过" if post_ok else f"未通过 ({post_check.reason})"
            print(
                f"[闭环确认] 视觉复核{status_tag}: 实测位置=({post_check.measured_xy_m[0]:.3f}, "
                f"{post_check.measured_xy_m[1]:.3f}) m, 水平偏差={post_check.xy_error_mm:.1f} mm, "
                f"杯口高度={post_check.rim_height_mm:.1f} mm, 纸杯像素={post_check.roi_cup_pixels}",
                flush=True,
            )

        overall_ok = bool(final_err_rad < 0.08 and post_ok)
        result_payload.update(
            confirm_real=True,
            commanded_grasp_gap_mm=grasp_gap_mm,
            measured_grasp_gap_mm=measured_grasp_gap_mm,
            final_joint_error_rad=final_err_rad,
            ok=overall_ok,
            success=overall_ok,
        )
        _write_json_report(report_path, result_payload)
        if not post_ok:
            raise SafetyGateError(
                f"放置后相机闭环确认未通过: {result_payload['post_place_verification']['reason']}"
            )
        print(
            f"[Real] 真机单周期执行与视觉闭环确认完成: 夹持实测开度={measured_grasp_gap_mm:.1f} mm, "
            f"回位关节误差={final_err_rad * 1000:.2f} mrad",
            flush=True,
        )
        return result_payload


def _write_json_report(report_path: str | Path | None, payload: dict[str, Any]) -> None:
    if not report_path:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
