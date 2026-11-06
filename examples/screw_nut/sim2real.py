# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Sim-to-Real pipeline for screw_nut: perception, digital-twin preflight, and hardware execution.

Modes: ``dry-run`` (camera + digital twin, no motion), ``sdk-sim`` (xArm controller simulation),
``real`` (one cycle on hardware, requires ``--confirm-real``). In ``real`` mode with camera
detection, the wrist camera re-measures the nut and the screw from directly above before the
grasp. Between threading strokes, the measured G2 gap is used as a tactile height sensor:
a gap wider than the 20 mm hex body means the fingers slid down onto the 29 mm flange, so the
TCP is raised and the nut is re-grasped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from ufactory_mujoco.calibration import KinematicsCalibration, load_kinematics_calibration
from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.ik import solve_ik
from ufactory_mujoco.safety import (
    SafetyGateError,
    authorize_real_motion,
    enforce_single_cycle,
    run_sdk_simulation,
    stream_sha256,
)
from ufactory_mujoco.xarm_robot import XArmRobot

from common.motion import (
    interpolate_waypoint_chain,
    read_real_gripper_gap_mm,
    stream_waypoints_on_real,
)
from common.paths import (
    BUILD_DIR,
    DEFAULT_CALIBRATION_YAML,
    DEFAULT_HANDEYE_YAML,
    DEFAULT_ROBOT_IP,
)
from common.sim2real import config_sha256, flange_pose, make_collision_checker, write_json_report
from common.vision import solve_camera_lookdown_q as _solve_camera_lookdown_q

from . import screw_nut as task
from .perception import (
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

MIN_CAMERA_TABLE_CLEARANCE_M = 0.088
MAX_JOINT_STEP_RAD = 0.15
CAM_HOVER_Z_M = 0.250
MIN_VALID_NUT_GRASP_GAP_MM = 12.0  # below: empty grasp
MAX_HEX_FLAT_GAP_MM = 21.2  # above at regrip: fingers are on the 29 mm flange, not the hex
MIN_VALID_NUT_OPEN_GAP_MM = 42.0  # the gripper must clear the 29 mm flange before the J6 rewind
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
DEFAULT_EXEC_LOG_PATH = BUILD_DIR / "screw_nut" / "real_execution.log"


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
    waypoints: task.ScrewNutWaypoints
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
    calibration: KinematicsCalibration | None = None,
) -> CollisionChecker:
    """Collision checker for the screw_nut scene (the nut and the screw are ignored)."""
    return make_collision_checker(task.SCENE, calibration, ignore_bodies=task.ALL_TASK_BODIES)


def solve_camera_lookdown_q(
    target_xy_m: tuple[float, float] | np.ndarray,
    *,
    z_cam_m: float = CAM_HOVER_Z_M,
    q_seed: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
) -> np.ndarray:
    """Joints that put the wrist camera directly above ``target_xy_m``, looking down."""
    return _solve_camera_lookdown_q(
        target_xy_m,
        z_cam_m=z_cam_m,
        q_seed=q_seed,
        scene_path=task.SCENE,
        calibration=calibration,
        handeye_yaml=handeye_yaml,
        ignore_bodies=task.ALL_TASK_BODIES,
    )


def _tcp_site(model: mujoco.MjModel) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
    return site_id


def waypoint_tcp_z(checker: CollisionChecker, q_wp: np.ndarray) -> float:
    """World TCP Z (m) of a 6-DoF joint waypoint in the (calibrated) checker model."""
    model = checker.model
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(q_wp, dtype=np.float64).reshape(6)
    mujoco.mj_kinematics(model, data)
    return float(data.site_xpos[_tcp_site(model)][2])


def shift_waypoints_z(
    checker: CollisionChecker,
    waypoints: list[np.ndarray],
    dz_m: float,
    *,
    min_tcp_z_m: float | None = None,
    max_tcp_z_m: float | None = None,
) -> list[np.ndarray]:
    """Shift joint waypoints vertically by ``dz_m`` (TCP XY, tilt, and J6 are kept).

    The shifted TCP height is clamped to ``[min_tcp_z_m, max_tcp_z_m]``. Every new waypoint
    is checked for joint limits and collisions.
    """
    if abs(dz_m) < 1e-6 and min_tcp_z_m is None and max_tcp_z_m is None:
        return [np.asarray(q, dtype=np.float64).reshape(6).copy() for q in waypoints]

    model = checker.model
    data = mujoco.MjData(model)
    site_id = _tcp_site(model)
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
            raise SafetyGateError(f"IK did not converge for the Z-shifted waypoint {target_pos}")
        q_new = np.asarray(q_sol[:6], dtype=np.float64).copy()
        q_new[5] = float(q6[5])
        if np.any(q_new < checker.q_lo - 1e-6) or np.any(q_new > checker.q_hi + 1e-6):
            raise SafetyGateError("Z-shifted waypoint exceeds the joint limits")
        if checker.in_collision(q_new):
            raise SafetyGateError("Z-shifted waypoint is in collision")
        shifted.append(q_new)
    return shifted


def build_shadow_stream(
    q_init: np.ndarray,
    waypoints: task.ScrewNutWaypoints,
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.65,
) -> np.ndarray:
    """Concatenate all phases into one smooth 50 Hz joint stream (N, 6)."""
    q_cur = np.asarray(q_init, dtype=np.float64).reshape(-1).copy()
    chunks: list[np.ndarray] = [q_cur.reshape(1, -1)]

    def append_segment(wps: list[np.ndarray], *, soft_landing: bool) -> None:
        nonlocal q_cur
        if not wps:
            return
        seg = interpolate_waypoint_chain(
            q_cur, wps, dt=dt, max_vel_rad_s=max_vel_rad_s, soft_landing=soft_landing
        )
        if len(seg) > 1:
            chunks.append(seg[1:])
            q_cur = seg[-1].copy()

    append_segment(waypoints.approach_waypoints, soft_landing=True)  # descend to the nut
    append_segment(waypoints.lift_align_waypoints, soft_landing=True)  # carry, capture screw tip
    for idx, stroke_wps in enumerate(waypoints.stroke_waypoints):  # threading strokes + rewinds
        append_segment(stroke_wps, soft_landing=True)
        if idx < len(waypoints.rewind_waypoints):
            append_segment(waypoints.rewind_waypoints[idx], soft_landing=True)
    append_segment(waypoints.retreat_waypoints, soft_landing=False)  # back to the start pose
    return np.vstack(chunks)


def run_digital_twin_preflight(
    nut_pose: NutPoseEstimate,
    screw_pose: ScrewPoseEstimate,
    *,
    thread_turns: float = task.DEFAULT_THREAD_TURNS,
    q_init: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    headless: bool = False,
) -> ScrewNutTwinReport:
    """Run the full task in the calibrated MuJoCo twin and check the resulting joint stream."""
    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = make_screw_nut_collision_checker(calibration)

    q_start = np.asarray(q_init, dtype=np.float64).reshape(6).copy()
    recorded_wps = task.ScrewNutWaypoints()
    failures: list[str] = []
    min_cam_z_m = float("inf")
    min_col_dist_m = float("inf")
    metrics: dict[str, Any] = {}
    table_z_ref = float(nut_pose.table_z_m)

    init_nut = task.InitialNutPose(
        xy_m=nut_pose.xy_m, yaw_deg=nut_pose.yaw_deg, table_z_m=table_z_ref
    )
    screw_xy = np.asarray(screw_pose.xy_m, dtype=np.float64).reshape(2)

    with task.ScrewNutRobot(viewer=not headless, calibration=calibration) as sim_robot:
        orig_step = sim_robot._step

        def monitored_step() -> None:
            nonlocal min_cam_z_m
            orig_step()
            t_cam = flange_pose(sim_robot.model, sim_robot.data) @ x_flange_cam
            min_cam_z_m = min(min_cam_z_m, float(t_cam[2, 3]) - table_z_ref)

        sim_robot._step = monitored_step  # type: ignore[method-assign]
        try:
            metrics = task.run_screw_nut(
                sim_robot,
                nut_pose=init_nut,
                screw_xy=screw_xy,
                thread_turns=thread_turns,
                q_init=q_start,
                recorded_waypoints=recorded_wps,
            )
        except Exception as exc:  # noqa: BLE001 - reported as a preflight failure
            failures.append(f"digital-twin rollout failed: {exc}")

    if not failures:
        recorded_wps.retreat_waypoints.append(q_start.copy())

    shadow_stream = build_shadow_stream(q_start, recorded_wps)
    max_step_rad = (
        float(np.max(np.abs(np.diff(shadow_stream, axis=0)))) if len(shadow_stream) > 1 else 0.0
    )

    fk_data = mujoco.MjData(checker.model)
    for idx, q_sample in enumerate(shadow_stream):
        if np.any(q_sample < checker.q_lo - 1e-6) or np.any(q_sample > checker.q_hi + 1e-6):
            failures.append(f"stream sample #{idx} exceeds the joint limits")
            break
        dist = float(checker.min_distance(q_sample))
        min_col_dist_m = min(min_col_dist_m, dist)
        if checker.in_collision(q_sample):
            failures.append(f"stream sample #{idx} is in collision (min_dist={dist * 1000:.2f} mm)")
            break
        fk_data.qpos[:6] = q_sample
        mujoco.mj_forward(checker.model, fk_data)
        t_cam = flange_pose(checker.model, fk_data) @ x_flange_cam
        min_cam_z_m = min(min_cam_z_m, float(t_cam[2, 3]) - table_z_ref)

    if min_cam_z_m < MIN_CAMERA_TABLE_CLEARANCE_M:
        failures.append(
            f"wrist camera {min_cam_z_m * 1000:.1f} mm above the table "
            f"(< {MIN_CAMERA_TABLE_CLEARANCE_M * 1000:.1f} mm)"
        )
    if max_step_rad > MAX_JOINT_STEP_RAD:
        failures.append(f"joint step {max_step_rad:.4f} rad > {MAX_JOINT_STEP_RAD} rad")

    total_wps = (
        len(recorded_wps.approach_waypoints)
        + len(recorded_wps.lift_align_waypoints)
        + len(recorded_wps.thread_waypoints)
        + len(recorded_wps.regrip_waypoints)
        + len(recorded_wps.retreat_waypoints)
    )
    return ScrewNutTwinReport(
        ok=not failures,
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


def replan_from_refined(
    refined_nut: NutPoseEstimate,
    refined_screw: ScrewPoseEstimate,
    *,
    thread_turns: float,
    q_start: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
) -> ScrewNutTwinReport:
    """Re-run the (headless) twin from ``q_start`` with the look-down refined poses."""
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
        raise SafetyGateError(
            f"Replanning after look-down refinement failed: {'; '.join(report.failures)}"
        )
    return report


def run_sim2real_pipeline(
    *,
    mode: str,
    thread_turns: float = task.DEFAULT_THREAD_TURNS,
    ip: str = DEFAULT_ROBOT_IP,
    calibration_path: str | Path = DEFAULT_CALIBRATION_YAML,
    handeye_path: str | Path = DEFAULT_HANDEYE_YAML,
    camera_serial: str = "",
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
    """Run the nut-threading Sim-to-Real pipeline (``dry-run``, ``sdk-sim``, or ``real``)."""
    if mode not in ("dry-run", "sdk-sim", "real"):
        raise ValueError(f"Unsupported hardware mode: {mode!r}")
    enforce_single_cycle(cycles, mode)

    calib = load_kinematics_calibration(calibration_path, robot="xarm6")

    with XArmRobot(
        ip,
        robot="xarm6",
        gripper="gripper_g2",
        model_path=task.SCENE,
        calibration=calib,
        read_only=True,
    ) as robot:
        q_obs = robot.get_joint_positions()
        t_base_flange = flange_pose(robot.model, robot.data)

        color_bgr: np.ndarray | None = None
        debug_info: dict[str, Any] = {}
        cam_stream: OrbbecCameraStream | None = None
        result_payload: dict[str, Any] = {"mode": mode, "ok": False}
        try:
            # 1. Perception: manual poses or one overview frame.
            if nut_pose_override is not None and screw_xy_override is not None:
                nut_est, screw_est = pose_estimates_from_manual(
                    (nut_pose_override[0], nut_pose_override[1]),
                    nut_pose_override[2],
                    screw_xy_override,
                )
                print(
                    f"[perception] manual nut=({nut_est.xy_m[0]:.4f}, {nut_est.xy_m[1]:.4f}) m "
                    f"yaw={nut_est.yaw_deg:.1f} deg, screw=({screw_est.xy_m[0]:.4f}, "
                    f"{screw_est.xy_m[1]:.4f}) m",
                    flush=True,
                )
            else:
                if mode == "real":
                    cam_stream = OrbbecCameraStream(camera_serial)
                    cam_stream.start()
                print("[perception] capturing the overview RGB-D frame ...", flush=True)
                color_bgr, depth_m, _ = capture_rgbd_frame(serial=camera_serial, stream=cam_stream)
                nut_est, screw_est, debug_info = detect_nut_and_screw_overview(
                    color_bgr, depth_m, t_base_flange, handeye_yaml=handeye_path
                )
                print(
                    f"[perception] nut=({nut_est.xy_m[0]:.4f}, {nut_est.xy_m[1]:.4f}) m "
                    f"yaw={nut_est.yaw_deg:.1f} deg ({nut_est.point_count} pts), "
                    f"screw=({screw_est.xy_m[0]:.4f}, {screw_est.xy_m[1]:.4f}) m "
                    f"height={screw_est.height_mm:.1f} mm ({screw_est.point_count} pts)",
                    flush=True,
                )

            if color_bgr is not None and save_vis_path is not None:
                vis_file = save_detection_overlay(
                    color_bgr, nut_est, screw_est, debug_info, save_vis_path
                )
                print(f"[perception] detection overlay: {vis_file}", flush=True)

            # 2. Digital-twin preflight.
            print(
                f"[preflight] digital twin ({'headless' if headless else 'viewer'}, "
                f"{thread_turns:.2f} turns) ...",
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
                raise SafetyGateError(
                    f"Digital-twin preflight failed; hardware motion blocked: "
                    f"{'; '.join(twin_report.failures)}"
                )
            print(
                f"[preflight] passed: {twin_report.metrics['stroke_count']} strokes "
                f"({twin_report.metrics['rotated_deg']:.1f} deg), "
                f"nut height={twin_report.metrics['nut_z_mm']:.2f} mm, "
                f"XY error={twin_report.metrics['xy_error_mm']:.2f} mm, "
                f"min camera height={twin_report.min_camera_z_mm:.1f} mm, "
                f"min clearance={twin_report.min_collision_dist_mm:.1f} mm",
                flush=True,
            )

            prog_hash = stream_sha256(twin_report.shadow_stream_rad)
            cfg_hash = config_sha256(
                {
                    "nut_pose": nut_est.as_dict(),
                    "screw_pose": screw_est.as_dict(),
                    "thread_turns": float(thread_turns),
                    "calibration_sha256": calib.sha256,
                }
            )
            result_payload = {
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
                write_json_report(report_path, result_payload)
                return result_payload

            # 3. Controller simulation (sdk-sim): the arm does not move.
            checker = make_screw_nut_collision_checker(calib)
            if mode == "sdk-sim":
                print(
                    "[sdk-sim] replaying the stream in controller simulation mode ...", flush=True
                )
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
                result_payload.update(sdk_sim=evidence.as_dict(), ok=ok, success=ok)
                write_json_report(report_path, result_payload)
                if not ok:
                    raise SafetyGateError(f"sdk-sim check failed: {evidence.failures}")
                print(
                    f"[sdk-sim] passed: {evidence.samples} samples, "
                    f"max joint error={evidence.max_joint_error_rad * 1000:.3f} mrad, "
                    f"physical drift={evidence.physical_joint_drift_rad * 1000:.4f} mrad",
                    flush=True,
                )
                return result_payload

            # 4. Real execution with look-down refinement and gripper-gap gates.
            return _execute_real(
                robot,
                calib=calib,
                checker=checker,
                twin_report=twin_report,
                result_payload=result_payload,
                nut_est=nut_est,
                screw_est=screw_est,
                use_lookdown=nut_pose_override is None or screw_xy_override is None,
                q_obs=q_obs,
                cam_stream=cam_stream,
                camera_serial=camera_serial,
                thread_turns=thread_turns,
                handeye_path=handeye_path,
                confirm_real=confirm_real,
                cycles=cycles,
                cfg_hash=cfg_hash,
                prog_hash=prog_hash,
                post_verify=post_verify,
                save_after_vis_path=save_after_vis_path,
                report_path=report_path,
            )
        finally:
            if mode == "real" and report_path:
                write_json_report(report_path, result_payload)
            if cam_stream is not None:
                cam_stream.close()


def _abort_to_start(
    robot: XArmRobot,
    path: list[np.ndarray],
    q_obs: np.ndarray,
    result_payload: dict[str, Any],
    report_path: str | Path | None,
    **fields: Any,
) -> dict[str, Any]:
    """Move back to the start pose and record an aborted (but safe) run."""
    stream_waypoints_on_real(robot, path, soft_landing=True)
    q_end = robot.get_joint_positions()
    result_payload.update(
        confirm_real=True,
        aborted=True,
        final_joint_error_rad=float(np.max(np.abs(q_end - q_obs))),
        ok=False,
        success=False,
        completed=False,
        **fields,
    )
    write_json_report(report_path, result_payload)
    return result_payload


def _execute_real(
    robot: XArmRobot,
    *,
    calib: KinematicsCalibration,
    checker: CollisionChecker,
    twin_report: ScrewNutTwinReport,
    result_payload: dict[str, Any],
    nut_est: NutPoseEstimate,
    screw_est: ScrewPoseEstimate,
    use_lookdown: bool,
    q_obs: np.ndarray,
    cam_stream: OrbbecCameraStream | None,
    camera_serial: str,
    thread_turns: float,
    handeye_path: str | Path,
    confirm_real: bool,
    cycles: int,
    cfg_hash: str,
    prog_hash: str,
    post_verify: bool,
    save_after_vis_path: str | Path | None,
    report_path: str | Path | None,
) -> dict[str, Any]:
    exec_log_path = DEFAULT_EXEC_LOG_PATH
    exec_log_path.parent.mkdir(parents=True, exist_ok=True)
    exec_log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with exec_log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def set_gap(gap_mm: float, speed: float, force: float, settle_s: float) -> float:
        robot.set_gripper_gap(gap_mm / 1000.0, speed_percent=speed, force_percent=force)
        if settle_s > 0:
            time.sleep(settle_s)
        return read_real_gripper_gap_mm(robot)

    def capture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        robot.get_joint_positions()  # refresh robot.data
        t_bf = flange_pose(robot.model, robot.data)
        bgr, depth, _ = capture_rgbd_frame(serial=camera_serial, stream=cam_stream)
        return bgr, depth, t_bf

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
    result_payload.update(confirm_real=True, completed=False, execution_log=str(exec_log_path))
    log("[real] motion unlocked for one cycle")

    init_open_gap_mm = set_gap(REAL_PRE_GRASP_GAP_MM, REAL_OPEN_SPEED_PCT, REAL_OPEN_FORCE_PCT, 0.0)
    log(
        f"  [0/6] open gripper: target {REAL_PRE_GRASP_GAP_MM:.1f} mm, measured {init_open_gap_mm:.1f} mm"
    )

    active_wps = twin_report.waypoints
    refined_log: dict[str, Any] = {}

    if use_lookdown:
        # 1. Re-measure the nut from directly above.
        q_cam_nut = solve_camera_lookdown_q(
            nut_est.xy_m, q_seed=q_obs, calibration=calib, handeye_yaml=handeye_path
        )
        log(f"  [1/6] look-down measurement of the nut (camera z={CAM_HOVER_Z_M * 1000:.0f} mm)")
        stream_waypoints_on_real(robot, [q_cam_nut], soft_landing=True)
        nut_bgr, nut_depth, t_bf_nut = capture()
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
            log(f"  [abort] nut not found from above ({nut_dbg.get('reason')}); returning to start")
            return _abort_to_start(
                robot,
                [q_obs],
                q_obs,
                result_payload,
                report_path,
                aborted_missing_nut=True,
                abort_reason=str(nut_dbg.get("reason", "missing_nut")),
            )

        # 2. Re-measure the fixed screw from directly above.
        q_cam_screw = solve_camera_lookdown_q(
            screw_est.xy_m, q_seed=q_cam_nut, calibration=calib, handeye_yaml=handeye_path
        )
        log(f"  [2/6] look-down measurement of the screw (camera z={CAM_HOVER_Z_M * 1000:.0f} mm)")
        stream_waypoints_on_real(robot, [q_cam_screw], soft_landing=True)
        screw_bgr, screw_depth, t_bf_screw = capture()
        q_hover_screw = robot.get_joint_positions()
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
            log(
                f"  [abort] screw not found from above ({screw_dbg.get('reason')}); returning to start"
            )
            return _abort_to_start(
                robot,
                [q_obs],
                q_obs,
                result_payload,
                report_path,
                aborted_missing_screw=True,
                abort_reason=str(screw_dbg.get("reason", "missing_screw")),
            )

        nut_dxy_mm = float(np.linalg.norm(np.subtract(refined_nut.xy_m, nut_est.xy_m)) * 1000.0)
        screw_dxy_mm = float(
            np.linalg.norm(np.subtract(refined_screw.xy_m, screw_est.xy_m)) * 1000.0
        )
        log(
            f"  [refined] nut=({refined_nut.xy_m[0]:.4f}, {refined_nut.xy_m[1]:.4f}) m "
            f"(dxy={nut_dxy_mm:.2f} mm, yaw={refined_nut.yaw_deg:.1f} deg), "
            f"screw=({refined_screw.xy_m[0]:.4f}, {refined_screw.xy_m[1]:.4f}) m "
            f"(dxy={screw_dxy_mm:.2f} mm, h={refined_screw.height_mm:.1f} mm)"
        )
        refined_log = {
            "nut_delta_xy_mm": round(nut_dxy_mm, 2),
            "screw_delta_xy_mm": round(screw_dxy_mm, 2),
            "refined_nut": refined_nut.as_dict(),
            "refined_screw": refined_screw.as_dict(),
        }
        nut_est, screw_est = refined_nut, refined_screw
        active_wps = replan_from_refined(
            nut_est,
            screw_est,
            thread_turns=thread_turns,
            q_start=q_hover_screw,
            calibration=calib,
            handeye_yaml=handeye_path,
        ).waypoints

    # 3. Grasp the nut.
    log("  [3/6] descend to the nut and close the gripper")
    stream_waypoints_on_real(robot, active_wps.approach_waypoints, soft_landing=True)
    meas_gap_mm = set_gap(REAL_GRASP_GAP_MM, REAL_GRASP_SPEED_PCT, REAL_GRASP_FORCE_PCT, 0.35)
    measured_gaps_mm = [float(meas_gap_mm)]
    measured_open_gaps_mm: list[float] = [float(init_open_gap_mm)]
    stroke_logs: list[dict[str, Any]] = []
    result_payload.update(
        measured_grasp_gaps_mm=measured_gaps_mm,
        measured_open_gaps_mm=measured_open_gaps_mm,
        stroke_logs=stroke_logs,
        refined_poses=refined_log,
    )
    log(f"  [grasp] target {REAL_GRASP_GAP_MM:.1f} mm, measured {meas_gap_mm:.1f} mm")
    if meas_gap_mm < MIN_VALID_NUT_GRASP_GAP_MM:
        log(
            f"  [abort] gap {meas_gap_mm:.1f} mm < {MIN_VALID_NUT_GRASP_GAP_MM:.1f} mm "
            "(empty grasp); opening and returning to start"
        )
        set_gap(REAL_PRE_GRASP_GAP_MM, REAL_OPEN_SPEED_PCT, REAL_OPEN_FORCE_PCT, 0.25)
        return _abort_to_start(
            robot,
            [active_wps.approach_waypoints[0], q_obs],
            q_obs,
            result_payload,
            report_path,
            aborted_empty_grasp=True,
            abort_reason="empty_grasp_nut",
        )

    # 4. Carry the nut and capture the screw tip.
    log("  [4/6] lift, align over the screw, and lower the nut onto the screw tip")
    stream_waypoints_on_real(robot, active_wps.lift_align_waypoints, soft_landing=True)

    # 5. Threading strokes with J6 rewinds and tactile Z adaptation.
    stroke_queue = [list(wps) for wps in active_wps.stroke_waypoints]
    rewind_queue = [list(wps) for wps in active_wps.rewind_waypoints]
    nominal_num_strokes = len(stroke_queue)
    table_z_m = float(screw_est.table_z_m)
    min_tcp_z_m = table_z_m + task.THREAD_END_TCP_Z_M
    max_tcp_z_m = table_z_m + task.CAPTURE_SCREW_TCP_Z_M + 0.0020
    z_offset_m = 0.0
    z_adapt_count = 0
    extra_strokes_used = 0
    z_adapt_events: list[dict[str, Any]] = []

    # Template 1.5-turn stroke and rewind for adaptive extra strokes.
    template_stroke = list(active_wps.stroke_waypoints[-2 if len(stroke_queue) >= 2 else -1])
    template_rewind = list(active_wps.rewind_waypoints[-1]) if rewind_queue else []
    template_stroke_start_z = waypoint_tcp_z(checker, template_stroke[0])
    template_stroke_end_z = waypoint_tcp_z(checker, template_stroke[-1])
    template_rewind_z = (
        waypoint_tcp_z(checker, template_rewind[-1]) if template_rewind else template_stroke_end_z
    )

    def shift(wps: list[np.ndarray], dz_m: float) -> list[np.ndarray]:
        return shift_waypoints_z(
            checker, wps, dz_m, min_tcp_z_m=min_tcp_z_m, max_tcp_z_m=max_tcp_z_m
        )

    def tcp_z_mm(q: np.ndarray) -> float:
        return (waypoint_tcp_z(checker, q) - table_z_m) * 1000.0

    log(
        f"  [5/6] {nominal_num_strokes} threading strokes (open {REAL_REGRIP_GAP_MM:.1f} mm, "
        f"min open {MIN_VALID_NUT_OPEN_GAP_MM:.1f} mm, max hex gap {MAX_HEX_FLAT_GAP_MM:.1f} mm, "
        f"force {REAL_GRASP_FORCE_PCT:.1f}%)"
    )
    s_idx = 0
    while s_idx < len(stroke_queue):
        base_stroke = stroke_queue[s_idx]
        q_before = robot.get_joint_positions()
        z_start_mm = tcp_z_mm(q_before)
        stream_waypoints_on_real(robot, shift(base_stroke, z_offset_m), soft_landing=True)
        q_after = robot.get_joint_positions()
        z_end_mm = tcp_z_mm(q_after)
        j6_start_deg = float(np.degrees(q_before[5]))
        j6_end_deg = float(np.degrees(q_after[5]))
        entry: dict[str, Any] = {
            "stroke_index": s_idx + 1,
            "is_adaptive_extra": bool(s_idx >= nominal_num_strokes),
            "j6_start_deg": round(j6_start_deg, 2),
            "j6_end_deg": round(j6_end_deg, 2),
            "tcp_z_start_mm": round(z_start_mm, 2),
            "tcp_z_end_mm": round(z_end_mm, 2),
            "z_adapt_offset_mm": round(z_offset_m * 1000.0, 2),
        }

        # If Z backoffs left the nut short of the seat after the last queued stroke, add one.
        if (
            s_idx == len(stroke_queue) - 1
            and z_adapt_count > 0
            and z_end_mm > task.THREAD_END_TCP_Z_M * 1000.0 + 1.2
            and extra_strokes_used < MAX_ADAPTIVE_EXTRA_STROKES
            and template_rewind
        ):
            extra_strokes_used += 1
            base_end_z = waypoint_tcp_z(checker, base_stroke[-1])
            rewind_queue.append(
                shift_waypoints_z(checker, template_rewind, base_end_z - template_rewind_z)
            )
            stroke_queue.append(
                shift_waypoints_z(checker, template_stroke, base_end_z - template_stroke_start_z)
            )
            log(
                f"    [extra stroke #{extra_strokes_used}] TCP Z={z_end_mm:.2f} mm after stroke "
                f"{s_idx + 1} > seat {task.THREAD_END_TCP_Z_M * 1000.0:.2f} mm "
                f"(Z offset +{z_offset_m * 1000.0:.1f} mm); adding a 1.5-turn stroke"
            )

        if s_idx >= len(rewind_queue):
            stroke_logs.append(entry)
            log(
                f"    stroke {s_idx + 1}/{len(stroke_queue)} (last): "
                f"J6 [{j6_start_deg:.1f} -> {j6_end_deg:.1f}] deg, "
                f"TCP Z [{z_start_mm:.2f} -> {z_end_mm:.2f}] mm"
            )
            s_idx += 1
            continue

        # Open; the gripper must clear the flange before rewinding J6.
        open_gap_mm = set_gap(REAL_REGRIP_GAP_MM, REAL_OPEN_SPEED_PCT, REAL_OPEN_FORCE_PCT, 0.30)
        if open_gap_mm < MIN_VALID_NUT_OPEN_GAP_MM:
            log(
                f"    [open retry] stroke {s_idx + 1}: gap {open_gap_mm:.1f} mm < "
                f"{MIN_VALID_NUT_OPEN_GAP_MM:.1f} mm, opening again with more force"
            )
            open_gap_mm = set_gap(REAL_REGRIP_GAP_MM, 75.0, 75.0, 0.35)
        measured_open_gaps_mm.append(float(open_gap_mm))
        entry["open_gap_mm"] = round(float(open_gap_mm), 2)
        if open_gap_mm < MIN_VALID_NUT_OPEN_GAP_MM:
            stroke_logs.append(entry)
            raise SafetyGateError(
                f"Stroke {s_idx + 1}: gripper opened only {open_gap_mm:.1f} mm "
                f"(< {MIN_VALID_NUT_OPEN_GAP_MM:.1f} mm); rewind blocked so the nut is not loosened"
            )

        stream_waypoints_on_real(robot, shift(rewind_queue[s_idx], z_offset_m), soft_landing=True)
        entry["rewind_j6_deg"] = round(float(np.degrees(robot.get_joint_positions()[5])), 2)
        regrip_gap = set_gap(REAL_GRASP_GAP_MM, 50.0, REAL_GRASP_FORCE_PCT, 0.30)
        initial_regrip_gap = float(regrip_gap)
        entry["initial_close_gap_mm"] = round(initial_regrip_gap, 2)

        # Tactile Z adaptation: a gap wider than the hex body means the fingertips are below
        # the hex/flange boundary (TCP ahead of the nut). Open, raise the TCP, and re-grasp.
        retries = 0
        while regrip_gap > MAX_HEX_FLAT_GAP_MM and retries < MAX_Z_ADAPT_RETRIES_PER_STROKE:
            retries += 1
            z_adapt_count += 1
            dz_up = float(
                np.clip(
                    0.0035 + 0.0004 * (regrip_gap - MAX_HEX_FLAT_GAP_MM),
                    Z_ADAPT_MIN_STEP_M,
                    Z_ADAPT_MAX_STEP_M,
                )
            )
            z_offset_m += dz_up
            q_lift = shift([rewind_queue[s_idx][-1]], z_offset_m)[0]
            lifted_z_mm = tcp_z_mm(q_lift)
            log(
                f"    [Z adapt #{z_adapt_count}] stroke {s_idx + 1}: regrip gap {regrip_gap:.1f} mm > "
                f"{MAX_HEX_FLAT_GAP_MM:.1f} mm (fingers on the flange); raising TCP "
                f"+{dz_up * 1000.0:.1f} mm to {lifted_z_mm:.2f} mm "
                f"(total +{z_offset_m * 1000.0:.1f} mm) and re-grasping"
            )
            set_gap(REAL_REGRIP_GAP_MM, REAL_OPEN_SPEED_PCT, REAL_OPEN_FORCE_PCT, 0.30)
            stream_waypoints_on_real(robot, [q_lift], soft_landing=True)
            regrip_gap = set_gap(REAL_GRASP_GAP_MM, 50.0, REAL_GRASP_FORCE_PCT, 0.30)
            z_adapt_events.append(
                {
                    "stroke_index": s_idx + 1,
                    "attempt": retries,
                    "initial_gap_mm": round(initial_regrip_gap, 2),
                    "dz_up_mm": round(dz_up * 1000.0, 2),
                    "cumulative_z_offset_mm": round(z_offset_m * 1000.0, 2),
                    "lifted_tcp_z_mm": round(lifted_z_mm, 2),
                    "regrip_gap_after_mm": round(float(regrip_gap), 2),
                }
            )

        measured_gaps_mm.append(float(regrip_gap))
        entry["close_gap_mm"] = round(float(regrip_gap), 2)
        entry["z_adapt_retries"] = retries
        stroke_logs.append(entry)
        log(
            f"    stroke {s_idx + 1}/{len(stroke_queue)}: "
            f"J6 [{j6_start_deg:.1f} -> {j6_end_deg:.1f}] deg, "
            f"TCP Z [{z_start_mm:.2f} -> {z_end_mm:.2f}] mm, open {open_gap_mm:.1f} mm, "
            f"rewind to J6={entry['rewind_j6_deg']:.1f} deg, regrip {regrip_gap:.1f} mm "
            f"(first {initial_regrip_gap:.1f} mm, Z offset +{z_offset_m * 1000.0:.1f} mm)"
        )
        if regrip_gap < MIN_VALID_NUT_GRASP_GAP_MM:
            raise SafetyGateError(
                f"Stroke {s_idx + 1}: regrip gap {regrip_gap:.1f} mm is too small (nut lost?)"
            )
        s_idx += 1

    # 6. Release and return.
    log("  [6/6] release the nut and return to the start pose")
    final_open_gap_mm = set_gap(
        REAL_PRE_GRASP_GAP_MM, REAL_OPEN_SPEED_PCT, REAL_OPEN_FORCE_PCT, 0.25
    )
    measured_open_gaps_mm.append(float(final_open_gap_mm))
    retreat_chain = list(active_wps.retreat_waypoints)
    if not retreat_chain or float(np.max(np.abs(retreat_chain[-1] - q_obs))) > 1e-4:
        retreat_chain.append(q_obs.copy())
    stream_waypoints_on_real(robot, retreat_chain, soft_landing=False)
    final_err_rad = float(np.max(np.abs(robot.get_joint_positions() - q_obs)))

    post_ok = True
    if post_verify:
        q_cam_after = solve_camera_lookdown_q(
            screw_est.xy_m, q_seed=q_obs, calibration=calib, handeye_yaml=handeye_path
        )
        log("[verify] moving the camera above the screw to check the result")
        stream_waypoints_on_real(robot, [q_cam_after], soft_landing=True)
        after_bgr, after_depth, t_bf_after = capture()
        expected_bottom_mm = float(
            twin_report.metrics.get("expected_nut_z_mm", task.SCREW_HEAD_LEN_M * 1000.0)
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
            log(f"[verify] overlay: {after_file}")
        stream_waypoints_on_real(robot, [q_obs], soft_landing=False)
        final_err_rad = float(np.max(np.abs(robot.get_joint_positions() - q_obs)))
        result_payload["post_thread_verification"] = post_check.as_dict()
        post_ok = bool(post_check.ok)
        log(
            f"[verify] {'passed' if post_ok else f'failed ({post_check.reason})'}: "
            f"XY error={post_check.xy_error_mm:.1f} mm, "
            f"nut top={post_check.nut_top_height_mm:.1f} mm, ring points={post_check.nut_ring_points}"
        )

    overall_ok = bool(final_err_rad < 0.08 and post_ok)
    result_payload.update(
        confirm_real=True,
        measured_grasp_gaps_mm=measured_gaps_mm,
        measured_open_gaps_mm=measured_open_gaps_mm,
        stroke_logs=stroke_logs,
        z_adapt_count=z_adapt_count,
        z_adapt_total_mm=round(z_offset_m * 1000.0, 2),
        adaptive_extra_strokes=extra_strokes_used,
        z_adapt_events=z_adapt_events,
        refined_poses=refined_log,
        final_joint_error_rad=final_err_rad,
        ok=overall_ok,
        success=overall_ok,
        completed=True,
    )
    write_json_report(report_path, result_payload)
    if not post_ok:
        raise SafetyGateError(
            f"Post-threading check failed: {result_payload['post_thread_verification']['reason']}"
        )
    return result_payload
