# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Sim-to-Real pipeline for pick_cup: perception, digital-twin preflight, and hardware execution.

Modes: ``dry-run`` (camera + digital twin, no motion), ``sdk-sim`` (xArm controller simulation),
``real`` (one cycle on hardware, requires ``--confirm-real``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from ufactory_mujoco.calibration import KinematicsCalibration, load_kinematics_calibration
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

from common.paths import DEFAULT_CALIBRATION_YAML, DEFAULT_HANDEYE_YAML, DEFAULT_ROBOT_IP
from common.sim2real import config_sha256, flange_pose, make_collision_checker, write_json_report

from . import pick_cup as task
from .perception import (
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
    """Joint waypoints recorded per phase while the digital twin runs the task."""

    q_init: np.ndarray
    approach_waypoints: list[np.ndarray] = field(default_factory=list)
    lift_flip_place_waypoints: list[np.ndarray] = field(default_factory=list)
    retreat_waypoints: list[np.ndarray] = field(default_factory=list)
    grasp_gap_mm: float = REAL_GRASP_GAP_MM


@dataclass(frozen=True)
class DigitalTwinReport:
    """Result of the calibrated MuJoCo digital-twin rollout."""

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
    """Arm self-collision and arm/gripper-vs-table checker (the cup is ignored)."""
    return make_collision_checker(scene_path, calibration, ignore_bodies=(task.CUP,))


def select_safe_target_xy(
    cup_pose: CupPoseEstimate,
    requested_target_xy: tuple[float, float] | np.ndarray | None = None,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> np.ndarray:
    """Validate the requested placement target, or pick a default one clear of the cup."""
    cup_mid = np.asarray(cup_pose.cup_center_xy_m, dtype=np.float64)
    cup_bot = np.asarray(cup_pose.cup_bottom_xy_m, dtype=np.float64)
    if requested_target_xy is not None:
        target = np.asarray(requested_target_xy, dtype=np.float64).reshape(2)
        validate_xy_in_workspace(target, "Requested placement target", bounds)
        if (
            np.linalg.norm(target - cup_mid) < MIN_PICK_PLACE_SEPARATION_M
            or np.linalg.norm(target - cup_bot) < MIN_PICK_PLACE_SEPARATION_M
        ):
            raise SafetyGateError(
                f"Placement target ({target[0]:.3f}, {target[1]:.3f}) m is too close to the cup "
                f"(< {MIN_PICK_PLACE_SEPARATION_M * 1000:.0f} mm)"
            )
        return target

    for candidate in (DEFAULT_REAL_TARGET_XY, FALLBACK_REAL_TARGET_XY):
        if (
            np.linalg.norm(candidate - cup_mid) >= MIN_PICK_PLACE_SEPARATION_M
            and np.linalg.norm(candidate - cup_bot) >= MIN_PICK_PLACE_SEPARATION_M
        ):
            validate_xy_in_workspace(candidate, "Default placement target", bounds)
            return candidate.copy()
    raise SafetyGateError(
        "All default placement targets overlap the cup; pass a clear one with --target-xy"
    )


def _interpolate_waypoint_chain(
    q_start: np.ndarray,
    waypoints: list[np.ndarray],
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.65,
) -> np.ndarray:
    """Convert recorded waypoints into a 50 Hz velocity-bounded joint stream (N, 6).

    Large moves use a quintic point-to-point profile; dense Cartesian IK waypoints are linearly
    resampled. (This example keeps this hardware-validated stream instead of
    ``common.motion.interpolate_waypoint_chain``.)
    """
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
            # Large point-to-point move: quintic profile.
            duration = max(1.5, disp * 1.875 / max_vel_rad_s)
            traj = plan_ptp_quintic(q0, q1, duration=duration, dt=dt)
            seg = traj.q
        else:
            # Consecutive fine Cartesian IK waypoints.
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
    """Concatenate all recorded phases into one continuous joint stream (N, 6)."""
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
    """Run the full task in the calibrated digital twin and check the resulting joint stream.

    Checks every stream sample for joint limits, collisions and wrist-camera table clearance.
    """
    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = make_cup_collision_checker(task.SCENE, calibration=calibration)

    q_start = np.asarray(q_init, dtype=np.float64).reshape(6).copy()
    phases = TwinPhaseWaypoints(q_init=q_start)
    failures: list[str] = []
    min_cam_z_m = float("inf")
    min_col_dist_m = float("inf")
    metrics: dict[str, float] = {}

    with task.CupRobot(viewer=not headless, calibration=calibration) as sim_robot:

        def update_camera_clearance() -> None:
            nonlocal min_cam_z_m
            t_cam = flange_pose(sim_robot.model, sim_robot.data) @ x_flange_cam
            min_cam_z_m = min(min_cam_z_m, float(t_cam[2, 3]) - cup_pose.table_z_m)

        orig_step = sim_robot._step

        def monitored_step() -> None:
            orig_step()
            update_camera_clearance()

        sim_robot._step = monitored_step  # type: ignore[method-assign]

        try:
            metrics = task.run_pick_cup(
                sim_robot,
                cup_pose=cup_pose,
                target_xy=target_xy_m,
                q_init=q_start,
                recorded_phases=phases,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Digital-twin rollout failed: {exc}")

    # Return to the observation pose at the end of the retreat phase.
    if not failures:
        phases.retreat_waypoints.append(q_start.copy())

    shadow_stream = _build_dense_shadow_stream(phases)
    max_step_rad = (
        float(np.max(np.abs(np.diff(shadow_stream, axis=0)))) if len(shadow_stream) > 1 else 0.0
    )

    fk_model = checker.model
    fk_data = mujoco.MjData(fk_model)
    for idx, q_sample in enumerate(shadow_stream):
        if np.any(q_sample < checker.q_lo - 1e-6) or np.any(q_sample > checker.q_hi + 1e-6):
            failures.append(f"Stream sample #{idx} exceeds joint limits")
            break
        dist = float(checker.min_distance(q_sample))
        min_col_dist_m = min(min_col_dist_m, dist)
        if checker.in_collision(q_sample):
            failures.append(f"Stream sample #{idx} is in collision (min_dist={dist * 1000:.2f} mm)")
            break
        fk_data.qpos[:6] = q_sample
        mujoco.mj_forward(fk_model, fk_data)
        t_cam = flange_pose(fk_model, fk_data) @ x_flange_cam
        min_cam_z_m = min(min_cam_z_m, float(t_cam[2, 3]) - cup_pose.table_z_m)

    if min_cam_z_m < MIN_CAMERA_TABLE_CLEARANCE_M:
        failures.append(
            f"Wrist camera table clearance {min_cam_z_m * 1000:.1f} mm is below "
            f"{MIN_CAMERA_TABLE_CLEARANCE_M * 1000:.1f} mm"
        )
    if max_step_rad > 0.15:
        failures.append(f"Joint step between samples too large ({max_step_rad:.4f} rad > 0.15 rad)")

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
    """Stream joint waypoints to the real xArm in servo_j mode."""
    if not waypoints:
        return
    q_cur = robot.get_joint_positions()
    stream = _interpolate_waypoint_chain(q_cur, waypoints, dt=dt, max_vel_rad_s=max_vel_rad_s)
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
    camera_serial: str = "",
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
    """End-to-end pipeline: perceive -> digital-twin preflight -> dry-run / sdk-sim / real."""
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

        # 1. Cup pose from the wrist camera (or a manual override).
        if cup_pose_override is not None:
            cup_pose = pose_estimate_from_manual(*cup_pose_override)
            print(
                f"[Perception] Manual cup pose: bottom=({cup_pose.cup_bottom_xy_m[0]:.3f}, "
                f"{cup_pose.cup_bottom_xy_m[1]:.3f}) m, yaw={cup_pose.yaw_deg:.1f} deg",
                flush=True,
            )
        else:
            print("[Perception] Capturing RGB-D from Orbbec Gemini 305 ...", flush=True)
            color_bgr, depth_m, _ = capture_rgbd_frame(serial=camera_serial)
            cup_pose, debug_info = detect_sideways_cup(
                color_bgr,
                depth_m,
                t_base_flange,
                handeye_yaml=handeye_path,
            )
            if save_vis_path is not None:
                vis_file = save_detection_overlay(color_bgr, cup_pose, debug_info, save_vis_path)
                print(f"[Perception] Saved detection overlay: {vis_file}", flush=True)
            print(
                f"[Perception] Cup detected: bottom=({cup_pose.cup_bottom_xy_m[0]:.3f}, "
                f"{cup_pose.cup_bottom_xy_m[1]:.3f}) m, rim=({cup_pose.cup_top_xy_m[0]:.3f}, "
                f"{cup_pose.cup_top_xy_m[1]:.3f}) m, yaw={cup_pose.yaw_deg:.1f} deg, "
                f"table Z={cup_pose.table_z_m * 1000:.1f} mm ({cup_pose.point_count} points)",
                flush=True,
            )

        target_xy = select_safe_target_xy(cup_pose, target_xy_override)
        print(
            f"[Plan] Upright target ({target_xy[0]:.3f}, {target_xy[1]:.3f}) m; running the "
            f"MuJoCo digital twin ({'headless' if headless else 'viewer'}) ...",
            flush=True,
        )

        # 2. Full physics rollout in the calibrated digital twin.
        twin_report = run_digital_twin_preflight(
            cup_pose,
            target_xy,
            q_init=q_obs,
            calibration=calib,
            handeye_yaml=handeye_path,
            headless=headless,
        )
        if not twin_report.ok:
            reason_str = "; ".join(twin_report.failures)
            raise SafetyGateError(f"Digital-twin preflight failed; hardware blocked: {reason_str}")

        print(
            f"[Preflight] Passed: placement error={twin_report.metrics['error_mm']:.2f} mm, "
            f"tilt={twin_report.metrics['tilt_deg']:.2f} deg, "
            f"min camera clearance={twin_report.min_camera_z_mm:.1f} mm, "
            f"min collision distance={twin_report.min_collision_dist_mm:.1f} mm",
            flush=True,
        )

        prog_hash = stream_sha256(twin_report.shadow_stream_rad)
        cfg_hash = config_sha256(
            {
                "cup_pose": cup_pose.as_dict(),
                "target_xy": target_xy.tolist(),
                "calibration_sha256": calib.sha256,
            }
        )

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
            write_json_report(report_path, result_payload)
            return result_payload

        checker = make_cup_collision_checker(task.SCENE, calibration=calib)
        if mode == "sdk-sim":
            print(
                "[SDK-Sim] Replaying in the controller simulation (arm stays still) ...", flush=True
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
            result_payload["sdk_sim"] = evidence.as_dict()
            result_payload["ok"] = ok
            result_payload["success"] = ok
            write_json_report(report_path, result_payload)
            if not ok:
                raise SafetyGateError(f"sdk-sim check failed: {evidence.failures}")
            print(
                f"[SDK-Sim] Passed: samples={evidence.samples}, "
                f"max joint error={evidence.max_joint_error_rad * 1000:.3f} mrad, "
                f"physical drift={evidence.physical_joint_drift_rad * 1000:.4f} mrad",
                flush=True,
            )
            return result_payload

        # 3. Real hardware (requires --confirm-real; approval is valid for one cycle).
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
        print("[Real] Motion unlocked for one cycle", flush=True)

        print("  1/5 Open the gripper and move to the grasp pose ...", flush=True)
        robot.set_gripper_gap(
            REAL_RELEASE_GAP_MM / 1000.0,
            speed_percent=50.0,
            force_percent=50.0,
        )
        _stream_waypoints_on_real(robot, twin_report.phases.approach_waypoints)

        grasp_gap_mm = float(np.clip(twin_report.phases.grasp_gap_mm, 50.0, 55.0))
        print(
            f"  2/5 Close the gripper (target gap={grasp_gap_mm:.1f} mm, "
            f"speed={REAL_GRASP_SPEED_PCT:.0f}%, force={REAL_GRASP_FORCE_PCT:.0f}%) ...",
            flush=True,
        )
        robot.set_gripper_gap(
            grasp_gap_mm / 1000.0,
            speed_percent=REAL_GRASP_SPEED_PCT,
            force_percent=REAL_GRASP_FORCE_PCT,
        )
        time.sleep(0.5)
        measured_grasp_gap_mm = robot.get_gripper_gap() * 1000.0

        print("  3/5 Lift, flip 90 deg, and carry to the target ...", flush=True)
        _stream_waypoints_on_real(robot, twin_report.phases.lift_flip_place_waypoints)
        time.sleep(0.3)

        print("  4/5 Release the cup slowly ...", flush=True)
        robot.set_gripper_gap(
            REAL_RELEASE_GAP_MM / 1000.0,
            speed_percent=15.0,
            force_percent=40.0,
        )
        time.sleep(0.5)

        print("  5/5 Retreat and return to the observation pose ...", flush=True)
        _stream_waypoints_on_real(robot, twin_report.phases.retreat_waypoints)

        q_final = robot.get_joint_positions()
        final_err_rad = float(np.max(np.abs(q_final - q_obs)))
        t_base_flange_after = flange_pose(robot.model, robot.data)

        post_ok = True
        if post_verify:
            print("[Verify] Capturing RGB-D to verify the upright cup ...", flush=True)
            after_bgr, after_depth, _ = capture_rgbd_frame(serial=camera_serial)
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
                print(f"[Verify] Saved verification overlay: {after_file}", flush=True)
            result_payload["post_place_verification"] = post_check.as_dict()
            post_ok = bool(post_check.ok)
            status_tag = "PASS" if post_ok else f"FAIL ({post_check.reason})"
            print(
                f"[Verify] {status_tag}: measured=({post_check.measured_xy_m[0]:.3f}, "
                f"{post_check.measured_xy_m[1]:.3f}) m, XY error={post_check.xy_error_mm:.1f} mm, "
                f"rim height={post_check.rim_height_mm:.1f} mm, cup pixels={post_check.roi_cup_pixels}",
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
        write_json_report(report_path, result_payload)
        if not post_ok:
            raise SafetyGateError(
                f"Post-place verification failed: {result_payload['post_place_verification']['reason']}"
            )
        print(
            f"[Real] Cycle complete: measured grasp gap={measured_grasp_gap_mm:.1f} mm, "
            f"return joint error={final_err_rad * 1000:.2f} mrad",
            flush=True,
        )
        return result_payload
