# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Sim-to-Real pipeline for stack_cups: perception, digital-twin preflight, and hardware execution.

Modes: ``dry-run`` (camera + digital twin, no motion), ``sdk-sim`` (xArm controller simulation),
``real`` (one cycle on hardware, requires ``--confirm-real``). In ``real`` mode with camera
detection, the wrist camera re-measures each cup from directly above before grasping it.
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
from ufactory_mujoco.safety import (
    SafetyGateError,
    authorize_real_motion,
    enforce_single_cycle,
    run_sdk_simulation,
    stream_sha256,
)
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT
from ufactory_mujoco.xarm_robot import XArmRobot

from common.camera import CameraStream, create_camera_stream
from common.geometry import yaw_rotate_quat
from common.motion import (
    interpolate_waypoint_chain,
    read_real_gripper_gap_mm,
    solve_cartesian_segment,
    stream_waypoints_on_real,
)
from common.objects.paper_cup import GRASP_AXIS_OFFSET_M
from common.paths import DEFAULT_CALIBRATION_YAML, DEFAULT_HANDEYE_YAML, DEFAULT_ROBOT_IP
from common.sim2real import (
    camera_run_context,
    config_sha256,
    flange_pose,
    make_collision_checker,
    pipeline_error_report,
    record_camera_frame,
    save_rgbd_capture,
    verify_controller_tcp_offset_and_fk,
    write_json_report,
)
from common.vision import cup_grasp_yaw_offset, target_cylinder_in_camera_view
from common.vision import solve_camera_lookdown_q as _solve_camera_lookdown_q

from . import stack_cups as task
from .perception import (
    WORKSPACE_BOUNDS_M,
    CupPoseEstimate,
    capture_rgbd_frame,
    detect_sideways_cups,
    detect_single_cup_near,
    load_handeye_calibration,
    pose_estimates_from_manual,
    save_multi_detection_overlay,
    save_post_stack_overlay,
    validate_xy_in_workspace,
    verify_stacked_cups,
)

CANDIDATE_REAL_TARGETS_XY: tuple[tuple[float, float], ...] = (
    (0.36, -0.06),
    (0.38, -0.18),
    (0.38, 0.04),
    (0.34, -0.10),
    (0.34, 0.02),
    (0.30, -0.18),
    (0.30, 0.20),
)
MIN_PICK_PLACE_SEPARATION_M = 0.110
MIN_CAMERA_TABLE_CLEARANCE_M = 0.095
CAM_HOVER_Z_M = 0.280
CAM_HOVER_Z_REALSENSE_M = 0.360
POST_VERIFY_CAM_Z_M = 0.420
MIN_VALID_GRASP_GAP_MM = 30.0
REAL_GRASP_GAP_MM = 58.0
REAL_GRASP_SPEED_PCT = 10.0
REAL_GRASP_FORCE_PCT = 5.0
REAL_RELEASE_GAP_MM = 84.0
REAL_RELEASE_SPEED_PCT = 25.0


@dataclass(frozen=True)
class MultiCupTwinReport:
    """Verification report produced by the MuJoCo multi-cup stacking digital twin rollout."""

    ok: bool
    num_cups: int
    cup_poses: list[CupPoseEstimate]
    target_xy_m: tuple[float, float]
    metrics: dict[str, Any]
    min_camera_z_mm: float
    min_collision_dist_mm: float
    max_joint_step_rad: float
    waypoint_count: int
    shadow_stream_rad: np.ndarray
    cycles: list[task.CupCycleWaypoints]
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "num_cups": int(self.num_cups),
            "cup_poses": [c.as_dict() for c in self.cup_poses],
            "target_xy_m": list(self.target_xy_m),
            "metrics": self.metrics,
            "min_camera_z_mm": float(self.min_camera_z_mm),
            "min_collision_dist_mm": float(self.min_collision_dist_mm),
            "max_joint_step_rad": float(self.max_joint_step_rad),
            "waypoint_count": int(self.waypoint_count),
            "shadow_samples": len(self.shadow_stream_rad),
            "failures": list(self.failures),
        }


def make_stack_collision_checker(
    scene_path: str | Path,
    calibration: KinematicsCalibration | None = None,
) -> CollisionChecker:
    """Arm self-collision and arm/gripper-vs-table checker (all cups are ignored)."""
    return make_collision_checker(scene_path, calibration, ignore_bodies=task.ALL_CUP_BODIES)


def _min_distance_to_cups(
    target_xy: np.ndarray,
    cup_poses: list[CupPoseEstimate],
) -> float:
    """Return minimum 2D distance from target_xy to any cup's bottom, center, or top."""
    min_d = float("inf")
    for cup in cup_poses:
        for pt in (cup.cup_center_xy_m, cup.cup_bottom_xy_m, cup.cup_top_xy_m):
            d = float(np.linalg.norm(target_xy - np.asarray(pt, dtype=np.float64)))
            min_d = min(min_d, d)
    return min_d


def select_safe_stack_target_xy(
    cup_poses: list[CupPoseEstimate],
    requested_target_xy: tuple[float, float] | np.ndarray | None = None,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> np.ndarray:
    """Choose and validate a safe stacking target XY separated from all detected cups."""
    if not cup_poses:
        raise SafetyGateError("No cup poses given; cannot plan a stack target")

    if requested_target_xy is not None:
        target = np.asarray(requested_target_xy, dtype=np.float64).reshape(2)
        validate_xy_in_workspace(target, "Requested stack target", bounds)
        min_d = _min_distance_to_cups(target, cup_poses)
        if min_d < MIN_PICK_PLACE_SEPARATION_M:
            raise SafetyGateError(
                f"Stack target ({target[0]:.3f}, {target[1]:.3f}) m is too close to a cup "
                f"({min_d * 1000:.1f} mm < {MIN_PICK_PLACE_SEPARATION_M * 1000:.0f} mm)"
            )
        return target

    valid_candidates: list[tuple[float, np.ndarray]] = []
    for cand_xy in CANDIDATE_REAL_TARGETS_XY:
        cand = np.asarray(cand_xy, dtype=np.float64)
        validate_xy_in_workspace(cand, "Candidate stack target", bounds)
        min_d = _min_distance_to_cups(cand, cup_poses)
        if min_d >= MIN_PICK_PLACE_SEPARATION_M:
            valid_candidates.append((min_d, cand))

    if not valid_candidates:
        raise SafetyGateError(
            "All default stack targets are too close to the cups; pass a clear one with --target-xy"
        )

    # Prefer the first candidate if it has ample clearance (>= 145 mm), otherwise pick max clearance
    if valid_candidates[0][0] >= 0.145:
        return valid_candidates[0][1].copy()
    valid_candidates.sort(key=lambda item: item[0], reverse=True)
    return valid_candidates[0][1].copy()


def _build_multi_cup_shadow_stream(
    q_init: np.ndarray,
    cycles: list[task.CupCycleWaypoints],
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.65,
) -> np.ndarray:
    """Concatenate all cup cycles into a continuous 50 Hz joint trajectory array (N, 6)."""
    q_cur = np.asarray(q_init, dtype=np.float64).reshape(-1).copy()
    chunks: list[np.ndarray] = [q_cur.reshape(1, -1)]
    if not cycles:
        return np.vstack(chunks)

    # Phase 1: Initial approach from observation pose to Cup #1 grasp pose
    if cycles[0].approach_waypoints:
        seg = interpolate_waypoint_chain(
            q_cur,
            cycles[0].approach_waypoints,
            dt=dt,
            max_vel_rad_s=max_vel_rad_s,
            soft_landing=True,
        )
        if len(seg) > 1:
            chunks.append(seg[1:])
            q_cur = seg[-1].copy()

    for idx, cycle in enumerate(cycles):
        # Lift, 90-degree flip, and soft-landing placement / nesting
        if cycle.lift_flip_place_waypoints:
            seg = interpolate_waypoint_chain(
                q_cur,
                cycle.lift_flip_place_waypoints,
                dt=dt,
                max_vel_rad_s=max_vel_rad_s,
                soft_landing=True,
            )
            if len(seg) > 1:
                chunks.append(seg[1:])
                q_cur = seg[-1].copy()

        # Merge retreat of cup idx with approach of cup idx+1 into one continuous trajectory
        if idx < len(cycles) - 1:
            merged_wps = list(cycle.retreat_waypoints) + list(cycles[idx + 1].approach_waypoints)
            if merged_wps:
                seg = interpolate_waypoint_chain(
                    q_cur,
                    merged_wps,
                    dt=dt,
                    max_vel_rad_s=max_vel_rad_s,
                    soft_landing=True,
                )
                if len(seg) > 1:
                    chunks.append(seg[1:])
                    q_cur = seg[-1].copy()
        elif cycle.retreat_waypoints:
            seg = interpolate_waypoint_chain(
                q_cur,
                cycle.retreat_waypoints,
                dt=dt,
                max_vel_rad_s=max_vel_rad_s,
                soft_landing=False,
            )
            if len(seg) > 1:
                chunks.append(seg[1:])
                q_cur = seg[-1].copy()

    return np.vstack(chunks)


def run_digital_twin_preflight(
    cup_poses: list[CupPoseEstimate],
    target_xy_m: np.ndarray,
    *,
    q_init: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    headless: bool = False,
) -> MultiCupTwinReport:
    """Execute the full multi-cup pick, flip, and coaxial stack in the calibrated MuJoCo twin."""
    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = make_stack_collision_checker(task.SCENE, calibration=calibration)

    q_start = np.asarray(q_init, dtype=np.float64).reshape(6).copy()
    cycles: list[task.CupCycleWaypoints] = []
    failures: list[str] = []
    min_cam_z_m = float("inf")
    min_col_dist_m = float("inf")
    metrics: dict[str, Any] = {}
    table_z_ref = float(cup_poses[0].table_z_m) if cup_poses else 0.0

    initial_poses = [
        task.InitialCupPose(
            bottom_xy_m=c.cup_bottom_xy_m,
            yaw_deg=c.yaw_deg,
            table_z_m=c.table_z_m,
        )
        for c in cup_poses
    ]

    with task.StackCupRobot(viewer=not headless, calibration=calibration) as sim_robot:

        def update_camera_clearance() -> None:
            nonlocal min_cam_z_m
            t_cam = flange_pose(sim_robot.model, sim_robot.data) @ x_flange_cam
            min_cam_z_m = min(min_cam_z_m, float(t_cam[2, 3]) - table_z_ref)

        orig_step = sim_robot._step

        def monitored_step() -> None:
            orig_step()
            update_camera_clearance()

        sim_robot._step = monitored_step  # type: ignore[method-assign]

        try:
            metrics = task.run_stack_cups(
                sim_robot,
                num_cups=len(cup_poses),
                cup_poses=initial_poses,
                target_xy=target_xy_m,
                q_init=q_start,
                recorded_cycles=cycles,
                grasp_yaw_offset_rad=cup_grasp_yaw_offset(handeye),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Digital-twin rollout failed: {exc}")

    # Append final return-to-observation waypoint at the end of the last cup's retreat
    if not failures and cycles:
        cycles[-1].retreat_waypoints.append(q_start.copy())

    shadow_stream = _build_multi_cup_shadow_stream(q_start, cycles)
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
        cam_clear = float(t_cam[2, 3]) - table_z_ref
        min_cam_z_m = min(min_cam_z_m, cam_clear)

    if min_cam_z_m < MIN_CAMERA_TABLE_CLEARANCE_M:
        failures.append(
            f"Wrist camera table clearance {min_cam_z_m * 1000:.1f} mm is below "
            f"{MIN_CAMERA_TABLE_CLEARANCE_M * 1000:.1f} mm"
        )
    if max_step_rad > 0.15:
        failures.append(f"Joint step between samples too large ({max_step_rad:.4f} rad > 0.15 rad)")

    total_wps = sum(
        len(c.approach_waypoints) + len(c.lift_flip_place_waypoints) + len(c.retreat_waypoints)
        for c in cycles
    )
    return MultiCupTwinReport(
        ok=len(failures) == 0,
        num_cups=len(cup_poses),
        cup_poses=cup_poses,
        target_xy_m=(float(target_xy_m[0]), float(target_xy_m[1])),
        metrics=metrics,
        min_camera_z_mm=min_cam_z_m * 1000.0,
        min_collision_dist_mm=min_col_dist_m * 1000.0,
        max_joint_step_rad=max_step_rad,
        waypoint_count=total_wps,
        shadow_stream_rad=shadow_stream,
        cycles=cycles,
        failures=tuple(failures),
    )


def solve_camera_lookdown_q(
    cup_center_xy_m: tuple[float, float] | np.ndarray,
    *,
    z_cam_m: float = CAM_HOVER_Z_M,
    q_seed: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
) -> np.ndarray:
    """Solve collision-free joint angles that place the Eye-in-Hand camera directly over cup_center_xy_m."""
    return _solve_camera_lookdown_q(
        cup_center_xy_m,
        z_cam_m=z_cam_m,
        q_seed=q_seed,
        scene_path=task.SCENE,
        calibration=calibration,
        handeye_yaml=handeye_yaml,
        ignore_bodies=task.ALL_CUP_BODIES,
    )


def replan_single_cup_cycle(
    base_cycle: task.CupCycleWaypoints,
    refined_pose: CupPoseEstimate,
    target_xy_m: np.ndarray,
    *,
    q_start: np.ndarray,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
) -> task.CupCycleWaypoints:
    """Recompute and collision-verify a single cup's pick & lift waypoints from refined_pose."""
    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = make_stack_collision_checker(task.SCENE, calibration=calibration)
    tgt_xy = np.asarray(target_xy_m, dtype=np.float64).reshape(2)

    yaw_rad = float(np.deg2rad(refined_pose.yaw_deg))
    axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=np.float64)
    bot_xy = np.asarray(refined_pose.cup_bottom_xy_m, dtype=np.float64).reshape(2)
    table_z = float(refined_pose.table_z_m)
    grasp_center = bot_xy + GRASP_AXIS_OFFSET_M * axis[:2]
    grasp = np.array([grasp_center[0], grasp_center[1], table_z + task.GRASP_TCP_Z_M])
    approach = np.array([grasp_center[0], grasp_center[1], table_z + 0.16])
    grasp_quat = yaw_rotate_quat(yaw_rad + cup_grasp_yaw_offset(handeye), GRIP_DOWN_QUAT)

    with task.StackCupRobot(viewer=False, calibration=calibration) as sim_robot:
        q_full = sim_robot.data.qpos.copy()
        q_full[:6] = np.asarray(q_start, dtype=np.float64).reshape(6)
        q_app, ok = sim_robot.solve_tcp(approach, grasp_quat, q_init=q_full)
        if not ok:
            raise SafetyGateError(f"IK did not converge for the refined approach pose {approach}")
        grasp_wps = solve_cartesian_segment(sim_robot, q_app[:6], grasp, grasp_quat, steps=30)
        new_approach = [np.asarray(q_app[:6], dtype=np.float64).copy()] + grasp_wps

        lift_wps = solve_cartesian_segment(
            sim_robot,
            grasp_wps[-1],
            grasp + np.array([0.0, 0.0, 0.14], dtype=np.float64),
            grasp_quat,
            steps=28,
        )
        place_side = -1.0 if tgt_xy[1] <= 0.0 else 1.0
        pre_flip_quat = yaw_rotate_quat(place_side * 0.5 * np.pi, GRIP_DOWN_QUAT)
        mid_pos = np.array([0.36, -0.05 * place_side, table_z + 0.22], dtype=np.float64)
        mid_wps = solve_cartesian_segment(sim_robot, lift_wps[-1], mid_pos, pre_flip_quat, steps=35)
        # Reuse the twin's flip/place waypoints that follow its lift (28) + pre-flip (35) steps.
        new_lift_flip_place = lift_wps + mid_wps + list(base_cycle.lift_flip_place_waypoints[63:])

    # Verify every 50 Hz sample of the updated trajectory segments
    s_app = interpolate_waypoint_chain(q_start, new_approach, soft_landing=True)
    s_lfp = interpolate_waypoint_chain(new_approach[-1], new_lift_flip_place, soft_landing=True)
    full_check = np.vstack([s_app, s_lfp[1:]])
    fk_model = checker.model
    fk_data = mujoco.MjData(fk_model)
    for idx_s, q_s in enumerate(full_check):
        if np.any(q_s < checker.q_lo - 1e-6) or np.any(q_s > checker.q_hi + 1e-6):
            raise SafetyGateError(f"Replanned sample #{idx_s} exceeds joint limits")
        if checker.in_collision(q_s):
            raise SafetyGateError(f"Replanned sample #{idx_s} is in collision")
        fk_data.qpos[:6] = q_s
        mujoco.mj_forward(fk_model, fk_data)
        t_bc = flange_pose(fk_model, fk_data) @ x_flange_cam
        if float(t_bc[2, 3]) - table_z < MIN_CAMERA_TABLE_CLEARANCE_M:
            raise SafetyGateError(
                "Replanned trajectory brings the wrist camera too close to the table"
            )

    return task.CupCycleWaypoints(
        cup_name=base_cycle.cup_name,
        approach_waypoints=new_approach,
        lift_flip_place_waypoints=new_lift_flip_place,
        retreat_waypoints=list(base_cycle.retreat_waypoints),
        grasp_gap_mm=base_cycle.grasp_gap_mm,
    )


def run_sim2real_pipeline(
    *,
    mode: str,
    num_cups: int = 2,
    ip: str = DEFAULT_ROBOT_IP,
    calibration_path: str | Path = DEFAULT_CALIBRATION_YAML,
    handeye_path: str | Path | None = None,
    camera_type: str = "realsense",
    camera_serial: str = "",
    cup_poses_override: list[float] | tuple[float, ...] | None = None,
    target_xy_override: tuple[float, float] | None = None,
    headless: bool = False,
    confirm_real: bool = False,
    cycles: int = 1,
    post_verify: bool = True,
    save_vis_path: str | Path | None = None,
    save_after_vis_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the end-to-end multi-cup stacking Sim-to-Real pipeline (dry-run, sdk-sim, or real)."""
    if mode not in ("dry-run", "sdk-sim", "real"):
        raise ValueError(f"Unsupported hardware mode: {mode!r}")
    enforce_single_cycle(cycles, mode)

    handeye_path, camera_context = camera_run_context(camera_type, handeye_path)
    result_payload: dict[str, Any] = {
        "mode": mode,
        "ok": False,
        "completed": False,
        **camera_context,
    }
    calib = load_kinematics_calibration(calibration_path, robot="xarm6")

    with (
        pipeline_error_report(report_path, result_payload),
        XArmRobot(
            ip,
            robot="xarm6",
            gripper="gripper_g2",
            model_path=task.SCENE,
            calibration=calib,
            read_only=True,
        ) as robot,
    ):
        tcp_check = verify_controller_tcp_offset_and_fk(robot)
        result_payload["controller_tcp_check"] = tcp_check
        print(
            f"[Kinematics] SDK tcp_offset={tcp_check['tcp_offset_mm_rad']} (mm, rad), "
            f"FK flange error={tcp_check['fk_flange_pos_error_mm']:.3f} mm / "
            f"{tcp_check['fk_flange_rot_error_deg']:.4f} deg",
            flush=True,
        )
        q_obs = robot.get_joint_positions()
        t_base_flange = flange_pose(robot.model, robot.data)

        # Step 1: Acquire multiple cup 6D poses (from the wrist RGB-D camera or manual override)
        color_bgr: np.ndarray | None = None
        debug_info: dict[str, Any] = {}
        cam_stream: CameraStream | None = None
        try:
            if cup_poses_override is not None:
                cup_poses = pose_estimates_from_manual(cup_poses_override, num_cups=num_cups)
                for idx, cp in enumerate(cup_poses, start=1):
                    print(
                        f"[Perception] Manual cup #{idx}: bottom=({cp.cup_bottom_xy_m[0]:.3f}, "
                        f"{cp.cup_bottom_xy_m[1]:.3f}) m, yaw={cp.yaw_deg:.1f} deg",
                        flush=True,
                    )
            else:
                if mode == "real":
                    cam_stream = create_camera_stream(camera_type, camera_serial)
                    cam_stream.start()
                print(
                    f"[Perception] Capturing RGB-D from {camera_type} to detect {num_cups} cups ...",
                    flush=True,
                )
                color_bgr, depth_m, camera_meta = capture_rgbd_frame(
                    camera_type=camera_type, serial=camera_serial, stream=cam_stream
                )
                record_camera_frame(camera_context, "overview", camera_meta, t_base_flange)
                save_rgbd_capture(save_vis_path, color_bgr, depth_m, camera_meta, t_base_flange)
                cup_poses, debug_info = detect_sideways_cups(
                    color_bgr,
                    depth_m,
                    t_base_flange,
                    num_cups=num_cups,
                    camera_metadata=camera_meta,
                    handeye_yaml=handeye_path,
                )
                for idx, cp in enumerate(cup_poses, start=1):
                    role_label = "base" if idx == 1 else f"nest level {idx}"
                    print(
                        f"[Perception] Cup #{idx} ({role_label}): bottom=({cp.cup_bottom_xy_m[0]:.3f}, "
                        f"{cp.cup_bottom_xy_m[1]:.3f}) m, rim=({cp.cup_top_xy_m[0]:.3f}, "
                        f"{cp.cup_top_xy_m[1]:.3f}) m, yaw={cp.yaw_deg:.1f} deg "
                        f"({cp.point_count} points)",
                        flush=True,
                    )

            target_xy = select_safe_stack_target_xy(cup_poses, target_xy_override)
            if color_bgr is not None and save_vis_path is not None:
                vis_file = save_multi_detection_overlay(
                    color_bgr,
                    cup_poses,
                    debug_info,
                    save_vis_path,
                    target_xy_m=target_xy,
                )
                print(f"[Perception] Saved detection overlay: {vis_file}", flush=True)

            print(
                f"[Plan] Stack target ({target_xy[0]:.3f}, {target_xy[1]:.3f}) m; running the "
                f"MuJoCo digital twin ({'headless' if headless else 'viewer'}) ...",
                flush=True,
            )

            # Step 2: Run full multi-cup physics rollout in MuJoCo digital twin
            twin_report = run_digital_twin_preflight(
                cup_poses,
                target_xy,
                q_init=q_obs,
                calibration=calib,
                handeye_yaml=handeye_path,
                headless=headless,
            )
            result_payload["digital_twin"] = twin_report.as_dict()
            if not twin_report.ok:
                reason_str = "; ".join(twin_report.failures)
                raise SafetyGateError(
                    f"Digital-twin preflight failed; hardware blocked: {reason_str}"
                )

            steps_str = ", ".join(
                f"{s:.1f} mm" for s in twin_report.metrics.get("nested_steps_mm", [])
            )
            print(
                f"[Preflight] Passed ({num_cups} cups): "
                f"placement error={twin_report.metrics['error_mm']:.2f} mm, "
                f"coaxial error={twin_report.metrics['coaxial_error_mm']:.2f} mm, "
                f"nesting steps=[{steps_str}], "
                f"min camera clearance={twin_report.min_camera_z_mm:.1f} mm, "
                f"min collision distance={twin_report.min_collision_dist_mm:.1f} mm",
                flush=True,
            )

            prog_hash = stream_sha256(twin_report.shadow_stream_rad)
            cfg_hash = config_sha256(
                {
                    "num_cups": num_cups,
                    "cup_poses": [c.as_dict() for c in cup_poses],
                    "target_xy": target_xy.tolist(),
                    "calibration_sha256": calib.sha256,
                    "tcp_offset_mm_rad": tcp_check["tcp_offset_mm_rad"],
                    **camera_context,
                }
            )

            result_payload.update(
                {
                    "mode": mode,
                    "robot": "xarm6",
                    "gripper": "gripper_g2",
                    "num_cups": num_cups,
                    "serial_number": robot.serial_number,
                    "calibration_sha256": calib.sha256,
                    **camera_context,
                    "config_sha256": cfg_hash,
                    "program_sha256": prog_hash,
                    "digital_twin": twin_report.as_dict(),
                    "ok": True,
                    "success": True,
                    "completed": True,
                }
            )

            if mode == "dry-run":
                write_json_report(report_path, result_payload)
                return result_payload

            checker = make_stack_collision_checker(task.SCENE, calibration=calib)
            if mode == "sdk-sim":
                print(
                    "[SDK-Sim] Replaying in the controller simulation (arm stays still) ...",
                    flush=True,
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
            result_payload.update(confirm_real=True, completed=False)
            print(
                f"[Real] Motion unlocked for one cycle; stacking {num_cups} cups ...",
                flush=True,
            )

            measured_grasp_gaps_mm: list[float] = []
            refined_cup_poses_log: list[dict[str, Any]] = []
            grasp_gap_mm = float(REAL_GRASP_GAP_MM)
            use_closed_loop_cam = cup_poses_override is None

            if twin_report.cycles:
                robot.set_gripper_gap(
                    REAL_RELEASE_GAP_MM / 1000.0,
                    speed_percent=50.0,
                    force_percent=50.0,
                )
                cam_hover_z_m = (
                    CAM_HOVER_Z_REALSENSE_M if camera_type == "realsense" else CAM_HOVER_Z_M
                )
                if use_closed_loop_cam:
                    q_cam_hover_1 = solve_camera_lookdown_q(
                        cup_poses[0].cup_center_xy_m,
                        z_cam_m=cam_hover_z_m,
                        q_seed=q_obs,
                        calibration=calib,
                        handeye_yaml=handeye_path,
                    )
                    print(
                        f"  [Cup 1/{num_cups} - 1/5] Open the gripper and move the camera above cup #1 "
                        f"(z_cam={cam_hover_z_m * 1000:.0f} mm) ...",
                        flush=True,
                    )
                    stream_waypoints_on_real(
                        robot,
                        [q_cam_hover_1],
                        soft_landing=True,
                    )
                else:
                    print(
                        f"  [Cup 1/{num_cups} - 1/4] Open the gripper and approach the cup #1 grasp pose ...",
                        flush=True,
                    )
                    stream_waypoints_on_real(
                        robot,
                        twin_report.cycles[0].approach_waypoints,
                        soft_landing=True,
                    )

            for idx, cycle in enumerate(twin_report.cycles, start=1):
                if use_closed_loop_cam:
                    q_hover_now = robot.get_joint_positions()
                    t_bf_hover = flange_pose(robot.model, robot.data)
                    hover_bgr, hover_depth, hover_meta = capture_rgbd_frame(
                        camera_type=camera_type, serial=camera_serial, stream=cam_stream
                    )
                    record_camera_frame(camera_context, f"near_cup_{idx}", hover_meta, t_bf_hover)
                    near_vis = (
                        Path(save_vis_path).with_name(f"real_hover_cup{idx}.png")
                        if save_vis_path
                        else None
                    )
                    save_rgbd_capture(near_vis, hover_bgr, hover_depth, hover_meta, t_bf_hover)
                    refined_cup, hover_dbg = detect_single_cup_near(
                        hover_bgr,
                        hover_depth,
                        t_bf_hover,
                        cup_poses[idx - 1].cup_center_xy_m,
                        search_radius_m=0.12,
                        exclude_xy_m=(target_xy if idx > 1 else None),
                        camera_metadata=hover_meta,
                        handeye_yaml=handeye_path,
                    )
                    if refined_cup is None:
                        if save_vis_path is not None:
                            abort_vis = Path(save_vis_path).with_name(
                                f"real_abort_missing_cup{idx}.png"
                            )
                            save_multi_detection_overlay(
                                hover_bgr, [], hover_dbg, abort_vis, target_xy_m=target_xy
                            )
                        print(
                            f"  [Abort] Cup #{idx} not found from the hover pose "
                            f"(reason: {hover_dbg.get('reason', 'missing')}); "
                            "returning to the observation pose",
                            flush=True,
                        )
                        stream_waypoints_on_real(robot, [q_obs], soft_landing=True)
                        q_end_abort = robot.get_joint_positions()
                        result_payload.update(
                            confirm_real=True,
                            aborted=True,
                            aborted_missing_cup=True,
                            abort_reason=f"missing_cup_{idx}",
                            missing_cup_index=idx,
                            completed_cups=idx - 1,
                            measured_grasp_gaps_mm=measured_grasp_gaps_mm,
                            refined_cup_poses=refined_cup_poses_log,
                            final_joint_error_rad=float(np.max(np.abs(q_end_abort - q_obs))),
                            ok=False,
                            success=False,
                            completed=False,
                        )
                        write_json_report(report_path, result_payload)
                        return result_payload

                    delta_xy_mm = float(
                        np.linalg.norm(
                            np.asarray(refined_cup.cup_bottom_xy_m)
                            - np.asarray(cup_poses[idx - 1].cup_bottom_xy_m)
                        )
                        * 1000.0
                    )
                    delta_yaw_deg = float(refined_cup.yaw_deg - cup_poses[idx - 1].yaw_deg)
                    print(
                        f"  [Cup {idx}/{num_cups} - 2/5] Cup confirmed from above "
                        f"({refined_cup.point_count} points): refined bottom="
                        f"({refined_cup.cup_bottom_xy_m[0]:.3f}, "
                        f"{refined_cup.cup_bottom_xy_m[1]:.3f}) m, yaw={refined_cup.yaw_deg:.1f} deg "
                        f"(correction dxy={delta_xy_mm:.1f} mm, dyaw={delta_yaw_deg:+.1f} deg)",
                        flush=True,
                    )
                    if save_vis_path is not None:
                        hover_vis = Path(save_vis_path).with_name(f"real_hover_cup{idx}.png")
                        save_multi_detection_overlay(
                            hover_bgr,
                            [refined_cup],
                            hover_dbg,
                            hover_vis,
                            target_xy_m=target_xy,
                        )
                    cup_poses[idx - 1] = refined_cup
                    refined_cup_poses_log.append(
                        {
                            "cup_index": idx,
                            "delta_xy_mm": round(delta_xy_mm, 2),
                            "delta_yaw_deg": round(delta_yaw_deg, 2),
                            "refined_pose": refined_cup.as_dict(),
                        }
                    )
                    cycle = replan_single_cup_cycle(
                        cycle,
                        refined_cup,
                        target_xy,
                        q_start=q_hover_now,
                        calibration=calib,
                        handeye_yaml=handeye_path,
                    )
                    print(
                        f"  [Cup {idx}/{num_cups} - 3/5] Replanned trajectory passed; "
                        f"descending to the cup #{idx} grasp pose ...",
                        flush=True,
                    )
                    stream_waypoints_on_real(
                        robot,
                        cycle.approach_waypoints,
                        soft_landing=True,
                    )

                action_desc = (
                    "place it upright as the base cup"
                    if idx == 1
                    else f"nest it into cup #{idx - 1}"
                )
                step_tag = "4/5" if use_closed_loop_cam else "2/4"
                print(
                    f"  [Cup {idx}/{num_cups} - {step_tag}] Close the gripper "
                    f"(target gap={grasp_gap_mm:.1f} mm, speed={REAL_GRASP_SPEED_PCT:.0f}%, "
                    f"force={REAL_GRASP_FORCE_PCT:.0f}%) ...",
                    flush=True,
                )
                robot.set_gripper_gap(
                    grasp_gap_mm / 1000.0,
                    speed_percent=REAL_GRASP_SPEED_PCT,
                    force_percent=REAL_GRASP_FORCE_PCT,
                )
                time.sleep(0.35)
                meas_gap = read_real_gripper_gap_mm(robot)
                measured_grasp_gaps_mm.append(float(meas_gap))
                if meas_gap < MIN_VALID_GRASP_GAP_MM:
                    print(
                        f"  [Abort] Cup #{idx} measured grasp gap too small ({meas_gap:.1f} mm < "
                        f"{MIN_VALID_GRASP_GAP_MM:.1f} mm, likely an empty grasp); "
                        "opening the gripper and returning to the observation pose",
                        flush=True,
                    )
                    robot.set_gripper_gap(
                        REAL_RELEASE_GAP_MM / 1000.0,
                        speed_percent=50.0,
                        force_percent=50.0,
                    )
                    time.sleep(0.25)
                    stream_waypoints_on_real(
                        robot,
                        [cycle.approach_waypoints[0], q_obs],
                        soft_landing=True,
                    )
                    q_end_abort = robot.get_joint_positions()
                    result_payload.update(
                        confirm_real=True,
                        aborted=True,
                        aborted_empty_grasp=True,
                        abort_reason=f"empty_grasp_cup_{idx}",
                        completed_cups=idx - 1,
                        measured_grasp_gaps_mm=measured_grasp_gaps_mm,
                        refined_cup_poses=refined_cup_poses_log,
                        final_joint_error_rad=float(np.max(np.abs(q_end_abort - q_obs))),
                        ok=False,
                        success=False,
                        completed=False,
                    )
                    write_json_report(report_path, result_payload)
                    return result_payload

                place_tag = "5/5" if use_closed_loop_cam else "3/4"
                print(
                    f"  [Cup {idx}/{num_cups} - {place_tag}] Lift cup #{idx}, flip it 90 deg, "
                    f"and {action_desc} ...",
                    flush=True,
                )
                stream_waypoints_on_real(
                    robot,
                    cycle.lift_flip_place_waypoints,
                    soft_landing=True,
                )

                print(
                    f"  [Cup {idx}/{num_cups} - release] Open the gripper to release cup #{idx} ...",
                    flush=True,
                )
                robot.set_gripper_gap(
                    REAL_RELEASE_GAP_MM / 1000.0,
                    speed_percent=REAL_RELEASE_SPEED_PCT,
                    force_percent=40.0,
                )
                time.sleep(0.25)

                if idx < num_cups:
                    if use_closed_loop_cam:
                        q_cam_hover_next = solve_camera_lookdown_q(
                            cup_poses[idx].cup_center_xy_m,
                            z_cam_m=cam_hover_z_m,
                            q_seed=q_obs,
                            calibration=calib,
                            handeye_yaml=handeye_path,
                        )
                        merged_retreat_hover = list(cycle.retreat_waypoints) + [
                            q_obs,
                            q_cam_hover_next,
                        ]
                        print(
                            f"  [Cup {idx}->{idx + 1}/{num_cups}] Retreat and move the camera above "
                            f"cup #{idx + 1} (z_cam={cam_hover_z_m * 1000:.0f} mm) ...",
                            flush=True,
                        )
                        stream_waypoints_on_real(
                            robot,
                            merged_retreat_hover,
                            soft_landing=True,
                        )
                    else:
                        next_cycle = twin_report.cycles[idx]
                        merged_retreat_approach = list(cycle.retreat_waypoints) + list(
                            next_cycle.approach_waypoints
                        )
                        print(
                            f"  [Cup {idx}->{idx + 1}/{num_cups}] Retreat and continue to the "
                            f"cup #{idx + 1} grasp pose ...",
                            flush=True,
                        )
                        stream_waypoints_on_real(
                            robot,
                            merged_retreat_approach,
                            soft_landing=True,
                        )
                else:
                    print(
                        f"  [Cup {idx}/{num_cups} - done] Retreat and return to the observation pose ...",
                        flush=True,
                    )
                    stream_waypoints_on_real(
                        robot,
                        cycle.retreat_waypoints,
                        soft_landing=False,
                    )

            q_final = robot.get_joint_positions()
            final_err_rad = float(np.max(np.abs(q_final - q_obs)))
            t_base_flange_after = flange_pose(robot.model, robot.data)

            post_ok = True
            if post_verify:
                moved_for_verify = False
                min_clear = 0.28 if camera_type == "realsense" else 0.15
                stack_top_z = (
                    cup_poses[0].table_z_m + task.CUP_HEIGHT_M + 0.025 * max(1, num_cups - 1)
                )
                if not target_cylinder_in_camera_view(
                    t_base_flange_after,
                    target_xy,
                    top_z_m=stack_top_z,
                    table_z_m=cup_poses[0].table_z_m,
                    min_cam_clearance_m=min_clear,
                    handeye_yaml=handeye_path,
                    camera_meta=camera_context.get("camera"),
                ):
                    q_verify = solve_camera_lookdown_q(
                        target_xy,
                        z_cam_m=POST_VERIFY_CAM_Z_M,
                        q_seed=q_obs,
                        calibration=calib,
                        handeye_yaml=handeye_path,
                    )
                    print(
                        f"[Verify] Moving the camera {POST_VERIFY_CAM_Z_M * 1000:.0f} mm above the cup stack ...",
                        flush=True,
                    )
                    stream_waypoints_on_real(robot, [q_verify], soft_landing=True)
                    robot.get_joint_positions()
                    t_base_flange_after = flange_pose(robot.model, robot.data)
                    moved_for_verify = True

                print("[Verify] Capturing RGB-D to verify the cup stack ...", flush=True)
                after_bgr, after_depth, after_meta = capture_rgbd_frame(
                    camera_type=camera_type, serial=camera_serial, stream=cam_stream
                )
                record_camera_frame(camera_context, "post_task", after_meta, t_base_flange_after)
                save_rgbd_capture(
                    save_after_vis_path, after_bgr, after_depth, after_meta, t_base_flange_after
                )
                post_check, after_debug = verify_stacked_cups(
                    after_bgr,
                    after_depth,
                    t_base_flange_after,
                    target_xy,
                    num_cups=num_cups,
                    initial_cups=cup_poses,
                    camera_metadata=after_meta,
                    handeye_yaml=handeye_path,
                    table_z_m=cup_poses[0].table_z_m,
                )
                if save_after_vis_path is not None:
                    after_file = save_post_stack_overlay(
                        after_bgr, post_check, after_debug, save_after_vis_path
                    )
                    print(f"[Verify] Saved verification overlay: {after_file}", flush=True)
                if moved_for_verify:
                    stream_waypoints_on_real(robot, [q_obs], soft_landing=False)
                    q_final = robot.get_joint_positions()
                    final_err_rad = float(np.max(np.abs(q_final - q_obs)))
                result_payload["post_stack_verification"] = post_check.as_dict()
                post_ok = bool(post_check.ok)
                status_tag = "PASS" if post_ok else f"FAIL ({post_check.reason})"
                print(
                    f"[Verify] {status_tag}: measured=({post_check.measured_xy_m[0]:.3f}, "
                    f"{post_check.measured_xy_m[1]:.3f}) m, XY error={post_check.xy_error_mm:.1f} mm, "
                    f"top rim height={post_check.rim_height_mm:.1f} mm, "
                    f"cup pixels={post_check.roi_cup_pixels}",
                    flush=True,
                )

            overall_ok = bool(final_err_rad < 0.08 and post_ok)
            result_payload.update(
                confirm_real=True,
                completed=True,
                commanded_grasp_gap_mm=grasp_gap_mm,
                measured_grasp_gaps_mm=measured_grasp_gaps_mm,
                refined_cup_poses=refined_cup_poses_log,
                final_joint_error_rad=final_err_rad,
                ok=overall_ok,
                success=overall_ok,
            )
            write_json_report(report_path, result_payload)
            if not post_ok:
                raise SafetyGateError(
                    "Post-stack verification failed: "
                    f"{result_payload['post_stack_verification']['reason']}"
                )
            gaps_str = ", ".join(f"{g:.1f} mm" for g in measured_grasp_gaps_mm)
            print(
                f"[Real] Stacked {num_cups} cups: measured grasp gaps=[{gaps_str}], "
                f"return joint error={final_err_rad * 1000:.2f} mrad",
                flush=True,
            )
            return result_payload
        finally:
            if cam_stream is not None:
                cam_stream.close()
