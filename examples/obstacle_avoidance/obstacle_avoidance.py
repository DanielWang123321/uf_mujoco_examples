# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Pick a block, carry it around a table-mounted pillar, and place it in simulation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from ufactory_mujoco.grippers import measured_g2_pad_gap_mm
from ufactory_mujoco.viewer import PassiveViewer

from common.paths import REPO_ROOT, resolve_repo_path
from common.sim import TaskSimRobot, configure_finger_contact_priority

from .planning import (
    DT,
    GRIPPER_MECHANISM_CLEARANCE_M,
    GRIPPER_QPOS,
    PAYLOAD,
    SCENE,
    CarriedState,
    MotionError,
    MotionPlan,
    Options,
    SceneChecker,
    plan_motion,
    pose_to_target,
    tcp_error,
)

CONFIG = Path(__file__).with_name("config.yaml")
DEFAULT_REPORT = "build/obstacle_avoidance/sim_report.json"


class ObstacleRobot(TaskSimRobot):
    def __init__(self, *, viewer: bool, scene=SCENE):
        self.scene_path = scene
        self.runtime_checker: SceneChecker | None = None
        self.options = Options()
        self.report: dict[str, Any] = {}
        self.payload_monitor = None
        self.gripper_expected = "open"
        self.viewer_requested = viewer
        super().__init__(scene, viewer=False)

    def _configure_model(self) -> None:
        configure_finger_contact_priority(self.model)

    def start_viewer(self) -> None:
        if self.viewer_requested and self._viewer is None:
            self._viewer = PassiveViewer(self.model, self.data)

    def observe(self) -> None:
        """Check actual arm and all six gripper joints after every physics step."""
        checker = self.runtime_checker
        if checker is None:
            return
        distance = checker.actual_distance(self.data.qpos.copy())
        report = self.report
        previous = report.get("actual_min_clearance_m")
        report["actual_min_clearance_m"] = distance if previous is None else min(previous, distance)
        report["actual_gripper_internal_min_clearance_m"] = min(
            report.get("actual_gripper_internal_min_clearance_m", checker.last_gripper_distance),
            checker.last_gripper_distance,
        )
        report["executed_physics_steps"] = round(self.data.time / self.options.timestep_s)
        report["simulated_time_s"] = float(self.data.time)
        if not np.all(np.isfinite(self.data.qvel)):
            raise MotionError("execution", "actual joint velocities are non-finite")
        q = self.data.qpos[: self.n_arm]
        if np.any(q < checker.q_lo - 1e-5) or np.any(q > checker.q_hi + 1e-5):
            raise MotionError("execution", "actual arm joint limits exceeded")
        gap_mm = measured_g2_pad_gap_mm(checker.model, checker.data)
        report["min_gripper_gap_mm"] = min(report.get("min_gripper_gap_mm", gap_mm), gap_mm)
        report["max_gripper_gap_mm"] = max(report.get("max_gripper_gap_mm", gap_mm), gap_mm)
        gripper = self.data.qpos[GRIPPER_QPOS]
        limits = self.model.jnt_range[6:12]
        if np.any(gripper < limits[:, 0] - 0.03) or np.any(gripper > limits[:, 1] + 0.03):
            raise MotionError("execution", "actual gripper joint limits exceeded")
        if self.gripper_expected == "open" and not 83.0 <= gap_mm <= 85.0:
            raise MotionError("execution", "G2 did not remain fully open (84 +/- 1 mm)")
        if self.gripper_expected == "holding" and not 47.0 <= gap_mm <= 53.0:
            raise MotionError("grasp_monitor", f"unexpected held-block gap {gap_mm:.2f} mm")
        speed = float(np.max(np.abs(self.data.qvel[: self.n_arm])))
        report["max_actual_velocity_rad_s"] = max(report.get("max_actual_velocity_rad_s", 0), speed)
        if distance < self.options.execution_clearance_m:
            raise MotionError(
                "execution", f"actual clearance {distance * 1000:.3f} mm is too small"
            )
        if checker.last_gripper_distance < GRIPPER_MECHANISM_CLEARANCE_M:
            raise MotionError("execution", "G2 internal mechanism clearance is too small")
        if self.payload_monitor is not None:
            self.payload_monitor.observe(checker.model, checker.data, self.options.timestep_s)
            if report.get("stage") in ("execution", "settling"):
                report["carry_actual_min_clearance_m"] = min(
                    report.get("carry_actual_min_clearance_m", distance), distance
                )

    def _step(self) -> None:
        super()._step()
        self.observe()


def draw_overlays(scene, baseline_xyz, planned_xyz, options: Options) -> None:
    """Populate a viewer user scene (or an offscreen MjvScene) without physics geoms."""
    scene.ngeom = 0

    def geom(kind, size, pos, color):
        if scene.ngeom >= scene.maxgeom:
            raise MotionError("viewer", "viewer overlay geometry capacity exceeded")
        item = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(item, kind, size, pos, np.eye(3).ravel(), color)
        scene.ngeom += 1
        return item

    def connector(a, b, color, *, radius=0.002, kind=mujoco.mjtGeom.mjGEOM_CAPSULE):
        if np.linalg.norm(b - a) < 1e-10:
            return
        item = geom(kind, np.zeros(3), np.zeros(3), color)
        mujoco.mjv_connector(item, kind, radius, a, b)

    red, green = [0.95, 0.1, 0.1, 0.85], [0.1, 0.9, 0.2, 1.0]
    for points, color in ((baseline_xyz, red), (planned_xyz, green)):
        for a, b in zip(points[:-1], points[1:], strict=True):
            connector(a, b, color)
    for name, pose, color in (
        ("start", options.start_pose, [0.1, 0.6, 1, 1]),
        ("goal", options.goal_pose, [0.85, 0.2, 1, 1]),
    ):
        pos, quat = pose_to_target(pose)
        marker = geom(mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.008), pos, color)
        marker.label = name
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        for axis, axis_color in enumerate(([1, 0, 0, 1], [0, 1, 0, 1], [0.2, 0.4, 1, 1])):
            connector(
                pos,
                pos + 0.05 * rotation[:, axis],
                axis_color,
                radius=0.004,
                kind=mujoco.mjtGeom.mjGEOM_ARROW,
            )


def _show_plan(robot: ObstacleRobot, plan: MotionPlan, options: Options) -> None:
    if robot._viewer is None:
        return
    # The shared viewer owns a public MuJoCo Handle; its user scene is visual only.
    handle = robot._viewer._handle
    stride = max(1, len(plan.trajectory.q) // 350)
    indices = np.unique(
        np.r_[np.arange(0, len(plan.trajectory.q), stride), len(plan.trajectory.q) - 1]
    )
    baseline_xyz = robot.fk_positions(plan.baseline)
    planned_xyz = robot.fk_positions(plan.trajectory.q[indices])
    with handle.lock():
        draw_overlays(handle.user_scn, baseline_xyz, planned_xyz, options)
    robot._sync(force=True)


def _record_final_state(robot, plan, options, report, hold_started) -> None:
    report.update(
        simulated_time_s=float(robot.data.time),
        executed_physics_steps=round(robot.data.time / options.timestep_s),
        hold_duration_s=(
            round(max(0, float(robot.data.time) - hold_started), 9)
            if hold_started is not None
            else 0.0
        ),
        final_joints_rad=robot.get_joint_positions().tolist(),
    )
    if np.all(np.isfinite(robot.data.qpos)):
        mujoco.mj_forward(robot.model, robot.data)
        position_mm, orientation_deg = tcp_error(robot.model, robot.data, options.goal_pose)
        report.update(
            final_joint_error_rad=float(
                np.max(np.abs(robot.get_joint_positions() - plan.path[-1]))
            ),
            final_tcp_position_error_mm=position_mm,
            final_tcp_orientation_error_deg=orientation_deg,
        )


def replay(robot: ObstacleRobot, plan: MotionPlan, options: Options, report: dict) -> None:
    """Replay an already-grasped payload without teleporting any physical state."""
    if not np.isclose(robot.model.opt.timestep, DT, rtol=0, atol=1e-12):
        raise MotionError("execution", "unexpected physics timestep")
    report["stage"] = "execution"
    robot.options, robot.report = options, report
    if robot.payload_monitor is None or plan.checker.carried is None:
        raise MotionError("execution", "carry replay requires a verified physical grasp")
    if not np.allclose(robot.get_joint_positions(), plan.path[0], atol=1e-10, rtol=0):
        raise MotionError("execution", "actual start differs from the validated carry trajectory")
    robot.runtime_checker.allow_table = False
    robot.runtime_checker.allow_fingers = True
    hold_started = None
    try:
        robot.check_viewer()
        robot.observe()
        _show_plan(robot, plan, options)
        # Planning time must not count toward GUI real-time playback.
        robot._wall_start = time.monotonic() - robot.data.time
        for q in plan.trajectory.q[1:]:
            robot.set_joint_positions(q)
            robot._step()
        report["stage"] = "settling"
        hold_started = float(robot.data.time)
        for _ in range(round(1.0 / options.timestep_s)):
            robot._step()
    finally:
        # Also retain the stopped state and errors when clearance or viewer checks abort.
        _record_final_state(robot, plan, options, report, hold_started)
    joint_error = report["final_joint_error_rad"]
    position_mm = report["final_tcp_position_error_mm"]
    orientation_deg = report["final_tcp_orientation_error_deg"]
    if joint_error > 0.01 or position_mm > 2.0 or orientation_deg > 1.0:
        raise MotionError(
            "endpoint_validation",
            f"goal error: {joint_error:.5f} rad, {position_mm:.3f} mm, {orientation_deg:.3f} deg",
        )


def run_task(robot: ObstacleRobot, options: Options, report: dict) -> None:
    from .manipulation import initialize, pickup, place

    initialize(robot, options, report)
    monitor = None
    try:
        monitor = pickup(robot, options, report)
        carried = CarriedState.from_data(robot.data)
        report["carried_payload_model"] = {
            "tcp_relative_position_m": carried.position.tolist(),
            "tcp_relative_rotation": carried.rotation.tolist(),
            "gripper_joints_rad": carried.gripper_qpos.tolist(),
        }
        checker = SceneChecker(robot.scene_path, clearance_m=options.clearance_m, carried=carried)
        planning_started = time.perf_counter()
        try:
            plan = plan_motion(
                options, report, checker=checker, q_start=robot.get_joint_positions()
            )
        finally:
            report.setdefault("planning_time_s", time.perf_counter() - planning_started)
        print("Carry the grasped block around the pillar", flush=True)
        replay(robot, plan, options, report)
        place(robot, options, report, monitor)
    finally:
        report["simulated_time_s"] = float(robot.data.time)
        report["stopped_joints_rad"] = robot.get_joint_positions().tolist()
        if np.all(np.isfinite(robot.data.qpos)):
            mujoco.mj_kinematics(robot.model, robot.data)
            position_mm, orientation_deg = tcp_error(robot.model, robot.data, options.goal_pose)
            report["final_tcp_position_error_mm"] = position_mm
            report["final_tcp_orientation_error_deg"] = orientation_deg
            report["final_joints_rad"] = robot.get_joint_positions().tolist()
            if "goal_joints_rad" in report:
                report["final_joint_error_rad"] = float(
                    np.max(np.abs(robot.get_joint_positions() - report["goal_joints_rad"]))
                )
        report["final_payload_position_m"] = robot.data.body(PAYLOAD).xpos.copy().tolist()
        if monitor is None:
            monitor = robot.payload_monitor
        if monitor is not None:
            report.update(
                max_grasp_shift_mm=monitor.max_shift_m * 1000,
                max_grasp_rotation_deg=monitor.max_rotation_deg,
                max_contact_loss_s=monitor.max_lost_s,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG), help="YAML configuration path")
    parser.add_argument("--mode", choices=("sim",), default=None, help="Simulation only")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    for name in ("start", "goal"):
        parser.add_argument(
            f"--{name}-pose",
            type=float,
            nargs=6,
            default=None,
            metavar=("X_MM", "Y_MM", "Z_MM", "ROLL_DEG", "PITCH_DEG", "YAW_DEG"),
        )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report", default=None, help="JSON report path")
    return parser


def _load_options(args, report: dict) -> tuple[Options, bool, str]:
    config_path = resolve_repo_path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a YAML mapping")
    report["config"] = str(config_path)
    report_path = args.report if args.report is not None else cfg.get("report", DEFAULT_REPORT)
    if not isinstance(report_path, str) or not report_path.strip():
        raise ValueError("report must be a non-empty path string")
    report["report_path"] = report_path
    mode = args.mode if args.mode is not None else cfg.get("mode", "sim")
    if mode != "sim":
        raise ValueError("only mode=sim is supported")
    headless = args.headless if args.headless is not None else cfg.get("headless", False)
    if not isinstance(headless, bool):
        raise ValueError("headless must be a boolean")
    raw = {}
    for section in ("task", "planning", "motion"):
        values = cfg.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a YAML mapping")
        raw.update(values)
    raw["seed"] = cfg.get("seed", 0)
    for key in ("start_pose", "goal_pose", "seed"):
        value = getattr(args, key)
        if value is not None:
            raw[key] = value
    unknown = set(raw) - {field.name for field in fields(Options)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    options = Options(**raw)
    report["inputs"] = options.to_dict()
    report["headless"] = headless
    options.validate()
    return options, headless, str(report_path)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(path: str, report: dict) -> None:
    destination = (REPO_ROOT / Path(path).expanduser()).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "success": False,
        "mode": "sim",
        "stage": "configuration",
        "failure_stage": None,
        "actual_min_clearance_m": None,
    }
    report_path = args.report if args.report is not None else DEFAULT_REPORT
    started = time.perf_counter()
    try:
        options, headless, report_path = _load_options(args, report)
        # Check the destination before planning or opening a viewer.
        _write_report(report_path, report)
        report["stage"] = "execution" if headless else "viewer"
        with ObstacleRobot(viewer=not headless) as robot:
            run_task(robot, options, report)
        report.update(success=True, stage="complete")
    except Exception as exc:  # CLI preserves partial metrics for every runtime failure.
        report["failure_stage"] = getattr(exc, "stage", report["stage"])
        report["error"] = str(exc)
        print(f"FAILED [{report['failure_stage']}]: {exc}", file=sys.stderr, flush=True)
    finally:
        report["elapsed_time_s"] = time.perf_counter() - started
        report_path = report.get("report_path", report_path)
        try:
            _write_report(report_path, report)
        except (OSError, ValueError) as exc:
            report["success"] = False
            print(f"FAILED [report]: {exc}", file=sys.stderr, flush=True)
    if report["success"]:
        print(
            f"OK: {report['interpolation']}, {report['trajectory_duration_s']:.3f} s; "
            f"clearance planned/actual {report['planned_min_clearance_m'] * 1000:.2f}/"
            f"{report['actual_min_clearance_m'] * 1000:.2f} mm; "
            f"TCP error {report['final_tcp_position_error_mm']:.3f} mm / "
            f"{report['final_tcp_orientation_error_deg']:.3f} deg; "
            f"placement error {report['placement_error_mm']:.3f} mm; report: {report_path}",
            flush=True,
        )
    return 0 if report["success"] else 1
