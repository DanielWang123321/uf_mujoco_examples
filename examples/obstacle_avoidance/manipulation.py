# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Physical pick/lift/place phases; payload motion is driven only by contacts."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from ufactory_mujoco.grippers import bodies_in_contact
from ufactory_mujoco.ik import solve_ik_trajectory

from .planning import (
    GRIPPER_QPOS,
    PAYLOAD,
    SITE,
    CarriedState,
    MotionError,
    Options,
    SceneChecker,
    endpoint_candidates,
    pose_to_target,
    time_parameterize,
)


def bottom_height(model, data) -> float:
    body = data.body(PAYLOAD)
    half_size = model.geom("payload_geom").size
    return float(body.xpos[2] - np.abs(body.xmat.reshape(3, 3)[2]) @ half_size)


def on_table(model, data) -> bool:
    table, payload = model.geom("table").id, model.geom("payload_geom").id
    return any(
        {int(c.geom1), int(c.geom2)} == {table, payload} and c.dist <= 1e-4
        for c in data.contact[: data.ncon]
    )


@dataclass
class PayloadMonitor:
    reference: CarriedState
    max_shift_m: float = 0
    max_rotation_deg: float = 0
    lost_s: float = 0
    max_lost_s: float = 0

    def observe(self, model, data, dt: float) -> None:
        state = CarriedState.from_data(data)
        shift = float(np.linalg.norm(state.position - self.reference.position))
        rotation = float(
            np.rad2deg(Rotation.from_matrix(state.rotation @ self.reference.rotation.T).magnitude())
        )
        self.max_shift_m = max(self.max_shift_m, shift)
        self.max_rotation_deg = max(self.max_rotation_deg, rotation)
        bilateral = all(
            bodies_in_contact(model, data, finger, PAYLOAD, max_dist=1e-4)
            for finger in ("left_finger", "right_finger")
        )
        self.lost_s = 0 if bilateral else self.lost_s + dt
        self.max_lost_s = max(self.max_lost_s, self.lost_s)
        if shift > 0.005 or rotation > 5 or self.lost_s > 0.1:
            raise MotionError(
                "grasp_monitor",
                f"payload slipped: {shift * 1000:.2f} mm / {rotation:.2f} deg; "
                f"bilateral contact lost for {self.lost_s:.3f} s",
            )


def initialize(robot, options: Options, report: dict) -> None:
    """Set only the initial robot pose; the object starts on the table under gravity."""
    report["stage"] = "initialization"
    checker = SceneChecker(robot.scene_path, clearance_m=options.clearance_m)
    candidates = endpoint_candidates(checker, options.start_pose, seed=options.seed, name="start")
    robot.options, robot.report = options, report
    robot.runtime_checker = SceneChecker(
        robot.scene_path, clearance_m=options.execution_clearance_m
    )
    robot.data.qpos[:] = checker.home
    robot.data.qvel[:] = 0
    robot.set_arm_qpos(candidates[0])
    robot.data.ctrl[robot.n_arm :] = checker.home[GRIPPER_QPOS][0]
    robot.g2_controller.reset(position_mm=84)
    robot.set_gripper_command(84, speed_percent=40, force_percent=20)
    robot.gripper_expected = "open"
    robot.model.geom_pos[robot.model.geom("place_target").id, :2] = (
        np.asarray(options.goal_pose[:2]) / 1000
    )
    mujoco.mj_forward(robot.model, robot.data)
    robot.start_viewer()
    robot._wall_start = time.monotonic() - robot.data.time
    robot.observe()
    robot.settle(0.4)
    report["initial_payload_position_m"] = robot.data.body(PAYLOAD).xpos.copy().tolist()
    if not on_table(robot.model, robot.data):
        raise MotionError("initialization", "payload is not supported by the table")


def cartesian_move(robot, pos, quat, *, stage: str, holding: bool, allow_table: bool) -> None:
    """Precheck a complete local Cartesian phase before advancing any physics."""
    robot.report["stage"] = stage
    mujoco.mj_forward(robot.model, robot.data)
    carried = CarriedState.from_data(robot.data) if holding else None
    checker = SceneChecker(robot.scene_path, clearance_m=robot.options.clearance_m, carried=carried)
    if not holding:
        adr = checker.payload_adr
        checker.home[adr : adr + 7] = robot.data.qpos[adr : adr + 7]
    checker.allow_table = allow_table
    start_pos, start_quat = robot.get_tcp_pose()
    length = float(np.linalg.norm(np.asarray(pos) - start_pos))
    rotations = Rotation.from_quat(np.array([start_quat, quat])[:, [1, 2, 3, 0]])
    angle = float((rotations[1] * rotations[0].inv()).magnitude())
    count = max(2, int(np.ceil(length / 0.01)), int(np.ceil(angle / 0.08)))
    u = np.linspace(0, 1, count + 1)[1:]
    positions = start_pos + u[:, None] * (np.asarray(pos) - start_pos)
    quaternions = Slerp([0, 1], rotations)(u).as_quat()[:, [3, 0, 1, 2]]
    data = mujoco.MjData(checker.model)
    q_path, ok = solve_ik_trajectory(
        checker.model,
        data,
        SITE,
        positions,
        quaternions,
        robot.data.qpos.copy(),
        pos_tol=5e-5,
        rot_tol=2e-4,
        max_iter=250,
    )
    if not ok:
        raise MotionError(stage, f"IK failed for {stage}")
    path = np.vstack((robot.get_joint_positions(), q_path[:, : robot.n_arm]))
    try:
        trajectory, method, minimum, rejection = time_parameterize(path, checker, robot.options)
    except MotionError as exc:
        raise MotionError(stage, str(exc)) from exc
    robot.report.setdefault("local_phases", []).append(
        {
            "stage": stage,
            "duration_s": trajectory.duration,
            "interpolation": method,
            "planned_min_clearance_m": minimum,
            "spline_rejection": rejection,
        }
    )
    robot.runtime_checker.allow_table = allow_table
    robot.runtime_checker.allow_fingers = holding
    robot._wall_start = time.monotonic() - robot.data.time
    for q in trajectory.q[1:]:
        robot.set_joint_positions(q)
        robot._step()
    robot.settle(0.35)
    mujoco.mj_forward(robot.model, robot.data)


def pickup(robot, options: Options, report: dict) -> PayloadMonitor:
    pos = robot.data.body(PAYLOAD).xpos.copy()
    half_height = float(robot.model.geom("payload_geom").size[2])
    initial_bottom = bottom_height(robot.model, robot.data)
    _, down_quat = pose_to_target([0, 0, 0, 180, 0, 0])
    approach = np.array([pos[0], pos[1], 0.15])
    grasp = np.array([pos[0], pos[1], pos[2] + 0.005])
    print("Approach and grasp the table-supported block", flush=True)
    cartesian_move(robot, approach, down_quat, stage="approach", holding=False, allow_table=True)
    cartesian_move(robot, grasp, down_quat, stage="grasp_descent", holding=False, allow_table=True)
    report["stage"] = "grasp"
    robot.runtime_checker.allow_fingers = True
    robot.gripper_expected = "transition"
    robot.set_gripper_command(45, speed_percent=20, force_percent=20)
    robot.settle(2)
    mujoco.mj_forward(robot.model, robot.data)
    if not robot.bilateral_finger_contact(PAYLOAD):
        raise MotionError("grasp", "no bilateral finger contact; refusing to lift")
    gap_mm = robot.get_measured_gripper_gap() * 1000
    if not 47 <= gap_mm <= 53:
        raise MotionError("grasp", f"unexpected block grasp opening: {gap_mm:.2f} mm")
    report["grasp_gap_mm"] = gap_mm
    report["grasp_forces_n"] = list(robot.get_gripper_contact_forces(PAYLOAD))
    monitor = PayloadMonitor(CarriedState.from_data(robot.data))
    robot.payload_monitor = monitor
    robot.gripper_expected = "holding"
    print("Lift the block, then plan with its measured grasp geometry", flush=True)
    hover_pos, hover_quat = pose_to_target(options.start_pose)
    cartesian_move(robot, hover_pos, hover_quat, stage="lift", holding=True, allow_table=True)
    height = bottom_height(robot.model, robot.data)
    if height < initial_bottom + 0.10 or on_table(robot.model, robot.data):
        raise MotionError("lift", "payload did not leave the table")
    report["lift_mm"] = (height - initial_bottom) * 1000
    report["payload_mass_kg"] = float(robot.model.body(PAYLOAD).mass[0])
    report["payload_size_m"] = (robot.model.geom("payload_geom").size * 2).tolist()
    report["payload_half_height_m"] = half_height
    robot.runtime_checker.allow_table = False
    robot.observe()
    return monitor


def place(robot, options: Options, report: dict, monitor: PayloadMonitor) -> None:
    print("Lower, release, and verify the block on the table", flush=True)
    position, quat = robot.get_tcp_pose()
    # First return to a vertical placement orientation above the destination.
    target = np.asarray(options.goal_pose[:2], dtype=float) / 1000
    _, down_quat = pose_to_target([0, 0, 0, 180, 0, 0])
    cartesian_move(
        robot, position, down_quat, stage="place_orientation", holding=True, allow_table=False
    )
    position, quat = robot.get_tcp_pose()
    body_pos = robot.data.body(PAYLOAD).xpos.copy()
    alignment = position + np.r_[target - body_pos[:2], 0]
    cartesian_move(robot, alignment, quat, stage="place_alignment", holding=True, allow_table=False)
    position, quat = robot.get_tcp_pose()
    bottom = bottom_height(robot.model, robot.data)
    destination = position + np.array([0, 0, -0.0003 - bottom])
    cartesian_move(robot, destination, quat, stage="place_descent", holding=True, allow_table=True)
    if not on_table(robot.model, robot.data):
        raise MotionError("place", "payload has not contacted the table; refusing release")
    robot.payload_monitor = None
    report["stage"] = "release"
    robot.gripper_expected = "transition"
    robot.set_gripper_command(84, speed_percent=40, force_percent=20)
    robot.settle(1.2)
    robot.gripper_expected = "open"
    robot.runtime_checker.allow_fingers = False
    robot.observe()
    retreat, quat = pose_to_target(options.goal_pose)
    cartesian_move(robot, retreat, quat, stage="retreat", holding=False, allow_table=True)
    report["stage"] = "placement_validation"
    start = robot.data.body(PAYLOAD).xpos.copy()
    max_drift = max_tilt = max_error = 0.0
    for _ in range(round(1 / options.timestep_s)):
        robot._step()
        data = robot.runtime_checker.data
        body = data.body(PAYLOAD)
        drift = float(np.linalg.norm(body.xpos - start))
        error = float(np.linalg.norm(body.xpos[:2] - target))
        tilt = float(np.rad2deg(np.arccos(np.clip(body.xmat.reshape(3, 3)[2, 2], -1, 1))))
        max_drift, max_tilt, max_error = (
            max(max_drift, drift),
            max(max_tilt, tilt),
            max(max_error, error),
        )
        if not on_table(robot.runtime_checker.model, data):
            raise MotionError("placement_validation", "payload lost table support")
        if any(
            bodies_in_contact(robot.runtime_checker.model, data, name, PAYLOAD)
            for name in ("left_finger", "right_finger")
        ):
            raise MotionError("placement_validation", "payload still has finger support")
        if (
            error > 0.005
            or drift > 0.001
            or tilt > 5
            or abs(bottom_height(robot.model, data)) > 0.001
        ):
            raise MotionError("placement_validation", "block placement is inaccurate or unstable")
    report.update(
        placement_error_mm=max_error * 1000,
        placement_drift_mm=max_drift * 1000,
        placement_tilt_deg=max_tilt,
        placement_stable_s=1.0,
        final_payload_position_m=robot.data.body(PAYLOAD).xpos.copy().tolist(),
        max_grasp_shift_mm=monitor.max_shift_m * 1000,
        max_grasp_rotation_deg=monitor.max_rotation_deg,
        max_contact_loss_s=monitor.max_lost_s,
    )
