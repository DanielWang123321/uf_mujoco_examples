# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Pick up several (2 or 3) sideways paper cups, flip each upright, and nest them into a stack.

Run ``python -m examples.stack_cups --help`` for options. ``--mode sim`` runs MuJoCo only;
the hardware modes are implemented in ``sim2real.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from ufactory_mujoco.calibration import KinematicsCalibration
from ufactory_mujoco.grippers import bodies_in_contact
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

from common import cli
from common.geometry import angle_deg, rotate_axis_up_quat, yaw_rotate_quat
from common.objects.paper_cup import CUP_HEIGHT_M, GRASP_AXIS_OFFSET_M, sideways_cup_mujoco_pose
from common.sim import (
    TaskError,
    TaskSimRobot,
    body_touches_geom,
    configure_finger_contact_priority,
    contact_between,
    move_tcp_checked,
)
from common.sim2real import write_json_report

SCENE = Path(__file__).with_name("scene.xml")
DEFAULT_CONFIG_YAML = Path(__file__).with_name("config.yaml")

ALL_CUP_BODIES: tuple[str, ...] = ("paper_cup_1", "paper_cup_2", "paper_cup_3")

DEFAULT_TARGET_XY = np.array([0.30, 0.16], dtype=np.float64)
GRASP_TCP_Z_M = 0.014
INITIAL_GRASP_GAP_MM = 53.0  # cup outer diameter at the grasp height is 63.0 mm
NEST_HOVER_CLEARANCE_M = 0.025  # hover the cup bottom 25 mm above the lower cup's rim
NEST_INSERT_DEPTH_M = 0.004  # lower the cup bottom 4 mm into the lower cup's mouth (fingers stay ~22 mm above the rim)
MIN_NEST_STEP_Z_M = 0.004  # min bottom-to-bottom height step between nested cups
MAX_NEST_STEP_Z_M = 0.035  # max bottom-to-bottom height step (larger means jammed near the rim)
MAX_COAXIAL_XY_ERR_M = 0.008  # max XY offset between adjacent nested cups
MAX_TARGET_XY_ERR_M = 0.030  # max XY error of the stack relative to target_xy
MAX_TILT_DEG = 10.0
MAX_REST_DRIFT_M = 0.002

MAX_GRASP_SHIFT_M = 0.025
MAX_GRASP_ROTATION_DEG = 25.0
MAX_CONTACT_LOSS_S = 0.10


@dataclass(frozen=True)
class InitialCupPose:
    """Initial sideways paper cup pose on the table."""

    bottom_xy_m: tuple[float, float]
    yaw_deg: float = 0.0
    table_z_m: float = 0.0

    def mujoco_pos_quat(self) -> tuple[np.ndarray, np.ndarray]:
        return sideways_cup_mujoco_pose(self.bottom_xy_m, self.yaw_deg, table_z_m=self.table_z_m)


@dataclass
class CupCycleWaypoints:
    """Joint waypoints recorded for one cup's pick-flip-stack cycle."""

    cup_name: str
    approach_waypoints: list[np.ndarray] = field(default_factory=list)
    lift_flip_place_waypoints: list[np.ndarray] = field(default_factory=list)
    retreat_waypoints: list[np.ndarray] = field(default_factory=list)
    grasp_gap_mm: float = INITIAL_GRASP_GAP_MM


def _cup_index(cup_name: str) -> int:
    for idx, name in enumerate(ALL_CUP_BODIES, start=1):
        if name == cup_name:
            return idx
    raise ValueError(f"Unknown cup body name: {cup_name!r}")


def _cup_bottom_geom(cup_name: str) -> str:
    return f"cup_{_cup_index(cup_name)}_bottom"


def _cup_pose(robot: TaskSimRobot, cup_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Cup bottom position and bottom->rim axis in the world frame."""
    return robot.body_pose_axis(cup_name)


def _tool_relative_cup(robot: TaskSimRobot, cup_name: str) -> tuple[np.ndarray, np.ndarray]:
    return robot.tool_relative_pose_axis(cup_name)


def _contact_between(robot: TaskSimRobot, geom_a: str, geom_b: str) -> bool:
    return contact_between(robot.model, robot.data, geom_a, geom_b)


@dataclass
class GraspMonitor:
    """Abort on loss of bilateral finger contact or excessive cup motion relative to the TCP."""

    cup_name: str
    position: np.ndarray
    axis: np.ndarray
    lost_s: float = 0.0
    max_lost_s: float = 0.0
    max_shift_m: float = 0.0
    max_rotation_deg: float = 0.0

    @classmethod
    def start(cls, robot: TaskSimRobot, cup_name: str) -> GraspMonitor:
        position, axis = _tool_relative_cup(robot, cup_name)
        return cls(cup_name, position, axis)

    def observe(self, robot: TaskSimRobot) -> None:
        position, axis = _tool_relative_cup(robot, self.cup_name)
        shift = float(np.linalg.norm(position - self.position))
        angle = angle_deg(axis, self.axis)
        self.max_shift_m = max(self.max_shift_m, shift)
        self.max_rotation_deg = max(self.max_rotation_deg, angle)
        self.lost_s = (
            0.0
            if robot.bilateral_finger_contact(self.cup_name)
            else (self.lost_s + robot.model.opt.timestep)
        )
        self.max_lost_s = max(self.max_lost_s, self.lost_s)
        if not np.isfinite(shift) or not np.isfinite(angle):
            raise TaskError(f"Invalid pose of {self.cup_name}")
        if self.lost_s > MAX_CONTACT_LOSS_S:
            raise TaskError(
                f"{self.cup_name} slipped: bilateral contact lost for {self.lost_s:.3f} s"
            )
        if shift > MAX_GRASP_SHIFT_M or angle > MAX_GRASP_ROTATION_DEG:
            raise TaskError(
                f"{self.cup_name} slipped: relative shift {shift * 1000:.1f} mm, "
                f"rotation {angle:.1f} deg"
            )


class StackCupRobot(TaskSimRobot):
    """Task robot for several cups (G2 force feedback on the active cup, priority finger friction)."""

    def __init__(self, *, viewer: bool, calibration: KinematicsCalibration | None = None):
        super().__init__(SCENE, viewer=viewer, calibration=calibration)

    def _configure_model(self) -> None:
        # High-friction fingers win via contact priority; cup-to-cup contacts keep low friction.
        configure_finger_contact_priority(self.model)


def _require_bilateral(robot: StackCupRobot, cup_name: str) -> None:
    if not robot.bilateral_finger_contact(cup_name):
        raise TaskError(f"No bilateral finger contact on {cup_name}; refusing to lift or release")


def _configure_initial_scene(
    robot: StackCupRobot,
    num_cups: int,
    cup_poses: list[InitialCupPose] | None,
    target_xy: np.ndarray,
    q_init: np.ndarray | None,
) -> tuple[list[str], float]:
    """Place the active cups, hide unused ones, and move the target marker; return (cups, table Z)."""
    if num_cups < 2 or num_cups > len(ALL_CUP_BODIES):
        raise ValueError(f"num_cups must be in [2, {len(ALL_CUP_BODIES)}], got: {num_cups}")

    if q_init is not None:
        robot.set_arm_qpos(q_init)

    table_z = 0.0
    if cup_poses is not None:
        if len(cup_poses) < num_cups:
            raise ValueError(
                f"Got {len(cup_poses)} initial cup poses, fewer than num_cups ({num_cups})"
            )
        table_z = float(cup_poses[0].table_z_m)
        for idx in range(num_cups):
            pos, quat = cup_poses[idx].mujoco_pos_quat()
            robot.set_free_joint_pose(f"{ALL_CUP_BODIES[idx]}_joint", pos, quat)

    # Park unused cups (e.g. paper_cup_3 when num_cups == 2) and disable their collision/visibility.
    for idx, cup_name in enumerate(ALL_CUP_BODIES):
        if idx < num_cups:
            continue
        robot.set_free_joint_pose(
            f"{cup_name}_joint",
            np.array([-0.60 - 0.15 * idx, 0.0, 0.0375]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        b_id = robot.model.body(cup_name).id
        for g_id in range(robot.model.ngeom):
            if int(robot.model.geom_bodyid[g_id]) == b_id:
                robot.model.geom_contype[g_id] = 0
                robot.model.geom_conaffinity[g_id] = 0
                robot.model.geom_rgba[g_id, 3] = 0.0

    table_geom_id = robot.model.geom("table").id
    robot.model.geom_pos[table_geom_id, 2] = table_z

    target_geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    if target_geom_id >= 0:
        robot.model.geom_pos[target_geom_id] = [
            float(target_xy[0]),
            float(target_xy[1]),
            table_z + 0.0002,
        ]
    mujoco.mj_forward(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.eval_data)
    return list(ALL_CUP_BODIES[:num_cups]), table_z


def _verify_stack_rest(
    robot: StackCupRobot,
    active_cups: list[str],
    *,
    target_xy: np.ndarray,
    duration: float = 1.0,
) -> dict[str, Any]:
    """Require ``duration`` seconds of all cups standing upright, nested coaxially, without fingers."""
    robot.check_viewer()
    up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    start_positions = {cup: _cup_pose(robot, cup)[0] for cup in active_cups}
    max_drift = 0.0
    max_tilt = 0.0
    max_target_error = 0.0
    max_coaxial_error = 0.0
    start_time = float(robot.data.time)

    while robot.data.time - start_time < duration:
        robot._step()
        poses = {cup: _cup_pose(robot, cup) for cup in active_cups}

        for cup in active_cups:
            bottom, axis = poses[cup]
            if not np.all(np.isfinite(bottom)) or not np.all(np.isfinite(axis)):
                raise TaskError(f"Invalid final pose of {cup}")
            if bodies_in_contact(robot.model, robot.data, "left_finger", cup) or bodies_in_contact(
                robot.model, robot.data, "right_finger", cup
            ):
                raise TaskError(f"Rest check failed: gripper still touches {cup}")

            tilt = angle_deg(axis, up_vec)
            err = float(np.linalg.norm(bottom[:2] - target_xy))
            drift = float(np.linalg.norm(bottom - start_positions[cup]))
            max_tilt = max(max_tilt, tilt)
            max_target_error = max(max_target_error, err)
            max_drift = max(max_drift, drift)

            if tilt > MAX_TILT_DEG or err > MAX_TARGET_XY_ERR_M or drift > MAX_REST_DRIFT_M:
                raise TaskError(
                    f"Rest check failed ({cup}): tilt {tilt:.1f} deg, placement error "
                    f"{err * 1000:.1f} mm, drift {drift * 1000:.1f} mm"
                )

        # The base cup must stay on the table.
        base_cup = active_cups[0]
        if not _contact_between(robot, "table", _cup_bottom_geom(base_cup)):
            raise TaskError(f"Rest check failed: base cup {base_cup} bottom lost table contact")

        # Each upper cup must touch the cup below it and sit nested inside it.
        for idx in range(1, len(active_cups)):
            lower_cup = active_cups[idx - 1]
            upper_cup = active_cups[idx]
            bot_low, _ = poses[lower_cup]
            bot_high, _ = poses[upper_cup]
            dz = float(bot_high[2] - bot_low[2])
            coax_err = float(np.linalg.norm(bot_high[:2] - bot_low[:2]))
            max_coaxial_error = max(max_coaxial_error, coax_err)

            if not bodies_in_contact(robot.model, robot.data, lower_cup, upper_cup):
                raise TaskError(
                    f"Stack check failed: {upper_cup} does not touch the lower cup {lower_cup}"
                )
            if not (MIN_NEST_STEP_Z_M <= dz <= MAX_NEST_STEP_Z_M):
                raise TaskError(
                    f"Stack check failed: nesting step of {upper_cup} on {lower_cup} is "
                    f"{dz * 1000:.1f} mm, outside [{MIN_NEST_STEP_Z_M * 1000:.0f}, "
                    f"{MAX_NEST_STEP_Z_M * 1000:.0f}] mm"
                )
            if coax_err > MAX_COAXIAL_XY_ERR_M:
                raise TaskError(
                    f"Stack check failed: coaxial offset between {upper_cup} and {lower_cup} "
                    f"too large ({coax_err * 1000:.1f} mm > {MAX_COAXIAL_XY_ERR_M * 1000:.1f} mm)"
                )

    final_poses = {cup: _cup_pose(robot, cup) for cup in active_cups}
    nested_steps_mm = [
        float((final_poses[active_cups[i]][0][2] - final_poses[active_cups[i - 1]][0][2]) * 1000.0)
        for i in range(1, len(active_cups))
    ]
    top_rim_z_mm = float((final_poses[active_cups[-1]][0][2] + CUP_HEIGHT_M) * 1000.0)

    return {
        "num_cups": len(active_cups),
        "tilt_deg": max_tilt,
        "error_mm": max_target_error * 1000.0,
        "coaxial_error_mm": max_coaxial_error * 1000.0,
        "drift_mm": max_drift * 1000.0,
        "stable_s": float(robot.data.time - start_time),
        "nested_steps_mm": nested_steps_mm,
        "stack_top_rim_z_mm": top_rim_z_mm,
    }


def run_stack_cups(
    robot: StackCupRobot,
    *,
    num_cups: int = 2,
    cup_poses: list[InitialCupPose] | None = None,
    target_xy: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    recorded_cycles: list[CupCycleWaypoints] | None = None,
) -> dict[str, Any]:
    """Pick, flip 90 degrees, and coaxially nest ``num_cups`` cups. Cups move only via contacts.

    When ``recorded_cycles`` is given, the commanded joint waypoints of each cup cycle are
    recorded for replay on the real robot.
    """
    active_target_xy = (
        DEFAULT_TARGET_XY.copy()
        if target_xy is None
        else np.asarray(target_xy, dtype=np.float64).reshape(2)
    )
    active_cups, table_z = _configure_initial_scene(
        robot, num_cups, cup_poses, active_target_xy, q_init
    )
    home_q = robot.model.key("home").qpos[: robot.n_arm].copy()
    transition_q = (
        np.asarray(q_init, dtype=np.float64).reshape(robot.n_arm).copy()
        if q_init is not None
        else home_q
    )

    print(f"Prepare: open gripper and let {len(active_cups)} cups settle", flush=True)
    robot.open_gripper(speed_percent=40, force_percent=10)
    robot.settle(1.0)
    robot.check_viewer()
    initial_poses_map = {name: _cup_pose(robot, name) for name in active_cups}

    max_lift_mm = 0.0
    max_shift_mm = 0.0
    max_rot_deg = 0.0
    max_loss_s = 0.0

    for idx, cup_name in enumerate(active_cups):
        tag = f"[{idx + 1}/{len(active_cups)}]"
        cycle_wps = CupCycleWaypoints(cup_name=cup_name)
        if recorded_cycles is not None:
            recorded_cycles.append(cycle_wps)

        # Between cups, pass through a high transition posture so the gripper stays above all
        # cups while the wrist unflips; sim2real blends through this via-point without stopping.
        if idx > 0:
            robot.active_waypoint_sink = cycle_wps.approach_waypoints
            robot.move_joints(transition_q, 2.0, wait=False)
            robot.check_viewer()

        initial_bottom, axis = initial_poses_map[cup_name]
        grasp_center = initial_bottom + GRASP_AXIS_OFFSET_M * axis
        grasp = np.array([grasp_center[0], grasp_center[1], table_z + GRASP_TCP_Z_M])
        approach = grasp.copy()
        approach[2] = table_z + 0.16

        axis_yaw_rad = float(np.arctan2(axis[1], axis[0]))
        grasp_quat = yaw_rotate_quat(axis_yaw_rad, GRIP_DOWN_QUAT)

        print(f"{tag} Approach the body of {cup_name}", flush=True)
        robot.active_waypoint_sink = cycle_wps.approach_waypoints
        q_seed = robot.data.qpos.copy()
        q_seed[: robot.n_arm] = transition_q
        q_app, ok = robot.solve_tcp(approach, grasp_quat, q_init=q_seed)
        robot.check_viewer()
        if not ok:
            raise TaskError(f"IK did not converge for the {cup_name} approach pose {approach}")
        robot.move_joints(q_app[: robot.n_arm], 2.2, wait=False)
        robot.check_viewer()
        move_tcp_checked(robot, grasp, grasp_quat, 1.8, steps=30, wait=True)
        robot.active_waypoint_sink = None

        print(f"{tag} Grasp {cup_name}: confirm bilateral contact", flush=True)
        robot.set_gripper_command(INITIAL_GRASP_GAP_MM, speed_percent=20, force_percent=10)
        robot.settle(1.6)
        robot.check_viewer()
        # Back off the initial force-limited closure by 1 mm before enabling force feedback.
        gap = max(0.0, robot.get_measured_gripper_gap() * 1000.0 - 1.0)
        robot.set_gripper_command(gap, speed_percent=20, force_percent=10)
        robot.settle(0.8)
        robot.check_viewer()
        _require_bilateral(robot, cup_name)

        monitor = GraspMonitor.start(robot, cup_name)
        robot.grasp_monitor = monitor
        robot.force_body = cup_name
        robot.force_feedback = True
        robot.last_force_update = -1.0

        print(f"{tag} Lift and flip {cup_name}", flush=True)
        robot.active_waypoint_sink = cycle_wps.lift_flip_place_waypoints
        move_tcp_checked(
            robot, grasp + np.array([0.0, 0.0, 0.14]), grasp_quat, 1.8, steps=28, wait=False
        )
        lifted_bottom, _ = _cup_pose(robot, cup_name)
        lift_mm = float((lifted_bottom[2] - initial_bottom[2]) * 1000.0)
        max_lift_mm = max(max_lift_mm, lift_mm)
        if lifted_bottom[2] < initial_bottom[2] + 0.05 or body_touches_geom(
            robot.model, robot.data, cup_name, "table"
        ):
            raise TaskError(f"Lift failed: {cup_name} did not leave the table")

        # With the gripper still vertical, align the cup axis with +Y (target on the left) or -Y
        # (target on the right), then flip the cup upright in one pitch move toward that side.
        place_side = -1.0 if active_target_xy[1] <= 0.0 else 1.0
        pre_flip_quat = yaw_rotate_quat(place_side * 0.5 * np.pi, GRIP_DOWN_QUAT)
        mid_pos = np.array([0.36, -0.05 * place_side, table_z + 0.22], dtype=np.float64)
        move_tcp_checked(robot, mid_pos, pre_flip_quat, 2.2, steps=35, wait=True)
        position, quat = robot.get_tcp_pose()
        _, axis = _cup_pose(robot, cup_name)
        move_tcp_checked(robot, position, rotate_axis_up_quat(axis, quat), 2.6, steps=45, wait=True)

        if idx == 0:
            print(f"{tag} Place base cup {cup_name}", flush=True)
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            hover_place = (
                position
                + np.r_[
                    active_target_xy - bottom[:2],
                    (table_z + NEST_HOVER_CLEARANCE_M) - bottom[2],
                ]
            )
            move_tcp_checked(robot, hover_place, quat, 2.5, steps=35, wait=True)

            target_z_touch = 0.0001 if q_init is None else -0.0004
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            destination = (
                position
                + np.r_[
                    active_target_xy - bottom[:2],
                    (table_z + target_z_touch) - bottom[2],
                ]
            )
            move_tcp_checked(robot, destination, quat, 2.2, steps=35, wait=True)
            cycle_wps.grasp_gap_mm = float(robot.g2_controller.position_mm[0])

            if not _contact_between(robot, "table", _cup_bottom_geom(cup_name)):
                position, quat = robot.get_tcp_pose()
                bottom, _ = _cup_pose(robot, cup_name)
                destination = (
                    position
                    + np.r_[
                        active_target_xy - bottom[:2],
                        (table_z - 0.0005) - bottom[2],
                    ]
                )
                move_tcp_checked(robot, destination, quat, 0.8, steps=12, wait=True)
            robot.active_waypoint_sink = None
            if not _contact_between(robot, "table", _cup_bottom_geom(cup_name)):
                raise TaskError(f"Place failed: {cup_name} bottom does not touch the table")
        else:
            prev_cup = active_cups[idx - 1]
            prev_bottom, prev_axis = _cup_pose(robot, prev_cup)
            prev_tilt = angle_deg(prev_axis, np.array([0.0, 0.0, 1.0]))
            if prev_tilt > MAX_TILT_DEG:
                raise TaskError(
                    f"Lower cup {prev_cup} has tipped over (tilt {prev_tilt:.1f} deg); "
                    "aborting the stack"
                )
            prev_rim_z = float(prev_bottom[2]) + CUP_HEIGHT_M

            print(f"{tag} Align coaxially and nest {cup_name} into {prev_cup}", flush=True)
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            hover_dest = (
                position
                + np.r_[
                    prev_bottom[:2] - bottom[:2],
                    (prev_rim_z + NEST_HOVER_CLEARANCE_M) - bottom[2],
                ]
            )
            move_tcp_checked(robot, hover_dest, quat, 2.5, steps=35, wait=True)

            # Re-measure the live cup poses, then lower into the lower cup's mouth (fingers stay
            # above its rim).
            prev_bottom, _ = _cup_pose(robot, prev_cup)
            prev_rim_z = float(prev_bottom[2]) + CUP_HEIGHT_M
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            insert_dest = (
                position
                + np.r_[
                    prev_bottom[:2] - bottom[:2],
                    (prev_rim_z - NEST_INSERT_DEPTH_M) - bottom[2],
                ]
            )
            move_tcp_checked(robot, insert_dest, quat, 2.2, steps=35, wait=True)
            cycle_wps.grasp_gap_mm = float(robot.g2_controller.position_mm[0])
            robot.active_waypoint_sink = None

        _require_bilateral(robot, cup_name)
        robot.force_feedback = False
        robot.force_body = None
        robot.grasp_monitor = None

        max_shift_mm = max(max_shift_mm, monitor.max_shift_m * 1000.0)
        max_rot_deg = max(max_rot_deg, monitor.max_rotation_deg)
        max_loss_s = max(max_loss_s, monitor.max_lost_s)

        print(f"{tag} Release {cup_name} and retreat", flush=True)
        robot.open_gripper(speed_percent=25, force_percent=40)
        robot.settle(0.6)
        robot.check_viewer()

        robot.active_waypoint_sink = cycle_wps.retreat_waypoints
        position, quat = robot.get_tcp_pose()
        retreat_pos = position + np.array([0.0, -0.095 * place_side, 0.015], dtype=np.float64)
        move_tcp_checked(robot, retreat_pos, quat, 1.5, steps=22, wait=False)
        move_tcp_checked(
            robot, retreat_pos + np.array([0.0, 0.0, 0.10]), quat, 1.4, steps=20, wait=True
        )
        robot.active_waypoint_sink = None
        robot.settle(0.3)
        robot.check_viewer()

    metrics = _verify_stack_rest(robot, active_cups, target_xy=active_target_xy)
    metrics.update(
        lift_mm=max_lift_mm,
        max_grasp_shift_mm=max_shift_mm,
        max_grasp_rotation_deg=max_rot_deg,
        max_contact_loss_s=max_loss_s,
    )
    steps_str = ", ".join(f"{s:.1f} mm" for s in metrics["nested_steps_mm"])
    print(
        f"Success: stacked {metrics['num_cups']} cups; target error {metrics['error_mm']:.2f} mm, "
        f"coaxial error {metrics['coaxial_error_mm']:.2f} mm, nesting steps [{steps_str}], "
        f"max tilt {metrics['tilt_deg']:.2f} deg, stable {metrics['stable_s']:.3f} s.",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def load_config(config_path: str | Path = DEFAULT_CONFIG_YAML) -> dict[str, Any]:
    """Load this example's YAML config (``{}`` if missing)."""
    return cli.load_config(config_path)


def _build_parser():
    parser = cli.build_parser(__doc__, default_config=DEFAULT_CONFIG_YAML)
    parser.add_argument(
        "--num-cups",
        type=int,
        choices=(2, 3),
        default=None,
        help="Number of cups to pick and stack (default: 2)",
    )
    parser.add_argument(
        "--target-xy",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
        default=None,
        help="Stack target position",
    )
    parser.add_argument(
        "--cup-poses",
        nargs="+",
        type=float,
        metavar="VAL",
        default=None,
        help="Sideways cup poses as X1 Y1 YAW1 X2 Y2 YAW2 ... (skips camera detection)",
    )
    return parser


def _run(args) -> None:
    opts = cli.resolve_run_options(args)
    task_cfg = opts.task
    num_cups = int(args.num_cups) if args.num_cups is not None else int(task_cfg.get("num_cups", 2))

    if opts.mode == "sim":
        if args.target_xy is not None:
            target_xy = np.asarray(args.target_xy, dtype=np.float64)
        elif "target_xy" in task_cfg:
            target_xy = np.asarray(task_cfg["target_xy"], dtype=np.float64)
        else:
            target_xy = DEFAULT_TARGET_XY.copy()

        cup_poses: list[InitialCupPose] | None = None
        if args.cup_poses is not None:
            vals = list(args.cup_poses)
            if len(vals) % 3 != 0 or len(vals) // 3 < num_cups:
                raise TaskError(
                    f"--cup-poses needs a multiple of 3 values covering at least {num_cups} cups "
                    f"({3 * num_cups} values)"
                )
            cup_poses = [
                InitialCupPose(
                    bottom_xy_m=(float(vals[3 * i]), float(vals[3 * i + 1])),
                    yaw_deg=float(vals[3 * i + 2]),
                )
                for i in range(len(vals) // 3)
            ]

        with StackCupRobot(viewer=not opts.headless) as robot:
            metrics = run_stack_cups(
                robot,
                num_cups=num_cups,
                cup_poses=cup_poses,
                target_xy=target_xy,
            )
        write_json_report(
            opts.report_path,
            {
                "mode": "sim",
                "ok": True,
                "success": True,
                "num_cups": num_cups,
                "target_xy_m": target_xy.tolist(),
                "metrics": metrics,
            },
        )
        return

    from .sim2real import run_sim2real_pipeline

    target_override: tuple[float, float] | None = None
    if args.target_xy is not None:
        target_override = (float(args.target_xy[0]), float(args.target_xy[1]))
    elif task_cfg.get("target_xy") is not None:
        t_cfg = task_cfg["target_xy"]
        target_override = (float(t_cfg[0]), float(t_cfg[1]))
    run_sim2real_pipeline(
        mode=opts.mode,
        num_cups=num_cups,
        ip=opts.ip,
        calibration_path=opts.calibration_path,
        handeye_path=opts.handeye_path,
        camera_serial=opts.camera_serial,
        cup_poses_override=args.cup_poses,
        target_xy_override=target_override,
        headless=opts.headless,
        confirm_real=opts.confirm_real,
        cycles=opts.cycles,
        post_verify=opts.post_verify,
        save_vis_path=opts.save_vis,
        save_after_vis_path=opts.save_after_vis,
        report_path=opts.report_path,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return cli.run_cli(lambda: _run(args))
