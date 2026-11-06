# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Pick a sideways paper cup, flip it 90 degrees in the air, and place it upright.

``--mode sim`` runs MuJoCo only; the hardware modes are implemented in ``sim2real.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from ufactory_mujoco.calibration import KinematicsCalibration
from ufactory_mujoco.grippers import bodies_in_contact
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

from common import cli
from common.geometry import angle_deg, rotate_axis_up_quat, yaw_rotate_quat
from common.objects.paper_cup import GRASP_AXIS_OFFSET_M
from common.sim import (
    TaskError,
    TaskSimRobot,
    body_touches_geom,
    contact_between,
    move_tcp_checked,
)
from common.sim2real import write_json_report

if TYPE_CHECKING:
    from common.objects.paper_cup import CupPoseEstimate

    from .sim2real import TwinPhaseWaypoints

SCENE = Path(__file__).with_name("scene.xml")
DEFAULT_CONFIG_YAML = Path(__file__).with_name("config.yaml")
CUP = "paper_cup"
TARGET_XY = np.array([0.30, 0.15])
GRASP_TCP_Z_M = 0.014
INITIAL_GRASP_GAP_MM = 53.0  # cup outer diameter at the grasp height is 63.0 mm
MAX_GRASP_SHIFT_M = 0.025
MAX_GRASP_ROTATION_DEG = 25.0
MAX_CONTACT_LOSS_S = 0.10


def _cup_pose(robot: TaskSimRobot) -> tuple[np.ndarray, np.ndarray]:
    """Cup bottom position and bottom->rim axis in the world frame."""
    return robot.body_pose_axis(CUP)


def _tool_relative_cup(robot: TaskSimRobot) -> tuple[np.ndarray, np.ndarray]:
    return robot.tool_relative_pose_axis(CUP)


def _contact_between(robot: TaskSimRobot, geom_a: str, geom_b: str) -> bool:
    return contact_between(robot.model, robot.data, geom_a, geom_b)


def _configure_initial_scene(
    robot: TaskSimRobot,
    cup_pose: CupPoseEstimate | None,
    target_xy: np.ndarray,
    q_init: np.ndarray | None,
) -> float:
    """Apply the optional initial arm state, cup pose, and target marker; return table Z."""
    if q_init is not None:
        robot.set_arm_qpos(q_init)

    table_z = 0.0
    if cup_pose is not None:
        table_z = float(cup_pose.table_z_m)
        robot.set_free_joint_pose(
            "paper_cup_joint", cup_pose.mujoco_pos_m, cup_pose.mujoco_quat_wxyz
        )
        robot.model.geom_pos[robot.model.geom("table").id, 2] = table_z

    target_geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    if target_geom_id >= 0:
        robot.model.geom_pos[target_geom_id] = [
            float(target_xy[0]),
            float(target_xy[1]),
            table_z + 0.0002,
        ]
    mujoco.mj_forward(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.eval_data)
    return table_z


@dataclass
class GraspMonitor:
    """Abort on loss of bilateral finger contact or excessive cup motion relative to the TCP."""

    position: np.ndarray
    axis: np.ndarray
    lost_s: float = 0.0
    max_lost_s: float = 0.0
    max_shift_m: float = 0.0
    max_rotation_deg: float = 0.0

    @classmethod
    def start(cls, robot: TaskSimRobot) -> GraspMonitor:
        position, axis = _tool_relative_cup(robot)
        return cls(position, axis)

    def observe(self, robot: TaskSimRobot) -> None:
        position, axis = _tool_relative_cup(robot)
        shift = float(np.linalg.norm(position - self.position))
        angle = angle_deg(axis, self.axis)
        self.max_shift_m = max(self.max_shift_m, shift)
        self.max_rotation_deg = max(self.max_rotation_deg, angle)
        self.lost_s = (
            0.0 if robot.bilateral_finger_contact(CUP) else (self.lost_s + robot.model.opt.timestep)
        )
        self.max_lost_s = max(self.max_lost_s, self.lost_s)
        if not np.isfinite(shift) or not np.isfinite(angle):
            raise TaskError("Invalid cup pose")
        if self.lost_s > MAX_CONTACT_LOSS_S:
            raise TaskError(f"Cup slipped: bilateral contact lost for {self.lost_s:.3f} s")
        if shift > MAX_GRASP_SHIFT_M or angle > MAX_GRASP_ROTATION_DEG:
            raise TaskError(
                f"Cup slipped: relative shift {shift * 1000:.1f} mm, rotation {angle:.1f} deg"
            )


class CupRobot(TaskSimRobot):
    """Task robot for one paper cup (G2 force feedback: 2 N target on the cup)."""

    def __init__(self, *, viewer: bool, calibration: KinematicsCalibration | None = None):
        super().__init__(SCENE, viewer=viewer, calibration=calibration)
        self.force_body = CUP


def _require_bilateral(robot: CupRobot) -> None:
    if not robot.bilateral_finger_contact(CUP):
        raise TaskError("No bilateral finger contact; refusing to lift")


def _verify_rest(
    robot: CupRobot,
    duration: float = 1.0,
    *,
    target_xy: np.ndarray = TARGET_XY,
) -> dict[str, float]:
    """Require ``duration`` seconds of the cup standing on its bottom without finger support."""
    robot.check_viewer()
    start, _ = _cup_pose(robot)
    max_drift = 0.0
    max_tilt = 0.0
    max_error = 0.0
    start_time = float(robot.data.time)
    while robot.data.time - start_time < duration:
        robot._step()
        bottom, axis = _cup_pose(robot)
        if not np.all(np.isfinite(bottom)) or not np.all(np.isfinite(axis)):
            raise TaskError("Invalid final cup pose")
        tilt = angle_deg(axis, np.array([0.0, 0.0, 1.0]))
        error = float(np.linalg.norm(bottom[:2] - target_xy))
        drift = float(np.linalg.norm(bottom - start))
        max_tilt, max_error, max_drift = (
            max(max_tilt, tilt),
            max(max_error, error),
            max(max_drift, drift),
        )
        if not _contact_between(robot, "table", "cup_bottom"):
            raise TaskError("Rest check failed: cup bottom lost table contact")
        if bodies_in_contact(robot.model, robot.data, "left_finger", CUP) or bodies_in_contact(
            robot.model, robot.data, "right_finger", CUP
        ):
            raise TaskError("Rest check failed: gripper still touches the cup")
        if tilt > 10.0 or error > 0.030 or drift > 0.002:
            raise TaskError(
                f"Rest check failed: tilt {tilt:.1f} deg, placement error {error * 1000:.1f} mm, "
                f"drift {drift * 1000:.1f} mm"
            )
    return {
        "tilt_deg": max_tilt,
        "error_mm": max_error * 1000,
        "drift_mm": max_drift * 1000,
        "stable_s": float(robot.data.time - start_time),
    }


def run_pick_cup(
    robot: CupRobot,
    *,
    cup_pose: CupPoseEstimate | None = None,
    target_xy: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    recorded_phases: TwinPhaseWaypoints | None = None,
) -> dict[str, float]:
    """Run one pick-flip-place cycle. The cup moves only through contacts (no welds).

    When ``recorded_phases`` is given, the commanded joint waypoints of each phase are recorded
    for replay on the real robot.
    """
    active_target_xy = TARGET_XY if target_xy is None else np.asarray(target_xy, dtype=np.float64)
    table_z = _configure_initial_scene(robot, cup_pose, active_target_xy, q_init)

    print("Prepare: open gripper and let the cup settle", flush=True)
    robot.open_gripper(speed_percent=40, force_percent=10)
    robot.settle(1.0)
    robot.check_viewer()
    initial_bottom, axis = _cup_pose(robot)
    grasp_center = initial_bottom + GRASP_AXIS_OFFSET_M * axis
    grasp = np.array([grasp_center[0], grasp_center[1], table_z + GRASP_TCP_Z_M])
    approach = grasp.copy()
    approach[2] = table_z + 0.16

    axis_yaw_rad = float(np.arctan2(axis[1], axis[0]))
    grasp_quat = yaw_rotate_quat(axis_yaw_rad, GRIP_DOWN_QUAT)

    print("Approach the cup body", flush=True)
    if recorded_phases is not None:
        robot.active_waypoint_sink = recorded_phases.approach_waypoints
    q, ok = robot.solve_tcp(approach, grasp_quat, q_init=robot.data.qpos.copy())
    robot.check_viewer()
    if not ok:
        raise TaskError(f"IK did not converge for the approach pose {approach}")
    robot.move_joints(q[: robot.n_arm], 3.0)
    robot.check_viewer()
    move_tcp_checked(robot, grasp, grasp_quat, 2.0, steps=30)
    robot.active_waypoint_sink = None

    print("Grasp: confirm bilateral contact", flush=True)
    robot.set_gripper_command(INITIAL_GRASP_GAP_MM, speed_percent=20, force_percent=10)
    robot.settle(2.0)
    robot.check_viewer()
    # Back off the initial force-limited closure by 1 mm before enabling force feedback.
    gap = max(0.0, robot.get_measured_gripper_gap() * 1000 - 1.0)
    robot.set_gripper_command(gap, speed_percent=20, force_percent=10)
    robot.settle(1.0)
    robot.check_viewer()
    _require_bilateral(robot)
    monitor = GraspMonitor.start(robot)
    robot.grasp_monitor = monitor
    robot.force_feedback = True
    robot.last_force_update = -1.0

    print("Lift and flip", flush=True)
    if recorded_phases is not None:
        robot.active_waypoint_sink = recorded_phases.lift_flip_place_waypoints
    move_tcp_checked(robot, grasp + np.array([0.0, 0.0, 0.13]), grasp_quat, 2.0, steps=30)
    lifted_bottom, _ = _cup_pose(robot)
    if lifted_bottom[2] < initial_bottom[2] + 0.05 or body_touches_geom(
        robot.model, robot.data, CUP, "table"
    ):
        raise TaskError("Lift failed: cup did not leave the table")

    position, quat = robot.get_tcp_pose()
    default_scene = cup_pose is None and np.allclose(active_target_xy, TARGET_XY)
    if default_scene:
        move_tcp_checked(robot, position + np.array([0.12, 0.0, 0.08]), quat, 2.5, steps=40)
        position, quat = robot.get_tcp_pose()
        _, axis = _cup_pose(robot)
        move_tcp_checked(robot, position, rotate_axis_up_quat(axis, quat), 3.0, steps=50)
        # Turn the gripper about the vertical axis for reachability and a clear retreat.
        position, quat = robot.get_tcp_pose()
        yaw = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        yawed = np.empty(4)
        mujoco.mju_mulQuat(yawed, yaw, quat)
        move_tcp_checked(robot, position, yawed, 2.5, steps=50)
        retreat_offset = np.array([0.0, -0.12, 0.0])
    else:
        # With the gripper still vertical, align the cup axis with -Y (target on the right) or
        # +Y (target on the left) so the flip keeps the flange near x = 0.40 m. This avoids the
        # wrist camera pointing at the table and controller self-collision near the base.
        place_side = -1.0 if active_target_xy[1] <= 0.0 else 1.0
        pre_flip_quat = yaw_rotate_quat(place_side * 0.5 * np.pi, GRIP_DOWN_QUAT)
        mid_pos = np.array([0.40, -0.06 * (-place_side), table_z + 0.22])
        move_tcp_checked(robot, mid_pos, pre_flip_quat, 2.5, steps=40)
        position, quat = robot.get_tcp_pose()
        _, axis = _cup_pose(robot)
        move_tcp_checked(robot, position, rotate_axis_up_quat(axis, quat), 3.0, steps=50)
        retreat_offset = np.array([0.0, -0.10 * place_side, 0.0])

    print("Carry and place using the live cup position", flush=True)
    position, quat = robot.get_tcp_pose()
    bottom, _ = _cup_pose(robot)
    destination = position + np.r_[active_target_xy - bottom[:2], 0.0]
    move_tcp_checked(robot, destination, quat, 4.0, steps=50)
    position, quat = robot.get_tcp_pose()
    bottom, _ = _cup_pose(robot)
    destination = position + np.r_[active_target_xy - bottom[:2], table_z + 0.002 - bottom[2]]
    move_tcp_checked(robot, destination, quat, 3.0, steps=50)
    if recorded_phases is not None:
        recorded_phases.grasp_gap_mm = float(robot.g2_controller.position_mm[0])
        robot.active_waypoint_sink = None

    if not _contact_between(robot, "table", "cup_bottom"):
        target_z_touch = 0.0001 if default_scene else -0.0004
        position, quat = robot.get_tcp_pose()
        bottom, _ = _cup_pose(robot)
        destination = (
            position + np.r_[active_target_xy - bottom[:2], table_z + target_z_touch - bottom[2]]
        )
        move_tcp_checked(robot, destination, quat, 1.2, steps=20)
    if not _contact_between(robot, "table", "cup_bottom"):
        raise TaskError("Place failed: cup bottom does not touch the table")
    _require_bilateral(robot)
    robot.force_feedback = False
    robot.grasp_monitor = None

    print("Release and retreat", flush=True)
    robot.open_gripper(speed_percent=15, force_percent=40)
    robot.settle(1.0)
    robot.check_viewer()
    if recorded_phases is not None:
        robot.active_waypoint_sink = recorded_phases.retreat_waypoints
    position, quat = robot.get_tcp_pose()
    retreat_pos = position + retreat_offset
    move_tcp_checked(robot, retreat_pos, quat, 2.0, steps=30)
    if recorded_phases is not None:
        move_tcp_checked(robot, retreat_pos + np.array([0.0, 0.0, 0.10]), quat, 1.5, steps=20)
        robot.active_waypoint_sink = None
    robot.settle(0.5)
    robot.check_viewer()
    metrics = _verify_rest(robot, target_xy=active_target_xy)
    metrics.update(
        lift_mm=float((lifted_bottom[2] - initial_bottom[2]) * 1000),
        max_grasp_shift_mm=monitor.max_shift_m * 1000,
        max_grasp_rotation_deg=monitor.max_rotation_deg,
        max_contact_loss_s=monitor.max_lost_s,
    )
    print(
        f"Success: placement error {metrics['error_mm']:.2f} mm, max tilt {metrics['tilt_deg']:.2f} deg, "
        f"stable {metrics['stable_s']:.3f} s; max grasp shift {metrics['max_grasp_shift_mm']:.1f} mm.",
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
        "--cup-pose",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "YAW_DEG"),
        help="Sideways cup bottom position and bottom->rim yaw (skips camera detection)",
    )
    parser.add_argument(
        "--target-xy",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
        help="Upright placement target",
    )
    return parser


def _run(args) -> None:
    opts = cli.resolve_run_options(args)
    cup_override = tuple(args.cup_pose) if args.cup_pose is not None else None
    target_override = tuple(args.target_xy) if args.target_xy is not None else None

    if opts.mode == "sim":
        cup_est = None
        if cup_override is not None:
            from .perception import pose_estimate_from_manual

            cup_est = pose_estimate_from_manual(*cup_override)
        target_arr = (
            np.asarray(target_override, dtype=np.float64) if target_override is not None else None
        )
        with CupRobot(viewer=not opts.headless) as robot:
            metrics = run_pick_cup(robot, cup_pose=cup_est, target_xy=target_arr)
        write_json_report(
            opts.report_path, {"mode": "sim", "ok": True, "success": True, "metrics": metrics}
        )
        return

    from .sim2real import run_sim2real_pipeline

    run_sim2real_pipeline(
        mode=opts.mode,
        ip=opts.ip,
        calibration_path=opts.calibration_path,
        handeye_path=opts.handeye_path,
        camera_serial=opts.camera_serial,
        cup_pose_override=cup_override,
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
