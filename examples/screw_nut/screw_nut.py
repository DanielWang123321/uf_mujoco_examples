# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Pick a hex flange nut and thread it onto an M12 screw glued upright on the table.

Run ``python -m examples.screw_nut --help`` for options. ``--mode sim`` runs MuJoCo only;
the hardware modes are implemented in ``sim2real.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ufactory_mujoco.calibration import KinematicsCalibration
from ufactory_mujoco.grippers import bodies_in_contact
from ufactory_mujoco.ik import solve_ik_trajectory
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

from common import cli
from common.geometry import angle_deg, canonical_quat
from common.motion import smooth_step_durations
from common.sim import (
    ForceFeedbackGains,
    TaskError,
    TaskSimRobot,
    body_touches_geom,
    configure_finger_contact_priority,
)
from common.sim2real import write_json_report

SCENE = Path(__file__).with_name("scene.xml")
DEFAULT_CONFIG_YAML = Path(__file__).with_name("config.yaml")

SCREW_BODY = "screw"
NUT_BODY = "nut"
ALL_TASK_BODIES: tuple[str, ...] = (SCREW_BODY, NUT_BODY)

# Physical dimensions (measured with a Gemini 305 3D scan).
SCREW_TOTAL_LEN_M = 0.045
SCREW_HEAD_LEN_M = 0.014
SCREW_HEAD_DIAM_M = 0.020
SCREW_THREAD_DIAM_M = 0.012
SCREW_MASS_KG = 0.056500

NUT_BOT_DIAM_M = 0.029
NUT_TOP_DIAM_M = 0.020
NUT_BORE_DIAM_M = 0.012
NUT_HEIGHT_M = 0.0145
NUT_FLANGE_HEIGHT_M = 0.0038
NUT_MASS_KG = 0.029000

DEFAULT_NUT_XY = np.array([0.3538, -0.0114], dtype=np.float64)
DEFAULT_SCREW_XY = np.array([0.3560, -0.1028], dtype=np.float64)
SCREW_PITCH_M = 0.0024  # 12 helical ridges x 2.4 mm pitch on a 31 mm shaft
DEFAULT_THREAD_TURNS = 10.0  # 10 full turns = 3600 deg across 7 ratchet strokes
MAX_SINGLE_STROKE_TURNS = 1.65  # max turns in one grasp within J6 [-360, +360] deg
RATCHET_STROKE_TURNS = 1.5  # 540 deg = 3 x 180 deg (keeps parallel-jaw symmetry on regrip)
J6_LIMIT_MARGIN_RAD = math.radians(3.0)  # safety margin from the J6 [-360, +360] deg hard stops

# Grasp and assembly geometry.
# At gap = 17..20 mm the G2 fingertip min Z is tcp_z - 3.10 mm. GRASP_TCP_Z_M = 9.0 mm puts the
# fingertips at z = 5.90 mm (above the 3.80 mm flange and 5.0 mm above the table), clamping the
# 20 mm upper hex body (z in [5.9, 14.5] mm).
PRE_GRASP_GAP_MM = 50.0
REGRIP_OPEN_GAP_MM = 50.0
INITIAL_GRASP_GAP_MM = 17.0
MIN_VALID_NUT_GRASP_GAP_MM = 12.0  # below: empty grasp (missed the 20 mm hex body)
MAX_HEX_FLAT_GAP_MM = 21.2  # above at regrip: fingertips touched the 29 mm flange/chamfer
MAX_VALID_NUT_GRASP_GAP_MM = 24.5  # above: jammed on the 29 mm flange or failed to close
Z_ADAPT_STEP_M = 0.0036  # +3.6 mm upward TCP backoff per tactile flange-contact trigger (1.5 turns)
MAX_Z_ADAPT_RETRIES_PER_STROKE = 3
MAX_ADAPTIVE_EXTRA_STROKES = 3
PRE_GRASP_SEARCH_RADIUS_M = 0.080
APPROACH_TCP_Z_M = 0.120
GRASP_TCP_Z_M = 0.0090
LIFT_TCP_Z_M = 0.095
HOVER_SCREW_TCP_Z_M = 0.062  # nut bottom at ~53 mm (above the 45 mm screw tip)
CAPTURE_SCREW_TCP_Z_M = 0.0465  # nut bottom piloted at ~37.4 mm (engaged on the top ridge)
THREAD_END_TCP_Z_M = 0.0230  # nut bottom at the ~14.0 mm screw head shoulder
RETREAT_TCP_Z_M = 0.105

MAX_GRASP_SHIFT_M = 0.018
MAX_GRASP_ROTATION_DEG = 20.0
MAX_CONTACT_LOSS_S = 0.10
MAX_REGRIP_THREAD_SETTLE_M = 0.0012  # max axial settle when the gripper opens for the J6 rewind

# Post-threading seating verification thresholds.
MAX_SEATED_XY_ERR_M = 0.0015  # 1.5 mm coaxial tolerance
MIN_SEATED_Z_M = 0.0130  # screw head shoulder is at 14.0 mm
MAX_SEATED_Z_M = 0.0155
MAX_THREAD_Z_ERR_M = 0.0020
MAX_TILT_DEG = 5.0
MAX_REST_DRIFT_M = 0.0010


@dataclass(frozen=True)
class InitialNutPose:
    """Initial tabletop pose of the hex flange nut."""

    xy_m: tuple[float, float] = (float(DEFAULT_NUT_XY[0]), float(DEFAULT_NUT_XY[1]))
    yaw_deg: float = 0.0
    table_z_m: float = 0.0

    def mujoco_pos_quat(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array([self.xy_m[0], self.xy_m[1], self.table_z_m], dtype=np.float64)
        q_xyzw = Rotation.from_euler("z", float(self.yaw_deg), degrees=True).as_quat()
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        return pos, quat_wxyz


@dataclass
class ScrewNutWaypoints:
    """Joint waypoints recorded across the pick, coaxial capture, and helical threading phases."""

    approach_waypoints: list[np.ndarray] = field(default_factory=list)
    lift_align_waypoints: list[np.ndarray] = field(default_factory=list)
    thread_waypoints: list[np.ndarray] = field(default_factory=list)
    regrip_waypoints: list[np.ndarray] = field(default_factory=list)
    retreat_waypoints: list[np.ndarray] = field(default_factory=list)
    stroke_waypoints: list[list[np.ndarray]] = field(default_factory=list)
    rewind_waypoints: list[list[np.ndarray]] = field(default_factory=list)
    grasp_gap_mm: float = INITIAL_GRASP_GAP_MM


def _nut_pose(robot: TaskSimRobot) -> tuple[np.ndarray, np.ndarray]:
    return robot.body_pose_axis(NUT_BODY)


def _screw_pose(robot: TaskSimRobot) -> tuple[np.ndarray, np.ndarray]:
    return robot.body_pose_axis(SCREW_BODY)


def _tool_relative_nut(robot: TaskSimRobot) -> tuple[np.ndarray, np.ndarray]:
    return robot.tool_relative_pose_axis(NUT_BODY)


def _yaw_rotate_quat(yaw_rad: float, base_quat: np.ndarray) -> np.ndarray:
    """Left-multiply ``base_quat`` by a world +Z rotation of ``yaw_rad`` and canonicalize sign."""
    if abs(yaw_rad) < 1e-9:
        return canonical_quat(base_quat)
    z_rot = np.array([math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)], dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, z_rot, base_quat)
    return canonical_quat(out)


def _plan_j6_grasp_and_quat(
    robot: TaskSimRobot,
    approach_pos: np.ndarray,
    capture_pos: np.ndarray,
    nut_yaw_deg: float,
    thread_turns: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Solve the approach IK and shift J6 by k*60 deg to maximize [-360, +360] deg headroom."""
    # Hexagon flats repeat every 60 deg.
    base_yaw_rad = math.radians((float(nut_yaw_deg) % 60.0) - 30.0)
    base_quat = _yaw_rotate_quat(base_yaw_rad, GRIP_DOWN_QUAT)

    q_app_base, ok_app = robot.solve_tcp(approach_pos, base_quat, q_init=robot.data.qpos.copy())
    if not ok_app:
        raise TaskError(f"IK did not converge for the nut approach pose {approach_pos}")
    q_cap_base, ok_cap = robot.solve_tcp(capture_pos, base_quat, q_init=q_app_base)
    if not ok_cap:
        raise TaskError(f"IK did not converge for the screw capture pose {capture_pos}")

    j6_id = robot.model.joint("joint6").id
    j6_lo, j6_hi = (float(v) for v in robot.model.jnt_range[j6_id])
    req_delta_rad = 2.0 * math.pi * float(thread_turns)

    # Search integer multiples k of 60 deg (hexagon + parallel-jaw symmetry) so that:
    # 1. approach J6 and capture J6 stay within the limits minus margin,
    # 2. the threading end J6 (j6_cap + req_delta_rad) stays below j6_hi - margin,
    # 3. the stroke midpoint is as close to 0 rad as possible (margin on both ends).
    best_k: int | None = None
    best_mid_abs = float("inf")
    hex_step_rad = math.pi / 3.0
    for k in range(-12, 13):
        shift = k * hex_step_rad
        j6_app = float(q_app_base[5]) - shift
        j6_cap = float(q_cap_base[5]) - shift
        j6_end = j6_cap + req_delta_rad
        if (
            j6_lo + J6_LIMIT_MARGIN_RAD <= j6_app <= j6_hi - J6_LIMIT_MARGIN_RAD
            and j6_lo + J6_LIMIT_MARGIN_RAD <= j6_cap <= j6_hi - J6_LIMIT_MARGIN_RAD
            and j6_lo + J6_LIMIT_MARGIN_RAD <= j6_end <= j6_hi - J6_LIMIT_MARGIN_RAD
        ):
            mid_abs = abs(j6_cap + 0.5 * req_delta_rad)
            if mid_abs < best_mid_abs:
                best_mid_abs = mid_abs
                best_k = k

    if best_k is None:
        max_possible_turns = (j6_hi - j6_lo - 2.0 * J6_LIMIT_MARGIN_RAD) / (2.0 * math.pi)
        raise TaskError(
            f"Requested stroke ({thread_turns:.2f} turns = {math.degrees(req_delta_rad):.1f} deg) "
            f"exceeds the J6 [-360, +360] deg single-grasp range (max ~{max_possible_turns:.2f} turns)"
        )

    shift_rad = best_k * hex_step_rad
    q_app = q_app_base.copy()
    q_app[5] -= shift_rad

    # Use the exact FK quaternion at the shifted q_app so later Cartesian moves start with zero
    # orientation error and stay on the selected J6 branch.
    robot.eval_data.qpos[:] = q_app
    mujoco.mj_kinematics(robot.model, robot.eval_data)
    grasp_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(grasp_quat, robot.eval_data.site_xmat[robot._site])

    j6_cap_expected = float(q_cap_base[5]) - shift_rad
    max_cw_deg = math.degrees((j6_hi - J6_LIMIT_MARGIN_RAD) - j6_cap_expected)
    plan_info = {
        "j6_approach_deg": math.degrees(float(q_app[5])),
        "j6_capture_expected_deg": math.degrees(j6_cap_expected),
        "j6_max_cw_available_deg": max_cw_deg,
    }
    return q_app, canonical_quat(grasp_quat), plan_info


@dataclass
class GraspMonitor:
    """Abort on loss of bilateral finger contact or excessive nut slip relative to the TCP."""

    z_offset_m: float
    axis: np.ndarray
    lost_s: float = 0.0
    max_lost_s: float = 0.0
    max_shift_m: float = 0.0
    max_rotation_deg: float = 0.0

    @classmethod
    def start(cls, robot: TaskSimRobot) -> GraspMonitor:
        rel_pos, rel_axis = _tool_relative_nut(robot)
        return cls(float(rel_pos[2]), rel_axis)

    def observe(self, robot: TaskSimRobot) -> None:
        rel_pos, rel_axis = _tool_relative_nut(robot)
        radial_shift = float(math.hypot(rel_pos[0], rel_pos[1]))
        axial_shift = abs(float(rel_pos[2]) - self.z_offset_m)
        shift = math.hypot(radial_shift, axial_shift)
        angle = angle_deg(rel_axis, self.axis)
        self.max_shift_m = max(self.max_shift_m, shift)
        self.max_rotation_deg = max(self.max_rotation_deg, angle)
        self.lost_s = (
            0.0
            if robot.bilateral_finger_contact(NUT_BODY)
            else (self.lost_s + robot.model.opt.timestep)
        )
        self.max_lost_s = max(self.max_lost_s, self.lost_s)
        if not np.isfinite(shift) or not np.isfinite(angle):
            raise TaskError("Invalid nut pose")
        if self.lost_s > MAX_CONTACT_LOSS_S:
            raise TaskError(f"Nut slipped: bilateral contact lost for {self.lost_s:.3f} s")
        if shift > MAX_GRASP_SHIFT_M or angle > MAX_GRASP_ROTATION_DEG:
            raise TaskError(
                f"Nut slipped: relative shift {shift * 1000:.1f} mm, rotation {angle:.1f} deg"
            )


class ScrewNutRobot(TaskSimRobot):
    """Task robot for the nut (G2 force feedback: 6 N target) with helical wrist motions."""

    force_gains = ForceFeedbackGains(
        target_n=6.0, gain_mm_per_n=0.03, max_step_mm=0.15, speed_percent=20, force_percent=25
    )

    def __init__(self, *, viewer: bool, calibration: KinematicsCalibration | None = None):
        super().__init__(SCENE, viewer=viewer, calibration=calibration)
        self.force_body = NUT_BODY

    def _configure_model(self) -> None:
        # High-friction fingers win over the nut; nut-to-screw contact keeps its low friction.
        configure_finger_contact_priority(self.model)

    def move_cartesian_smooth(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        duration: float,
        *,
        steps: int = 25,
        soft_landing: bool = True,
        wait: bool = True,
    ) -> None:
        """Straight Cartesian move with smooth acceleration and soft-landing deceleration."""
        self.check_viewer()
        pos0, quat0 = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (np.asarray(pos, dtype=np.float64) - pos0)
        quat_goal = canonical_quat(quat)
        q0_conj = np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]], dtype=np.float64)
        q_delta_local = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(q_delta_local, q0_conj, quat_goal)
        rotvec_local = np.empty(3, dtype=np.float64)
        mujoco.mju_quat2Vel(rotvec_local, q_delta_local, 1.0)
        quat_traj = np.empty((steps, 4), dtype=np.float64)
        for i in range(steps):
            quat_traj[i] = quat0
            mujoco.mju_quatIntegrate(quat_traj[i], rotvec_local, float(s[i]))

        q_start = self.data.qpos.copy()
        q_traj, ok = solve_ik_trajectory(
            self.model,
            self.eval_data,
            self._tcp_site_name(),
            waypoints,
            quat_traj,
            q_start,
        )
        if not ok:
            raise TaskError(f"IK did not converge: pos={pos}")

        dts = smooth_step_durations(
            duration,
            steps,
            soft_landing=soft_landing,
            min_dt=4 * self.model.opt.timestep,
        )
        for q, seg in zip(q_traj, dts, strict=True):
            self.move_joints(q[: self.n_arm], float(seg), wait=False)
        if wait:
            self.settle(0.25 if soft_landing else 0.15)
        self.check_viewer()

    def move_helical_down(
        self,
        target_pos: np.ndarray,
        base_quat: np.ndarray,
        *,
        turns: float,
        duration: float,
        steps: int = 60,
        soft_landing: bool = True,
    ) -> dict[str, float]:
        """Descend along a vertical line while rotating joint6 coaxially by +2*pi*turns.

        ``smooth_step_durations`` ramps translation and rotation up smoothly and decelerates to a
        soft landing while keeping the spatial waypoints uniform.
        """
        self.check_viewer()
        pos0, _ = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (np.asarray(target_pos, dtype=np.float64) - pos0)
        q_start = self.data.qpos.copy()
        q_traj, ok = solve_ik_trajectory(
            self.model,
            self.eval_data,
            self._tcp_site_name(),
            waypoints,
            canonical_quat(base_quat),
            q_start,
        )
        if not ok:
            raise TaskError(f"Helical threading IK did not converge: target_pos={target_pos}")

        j6_start = float(self.data.qpos[5])
        j6_id = self.model.joint("joint6").id
        j6_lo, j6_hi = (float(v) for v in self.model.jnt_range[j6_id])
        max_cw_rad = (j6_hi - J6_LIMIT_MARGIN_RAD) - j6_start
        total_delta_rad = 2.0 * math.pi * float(turns)
        if total_delta_rad > max_cw_rad:
            raise TaskError(
                f"J6 clockwise rotation exceeds the limit: J6={math.degrees(j6_start):.1f} deg, "
                f"requested +{math.degrees(total_delta_rad):.1f} deg, "
                f"only +{math.degrees(max_cw_rad):.1f} deg available before +360 deg"
            )

        dts = smooth_step_durations(
            duration,
            steps,
            soft_landing=soft_landing,
            min_dt=4 * self.model.opt.timestep,
        )
        for idx, (q, seg) in enumerate(zip(q_traj, dts, strict=True)):
            q_arm = q[: self.n_arm].copy()
            q_arm[5] = j6_start + total_delta_rad * float(s[idx])
            self.move_joints(q_arm, float(seg), wait=False)
        self.settle(0.35)
        self.check_viewer()
        j6_end = float(self.data.qpos[5])
        return {
            "rotated_deg": float(np.degrees(j6_end - j6_start)),
            "j6_start_deg": float(np.degrees(j6_start)),
            "j6_end_deg": float(np.degrees(j6_end)),
            "j6_max_cw_available_deg": float(np.degrees(max_cw_rad)),
            "j6_upper_margin_deg": float(np.degrees(j6_hi - j6_end)),
            "j6_lower_margin_deg": float(np.degrees(j6_start - j6_lo)),
        }

    def move_vertical_hold_wrist(
        self,
        target_pos: np.ndarray,
        _base_quat: np.ndarray | None = None,
        *,
        duration: float,
        steps: int = 25,
        soft_landing: bool = False,
    ) -> None:
        """Move vertically while holding the current joint6 angle (smooth acceleration/deceleration)."""
        self.check_viewer()
        pos0, cur_quat = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (np.asarray(target_pos, dtype=np.float64) - pos0)
        j6_hold = float(self.data.qpos[5])
        q_traj, ok = solve_ik_trajectory(
            self.model,
            self.eval_data,
            self._tcp_site_name(),
            waypoints,
            canonical_quat(cur_quat),
            self.data.qpos.copy(),
        )
        if not ok:
            raise TaskError(f"Vertical retreat IK did not converge: target_pos={target_pos}")
        dts = smooth_step_durations(
            duration,
            steps,
            soft_landing=soft_landing,
            min_dt=4 * self.model.opt.timestep,
        )
        for q, seg in zip(q_traj, dts, strict=True):
            q_arm = q[: self.n_arm].copy()
            q_arm[5] = j6_hold
            self.move_joints(q_arm, float(seg), wait=False)
        self.settle(0.25)
        self.check_viewer()


def _require_bilateral(robot: ScrewNutRobot) -> None:
    if not robot.bilateral_finger_contact(NUT_BODY):
        raise TaskError("No bilateral finger contact on the nut; refusing to lift or release")


def _measure_nut_before_grasp(
    robot: ScrewNutRobot,
    expected_xy: np.ndarray,
    *,
    table_z: float,
    search_radius_m: float = PRE_GRASP_SEARCH_RADIUS_M,
) -> tuple[np.ndarray, float]:
    """Closed-loop pre-grasp measurement of the nut from the overhead approach pose."""
    nut_pos, nut_axis = _nut_pose(robot)
    if not np.all(np.isfinite(nut_pos)) or not np.all(np.isfinite(nut_axis)):
        raise TaskError("Pre-grasp check failed: invalid nut pose")
    xy_delta = float(np.linalg.norm(nut_pos[:2] - expected_xy[:2]))
    z_rel = float(nut_pos[2] - table_z)
    tilt = angle_deg(nut_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if xy_delta > search_radius_m or abs(z_rel) > 0.015:
        raise TaskError(
            f"Pre-grasp gate: nut missing or outside the search radius "
            f"(XY offset {xy_delta * 1000:.1f} mm > {search_radius_m * 1000:.0f} mm, "
            f"z_rel={z_rel * 1000:.1f} mm)"
        )
    if tilt > 20.0:
        raise TaskError(f"Pre-grasp gate: abnormal nut tilt ({tilt:.1f} deg > 20.0 deg)")
    return nut_pos.copy(), xy_delta * 1000.0


def _verify_gripper_grasp_gap(robot: ScrewNutRobot) -> float:
    """Check that the measured G2 opening matches the 20 mm upper hex body (rejects empty grasps)."""
    meas_gap_mm = float(robot.get_measured_gripper_gap() * 1000.0)
    if meas_gap_mm < MIN_VALID_NUT_GRASP_GAP_MM:
        raise TaskError(
            f"Grasp gate: measured gap too small ({meas_gap_mm:.1f} mm < "
            f"{MIN_VALID_NUT_GRASP_GAP_MM:.1f} mm), likely an empty grasp"
        )
    if meas_gap_mm > MAX_VALID_NUT_GRASP_GAP_MM:
        raise TaskError(
            f"Grasp gate: measured gap too large ({meas_gap_mm:.1f} mm > "
            f"{MAX_VALID_NUT_GRASP_GAP_MM:.1f} mm), likely jammed on the bottom flange"
        )
    return meas_gap_mm


def _configure_initial_scene(
    robot: ScrewNutRobot,
    nut_pose: InitialNutPose | None,
    screw_xy: np.ndarray,
    q_init: np.ndarray | None,
) -> float:
    """Apply the optional initial arm state, the glued screw position, and the free nut pose."""
    active_nut = nut_pose if nut_pose is not None else InitialNutPose()
    table_z = float(active_nut.table_z_m)

    table_geom_id = robot.model.geom("table").id
    robot.model.geom_pos[table_geom_id, 2] = table_z

    # The screw body is fixed (glued upright on the table at screw_xy).
    screw_body_id = robot.model.body(SCREW_BODY).id
    robot.model.body_pos[screw_body_id] = [
        float(screw_xy[0]),
        float(screw_xy[1]),
        table_z,
    ]
    glue_geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "glue_pad")
    if glue_geom_id >= 0:
        robot.model.geom_pos[glue_geom_id] = [
            float(screw_xy[0]),
            float(screw_xy[1]),
            table_z + 0.0003,
        ]

    # Recompute model constants before writing custom qpos (mj_setConst resets qpos to qpos0).
    mujoco.mj_setConst(robot.model, robot.data)
    mujoco.mj_setConst(robot.model, robot.eval_data)

    if q_init is not None:
        robot.set_arm_qpos(q_init)

    pos, quat = active_nut.mujoco_pos_quat()
    robot.set_free_joint_pose("nut_joint", pos, quat)

    mujoco.mj_forward(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.eval_data)
    return table_z


def _plan_ratchet_strokes(thread_turns: float) -> list[float]:
    """Split the requested turns into J6-bounded strokes (intermediate strokes: 1.5 turns)."""
    if thread_turns <= 0.0 or thread_turns > 12.0:
        raise ValueError(f"thread_turns must be in (0.0, 12.0], got {thread_turns}")
    strokes: list[float] = []
    rem = float(thread_turns)
    while rem > MAX_SINGLE_STROKE_TURNS + 1e-6:
        strokes.append(RATCHET_STROKE_TURNS)
        rem -= RATCHET_STROKE_TURNS
    if rem > 1e-6:
        strokes.append(round(rem, 6))
    return strokes


def _verify_nut_seated(
    robot: ScrewNutRobot,
    *,
    screw_xy: np.ndarray,
    table_z: float,
    expected_rel_z: float = SCREW_HEAD_LEN_M,
    duration: float = 1.0,
) -> dict[str, float]:
    """Require ``duration`` seconds of the nut resting coaxially on the screw at ``expected_rel_z``."""
    robot.check_viewer()
    up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    start_pos, _ = _nut_pose(robot)
    max_drift = 0.0
    max_tilt = 0.0
    max_xy_err = 0.0
    lost_screw_s = 0.0
    start_time = float(robot.data.time)
    z_lo = max(MIN_SEATED_Z_M, expected_rel_z - MAX_THREAD_Z_ERR_M)
    z_hi = max(MAX_SEATED_Z_M, expected_rel_z + MAX_THREAD_Z_ERR_M)

    while robot.data.time - start_time < duration:
        robot._step()
        nut_pos, nut_axis = _nut_pose(robot)
        if not np.all(np.isfinite(nut_pos)) or not np.all(np.isfinite(nut_axis)):
            raise TaskError("Invalid final nut pose")

        if bodies_in_contact(robot.model, robot.data, "left_finger", NUT_BODY) or bodies_in_contact(
            robot.model, robot.data, "right_finger", NUT_BODY
        ):
            raise TaskError("Rest check failed: gripper still touches the nut")

        if bodies_in_contact(robot.model, robot.data, NUT_BODY, SCREW_BODY):
            lost_screw_s = 0.0
        else:
            lost_screw_s += float(robot.model.opt.timestep)
            if lost_screw_s > 0.02:
                raise TaskError("Assembly check failed: nut lost contact with the fixed screw")

        tilt = angle_deg(nut_axis, up_vec)
        xy_err = float(np.linalg.norm(nut_pos[:2] - screw_xy))
        rel_z = float(nut_pos[2] - table_z)
        drift = float(np.linalg.norm(nut_pos - start_pos))

        max_tilt = max(max_tilt, tilt)
        max_xy_err = max(max_xy_err, xy_err)
        max_drift = max(max_drift, drift)

        if (
            tilt > MAX_TILT_DEG
            or xy_err > MAX_SEATED_XY_ERR_M
            or not (z_lo <= rel_z <= z_hi)
            or drift > MAX_REST_DRIFT_M
        ):
            raise TaskError(
                f"Nut seating / thread self-lock check failed: coaxial error {xy_err * 1000:.2f} mm, "
                f"height {rel_z * 1000:.2f} mm (expected ~{expected_rel_z * 1000:.2f} mm), "
                f"tilt {tilt:.2f} deg, drift {drift * 1000:.2f} mm"
            )

    final_pos, _ = _nut_pose(robot)
    return {
        "xy_error_mm": max_xy_err * 1000.0,
        "nut_z_mm": float((final_pos[2] - table_z) * 1000.0),
        "expected_nut_z_mm": float(expected_rel_z * 1000.0),
        "tilt_deg": max_tilt,
        "drift_mm": max_drift * 1000.0,
        "stable_s": float(robot.data.time - start_time),
    }


def _verify_joint_limits(
    robot: ScrewNutRobot,
    waypoints: ScrewNutWaypoints,
) -> dict[str, Any]:
    """Check that every recorded arm waypoint stays within ``model.jnt_range``."""
    blocks = (
        waypoints.approach_waypoints
        + waypoints.lift_align_waypoints
        + waypoints.thread_waypoints
        + waypoints.regrip_waypoints
        + waypoints.retreat_waypoints
    )
    if not blocks:
        raise TaskError("No joint waypoints were recorded")
    all_q = np.vstack(blocks)
    per_joint: dict[str, dict[str, float]] = {}
    min_margin_deg = float("inf")

    for j in range(robot.n_arm):
        lo_rad, hi_rad = (float(v) for v in robot.model.jnt_range[j])
        q_min_rad = float(all_q[:, j].min())
        q_max_rad = float(all_q[:, j].max())
        margin_lo_deg = math.degrees(q_min_rad - lo_rad)
        margin_hi_deg = math.degrees(hi_rad - q_max_rad)
        joint_margin_deg = min(margin_lo_deg, margin_hi_deg)
        min_margin_deg = min(min_margin_deg, joint_margin_deg)
        if q_min_rad < lo_rad or q_max_rad > hi_rad:
            raise TaskError(
                f"joint{j + 1} exceeds its limits: actual [{math.degrees(q_min_rad):.2f}, "
                f"{math.degrees(q_max_rad):.2f}] deg, limits [{math.degrees(lo_rad):.2f}, "
                f"{math.degrees(hi_rad):.2f}] deg"
            )
        per_joint[f"joint{j + 1}"] = {
            "range_deg": [round(math.degrees(lo_rad), 2), round(math.degrees(hi_rad), 2)],
            "actual_deg": [round(math.degrees(q_min_rad), 2), round(math.degrees(q_max_rad), 2)],
            "margin_deg": round(joint_margin_deg, 2),
        }

    return {
        "within_limits": True,
        "min_margin_deg": round(min_margin_deg, 2),
        "joints": per_joint,
    }


def run_screw_nut(
    robot: ScrewNutRobot,
    *,
    nut_pose: InitialNutPose | None = None,
    screw_xy: np.ndarray | None = None,
    thread_turns: float = DEFAULT_THREAD_TURNS,
    commanded_pitch_m: float = SCREW_PITCH_M,
    simulated_slip_ratio: float = 0.0,
    q_init: np.ndarray | None = None,
    recorded_waypoints: ScrewNutWaypoints | None = None,
) -> dict[str, Any]:
    """Pick the nut, pilot it onto the screw, and thread it down in multiple clockwise strokes."""
    strokes = list(_plan_ratchet_strokes(thread_turns))
    nominal_stroke_count = len(strokes)

    active_wps = recorded_waypoints if recorded_waypoints is not None else ScrewNutWaypoints()
    active_screw_xy = (
        DEFAULT_SCREW_XY.copy()
        if screw_xy is None
        else np.asarray(screw_xy, dtype=np.float64).reshape(2)
    )
    active_nut_pose = nut_pose if nut_pose is not None else InitialNutPose()
    table_z = _configure_initial_scene(robot, active_nut_pose, active_screw_xy, q_init)

    print(f"Prepare: open gripper to {PRE_GRASP_GAP_MM:.1f} mm and let the nut settle", flush=True)
    robot.set_gripper_command(PRE_GRASP_GAP_MM, speed_percent=40, force_percent=20)
    robot.settle(0.6)
    robot.check_viewer()

    expected_nut_xy = np.array([active_nut_pose.xy_m[0], active_nut_pose.xy_m[1]], dtype=np.float64)
    approach_pos = np.array(
        [expected_nut_xy[0], expected_nut_xy[1], table_z + APPROACH_TCP_Z_M],
        dtype=np.float64,
    )
    capture_pos = np.array(
        [active_screw_xy[0], active_screw_xy[1], table_z + CAPTURE_SCREW_TCP_Z_M],
        dtype=np.float64,
    )

    q_app, grasp_quat, j6_plan = _plan_j6_grasp_and_quat(
        robot,
        approach_pos,
        capture_pos,
        active_nut_pose.yaw_deg,
        strokes[0],
    )

    print(
        f"Approach the pre-grasp pose above the nut (J6 preset {j6_plan['j6_approach_deg']:.1f} deg, "
        f"clockwise headroom +{j6_plan['j6_max_cw_available_deg']:.1f} deg)",
        flush=True,
    )
    robot.active_waypoint_sink = active_wps.approach_waypoints
    robot.move_joints(q_app[: robot.n_arm], 1.6, wait=True)
    robot.check_viewer()

    # Pre-grasp gate: confirm the nut is present and refine its XY before the soft-landing descent.
    measured_nut_pos, pre_grasp_delta_mm = _measure_nut_before_grasp(
        robot, expected_nut_xy, table_z=table_z
    )
    grasp_pos = np.array(
        [measured_nut_pos[0], measured_nut_pos[1], table_z + GRASP_TCP_Z_M],
        dtype=np.float64,
    )
    print(
        f"Nut confirmed from above (XY correction {pre_grasp_delta_mm:.2f} mm); "
        "soft-landing descent to grasp",
        flush=True,
    )
    robot.move_cartesian_smooth(grasp_pos, grasp_quat, 1.2, steps=24, soft_landing=True, wait=True)
    robot.active_waypoint_sink = None

    print("Grasp the upper hex body: check measured gap and bilateral contact", flush=True)
    robot.set_gripper_command(INITIAL_GRASP_GAP_MM, speed_percent=20, force_percent=25)
    robot.settle(1.1)
    robot.check_viewer()
    measured_grasp_gap_mm = _verify_gripper_grasp_gap(robot)
    gap_mm = max(0.0, measured_grasp_gap_mm - 1.5)
    robot.set_gripper_command(gap_mm, speed_percent=20, force_percent=25)
    robot.settle(0.4)
    robot.check_viewer()
    active_wps.grasp_gap_mm = float(robot.g2_controller.position_mm[0])
    _require_bilateral(robot)

    monitor = GraspMonitor.start(robot)
    robot.grasp_monitor = monitor
    robot.force_feedback = True
    robot.last_force_update = -1.0

    print("Lift the nut and move above the glued screw", flush=True)
    robot.active_waypoint_sink = active_wps.lift_align_waypoints
    lift_pos = np.array(
        [measured_nut_pos[0], measured_nut_pos[1], table_z + LIFT_TCP_Z_M],
        dtype=np.float64,
    )
    robot.move_cartesian_smooth(lift_pos, grasp_quat, 1.1, steps=22, soft_landing=False, wait=False)
    lifted_nut_pos, _ = _nut_pose(robot)
    lift_mm = float((lifted_nut_pos[2] - measured_nut_pos[2]) * 1000.0)
    if lifted_nut_pos[2] < measured_nut_pos[2] + 0.050 or body_touches_geom(
        robot.model, robot.data, NUT_BODY, "table"
    ):
        raise TaskError("Lift failed: nut did not leave the table")

    hover_pos = np.array(
        [active_screw_xy[0], active_screw_xy[1], table_z + HOVER_SCREW_TCP_Z_M],
        dtype=np.float64,
    )
    robot.move_cartesian_smooth(hover_pos, grasp_quat, 1.4, steps=28, soft_landing=True, wait=True)

    # Pre-capture gate: the screw must be upright in place and coaxial with the nut in hand.
    live_screw_pos, live_screw_axis = _screw_pose(robot)
    live_nut_pos, _ = _nut_pose(robot)
    screw_delta = float(np.linalg.norm(live_screw_pos[:2] - active_screw_xy[:2]))
    screw_tilt = angle_deg(live_screw_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if screw_delta > PRE_GRASP_SEARCH_RADIUS_M or screw_tilt > 10.0:
        raise TaskError(
            f"Pre-capture gate: fixed screw missing or displaced "
            f"(offset {screw_delta * 1000:.1f} mm, tilt {screw_tilt:.1f} deg)"
        )
    xy_offset = live_screw_pos[:2] - live_nut_pos[:2]
    pre_capture_comp_mm = float(np.linalg.norm(xy_offset) * 1000.0)
    if pre_capture_comp_mm > 6.0:
        raise TaskError(
            f"Pre-capture gate: nut-to-screw coaxial offset too large "
            f"({pre_capture_comp_mm:.2f} mm > 6.0 mm)"
        )
    capture_pos = np.array(
        [
            live_screw_pos[0],
            live_screw_pos[1],
            table_z + CAPTURE_SCREW_TCP_Z_M,
        ],
        dtype=np.float64,
    )
    print(
        f"Coaxial check passed (offset {pre_capture_comp_mm:.2f} mm); "
        "soft-landing pilot onto the first M12 thread",
        flush=True,
    )
    robot.move_cartesian_smooth(
        capture_pos, grasp_quat, 1.0, steps=22, soft_landing=True, wait=True
    )
    robot.active_waypoint_sink = None
    capture_nut_pos, _ = _nut_pose(robot)
    capture_rel_z = float(capture_nut_pos[2] - table_z)

    print(
        f"Helical threading: {thread_turns:.2f} turns ({thread_turns * 360.0:.0f} deg) "
        f"in {len(strokes)} strokes (pitch {SCREW_PITCH_M * 1000:.2f} mm/turn)",
        flush=True,
    )
    total_rotated_deg = 0.0
    max_shift_m = 0.0
    max_rot_deg = 0.0
    max_lost_s = 0.0
    max_regrip_settle_m = 0.0
    first_j6_start_deg = 0.0
    last_j6_end_deg = 0.0
    min_j6_upper_margin_deg = float("inf")
    min_j6_lower_margin_deg = float("inf")
    max_cw_avail_deg = 0.0
    current_tcp_z = float(capture_pos[2])
    z_adapt_count = 0
    z_adapt_total_m = 0.0
    extra_strokes_used = 0
    regrip_gaps_mm: list[float] = []

    stroke_idx = 0
    while stroke_idx < len(strokes):
        st_turns = strokes[stroke_idx]
        nut_before, _ = _nut_pose(robot)
        dz = st_turns * float(commanded_pitch_m)
        current_tcp_z = max(table_z + THREAD_END_TCP_Z_M, current_tcp_z - dz - 0.0002)
        stroke_target_pos = np.array(
            [active_screw_xy[0], active_screw_xy[1], current_tcp_z],
            dtype=np.float64,
        )
        stroke_sink: list[np.ndarray] = []
        robot.active_waypoint_sink = stroke_sink
        stroke_dur = max(0.8, 0.75 * st_turns)
        stroke_steps = max(20, round(20 * st_turns))
        eff_turns = st_turns * max(0.1, 1.0 - float(simulated_slip_ratio))
        helical_info = robot.move_helical_down(
            stroke_target_pos,
            grasp_quat,
            turns=eff_turns,
            duration=stroke_dur,
            steps=stroke_steps,
            soft_landing=True,
        )
        robot.active_waypoint_sink = None
        active_wps.thread_waypoints.extend(stroke_sink)
        active_wps.stroke_waypoints.append(stroke_sink)
        _require_bilateral(robot)

        nut_after, _ = _nut_pose(robot)
        total_rotated_deg += helical_info["rotated_deg"]
        if stroke_idx == 0:
            first_j6_start_deg = helical_info["j6_start_deg"]
            max_cw_avail_deg = helical_info["j6_max_cw_available_deg"]
        last_j6_end_deg = helical_info["j6_end_deg"]
        min_j6_upper_margin_deg = min(min_j6_upper_margin_deg, helical_info["j6_upper_margin_deg"])
        min_j6_lower_margin_deg = min(min_j6_lower_margin_deg, helical_info["j6_lower_margin_deg"])

        if robot.grasp_monitor is not None:
            max_shift_m = max(max_shift_m, robot.grasp_monitor.max_shift_m)
            max_rot_deg = max(max_rot_deg, robot.grasp_monitor.max_rotation_deg)
            max_lost_s = max(max_lost_s, robot.grasp_monitor.max_lost_s)

        print(
            f"  Stroke {stroke_idx + 1}/{len(strokes)} ({st_turns:.2f} turns): "
            f"J6 [{helical_info['j6_start_deg']:.1f} -> {helical_info['j6_end_deg']:.1f}] deg, "
            f"TCP Z={(current_tcp_z - table_z) * 1000:.2f} mm, "
            f"nut height {(nut_before[2] - table_z) * 1000:.2f} -> "
            f"{(nut_after[2] - table_z) * 1000:.2f} mm",
            flush=True,
        )

        # If the Z backoff left the TCP above the seating target, append a continuation stroke.
        if (
            stroke_idx == len(strokes) - 1
            and z_adapt_count > 0
            and current_tcp_z > table_z + THREAD_END_TCP_Z_M + 0.0012
            and extra_strokes_used < MAX_ADAPTIVE_EXTRA_STROKES
        ):
            strokes.append(RATCHET_STROKE_TURNS)
            extra_strokes_used += 1
            print(
                f"    [Extra stroke #{extra_strokes_used}] TCP Z={(current_tcp_z - table_z) * 1000:.2f} mm "
                f"> seating target {THREAD_END_TCP_Z_M * 1000:.2f} mm; "
                f"appending a {RATCHET_STROKE_TURNS:.2f}-turn stroke",
                flush=True,
            )

        if stroke_idx < len(strokes) - 1:
            # Open wide (50 mm >> 29 mm flange / 23.1 mm hex corners), rewind J6, and regrip.
            robot.force_feedback = False
            robot.grasp_monitor = None
            robot.set_gripper_command(REGRIP_OPEN_GAP_MM, speed_percent=60, force_percent=50)
            robot.settle(0.55)
            robot.check_viewer()

            nut_hold, _ = _nut_pose(robot)
            settle_m = abs(float(nut_hold[2] - nut_after[2]))
            max_regrip_settle_m = max(max_regrip_settle_m, settle_m)
            if settle_m > MAX_REGRIP_THREAD_SETTLE_M:
                raise TaskError(
                    f"Nut did not self-lock on the thread during regrip: sank {settle_m * 1000:.2f} mm"
                )

            regrip_pos = np.array(
                [active_screw_xy[0], active_screw_xy[1], current_tcp_z],
                dtype=np.float64,
            )
            q_seed = robot.data.qpos.copy()
            j6_rewind_rad = math.radians(helical_info["j6_start_deg"])
            q_seed[5] = j6_rewind_rad
            q_regrip, ok_regrip = robot.solve_tcp(regrip_pos, grasp_quat, q_init=q_seed)
            if not ok_regrip:
                raise TaskError(f"Stroke {stroke_idx + 1} rewind IK did not converge: {regrip_pos}")
            q_regrip[5] = j6_rewind_rad

            rewind_sink: list[np.ndarray] = []
            robot.active_waypoint_sink = rewind_sink
            robot.move_joints(q_regrip[: robot.n_arm], 0.80, wait=True)
            robot.active_waypoint_sink = None
            robot.check_viewer()

            robot.set_gripper_command(gap_mm, speed_percent=50, force_percent=25)
            robot.settle(0.65)
            robot.check_viewer()
            meas_regrip_gap_mm = float(robot.get_measured_gripper_gap() * 1000.0)

            adapt_retries = 0
            while (
                meas_regrip_gap_mm > MAX_HEX_FLAT_GAP_MM
                and adapt_retries < MAX_Z_ADAPT_RETRIES_PER_STROKE
            ):
                adapt_retries += 1
                z_adapt_count += 1
                dz_up = Z_ADAPT_STEP_M
                z_adapt_total_m += dz_up
                current_tcp_z = min(
                    table_z + CAPTURE_SCREW_TCP_Z_M,
                    current_tcp_z + dz_up,
                )
                print(
                    f"    [Z adaptation #{z_adapt_count}] stroke {stroke_idx + 1} regrip gap "
                    f"{meas_regrip_gap_mm:.2f} mm > {MAX_HEX_FLAT_GAP_MM:.1f} mm (touching the 29 mm "
                    f"flange); raising TCP Z +{dz_up * 1000:.1f} mm to "
                    f"{(current_tcp_z - table_z) * 1000:.2f} mm to regrip the hex body",
                    flush=True,
                )
                robot.set_gripper_command(REGRIP_OPEN_GAP_MM, speed_percent=60, force_percent=50)
                robot.settle(0.50)
                robot.check_viewer()
                regrip_pos = np.array(
                    [active_screw_xy[0], active_screw_xy[1], current_tcp_z],
                    dtype=np.float64,
                )
                q_regrip_up, ok_up = robot.solve_tcp(
                    regrip_pos, grasp_quat, q_init=robot.data.qpos.copy()
                )
                if not ok_up:
                    raise TaskError(
                        f"Stroke {stroke_idx + 1} Z-adaptation IK did not converge: {regrip_pos}"
                    )
                q_regrip_up[5] = j6_rewind_rad
                robot.active_waypoint_sink = rewind_sink
                robot.move_joints(q_regrip_up[: robot.n_arm], 0.45, wait=True)
                robot.active_waypoint_sink = None
                robot.check_viewer()
                robot.set_gripper_command(gap_mm, speed_percent=50, force_percent=25)
                robot.settle(0.65)
                robot.check_viewer()
                meas_regrip_gap_mm = float(robot.get_measured_gripper_gap() * 1000.0)

            active_wps.regrip_waypoints.extend(rewind_sink)
            active_wps.rewind_waypoints.append(rewind_sink)
            regrip_gaps_mm.append(round(meas_regrip_gap_mm, 2))
            _verify_gripper_grasp_gap(robot)
            _require_bilateral(robot)
            robot.grasp_monitor = GraspMonitor.start(robot)
            robot.force_feedback = True
            robot.last_force_update = -1.0

        stroke_idx += 1

    robot.force_feedback = False
    robot.grasp_monitor = None

    print("Release the nut and retreat vertically", flush=True)
    robot.set_gripper_command(PRE_GRASP_GAP_MM, speed_percent=30, force_percent=20)
    robot.settle(0.6)
    robot.check_viewer()

    robot.active_waypoint_sink = active_wps.retreat_waypoints
    retreat_pos = np.array(
        [active_screw_xy[0], active_screw_xy[1], table_z + RETREAT_TCP_Z_M],
        dtype=np.float64,
    )
    robot.move_vertical_hold_wrist(
        retreat_pos, grasp_quat, duration=1.4, steps=25, soft_landing=False
    )
    robot.active_waypoint_sink = None
    robot.settle(0.3)
    robot.check_viewer()

    effective_turns = total_rotated_deg / 360.0
    expected_rel_z = max(
        SCREW_HEAD_LEN_M,
        capture_rel_z - float(effective_turns) * SCREW_PITCH_M,
    )
    joint_limit_summary = _verify_joint_limits(robot, active_wps)
    metrics = _verify_nut_seated(
        robot,
        screw_xy=active_screw_xy,
        table_z=table_z,
        expected_rel_z=expected_rel_z,
        duration=1.0,
    )
    metrics.update(
        lift_mm=lift_mm,
        pre_grasp_nut_delta_xy_mm=pre_grasp_delta_mm,
        measured_grasp_gap_mm=measured_grasp_gap_mm,
        regrip_gaps_mm=regrip_gaps_mm,
        z_adapt_count=z_adapt_count,
        z_adapt_total_mm=round(z_adapt_total_m * 1000.0, 2),
        adaptive_extra_strokes=extra_strokes_used,
        nominal_stroke_count=nominal_stroke_count,
        pre_capture_offset_comp_mm=pre_capture_comp_mm,
        rotated_deg=total_rotated_deg,
        thread_turns=float(thread_turns),
        stroke_count=len(strokes),
        j6_start_deg=first_j6_start_deg,
        j6_end_deg=last_j6_end_deg,
        j6_max_cw_available_deg=max_cw_avail_deg,
        j6_upper_margin_deg=min_j6_upper_margin_deg,
        j6_lower_margin_deg=min_j6_lower_margin_deg,
        max_regrip_settle_mm=max_regrip_settle_m * 1000.0,
        joint_limits=joint_limit_summary,
        max_grasp_shift_mm=max_shift_m * 1000.0,
        max_grasp_rotation_deg=max_rot_deg,
        max_contact_loss_s=max_lost_s,
    )
    print(
        f"Success: {len(strokes)} strokes, {thread_turns:.2f} turns "
        f"(net rotation {metrics['rotated_deg']:.1f} deg, min J6 upper margin "
        f"{metrics['j6_upper_margin_deg']:.1f} deg); nut height {metrics['nut_z_mm']:.2f} mm "
        f"(expected {metrics['expected_nut_z_mm']:.2f} mm), coaxial error "
        f"{metrics['xy_error_mm']:.2f} mm, max tilt {metrics['tilt_deg']:.2f} deg, "
        f"stable {metrics['stable_s']:.3f} s.",
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
        "--nut-pose",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "YAW_DEG"),
        default=None,
        help="Initial nut position and yaw on the table",
    )
    parser.add_argument(
        "--screw-xy",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
        default=None,
        help="Position of the screw glued on the table",
    )
    parser.add_argument(
        "--thread-turns",
        type=float,
        default=None,
        help=f"Clockwise turns after capture (default: {DEFAULT_THREAD_TURNS:.1f} = 3600 deg)",
    )
    return parser


def _run(args) -> None:
    opts = cli.resolve_run_options(args)
    task_cfg = opts.task
    thread_turns = (
        float(args.thread_turns)
        if args.thread_turns is not None
        else float(task_cfg.get("thread_turns", DEFAULT_THREAD_TURNS))
    )

    nut_override: tuple[float, float, float] | None = None
    if args.nut_pose is not None:
        nut_override = (float(args.nut_pose[0]), float(args.nut_pose[1]), float(args.nut_pose[2]))
        nut_pose = InitialNutPose(xy_m=(nut_override[0], nut_override[1]), yaw_deg=nut_override[2])
    else:
        cfg_nut_xy = task_cfg.get("nut_xy", DEFAULT_NUT_XY.tolist())
        nut_pose = InitialNutPose(
            xy_m=(float(cfg_nut_xy[0]), float(cfg_nut_xy[1])),
            yaw_deg=float(task_cfg.get("nut_yaw_deg", 0.0)),
        )

    screw_override: tuple[float, float] | None = None
    if args.screw_xy is not None:
        screw_override = (float(args.screw_xy[0]), float(args.screw_xy[1]))
        screw_xy = np.array([screw_override[0], screw_override[1]], dtype=np.float64)
    else:
        cfg_screw_xy = task_cfg.get("screw_xy", DEFAULT_SCREW_XY.tolist())
        screw_xy = np.array([float(cfg_screw_xy[0]), float(cfg_screw_xy[1])], dtype=np.float64)

    if opts.mode == "sim":
        with ScrewNutRobot(viewer=not opts.headless) as robot:
            metrics = run_screw_nut(
                robot, nut_pose=nut_pose, screw_xy=screw_xy, thread_turns=thread_turns
            )
        write_json_report(
            opts.report_path,
            {
                "ok": True,
                "mode": "sim",
                "nut_xy_m": list(nut_pose.xy_m),
                "nut_yaw_deg": float(nut_pose.yaw_deg),
                "screw_xy_m": [float(screw_xy[0]), float(screw_xy[1])],
                "metrics": metrics,
            },
        )
        return

    from .sim2real import run_sim2real_pipeline

    run_sim2real_pipeline(
        mode=opts.mode,
        thread_turns=thread_turns,
        ip=opts.ip,
        calibration_path=opts.calibration_path,
        handeye_path=opts.handeye_path,
        camera_serial=opts.camera_serial,
        nut_pose_override=nut_override,
        screw_xy_override=screw_override,
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
