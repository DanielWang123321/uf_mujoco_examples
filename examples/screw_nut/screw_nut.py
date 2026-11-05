#!/usr/bin/env python3
"""Pick up a hex flange nut and thread it onto a tabletop-glued M12 screw in MuJoCo (0.0.3)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# Reference shared common package and read-only ufactory_mujoco package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ufactory_mujoco"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ufactory_mujoco.calibration import (
    KinematicsCalibration,
    apply_calibration_to_model,
)
from ufactory_mujoco.grippers import bodies_in_contact
from ufactory_mujoco.ik import solve_ik_trajectory
from ufactory_mujoco.sim_robot import SimRobot
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

from common.motion import smooth_step_durations

SCENE = Path(__file__).with_name("scene.xml")
DEFAULT_CONFIG_YAML = Path(__file__).with_name("config.yaml")
MODES = ("sim", "dry-run", "sdk-sim", "real")

SCREW_BODY = "screw"
NUT_BODY = "nut"
ALL_TASK_BODIES: tuple[str, ...] = (SCREW_BODY, NUT_BODY)

# Physical dimensions from prd.md 0.0.3 and Gemini 305 3D scan
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
SCREW_PITCH_M = 0.0024  # 12 helical ridges x 2.4 mm pitch on 31 mm shaft
DEFAULT_THREAD_TURNS = 10.0  # 10 full turns = 3600 deg across 7 ratchet strokes
MAX_SINGLE_STROKE_TURNS = 1.65  # Max turns in one J6 [-360, +360] deg grasp
RATCHET_STROKE_TURNS = 1.5  # 540 deg = 3 x 180 deg (preserves parallel-jaw symmetry on regrip)
J6_LIMIT_MARGIN_RAD = math.radians(3.0)  # 3.0 deg safety margin from [-360, +360] deg hard stops

# Grasp & assembly geometry constants
# At gap=17..20mm, Gripper G2 fingertip min_z is tcp_z - 3.10mm.
# Setting GRASP_TCP_Z_M = 9.0mm places fingertips at z = 5.90mm (> 3.80mm bottom flange
# and > 5.0mm above the table), clamping the 20mm upper hex body (z in [5.9, 14.5]mm).
PRE_GRASP_GAP_MM = 50.0
REGRIP_OPEN_GAP_MM = 50.0
INITIAL_GRASP_GAP_MM = 17.0
MIN_VALID_NUT_GRASP_GAP_MM = 12.0  # Below 12.0 mm means empty grasp (missed the 20mm hex body)
MAX_HEX_FLAT_GAP_MM = 21.2  # Above 21.2 mm at regrip means fingertips touched 29mm flange/chamfer
MAX_VALID_NUT_GRASP_GAP_MM = 24.5  # Above 24.5 mm means jammed on 29mm flange or failed to close
Z_ADAPT_STEP_M = 0.0036  # +3.6 mm upward Z backoff per tactile flange-contact trigger (1.5 turns)
MAX_Z_ADAPT_RETRIES_PER_STROKE = 3
MAX_ADAPTIVE_EXTRA_STROKES = 3
PRE_GRASP_SEARCH_RADIUS_M = 0.080
APPROACH_TCP_Z_M = 0.120
GRASP_TCP_Z_M = 0.0090
LIFT_TCP_Z_M = 0.095
HOVER_SCREW_TCP_Z_M = 0.062  # Nut bottom at ~53mm (> 45mm screw tip)
CAPTURE_SCREW_TCP_Z_M = 0.0465  # Nut bottom piloted at ~37.4mm (engaged at top helical ridge)
THREAD_END_TCP_Z_M = 0.0230  # Nut bottom at ~14.0mm screw head shoulder
RETREAT_TCP_Z_M = 0.105

MAX_GRASP_SHIFT_M = 0.018
MAX_GRASP_ROTATION_DEG = 20.0
MAX_CONTACT_LOSS_S = 0.10
MAX_REGRIP_THREAD_SETTLE_M = 0.0012  # Max mid-shaft axial settle when gripper opens for J6 rewind
FORCE_TARGET_N = 6.0
FORCE_UPDATE_S = 0.05

# Post-threading static seating verification thresholds
MAX_SEATED_XY_ERR_M = 0.0015  # 1.5 mm coaxial tolerance
MIN_SEATED_Z_M = 0.0130  # Screw head shoulder is at 14.0 mm
MAX_SEATED_Z_M = 0.0155
MAX_THREAD_Z_ERR_M = 0.0020
MAX_TILT_DEG = 5.0
MAX_REST_DRIFT_M = 0.0010


class DemoError(RuntimeError):
    """A task, contact, or viewer condition prevented a successful nut-threading run."""


@dataclass(frozen=True)
class InitialNutPose:
    """Initial tabletop hex flange nut configuration."""

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
    """Recorded joint waypoints across the pick, coaxial capture, and helical threading phases."""

    approach_waypoints: list[np.ndarray] = field(default_factory=list)
    lift_align_waypoints: list[np.ndarray] = field(default_factory=list)
    thread_waypoints: list[np.ndarray] = field(default_factory=list)
    regrip_waypoints: list[np.ndarray] = field(default_factory=list)
    retreat_waypoints: list[np.ndarray] = field(default_factory=list)
    stroke_waypoints: list[list[np.ndarray]] = field(default_factory=list)
    rewind_waypoints: list[list[np.ndarray]] = field(default_factory=list)
    grasp_gap_mm: float = INITIAL_GRASP_GAP_MM


def _nut_pose(robot: SimRobot) -> tuple[np.ndarray, np.ndarray]:
    body = robot.data.body(NUT_BODY)
    return body.xpos.copy(), body.xmat.reshape(3, 3)[:, 2].copy()


def _screw_pose(robot: SimRobot) -> tuple[np.ndarray, np.ndarray]:
    body = robot.data.body(SCREW_BODY)
    return body.xpos.copy(), body.xmat.reshape(3, 3)[:, 2].copy()


def _tool_relative_nut(robot: SimRobot) -> tuple[np.ndarray, np.ndarray]:
    nut_pos, nut_axis = _nut_pose(robot)
    tcp_pos = robot.data.site_xpos[robot._site]
    tcp_rot = robot.data.site_xmat[robot._site].reshape(3, 3)
    return tcp_rot.T @ (nut_pos - tcp_pos), tcp_rot.T @ nut_axis


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _nut_touches_table(robot: SimRobot) -> bool:
    table_id = robot.model.geom("table").id
    nut_body_id = robot.model.body(NUT_BODY).id
    for contact in robot.data.contact[: robot.data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if contact.dist <= 0.0 and (
            (a == table_id and robot.model.geom_bodyid[b] == nut_body_id)
            or (b == table_id and robot.model.geom_bodyid[a] == nut_body_id)
        ):
            return True
    return False


def _canonical_quat(quat: np.ndarray) -> np.ndarray:
    """Return quaternion in MuJoCo's canonical mju_mat2Quat hemisphere."""
    mat = np.empty(9, dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    mujoco.mju_mat2Quat(out, mat)
    return out


def _yaw_rotate_quat(yaw_rad: float, base_quat: np.ndarray) -> np.ndarray:
    """Left-multiply base_quat by a world +Z rotation of yaw_rad and canonicalize sign."""
    if abs(yaw_rad) < 1e-9:
        return _canonical_quat(base_quat)
    z_rot = np.array([math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)], dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, z_rot, base_quat)
    return _canonical_quat(out)


def _plan_j6_grasp_and_quat(
    robot: SimRobot,
    approach_pos: np.ndarray,
    capture_pos: np.ndarray,
    nut_yaw_deg: float,
    thread_turns: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Solve approach IK and shift J6 in joint space (by k*pi) to maximize [-360, +360] deg headroom."""
    # Regular hexagon flats repeat every 60 degrees (pi/3 rad).
    base_yaw_rad = math.radians((float(nut_yaw_deg) % 60.0) - 30.0)
    base_quat = _yaw_rotate_quat(base_yaw_rad, GRIP_DOWN_QUAT)

    q_app_base, ok_app = robot.solve_tcp(approach_pos, base_quat, q_init=robot.data.qpos.copy())
    if not ok_app:
        raise DemoError(f"逆运动学不收敛：螺母接近位置 {approach_pos}")
    q_cap_base, ok_cap = robot.solve_tcp(capture_pos, base_quat, q_init=q_app_base)
    if not ok_cap:
        raise DemoError(f"逆运动学不收敛：螺杆捕获位置 {capture_pos}")

    j6_id = robot.model.joint("joint6").id
    j6_lo, j6_hi = (float(v) for v in robot.model.jnt_range[j6_id])
    req_delta_rad = 2.0 * math.pi * float(thread_turns)

    # Search integer multiples k of pi/3 (60-deg regular hexagonal nut + parallel-jaw symmetry) so:
    # 1. Both approach J6 and capture J6 stay above j6_lo + margin
    # 2. Threading end J6 (j6_cap + req_delta_rad) stays below j6_hi - margin
    # 3. The stroke midpoint is as close to 0 rad as possible (maximizing margin on both ends)
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
        raise DemoError(
            f"请求的旋紧圈数 ({thread_turns:.2f} 圈 = {math.degrees(req_delta_rad):.1f}°) "
            f"超出 J6 [-360°, +360°] 单次最大可用余量 (上限约 {max_possible_turns:.2f} 圈)"
        )

    shift_rad = best_k * hex_step_rad
    q_app = q_app_base.copy()
    q_app[5] -= shift_rad

    # Evaluate exact FK quaternion at the shifted q_app so subsequent Cartesian moves
    # start with zero orientation error and stay on the selected J6 sheet.
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
    return q_app, _canonical_quat(grasp_quat), plan_info


@dataclass
class GraspMonitor:
    """Detect loss of bilateral finger contact and excessive nut slip relative to the TCP."""

    z_offset_m: float
    axis: np.ndarray
    lost_s: float = 0.0
    max_lost_s: float = 0.0
    max_shift_m: float = 0.0
    max_rotation_deg: float = 0.0

    @classmethod
    def start(cls, robot: SimRobot) -> GraspMonitor:
        rel_pos, rel_axis = _tool_relative_nut(robot)
        return cls(float(rel_pos[2]), rel_axis)

    def observe(self, robot: SimRobot) -> None:
        rel_pos, rel_axis = _tool_relative_nut(robot)
        radial_shift = float(math.hypot(rel_pos[0], rel_pos[1]))
        axial_shift = abs(float(rel_pos[2]) - self.z_offset_m)
        shift = math.hypot(radial_shift, axial_shift)
        angle = _angle_deg(rel_axis, self.axis)
        self.max_shift_m = max(self.max_shift_m, shift)
        self.max_rotation_deg = max(self.max_rotation_deg, angle)
        self.lost_s = (
            0.0
            if robot.bilateral_finger_contact(NUT_BODY)
            else (self.lost_s + robot.model.opt.timestep)
        )
        self.max_lost_s = max(self.max_lost_s, self.lost_s)
        if not np.isfinite(shift) or not np.isfinite(angle):
            raise DemoError("螺母位姿无效")
        if self.lost_s > MAX_CONTACT_LOSS_S:
            raise DemoError(f"夹持螺母滑脱：双侧接触中断 {self.lost_s:.3f} s")
        if shift > MAX_GRASP_SHIFT_M or angle > MAX_GRASP_ROTATION_DEG:
            raise DemoError(
                f"夹持螺母滑脱：相对位移 {shift * 1000:.1f} mm，倾角变化 {angle:.1f}°"
            )


class ScrewNutRobot(SimRobot):
    """SimRobot with G2 contact-force feedback, priority-decoupled finger friction, and helical wrist drive."""

    def __init__(
        self,
        *,
        viewer: bool,
        calibration: KinematicsCalibration | None = None,
    ):
        self.grasp_monitor: GraspMonitor | None = None
        self.force_feedback = False
        self.last_force_update = -1.0
        self.active_waypoint_sink: list[np.ndarray] | None = None
        super().__init__(SCENE, viewer=viewer)
        self._configure_finger_contact_priority()
        if calibration is not None:
            apply_calibration_to_model(self.model, calibration, self.data)
            mujoco.mj_setConst(self.model, self.eval_data)
            mujoco.mj_forward(self.model, self.eval_data)

    def _configure_finger_contact_priority(self) -> None:
        """Assign priority 2 and high friction to Gripper G2 fingers while nut-to-screw stays low friction."""
        for finger_body in ("left_finger", "right_finger"):
            b_id = self.model.body(finger_body).id
            for g_id in range(self.model.ngeom):
                if (
                    int(self.model.geom_bodyid[g_id]) == b_id
                    and int(self.model.geom_group[g_id]) == 3
                ):
                    self.model.geom_priority[g_id] = 2
                    self.model.geom_condim[g_id] = 4
                    self.model.geom_friction[g_id] = [4.0, 0.015, 0.002]
                    self.model.geom_solref[g_id] = [0.003, 1.0]
                    self.model.geom_solimp[g_id] = [0.99, 0.999, 0.0001, 0.5, 2.0]

    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        if self.active_waypoint_sink is not None:
            self.active_waypoint_sink.append(
                np.asarray(q_goal[: self.n_arm], dtype=np.float64).copy()
            )
        super().move_joints(q_goal, duration, wait=wait)

    def check_viewer(self) -> None:
        if self._viewer is not None and not self._viewer.is_running():
            raise DemoError("可视化窗口提前关闭")

    def _step(self) -> None:
        self.check_viewer()
        if self.force_feedback and self.data.time - self.last_force_update >= FORCE_UPDATE_S:
            self.last_force_update = float(self.data.time)
            force = min(self.get_gripper_contact_forces(NUT_BODY))
            target = float(self.g2_controller.position_mm[0])
            adjustment_mm = float(np.clip((force - FORCE_TARGET_N) * 0.03, -0.15, 0.15))
            self.set_gripper_command(
                np.clip(target + adjustment_mm, 0.0, 84.0),
                speed_percent=20,
                force_percent=25,
            )
        super()._step()
        self.check_viewer()
        if self.grasp_monitor is not None:
            self.grasp_monitor.observe(self)

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
        """Move along a straight Cartesian path using smooth acceleration and soft-landing deceleration."""
        self.check_viewer()
        pos0, quat0 = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (np.asarray(pos, dtype=np.float64) - pos0)
        quat_goal = _canonical_quat(quat)
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
            raise DemoError(f"逆运动学不收敛：pos={pos}")

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
        """Descend along a vertical Cartesian line while rotating joint6 coaxially by +2*pi*turns.

        Uses ``smooth_step_durations`` so vertical translation and wrist rotation ramp up
        smoothly and decelerate to a soft landing while keeping spatial waypoints uniform.
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
            _canonical_quat(base_quat),
            q_start,
        )
        if not ok:
            raise DemoError(f"螺旋拧入逆运动学不收敛：target_pos={target_pos}")

        j6_start = float(self.data.qpos[5])
        j6_id = self.model.joint("joint6").id
        j6_lo, j6_hi = (float(v) for v in self.model.jnt_range[j6_id])
        max_cw_rad = (j6_hi - J6_LIMIT_MARGIN_RAD) - j6_start
        total_delta_rad = 2.0 * math.pi * float(turns)
        if total_delta_rad > max_cw_rad:
            raise DemoError(
                f"J6 顺时针旋转超限：当前 J6={math.degrees(j6_start):.1f}°，"
                f"请求旋转 +{math.degrees(total_delta_rad):.1f}°，"
                f"距 +360° 限位最大可用余量仅 +{math.degrees(max_cw_rad):.1f}°"
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
        """Move vertically while preserving the current joint6 wrist angle with smooth acc/dec."""
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
            _canonical_quat(cur_quat),
            self.data.qpos.copy(),
        )
        if not ok:
            raise DemoError(f"垂直撤离逆运动学不收敛：target_pos={target_pos}")
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


def _move_tcp(
    robot: ScrewNutRobot,
    pos: np.ndarray,
    quat: np.ndarray,
    duration: float,
    *,
    steps: int,
    soft_landing: bool = True,
    wait: bool = True,
) -> None:
    robot.move_cartesian_smooth(
        pos,
        quat,
        duration,
        steps=steps,
        soft_landing=soft_landing,
        wait=wait,
    )


def _require_bilateral(robot: ScrewNutRobot) -> None:
    if not robot.bilateral_finger_contact(NUT_BODY):
        raise DemoError("螺母无双侧指垫接触，拒绝抬起或释放")


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
        raise DemoError("抓取前复测失败：螺母位姿无效")
    xy_delta = float(np.linalg.norm(nut_pos[:2] - expected_xy[:2]))
    z_rel = float(nut_pos[2] - table_z)
    tilt = _angle_deg(nut_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if xy_delta > search_radius_m or abs(z_rel) > 0.015:
        raise DemoError(
            f"抓取前闭环拦截：螺母缺失或移出搜索半径 "
            f"(水平偏差 {xy_delta * 1000:.1f} mm > {search_radius_m * 1000:.0f} mm, z_rel={z_rel * 1000:.1f} mm)"
        )
    if tilt > 20.0:
        raise DemoError(f"抓取前闭环拦截：螺母倾角异常 ({tilt:.1f}° > 20.0°)")
    return nut_pos.copy(), xy_delta * 1000.0


def _verify_gripper_grasp_gap(robot: ScrewNutRobot) -> float:
    """Verify measured Gripper G2 opening matches the 20 mm upper hex body (rejecting empty grasp)."""
    meas_gap_mm = float(robot.get_measured_gripper_gap() * 1000.0)
    if meas_gap_mm < MIN_VALID_NUT_GRASP_GAP_MM:
        raise DemoError(
            f"闭环夹持拦截：实测夹持开度过小 ({meas_gap_mm:.1f} mm < "
            f"{MIN_VALID_NUT_GRASP_GAP_MM:.1f} mm)，疑似空抓！"
        )
    if meas_gap_mm > MAX_VALID_NUT_GRASP_GAP_MM:
        raise DemoError(
            f"闭环夹持拦截：实测夹持开度过大 ({meas_gap_mm:.1f} mm > "
            f"{MAX_VALID_NUT_GRASP_GAP_MM:.1f} mm)，疑似卡在底法兰上！"
        )
    return meas_gap_mm


def _configure_initial_scene(
    robot: ScrewNutRobot,
    nut_pose: InitialNutPose | None,
    screw_xy: np.ndarray,
    q_init: np.ndarray | None,
) -> float:
    """Configure initial arm state, tabletop-glued screw position, and free nut pose."""
    active_nut = nut_pose if nut_pose is not None else InitialNutPose()
    table_z = float(active_nut.table_z_m)

    # Set tabletop plane Z
    table_geom_id = robot.model.geom("table").id
    robot.model.geom_pos[table_geom_id, 2] = table_z

    # Set fixed screw body position (glued upright to tabletop at screw_xy)
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

    # Recompute model constants before writing custom qpos (mj_setConst resets data.qpos to qpos0)
    mujoco.mj_setConst(robot.model, robot.data)
    mujoco.mj_setConst(robot.model, robot.eval_data)

    if q_init is not None:
        q_arr = np.asarray(q_init, dtype=np.float64).reshape(robot.n_arm)
        robot.data.qpos[: robot.n_arm] = q_arr
        robot.data.ctrl[: robot.n_arm] = q_arr
        robot.eval_data.qpos[: robot.n_arm] = q_arr

    # Set free nut joint qpos/qvel
    jnt_id = robot.model.joint("nut_joint").id
    qpos_adr = int(robot.model.jnt_qposadr[jnt_id])
    dof_adr = int(robot.model.jnt_dofadr[jnt_id])
    pos, quat = active_nut.mujoco_pos_quat()
    robot.data.qpos[qpos_adr : qpos_adr + 3] = pos
    robot.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
    robot.data.qvel[dof_adr : dof_adr + 6] = 0.0

    mujoco.mj_forward(robot.model, robot.data)
    mujoco.mj_forward(robot.model, robot.eval_data)
    return table_z


def _plan_ratchet_strokes(thread_turns: float) -> list[float]:
    """Decompose total requested turns into J6-bounded strokes (intermediate strokes = 1.5 turns = 540 deg)."""
    if thread_turns <= 0.0 or thread_turns > 12.0:
        raise ValueError(f"thread_turns 必须在 (0.0, 12.0] 范围内，当前: {thread_turns}")
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
    """Verify the nut sits coaxially and self-locks/seats at expected_rel_z on the screw for 1.0s."""
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
            raise DemoError("螺母最终位姿无效")

        if bodies_in_contact(robot.model, robot.data, "left_finger", NUT_BODY) or bodies_in_contact(
            robot.model, robot.data, "right_finger", NUT_BODY
        ):
            raise DemoError("稳定性检查失败：夹爪仍接触螺母")

        if bodies_in_contact(robot.model, robot.data, NUT_BODY, SCREW_BODY):
            lost_screw_s = 0.0
        else:
            lost_screw_s += float(robot.model.opt.timestep)
            if lost_screw_s > 0.02:
                raise DemoError("装配检查失败：螺母未与固定螺丝保持物理接触")

        tilt = _angle_deg(nut_axis, up_vec)
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
            raise DemoError(
                f"螺母落座/螺纹自锁稳定性检查失败：同心误差 {xy_err * 1000:.2f} mm，"
                f"高度 {rel_z * 1000:.2f} mm (期望 ~{expected_rel_z * 1000:.2f} mm)，"
                f"倾角 {tilt:.2f}°，漂移 {drift * 1000:.2f} mm"
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
    """Verify every recorded arm waypoint stays strictly within model.jnt_range."""
    blocks = (
        waypoints.approach_waypoints
        + waypoints.lift_align_waypoints
        + waypoints.thread_waypoints
        + waypoints.regrip_waypoints
        + waypoints.retreat_waypoints
    )
    if not blocks:
        raise DemoError("未记录到任何关节轨迹航点")
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
            raise DemoError(
                f"joint{j + 1} 超出关节限位：实测 [{math.degrees(q_min_rad):.2f}°, "
                f"{math.degrees(q_max_rad):.2f}°]，限位 [{math.degrees(lo_rad):.2f}°, "
                f"{math.degrees(hi_rad):.2f}°]"
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
    """Run nut pick, coaxial pilot capture, and multi-stroke clockwise helical threading."""
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

    print(f"准备：张开夹爪至 {PRE_GRASP_GAP_MM:.1f} mm 并等待螺母静置稳定", flush=True)
    robot.set_gripper_command(PRE_GRASP_GAP_MM, speed_percent=40, force_percent=20)
    robot.settle(0.6)
    robot.check_viewer()

    expected_nut_xy = np.array(
        [active_nut_pose.xy_m[0], active_nut_pose.xy_m[1]], dtype=np.float64
    )
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
        f"接近螺母正上方观测/预抓取位 (空爪预置 J6={j6_plan['j6_approach_deg']:.1f}°，"
        f"单把顺时针可用余量 +{j6_plan['j6_max_cw_available_deg']:.1f}°)",
        flush=True,
    )
    robot.active_waypoint_sink = active_wps.approach_waypoints
    robot.move_joints(q_app[: robot.n_arm], 1.6, wait=True)
    robot.check_viewer()

    # Pre-grasp measurement gate: verify nut presence & refine exact XY before soft-landing descent
    measured_nut_pos, pre_grasp_delta_mm = _measure_nut_before_grasp(
        robot, expected_nut_xy, table_z=table_z
    )
    grasp_pos = np.array(
        [measured_nut_pos[0], measured_nut_pos[1], table_z + GRASP_TCP_Z_M],
        dtype=np.float64,
    )
    print(
        f"抓取前正上方复测确认螺母在位 (修正 Δxy={pre_grasp_delta_mm:.2f} mm)，软着陆减速下探夹持",
        flush=True,
    )
    _move_tcp(robot, grasp_pos, grasp_quat, 1.2, steps=24, soft_landing=True, wait=True)
    robot.active_waypoint_sink = None

    print("夹持螺母上层六角体：闭环校验实测开度与双侧指垫接触", flush=True)
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

    print("垂直抬起螺母并连续平移至胶水固定螺丝顶端正上方", flush=True)
    robot.active_waypoint_sink = active_wps.lift_align_waypoints
    lift_pos = np.array(
        [measured_nut_pos[0], measured_nut_pos[1], table_z + LIFT_TCP_Z_M],
        dtype=np.float64,
    )
    _move_tcp(robot, lift_pos, grasp_quat, 1.1, steps=22, soft_landing=False, wait=False)
    lifted_nut_pos, _ = _nut_pose(robot)
    lift_mm = float((lifted_nut_pos[2] - measured_nut_pos[2]) * 1000.0)
    if lifted_nut_pos[2] < measured_nut_pos[2] + 0.050 or _nut_touches_table(robot):
        raise DemoError("抬起失败：螺母未完全离开桌面")

    hover_pos = np.array(
        [active_screw_xy[0], active_screw_xy[1], table_z + HOVER_SCREW_TCP_Z_M],
        dtype=np.float64,
    )
    _move_tcp(robot, hover_pos, grasp_quat, 1.4, steps=28, soft_landing=True, wait=True)

    # Pre-capture coaxial alignment check: verify upright fixed screw and in-hand nut coaxiality
    live_screw_pos, live_screw_axis = _screw_pose(robot)
    live_nut_pos, _ = _nut_pose(robot)
    screw_delta = float(np.linalg.norm(live_screw_pos[:2] - active_screw_xy[:2]))
    screw_tilt = _angle_deg(live_screw_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if screw_delta > PRE_GRASP_SEARCH_RADIUS_M or screw_tilt > 10.0:
        raise DemoError(
            f"套入前闭环拦截：固定螺丝缺失或偏离 "
            f"(偏差 {screw_delta * 1000:.1f} mm, 倾角 {screw_tilt:.1f}°)"
        )
    xy_offset = live_screw_pos[:2] - live_nut_pos[:2]
    pre_capture_comp_mm = float(np.linalg.norm(xy_offset) * 1000.0)
    if pre_capture_comp_mm > 6.0:
        raise DemoError(f"套入前闭环拦截：螺母与固定螺丝同心度超限 ({pre_capture_comp_mm:.2f} mm > 6.0 mm)")
    capture_pos = np.array(
        [
            live_screw_pos[0],
            live_screw_pos[1],
            table_z + CAPTURE_SCREW_TCP_Z_M,
        ],
        dtype=np.float64,
    )
    print(
        f"套入前同心复测通过 (同心偏差 {pre_capture_comp_mm:.2f} mm)，软着陆导引套入 M12 螺杆首圈螺纹",
        flush=True,
    )
    _move_tcp(robot, capture_pos, grasp_quat, 1.0, steps=22, soft_landing=True, wait=True)
    robot.active_waypoint_sink = None
    capture_nut_pos, _ = _nut_pose(robot)
    capture_rel_z = float(capture_nut_pos[2] - table_z)

    print(
        f"开始 3D 螺旋旋紧：共 {thread_turns:.2f} 圈 ({thread_turns * 360.0:.0f}°)，"
        f"分 {len(strokes)} 把倒手执行 (螺距 {SCREW_PITCH_M * 1000:.2f} mm/圈)",
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
            f"  第 {stroke_idx + 1}/{len(strokes)} 把 ({st_turns:.2f} 圈): "
            f"J6 [{helical_info['j6_start_deg']:.1f}° -> {helical_info['j6_end_deg']:.1f}°], "
            f"TCP Z={(current_tcp_z - table_z) * 1000:.2f} mm, "
            f"螺母高度 {(nut_before[2] - table_z) * 1000:.2f} -> {(nut_after[2] - table_z) * 1000:.2f} mm",
            flush=True,
        )

        # If closed-loop Z backoff lifted TCP Z above the bottom target, append adaptive continuation stroke
        if (
            stroke_idx == len(strokes) - 1
            and z_adapt_count > 0
            and current_tcp_z > table_z + THREAD_END_TCP_Z_M + 0.0012
            and extra_strokes_used < MAX_ADAPTIVE_EXTRA_STROKES
        ):
            strokes.append(RATCHET_STROKE_TURNS)
            extra_strokes_used += 1
            print(
                f"    [自适应补拧 #{extra_strokes_used}] 当前 TCP Z={(current_tcp_z - table_z) * 1000:.2f} mm "
                f"> 落座目标 {THREAD_END_TCP_Z_M * 1000:.2f} mm，自动追加 {RATCHET_STROKE_TURNS:.2f} 圈补拧行程",
                flush=True,
            )

        if stroke_idx < len(strokes) - 1:
            # Open gripper wide (50 mm >> 29 mm flange / 23.1 mm hex corners) to release nut, rewind J6, and regrip.
            robot.force_feedback = False
            robot.grasp_monitor = None
            robot.set_gripper_command(REGRIP_OPEN_GAP_MM, speed_percent=60, force_percent=50)
            robot.settle(0.55)
            robot.check_viewer()

            nut_hold, _ = _nut_pose(robot)
            settle_m = abs(float(nut_hold[2] - nut_after[2]))
            max_regrip_settle_m = max(max_regrip_settle_m, settle_m)
            if settle_m > MAX_REGRIP_THREAD_SETTLE_M:
                raise DemoError(
                    f"半途松爪倒手时螺母未在螺纹上自锁：下沉 {settle_m * 1000:.2f} mm"
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
                raise DemoError(f"第 {stroke_idx + 1} 把倒手回转逆运动学不收敛：{regrip_pos}")
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
                    f"    [Z轴闭环自适应 #{z_adapt_count}] 第 {stroke_idx + 1} 把重夹实测开度 "
                    f"{meas_regrip_gap_mm:.2f} mm > {MAX_HEX_FLAT_GAP_MM:.1f} mm (触碰 29 mm 底法兰)，"
                    f"上抬 TCP Z +{dz_up * 1000:.1f} mm 至 {(current_tcp_z - table_z) * 1000:.2f} mm 重新夹持六角体",
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
                    raise DemoError(f"第 {stroke_idx + 1} 把 Z 轴自适应上抬逆运动学不收敛：{regrip_pos}")
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

    print("释放螺母并平滑垂直撤离机械臂", flush=True)
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
        f"成功完成拧螺母仿真：共 {len(strokes)} 把旋紧 {thread_turns:.2f} 圈 "
        f"(累计净转角 {metrics['rotated_deg']:.1f}°，J6 上限最小余量 {metrics['j6_upper_margin_deg']:.1f}°)，"
        f"螺母高度 {metrics['nut_z_mm']:.2f} mm (期望 {metrics['expected_nut_z_mm']:.2f} mm)，"
        f"同心误差 {metrics['xy_error_mm']:.2f} mm，最大倾角 {metrics['tilt_deg']:.2f}°，"
        f"独立稳定 {metrics['stable_s']:.3f} s。",
        flush=True,
    )
    return metrics


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration dictionary from YAML file (falls back to empty dict if missing)."""
    path = DEFAULT_CONFIG_YAML if config_path is None else Path(config_path)
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _write_json_report(report_path: str | Path | None, payload: dict[str, Any]) -> None:
    if not report_path:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_YAML),
        help="配置文件 YAML 路径 (默认: examples/screw_nut/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="运行模式: sim (默认纯仿真), dry-run (实机视觉+数字孪生预演), sdk-sim (控制器内仿真), real (真机闭环执行)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否以无头模式运行 MuJoCo (默认读取 config.yaml 中的 headless，未指定时为 false)",
    )
    parser.add_argument("--ip", default=None, help="xArm 6 控制器 IP 地址")
    parser.add_argument(
        "--calibration",
        default=None,
        help="xArm 6 单机运动学标定 YAML 路径",
    )
    parser.add_argument(
        "--handeye",
        default=None,
        help="Eye-in-Hand 手眼标定结果 YAML 路径",
    )
    parser.add_argument(
        "--nut-pose",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "YAW_DEG"),
        default=None,
        help="指定螺母初始桌面坐标与偏航角 (x_m, y_m, yaw_deg)",
    )
    parser.add_argument(
        "--screw-xy",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
        default=None,
        help="指定桌面胶水固定螺丝的水平坐标 (x_m, y_m)",
    )
    parser.add_argument(
        "--thread-turns",
        type=float,
        default=None,
        help="螺母套入螺杆后的顺时针旋紧圈数 (默认 10.0 圈 = 3600°)",
    )
    parser.add_argument(
        "--confirm-real",
        action="store_true",
        help="显式授权单周期真机运动 (--mode real 必填)",
    )
    parser.add_argument("--cycles", type=int, default=1, help="执行周期数 (硬件模式强制为 1)")
    parser.add_argument(
        "--post-verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="真机完成拧螺母回到观测位后是否自动拍照闭环确认 (默认读取 config.yaml)",
    )
    parser.add_argument("--save-vis", default=None, help="保存抓取前相机检测叠加图像路径 (PNG)")
    parser.add_argument(
        "--save-after-vis",
        default=None,
        help="保存拧螺母后相机闭环确认叠加图像路径 (PNG)",
    )
    parser.add_argument("--report", default=None, help="保存结构化运行报告路径 (JSON)")
    return parser


def main(argv: list[str] | None = None) -> int:
    from perception import DEFAULT_HANDEYE_YAML
    from sim2real import (
        DEFAULT_CALIBRATION_YAML,
        DEFAULT_ROBOT_IP,
        run_sim2real_pipeline,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        robot_cfg = cfg.get("robot", {}) if isinstance(cfg.get("robot"), dict) else {}
        cam_cfg = cfg.get("camera", {}) if isinstance(cfg.get("camera"), dict) else {}
        task_cfg = cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}

        mode = args.mode if args.mode is not None else str(cfg.get("mode", "sim"))
        headless = (
            bool(args.headless)
            if args.headless is not None
            else bool(cfg.get("headless", False))
        )
        thread_turns = (
            float(args.thread_turns)
            if args.thread_turns is not None
            else float(task_cfg.get("thread_turns", DEFAULT_THREAD_TURNS))
        )
        ip = args.ip if args.ip is not None else str(robot_cfg.get("ip", DEFAULT_ROBOT_IP))
        repo_root = Path(__file__).resolve().parents[2]

        def _resolve_cfg_path(val: str | Path) -> str:
            p = Path(val).expanduser()
            if not p.is_absolute() and not p.exists() and (repo_root / p).exists():
                return str((repo_root / p).resolve())
            return str(p)

        calibration_path = _resolve_cfg_path(
            args.calibration
            if args.calibration is not None
            else str(robot_cfg.get("calibration", DEFAULT_CALIBRATION_YAML))
        )
        handeye_path = _resolve_cfg_path(
            args.handeye
            if args.handeye is not None
            else str(cam_cfg.get("handeye", DEFAULT_HANDEYE_YAML))
        )
        post_verify = (
            bool(args.post_verify)
            if args.post_verify is not None
            else bool(task_cfg.get("post_verify", True))
        )
        save_vis = args.save_vis if args.save_vis is not None else task_cfg.get("save_vis")
        save_after_vis = (
            args.save_after_vis
            if args.save_after_vis is not None
            else task_cfg.get("save_after_vis")
        )
        report_path = args.report if args.report is not None else task_cfg.get("report")

        nut_override: tuple[float, float, float] | None = None
        if args.nut_pose is not None:
            nut_override = (
                float(args.nut_pose[0]),
                float(args.nut_pose[1]),
                float(args.nut_pose[2]),
            )
            nut_pose = InitialNutPose(
                xy_m=(nut_override[0], nut_override[1]),
                yaw_deg=nut_override[2],
            )
        else:
            cfg_nut_xy = task_cfg.get("nut_xy", DEFAULT_NUT_XY.tolist())
            cfg_nut_yaw = float(task_cfg.get("nut_yaw_deg", 0.0))
            nut_pose = InitialNutPose(
                xy_m=(float(cfg_nut_xy[0]), float(cfg_nut_xy[1])),
                yaw_deg=cfg_nut_yaw,
            )

        screw_override: tuple[float, float] | None = None
        if args.screw_xy is not None:
            screw_override = (float(args.screw_xy[0]), float(args.screw_xy[1]))
            screw_xy = np.array([screw_override[0], screw_override[1]], dtype=np.float64)
        else:
            cfg_screw_xy = task_cfg.get("screw_xy", DEFAULT_SCREW_XY.tolist())
            screw_xy = np.array([float(cfg_screw_xy[0]), float(cfg_screw_xy[1])], dtype=np.float64)

        if mode == "sim":
            with ScrewNutRobot(viewer=not headless) as robot:
                metrics = run_screw_nut(
                    robot,
                    nut_pose=nut_pose,
                    screw_xy=screw_xy,
                    thread_turns=thread_turns,
                )
            report = {
                "ok": True,
                "mode": "sim",
                "nut_xy_m": list(nut_pose.xy_m),
                "nut_yaw_deg": float(nut_pose.yaw_deg),
                "screw_xy_m": [float(screw_xy[0]), float(screw_xy[1])],
                "metrics": metrics,
            }
            _write_json_report(report_path, report)
            return 0

        run_sim2real_pipeline(
            mode=mode,
            thread_turns=thread_turns,
            ip=ip,
            calibration_path=calibration_path,
            handeye_path=handeye_path,
            nut_pose_override=nut_override,
            screw_xy_override=screw_override,
            headless=headless,
            confirm_real=bool(args.confirm_real),
            cycles=int(args.cycles),
            post_verify=post_verify,
            save_vis_path=save_vis,
            save_after_vis_path=save_after_vis,
            report_path=report_path,
        )
        return 0
    except (DemoError, ValueError, RuntimeError, OSError) as exc:
        print(f"拧螺母任务失败：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
