#!/usr/bin/env python3
"""Pick up multiple (>= 2) sideways paper cups and stack (nest) them upright in MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# Reference the read-only ufactory_mujoco package in the sibling repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ufactory_mujoco"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ufactory_mujoco.calibration import (
    KinematicsCalibration,
    apply_calibration_to_model,
)
from ufactory_mujoco.grippers import bodies_in_contact
from ufactory_mujoco.sim_robot import SimRobot
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

SCENE = Path(__file__).with_name("scene.xml")
DEFAULT_CONFIG_YAML = Path(__file__).with_name("config.yaml")
MODES = ("sim", "dry-run", "sdk-sim", "real")

ALL_CUP_BODIES: tuple[str, ...] = ("paper_cup_1", "paper_cup_2", "paper_cup_3")
CUP_HEIGHT_M = 0.088
CUP_BOT_DIAM_M = 0.053
CUP_TOP_DIAM_M = 0.075
CUP_SPAWN_Z_OFFSET_M = 0.0375

DEFAULT_TARGET_XY = np.array([0.30, 0.16], dtype=np.float64)
GRASP_AXIS_OFFSET_M = 0.040
GRASP_TCP_Z_M = 0.014
INITIAL_GRASP_GAP_MM = 53.0  # Cup outer diameter at 40mm height is 63.0mm
NEST_HOVER_CLEARANCE_M = 0.025  # Hover cup bottom 25mm above lower cup's top rim
NEST_INSERT_DEPTH_M = 0.004  # Lower cup bottom 4mm inside lower cup's mouth (fingers still ~22mm above rim)
MIN_NEST_STEP_Z_M = 0.004  # Minimum bottom-to-bottom vertical step between nested cups
MAX_NEST_STEP_Z_M = 0.035  # Maximum bottom-to-bottom vertical step (must not jam near rim)
MAX_COAXIAL_XY_ERR_M = 0.008  # Maximum XY offset between adjacent nested cups (8 mm)
MAX_TARGET_XY_ERR_M = 0.030  # Maximum XY error of the stack relative to target_xy (30 mm)
MAX_TILT_DEG = 10.0
MAX_REST_DRIFT_M = 0.002

MAX_GRASP_SHIFT_M = 0.025
MAX_GRASP_ROTATION_DEG = 25.0
MAX_CONTACT_LOSS_S = 0.10
FORCE_TARGET_N = 2.0
FORCE_UPDATE_S = 0.05


class DemoError(RuntimeError):
    """A task, contact, or viewer condition prevented a successful stacking run."""


@dataclass(frozen=True)
class InitialCupPose:
    """Initial sideways paper cup configuration on the table."""

    bottom_xy_m: tuple[float, float]
    yaw_deg: float = 0.0
    table_z_m: float = 0.0

    def mujoco_pos_quat(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(
            [self.bottom_xy_m[0], self.bottom_xy_m[1], self.table_z_m + CUP_SPAWN_Z_OFFSET_M],
            dtype=np.float64,
        )
        r_sideways = Rotation.from_euler("z", float(self.yaw_deg), degrees=True) * (
            Rotation.from_euler("y", 90.0, degrees=True)
        )
        q_xyzw = r_sideways.as_quat()
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        return pos, quat_wxyz


@dataclass
class CupCycleWaypoints:
    """Recorded joint waypoints for a single cup's pick-flip-stack cycle."""

    cup_name: str
    approach_waypoints: list[np.ndarray] = field(default_factory=list)
    lift_flip_place_waypoints: list[np.ndarray] = field(default_factory=list)
    retreat_waypoints: list[np.ndarray] = field(default_factory=list)
    grasp_gap_mm: float = INITIAL_GRASP_GAP_MM


def _cup_index(cup_name: str) -> int:
    for idx, name in enumerate(ALL_CUP_BODIES, start=1):
        if name == cup_name:
            return idx
    raise ValueError(f"未知的纸杯刚体名: {cup_name!r}")


def _cup_bottom_geom(cup_name: str) -> str:
    return f"cup_{_cup_index(cup_name)}_bottom"


def _cup_pose(robot: SimRobot, cup_name: str) -> tuple[np.ndarray, np.ndarray]:
    body = robot.data.body(cup_name)
    return body.xpos.copy(), body.xmat.reshape(3, 3)[:, 2].copy()


def _tool_relative_cup(robot: SimRobot, cup_name: str) -> tuple[np.ndarray, np.ndarray]:
    bottom, axis = _cup_pose(robot, cup_name)
    position = robot.data.site_xpos[robot._site]
    rotation = robot.data.site_xmat[robot._site].reshape(3, 3)
    return rotation.T @ (bottom - position), rotation.T @ axis


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _contact_between(robot: SimRobot, geom_a: str, geom_b: str) -> bool:
    a, b = robot.model.geom(geom_a).id, robot.model.geom(geom_b).id
    return any(
        {int(c.geom1), int(c.geom2)} == {a, b} and c.dist <= 0.0
        for c in robot.data.contact[: robot.data.ncon]
    )


def _cup_touches_table(robot: SimRobot, cup_name: str) -> bool:
    table = robot.model.geom("table").id
    cup_body = robot.model.body(cup_name).id
    for contact in robot.data.contact[: robot.data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if contact.dist <= 0.0 and (
            (a == table and robot.model.geom_bodyid[b] == cup_body)
            or (b == table and robot.model.geom_bodyid[a] == cup_body)
        ):
            return True
    return False


def _yaw_rotate_quat(yaw_rad: float, base_quat: np.ndarray) -> np.ndarray:
    """Left-multiply base_quat by a world +Z rotation of yaw_rad."""
    if abs(yaw_rad) < 1e-6:
        return base_quat.copy()
    z_rot = np.array([np.cos(0.5 * yaw_rad), 0.0, 0.0, np.sin(0.5 * yaw_rad)], dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, z_rot, base_quat)
    return out


def _rotate_cup_up(axis: np.ndarray, tool_quat: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    cross = np.cross(axis, up)
    length = float(np.linalg.norm(cross))
    if length < 1e-9:
        correction = (
            np.array([1.0, 0.0, 0.0, 0.0]) if axis[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        )
    else:
        angle = float(np.arctan2(length, np.dot(axis, up)))
        correction = np.r_[np.cos(angle / 2), cross / length * np.sin(angle / 2)]
    result = np.empty(4)
    mujoco.mju_mulQuat(result, correction, tool_quat)
    return result


@dataclass
class GraspMonitor:
    """Detect loss of bilateral contact and excessive cup motion relative to the TCP."""

    cup_name: str
    position: np.ndarray
    axis: np.ndarray
    lost_s: float = 0.0
    max_lost_s: float = 0.0
    max_shift_m: float = 0.0
    max_rotation_deg: float = 0.0

    @classmethod
    def start(cls, robot: SimRobot, cup_name: str) -> GraspMonitor:
        position, axis = _tool_relative_cup(robot, cup_name)
        return cls(cup_name, position, axis)

    def observe(self, robot: SimRobot) -> None:
        position, axis = _tool_relative_cup(robot, self.cup_name)
        shift = float(np.linalg.norm(position - self.position))
        angle = _angle_deg(axis, self.axis)
        self.max_shift_m = max(self.max_shift_m, shift)
        self.max_rotation_deg = max(self.max_rotation_deg, angle)
        self.lost_s = (
            0.0
            if robot.bilateral_finger_contact(self.cup_name)
            else (self.lost_s + robot.model.opt.timestep)
        )
        self.max_lost_s = max(self.max_lost_s, self.lost_s)
        if not np.isfinite(shift) or not np.isfinite(angle):
            raise DemoError(f"{self.cup_name} 位姿无效")
        if self.lost_s > MAX_CONTACT_LOSS_S:
            raise DemoError(f"{self.cup_name} 夹持中滑脱：双侧接触中断 {self.lost_s:.3f} s")
        if shift > MAX_GRASP_SHIFT_M or angle > MAX_GRASP_ROTATION_DEG:
            raise DemoError(
                f"{self.cup_name} 夹持中滑脱：相对位移 {shift * 1000:.1f} mm，转角 {angle:.1f}°"
            )


class StackCupRobot(SimRobot):
    """SimRobot with G2 contact-force feedback and priority-decoupled finger friction."""

    def __init__(
        self,
        *,
        viewer: bool,
        calibration: KinematicsCalibration | None = None,
    ):
        self.grasp_monitor: GraspMonitor | None = None
        self.active_cup_body: str | None = None
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
        """Assign priority 2 and high friction to Gripper G2 fingers so cup-to-cup stays low friction."""
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
        if (
            self.force_feedback
            and self.active_cup_body is not None
            and self.data.time - self.last_force_update >= FORCE_UPDATE_S
        ):
            self.last_force_update = float(self.data.time)
            force = min(self.get_gripper_contact_forces(self.active_cup_body))
            target = float(self.g2_controller.position_mm[0])
            adjustment_mm = float(np.clip((force - FORCE_TARGET_N) * 0.04, -0.2, 0.2))
            self.set_gripper_command(
                np.clip(target + adjustment_mm, 0.0, 84.0),
                speed_percent=20,
                force_percent=10,
            )
        super()._step()
        self.check_viewer()
        if self.grasp_monitor is not None:
            self.grasp_monitor.observe(self)


def _move_tcp(
    robot: StackCupRobot,
    pos: np.ndarray,
    quat: np.ndarray,
    duration: float,
    *,
    steps: int,
    wait: bool = True,
) -> None:
    robot.check_viewer()
    try:
        robot.move_tcp(pos, quat, duration, wait=wait, steps=steps)
    except RuntimeError as exc:
        if isinstance(exc, DemoError):
            raise
        raise DemoError(f"逆运动学或轨迹执行失败：{exc}") from exc
    robot.check_viewer()


def _require_bilateral(robot: StackCupRobot, cup_name: str) -> None:
    if not robot.bilateral_finger_contact(cup_name):
        raise DemoError(f"{cup_name} 无双侧指垫接触，拒绝抬起或释放")


def _configure_initial_scene(
    robot: StackCupRobot,
    num_cups: int,
    cup_poses: list[InitialCupPose] | None,
    target_xy: np.ndarray,
    q_init: np.ndarray | None,
) -> tuple[list[str], float]:
    """Configure active cups, hide unused cups, and set target marker position."""
    if num_cups < 2 or num_cups > len(ALL_CUP_BODIES):
        raise ValueError(f"纸杯数量 num_cups 必须在 [2, {len(ALL_CUP_BODIES)}] 范围内，当前: {num_cups}")

    if q_init is not None:
        q_arr = np.asarray(q_init, dtype=np.float64).reshape(robot.n_arm)
        robot.data.qpos[: robot.n_arm] = q_arr
        robot.data.ctrl[: robot.n_arm] = q_arr
        robot.eval_data.qpos[: robot.n_arm] = q_arr

    table_z = 0.0
    if cup_poses is not None:
        if len(cup_poses) < num_cups:
            raise ValueError(f"提供的纸杯初始位姿数量 ({len(cup_poses)}) 少于 num_cups ({num_cups})")
        table_z = float(cup_poses[0].table_z_m)
        for idx in range(num_cups):
            cup_name = ALL_CUP_BODIES[idx]
            jnt_id = robot.model.joint(f"{cup_name}_joint").id
            qpos_adr = int(robot.model.jnt_qposadr[jnt_id])
            dof_adr = int(robot.model.jnt_dofadr[jnt_id])
            pos, quat = cup_poses[idx].mujoco_pos_quat()
            robot.data.qpos[qpos_adr : qpos_adr + 3] = pos
            robot.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
            robot.data.qvel[dof_adr : dof_adr + 6] = 0.0

    # Park and disable collision/visibility for unused cup bodies (e.g. paper_cup_3 when num_cups == 2)
    for idx, cup_name in enumerate(ALL_CUP_BODIES):
        b_id = robot.model.body(cup_name).id
        active = idx < num_cups
        if not active:
            jnt_id = robot.model.joint(f"{cup_name}_joint").id
            qpos_adr = int(robot.model.jnt_qposadr[jnt_id])
            dof_adr = int(robot.model.jnt_dofadr[jnt_id])
            robot.data.qpos[qpos_adr : qpos_adr + 3] = [-0.60 - 0.15 * idx, 0.0, 0.0375]
            robot.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            robot.data.qvel[dof_adr : dof_adr + 6] = 0.0
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
    """Verify that all N stacked cups stand upright, nested coaxially, without finger support."""
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
                raise DemoError(f"{cup} 最终位姿无效")
            if bodies_in_contact(robot.model, robot.data, "left_finger", cup) or bodies_in_contact(
                robot.model, robot.data, "right_finger", cup
            ):
                raise DemoError(f"稳定性检查失败：夹爪仍接触 {cup}")

            tilt = _angle_deg(axis, up_vec)
            err = float(np.linalg.norm(bottom[:2] - target_xy))
            drift = float(np.linalg.norm(bottom - start_positions[cup]))
            max_tilt = max(max_tilt, tilt)
            max_target_error = max(max_target_error, err)
            max_drift = max(max_drift, drift)

            if tilt > MAX_TILT_DEG or err > MAX_TARGET_XY_ERR_M or drift > MAX_REST_DRIFT_M:
                raise DemoError(
                    f"稳定性检查失败 ({cup})：倾角 {tilt:.1f}°，落点误差 {err * 1000:.1f} mm，"
                    f"漂移 {drift * 1000:.1f} mm"
                )

        # Base cup must remain in contact with the table
        base_cup = active_cups[0]
        if not _contact_between(robot, "table", _cup_bottom_geom(base_cup)):
            raise DemoError(f"稳定性检查失败：底层基准杯 {base_cup} 杯底未持续接触桌面")

        # Each upper cup k must contact cup k-1 and sit nested inside it
        for idx in range(1, len(active_cups)):
            lower_cup = active_cups[idx - 1]
            upper_cup = active_cups[idx]
            bot_low, _ = poses[lower_cup]
            bot_high, _ = poses[upper_cup]
            dz = float(bot_high[2] - bot_low[2])
            coax_err = float(np.linalg.norm(bot_high[:2] - bot_low[:2]))
            max_coaxial_error = max(max_coaxial_error, coax_err)

            if not bodies_in_contact(robot.model, robot.data, lower_cup, upper_cup):
                raise DemoError(
                    f"堆叠检查失败：{upper_cup} 与下层 {lower_cup} 未发生物理接触"
                )
            if not (MIN_NEST_STEP_Z_M <= dz <= MAX_NEST_STEP_Z_M):
                raise DemoError(
                    f"堆叠检查失败：{upper_cup} 相对 {lower_cup} 的嵌套高度差异常 "
                    f"({dz * 1000:.1f} mm 不在 [{MIN_NEST_STEP_Z_M * 1000:.0f}, "
                    f"{MAX_NEST_STEP_Z_M * 1000:.0f}] mm 范围内)"
                )
            if coax_err > MAX_COAXIAL_XY_ERR_M:
                raise DemoError(
                    f"堆叠检查失败：{upper_cup} 与 {lower_cup} 同心度偏差过大 "
                    f"({coax_err * 1000:.1f} mm > {MAX_COAXIAL_XY_ERR_M * 1000:.1f} mm)"
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
    """Run multi-cup pick, 90-degree flip, and coaxial nesting stack task."""
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

    print(f"准备：张开夹爪并等待 {len(active_cups)} 个纸杯稳定", flush=True)
    robot.open_gripper(speed_percent=40, force_percent=10)
    robot.settle(1.0)
    robot.check_viewer()
    initial_poses_map = {name: _cup_pose(robot, name) for name in active_cups}

    max_lift_mm = 0.0
    max_shift_mm = 0.0
    max_rot_deg = 0.0
    max_loss_s = 0.0

    for idx, cup_name in enumerate(active_cups):
        cycle_wps = CupCycleWaypoints(cup_name=cup_name)
        if recorded_cycles is not None:
            recorded_cycles.append(cycle_wps)

        # Pass through overhead transition posture between cups so the gripper stays high above
        # all cups on the table while unflipping the wrist; sim2real blends through this via-point
        # continuously without stopping.
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
        grasp_quat = _yaw_rotate_quat(axis_yaw_rad, GRIP_DOWN_QUAT)

        print(f"[{idx + 1}/{len(active_cups)}] 接近 {cup_name} 杯身", flush=True)
        robot.active_waypoint_sink = cycle_wps.approach_waypoints
        q_seed = robot.data.qpos.copy()
        q_seed[: robot.n_arm] = transition_q
        q_app, ok = robot.solve_tcp(approach, grasp_quat, q_init=q_seed)
        robot.check_viewer()
        if not ok:
            raise DemoError(f"逆运动学不收敛：{cup_name} 接近位置 {approach}")
        robot.move_joints(q_app[: robot.n_arm], 2.2, wait=False)
        robot.check_viewer()
        _move_tcp(robot, grasp, grasp_quat, 1.8, steps=30, wait=True)
        robot.active_waypoint_sink = None

        print(f"[{idx + 1}/{len(active_cups)}] 夹持 {cup_name}：确认双侧接触", flush=True)
        robot.set_gripper_command(INITIAL_GRASP_GAP_MM, speed_percent=20, force_percent=10)
        robot.settle(1.6)
        robot.check_viewer()
        gap = max(0.0, robot.get_measured_gripper_gap() * 1000.0 - 1.0)
        robot.set_gripper_command(gap, speed_percent=20, force_percent=10)
        robot.settle(0.8)
        robot.check_viewer()
        _require_bilateral(robot, cup_name)

        monitor = GraspMonitor.start(robot, cup_name)
        robot.grasp_monitor = monitor
        robot.active_cup_body = cup_name
        robot.force_feedback = True
        robot.last_force_update = -1.0

        print(f"[{idx + 1}/{len(active_cups)}] 抬起并翻转 {cup_name}", flush=True)
        robot.active_waypoint_sink = cycle_wps.lift_flip_place_waypoints
        _move_tcp(robot, grasp + np.array([0.0, 0.0, 0.14]), grasp_quat, 1.8, steps=28, wait=False)
        lifted_bottom, _ = _cup_pose(robot, cup_name)
        lift_mm = float((lifted_bottom[2] - initial_bottom[2]) * 1000.0)
        max_lift_mm = max(max_lift_mm, lift_mm)
        if lifted_bottom[2] < initial_bottom[2] + 0.05 or _cup_touches_table(robot, cup_name):
            raise DemoError(f"抬起失败：{cup_name} 未完全离开桌面")

        # Align cup horizontal axis along +Y (for +Y target) or -Y (for -Y target) while gripper
        # is still vertical, then flip cup upright in a single pitch move toward the target side.
        place_side = -1.0 if active_target_xy[1] <= 0.0 else 1.0
        pre_flip_quat = _yaw_rotate_quat(place_side * 0.5 * np.pi, GRIP_DOWN_QUAT)
        mid_pos = np.array([0.36, -0.05 * place_side, table_z + 0.22], dtype=np.float64)
        _move_tcp(robot, mid_pos, pre_flip_quat, 2.2, steps=35, wait=True)
        position, quat = robot.get_tcp_pose()
        _, axis = _cup_pose(robot, cup_name)
        _move_tcp(robot, position, _rotate_cup_up(axis, quat), 2.6, steps=45, wait=True)

        if idx == 0:
            print(f"[{idx + 1}/{len(active_cups)}] 放置底层基准杯 {cup_name}", flush=True)
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            hover_place = position + np.r_[
                active_target_xy - bottom[:2],
                (table_z + NEST_HOVER_CLEARANCE_M) - bottom[2],
            ]
            _move_tcp(robot, hover_place, quat, 2.5, steps=35, wait=True)

            target_z_touch = 0.0001 if q_init is None else -0.0004
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            destination = position + np.r_[
                active_target_xy - bottom[:2],
                (table_z + target_z_touch) - bottom[2],
            ]
            _move_tcp(robot, destination, quat, 2.2, steps=35, wait=True)
            cycle_wps.grasp_gap_mm = float(robot.g2_controller.position_mm[0])

            if not _contact_between(robot, "table", _cup_bottom_geom(cup_name)):
                position, quat = robot.get_tcp_pose()
                bottom, _ = _cup_pose(robot, cup_name)
                destination = position + np.r_[
                    active_target_xy - bottom[:2],
                    (table_z - 0.0005) - bottom[2],
                ]
                _move_tcp(robot, destination, quat, 0.8, steps=12, wait=True)
            robot.active_waypoint_sink = None
            if not _contact_between(robot, "table", _cup_bottom_geom(cup_name)):
                raise DemoError(f"放置失败：{cup_name} 杯底未接触桌面")
        else:
            prev_cup = active_cups[idx - 1]
            prev_bottom, prev_axis = _cup_pose(robot, prev_cup)
            prev_tilt = _angle_deg(prev_axis, np.array([0.0, 0.0, 1.0]))
            if prev_tilt > MAX_TILT_DEG:
                raise DemoError(f"下层纸杯 {prev_cup} 已倾倒 (倾角 {prev_tilt:.1f}°)，中止堆叠")
            prev_rim_z = float(prev_bottom[2]) + CUP_HEIGHT_M

            print(
                f"[{idx + 1}/{len(active_cups)}] 同心对准并将 {cup_name} 套入 {prev_cup}",
                flush=True,
            )
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            hover_dest = position + np.r_[
                prev_bottom[:2] - bottom[:2],
                (prev_rim_z + NEST_HOVER_CLEARANCE_M) - bottom[2],
            ]
            _move_tcp(robot, hover_dest, quat, 2.5, steps=35, wait=True)

            # Re-measure live cup bottom and lower into the mouth of prev_cup (fingers stay above rim)
            prev_bottom, _ = _cup_pose(robot, prev_cup)
            prev_rim_z = float(prev_bottom[2]) + CUP_HEIGHT_M
            position, quat = robot.get_tcp_pose()
            bottom, _ = _cup_pose(robot, cup_name)
            insert_dest = position + np.r_[
                prev_bottom[:2] - bottom[:2],
                (prev_rim_z - NEST_INSERT_DEPTH_M) - bottom[2],
            ]
            _move_tcp(robot, insert_dest, quat, 2.2, steps=35, wait=True)
            cycle_wps.grasp_gap_mm = float(robot.g2_controller.position_mm[0])
            robot.active_waypoint_sink = None

        _require_bilateral(robot, cup_name)
        robot.force_feedback = False
        robot.active_cup_body = None
        robot.grasp_monitor = None

        max_shift_mm = max(max_shift_mm, monitor.max_shift_m * 1000.0)
        max_rot_deg = max(max_rot_deg, monitor.max_rotation_deg)
        max_loss_s = max(max_loss_s, monitor.max_lost_s)

        print(f"[{idx + 1}/{len(active_cups)}] 释放 {cup_name} 并安全撤离", flush=True)
        robot.open_gripper(speed_percent=25, force_percent=40)
        robot.settle(0.6)
        robot.check_viewer()

        robot.active_waypoint_sink = cycle_wps.retreat_waypoints
        position, quat = robot.get_tcp_pose()
        retreat_pos = position + np.array([0.0, -0.095 * place_side, 0.015], dtype=np.float64)
        _move_tcp(robot, retreat_pos, quat, 1.5, steps=22, wait=False)
        _move_tcp(robot, retreat_pos + np.array([0.0, 0.0, 0.10]), quat, 1.4, steps=20, wait=True)
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
        f"成功完成 {metrics['num_cups']} 杯同心堆叠：目标落点误差 {metrics['error_mm']:.2f} mm，"
        f"层间同心误差 {metrics['coaxial_error_mm']:.2f} mm，层间嵌套高度差 [{steps_str}]，"
        f"最大倾角 {metrics['tilt_deg']:.2f}°，独立稳定 {metrics['stable_s']:.3f} s。",
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
        help="配置文件 YAML 路径 (默认: examples/stack_cups/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="运行模式: sim (默认纯仿真), dry-run (实机视觉+数字孪生预演), sdk-sim (控制器内仿真), real (真机执行)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否以无头模式运行 MuJoCo (默认读取 config.yaml 中的 headless，未指定时为 false)",
    )
    parser.add_argument(
        "--num-cups",
        type=int,
        choices=(2, 3),
        default=None,
        help="参与夹取与同心堆叠的纸杯数量 (支持 2 或 3，默认 2)",
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
        "--target-xy",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
        default=None,
        help="指定目标堆叠点水平坐标 (x_m, y_m)",
    )
    parser.add_argument(
        "--cup-poses",
        nargs="+",
        type=float,
        metavar="VAL",
        default=None,
        help="手动指定多个侧卧纸杯位姿，按 X1 Y1 YAW1 X2 Y2 YAW2 ... 顺序传入 (3 的倍数个浮点数)",
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
        help="真机堆叠回到观测位后是否自动拍照闭环确认 (默认读取 config.yaml)",
    )
    parser.add_argument("--save-vis", default=None, help="保存抓取前相机多纸杯检测叠加图像路径 (PNG)")
    parser.add_argument(
        "--save-after-vis",
        default=None,
        help="保存堆叠后相机闭环确认叠加图像路径 (PNG)",
    )
    parser.add_argument("--report", default=None, help="保存结构化运行报告路径 (JSON)")
    return parser


def main(argv: list[str] | None = None) -> int:
    from perception import DEFAULT_HANDEYE_YAML
    from sim2real import DEFAULT_CALIBRATION_YAML, DEFAULT_ROBOT_IP

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
        num_cups = (
            int(args.num_cups)
            if args.num_cups is not None
            else int(task_cfg.get("num_cups", 2))
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

        if mode == "sim":
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
                    raise DemoError(
                        f"--cup-poses 参数数量必须是 3 的倍数且不少于 {num_cups} 个纸杯 "
                        f"(共需 {3 * num_cups} 个数值)"
                    )
                cup_poses = [
                    InitialCupPose(
                        bottom_xy_m=(float(vals[3 * i]), float(vals[3 * i + 1])),
                        yaw_deg=float(vals[3 * i + 2]),
                    )
                    for i in range(len(vals) // 3)
                ]

            with StackCupRobot(viewer=not headless) as robot:
                metrics = run_stack_cups(
                    robot,
                    num_cups=num_cups,
                    cup_poses=cup_poses,
                    target_xy=target_xy,
                )

            if report_path:
                _write_json_report(
                    report_path,
                    {
                        "mode": "sim",
                        "ok": True,
                        "success": True,
                        "num_cups": num_cups,
                        "target_xy_m": target_xy.tolist(),
                        "metrics": metrics,
                    },
                )
            return 0

        from sim2real import run_sim2real_pipeline

        if args.target_xy is not None:
            target_override: tuple[float, float] | None = (
                float(args.target_xy[0]),
                float(args.target_xy[1]),
            )
        elif "target_xy" in task_cfg and task_cfg["target_xy"] is not None:
            t_cfg = task_cfg["target_xy"]
            target_override = (float(t_cfg[0]), float(t_cfg[1]))
        else:
            target_override = None
        run_sim2real_pipeline(
            mode=mode,
            num_cups=num_cups,
            ip=ip,
            calibration_path=calibration_path,
            handeye_path=handeye_path,
            cup_poses_override=args.cup_poses,
            target_xy_override=target_override,
            headless=headless,
            confirm_real=bool(args.confirm_real),
            cycles=int(args.cycles),
            post_verify=post_verify,
            save_vis_path=save_vis,
            save_after_vis_path=save_after_vis,
            report_path=report_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"失败：{exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
