#!/usr/bin/env python3
"""Pick up a sideways paper cup and place it upright (MuJoCo simulation & Sim-to-Real)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import yaml

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

if TYPE_CHECKING:
    from perception import CupPoseEstimate
    from sim2real import TwinPhaseWaypoints

SCENE = Path(__file__).with_name("scene.xml")
DEFAULT_CONFIG_YAML = Path(__file__).with_name("config.yaml")
CUP = "paper_cup"
TARGET_XY = np.array([0.30, 0.15])
GRASP_AXIS_OFFSET_M = 0.040
GRASP_TCP_Z_M = 0.014
INITIAL_GRASP_GAP_MM = 53.0  # Cup outer diameter at 40mm height is 63.0mm
MAX_GRASP_SHIFT_M = 0.025
MAX_GRASP_ROTATION_DEG = 25.0
MAX_CONTACT_LOSS_S = 0.10
FORCE_TARGET_N = 2.0
FORCE_UPDATE_S = 0.05
MODES = ("sim", "dry-run", "sdk-sim", "real")


class DemoError(RuntimeError):
    """A task, contact, or viewer condition prevented a successful run."""


def _cup_pose(robot: SimRobot) -> tuple[np.ndarray, np.ndarray]:
    body = robot.data.body(CUP)
    return body.xpos.copy(), body.xmat.reshape(3, 3)[:, 2].copy()


def _tool_relative_cup(robot: SimRobot) -> tuple[np.ndarray, np.ndarray]:
    bottom, axis = _cup_pose(robot)
    position = robot.data.site_xpos[robot._site]
    rotation = robot.data.site_xmat[robot._site].reshape(3, 3)
    return rotation.T @ (bottom - position), rotation.T @ axis


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _contact_between(robot: SimRobot, geom_a: str, geom_b: str) -> bool:
    a, b = robot.model.geom(geom_a).id, robot.model.geom(geom_b).id
    return any(
        {int(c.geom1), int(c.geom2)} == {a, b} and c.dist <= 0.0
        for c in robot.data.contact[:robot.data.ncon]
    )


def _cup_touches_table(robot: SimRobot) -> bool:
    table = robot.model.geom("table").id
    cup_body = robot.model.body(CUP).id
    for contact in robot.data.contact[:robot.data.ncon]:
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


def _configure_initial_scene(
    robot: SimRobot,
    cup_pose: CupPoseEstimate | None,
    target_xy: np.ndarray,
    q_init: np.ndarray | None,
) -> float:
    """Apply optional initial arm joint state, cup pose, and target disk position."""
    if q_init is not None:
        q_arr = np.asarray(q_init, dtype=np.float64).reshape(robot.n_arm)
        robot.data.qpos[:robot.n_arm] = q_arr
        robot.data.ctrl[:robot.n_arm] = q_arr
        robot.eval_data.qpos[:robot.n_arm] = q_arr

    table_z = 0.0
    if cup_pose is not None:
        table_z = float(cup_pose.table_z_m)
        jnt_id = robot.model.joint("paper_cup_joint").id
        qpos_adr = int(robot.model.jnt_qposadr[jnt_id])
        dof_adr = int(robot.model.jnt_dofadr[jnt_id])
        robot.data.qpos[qpos_adr:qpos_adr + 3] = cup_pose.mujoco_pos_m
        robot.data.qpos[qpos_adr + 3:qpos_adr + 7] = cup_pose.mujoco_quat_wxyz
        robot.data.qvel[dof_adr:dof_adr + 6] = 0.0
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
    return table_z


@dataclass
class GraspMonitor:
    """Detect loss of either contact and excessive cup motion relative to the TCP."""

    position: np.ndarray
    axis: np.ndarray
    lost_s: float = 0.0
    max_lost_s: float = 0.0
    max_shift_m: float = 0.0
    max_rotation_deg: float = 0.0

    @classmethod
    def start(cls, robot: SimRobot) -> GraspMonitor:
        position, axis = _tool_relative_cup(robot)
        return cls(position, axis)

    def observe(self, robot: SimRobot) -> None:
        position, axis = _tool_relative_cup(robot)
        shift = float(np.linalg.norm(position - self.position))
        angle = _angle_deg(axis, self.axis)
        self.max_shift_m = max(self.max_shift_m, shift)
        self.max_rotation_deg = max(self.max_rotation_deg, angle)
        self.lost_s = 0.0 if robot.bilateral_finger_contact(CUP) else (
            self.lost_s + robot.model.opt.timestep
        )
        self.max_lost_s = max(self.max_lost_s, self.lost_s)
        if not np.isfinite(shift) or not np.isfinite(angle):
            raise DemoError("纸杯位姿无效")
        if self.lost_s > MAX_CONTACT_LOSS_S:
            raise DemoError(f"夹持中滑脱：双侧接触中断 {self.lost_s:.3f} s")
        if shift > MAX_GRASP_SHIFT_M or angle > MAX_GRASP_ROTATION_DEG:
            raise DemoError(f"夹持中滑脱：相对位移 {shift * 1000:.1f} mm，转角 {angle:.1f}°")


class CupRobot(SimRobot):
    """Use the existing G2 servo, with a small contact-force feedback loop."""

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
        if calibration is not None:
            apply_calibration_to_model(self.model, calibration, self.data)
            mujoco.mj_setConst(self.model, self.eval_data)
            mujoco.mj_forward(self.model, self.eval_data)

    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        if self.active_waypoint_sink is not None:
            self.active_waypoint_sink.append(
                np.asarray(q_goal[:self.n_arm], dtype=np.float64).copy()
            )
        super().move_joints(q_goal, duration, wait=wait)

    def check_viewer(self) -> None:
        if self._viewer is not None and not self._viewer.is_running():
            raise DemoError("可视化窗口提前关闭")

    def _step(self) -> None:
        self.check_viewer()
        if self.force_feedback and self.data.time - self.last_force_update >= FORCE_UPDATE_S:
            self.last_force_update = float(self.data.time)
            force = min(self.get_gripper_contact_forces(CUP))
            target = float(self.g2_controller.position_mm[0])
            # 20 Hz outer loop; the G2 controller remains responsible for
            # position, speed and torque limiting at every simulation step.
            adjustment_mm = float(np.clip((force - FORCE_TARGET_N) * 0.04, -0.2, 0.2))
            self.set_gripper_command(
                np.clip(target + adjustment_mm, 0.0, 84.0),
                speed_percent=20, force_percent=10,
            )
        super()._step()
        self.check_viewer()
        if self.grasp_monitor is not None:
            self.grasp_monitor.observe(self)


def _move_tcp(robot: CupRobot, pos: np.ndarray, quat: np.ndarray, duration: float,
              *, steps: int) -> None:
    robot.check_viewer()
    try:
        robot.move_tcp(pos, quat, duration, steps=steps)
    except RuntimeError as exc:
        if isinstance(exc, DemoError):
            raise
        raise DemoError(f"逆运动学或轨迹执行失败：{exc}") from exc
    robot.check_viewer()


def _rotate_cup_up(axis: np.ndarray, tool_quat: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    cross = np.cross(axis, up)
    length = float(np.linalg.norm(cross))
    if length < 1e-9:
        correction = np.array([1.0, 0.0, 0.0, 0.0]) if axis[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        angle = float(np.arctan2(length, np.dot(axis, up)))
        correction = np.r_[np.cos(angle / 2), cross / length * np.sin(angle / 2)]
    result = np.empty(4)
    mujoco.mju_mulQuat(result, correction, tool_quat)
    return result


def _require_bilateral(robot: CupRobot) -> None:
    if not robot.bilateral_finger_contact(CUP):
        raise DemoError("无双侧指垫接触，拒绝抬起")


def _verify_rest(
    robot: CupRobot,
    duration: float = 1.0,
    *,
    target_xy: np.ndarray = TARGET_XY,
) -> dict[str, float]:
    """Require one continuous second on the cup bottom without finger support."""
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
            raise DemoError("纸杯最终位姿无效")
        tilt = _angle_deg(axis, np.array([0.0, 0.0, 1.0]))
        error = float(np.linalg.norm(bottom[:2] - target_xy))
        drift = float(np.linalg.norm(bottom - start))
        max_tilt, max_error, max_drift = max(max_tilt, tilt), max(max_error, error), max(max_drift, drift)
        if not _contact_between(robot, "table", "cup_bottom"):
            raise DemoError("稳定性检查失败：杯底未持续接触桌面")
        if bodies_in_contact(robot.model, robot.data, "left_finger", CUP) or bodies_in_contact(
            robot.model, robot.data, "right_finger", CUP
        ):
            raise DemoError("稳定性检查失败：夹爪仍接触纸杯")
        if tilt > 10.0 or error > 0.030 or drift > 0.002:
            raise DemoError(f"稳定性检查失败：倾角 {tilt:.1f}°，落点误差 {error * 1000:.1f} mm，"
                            f"漂移 {drift * 1000:.1f} mm")
    return {"tilt_deg": max_tilt, "error_mm": max_error * 1000,
            "drift_mm": max_drift * 1000, "stable_s": float(robot.data.time - start_time)}


def run_pick_cup(
    robot: CupRobot,
    *,
    cup_pose: CupPoseEstimate | None = None,
    target_xy: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    recorded_phases: TwinPhaseWaypoints | None = None,
) -> dict[str, float]:
    """Run one pick-flip-place task. Object motion is exclusively produced by contacts."""
    active_target_xy = TARGET_XY if target_xy is None else np.asarray(target_xy, dtype=np.float64)
    table_z = _configure_initial_scene(robot, cup_pose, active_target_xy, q_init)

    print("准备：张开夹爪并等待纸杯稳定", flush=True)
    robot.open_gripper(speed_percent=40, force_percent=10)
    robot.settle(1.0)
    robot.check_viewer()
    initial_bottom, axis = _cup_pose(robot)
    grasp_center = initial_bottom + GRASP_AXIS_OFFSET_M * axis
    grasp = np.array([grasp_center[0], grasp_center[1], table_z + GRASP_TCP_Z_M])
    approach = grasp.copy()
    approach[2] = table_z + 0.16

    axis_yaw_rad = float(np.arctan2(axis[1], axis[0]))
    grasp_quat = _yaw_rotate_quat(axis_yaw_rad, GRIP_DOWN_QUAT)

    print("接近杯身", flush=True)
    if recorded_phases is not None:
        robot.active_waypoint_sink = recorded_phases.approach_waypoints
    q, ok = robot.solve_tcp(approach, grasp_quat, q_init=robot.data.qpos.copy())
    robot.check_viewer()
    if not ok:
        raise DemoError(f"逆运动学不收敛：接近位置 {approach}")
    robot.move_joints(q[:robot.n_arm], 3.0)
    robot.check_viewer()
    _move_tcp(robot, grasp, grasp_quat, 2.0, steps=30)
    robot.active_waypoint_sink = None

    print("夹持：确认双侧接触", flush=True)
    robot.set_gripper_command(INITIAL_GRASP_GAP_MM, speed_percent=20, force_percent=10)
    robot.settle(2.0)
    robot.check_viewer()
    # Back off the initial force-limited closure before enabling force feedback.
    gap = max(0.0, robot.get_measured_gripper_gap() * 1000 - 1.0)
    robot.set_gripper_command(gap, speed_percent=20, force_percent=10)
    robot.settle(1.0)
    robot.check_viewer()
    _require_bilateral(robot)
    monitor = GraspMonitor.start(robot)
    robot.grasp_monitor = monitor
    robot.force_feedback = True
    robot.last_force_update = -1.0

    print("抬起并翻转", flush=True)
    if recorded_phases is not None:
        robot.active_waypoint_sink = recorded_phases.lift_flip_place_waypoints
    _move_tcp(robot, grasp + np.array([0.0, 0.0, 0.13]), grasp_quat, 2.0, steps=30)
    lifted_bottom, _ = _cup_pose(robot)
    if lifted_bottom[2] < initial_bottom[2] + 0.05 or _cup_touches_table(robot):
        raise DemoError("抬起失败：纸杯未完全离开桌面")

    position, quat = robot.get_tcp_pose()
    if cup_pose is None and np.allclose(active_target_xy, TARGET_XY):
        _move_tcp(robot, position + np.array([0.12, 0.0, 0.08]), quat, 2.5, steps=40)
        position, quat = robot.get_tcp_pose()
        _, axis = _cup_pose(robot)
        _move_tcp(robot, position, _rotate_cup_up(axis, quat), 3.0, steps=50)
        # Turn the gripper around the vertical axis for reachability and a clear retreat.
        position, quat = robot.get_tcp_pose()
        yaw = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        yawed = np.empty(4)
        mujoco.mju_mulQuat(yawed, yaw, quat)
        _move_tcp(robot, position, yawed, 2.5, steps=50)
        retreat_offset = np.array([0.0, -0.12, 0.0])
    else:
        # Align cup horizontal axis along -Y (for right-side target) or +Y (for left-side target)
        # while the gripper is still vertical, so flipping up keeps flange X ~ 0.40 m from base.
        place_side = -1.0 if active_target_xy[1] <= 0.0 else 1.0
        pre_flip_quat = _yaw_rotate_quat(place_side * 0.5 * np.pi, GRIP_DOWN_QUAT)
        mid_pos = np.array([0.40, -0.06 * (-place_side), table_z + 0.22])
        _move_tcp(robot, mid_pos, pre_flip_quat, 2.5, steps=40)
        position, quat = robot.get_tcp_pose()
        _, axis = _cup_pose(robot)
        _move_tcp(robot, position, _rotate_cup_up(axis, quat), 3.0, steps=50)
        retreat_offset = np.array([0.0, -0.10 * place_side, 0.0])

    print("搬运并根据纸杯实时位置放置", flush=True)
    position, quat = robot.get_tcp_pose()
    bottom, _ = _cup_pose(robot)
    destination = position + np.r_[active_target_xy - bottom[:2], 0.0]
    _move_tcp(robot, destination, quat, 4.0, steps=50)
    position, quat = robot.get_tcp_pose()
    bottom, _ = _cup_pose(robot)
    destination = position + np.r_[active_target_xy - bottom[:2], table_z + 0.002 - bottom[2]]
    _move_tcp(robot, destination, quat, 3.0, steps=50)
    if recorded_phases is not None:
        recorded_phases.grasp_gap_mm = float(robot.g2_controller.position_mm[0])
        robot.active_waypoint_sink = None

    if not _contact_between(robot, "table", "cup_bottom"):
        target_z_touch = 0.0001 if (cup_pose is None and np.allclose(active_target_xy, TARGET_XY)) else -0.0004
        position, quat = robot.get_tcp_pose()
        bottom, _ = _cup_pose(robot)
        destination = position + np.r_[active_target_xy - bottom[:2], table_z + target_z_touch - bottom[2]]
        _move_tcp(robot, destination, quat, 1.2, steps=20)
    if not _contact_between(robot, "table", "cup_bottom"):
        raise DemoError("放置失败：杯底未接触桌面")
    _require_bilateral(robot)
    robot.force_feedback = False
    robot.grasp_monitor = None

    print("释放并撤离", flush=True)
    robot.open_gripper(speed_percent=15, force_percent=40)
    robot.settle(1.0)
    robot.check_viewer()
    if recorded_phases is not None:
        robot.active_waypoint_sink = recorded_phases.retreat_waypoints
    position, quat = robot.get_tcp_pose()
    retreat_pos = position + retreat_offset
    _move_tcp(robot, retreat_pos, quat, 2.0, steps=30)
    if recorded_phases is not None:
        _move_tcp(robot, retreat_pos + np.array([0.0, 0.0, 0.10]), quat, 1.5, steps=20)
        robot.active_waypoint_sink = None
    robot.settle(0.5)
    robot.check_viewer()
    metrics = _verify_rest(robot, target_xy=active_target_xy)
    metrics.update(lift_mm=float((lifted_bottom[2] - initial_bottom[2]) * 1000),
                   max_grasp_shift_mm=monitor.max_shift_m * 1000,
                   max_grasp_rotation_deg=monitor.max_rotation_deg,
                   max_contact_loss_s=monitor.max_lost_s)
    print(f"成功：杯底误差 {metrics['error_mm']:.2f} mm，最大倾角 {metrics['tilt_deg']:.2f}°，"
          f"稳定 {metrics['stable_s']:.3f} s；夹持相对位移最大 {metrics['max_grasp_shift_mm']:.1f} mm。",
          flush=True)
    return metrics


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration dictionary from YAML file (falls back to empty dict if missing)."""
    path = DEFAULT_CONFIG_YAML if config_path is None else Path(config_path)
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_YAML),
        help="配置文件 YAML 路径 (默认: examples/pick_cup/config.yaml)",
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
        help="是否以无头模式运行 MuJoCo (默认读取 config.yaml 中的 headless，未指定时为 false 即开启可视化)",
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
        "--cup-pose",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "YAW_DEG"),
        help="手动指定侧卧纸杯杯底坐标 (x_m, y_m) 与水平朝向角 yaw_deg",
    )
    parser.add_argument(
        "--target-xy",
        nargs=2,
        type=float,
        metavar=("X_M", "Y_M"),
        help="指定竖直放置目标点水平坐标 (x_m, y_m)",
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
        help="真机放置回到观测位后是否自动拍照闭环确认 (默认读取 config.yaml)",
    )
    parser.add_argument("--save-vis", default=None, help="保存抓取前相机纸杯检测叠加图像路径 (PNG)")
    parser.add_argument("--save-after-vis", default=None, help="保存放置后相机闭环确认叠加图像路径 (PNG)")
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

        cup_override = tuple(args.cup_pose) if args.cup_pose is not None else None
        target_override = tuple(args.target_xy) if args.target_xy is not None else None

        if mode == "sim":
            cup_est = None
            if cup_override is not None:
                from perception import pose_estimate_from_manual
                cup_est = pose_estimate_from_manual(
                    cup_override[0], cup_override[1], cup_override[2]
                )
            target_arr = (
                np.asarray(target_override, dtype=np.float64)
                if target_override is not None
                else None
            )
            with CupRobot(viewer=not headless) as robot:
                metrics = run_pick_cup(robot, cup_pose=cup_est, target_xy=target_arr)
            if report_path:
                from sim2real import _write_json_report
                _write_json_report(
                    report_path,
                    {"mode": "sim", "ok": True, "success": True, "metrics": metrics},
                )
            return 0

        from sim2real import run_sim2real_pipeline

        run_sim2real_pipeline(
            mode=mode,
            ip=ip,
            calibration_path=calibration_path,
            handeye_path=handeye_path,
            cup_pose_override=cup_override,
            target_xy_override=target_override,
            headless=headless,
            confirm_real=bool(args.confirm_real),
            cycles=int(args.cycles),
            post_verify=post_verify,
            save_vis_path=save_vis,
            save_after_vis_path=save_after_vis,
            report_path=report_path,
        )
    except Exception as exc:  # noqa: BLE001 - the CLI must report any failure and return nonzero
        print(f"失败：{exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
