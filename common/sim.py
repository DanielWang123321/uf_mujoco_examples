# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""MuJoCo task-robot base class and contact helpers shared by all simulation examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from ufactory_mujoco.calibration import KinematicsCalibration, apply_calibration_to_model
from ufactory_mujoco.ik import solve_ik_trajectory
from ufactory_mujoco.sim_robot import SimRobot

from .motion import smooth_step_durations


class TaskError(RuntimeError):
    """A task, contact, or viewer condition prevented a successful run."""


@dataclass(frozen=True)
class ForceFeedbackGains:
    """Outer-loop gains that nudge the G2 position target toward a desired contact force."""

    target_n: float
    gain_mm_per_n: float
    max_step_mm: float
    speed_percent: float
    force_percent: float
    period_s: float = 0.05


class TaskSimRobot(SimRobot):
    """``SimRobot`` extended with the features every example needs.

    - optional per-robot kinematic calibration (digital twin of the real arm),
    - joint waypoint recording (``active_waypoint_sink``) for Sim-to-Real replay,
    - viewer-close abort (``check_viewer``),
    - optional G2 contact-force feedback loop (``force_feedback`` + ``force_body``),
    - optional per-step grasp monitor (any object with ``observe(robot)``).

    Subclasses customize the model in ``_configure_model`` and add per-step logic in
    ``_before_step``.
    """

    #: Force-feedback gains; subclasses override to enable a different contact force.
    force_gains = ForceFeedbackGains(
        target_n=2.0, gain_mm_per_n=0.04, max_step_mm=0.2, speed_percent=20, force_percent=10
    )

    def __init__(
        self,
        scene: str | Path,
        *,
        viewer: bool,
        calibration: KinematicsCalibration | None = None,
    ) -> None:
        # Attributes used by ``_step`` must exist before ``SimRobot.__init__`` runs.
        self.grasp_monitor = None
        self.active_waypoint_sink: list[np.ndarray] | None = None
        self.force_feedback = False
        self.force_body: str | None = None
        self.last_force_update = -1.0
        super().__init__(scene, viewer=viewer)
        self._configure_model()
        if calibration is not None:
            apply_calibration_to_model(self.model, calibration, self.data)
            mujoco.mj_setConst(self.model, self.eval_data)
            mujoco.mj_forward(self.model, self.eval_data)

    def _configure_model(self) -> None:
        """Hook for model edits (friction, priorities, ...) before calibration is applied."""

    def _before_step(self) -> None:
        """Hook executed before every physics step (after the viewer check)."""

    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        if self.active_waypoint_sink is not None:
            self.active_waypoint_sink.append(
                np.asarray(q_goal[: self.n_arm], dtype=np.float64).copy()
            )
        super().move_joints(q_goal, duration, wait=wait)

    def move_tcp(
        self,
        pos: np.ndarray,
        quat: np.ndarray | None,
        duration: float,
        wait: bool = True,
        *,
        steps: int = 50,
        soft_landing: bool = False,
    ) -> None:
        """Move along a straight Cartesian path, optionally with soft-landing deceleration."""
        if not soft_landing:
            super().move_tcp(pos, quat, duration, wait=wait, steps=steps)
            return
        pos0, quat0 = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (np.asarray(pos, dtype=np.float64) - pos0)
        quat_traj = None
        if quat is not None:
            quat_goal = np.asarray(quat, dtype=np.float64)
            q0_conj = np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]], dtype=np.float64)
            # The integrator right-multiplies a tool-frame rotation increment.
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
            self.model, self.eval_data, self._tcp_site_name(), waypoints, quat_traj, q_start
        )
        if not ok:
            raise RuntimeError(f"IK did not converge for target pos={pos} quat={quat}")
        dts = smooth_step_durations(
            duration,
            steps,
            soft_landing=True,
            min_dt=4 * self.model.opt.timestep,
        )
        for q, seg in zip(q_traj, dts, strict=True):
            self.move_joints(q[: self.n_arm], float(seg), wait=False)
        if wait:
            self.settle(0.3)

    def check_viewer(self) -> None:
        if self._viewer is not None and not self._viewer.is_running():
            raise TaskError("Viewer window was closed")

    def _update_force_feedback(self) -> None:
        g = self.force_gains
        if (
            not self.force_feedback
            or self.force_body is None
            or self.data.time - self.last_force_update < g.period_s
        ):
            return
        self.last_force_update = float(self.data.time)
        force = min(self.get_gripper_contact_forces(self.force_body))
        target = float(self.g2_controller.position_mm[0])
        # Slow outer loop; the G2 controller still limits position, speed and torque every step.
        adjustment_mm = float(
            np.clip((force - g.target_n) * g.gain_mm_per_n, -g.max_step_mm, g.max_step_mm)
        )
        self.set_gripper_command(
            np.clip(target + adjustment_mm, 0.0, 84.0),
            speed_percent=g.speed_percent,
            force_percent=g.force_percent,
        )

    def _step(self) -> None:
        self.check_viewer()
        self._update_force_feedback()
        self._before_step()
        super()._step()
        self.check_viewer()
        if self.grasp_monitor is not None:
            self.grasp_monitor.observe(self)

    def set_arm_qpos(self, q: np.ndarray) -> None:
        """Teleport the arm joints (data, ctrl and IK data) to ``q`` without stepping physics."""
        q_arr = np.asarray(q, dtype=np.float64).reshape(self.n_arm)
        self.data.qpos[: self.n_arm] = q_arr
        self.data.ctrl[: self.n_arm] = q_arr
        self.eval_data.qpos[: self.n_arm] = q_arr

    def set_free_joint_pose(self, joint: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Place a free-joint body at ``pos`` / ``quat_wxyz`` with zero velocity."""
        jnt_id = self.model.joint(joint).id
        qpos_adr = int(self.model.jnt_qposadr[jnt_id])
        dof_adr = int(self.model.jnt_dofadr[jnt_id])
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0

    def body_pose_axis(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (world position, world local-Z axis) of ``body``."""
        b = self.data.body(body)
        return b.xpos.copy(), b.xmat.reshape(3, 3)[:, 2].copy()

    def tool_relative_pose_axis(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``body`` position and local-Z axis expressed in the TCP frame."""
        pos, axis = self.body_pose_axis(body)
        tcp_pos = self.data.site_xpos[self._site]
        tcp_rot = self.data.site_xmat[self._site].reshape(3, 3)
        return tcp_rot.T @ (pos - tcp_pos), tcp_rot.T @ axis


def configure_finger_contact_priority(model: mujoco.MjModel) -> None:
    """Give G2 finger collision geoms priority 2 and high friction.

    With MuJoCo contact priorities the finger parameters win against the grasped object, so
    object-to-object contacts (cup-cup, nut-screw) can keep their own low friction.
    """
    for finger_body in ("left_finger", "right_finger"):
        b_id = model.body(finger_body).id
        for g_id in range(model.ngeom):
            if int(model.geom_bodyid[g_id]) == b_id and int(model.geom_group[g_id]) == 3:
                model.geom_priority[g_id] = 2
                model.geom_condim[g_id] = 4
                model.geom_friction[g_id] = [4.0, 0.015, 0.002]
                model.geom_solref[g_id] = [0.003, 1.0]
                model.geom_solimp[g_id] = [0.99, 0.999, 0.0001, 0.5, 2.0]


def contact_between(model: mujoco.MjModel, data: mujoco.MjData, geom_a: str, geom_b: str) -> bool:
    """True if the two named geoms are in penetrating contact."""
    a, b = model.geom(geom_a).id, model.geom(geom_b).id
    return any(
        {int(c.geom1), int(c.geom2)} == {a, b} and c.dist <= 0.0 for c in data.contact[: data.ncon]
    )


def body_touches_geom(model: mujoco.MjModel, data: mujoco.MjData, body: str, geom: str) -> bool:
    """True if any geom of ``body`` is in penetrating contact with geom ``geom``."""
    geom_id = model.geom(geom).id
    body_id = model.body(body).id
    for contact in data.contact[: data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if contact.dist <= 0.0 and (
            (a == geom_id and model.geom_bodyid[b] == body_id)
            or (b == geom_id and model.geom_bodyid[a] == body_id)
        ):
            return True
    return False


def move_tcp_checked(
    robot: TaskSimRobot,
    pos: np.ndarray,
    quat: np.ndarray,
    duration: float,
    *,
    steps: int,
    wait: bool = True,
    soft_landing: bool = False,
) -> None:
    """``robot.move_tcp`` that converts IK / trajectory failures into ``TaskError``."""
    robot.check_viewer()
    try:
        robot.move_tcp(pos, quat, duration, wait=wait, steps=steps, soft_landing=soft_landing)
    except RuntimeError as exc:
        if isinstance(exc, TaskError):
            raise
        raise TaskError(f"IK or trajectory execution failed: {exc}") from exc
    robot.check_viewer()
