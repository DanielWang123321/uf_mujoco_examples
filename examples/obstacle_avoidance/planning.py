# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Deterministic endpoint IK, geometric planning, and checked time parameterization."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.ik import solve_ik
from ufactory_mujoco.path_planning import (
    PlanningError,
    path_length,
    plan_collision_free,
    shortcut_path,
)
from ufactory_mujoco.trajectory import JointTrajectory, plan_joint_waypoints

SCENE = Path(__file__).with_name("scene.xml")
SITE = "attachment_site"
MOUNT_PAIRS = (("world", "link_base"), ("world", "link1"))
START_POSE = (350.0, -200.0, 300.0, 180.0, 0.0, 0.0)
GOAL_POSE = (350.0, 200.0, 300.0, 180.0, 0.0, 0.0)
DT = 0.001
PAYLOAD = "payload"
GRIPPER_QPOS = slice(6, 12)
GRIPPER_MECHANISM_CLEARANCE_M = 0.005


@dataclass(frozen=True)
class CarriedState:
    """Measured object pose relative to TCP, used only in the private planning model."""

    gripper_qpos: np.ndarray
    position: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_data(cls, data: mujoco.MjData) -> CarriedState:
        tool = data.site(SITE)
        tool_rotation = tool.xmat.reshape(3, 3)
        body = data.body(PAYLOAD)
        return cls(
            data.qpos[GRIPPER_QPOS].copy(),
            tool_rotation.T @ (body.xpos - tool.xpos),
            tool_rotation.T @ body.xmat.reshape(3, 3),
        )


class MotionError(RuntimeError):
    """A failure with a reportable pipeline stage."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class Options:
    start_pose: tuple[float, ...] = START_POSE
    goal_pose: tuple[float, ...] = GOAL_POSE
    seed: int = 0
    step_rad: float = 0.15
    edge_resolution_rad: float = 0.02
    max_iterations: int = 2000
    shortcut_iterations: int = 200
    clearance_m: float = 0.01
    execution_clearance_m: float = 0.005
    max_velocity_rad_s: float = 0.65
    max_acceleration_rad_s2: float = 2.0
    timestep_s: float = DT

    def validate(self) -> None:
        pose_to_target(self.start_pose)
        pose_to_target(self.goal_pose)
        for key in ("seed", "max_iterations", "shortcut_iterations"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if self.max_iterations == 0:
            raise ValueError("max_iterations must be positive")
        for key in (
            "step_rad",
            "edge_resolution_rad",
            "clearance_m",
            "execution_clearance_m",
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
            "timestep_s",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        if self.edge_resolution_rad > 0.02:
            raise ValueError("edge_resolution_rad must be <= 0.02 rad")
        if self.clearance_m < self.execution_clearance_m:
            raise ValueError("planning clearance must be >= execution clearance")
        if not math.isclose(self.timestep_s, DT, abs_tol=1e-12, rel_tol=0):
            raise ValueError("timestep_s must be 0.001 s (the scene physics timestep)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pose_to_target(pose) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame mm / extrinsic XYZ degrees -> meters / MuJoCo wxyz quaternion."""
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("TCP pose must contain six finite values: x y z roll pitch yaw")
    quat_xyzw = Rotation.from_euler("xyz", values[3:], degrees=True).as_quat()
    return values[:3] / 1000.0, quat_xyzw[[3, 0, 1, 2]]


def tcp_error(model, data, pose) -> tuple[float, float]:
    """TCP position norm (mm) and shortest orientation error (degrees)."""
    pos, quat = pose_to_target(pose)
    site = data.site(SITE)
    actual = Rotation.from_matrix(site.xmat.reshape(3, 3))
    target = Rotation.from_quat(quat[[1, 2, 3, 0]])
    return (
        float(np.linalg.norm(site.xpos - pos) * 1000.0),
        float(np.rad2deg((target * actual.inv()).magnitude())),
    )


class SceneChecker(CollisionChecker):
    """Reference checker with measured G2/payload geometry and full-state runtime queries.

    Only the two base/table mounting pairs are ignored. Obstacles live on their
    own bodies. All other model collision filters and adjacent-link excludes apply.
    Distances are capped at query_margin, a conservative lower bound outside it.
    """

    def __init__(self, scene=SCENE, *, clearance_m=0.01, carried: CarriedState | None = None):
        super().__init__(
            scene,
            safety_margin=clearance_m,
            query_margin=max(0.05, clearance_m * 2),
            ignore_pairs=MOUNT_PAIRS,
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        for joint in range(self.model.njnt):
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[joint]
                self.data.qpos[adr : adr + 7] = self.model.qpos0[adr : adr + 7]
        self.home = self.data.qpos.copy()
        self.open_gripper = self.home[GRIPPER_QPOS].copy()
        self.carried = carried
        self.allow_table = carried is None
        self.allow_fingers = carried is not None
        self._actual = False
        self.payload_id = self.model.body(PAYLOAD).id
        self.payload_adr = int(self.model.jnt_qposadr[self.model.joint("payload_joint").id])
        self.finger_ids = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        self.gripper_body_ids = {
            self.model.body(name).id
            for name in (
                "xarm_gripper_base_link",
                "left_outer_knuckle",
                "left_inner_knuckle",
                "left_finger",
                "right_outer_knuckle",
                "right_inner_knuckle",
                "right_finger",
            )
        }
        self.last_gripper_distance = self.query_margin
        self.checked_gripper_min = self.query_margin

    def _set_q(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_arm_qpos,) or not np.all(np.isfinite(q)):
            raise ValueError("arm configuration must contain six finite joint positions")
        if not self._actual:
            self.data.qpos[:] = self.home
            if self.carried is not None:
                self.data.qpos[GRIPPER_QPOS] = self.carried.gripper_qpos
            self.data.qpos[: self.n_arm_qpos] = q
            if self.carried is not None:
                mujoco.mj_kinematics(self.model, self.data)
                tool = self.data.site(SITE)
                rotation = tool.xmat.reshape(3, 3)
                adr = self.payload_adr
                self.data.qpos[adr : adr + 3] = tool.xpos + rotation @ self.carried.position
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, (rotation @ self.carried.rotation).ravel())
                self.data.qpos[adr + 3 : adr + 7] = quat
        self.data.qpos[: self.n_arm_qpos] = q
        # Collision queries need poses and contact distances, not a constraint-force solve.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def _contacts(self, q):
        contacts = super()._contacts(q)
        result = []
        for first, second, distance in contacts:
            if self.payload_id in (first, second):
                other = second if first == self.payload_id else first
                if (self.allow_table and other == 0 and distance >= -0.001) or (
                    self.allow_fingers and other in self.finger_ids and distance >= -0.002
                ):
                    continue
            result.append((first, second, distance))
        return result

    def min_distance(self, q: np.ndarray) -> float:
        """Environment/arm/payload clearance; measure G2 mechanism clearance separately."""
        environment, mechanism = self.query_margin, self.query_margin
        for first, second, distance in self._contacts(q):
            if first in self.gripper_body_ids and second in self.gripper_body_ids:
                mechanism = min(mechanism, distance)
            else:
                environment = min(environment, distance)
        self.last_gripper_distance = mechanism
        return environment

    def in_collision(self, q, *, safety_margin=None):
        margin = self.safety_margin if safety_margin is None else safety_margin
        distance = self.min_distance(q)
        return distance < margin or self.last_gripper_distance < min(
            margin, GRIPPER_MECHANISM_CLEARANCE_M
        )

    def actual_distance(self, full_qpos: np.ndarray) -> float:
        if full_qpos.shape != (self.model.nq,) or not np.all(np.isfinite(full_qpos)):
            raise MotionError("execution", "actual joint state is non-finite or has wrong shape")
        self.data.qpos[:] = full_qpos
        self._actual = True
        try:
            return self.min_distance(full_qpos[: self.n_arm_qpos])
        finally:
            self._actual = False


def _within_limits(q: np.ndarray, checker: CollisionChecker, *, tolerance=1e-9) -> bool:
    return bool(
        np.all(np.isfinite(q))
        and np.all(q >= checker.q_lo - tolerance)
        and np.all(q <= checker.q_hi + tolerance)
    )


def endpoint_candidates(checker: SceneChecker, pose, *, seed: int, name: str) -> list[np.ndarray]:
    """At most eight IK solves, retaining up to four distinct, verified arm solutions."""
    pos, quat = pose_to_target(pose)
    rng = np.random.default_rng(seed)
    model = checker.model
    data = mujoco.MjData(model)
    candidates: list[np.ndarray] = []
    converged_count = 0
    for attempt in range(8):
        initial = checker.home.copy()
        if attempt:
            initial[: checker.n_arm_qpos] = rng.uniform(checker.q_lo, checker.q_hi)
        q_full, converged = solve_ik(
            model,
            data,
            SITE,
            pos,
            quat,
            initial,
            pos_tol=5e-5,
            rot_tol=2e-4,
            max_iter=250,
        )
        if not converged or not np.all(np.isfinite(q_full)):
            continue
        converged_count += 1
        q = q_full[: checker.n_arm_qpos].copy()
        # Equivalent full turns are represented nearest home, within the authored limits.
        q -= 2 * np.pi * np.round((q - checker.home[: checker.n_arm_qpos]) / (2 * np.pi))
        q = np.where(q < checker.q_lo, q + 2 * np.pi, q)
        q = np.where(q > checker.q_hi, q - 2 * np.pi, q)
        if not _within_limits(q, checker):
            continue
        data.qpos[:] = checker.home
        data.qpos[: checker.n_arm_qpos] = q
        mujoco.mj_forward(model, data)
        pos_mm, rot_deg = tcp_error(model, data, pose)
        if pos_mm > 0.2 or rot_deg > 0.1 or checker.in_collision(q):
            continue
        if any(np.linalg.norm(q - previous) < 1e-3 for previous in candidates):
            continue
        candidates.append(q)
        if len(candidates) == 4:
            break
    if not candidates:
        reason = (
            "IK solutions failed FK, joint-limit or clearance checks"
            if converged_count
            else "IK did not converge (unreachable or initial-guess budget exhausted)"
        )
        raise MotionError(f"{name}_ik", f"{name}: {reason}; at most 8 IK attempts")
    return candidates


def sample_edges(path: np.ndarray, resolution: float):
    """Yield endpoints and linear edge samples, with Euclidean spacing <= resolution."""
    yield path[0]
    for a, b in zip(path[:-1], path[1:], strict=True):
        count = max(1, math.ceil(float(np.linalg.norm(b - a)) / resolution))
        for i in range(1, count + 1):
            yield a + (b - a) * (i / count)


def check_path(path, checker: SceneChecker, *, resolution: float, clearance: float) -> float:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != checker.n_arm_qpos:
        raise MotionError("path_validation", "path must have shape (N >= 2, n_arm)")
    if not _within_limits(values, checker):
        raise MotionError(
            "path_validation", "path violates joint limits or contains non-finite values"
        )
    minimum = checker.query_margin
    checker.checked_gripper_min = checker.query_margin
    for q in sample_edges(values, resolution):
        minimum = min(minimum, checker.min_distance(q))
        checker.checked_gripper_min = min(
            checker.checked_gripper_min, checker.last_gripper_distance
        )
        if minimum < clearance:
            raise MotionError(
                "path_validation", f"path clearance {minimum * 1000:.3f} mm is too small"
            )
        if checker.last_gripper_distance < min(clearance, GRIPPER_MECHANISM_CLEARANCE_M):
            raise MotionError("path_validation", "G2 internal mechanism clearance is too small")
    return minimum


def check_trajectory(
    trajectory: JointTrajectory, path: np.ndarray, checker: SceneChecker, options: Options
) -> float:
    """Check every 1 ms command, derivatives, endpoints, and intervening joint-space edges."""
    tr = trajectory
    if (
        tr.n_steps < 2
        or tr.q.shape != tr.qd.shape
        or tr.q.shape != tr.qdd.shape
        or not all(np.all(np.isfinite(a)) for a in (tr.times, tr.q, tr.qd, tr.qdd))
        or not np.allclose(np.diff(tr.times), options.timestep_s, atol=1e-10, rtol=0)
        or abs(tr.times[0]) > 1e-10
    ):
        raise MotionError("trajectory_validation", "invalid trajectory samples or timestep")
    if not (
        np.allclose(tr.q[[0, -1]], path[[0, -1]], atol=1e-8, rtol=0)
        and np.max(np.abs(tr.qd[[0, -1]])) < 1e-8
    ):
        raise MotionError(
            "trajectory_validation", "trajectory changed endpoints or endpoint velocity"
        )
    if not _within_limits(tr.q, checker):
        raise MotionError("trajectory_validation", "trajectory violates joint limits")
    if np.max(np.abs(tr.qd)) > options.max_velocity_rad_s + 1e-6:
        raise MotionError("trajectory_validation", "trajectory exceeds velocity limit")
    if np.max(np.abs(tr.qdd)) > options.max_acceleration_rad_s2 + 1e-6:
        raise MotionError("trajectory_validation", "trajectory exceeds acceleration limit")
    # Also check derivatives of the actual sampled commands, including segment joins.
    velocity = np.diff(tr.q, axis=0) / options.timestep_s
    if np.max(np.abs(velocity)) > options.max_velocity_rad_s + 1e-6:
        raise MotionError("trajectory_validation", "sampled commands exceed velocity limit")
    if len(velocity) > 1:
        acceleration = np.diff(velocity, axis=0) / options.timestep_s
        if np.max(np.abs(acceleration)) > options.max_acceleration_rad_s2 + 1e-5:
            raise MotionError("trajectory_validation", "sampled commands exceed acceleration limit")
    try:
        return check_path(tr.q, checker, resolution=0.01, clearance=options.clearance_m)
    except MotionError as exc:
        raise MotionError("trajectory_validation", str(exc)) from exc


def time_parameterize(path, checker, options) -> tuple[JointTrajectory, str, float, str | None]:
    """Prefer the reference cubic spline; fall back to quintics on the checked polyline."""
    rejection = None
    for method in ("spline", "stop"):
        try:
            trajectory = plan_joint_waypoints(
                path,
                max_vel=options.max_velocity_rad_s,
                max_acc=options.max_acceleration_rad_s2,
                dt=options.timestep_s,
                method=method,
            )
            # Polynomial evaluation can differ from a waypoint by one floating-point ULP.
            if np.allclose(trajectory.q[[0, -1]], path[[0, -1]], atol=1e-12, rtol=0):
                trajectory.q[[0, -1]] = path[[0, -1]]
            minimum = check_trajectory(trajectory, path, checker, options)
            return (
                trajectory,
                "spline" if method == "spline" else "quintic_polyline",
                minimum,
                rejection,
            )
        except (MotionError, ValueError) as exc:
            if method == "stop":
                raise MotionError(
                    "trajectory_validation",
                    f"spline rejected: {rejection}; quintic rejected: {exc}",
                ) from exc
            rejection = str(exc)
    raise AssertionError("unreachable")


@dataclass
class MotionPlan:
    path: np.ndarray
    raw_path: np.ndarray
    baseline: np.ndarray
    trajectory: JointTrajectory
    checker: SceneChecker


def plan_motion(
    options: Options,
    report: dict[str, Any],
    *,
    scene=SCENE,
    checker: SceneChecker | None = None,
    q_start: np.ndarray | None = None,
) -> MotionPlan:
    options.validate()
    started = time.perf_counter()
    report["stage"] = "scene"
    checker = (
        checker if checker is not None else SceneChecker(scene, clearance_m=options.clearance_m)
    )
    if not math.isclose(checker.model.opt.timestep, options.timestep_s, rel_tol=0, abs_tol=1e-12):
        raise MotionError("scene", "scene physics timestep differs from trajectory timestep")
    report["scene"] = str(scene)
    report["clearance_query_cap_m"] = checker.query_margin
    report["stage"] = "start_ik"
    # Separate deterministic streams keep endpoint sampling independent of search effort.
    seeds = np.random.SeedSequence(options.seed).generate_state(6)
    if q_start is None:
        starts = endpoint_candidates(checker, options.start_pose, seed=int(seeds[0]), name="start")
    else:
        q_start = np.asarray(q_start, dtype=float)
        if q_start.shape != (checker.n_arm_qpos,) or not _within_limits(q_start, checker):
            raise MotionError("start_ik", "actual carry start violates joint limits")
        if checker.in_collision(q_start):
            raise MotionError("start_ik", "actual carry start has insufficient clearance")
        error_mm, error_deg = tcp_error(checker.model, checker.data, options.start_pose)
        if error_mm > 2 or error_deg > 1:
            raise MotionError(
                "start_ik", "actual carry start has not reached the requested TCP pose"
            )
        starts = [q_start.copy()]
    report["start_candidates_rad"] = [q.tolist() for q in starts]
    report["stage"] = "goal_ik"
    goals = endpoint_candidates(checker, options.goal_pose, seed=int(seeds[1]), name="goal")
    report["goal_candidates_rad"] = [q.tolist() for q in goals]
    pairs = sorted(
        ((a, b) for a in starts for b in goals), key=lambda pair: np.linalg.norm(pair[1] - pair[0])
    )[:4]
    # Search and shortcut edges at the same resolution used by independent validation.
    edge_resolution = min(options.edge_resolution_rad, 0.01)
    report["edge_resolution_used_rad"] = edge_resolution
    report["attempts"] = []
    for index, (start, goal) in enumerate(pairs):
        report["stage"] = "search"
        pair_seed = options.seed if index == 0 else int(seeds[index + 1])
        attempt = {"pair": index, "seed": pair_seed}
        report["attempts"].append(attempt)
        try:
            raw = plan_collision_free(
                start,
                goal,
                checker=checker,
                max_iter=options.max_iterations,
                step=options.step_rad,
                check_res=edge_resolution,
                seed=pair_seed,
            )
        except PlanningError as exc:
            attempt["error"] = str(exc)
            continue
        report["stage"] = "path_validation"
        path = shortcut_path(
            checker,
            raw,
            iterations=options.shortcut_iterations,
            check_res=edge_resolution,
            seed=pair_seed,
        )
        path_minimum = check_path(path, checker, resolution=0.01, clearance=options.clearance_m)
        if not np.array_equal(path[[0, -1]], np.vstack((start, goal))):
            raise MotionError("path_validation", "simplification changed endpoints")
        if path_length(path) > path_length(raw) + 1e-9:
            raise MotionError("path_validation", "simplification increased path length")
        report.update(
            start_joints_rad=start.tolist(),
            goal_joints_rad=goal.tolist(),
            raw_path_length_rad=path_length(raw),
            path_length_rad=path_length(path),
            path_waypoints_rad=path.tolist(),
            path_min_clearance_m=path_minimum,
        )
        report["stage"] = "trajectory_validation"
        trajectory, method, minimum, rejection = time_parameterize(path, checker, options)
        baseline = np.array(list(sample_edges(np.vstack((start, goal)), 0.01)))
        direct_minimum = min(checker.min_distance(q) for q in baseline)
        report.update(
            planning_time_s=time.perf_counter() - started,
            trajectory_duration_s=trajectory.duration,
            trajectory_samples=trajectory.n_steps,
            interpolation=method,
            spline_rejection=rejection,
            planned_min_clearance_m=minimum,
            planned_gripper_internal_min_clearance_m=checker.checked_gripper_min,
            baseline_min_clearance_m=direct_minimum,
            baseline_collides=direct_minimum < 0,
            baseline_clearance_passed=direct_minimum >= options.clearance_m,
            max_planned_velocity_rad_s=float(np.max(np.abs(trajectory.qd))),
            max_planned_acceleration_rad_s2=float(np.max(np.abs(trajectory.qdd))),
        )
        return MotionPlan(path, raw, baseline, trajectory, checker)
    raise MotionError("search", "预算内未找到路径 (no path found within the search budget)")
