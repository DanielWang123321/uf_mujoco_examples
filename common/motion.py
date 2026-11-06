# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Joint-space trajectory smoothing, soft-landing deceleration, and real-robot streaming."""

from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from ufactory_mujoco.ik import solve_ik_trajectory
from ufactory_mujoco.safety import SafetyGateError


def smooth_progress_profile(steps: int, *, soft_landing: bool = True) -> np.ndarray:
    """Return a monotonically increasing normalized progress array s in (0, 1] of length ``steps``."""
    n = max(1, int(steps))
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    u = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)[1:]
    s_quintic = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
    if not soft_landing:
        return s_quintic
    u_warped = np.sin(0.5 * np.pi * u) ** 0.85
    s_soft = 10.0 * (u_warped**3) - 15.0 * (u_warped**4) + 6.0 * (u_warped**5)
    s_soft[-1] = 1.0
    return np.maximum.accumulate(np.clip(s_soft, 0.0, 1.0))


def smooth_step_durations(
    duration: float,
    steps: int,
    *,
    soft_landing: bool = True,
    min_dt: float = 0.008,
) -> np.ndarray:
    """Compute per-step durations (summing to >= duration) for uniform spatial waypoints.

    Allocates longer durations at the start (smooth acceleration) and especially over the final
    steps when ``soft_landing=True`` (smooth deceleration to rest), while keeping spatial
    waypoints uniformly spaced.
    """
    n = max(1, int(steps))
    if n == 1:
        return np.array([max(float(duration), min_dt)], dtype=np.float64)
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    # Bell-shaped speed profile v(u) in [v_min, 1.0]
    v_prof = np.sin(np.pi * u) ** 0.85
    if soft_landing:
        # Taper speed further over the final 30% of the stroke for soft landing
        tail = np.clip((u - 0.70) / 0.30, 0.0, 1.0)
        v_prof *= 1.0 - 0.55 * (0.5 * (1.0 - np.cos(np.pi * tail)))
    v_prof = np.maximum(v_prof, 0.22)
    inv_v = 1.0 / v_prof
    dts = float(duration) * (inv_v / np.sum(inv_v))
    return np.maximum(dts, float(min_dt))


def interpolate_waypoint_chain(
    q_start: np.ndarray,
    waypoints: list[np.ndarray],
    *,
    dt: float = 0.02,
    max_vel_rad_s: float = 0.65,
    max_acc_rad_s2: float = 2.0,
    soft_landing: bool = False,
) -> np.ndarray:
    """Convert a waypoint list into a C2-smooth, curvature-limited 50 Hz joint stream (N, 6).

    Uses arc-length parameterization, endpoint-pinned corner smoothing, curvature-bounded speed
    caps, and an extended soft-landing deceleration ramp when ``soft_landing=True``.
    """
    pts: list[np.ndarray] = [np.asarray(q_start, dtype=np.float64).reshape(-1).copy()]
    for q_wp in waypoints:
        q_arr = np.asarray(q_wp, dtype=np.float64).reshape(-1)
        if float(np.max(np.abs(q_arr - pts[-1]))) > 1e-5:
            pts.append(q_arr.copy())

    if len(pts) == 1:
        return pts[0].reshape(1, -1)

    # 1. Densify coarse segments in joint space so arc-length grid is uniform
    dense: list[np.ndarray] = [pts[0]]
    for i in range(len(pts) - 1):
        dq = pts[i + 1] - pts[i]
        dist = float(np.max(np.abs(dq)))
        n_sub = max(1, int(np.ceil(dist / 0.004)))
        for k in range(1, n_sub + 1):
            dense.append(pts[i] + (k / n_sub) * dq)
    p_dense = np.vstack(dense)

    # 2. Parameterize by cumulative infinity-norm joint arc-length
    seg_ds = np.max(np.abs(np.diff(p_dense, axis=0)), axis=1)
    s_knots = np.concatenate([[0.0], np.cumsum(seg_ds)])
    s_total = float(s_knots[-1])
    if s_total < 1e-6:
        return p_dense[:1]

    ds_nom = 0.002
    n_grid = max(80, int(np.ceil(s_total / ds_nom)) + 1)
    u_grid = np.linspace(0.0, s_total, n_grid)
    ds_step = float(u_grid[1] - u_grid[0])
    q_raw = np.column_stack(
        [np.interp(u_grid, s_knots, p_dense[:, j]) for j in range(p_dense.shape[1])]
    )

    # 3. Smooth internal corners in arc-length space while pinning start & final vertical descent
    sigma_pts = max(2.0, 0.045 / max(ds_step, 1e-6))
    q_blur = gaussian_filter1d(q_raw, sigma=sigma_pts, axis=0, mode="nearest")
    pin_start = min(n_grid // 4, max(8, int(0.08 / max(ds_step, 1e-6))))
    pin_end = min(
        n_grid // 3,
        max(12, int((0.12 if soft_landing else 0.08) / max(ds_step, 1e-6))),
    )
    env = np.ones(n_grid, dtype=np.float64)
    env[:pin_start] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, pin_start)))
    env[-pin_end:] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(1.0, 0.0, pin_end)))
    q_grid = q_raw + env[:, None] * (q_blur - q_raw)

    # Recompute exact arc-length on smoothed geometric path
    seg_ds2 = np.max(np.abs(np.diff(q_grid, axis=0)), axis=1)
    s_grid = np.concatenate([[0.0], np.cumsum(seg_ds2)])
    s_total = float(s_grid[-1])
    if s_total < 1e-6:
        return q_grid[:1]
    u_grid = np.linspace(0.0, s_total, n_grid)
    ds_step = float(u_grid[1] - u_grid[0])
    q_grid = np.column_stack(
        [np.interp(u_grid, s_grid, q_grid[:, j]) for j in range(q_grid.shape[1])]
    )

    # 4. Curvature-bounded speed cap: ||d2q/ds2||_inf * v^2 <= a_corner
    q_s = np.gradient(q_grid, ds_step, axis=0)
    q_ss = np.gradient(q_s, ds_step, axis=0)
    curvature = np.max(np.abs(q_ss), axis=1)
    curvature = gaussian_filter1d(
        curvature, sigma=max(2.0, 0.02 / max(ds_step, 1e-6)), mode="nearest"
    )
    a_corner = 0.55 * max_acc_rad_s2
    v_cap = np.minimum(max_vel_rad_s, np.sqrt(a_corner / np.maximum(curvature, 1e-4)))

    # 5. Soft-landing speed taper over final placement / approach zone
    if soft_landing:
        land_zone = min(0.26, 0.40 * s_total)
        rem = s_total - u_grid
        in_land = rem <= land_zone
        ratio = np.clip(rem[in_land] / max(land_zone, 1e-6), 0.0, 1.0)
        v_land = 0.05 + (max_vel_rad_s - 0.05) * (0.5 * (1.0 - np.cos(np.pi * ratio)))
        v_cap[in_land] = np.minimum(v_cap[in_land], v_land)

    v_prof = v_cap.copy()
    v_prof[0] = 0.0
    v_prof[-1] = 0.0
    a_tan_up = 0.65 * max_acc_rad_s2
    a_tan_dn = (0.30 if soft_landing else 0.55) * max_acc_rad_s2

    for i in range(n_grid - 1):
        v_prof[i + 1] = min(v_prof[i + 1], np.sqrt(v_prof[i] ** 2 + 2.0 * a_tan_up * ds_step))
    for i in range(n_grid - 1, 0, -1):
        rem_s = s_total - u_grid[i - 1]
        a_local = (0.18 * max_acc_rad_s2) if (soft_landing and rem_s < 0.12) else a_tan_dn
        v_prof[i - 1] = min(v_prof[i - 1], np.sqrt(v_prof[i] ** 2 + 2.0 * a_local * ds_step))

    # 6. Time integration and C2 velocity smoothing with exact endpoint target preservation
    v_mid = np.maximum(0.5 * (v_prof[:-1] + v_prof[1:]), 1e-3)
    dt_seg = np.diff(u_grid) / v_mid
    t_grid = np.concatenate([[0.0], np.cumsum(dt_seg)])
    t_total = float(t_grid[-1])
    n_steps = max(4, int(np.ceil(t_total / dt)))
    t_samples = np.linspace(0.0, n_steps * dt, n_steps + 1)
    t_grid *= (n_steps * dt) / max(t_total, 1e-6)
    s_of_t = np.interp(t_samples, t_grid, u_grid)

    q_out = np.column_stack(
        [np.interp(s_of_t, u_grid, q_grid[:, j]) for j in range(q_grid.shape[1])]
    )
    qd_raw = np.diff(q_out, axis=0) / dt
    qd_pad = np.vstack([np.zeros((6, q_out.shape[1])), qd_raw, np.zeros((8, q_out.shape[1]))])
    qd_smooth = gaussian_filter1d(qd_pad, sigma=2.2, axis=0, mode="constant", cval=0.0)[3:-5]
    qd_smooth[0] = 0.0
    qd_smooth[-1] = 0.0
    q_recon = np.vstack([q_out[0:1], q_out[0:1] + np.cumsum(qd_smooth * dt, axis=0)])
    total_dq = q_out[-1] - q_out[0]
    recon_dq = q_recon[-1] - q_recon[0]
    for j in range(q_out.shape[1]):
        if abs(recon_dq[j]) > 1e-6:
            q_recon[:, j] = q_out[0, j] + (q_recon[:, j] - q_out[0, j]) * (
                total_dq[j] / recon_dq[j]
            )
        else:
            q_recon[:, j] = q_out[0, j]
    q_recon[-1] = pts[-1]
    return q_recon


def solve_cartesian_segment(
    sim_robot: Any,
    q_init6: np.ndarray,
    pos_goal: np.ndarray,
    quat_goal: np.ndarray,
    steps: int,
) -> list[np.ndarray]:
    """Solve a straight Cartesian path in calibrated MuJoCo without stepping physics."""
    sim_robot.data.qpos[:6] = np.asarray(q_init6, dtype=np.float64).reshape(6)
    mujoco.mj_forward(sim_robot.model, sim_robot.data)
    pos0, quat0 = sim_robot.get_tcp_pose()
    s = np.linspace(0.0, 1.0, steps + 1)[1:]
    wps = pos0 + s[:, None] * (np.asarray(pos_goal, dtype=np.float64) - pos0)
    q0_conj = np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]], dtype=np.float64)
    q_delta = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(q_delta, q0_conj, np.asarray(quat_goal, dtype=np.float64))
    rv = np.empty(3, dtype=np.float64)
    mujoco.mju_quat2Vel(rv, q_delta, 1.0)
    quat_traj = np.empty((steps, 4), dtype=np.float64)
    for i in range(steps):
        quat_traj[i] = quat0
        mujoco.mju_quatIntegrate(quat_traj[i], rv, s[i])
    q_start = sim_robot.data.qpos.copy()
    q_traj, ok = solve_ik_trajectory(
        sim_robot.model,
        sim_robot.eval_data,
        sim_robot._tcp_site_name(),
        wps,
        quat_traj,
        q_start,
    )
    if not ok:
        raise SafetyGateError(f"Cartesian segment IK did not converge: pos={pos_goal}")
    return [np.asarray(q[:6], dtype=np.float64).copy() for q in q_traj]


def read_real_gripper_gap_mm(robot: Any) -> float:
    """Read physical Gripper G2 opening in mm (preferring get_gripper_g2_position)."""
    g2_getter = getattr(robot._arm, "get_gripper_g2_position", None)
    if callable(g2_getter):
        code, pos_mm = g2_getter()
        if int(code) == 0 and pos_mm is not None:
            return float(pos_mm)
    return float(robot.get_gripper_gap() * 1000.0)


def stream_waypoints_on_real(
    robot: Any,
    waypoints: list[np.ndarray],
    *,
    max_vel_rad_s: float = 0.65,
    max_acc_rad_s2: float = 2.0,
    dt: float = 0.02,
    soft_landing: bool = False,
) -> np.ndarray:
    """Stream a sequence of joint waypoints smoothly in servo_j mode on the real xArm6."""
    q_cur = robot.get_joint_positions()
    if not waypoints:
        return np.asarray(q_cur, dtype=np.float64).reshape(1, -1)
    stream = interpolate_waypoint_chain(
        q_cur,
        waypoints,
        dt=dt,
        max_vel_rad_s=max_vel_rad_s,
        max_acc_rad_s2=max_acc_rad_s2,
        soft_landing=soft_landing,
    )
    for q_cmd in stream[1:]:
        robot.set_joint_positions(q_cmd)
        time.sleep(dt)
    robot.settle(0.18)
    robot.get_joint_positions()
    return stream
