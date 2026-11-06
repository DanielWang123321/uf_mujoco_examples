# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Unit and MuJoCo physics tests for the screw and nut 3D assets."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from generate_screw_nut import (
    NUT_BORE_DIAM,
    NUT_BOT_DIAM,
    NUT_HEIGHT,
    NUT_MASS,
    NUT_TOP_DIAM,
    SCREW_HEAD_DIAM,
    SCREW_HEAD_LEN,
    SCREW_MASS,
    SCREW_THREAD_DIAM,
    SCREW_TOTAL_LEN,
    build_nut_mesh,
    build_screw_mesh,
)

ASSET_DIR = Path(__file__).resolve().parent


def _check_watertight_manifold(vertices: np.ndarray, faces: np.ndarray) -> tuple[bool, int]:
    """Verify every directed edge (u, v) appears once and has a matching reverse edge (v, u)."""
    directed_edges: set[tuple[int, int]] = set()
    for tri in faces:
        u, v, w = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((u, v), (v, w), (w, u)):
            if (a, b) in directed_edges:
                return False, 0
            directed_edges.add((a, b))
    for a, b in directed_edges:
        if (b, a) not in directed_edges:
            return False, 0
    n_edges = len(directed_edges) // 2
    euler_chi = len(vertices) - n_edges + len(faces)
    return True, euler_chi


def test_screw_mesh_watertight_and_dimensions() -> None:
    """Verify screw mesh is watertight (genus 0, chi=2) and matches the measured specs."""
    verts, faces = build_screw_mesh()
    ok, chi = _check_watertight_manifold(verts, faces)
    assert ok, "Screw mesh is not a watertight 2-manifold"
    assert chi == 2, f"Expected genus-0 Euler characteristic 2, got {chi}"

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    span = maxs - mins
    assert np.isclose(mins[2], 0.0, atol=1e-6)
    assert np.isclose(maxs[2], SCREW_TOTAL_LEN, atol=1e-6)
    assert np.isclose(span[0], SCREW_HEAD_DIAM, atol=1e-4)
    assert np.isclose(span[1], SCREW_HEAD_DIAM, atol=1e-4)

    # Check shaft outer diameter above head (z in [0.015, 0.042])
    shaft_mask = (verts[:, 2] >= SCREW_HEAD_LEN + 0.001) & (verts[:, 2] <= SCREW_TOTAL_LEN - 0.002)
    shaft_r_max = float(np.max(np.hypot(verts[shaft_mask, 0], verts[shaft_mask, 1])))
    assert np.isclose(shaft_r_max * 2.0, SCREW_THREAD_DIAM, atol=1e-4)


def test_nut_mesh_watertight_and_dimensions() -> None:
    """Verify nut mesh is a watertight torus (genus 1, chi=0) and matches the measured + scanned specs."""
    verts, faces = build_nut_mesh()
    ok, chi = _check_watertight_manifold(verts, faces)
    assert ok, "Nut mesh is not a watertight 2-manifold"
    assert chi == 0, f"Expected genus-1 toroidal Euler characteristic 0, got {chi}"

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    span = maxs - mins
    assert np.isclose(mins[2], 0.0, atol=1e-6)
    assert np.isclose(maxs[2], NUT_HEIGHT, atol=1e-6)
    assert np.isclose(span[0], NUT_BOT_DIAM, atol=1e-4)
    assert np.isclose(span[1], NUT_BOT_DIAM, atol=1e-4)

    # Check upper hex circumdiameter (z in [0.005, 0.013])
    upper_mask = (verts[:, 2] >= 0.005) & (verts[:, 2] <= 0.013)
    radii_upper = np.hypot(verts[upper_mask, 0], verts[upper_mask, 1])
    assert np.isclose(float(np.max(radii_upper)) * 2.0, NUT_TOP_DIAM, atol=1e-4)
    assert np.isclose(float(np.min(radii_upper)) * 2.0, NUT_BORE_DIAM, atol=1e-4)


def test_mjcf_models_mass_and_table_stability() -> None:
    """Verify screw.xml (56.5g) and nut.xml (29.0g) compile and rest stably on the floor."""
    screw_model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "screw.xml"))
    screw_data = mujoco.MjData(screw_model)
    mujoco.mj_forward(screw_model, screw_data)
    screw_id = mujoco.mj_name2id(screw_model, mujoco.mjtObj.mjOBJ_BODY, "screw")
    assert np.isclose(screw_model.body_subtreemass[screw_id], SCREW_MASS, atol=1e-6)

    for _ in range(300):
        mujoco.mj_step(screw_model, screw_data)
    assert abs(float(screw_data.xpos[screw_id, 2])) < 0.001
    assert float(screw_data.xmat[screw_id].reshape(3, 3)[2, 2]) > 0.999

    nut_model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "nut.xml"))
    nut_data = mujoco.MjData(nut_model)
    mujoco.mj_forward(nut_model, nut_data)
    nut_id = mujoco.mj_name2id(nut_model, mujoco.mjtObj.mjOBJ_BODY, "nut")
    assert np.isclose(nut_model.body_subtreemass[nut_id], NUT_MASS, atol=1e-6)

    for _ in range(300):
        mujoco.mj_step(nut_model, nut_data)
    assert abs(float(nut_data.xpos[nut_id, 2])) < 0.001
    assert float(nut_data.xmat[nut_id].reshape(3, 3)[2, 2]) > 0.999


def test_nut_onto_screw_insertion_and_seating() -> None:
    """Verify the hollow nut self-locks on the top thread under gravity and threads down 10 turns to 14mm."""
    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "screw_nut.xml"))
    data = mujoco.MjData(model)
    screw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw")
    nut_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "nut")

    mujoco.mj_forward(model, data)
    screw_xy = data.xpos[screw_id, :2].copy()

    # 1. Spawn nut slightly off-center (+0.8 mm X, -0.6 mm Y) piloted at the top thread (z = 38.0 mm)
    data.qpos[0] = screw_xy[0] + 0.0008
    data.qpos[1] = screw_xy[1] - 0.0006
    data.qpos[2] = 0.0380
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Step 0.5s under gravity with zero external torque: verify helical thread self-locking
    for _ in range(500):
        mujoco.mj_step(model, data)

    locked_z_mm = float(data.xpos[nut_id, 2] * 1000.0)
    assert locked_z_mm > 35.0, (
        f"Nut failed to self-lock on top helical thread under gravity: z={locked_z_mm:.2f} mm"
    )

    # 2. Apply damped clockwise rotation (-4*pi rad/s = 2 turns/s) for 5.0s (10 turns)
    # and verify the nut advances down all helical ridges to seat at the 14.0mm shoulder.
    target_wz = -4.0 * np.pi
    for _ in range(5000):
        data.qfrc_applied[5] = 0.003 * (target_wz - data.qvel[5])
        mujoco.mj_step(model, data)
    data.qfrc_applied[5] = 0.0
    for _ in range(300):
        mujoco.mj_step(model, data)

    nut_pos = data.xpos[nut_id].copy()
    nut_z_axis = data.xmat[nut_id].reshape(3, 3)[:, 2]
    xy_err_mm = float(np.linalg.norm(nut_pos[:2] - screw_xy) * 1000.0)
    z_seated_mm = float(nut_pos[2] * 1000.0)
    tilt_deg = float(np.rad2deg(np.arccos(np.clip(nut_z_axis[2], -1.0, 1.0))))

    assert xy_err_mm < 0.6, f"Nut failed to center coaxially on screw shaft: err={xy_err_mm:.2f} mm"
    assert abs(z_seated_mm - SCREW_HEAD_LEN * 1000.0) < 0.8, (
        f"Nut did not seat on screw head shoulder (14.0 mm) after 10 helical turns: z={z_seated_mm:.2f} mm"
    )
    assert tilt_deg < 2.5, f"Nut tilted excessively on screw shaft: tilt={tilt_deg:.2f} deg"
