# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Helpers shared by the Sim-to-Real pipelines (dry-run / sdk-sim / real)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from ufactory_mujoco.calibration import KinematicsCalibration, apply_calibration_to_model
from ufactory_mujoco.collision import CollisionChecker

FLANGE_BODY = "link6"


def make_collision_checker(
    scene_path: str | Path,
    calibration: KinematicsCalibration | None = None,
    *,
    ignore_bodies: tuple[str, ...] = (),
) -> CollisionChecker:
    """Checker for arm self-collision and arm/gripper-vs-table, ignoring the task objects."""
    checker = CollisionChecker(
        scene_path,
        safety_margin=0.0,
        query_margin=0.01,
        ignore_pairs=[("world", "link_base"), ("world", "link1")],
        ignore_bodies=ignore_bodies,
    )
    if calibration is not None:
        apply_calibration_to_model(checker.model, calibration, checker.data)
    return checker


def flange_pose(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the current 4x4 base-to-flange (``link6``) transform from ``data``."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY)
    t_bf = np.eye(4, dtype=np.float64)
    t_bf[:3, :3] = data.xmat[body_id].reshape(3, 3)
    t_bf[:3, 3] = data.xpos[body_id]
    return t_bf


def config_sha256(payload: dict[str, Any]) -> str:
    """Stable SHA-256 of a JSON-serializable task configuration (used by the safety gates)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_json_report(report_path: str | Path | None, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty JSON; no-op when ``report_path`` is empty."""
    if not report_path:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
