# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Helpers shared by the Sim-to-Real pipelines (dry-run / sdk-sim / real)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from ufactory_mujoco.calibration import (
    KinematicsCalibration,
    apply_calibration_to_model,
    read_tcp_offset,
    tcp_mm_rpy_to_flange_m_quat,
)
from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.safety import SafetyGateError

from common.paths import default_handeye_path, resolve_repo_path

FLANGE_BODY = "link6"


@contextmanager
def pipeline_error_report(
    report_path: str | Path | None, payload: dict[str, Any]
) -> Iterator[None]:
    """Preserve calibration/capture evidence when detection or execution fails."""
    try:
        yield
    except Exception as exc:
        payload.update(ok=False, success=False, error=str(exc))
        write_json_report(report_path, payload)
        raise


def camera_run_context(
    camera_type: str,
    handeye_path: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the camera-specific calibration and capture its exact file hash for reports."""
    path = resolve_repo_path(handeye_path or default_handeye_path(camera_type)).resolve()
    return path, {
        "camera": {"type": camera_type},
        "camera_frames": [],
        "handeye_path": str(path),
        "handeye_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def record_camera_frame(
    context: dict[str, Any],
    stage: str,
    metadata: dict[str, Any],
    t_base_flange: np.ndarray | None = None,
) -> None:
    """Record actual SDK configuration per capture, including near and post-task frames."""
    context["camera"].update(metadata)
    frame = {"stage": stage, **metadata}
    if t_base_flange is not None:
        frame["T_base_flange"] = np.asarray(t_base_flange).tolist()
    context["camera_frames"].append(frame)


def save_rgbd_capture(
    image_path: str | Path | None,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    metadata: dict[str, Any],
    t_base_flange: np.ndarray,
) -> None:
    """Keep replayable input beside a requested overlay, including on detection failure."""
    if image_path is None:
        return
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path.with_suffix(".input.png")), color_bgr):
        raise OSError(f"Failed to save camera image: {path}")
    np.savez_compressed(
        path.with_suffix(".rgbd.npz"),
        color_bgr=color_bgr,
        depth_m=depth_m,
        metadata_json=json.dumps(metadata),
        T_base_flange=t_base_flange,
    )


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


def verify_controller_tcp_offset_and_fk(
    robot: Any,
    *,
    xy_tol_mm: float = 5.0,
    z_bounds_mm: tuple[float, float] = (168.0, 178.0),
    rpy_tol_rad: float = 0.05,
    fk_pos_tol_mm: float = 1.0,
    fk_rot_tol_deg: float = 0.2,
) -> dict[str, Any]:
    """Verify Gripper G2 SDK ``tcp_offset`` and pure FK agreement with calibrated MuJoCo ``link6``."""
    try:
        tcp_offset = np.asarray(read_tcp_offset(robot._arm), dtype=np.float64).reshape(6)
    except Exception as exc:
        raise SafetyGateError(f"Failed to read controller tcp_offset via SDK: {exc}") from exc

    x_mm, y_mm, z_mm = float(tcp_offset[0]), float(tcp_offset[1]), float(tcp_offset[2])
    rpy_rad = tcp_offset[3:6]
    if (
        abs(x_mm) > xy_tol_mm
        or abs(y_mm) > xy_tol_mm
        or not (z_bounds_mm[0] <= z_mm <= z_bounds_mm[1])
        or float(np.max(np.abs(rpy_rad))) > rpy_tol_rad
    ):
        raise SafetyGateError(
            f"Controller tcp_offset {tcp_offset.tolist()} (mm, rad) does not match Gripper G2 "
            f"(expected |x|,|y|<={xy_tol_mm:.1f} mm, z in [{z_bounds_mm[0]:.1f}, "
            f"{z_bounds_mm[1]:.1f}] mm, |rpy|<={rpy_tol_rad:.3f} rad)"
        )

    q_current = np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)
    code_fk, sdk_pose_raw = robot._arm.get_forward_kinematics(
        q_current.tolist(), input_is_radian=True, return_is_radian=True
    )
    if int(code_fk) != 0 or sdk_pose_raw is None:
        raise SafetyGateError(f"Controller get_forward_kinematics failed with SDK code {code_fk}")
    sdk_tcp = np.asarray(sdk_pose_raw[:6], dtype=np.float64)
    sdk_flange_pos_m, sdk_flange_quat = tcp_mm_rpy_to_flange_m_quat(sdk_tcp, tcp_offset)

    t_mj = flange_pose(robot.model, robot.data)
    mj_pos_m = t_mj[:3, 3].copy()
    mj_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(mj_quat, t_mj[:3, :3].reshape(9))

    pos_err_mm = float(np.linalg.norm(sdk_flange_pos_m - mj_pos_m) * 1000.0)
    q_sdk = sdk_flange_quat / np.linalg.norm(sdk_flange_quat)
    q_mj = mj_quat / np.linalg.norm(mj_quat)
    dot = float(min(1.0, abs(float(np.dot(q_sdk, q_mj)))))
    rot_err_deg = float(np.rad2deg(2.0 * np.arccos(dot)))

    if pos_err_mm > fk_pos_tol_mm or rot_err_deg > fk_rot_tol_deg:
        raise SafetyGateError(
            f"Controller FK minus tcp_offset disagrees with calibrated MuJoCo link6: "
            f"pos_err={pos_err_mm:.3f} mm (tol {fk_pos_tol_mm:.1f} mm), "
            f"rot_err={rot_err_deg:.4f} deg (tol {fk_rot_tol_deg:.2f} deg)"
        )

    raw_load = getattr(robot._arm, "tcp_load", None)
    tcp_load: list[Any] | None = None
    if isinstance(raw_load, (list, tuple)) and len(raw_load) == 2:
        mass = float(raw_load[0])
        com = [float(v) for v in np.asarray(raw_load[1], dtype=np.float64).reshape(-1)]
        tcp_load = [mass, com]

    return {
        "ok": True,
        "gripper": "gripper_g2",
        "tcp_offset_mm_rad": [float(v) for v in tcp_offset],
        "tcp_load": tcp_load,
        "sdk_tcp_mm_rad": [float(v) for v in sdk_tcp],
        "sdk_flange_pos_m": [float(v) for v in sdk_flange_pos_m],
        "mujoco_flange_pos_m": [float(v) for v in mj_pos_m],
        "fk_flange_pos_error_mm": pos_err_mm,
        "fk_flange_rot_error_deg": rot_err_deg,
    }


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
