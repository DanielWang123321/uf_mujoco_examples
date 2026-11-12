# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Repository paths, default calibration files, and the ``ufactory_mujoco`` import bootstrap.

This module only depends on the standard library so it can be imported before any
third-party package (see ``common/__init__.py`` and ``examples/__init__.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

# Sibling checkout expected next to this repository (see README "Installation").
UFACTORY_MUJOCO_ROOT = WORKSPACE_ROOT / "ufactory_mujoco"

DEFAULT_ROBOT_IP = "192.168.1.60"

# Calibration defaults; local per-robot files are ignored (see calibration/README.md).
CALIBRATION_DIR = REPO_ROOT / "calibration"
DEFAULT_CALIBRATION_YAML = CALIBRATION_DIR / "xarm6_kinematics.yaml"
ORBBEC_HANDEYE_YAML = CALIBRATION_DIR / "eye_in_hand.yaml"
DEFAULT_HANDEYE_YAML = CALIBRATION_DIR / "eye_in_hand.d435.yaml"

# Sample calibration files used by the tests (no hardware needed).
SAMPLE_CALIBRATION_YAML = UFACTORY_MUJOCO_ROOT / "calibrations" / "xarm6_kinematics_L56A01.yaml"
SAMPLE_HANDEYE_YAML = CALIBRATION_DIR / "eye_in_hand.example.yaml"

BUILD_DIR = REPO_ROOT / "build"


def default_handeye_path(camera_type: str = "realsense") -> Path:
    """Select the installed hand-eye file for the requested camera family."""
    if camera_type == "realsense":
        return DEFAULT_HANDEYE_YAML
    if camera_type == "orbbec":
        return ORBBEC_HANDEYE_YAML
    raise ValueError(f"Unsupported camera type: {camera_type!r}")


def ensure_ufactory_mujoco() -> None:
    """Make the sibling ``ufactory_mujoco`` checkout importable (no-op if already on sys.path)."""
    root = str(UFACTORY_MUJOCO_ROOT)
    if UFACTORY_MUJOCO_ROOT.is_dir() and root not in sys.path:
        sys.path.insert(0, root)


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a user/config path: relative paths that do not exist in the current directory
    are taken relative to the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.exists():
        return (REPO_ROOT / path).resolve()
    return path
