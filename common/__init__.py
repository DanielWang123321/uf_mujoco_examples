# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Shared building blocks for the uf_mujoco_examples tasks.

Modules
-------
- ``paths``        : repository / sibling-repository paths and the ``ufactory_mujoco`` bootstrap.
- ``cli``          : YAML config loading and the CLI options shared by every example.
- ``geometry``     : small quaternion / vector helpers.
- ``sim``          : ``TaskSimRobot`` base class and MuJoCo contact helpers.
- ``motion``       : joint-space smoothing, soft landing, Cartesian IK segments, real-robot streaming.
- ``camera``       : RealSense / Orbbec RGB-D capture (``pyrealsense2`` / ``pyorbbecsdk``), depth aligned to color.
- ``camera_check`` : read-only RGB-D stream health check CLI.
- ``vision``       : hand-eye calibration, Eye-in-Hand projection, camera look-down IK.
- ``sim2real``     : collision checker factory, config hashing, JSON reports.
- ``objects``      : physical object models and perception helpers shared by several examples.
"""

from __future__ import annotations

from common.paths import ensure_ufactory_mujoco

# Must run before any submodule imports ``ufactory_mujoco``.
ensure_ufactory_mujoco()
