"""Shared motion smoothing, soft-landing deceleration, and Eye-in-Hand vision closed-loop utilities."""

from __future__ import annotations

from common.motion import (
    interpolate_waypoint_chain,
    read_real_gripper_gap_mm,
    smooth_progress_profile,
    smooth_step_durations,
    solve_cartesian_segment,
    stream_waypoints_on_real,
)
from common.vision import (
    CAMERA_TOOL_SRC,
    DEFAULT_HANDEYE_YAML,
    PY312_PYTHON,
    WORKSPACE_BOUNDS_M,
    OrbbecCameraStream,
    PerceptionError,
    capture_rgbd_frame,
    compute_flange_and_cam_poses,
    deproject_pixels_to_cam,
    load_handeye_calibration,
    project_cam_to_pixels,
    solve_camera_lookdown_q,
    validate_xy_in_workspace,
)

__all__ = [
    "CAMERA_TOOL_SRC",
    "DEFAULT_HANDEYE_YAML",
    "PY312_PYTHON",
    "WORKSPACE_BOUNDS_M",
    "OrbbecCameraStream",
    "PerceptionError",
    "capture_rgbd_frame",
    "compute_flange_and_cam_poses",
    "deproject_pixels_to_cam",
    "interpolate_waypoint_chain",
    "load_handeye_calibration",
    "project_cam_to_pixels",
    "read_real_gripper_gap_mm",
    "smooth_progress_profile",
    "smooth_step_durations",
    "solve_camera_lookdown_q",
    "solve_cartesian_segment",
    "stream_waypoints_on_real",
    "validate_xy_in_workspace",
]
