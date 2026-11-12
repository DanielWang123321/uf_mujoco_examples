# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""A flange-Yaw 180 camera needs a symmetric grasp and a matching safe retreat."""

import numpy as np
import pytest

from common.paths import DEFAULT_HANDEYE_YAML, SAMPLE_CALIBRATION_YAML
from common.vision import cup_grasp_yaw_offset, load_handeye_calibration, solve_camera_lookdown_q
from examples.pick_cup import perception, sim2real


@pytest.mark.parametrize("near", [False, True])
def test_rotated_camera_pick_preflight_preserves_clearance_and_release(near):
    path = DEFAULT_HANDEYE_YAML.with_name("eye_in_hand.d435.yaw180.example.yaml")
    calibration = sim2real.load_kinematics_calibration(SAMPLE_CALIBRATION_YAML, robot="xarm6")
    pose = perception.pose_estimate_from_manual(0.284, 0.013, 72.6)
    q_obs = np.array([-0.01048, -0.45906, -1.41007, 0.00862, 1.86947, -0.00069])
    assert cup_grasp_yaw_offset(load_handeye_calibration(path)) == np.pi
    q_start = q_obs
    if near:
        q_start = solve_camera_lookdown_q(
            pose.cup_center_xy_m,
            q_seed=q_obs,
            z_cam_m=0.28,
            scene_path=sim2real.task.SCENE,
            calibration=calibration,
            handeye_yaml=path,
            ignore_bodies=(sim2real.task.CUP,),
        )
        sim2real._checked_hover_stream(
            q_obs, q_start, calibration=calibration, handeye_path=path, table_z_m=0.0
        )
    report = sim2real.run_digital_twin_preflight(
        pose,
        np.array([0.38, -0.18]),
        q_init=q_start,
        q_return=q_obs,
        calibration=calibration,
        handeye_yaml=path,
        headless=True,
    )
    assert report.ok, report.failures
    assert report.min_camera_z_mm >= 95 and report.min_collision_dist_mm > 0
    assert report.metrics["error_mm"] < 30 and report.metrics["stable_s"] >= 1.0
    assert np.array_equal(report.phases.retreat_waypoints[-1], q_obs)
