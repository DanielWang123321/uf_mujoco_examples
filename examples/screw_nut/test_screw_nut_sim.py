"""Unit and end-to-end physics tests for the 0.0.3 xArm 6 nut-threading simulation & closed-loop module."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screw_nut as demo
from screw_nut import (
    MAX_CONTACT_LOSS_S,
    NUT_BODY,
    NUT_MASS_KG,
    SCREW_BODY,
    SCREW_HEAD_LEN_M,
    SCREW_MASS_KG,
    DemoError,
    InitialNutPose,
    ScrewNutRobot,
    ScrewNutWaypoints,
    run_screw_nut,
)


@pytest.fixture(autouse=True)
def _isolate_local_modules() -> None:
    local_dir = Path(__file__).resolve().parent
    local_str = str(local_dir)
    if sys.path[0] != local_str:
        if local_str in sys.path:
            sys.path.remove(local_str)
        sys.path.insert(0, local_str)
    for mod_name in ("perception", "sim2real"):
        mod = sys.modules.get(mod_name)
        if mod is not None and Path(getattr(mod, "__file__", "")).resolve().parent != local_dir:
            sys.modules.pop(mod_name, None)


def test_scene_loading_and_fixed_screw() -> None:
    """Verify screw is glued to tabletop (0 DOFs, 56.5g, 192 helical capsules) and nut is free (29.0g)."""
    with ScrewNutRobot(viewer=False) as robot:
        screw_id = robot.model.body(SCREW_BODY).id
        nut_id = robot.model.body(NUT_BODY).id

        assert int(robot.model.body_dofnum[screw_id]) == 0, "Screw must have 0 DOFs (glued to table)"
        assert int(robot.model.body_dofnum[nut_id]) == 6, "Nut must have 6 DOFs (freejoint)"

        assert np.isclose(robot.model.body_subtreemass[screw_id], SCREW_MASS_KG, atol=1e-6)
        assert np.isclose(robot.model.body_subtreemass[nut_id], NUT_MASS_KG, atol=1e-6)

        # Verify 12-turn helical capsule geoms on screw (12 * 16 = 192 capsules) and 8 teeth on nut
        screw_geoms = [
            g for g in range(robot.model.ngeom) if int(robot.model.geom_bodyid[g]) == screw_id
        ]
        nut_geoms = [
            g for g in range(robot.model.ngeom) if int(robot.model.geom_bodyid[g]) == nut_id
        ]
        assert len(screw_geoms) >= 195
        assert len(nut_geoms) >= 48

        # Verify finger priority decoupling (priority=2, friction=4.0)
        lf_id = robot.model.body("left_finger").id
        lf_col_geoms = [
            g
            for g in range(robot.model.ngeom)
            if int(robot.model.geom_bodyid[g]) == lf_id and int(robot.model.geom_group[g]) == 3
        ]
        assert len(lf_col_geoms) > 0
        for g in lf_col_geoms:
            assert int(robot.model.geom_priority[g]) == 2
            assert np.isclose(robot.model.geom_friction[g, 0], 4.0)


def test_run_screw_nut_default_pose() -> None:
    """Verify 10-turn (3600 deg, 7 strokes) multi-stroke helical threading down to 14mm shoulder."""
    wps = ScrewNutWaypoints()
    with ScrewNutRobot(viewer=False) as robot:
        metrics = run_screw_nut(robot, recorded_waypoints=wps)

    assert metrics["xy_error_mm"] < 1.5
    assert abs(metrics["nut_z_mm"] - SCREW_HEAD_LEN_M * 1000.0) < 1.0
    assert metrics["tilt_deg"] < 3.0
    assert metrics["drift_mm"] < 0.5
    assert metrics["stroke_count"] == 7
    assert metrics["rotated_deg"] >= 3500.0
    assert metrics["max_regrip_settle_mm"] < 1.2
    assert metrics["max_contact_loss_s"] <= MAX_CONTACT_LOSS_S
    assert metrics["stable_s"] >= 1.0
    assert metrics["joint_limits"]["within_limits"] is True
    assert metrics["joint_limits"]["min_margin_deg"] > 10.0
    assert 12.0 <= metrics["measured_grasp_gap_mm"] <= 24.5
    assert metrics["pre_grasp_nut_delta_xy_mm"] < 2.0

    assert len(wps.approach_waypoints) > 0
    assert len(wps.lift_align_waypoints) > 0
    assert len(wps.thread_waypoints) > 0
    assert len(wps.regrip_waypoints) > 0
    assert len(wps.retreat_waypoints) > 0
    assert len(wps.stroke_waypoints) == 7
    assert len(wps.rewind_waypoints) == 6


def test_run_screw_nut_perturbed_poses() -> None:
    """Verify generalization to shifted nut/screw positions, rotated hex yaw, and 2-stroke (3.0 turns) threading."""
    nut_pose = InitialNutPose(xy_m=(0.33, 0.05), yaw_deg=25.0)
    screw_xy = np.array([0.38, -0.08], dtype=np.float64)
    wps = ScrewNutWaypoints()
    with ScrewNutRobot(viewer=False) as robot:
        metrics = run_screw_nut(
            robot,
            nut_pose=nut_pose,
            screw_xy=screw_xy,
            thread_turns=3.0,
            recorded_waypoints=wps,
        )

    assert metrics["xy_error_mm"] < 1.5
    assert abs(metrics["nut_z_mm"] - metrics["expected_nut_z_mm"]) < 1.8
    assert metrics["tilt_deg"] < 3.5
    assert metrics["stroke_count"] == 2
    assert metrics["rotated_deg"] >= 1050.0
    assert metrics["max_regrip_settle_mm"] < 1.2
    assert metrics["stable_s"] >= 1.0
    assert metrics["joint_limits"]["within_limits"] is True


def test_j6_high_turn_headroom_within_360_limits() -> None:
    """Verify 1.65 turns (594 deg) pre-positions J6 near -346 deg, stays in [-360, +360], and self-locks mid-shaft."""
    wps = ScrewNutWaypoints()
    with ScrewNutRobot(viewer=False) as robot:
        metrics = run_screw_nut(robot, thread_turns=1.65, recorded_waypoints=wps)

    assert metrics["joint_limits"]["within_limits"] is True
    assert metrics["stroke_count"] == 1
    assert metrics["j6_start_deg"] >= -357.0
    assert metrics["j6_end_deg"] <= 357.0
    assert metrics["rotated_deg"] >= 585.0
    assert abs(metrics["nut_z_mm"] - metrics["expected_nut_z_mm"]) < 1.8
    thread_j6 = np.array([q[5] for q in wps.thread_waypoints], dtype=np.float64)
    assert np.all(np.diff(thread_j6) >= -1e-6)


def test_screw_nut_trajectory_smooth_deceleration_and_soft_landing() -> None:
    """Verify common.motion trajectory interpolation provides zero endpoint vel, bounded acc, and soft-landing."""
    import perception
    import sim2real
    from ufactory_mujoco.calibration import load_kinematics_calibration

    calib = load_kinematics_calibration(sim2real.DEFAULT_CALIBRATION_YAML, robot="xarm6")
    nut_est, screw_est = perception.pose_estimates_from_manual(
        (0.3538, -0.0114),
        0.0,
        (0.3560, -0.1028),
    )
    q_obs = np.array([0.0, -0.55, -0.35, 0.0, 0.90, 0.0], dtype=np.float64)
    report = sim2real.run_digital_twin_preflight(
        nut_est,
        screw_est,
        thread_turns=3.0,
        q_init=q_obs,
        calibration=calib,
        headless=True,
    )
    assert report.ok, f"Digital twin preflight failed: {report.failures}"

    dt = 0.02
    q_cur = q_obs.copy()
    phases = [
        ("approach_grasp", report.waypoints.approach_waypoints, True),
        ("lift_align_capture", report.waypoints.lift_align_waypoints, True),
        ("helical_stroke_1", report.waypoints.stroke_waypoints[0], True),
        ("j6_rewind_1", report.waypoints.rewind_waypoints[0], True),
        ("helical_stroke_2", report.waypoints.stroke_waypoints[1], True),
        ("retreat_to_obs", report.waypoints.retreat_waypoints, False),
    ]
    for name, wps, soft_land in phases:
        seg = sim2real._interpolate_waypoint_chain(q_cur, wps, dt=dt, soft_landing=soft_land)
        qd = np.diff(seg, axis=0) / dt
        max_qd = np.max(np.abs(qd), axis=1)
        qdd = np.diff(qd, axis=0) / dt
        max_qdd = np.max(np.abs(qdd), axis=1)

        assert max_qd[0] < 1e-4, f"{name} start velocity not zero: {max_qd[0]}"
        assert max_qd[-1] < 1e-4, f"{name} end velocity not zero: {max_qd[-1]}"
        assert float(np.max(max_qdd)) <= 3.5, f"{name} peak qdd too high: {np.max(max_qdd)}"
        assert np.allclose(seg[-1], wps[-1], atol=1e-6), f"{name} endpoint mismatch"

        if soft_land:
            last5 = max_qd[-5:]
            assert float(np.max(last5)) < 0.025, f"{name} soft-landing too fast: {last5}"
            assert np.all(np.diff(last5) <= 1e-5), f"{name} not monotonically decelerating: {last5}"

        q_cur = seg[-1].copy()


def test_pre_grasp_missing_nut_or_empty_grasp_aborts(monkeypatch) -> None:
    """Verify pre-grasp nut displacement/missing gate and Gripper G2 empty-grasp gate."""
    # Case 1: Nut displaced outside search radius when robot reaches overhead approach pose
    with ScrewNutRobot(viewer=False) as robot:
        orig_pose = demo._nut_pose
        monkeypatch.setattr(
            demo,
            "_nut_pose",
            lambda r: (
                np.array([0.55, 0.18, 0.0], dtype=np.float64),
                orig_pose(r)[1],
            ),
        )
        with pytest.raises(DemoError, match="抓取前闭环拦截：螺母缺失或移出搜索半径"):
            run_screw_nut(robot, thread_turns=1.5)

    # Case 2: Gripper G2 closes to < 12.0 mm (empty grasp) -> must abort before lifting
    monkeypatch.setattr(demo, "_nut_pose", orig_pose)
    with ScrewNutRobot(viewer=False) as robot:
        monkeypatch.setattr(robot, "get_measured_gripper_gap", lambda: 0.005)
        with pytest.raises(DemoError, match="闭环夹持拦截：实测夹持开度过小"):
            run_screw_nut(robot, thread_turns=1.5)


def test_lookdown_camera_and_synthetic_rgbd_closed_loop(tmp_path) -> None:
    """Verify solve_camera_lookdown_q alignment, look-down nut/screw refinement, and post-thread verification."""
    import perception
    import sim2real
    from ufactory_mujoco.calibration import load_kinematics_calibration

    calib = load_kinematics_calibration(sim2real.DEFAULT_CALIBRATION_YAML, robot="xarm6")
    handeye = perception.load_handeye_calibration()
    x_fc = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    q_seed = np.array([0.0, -0.55, -0.35, 0.0, 0.90, 0.0], dtype=np.float64)

    # 1. Verify solve_camera_lookdown_q aligns optical center directly over target XY at z_cam = 250 mm
    target_xy = (0.354, -0.011)
    q_cam = sim2real.solve_camera_lookdown_q(
        target_xy,
        z_cam_m=0.250,
        q_seed=q_seed,
        calibration=calib,
    )
    with ScrewNutRobot(viewer=False, calibration=calib) as sim_robot:
        sim_robot.data.qpos[:6] = q_cam
        mujoco.mj_forward(sim_robot.model, sim_robot.data)
        link6_id = mujoco.mj_name2id(sim_robot.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        t_bf = np.eye(4, dtype=np.float64)
        t_bf[:3, :3] = sim_robot.data.xmat[link6_id].reshape(3, 3)
        t_bf[:3, 3] = sim_robot.data.xpos[link6_id]
        t_bc = t_bf @ x_fc
        assert np.allclose(t_bc[:3, 3], [0.354, -0.011, 0.250], atol=0.002)

    # 2. Synthetic look-down RGB-D test for detect_nut_near, detect_screw_near, and verify_threaded_nut
    h, w = 800, 1280
    fx, fy, cx, cy = 612.5, 612.7, 633.2, 399.8
    k_mat = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    t_base_flange = np.eye(4, dtype=np.float64)
    x_flange_cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.355],
            [1.0, 0.0, 0.0, -0.055],
            [0.0, 0.0, -1.0, 0.250],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    handeye_file = tmp_path / "handeye_screw_nut.yaml"
    handeye_file.write_text(
        yaml.safe_dump(
            {
                "camera": {
                    "K": k_mat,
                    "distortion": [0.0] * 8,
                    "model": "pinhole",
                    "width": w,
                    "height": h,
                },
                "extrinsics": {"X_flange_cam": x_flange_cam.tolist()},
            }
        ),
        encoding="utf-8",
    )

    color_bgr = np.full((h, w, 3), (120, 140, 40), dtype=np.uint8)
    depth_m = np.full((h, w), 0.250, dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w]

    # Empty mat must return None for both nut and screw
    nut_missing, nut_dbg = perception.detect_nut_near(
        color_bgr, depth_m, t_base_flange, (0.354, -0.011), handeye_yaml=handeye_file
    )
    assert nut_missing is None
    assert nut_dbg["reason"] == "nut_not_found_in_search_radius"

    screw_missing, screw_dbg = perception.detect_screw_near(
        color_bgr, depth_m, t_base_flange, (0.356, -0.103), handeye_yaml=handeye_file
    )
    assert screw_missing is None
    assert screw_dbg["reason"] == "screw_not_found_in_search_radius"

    # Synthesize nut at (0.358, -0.008) and upright screw at (0.357, -0.101) with exact perspective projection
    nut_xy_true = (0.358, -0.008)
    screw_xy_true = (0.357, -0.101)

    z_flange = 0.250 - 0.0038
    bx_flange = 0.355 - (ys - cy) * z_flange / fy
    by_flange = -0.055 + (xs - cx) * z_flange / fx
    r_flange = np.hypot(bx_flange - nut_xy_true[0], by_flange - nut_xy_true[1])
    in_flange = (r_flange >= 0.006) & (r_flange <= 0.0145)

    z_hex = 0.250 - 0.0145
    bx_hex = 0.355 - (ys - cy) * z_hex / fy
    by_hex = -0.055 + (xs - cx) * z_hex / fx
    r_hex = np.hypot(bx_hex - nut_xy_true[0], by_hex - nut_xy_true[1])
    in_hex = (r_hex >= 0.006) & (r_hex <= 0.0100)

    color_bgr[in_flange] = (180, 185, 190)
    depth_m[in_flange] = z_flange
    color_bgr[in_hex] = (180, 185, 190)
    depth_m[in_hex] = z_hex

    z_head = 0.250 - 0.0140
    bx_head = 0.355 - (ys - cy) * z_head / fy
    by_head = -0.055 + (xs - cx) * z_head / fx
    r_head = np.hypot(bx_head - screw_xy_true[0], by_head - screw_xy_true[1])
    in_screw_head = r_head <= 0.0100

    z_shaft = 0.250 - 0.0450
    bx_shaft = 0.355 - (ys - cy) * z_shaft / fy
    by_shaft = -0.055 + (xs - cx) * z_shaft / fx
    r_shaft = np.hypot(bx_shaft - screw_xy_true[0], by_shaft - screw_xy_true[1])
    in_screw_shaft = r_shaft <= 0.0060

    color_bgr[in_screw_head] = (175, 180, 185)
    depth_m[in_screw_head] = z_head
    color_bgr[in_screw_shaft] = (175, 180, 185)
    depth_m[in_screw_shaft] = z_shaft

    nut_det, _ = perception.detect_nut_near(
        color_bgr,
        depth_m,
        t_base_flange,
        (0.354, -0.011),
        exclude_xy_m=screw_xy_true,
        handeye_yaml=handeye_file,
    )
    assert nut_det is not None
    assert nut_det.xy_m[0] == pytest.approx(nut_xy_true[0], abs=0.003)
    assert nut_det.xy_m[1] == pytest.approx(nut_xy_true[1], abs=0.003)
    assert nut_det.height_mm == pytest.approx(14.5, abs=1.5)

    screw_det, _ = perception.detect_screw_near(
        color_bgr,
        depth_m,
        t_base_flange,
        (0.356, -0.103),
        exclude_xy_m=nut_xy_true,
        handeye_yaml=handeye_file,
    )
    assert screw_det is not None
    assert screw_det.xy_m[0] == pytest.approx(screw_xy_true[0], abs=0.003)
    assert screw_det.xy_m[1] == pytest.approx(screw_xy_true[1], abs=0.003)
    assert screw_det.height_mm == pytest.approx(45.0, abs=1.5)

    # 3. Synthesize post-threading state: nut removed from initial site and seated on screw at z_bot=14mm (z_top=28.5mm)
    color_after = np.full((h, w, 3), (120, 140, 40), dtype=np.uint8)
    depth_after = np.full((h, w), 0.250, dtype=np.float32)
    z_threaded = 0.250 - 0.0285
    bx_threaded = 0.355 - (ys - cy) * z_threaded / fy
    by_threaded = -0.055 + (xs - cx) * z_threaded / fx
    r_threaded = np.hypot(bx_threaded - screw_xy_true[0], by_threaded - screw_xy_true[1])
    in_threaded_nut = (r_threaded >= 0.006) & (r_threaded <= 0.0145)
    color_after[in_threaded_nut] = (180, 185, 190)
    depth_after[in_threaded_nut] = z_threaded
    color_after[in_screw_shaft] = (175, 180, 185)
    depth_after[in_screw_shaft] = z_shaft

    ver_ok, ver_dbg = perception.verify_threaded_nut(
        color_after,
        depth_after,
        t_base_flange,
        screw_xy_true,
        initial_nut_xy_m=nut_xy_true,
        expected_nut_bottom_z_mm=14.0,
        handeye_yaml=handeye_file,
    )
    assert ver_ok.ok, f"Post-thread verification failed: {ver_ok.reason}"
    assert ver_ok.cleared_initial_nut_site is True
    assert ver_ok.xy_error_mm < 3.0
    assert ver_ok.nut_top_height_mm == pytest.approx(28.5, abs=2.0)

    after_png = perception.save_post_thread_overlay(
        color_after, ver_ok, ver_dbg, tmp_path / "after_thread.png"
    )
    assert after_png.is_file()


def test_z_axis_closed_loop_adaptation_under_slip() -> None:
    """Verify tactile gripper-gap Z-axis adaptation and adaptive continuation strokes under 25% axial slip."""
    import sim2real
    from ufactory_mujoco.calibration import load_kinematics_calibration

    wps = ScrewNutWaypoints()
    with ScrewNutRobot(viewer=False) as robot:
        metrics = run_screw_nut(
            robot,
            thread_turns=10.0,
            simulated_slip_ratio=0.25,
            recorded_waypoints=wps,
        )

    assert metrics["z_adapt_count"] >= 2
    assert metrics["z_adapt_total_mm"] >= 7.0
    assert metrics["adaptive_extra_strokes"] >= 1
    assert metrics["stroke_count"] == metrics["nominal_stroke_count"] + metrics["adaptive_extra_strokes"]
    assert all(12.0 <= g <= demo.MAX_HEX_FLAT_GAP_MM for g in metrics["regrip_gaps_mm"])
    assert abs(metrics["nut_z_mm"] - SCREW_HEAD_LEN_M * 1000.0) < 1.2
    assert metrics["xy_error_mm"] < 1.5
    assert metrics["joint_limits"]["within_limits"] is True

    # Also verify sim2real._shift_waypoints_cartesian_z shifts TCP Z accurately while preserving J6
    calib = load_kinematics_calibration(sim2real.DEFAULT_CALIBRATION_YAML, robot="xarm6")
    checker = sim2real.make_screw_nut_collision_checker(demo.SCENE, calibration=calib)
    stroke0 = wps.stroke_waypoints[0]
    shifted = sim2real._shift_waypoints_cartesian_z(checker, stroke0, 0.0036)
    assert len(shifted) == len(stroke0)
    for q_orig, q_new in zip(stroke0, shifted, strict=True):
        assert np.isclose(q_orig[5], q_new[5], atol=1e-7)
        z0 = sim2real._get_waypoint_tcp_z(checker, q_orig)
        z1 = sim2real._get_waypoint_tcp_z(checker, q_new)
        assert abs((z1 - z0) - 0.0036) < 2e-4

