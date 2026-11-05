"""Physical acceptance and failure-path tests for multi-cup pick-and-stack in MuJoCo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack_cups as demo


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


@pytest.fixture
def robot():
    with demo.StackCupRobot(viewer=False) as instance:
        yield instance


def test_headless_two_cups_pick_and_stack(robot):
    result = demo.run_stack_cups(robot, num_cups=2)
    assert result["num_cups"] == 2
    assert result["lift_mm"] > 50.0
    assert result["error_mm"] <= 30.0
    assert result["coaxial_error_mm"] <= 8.0
    assert result["tilt_deg"] <= 10.0
    assert result["stable_s"] >= 1.0
    assert result["drift_mm"] <= 2.0
    assert len(result["nested_steps_mm"]) == 1
    assert 4.0 <= result["nested_steps_mm"][0] <= 35.0
    assert result["max_grasp_shift_mm"] < demo.MAX_GRASP_SHIFT_M * 1000.0
    assert result["max_grasp_rotation_deg"] < demo.MAX_GRASP_ROTATION_DEG
    assert result["max_contact_loss_s"] <= demo.MAX_CONTACT_LOSS_S

    assert demo._contact_between(robot, "table", "cup_1_bottom")
    assert demo.bodies_in_contact(robot.model, robot.data, "paper_cup_1", "paper_cup_2")
    for cup in ("paper_cup_1", "paper_cup_2"):
        assert not demo.bodies_in_contact(robot.model, robot.data, "left_finger", cup)
        assert not demo.bodies_in_contact(robot.model, robot.data, "right_finger", cup)
    assert all(int(kind) != int(mujoco.mjtEq.mjEQ_WELD) for kind in robot.model.eq_type)


def test_headless_three_cups_pick_and_stack(robot):
    result = demo.run_stack_cups(robot, num_cups=3)
    assert result["num_cups"] == 3
    assert result["error_mm"] <= 30.0
    assert result["coaxial_error_mm"] <= 8.0
    assert result["tilt_deg"] <= 10.0
    assert result["stable_s"] >= 1.0
    assert len(result["nested_steps_mm"]) == 2
    for step_mm in result["nested_steps_mm"]:
        assert 4.0 <= step_mm <= 35.0
    assert demo._contact_between(robot, "table", "cup_1_bottom")
    assert demo.bodies_in_contact(robot.model, robot.data, "paper_cup_1", "paper_cup_2")
    assert demo.bodies_in_contact(robot.model, robot.data, "paper_cup_2", "paper_cup_3")


def test_cannot_lift_without_bilateral_contact(robot, monkeypatch):
    monkeypatch.setattr(robot, "bilateral_finger_contact", lambda _: False)
    with pytest.raises(demo.DemoError, match="无双侧指垫接触"):
        demo.run_stack_cups(robot, num_cups=2)
    bottom, _ = demo._cup_pose(robot, "paper_cup_1")
    assert bottom[2] < 0.04
    assert robot.grasp_monitor is None


def test_lost_contact_during_lift_aborts(robot, monkeypatch):
    actual_contact = robot.bilateral_finger_contact
    monkeypatch.setattr(
        robot,
        "bilateral_finger_contact",
        lambda name: actual_contact(name) if robot.grasp_monitor is None else False,
    )
    with pytest.raises(demo.DemoError, match="夹持中滑脱：双侧接触中断"):
        demo.run_stack_cups(robot, num_cups=2)
    assert robot.grasp_monitor.max_lost_s > demo.MAX_CONTACT_LOSS_S


def test_excess_relative_motion_aborts(robot, monkeypatch):
    monitor = demo.GraspMonitor.start(robot, "paper_cup_1")
    monkeypatch.setattr(robot, "bilateral_finger_contact", lambda _: True)
    monkeypatch.setattr(
        demo,
        "_tool_relative_cup",
        lambda _r, _c: (
            monitor.position + np.array([demo.MAX_GRASP_SHIFT_M + 0.002, 0.0, 0.0]),
            monitor.axis,
        ),
    )
    with pytest.raises(demo.DemoError, match="夹持中滑脱：相对位移"):
        monitor.observe(robot)


def test_unseated_stack_fails_verification(robot):
    # Place cup 1 upright at target_xy, but keep cup 2 separated on the table
    target = np.array([0.30, 0.16], dtype=np.float64)
    j1 = robot.model.joint("paper_cup_1_joint").id
    adr1 = int(robot.model.jnt_qposadr[j1])
    robot.data.qpos[adr1 : adr1 + 3] = [0.30, 0.16, 0.0]
    robot.data.qpos[adr1 + 3 : adr1 + 7] = [1.0, 0.0, 0.0, 0.0]

    j2 = robot.model.joint("paper_cup_2_joint").id
    adr2 = int(robot.model.jnt_qposadr[j2])
    robot.data.qpos[adr2 : adr2 + 3] = [0.30, 0.16, 0.12]
    robot.data.qpos[adr2 + 3 : adr2 + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(robot.model, robot.data)

    with pytest.raises(demo.DemoError, match="堆叠检查失败"):
        demo._verify_stack_rest(
            robot,
            ["paper_cup_1", "paper_cup_2"],
            target_xy=target,
            duration=0.05,
        )


def test_config_and_cli_options(tmp_path, monkeypatch):
    default_cfg = demo.load_config()
    assert default_cfg.get("headless") is False
    assert default_cfg.get("task", {}).get("num_cups") == 2

    # Invalid --cup-poses count must fail with nonzero exit code
    assert demo.main(["--headless", "--num-cups", "2", "--cup-poses", "0.21", "-0.13"]) == 1

    # Valid config with headless: true and report output
    report_file = tmp_path / "report.json"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "mode": "sim",
                "headless": True,
                "task": {"num_cups": 2, "target_xy": [0.30, 0.16], "report": str(report_file)},
            }
        ),
        encoding="utf-8",
    )
    observed_viewer: list[bool] = []
    orig_init = demo.StackCupRobot.__init__

    def spy_init(self, *, viewer: bool, calibration=None):
        observed_viewer.append(viewer)
        orig_init(self, viewer=False, calibration=calibration)

    monkeypatch.setattr(demo.StackCupRobot, "__init__", spy_init)
    monkeypatch.setattr(
        demo,
        "run_stack_cups",
        lambda *_a, **_kw: {"num_cups": 2, "error_mm": 1.2, "nested_steps_mm": [9.6]},
    )

    assert demo.main(["--config", str(cfg_file)]) == 0
    assert observed_viewer == [False]
    assert report_file.is_file()
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["num_cups"] == 2


def test_perception_manual_multi_cup_validation():
    import perception

    estimates = perception.pose_estimates_from_manual(
        [0.54, -0.16, -140.0, 0.50, 0.06, -60.0],
        num_cups=2,
    )
    assert len(estimates) == 2
    assert estimates[0].source == "manual"
    assert estimates[0].cup_bottom_xy_m == pytest.approx((0.54, -0.16))
    assert estimates[1].cup_bottom_xy_m == pytest.approx((0.50, 0.06))

    # Out of workspace bounds must fail
    with pytest.raises(perception.PerceptionError, match="超出安全净空工作区"):
        perception.pose_estimates_from_manual(
            [0.54, -0.16, 0.0, 0.45, 0.25, 0.0],
            num_cups=2,
        )

    # Overlapping cups must fail separation check
    with pytest.raises(perception.PerceptionError, match="中心间距过近"):
        perception.pose_estimates_from_manual(
            [0.45, -0.10, 0.0, 0.47, -0.09, 10.0],
            num_cups=2,
        )


def test_perception_detects_multiple_sideways_cups_from_synthetic_rgbd(tmp_path):
    import perception

    h, w = 800, 1280
    fx, fy, cx, cy = 612.5, 612.7, 633.2, 399.8
    k_mat = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    t_base_flange = np.eye(4, dtype=np.float64)
    x_flange_cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.44],
            [1.0, 0.0, 0.0, -0.05],
            [0.0, 0.0, -1.0, 0.50],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    handeye_file = tmp_path / "handeye.yaml"
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
    depth_m = np.full((h, w), 0.50, dtype=np.float32)

    ys, xs = np.mgrid[0:h, 0:w]
    base_x = 0.44 - (ys - cy) * 0.50 / fy
    base_y = -0.05 + (xs - cx) * 0.50 / fx

    # Synthesize two sideways cups at (0.44, -0.11, yaw=30 deg) and (0.44, 0.02, yaw=-30 deg)
    synth_cups = [
        (0.44, -0.11, 30.0),
        (0.44, 0.02, -30.0),
    ]
    half_len = 0.044
    for cx_m, cy_m, yaw_deg in synth_cups:
        yaw_rad = np.deg2rad(yaw_deg)
        u_axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
        v_axis = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])
        dx = base_x - cx_m
        dy = base_y - cy_m
        s_along = dx * u_axis[0] + dy * u_axis[1]
        s_ortho = dx * v_axis[0] + dy * v_axis[1]
        frac = np.clip((s_along + half_len) / (2 * half_len), 0.0, 1.0)
        radius_s = 0.0265 + frac * (0.0375 - 0.0265)
        in_cup = (np.abs(s_along) <= half_len) & (np.abs(s_ortho) <= radius_s)
        cup_z = radius_s + np.sqrt(np.maximum(0.0, radius_s**2 - s_ortho**2))
        color_bgr[in_cup] = (238, 242, 245)
        depth_m[in_cup] = (0.50 - cup_z[in_cup]).astype(np.float32)

    estimates, debug = perception.detect_sideways_cups(
        color_bgr,
        depth_m,
        t_base_flange,
        num_cups=2,
        handeye_yaml=handeye_file,
    )
    assert len(estimates) == 2
    # Ordered by bottom Y ascending: Cup #1 is at y ~ -0.11, Cup #2 is at y ~ +0.02
    assert estimates[0].cup_center_xy_m[0] == pytest.approx(0.44, abs=0.010)
    assert estimates[0].cup_center_xy_m[1] == pytest.approx(-0.11, abs=0.012)
    assert estimates[0].yaw_deg == pytest.approx(30.0, abs=6.0)
    assert estimates[1].cup_center_xy_m[0] == pytest.approx(0.44, abs=0.010)
    assert estimates[1].cup_center_xy_m[1] == pytest.approx(0.02, abs=0.012)
    assert estimates[1].yaw_deg == pytest.approx(-30.0, abs=6.0)

    vis_path = perception.save_multi_detection_overlay(
        color_bgr,
        estimates,
        debug,
        tmp_path / "multi_vis.png",
        target_xy_m=(0.36, -0.06),
    )
    assert vis_path.is_file()


def test_sim2real_target_selection_and_safety_gates():
    import perception
    import sim2real
    from ufactory_mujoco.safety import SafetyGateError, enforce_single_cycle

    cups = perception.pose_estimates_from_manual(
        [0.568, -0.168, -142.2, 0.503, 0.081, -61.0],
        num_cups=2,
    )
    chosen = sim2real.select_safe_stack_target_xy(cups)
    assert np.allclose(chosen, [0.36, -0.06])

    # Explicit target too close to one of the cups must fail closed
    with pytest.raises(SafetyGateError, match="距离过近"):
        sim2real.select_safe_stack_target_xy(cups, requested_target_xy=(0.52, -0.18))

    # Multi-cycle hardware request must fail closed
    with pytest.raises(SafetyGateError, match="single-cycle"):
        enforce_single_cycle(2, "real")


def test_verify_stacked_cups_closed_loop(tmp_path):
    import perception

    h, w = 800, 1280
    fx, fy, cx, cy = 612.5, 612.7, 633.2, 399.8
    k_mat = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    t_base_flange = np.eye(4, dtype=np.float64)
    x_flange_cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.44],
            [1.0, 0.0, 0.0, -0.05],
            [0.0, 0.0, -1.0, 0.50],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    handeye_file = tmp_path / "handeye.yaml"
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
    depth_m = np.full((h, w), 0.50, dtype=np.float32)
    target_xy = (0.36, -0.06)

    # Empty table must fail post-stack verification
    v_empty, _ = perception.verify_stacked_cups(
        color_bgr,
        depth_m,
        t_base_flange,
        target_xy,
        num_cups=2,
        handeye_yaml=handeye_file,
    )
    assert not v_empty.ok

    # Synthesize 2-cup nested stack at (0.36, -0.06) with top rim height ~ 97.6 mm (depth = 0.4024 m)
    ys, xs = np.mgrid[0:h, 0:w]
    base_x = 0.44 - (ys - cy) * 0.4024 / fy
    base_y = -0.05 + (xs - cx) * 0.4024 / fx
    in_stack = np.hypot(base_x - target_xy[0], base_y - target_xy[1]) <= 0.0375
    color_bgr[in_stack] = (238, 242, 245)
    depth_m[in_stack] = 0.4024

    v_ok, dbg = perception.verify_stacked_cups(
        color_bgr,
        depth_m,
        t_base_flange,
        target_xy,
        num_cups=2,
        handeye_yaml=handeye_file,
    )
    assert v_ok.ok
    assert v_ok.xy_error_mm < 15.0
    assert v_ok.rim_height_mm >= 90.0
    after_img = perception.save_post_stack_overlay(
        color_bgr,
        v_ok,
        dbg,
        tmp_path / "after_stack.png",
    )
    assert after_img.is_file()


def test_trajectory_smooth_deceleration_and_continuity():
    import perception
    import sim2real
    from ufactory_mujoco.calibration import load_kinematics_calibration

    calib = load_kinematics_calibration(sim2real.DEFAULT_CALIBRATION_YAML, robot="xarm6")
    cups = perception.pose_estimates_from_manual(
        [0.5682, -0.1669, -142.19, 0.5026, 0.0815, -61.32],
        num_cups=2,
    )
    q_obs = np.array([0.0, -0.55, -0.35, 0.0, 0.90, 0.0], dtype=np.float64)
    report = sim2real.run_digital_twin_preflight(
        cups,
        np.array([0.36, -0.06], dtype=np.float64),
        q_init=q_obs,
        calibration=calib,
        headless=True,
    )
    assert report.ok, f"Digital twin preflight failed: {report.failures}"

    dt = 0.02
    q_cur = q_obs.copy()
    merged_retreat_approach = list(report.cycles[0].retreat_waypoints) + list(
        report.cycles[1].approach_waypoints
    )
    phases = [
        ("cup1_approach", report.cycles[0].approach_waypoints, True),
        ("cup1_lift_flip_place", report.cycles[0].lift_flip_place_waypoints, True),
        ("cup1_retreat_to_cup2_grasp", merged_retreat_approach, True),
        ("cup2_lift_flip_place", report.cycles[1].lift_flip_place_waypoints, True),
        ("cup2_retreat_to_obs", report.cycles[1].retreat_waypoints, False),
    ]

    for name, wps, soft_land in phases:
        seg = sim2real._interpolate_waypoint_chain(
            q_cur, wps, dt=dt, soft_landing=soft_land
        )
        qd = np.diff(seg, axis=0) / dt
        max_qd = np.max(np.abs(qd), axis=1)
        qdd = np.diff(qd, axis=0) / dt
        max_qdd = np.max(np.abs(qdd), axis=1)

        # Zero initial & final velocity, bounded acceleration (<= 3.5 rad/s^2)
        assert max_qd[0] < 1e-4, f"{name} start velocity not zero: {max_qd[0]}"
        assert max_qd[-1] < 1e-4, f"{name} end velocity not zero: {max_qd[-1]}"
        assert float(np.max(max_qdd)) <= 3.5, f"{name} peak qdd too high: {np.max(max_qdd)}"
        assert np.allclose(seg[-1], wps[-1], atol=1e-6), f"{name} endpoint mismatch"

        if soft_land:
            # Last 5 velocity samples must be < 0.025 rad/s and monotonically non-increasing
            last5 = max_qd[-5:]
            assert float(np.max(last5)) < 0.025, f"{name} soft-landing too fast: {last5}"
            assert np.all(np.diff(last5) <= 1e-5), f"{name} not monotonically decelerating: {last5}"

        q_cur = seg[-1].copy()


def test_detect_single_cup_near_present_and_missing(tmp_path):
    """Verify local look-down cup refinement when present/shifted and None when missing."""
    import perception
    import yaml

    h, w = 800, 1280
    fx, fy, cx, cy = 612.5, 612.7, 633.2, 399.8
    k_mat = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    t_base_flange = np.eye(4, dtype=np.float64)
    x_flange_cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.48],
            [1.0, 0.0, 0.0, 0.05],
            [0.0, 0.0, -1.0, 0.28],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    handeye_file = tmp_path / "handeye_local.yaml"
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
    depth_m = np.full((h, w), 0.28, dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    base_x = 0.48 - (ys - cy) * 0.28 / fy
    base_y = 0.05 + (xs - cx) * 0.28 / fx

    # Case 1: Empty table (cup was removed!) -> must return None
    est_missing, dbg_missing = perception.detect_single_cup_near(
        color_bgr,
        depth_m,
        t_base_flange,
        expected_xy_m=(0.48, 0.05),
        handeye_yaml=handeye_file,
    )
    assert est_missing is None
    assert dbg_missing["reason"] == "cup_not_found_in_search_radius"

    # Case 2: Cup shifted by (+35 mm, -25 mm) and rotated to 75 deg -> must refine pose
    cx_m, cy_m, yaw_deg = 0.515, 0.025, 75.0
    half_len = 0.044
    yaw_rad = np.deg2rad(yaw_deg)
    u_axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
    v_axis = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])
    dx = base_x - cx_m
    dy = base_y - cy_m
    s_along = dx * u_axis[0] + dy * u_axis[1]
    s_ortho = dx * v_axis[0] + dy * v_axis[1]
    frac = np.clip((s_along + half_len) / (2 * half_len), 0.0, 1.0)
    radius_s = 0.0265 + frac * (0.0375 - 0.0265)
    in_cup = (np.abs(s_along) <= half_len) & (np.abs(s_ortho) <= radius_s)
    cup_z = radius_s + np.sqrt(np.maximum(0.0, radius_s**2 - s_ortho**2))
    color_bgr[in_cup] = (238, 242, 245)
    depth_m[in_cup] = (0.28 - cup_z[in_cup]).astype(np.float32)

    est_present, dbg_present = perception.detect_single_cup_near(
        color_bgr,
        depth_m,
        t_base_flange,
        expected_xy_m=(0.48, 0.05),
        handeye_yaml=handeye_file,
    )
    assert est_present is not None
    assert dbg_present["reason"] == "ok"
    assert est_present.cup_center_xy_m[0] == pytest.approx(cx_m, abs=0.008)
    assert est_present.cup_center_xy_m[1] == pytest.approx(cy_m, abs=0.008)
    assert est_present.yaw_deg == pytest.approx(yaw_deg, abs=6.0)


def test_solve_camera_lookdown_q_alignment():
    """Verify solve_camera_lookdown_q aligns the Eye-in-Hand camera optical center over target XY."""
    import mujoco
    import sim2real
    import stack_cups as demo
    from perception import load_handeye_calibration
    from ufactory_mujoco.calibration import load_kinematics_calibration

    calib = load_kinematics_calibration(sim2real.DEFAULT_CALIBRATION_YAML, robot="xarm6")
    handeye = load_handeye_calibration()
    x_fc = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    q_seed = np.array([0.0, -0.55, -0.35, 0.0, 0.90, 0.0], dtype=np.float64)

    target_cup_xy = (0.490, 0.075)
    q_cam = sim2real.solve_camera_lookdown_q(
        target_cup_xy,
        z_cam_m=0.280,
        q_seed=q_seed,
        calibration=calib,
    )
    with demo.StackCupRobot(viewer=False, calibration=calib) as sim_robot:
        sim_robot.data.qpos[:6] = q_cam
        mujoco.mj_forward(sim_robot.model, sim_robot.data)
        link6_id = mujoco.mj_name2id(sim_robot.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        t_bf = np.eye(4, dtype=np.float64)
        t_bf[:3, :3] = sim_robot.data.xmat[link6_id].reshape(3, 3)
        t_bf[:3, 3] = sim_robot.data.xpos[link6_id]
        t_bc = t_bf @ x_fc
        assert np.allclose(t_bc[:3, 3], [0.490, 0.075, 0.280], atol=0.002)



