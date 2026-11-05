"""Physical acceptance and failure paths for the paper-cup pick-and-place example."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick_cup as demo


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
    with demo.CupRobot(viewer=False) as instance:
        yield instance


def test_headless_pick_places_and_stands_without_finger_support(robot):
    result = demo.run_pick_cup(robot)
    assert result["lift_mm"] > 50
    assert result["error_mm"] <= 30
    assert result["tilt_deg"] <= 10
    assert result["stable_s"] >= 1
    assert result["drift_mm"] <= 2
    assert result["max_grasp_shift_mm"] < demo.MAX_GRASP_SHIFT_M * 1000
    assert result["max_grasp_rotation_deg"] < demo.MAX_GRASP_ROTATION_DEG
    assert result["max_contact_loss_s"] <= demo.MAX_CONTACT_LOSS_S
    assert demo._contact_between(robot, "table", "cup_bottom")
    assert not demo.bodies_in_contact(robot.model, robot.data, "left_finger", demo.CUP)
    assert not demo.bodies_in_contact(robot.model, robot.data, "right_finger", demo.CUP)
    # Only the gripper linkage has equalities; no weld attaches the cup.
    assert all(int(kind) != int(mujoco.mjtEq.mjEQ_WELD) for kind in robot.model.eq_type)


def test_cannot_lift_without_bilateral_contact(robot, monkeypatch):
    monkeypatch.setattr(robot, "bilateral_finger_contact", lambda _: False)
    with pytest.raises(demo.DemoError, match="无双侧指垫接触"):
        demo.run_pick_cup(robot)
    bottom, _ = demo._cup_pose(robot)
    assert bottom[2] < 0.04
    assert robot.grasp_monitor is None


def test_lost_contact_during_lift_aborts(robot, monkeypatch):
    actual_contact = robot.bilateral_finger_contact
    monkeypatch.setattr(
        robot, "bilateral_finger_contact",
        lambda name: actual_contact(name) if robot.grasp_monitor is None else False,
    )
    with pytest.raises(demo.DemoError, match="夹持中滑脱：双侧接触中断"):
        demo.run_pick_cup(robot)
    assert robot.grasp_monitor.max_lost_s > demo.MAX_CONTACT_LOSS_S


def test_excess_relative_motion_aborts(robot, monkeypatch):
    monitor = demo.GraspMonitor.start(robot)
    monkeypatch.setattr(robot, "bilateral_finger_contact", lambda _: True)
    monkeypatch.setattr(
        demo, "_tool_relative_cup",
        lambda _: (monitor.position + np.array([demo.MAX_GRASP_SHIFT_M + 0.001, 0, 0]), monitor.axis),
    )
    with pytest.raises(demo.DemoError, match="夹持中滑脱：相对位移"):
        monitor.observe(robot)


def test_ik_failure_reports_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(
        demo.CupRobot, "solve_tcp",
        lambda self, *args, **kwargs: (self.data.qpos.copy(), False),
    )
    assert demo.main(["--headless"]) == 1
    assert "逆运动学不收敛" in capsys.readouterr().err


def test_waypoint_ik_failure_reports_nonzero_exit(monkeypatch, capsys):
    def failed_waypoint(_robot, *args, **kwargs):
        raise RuntimeError("IK did not converge")

    monkeypatch.setattr(demo.CupRobot, "move_tcp", failed_waypoint)
    assert demo.main(["--headless"]) == 1
    assert "逆运动学或轨迹执行失败" in capsys.readouterr().err


def test_early_viewer_close_reports_nonzero_exit(monkeypatch, capsys):
    def closed(_robot):
        raise demo.DemoError("可视化窗口提前关闭")

    monkeypatch.setattr(demo.CupRobot, "check_viewer", closed)
    assert demo.main(["--headless"]) == 1
    assert "可视化窗口提前关闭" in capsys.readouterr().err


def test_viewer_close_during_a_step_aborts(robot):
    class ClosingViewer:
        def __init__(self):
            self.checks = 0

        def is_running(self):
            self.checks += 1
            return self.checks < 2

        def sync(self):
            pass

    robot._viewer = ClosingViewer()
    try:
        with pytest.raises(demo.DemoError, match="可视化窗口提前关闭"):
            robot._step()
    finally:
        robot._viewer = None


def test_headless_mode_has_no_viewer(robot):
    assert robot._viewer is None
    assert robot.model.body(demo.CUP).mass[0] == pytest.approx(0.00574)
    joint = robot.model.joint("paper_cup_joint").id
    assert robot.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE


def test_perception_manual_pose_and_workspace_validation():
    import perception

    est = perception.pose_estimate_from_manual(0.46, -0.07, 140.0)
    assert est.source == "manual"
    assert est.cup_bottom_xy_m == pytest.approx((0.46, -0.07))
    assert est.yaw_deg == pytest.approx(140.0)
    assert len(est.mujoco_pos_m) == 3
    assert len(est.mujoco_quat_wxyz) == 4

    with pytest.raises(perception.PerceptionError, match="超出安全净空工作区"):
        perception.pose_estimate_from_manual(0.45, 0.25, 0.0)


def test_perception_detects_sideways_cup_from_synthetic_rgbd(tmp_path):
    import perception

    h, w = 800, 1280
    fx, fy, cx, cy = 612.5, 612.7, 633.2, 399.8
    k_mat = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    # Camera looking straight down from (0.44, -0.05, 0.50) m in base frame:
    # cam +X = base +Y, cam +Y = base -X, cam +Z = base -Z
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
    import yaml

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

    # Synthesize green table at Z=0 (depth=0.50m) and a white tapered sideways cup
    color_bgr = np.full((h, w, 3), (120, 140, 40), dtype=np.uint8)
    depth_m = np.full((h, w), 0.50, dtype=np.float32)

    ys, xs = np.mgrid[0:h, 0:w]
    # Base coordinates of each pixel on the Z=0 plane:
    # x_cam = (xs - cx) * 0.50 / fx, y_cam = (ys - cy) * 0.50 / fy
    # base_x = 0.44 - y_cam, base_y = -0.05 + x_cam
    base_x = 0.44 - (ys - cy) * 0.50 / fy
    base_y = -0.05 + (xs - cx) * 0.50 / fx

    # Place sideways cup centered at (0.44, -0.05) with yaw = 30 deg (bottom -> top)
    yaw_true = np.deg2rad(30.0)
    u_axis = np.array([np.cos(yaw_true), np.sin(yaw_true)])
    v_axis = np.array([-np.sin(yaw_true), np.cos(yaw_true)])
    dx = base_x - 0.44
    dy = base_y - (-0.05)
    s_along = dx * u_axis[0] + dy * u_axis[1]  # in [-0.044, +0.044]
    s_ortho = dx * v_axis[0] + dy * v_axis[1]

    half_len = 0.044
    # Radius tapers linearly from 0.0265m at s=-0.044 (bottom) to 0.0375m at s=+0.044 (top)
    frac = np.clip((s_along + half_len) / (2 * half_len), 0.0, 1.0)
    radius_s = 0.0265 + frac * (0.0375 - 0.0265)
    in_cup = (np.abs(s_along) <= half_len) & (np.abs(s_ortho) <= radius_s)

    cup_z = radius_s + np.sqrt(np.maximum(0.0, radius_s**2 - s_ortho**2))
    color_bgr[in_cup] = (235, 240, 242)
    depth_m[in_cup] = (0.50 - cup_z[in_cup]).astype(np.float32)

    est, debug = perception.detect_sideways_cup(
        color_bgr, depth_m, t_base_flange, handeye_yaml=handeye_file
    )
    assert abs(est.table_z_m) < 0.005
    assert est.yaw_deg == pytest.approx(30.0, abs=4.0)
    assert est.cup_center_xy_m[0] == pytest.approx(0.44, abs=0.006)
    assert est.cup_center_xy_m[1] == pytest.approx(-0.05, abs=0.006)
    assert est.width_top_mm > est.width_bot_mm

    vis_out = perception.save_detection_overlay(color_bgr, est, debug, tmp_path / "vis.png")
    assert vis_out.is_file()


def test_sim_oblique_cup_pose_pick_and_place(tmp_path):
    report_file = tmp_path / "sim_oblique.json"
    code = demo.main(
        [
            "--headless",
            "--cup-pose",
            "0.46",
            "-0.07",
            "140.0",
            "--target-xy",
            "0.38",
            "-0.18",
            "--report",
            str(report_file),
        ]
    )
    assert code == 0
    assert report_file.is_file()


def test_sim2real_safety_gates():
    import perception
    import sim2real
    from ufactory_mujoco.safety import SafetyGateError, enforce_single_cycle

    est = perception.pose_estimate_from_manual(0.38, -0.18, 0.0)
    # Automatic target selection must switch to fallback when default overlaps cup
    chosen = sim2real.select_safe_target_xy(est)
    assert np.allclose(chosen, sim2real.FALLBACK_REAL_TARGET_XY)

    # Explicit target too close to cup must fail closed
    with pytest.raises(SafetyGateError, match="距离过近"):
        sim2real.select_safe_target_xy(est, requested_target_xy=(0.39, -0.17))

    # Multi-cycle hardware request must fail closed
    with pytest.raises(SafetyGateError, match="single-cycle"):
        enforce_single_cycle(2, "real")


def test_config_yaml_headless_and_viewer_defaults(tmp_path, monkeypatch):
    import yaml

    default_cfg = demo.load_config()
    assert default_cfg.get("headless") is False
    assert default_cfg.get("task", {}).get("post_verify") is True

    # Setting headless: true in config file must run without opening a viewer
    cfg_file = tmp_path / "headless_config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({"mode": "sim", "headless": True}),
        encoding="utf-8",
    )
    observed_viewer: list[bool] = []
    orig_init = demo.CupRobot.__init__

    def spy_init(self, *, viewer: bool, calibration=None):
        observed_viewer.append(viewer)
        orig_init(self, viewer=False, calibration=calibration)

    monkeypatch.setattr(demo.CupRobot, "__init__", spy_init)
    monkeypatch.setattr(demo, "run_pick_cup", lambda *_a, **_kw: {"error_mm": 0.0})

    assert demo.main(["--config", str(cfg_file)]) == 0
    assert observed_viewer == [False]

    # Default config (headless: false) must request viewer=True unless --headless is passed
    observed_viewer.clear()
    assert demo.main([]) == 0
    assert observed_viewer == [True]


def test_verify_upright_cup_closed_loop(tmp_path):
    import perception
    import yaml

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
    target_xy = (0.38, -0.18)

    # Empty table must fail post-placement check
    v_empty, _ = perception.verify_upright_cup(
        color_bgr, depth_m, t_base_flange, target_xy, handeye_yaml=handeye_file
    )
    assert not v_empty.ok

    # Synthesize upright paper cup at (0.38, -0.18) with radius 37.5 mm and height 88 mm
    ys, xs = np.mgrid[0:h, 0:w]
    base_x = 0.44 - (ys - cy) * 0.412 / fy
    base_y = -0.05 + (xs - cx) * 0.412 / fx
    in_upright = np.hypot(base_x - target_xy[0], base_y - target_xy[1]) <= 0.0375
    color_bgr[in_upright] = (235, 240, 242)
    depth_m[in_upright] = 0.412  # Z = +0.088 m

    v_ok, dbg = perception.verify_upright_cup(
        color_bgr, depth_m, t_base_flange, target_xy, handeye_yaml=handeye_file
    )
    assert v_ok.ok
    assert v_ok.xy_error_mm < 15.0
    assert v_ok.rim_height_mm >= 80.0
    after_img = perception.save_post_place_overlay(
        color_bgr, v_ok, dbg, tmp_path / "after_verify.png"
    )
    assert after_img.is_file()


