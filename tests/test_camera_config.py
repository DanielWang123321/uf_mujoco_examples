# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Backend/configuration precedence, calibration provenance, and distortion checks."""

import importlib
import json
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from ufactory_mujoco.calibration import (
    apply_calibration_to_model,
    flange_m_quat_to_tcp_mm_rpy,
    load_kinematics_calibration,
)
from ufactory_mujoco.safety import SafetyGateError

from common.cli import build_parser, resolve_run_options
from common.paths import DEFAULT_HANDEYE_YAML, ORBBEC_HANDEYE_YAML, SAMPLE_CALIBRATION_YAML
from common.sim2real import (
    camera_run_context,
    flange_pose,
    pipeline_error_report,
    record_camera_frame,
    verify_controller_tcp_offset_and_fk,
)
from common.vision import (
    PerceptionError,
    deproject_pixels_to_cam,
    project_cam_to_pixels,
    resolve_camera_config,
)


def test_options_precedence_and_handeye_selection(tmp_path):
    path = tmp_path / "config.yaml"
    parser = build_parser(None, default_config=path)
    opts = resolve_run_options(parser.parse_args([]))
    assert opts.camera_type == "realsense" and opts.handeye_path == str(DEFAULT_HANDEYE_YAML)
    path.write_text(yaml.safe_dump({"camera": {"type": "orbbec", "serial": "OB-SDK"}}))
    opts = resolve_run_options(parser.parse_args([]))
    assert opts.camera_type == "orbbec" and opts.handeye_path == str(ORBBEC_HANDEYE_YAML)
    opts = resolve_run_options(
        parser.parse_args(["--camera-type", "realsense", "--camera-serial", "RS-SDK"])
    )
    assert opts.handeye_path == str(DEFAULT_HANDEYE_YAML) and opts.camera_serial == "RS-SDK"
    path.write_text(yaml.safe_dump({"camera": {"type": "orbbec", "handeye": "explicit.yaml"}}))
    opts = resolve_run_options(parser.parse_args(["--camera-type", "realsense"]))
    assert opts.handeye_path.endswith("/explicit.yaml")
    opts = resolve_run_options(parser.parse_args(["--handeye", str(tmp_path / "cli.yaml")]))
    assert opts.handeye_path == str(tmp_path / "cli.yaml")
    path.write_text("camera:\n  type: unsupported\n")
    with pytest.raises(ValueError, match="Unsupported camera"):
        resolve_run_options(parser.parse_args([]))


@pytest.mark.parametrize("task", ["pick_cup", "stack_cups", "screw_nut"])
def test_task_cli_forwards_camera_options(task, tmp_path, monkeypatch):
    demo = importlib.import_module(f"examples.{task}.{task}")
    pipeline = importlib.import_module(f"examples.{task}.sim2real")
    calls = []
    monkeypatch.setattr(pipeline, "run_sim2real_pipeline", lambda **kw: calls.append(kw))
    assert (
        demo.main(
            [
                "--mode",
                "dry-run",
                "--camera-type",
                "orbbec",
                "--camera-serial",
                "OB-SN",
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    assert calls[0]["camera_type"] == "orbbec" and calls[0]["camera_serial"] == "OB-SN"
    assert calls[0]["handeye_path"] == str(ORBBEC_HANDEYE_YAML)


def test_migrated_handeye_matches_reference_euler_and_hash(tmp_path):
    data = yaml.safe_load(DEFAULT_HANDEYE_YAML.read_text())
    euler = data["extrinsics"]["flange_cam_m_rad"]
    x = np.asarray(data["extrinsics"]["X_flange_cam"])
    assert np.allclose(x[:3, :3], Rotation.from_euler("xyz", euler[3:]).as_matrix(), atol=1e-9)
    assert np.allclose(x[:3, 3], euler[:3], atol=1e-9)
    assert data["extrinsics"]["tcp_offset_mm_rad"] == [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
    t_ft = np.eye(4)
    t_ft[2, 3] = 0.172
    x_tc = np.asarray(data["extrinsics"]["X_tcp_cam"])
    assert np.allclose(x_tc, np.linalg.inv(t_ft) @ x, atol=1e-9)

    ref = yaml.safe_load(
        DEFAULT_HANDEYE_YAML.with_name("eye_in_hand.d435.example.yaml").read_text()
    )
    ref_euler = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]
    x_ref = np.asarray(ref["extrinsics"]["X_flange_cam"])
    assert np.allclose(
        x_ref[:3, :3], Rotation.from_euler("xyz", ref_euler[3:]).as_matrix(), atol=1e-12
    )
    assert np.allclose(x_ref[:3, 3], ref_euler[:3], atol=1e-12)

    path = tmp_path / "handeye.yaml"
    path.write_text(DEFAULT_HANDEYE_YAML.read_text())
    _, context = camera_run_context("realsense", path)
    record_camera_frame(context, "overview", {"serial": "sdk-serial", "K": [[1, 0, 0]]})
    report = {**context}
    record_camera_frame(context, "post_task", {"frame_number": 5})
    assert report["camera"]["frame_number"] == 5 and len(report["camera_frames"]) == 2
    path.write_text(path.read_text() + "\n# modified\n")
    _, modified = camera_run_context("realsense", path)
    assert modified["handeye_sha256"] != context["handeye_sha256"]


def test_verify_controller_tcp_offset_and_fk_gate():
    from examples.pick_cup import pick_cup as task

    calib = load_kinematics_calibration(SAMPLE_CALIBRATION_YAML, robot="xarm6")
    model = mujoco.MjModel.from_xml_path(str(task.SCENE))
    data = mujoco.MjData(model)
    apply_calibration_to_model(model, calib, data)
    q = np.array([0.0, -0.55, -0.45, 0.0, 1.00, 0.0], dtype=np.float64)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)

    t_mj = flange_pose(model, data)
    mj_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(mj_quat, t_mj[:3, :3].reshape(9))
    g2_offset = [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
    sdk_tcp_172 = flange_m_quat_to_tcp_mm_rpy(t_mj[:3, 3], mj_quat, g2_offset)
    sdk_tcp_zero = flange_m_quat_to_tcp_mm_rpy(t_mj[:3, 3], mj_quat, [0.0] * 6)

    def make_robot(tcp_offset, fk_pose_out):
        arm = NS(
            tcp_offset=tcp_offset,
            tcp_load=[0.82, [0.0, 0.0, 48.0]],
            get_forward_kinematics=lambda angles, **kw: (0, fk_pose_out.tolist()),
        )
        return NS(_arm=arm, model=model, data=data, get_joint_positions=lambda: q.copy())

    checked = verify_controller_tcp_offset_and_fk(make_robot(g2_offset, sdk_tcp_172))
    assert checked["ok"] is True
    assert checked["tcp_offset_mm_rad"] == g2_offset
    assert checked["tcp_load"] == [0.82, [0.0, 0.0, 48.0]]
    assert checked["fk_flange_pos_error_mm"] < 1e-6

    with pytest.raises(SafetyGateError, match="does not match Gripper G2"):
        verify_controller_tcp_offset_and_fk(make_robot([0.0] * 6, sdk_tcp_zero))

    with pytest.raises(SafetyGateError, match="disagrees with calibrated MuJoCo link6"):
        verify_controller_tcp_offset_and_fk(make_robot(g2_offset, sdk_tcp_zero))


def test_flange_yaw_180_mount_rotates_translation_and_orientation():
    original = yaml.safe_load(
        DEFAULT_HANDEYE_YAML.with_name("eye_in_hand.d435.example.yaml").read_text()
    )
    rotated = yaml.safe_load(
        DEFAULT_HANDEYE_YAML.with_name("eye_in_hand.d435.yaw180.example.yaml").read_text()
    )
    x_ref = np.array(original["extrinsics"]["X_flange_cam"])
    x_current = np.array(rotated["extrinsics"]["X_flange_cam"])
    assert np.allclose(x_current, np.diag([-1.0, -1.0, 1.0, 1.0]) @ x_ref)
    assert np.allclose(x_current[:2, 3], -x_ref[:2, 3])
    assert x_current[2, 3] == x_ref[2, 3]


def test_live_intrinsics_override_yaml_and_validate_dimensions():
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    old = {"camera": {"width": 1280, "height": 800, "K": np.eye(3).tolist()}}
    meta = {
        "width": 640,
        "height": 480,
        "K": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
        "distortion": [0.0] * 5,
        "distortion_model": "inverse_brown_conrady",
    }
    cfg = resolve_camera_config(old, color, depth, meta)
    points = deproject_pixels_to_cam([[380, 300]], [0.5], cfg)
    assert np.allclose(points, [[0.05, 0.05, 0.5]])
    assert np.allclose(project_cam_to_pixels(points, cfg), [[380, 300]])
    d435_yaml = yaml.safe_load(DEFAULT_HANDEYE_YAML.read_text())
    offline_cfg = resolve_camera_config(
        d435_yaml,
        np.zeros((720, 1280, 3), dtype=np.uint8),
        np.ones((720, 1280), dtype=np.float32),
    )
    assert offline_cfg["type"] == "realsense"
    with pytest.raises(PerceptionError, match="dimensions"):
        resolve_camera_config(old, color, depth)
    with pytest.raises(PerceptionError, match="Incomplete"):
        resolve_camera_config(old, color, depth, {"width": 640, "height": 480})
    with pytest.raises(PerceptionError, match="intrinsic"):
        resolve_camera_config(old, color, depth, {**meta, "K": [[0, 0, 0]]})
    with pytest.raises(PerceptionError, match="Unsupported nonzero"):
        resolve_camera_config(old, color, depth, {**meta, "distortion": [0.1, 0, 0, 0, 0]})


def test_failure_report_preserves_capture_and_calibration_evidence(tmp_path):
    _, context = camera_run_context("realsense", None)
    payload = {"mode": "real", "ok": False, "completed": False, **context}
    report = tmp_path / "failure.json"
    with pytest.raises(PerceptionError, match="clipped"):
        with pipeline_error_report(report, payload):
            record_camera_frame(context, "overview", {"serial": "SDK-SN", "width": 640})
            raise PerceptionError("Cup is clipped")
    saved = json.loads(report.read_text())
    assert saved["ok"] is False and saved["completed"] is False
    assert saved["error"] == "Cup is clipped" and saved["camera"]["serial"] == "SDK-SN"
    assert saved["camera_frames"][0]["stage"] == "overview" and len(saved["handeye_sha256"]) == 64
