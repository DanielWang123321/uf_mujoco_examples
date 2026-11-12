# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Single-cup near measurement must replan, or retreat before any grasp."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from common.paths import DEFAULT_HANDEYE_YAML, REPO_ROOT, SAMPLE_CALIBRATION_YAML
from common.sim2real import camera_run_context
from examples.pick_cup import perception, sim2real


@pytest.mark.parametrize("outcome", ["success", "missing", "failed_preflight"])
def test_near_measurement_uses_live_metadata_and_retreats_on_failure(monkeypatch, outcome):
    initial = perception.pose_estimate_from_manual(0.40, -0.04, 0.0)
    refined = perception.pose_estimate_from_manual(0.41, -0.04, 2.0)
    q_obs, q_hover = np.zeros(6), np.ones(6) * 0.1
    calibration = sim2real.load_kinematics_calibration(SAMPLE_CALIBRATION_YAML, robot="xarm6")
    robot = NS(model=None, data=None, get_joint_positions=lambda: q_hover.copy())
    metadata = {
        "width": 640,
        "height": 480,
        "K": [[607, 0, 317], [0, 607, 238], [0, 0, 1]],
        "distortion": [0.0] * 5,
        "distortion_model": "inverse_brown_conrady",
        "serial": "SDK",
    }
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 0.28, dtype=np.float32)
    _, context = camera_run_context("realsense", None)
    payload, motions, captured, preflights = {}, [], [], []
    persistent_stream = object()
    lookdown_z = []
    monkeypatch.setattr(
        sim2real,
        "solve_camera_lookdown_q",
        lambda *a, **kw: (lookdown_z.append(kw.get("z_cam_m")), q_hover)[1],
    )
    monkeypatch.setattr(
        sim2real, "_checked_hover_stream", lambda *a, **kw: np.array([q_obs, q_hover])
    )

    def fake_stream(_robot, points, *, soft_landing=False, **_kwargs):
        motions.append((points, soft_landing))
        return np.asarray(points)

    monkeypatch.setattr(sim2real, "stream_waypoints_on_real", fake_stream)
    monkeypatch.setattr(sim2real, "flange_pose", lambda *a: np.eye(4))

    def capture(**kwargs):
        captured.append(kwargs)
        return color, depth, metadata

    def detect(*args, **kwargs):
        assert args[0] is color and args[1] is depth
        assert kwargs["camera_metadata"] is metadata
        return (None if outcome == "missing" else refined), {"reason": "missing"}

    twin = NS(
        ok=outcome != "failed_preflight",
        failures=("collision",),
        as_dict=lambda: {"ok": outcome != "failed_preflight"},
        shadow_stream_rad=np.zeros((2, 6)),
    )

    def preflight(pose, target, **kwargs):
        preflights.append((pose, kwargs))
        return twin

    monkeypatch.setattr(sim2real, "capture_rgbd_frame", capture)
    monkeypatch.setattr(sim2real, "detect_single_cup_near", detect)
    monkeypatch.setattr(sim2real, "run_digital_twin_preflight", preflight)

    def run():
        return sim2real._refine_cup_before_grasp(
            robot,
            initial,
            np.array([0.38, -0.18]),
            q_obs=q_obs,
            calibration=calibration,
            handeye_path=DEFAULT_HANDEYE_YAML,
            camera_type="realsense",
            camera_serial="SDK",
            cam_stream=persistent_stream,
            camera_context=context,
            save_vis_path=None,
            result_payload=payload,
        )

    if outcome == "success":
        pose, report = run()
        assert pose is refined and report is twin and len(motions) == 1
        assert motions[0][1] is True
        assert payload["refinement_xy_mm"] == pytest.approx(10.0, abs=0.2)
        assert len(payload["refined_program_sha256"]) == 64
    else:
        with pytest.raises((sim2real.SafetyGateError, perception.PerceptionError)):
            run()
        assert len(motions) == 2 and motions[-1][1] is True
        assert np.array_equal(motions[-1][0][0], q_obs)
    assert lookdown_z == [0.360]
    assert captured[0]["stream"] is persistent_stream
    assert captured[0]["camera_type"] == "realsense" and captured[0]["serial"] == "SDK"
    assert context["camera_frames"][0]["stage"] == "near_cup"
    assert context["camera_frames"][0]["T_base_flange"] == np.eye(4).tolist()
    if outcome != "missing":
        assert preflights[0][0] is refined
        assert np.array_equal(preflights[0][1]["q_init"], q_hover)
        assert np.array_equal(preflights[0][1]["q_return"], q_obs)
    else:
        assert not preflights


def test_target_cylinder_in_camera_view_distinguishes_high_and_low_observation():
    from common.vision import target_cylinder_in_camera_view

    t_high = np.diag([1.0, -1.0, -1.0, 1.0])
    t_high[:3, 3] = [0.30, 0.0, 0.572]  # TCP Z = 400 mm (flange Z = 572 mm)
    t_low = np.diag([1.0, -1.0, -1.0, 1.0])
    t_low[:3, 3] = [0.30, 0.0, 0.322]  # TCP Z = 150 mm (flange Z = 322 mm)

    assert target_cylinder_in_camera_view(
        t_high, (0.38, -0.18), top_z_m=0.088, min_cam_clearance_m=0.28
    )
    assert not target_cylinder_in_camera_view(
        t_low, (0.38, -0.18), top_z_m=0.088, min_cam_clearance_m=0.28
    )


def test_sim2real_modules_reference_only_existing_task_attributes():
    import ast

    from examples.pick_cup import pick_cup as pick_task
    from examples.screw_nut import screw_nut as screw_task
    from examples.stack_cups import stack_cups as stack_task

    for task_mod, rel_path in [
        (pick_task, "examples/pick_cup/sim2real.py"),
        (stack_task, "examples/stack_cups/sim2real.py"),
        (screw_task, "examples/screw_nut/sim2real.py"),
    ]:
        tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        missing = [
            (node.lineno, node.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "task"
            and not hasattr(task_mod, node.attr)
        ]
        assert not missing, f"{rel_path} references missing task attributes: {missing}"
