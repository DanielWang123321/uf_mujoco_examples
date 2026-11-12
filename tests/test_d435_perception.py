# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""1280x720 full-frame regressions with live D435i intrinsics and migrated optical-frame extrinsics."""

import importlib

import mujoco
import numpy as np
import pytest
import yaml

from common.paths import DEFAULT_HANDEYE_YAML, SAMPLE_CALIBRATION_YAML
from common.sim2real import flange_pose
from common.vision import PerceptionError
from examples.pick_cup import perception as pick
from examples.screw_nut import perception as screw
from examples.stack_cups import perception as stack


class Scene:
    def __init__(self, tmp_path, z=0.5):
        calibration = yaml.safe_load(DEFAULT_HANDEYE_YAML.read_text())
        self.meta = {
            **calibration["camera"],
            "type": "realsense",
            "distortion_model": "inverse_brown_conrady",
        }
        self.k = np.asarray(self.meta["K"])
        self.z = z
        self.xy = (0.41, -0.04)
        t_bc = np.diag([1.0, -1.0, -1.0, 1.0])
        t_bc[:3, 3] = [*self.xy, z]
        x_fc = np.asarray(calibration["extrinsics"]["X_flange_cam"])
        self.flange = t_bc @ np.linalg.inv(x_fc)
        # Deliberately stale YAML: live metadata must fully override its intrinsics.
        calibration["camera"] = {
            "width": 640,
            "height": 480,
            "K": np.eye(3).tolist(),
            "distortion": [0.0] * 5,
        }
        self.path = tmp_path / "stale_intrinsics.yaml"
        self.path.write_text(yaml.safe_dump(calibration))
        self.color = np.full((720, 1280, 3), [120, 140, 40], dtype=np.uint8)
        self.depth = np.full((720, 1280), z, dtype=np.float32)
        self.ys, self.xs = np.mgrid[:720, :1280]

    @property
    def kwargs(self):
        return {"handeye_yaml": self.path, "camera_metadata": self.meta}

    def plane_xy(self, height):
        d = self.z - height
        return (
            self.xy[0] + (self.xs - self.k[0, 2]) * d / self.k[0, 0],
            self.xy[1] - (self.ys - self.k[1, 2]) * d / self.k[1, 1],
        )

    def sideways_cup(self, xy, yaw=0.0, color=(238, 242, 245)):
        bx, by = self.plane_xy(0.055)
        angle = np.deg2rad(yaw)
        along = (bx - xy[0]) * np.cos(angle) + (by - xy[1]) * np.sin(angle)
        across = -(bx - xy[0]) * np.sin(angle) + (by - xy[1]) * np.cos(angle)
        radius = 0.0265 + np.clip((along + 0.044) / 0.088, 0, 1) * 0.011
        mask = (np.abs(along) <= 0.044) & (np.abs(across) <= radius)
        self.color[mask] = color
        self.depth[mask] = self.z - 0.055

    def disk(self, xy, height, outer, inner=0.0, color=(180, 185, 190)):
        bx, by = self.plane_xy(height)
        radius = np.hypot(bx - xy[0], by - xy[1])
        mask = (radius <= outer) & (radius >= inner)
        self.color[mask] = color
        self.depth[mask] = self.z - height


@pytest.mark.parametrize(
    "task, z",
    [("stack_cups", 0.36), ("screw_nut", 0.33), ("stack_cups", 0.28), ("screw_nut", 0.25)],
)
def test_d435_near_camera_ik_keeps_migrated_optical_center(task, z):
    pipeline = importlib.import_module(f"examples.{task}.sim2real")
    calibration = pipeline.load_kinematics_calibration(SAMPLE_CALIBRATION_YAML, robot="xarm6")
    seed = np.array([0.0, -0.55, -0.35, 0.0, 0.90, 0.0])
    target = (0.354, -0.011)
    kwargs = {"q_seed": seed, "calibration": calibration, "handeye_yaml": DEFAULT_HANDEYE_YAML}
    q = pipeline.solve_camera_lookdown_q(target, z_cam_m=z, **kwargs)
    with pipeline.task.__dict__["ScrewNutRobot" if task == "screw_nut" else "StackCupRobot"](
        viewer=False, calibration=calibration
    ) as robot:
        robot.data.qpos[:6] = q
        mujoco.mj_forward(robot.model, robot.data)
        x_fc = np.array(
            yaml.safe_load(DEFAULT_HANDEYE_YAML.read_text())["extrinsics"]["X_flange_cam"]
        )
        optical_pose = flange_pose(robot.model, robot.data) @ x_fc
        assert np.allclose(optical_pose[:3, 3], [*target, z], atol=0.002)
        assert optical_pose[2, 2] < -0.999


@pytest.mark.parametrize("color", [(238, 242, 245), (168, 193, 252)])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_d435_overview_cups_and_empty_table(tmp_path, count, color):
    scene = Scene(tmp_path)
    if count == 1:
        with pytest.raises(PerceptionError, match="No white paper cup"):
            pick.detect_sideways_cup(scene.color, scene.depth, scene.flange, **scene.kwargs)
        centers = [(0.41, -0.04)]
    else:
        with pytest.raises(PerceptionError, match="Only 0"):
            stack.detect_sideways_cups(
                scene.color, scene.depth, scene.flange, num_cups=count, **scene.kwargs
            )
        centers = [(x, -0.04) for x in ([0.35, 0.47] if count == 2 else [0.31, 0.42, 0.53])]
    for center in centers:
        scene.sideways_cup(center, color=color)
    if count == 1:
        est, debug = pick.detect_sideways_cup(
            scene.color, scene.depth, scene.flange, **scene.kwargs
        )
        estimates = [est]
    else:
        estimates, debug = stack.detect_sideways_cups(
            scene.color, scene.depth, scene.flange, num_cups=count, **scene.kwargs
        )
    assert (
        len(estimates) == count
        and debug["camera_cfg"]["width"] == 1280
        and debug["camera_cfg"]["height"] == 720
    )
    for est, center in zip(
        sorted(estimates, key=lambda e: e.cup_center_xy_m[0]), centers, strict=True
    ):
        assert np.allclose(est.cup_center_xy_m, center, atol=0.006)
        assert abs(est.yaw_deg) < 5 and abs(est.table_z_m) < 0.005


@pytest.mark.parametrize("color", [(238, 242, 245), (168, 193, 252)])
@pytest.mark.parametrize("yaw", [0.0, 25.0, 90.0, 140.0])
def test_d435_near_cup_at_360mm(tmp_path, yaw, color):
    scene = Scene(tmp_path, z=0.36)
    xy = scene.xy
    missing, _ = stack.detect_single_cup_near(
        scene.color, scene.depth, scene.flange, xy, **scene.kwargs
    )
    assert missing is None
    scene.sideways_cup(xy, yaw=yaw, color=color)
    found, _ = stack.detect_single_cup_near(
        scene.color, scene.depth, scene.flange, xy, **scene.kwargs
    )
    assert found is not None and np.allclose(found.cup_center_xy_m, xy, atol=0.006)
    assert found.yaw_deg == pytest.approx(yaw, abs=5.0)
    scene.depth[:] = 0
    missing, _ = stack.detect_single_cup_near(
        scene.color, scene.depth, scene.flange, xy, **scene.kwargs
    )
    assert missing is None


def test_d435_cropped_overview_cup_is_rejected(tmp_path):
    scene = Scene(tmp_path)
    # Straddles the physical bottom edge (y = 719) of the uncropped 1280x720 frame.
    scene.sideways_cup((0.41, -0.20))
    with pytest.raises(PerceptionError, match="clipped"):
        pick.detect_sideways_cup(scene.color, scene.depth, scene.flange, **scene.kwargs)
    scene.sideways_cup((0.34, -0.04))
    with pytest.raises(PerceptionError, match="clipped"):
        stack.detect_sideways_cups(
            scene.color, scene.depth, scene.flange, num_cups=2, **scene.kwargs
        )


@pytest.mark.parametrize("center", [(0.41, -0.13), (0.41, -0.165)])
def test_d435_complete_cup_near_margin_is_detected(tmp_path, center):
    scene = Scene(tmp_path)
    scene.sideways_cup(center)
    pose, _ = pick.detect_sideways_cup(scene.color, scene.depth, scene.flange, **scene.kwargs)
    assert np.allclose(pose.cup_center_xy_m, center, atol=0.006)


@pytest.mark.parametrize("color", [(238, 242, 245), (168, 193, 252), (97, 141, 216)])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_d435_post_place_and_stack_positive_negative(tmp_path, count, color):
    scene = Scene(tmp_path)
    target = scene.xy

    def verify():
        if count == 1:
            return pick.verify_upright_cup(
                scene.color, scene.depth, scene.flange, target, **scene.kwargs
            )[0]
        return stack.verify_stacked_cups(
            scene.color, scene.depth, scene.flange, target, num_cups=count, **scene.kwargs
        )[0]

    assert not verify().ok
    height = 0.088 + 0.0095 * (count - 1)
    scene.disk(target, height, 0.0375, color=color)
    result = verify()
    assert result.ok, result.reason
    assert result.xy_error_mm < 5 and result.rim_height_mm == pytest.approx(height * 1000, abs=1.0)
    # Low, tipped white object must fail the task completion check.
    scene.depth[scene.depth < scene.z] = scene.z - 0.025
    assert not verify().ok


@pytest.mark.parametrize("z", [0.25, 0.50])
def test_d435_nut_screw_detection_and_thread_verification(tmp_path, z):
    scene = Scene(tmp_path, z=z)
    scene.xy = (0.355, -0.055)
    # Update the virtual flange for the new center while retaining the migrated X_fc.
    x_fc = np.asarray(
        yaml.safe_load(DEFAULT_HANDEYE_YAML.read_text())["extrinsics"]["X_flange_cam"]
    )
    t_bc = np.diag([1.0, -1.0, -1.0, 1.0])
    t_bc[:3, 3] = [*scene.xy, z]
    scene.flange = t_bc @ np.linalg.inv(x_fc)
    nut_xy, screw_xy = (0.354, -0.011), (0.356, -0.103)
    nut, _ = screw.detect_nut_near(scene.color, scene.depth, scene.flange, nut_xy, **scene.kwargs)
    assert nut is None
    scene.disk(nut_xy, 0.0145, 0.0145, 0.006)
    scene.disk(screw_xy, 0.014, 0.010)
    scene.disk(screw_xy, 0.045, 0.006)
    nut, bolt, _ = screw.detect_nut_and_screw_overview(
        scene.color, scene.depth, scene.flange, **scene.kwargs
    )
    assert np.allclose(nut.xy_m, nut_xy, atol=0.003) and np.allclose(
        bolt.xy_m, screw_xy, atol=0.003
    )
    assert nut.height_mm == pytest.approx(14.5, abs=1.5) and bolt.height_mm == pytest.approx(
        45.0, abs=1.5
    )
    scene.color[:] = [120, 140, 40]
    scene.depth[:] = z
    verify_kw = {**scene.kwargs, "initial_nut_xy_m": nut_xy}
    empty, _ = screw.verify_threaded_nut(
        scene.color, scene.depth, scene.flange, screw_xy, **verify_kw
    )
    assert not empty.ok
    scene.disk(screw_xy, 0.0285, 0.0145, 0.006)
    scene.disk(screw_xy, 0.045, 0.006)
    result, _ = screw.verify_threaded_nut(
        scene.color, scene.depth, scene.flange, screw_xy, **verify_kw
    )
    assert result.ok, result.reason
    assert result.nut_top_height_mm == pytest.approx(28.5, abs=1.0)
    scene.disk(nut_xy, 0.0145, 0.0145, 0.006)
    uncleared, _ = screw.verify_threaded_nut(
        scene.color, scene.depth, scene.flange, screw_xy, **verify_kw
    )
    assert not uncleared.ok and not uncleared.cleared_initial_nut_site


@pytest.mark.parametrize("color", [(175, 195, 140), (10, 150, 240), (200, 70, 30)])
def test_d435_green_or_saturated_objects_are_rejected(tmp_path, color):
    scene = Scene(tmp_path)
    scene.sideways_cup(scene.xy, color=color)
    with pytest.raises(PerceptionError, match="No white paper cup"):
        pick.detect_sideways_cup(scene.color, scene.depth, scene.flange, **scene.kwargs)
    near, _ = stack.detect_single_cup_near(
        scene.color, scene.depth, scene.flange, scene.xy, **scene.kwargs
    )
    assert near is None


def test_d435_offline_1280x720_default_handeye_detects_cup():
    calibration = yaml.safe_load(DEFAULT_HANDEYE_YAML.read_text())
    k = np.asarray(calibration["camera"]["K"], dtype=np.float64)
    z = 0.5
    xy = (0.41, -0.04)
    t_bc = np.diag([1.0, -1.0, -1.0, 1.0])
    t_bc[:3, 3] = [*xy, z]
    x_fc = np.asarray(calibration["extrinsics"]["X_flange_cam"], dtype=np.float64)
    flange = t_bc @ np.linalg.inv(x_fc)

    color = np.full((720, 1280, 3), [120, 140, 40], dtype=np.uint8)
    depth = np.full((720, 1280), z, dtype=np.float32)
    ys, xs = np.mgrid[:720, :1280]
    d = z - 0.055
    bx = xy[0] + (xs - k[0, 2]) * d / k[0, 0]
    by = xy[1] - (ys - k[1, 2]) * d / k[1, 1]
    along = bx - xy[0]
    across = by - xy[1]
    radius = 0.0265 + np.clip((along + 0.044) / 0.088, 0, 1) * 0.011
    mask = (np.abs(along) <= 0.044) & (np.abs(across) <= radius)
    color[mask] = (168, 193, 252)
    depth[mask] = d

    pose, debug = pick.detect_sideways_cup(color, depth, flange, handeye_yaml=DEFAULT_HANDEYE_YAML)
    assert debug["camera_cfg"]["type"] == "realsense"
    assert debug["camera_cfg"]["width"] == 1280
    assert np.allclose(pose.cup_center_xy_m, xy, atol=0.006)


def test_d435_open_mouth_cavity_and_specular_dropout_keeps_true_yaw(tmp_path):
    """Open-mouth low interior Z + partial stereo IR dropout must not invert bottom->rim yaw."""
    scene = Scene(tmp_path, z=0.45)
    cup_xy = (0.32, -0.01)
    true_yaw = -34.0
    bx, by = scene.plane_xy(0.035)
    ang = np.deg2rad(true_yaw)
    along = (bx - cup_xy[0]) * np.cos(ang) + (by - cup_xy[1]) * np.sin(ang)
    across = -(bx - cup_xy[0]) * np.sin(ang) + (by - cup_xy[1]) * np.cos(ang)
    radius = 0.0265 + np.clip((along + 0.044) / 0.088, 0.0, 1.0) * 0.011
    mask = (np.abs(along) <= 0.044) & (np.abs(across) <= radius)
    scene.color[mask] = (238, 242, 245)

    # Simulate hollow open-mouth low interior floor Z (~24 mm) at the wide +along rim end,
    # higher outer wall Z (~58 mm) at the narrow -along bottom end, plus specular depth==0 dropout
    # across one flank of the cup crest.
    z_surf = np.where(along >= 0.0, 0.024, 0.058)
    scene.depth[mask] = (scene.z - z_surf[mask]).astype(np.float32)
    dropout = mask & (across > 0.004) & (np.abs(along) < 0.028)
    scene.depth[dropout] = 0.0

    pose, _ = pick.detect_sideways_cup(scene.color, scene.depth, scene.flange, **scene.kwargs)
    assert np.allclose(pose.cup_center_xy_m, cup_xy, atol=0.005)
    assert pose.yaw_deg == pytest.approx(true_yaw, abs=4.0)
    assert pose.width_top_mm > pose.width_bot_mm + 8.0

    near, _ = stack.detect_single_cup_near(
        scene.color, scene.depth, scene.flange, cup_xy, **scene.kwargs
    )
    assert near is not None
    assert np.allclose(near.cup_center_xy_m, cup_xy, atol=0.005)
    assert near.yaw_deg == pytest.approx(true_yaw, abs=4.0)


def test_d435_chrome_nut_one_sided_depth_dropout_recovers_center_and_hex_yaw(tmp_path):
    """One-sided stereo-IR glare dropout on a chrome flange nut must not bias XY or hex flat yaw."""
    scene = Scene(tmp_path, z=0.33)
    nut_xy = (0.312, 0.048)
    true_yaw = 14.0

    # Render 29 mm circular bottom flange (z = 3.8 mm), 20 mm across-flats upper hexagon (z = 14.5 mm),
    # and 12 mm central M12 bore, then zero out 60% of valid depth on one flank of the nut.
    bx_f, by_f = scene.plane_xy(0.0038)
    r_f = np.hypot(bx_f - nut_xy[0], by_f - nut_xy[1])
    flange_mask = (r_f >= 0.0060) & (r_f <= 0.0145)
    scene.color[flange_mask] = (172, 178, 184)
    scene.depth[flange_mask] = scene.z - 0.0038

    bx_h, by_h = scene.plane_xy(0.0145)
    dx_h, dy_h = bx_h - nut_xy[0], by_h - nut_xy[1]
    in_hex = np.hypot(dx_h, dy_h) >= 0.0060
    for k in range(3):
        phi = np.deg2rad(true_yaw + 60.0 * k)
        in_hex &= np.abs(dx_h * np.cos(phi) + dy_h * np.sin(phi)) <= 0.0100
    scene.color[in_hex] = (232, 238, 244)
    scene.depth[in_hex] = scene.z - 0.0145

    # Specular stereo-IR dropout leaving depth only on the +Y crescent (would pull 3D median +6 mm).
    dropout = (flange_mask | in_hex) & ((by_f - nut_xy[1]) < 0.0025)
    scene.depth[dropout] = 0.0

    nut, _ = screw.detect_nut_near(scene.color, scene.depth, scene.flange, nut_xy, **scene.kwargs)
    assert nut is not None
    assert np.allclose(nut.xy_m, nut_xy, atol=0.0015)
    assert nut.outer_diam_mm == pytest.approx(29.0, abs=2.0)
    assert nut.yaw_deg == pytest.approx(true_yaw, abs=3.0)

    # Same one-sided chrome dropout when seated on the M12 bolt during post-threading verification.
    bolt_xy = (0.407, 0.025)
    scene.color[:] = [120, 140, 40]
    scene.depth[:] = scene.z
    bx_tf, by_tf = scene.plane_xy(0.0178)
    nut_flange_on_bolt = (np.hypot(bx_tf - bolt_xy[0], by_tf - bolt_xy[1]) >= 0.0060) & (
        np.hypot(bx_tf - bolt_xy[0], by_tf - bolt_xy[1]) <= 0.0145
    )
    scene.color[nut_flange_on_bolt] = (195, 200, 206)
    scene.depth[nut_flange_on_bolt] = scene.z - 0.0178
    bx_th, by_th = scene.plane_xy(0.0285)
    nut_hex_on_bolt = (np.hypot(bx_th - bolt_xy[0], by_th - bolt_xy[1]) >= 0.0060) & (
        np.hypot(bx_th - bolt_xy[0], by_th - bolt_xy[1]) <= 0.0105
    )
    scene.color[nut_hex_on_bolt] = (225, 230, 236)
    scene.depth[nut_hex_on_bolt] = scene.z - 0.0285
    # Simulate 1.8 mm stereo block-matching foreground fattening around the r=6.0 mm bolt tip
    # (r <= 7.8 mm at z = 45.0 mm); nut_top_height_mm must stay at 28.5 mm, not jump to 45.0 mm.
    scene.disk(bolt_xy, 0.045, 0.0078, color=(190, 195, 200))
    scene.depth[(nut_flange_on_bolt | nut_hex_on_bolt) & ((by_tf - bolt_xy[1]) < 0.0020)] = 0.0

    ver, _ = screw.verify_threaded_nut(
        scene.color, scene.depth, scene.flange, bolt_xy, initial_nut_xy_m=nut_xy, **scene.kwargs
    )
    assert ver.ok, ver.reason
    assert ver.xy_error_mm <= 1.5
    assert ver.nut_top_height_mm == pytest.approx(28.5, abs=1.0)

    # Bare bolt (14 mm head + 7.8 mm stereo-fattened shaft, no nut) must still fail verification.
    scene.color[:] = [120, 140, 40]
    scene.depth[:] = scene.z
    scene.disk(bolt_xy, 0.014, 0.0100, color=(180, 185, 190))
    scene.disk(bolt_xy, 0.045, 0.0078, color=(190, 195, 200))
    bare_ver, _ = screw.verify_threaded_nut(
        scene.color, scene.depth, scene.flange, bolt_xy, initial_nut_xy_m=nut_xy, **scene.kwargs
    )
    assert not bare_ver.ok and bare_ver.nut_ring_points == 0
