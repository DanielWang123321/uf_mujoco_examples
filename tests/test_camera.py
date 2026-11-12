# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""SDK-independent regression tests for the RGB-D stream contract."""

import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from common import camera


@pytest.fixture
def sdk(monkeypatch):
    state = NS(
        starts=0,
        stops=0,
        pulls=0,
        aligns=0,
        missing=False,
        timeout=False,
        start_error=False,
        align_error=False,
        bad_shape=False,
    )
    bgr = np.full((720, 1280, 3), [10, 20, 30], dtype=np.uint8)
    raw = np.full((720, 1280), 200, dtype=np.uint16)
    raw[0, 0] = 0
    intr = NS(
        width=1280,
        height=720,
        fx=911.6868,
        fy=911.9044,
        ppx=636.0090,
        ppy=356.5287,
        coeffs=[0.0] * 5,
        model="distortion.inverse_brown_conrady",
    )

    class Profile:
        def __init__(self, stream):
            self.stream = stream

        def as_video_stream_profile(self):
            return self

        def get_intrinsics(self):
            return intr

        def width(self):
            return 1280

        def height(self):
            return 720

        def fps(self):
            return 30

        def format(self):
            return "bgr8" if self.stream == "color" else "z16"

    class Device:
        def __init__(self, serial):
            self.serial = serial

        def get_info(self, field):
            return self.serial if field == "serial" else "RealSense D435I"

        def first_depth_sensor(self):
            return NS(get_depth_scale=lambda: 0.002)

    class Config:
        def __init__(self):
            self.streams = []
            state.config = self

        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, *args):
            self.streams.append(args)

    class Frame:
        def __init__(self, is_color):
            self.is_color = is_color

        def get_data(self):
            if self.is_color:
                return bgr
            return raw[:1] if state.bad_shape else raw

        def get_frame_number(self):
            return state.pulls

        def get_timestamp(self):
            return state.pulls * 1000 / 30

    class Frames:
        def get_color_frame(self):
            return Frame(True)

        def get_depth_frame(self):
            return None if state.missing else Frame(False)

    class Pipeline:
        def __init__(self, ctx):
            state.context = ctx

        def start(self, config):
            state.starts += 1
            if state.start_error:
                raise RuntimeError("USB failed")
            return NS(get_device=lambda: Device(config.serial), get_stream=Profile)

        def wait_for_frames(self, timeout_ms):
            assert timeout_ms == 1000
            state.pulls += 1
            if state.timeout:
                raise RuntimeError("Frame did not arrive")
            # The raw frames deliberately lack get_color_frame/get_depth_frame.
            return object()

        def stop(self):
            state.stops += 1

    class Align:
        def __init__(self, target):
            assert target == "color"

        def process(self, frames):
            state.aligns += 1
            if state.align_error:
                raise RuntimeError("Align failed")
            return Frames()

    state.devices = [Device("SDK-1"), Device("SDK-2")]
    rs = NS(
        context=lambda: NS(query_devices=lambda: state.devices),
        config=Config,
        pipeline=Pipeline,
        align=Align,
        stream=NS(color="color", depth="depth"),
        format=NS(bgr8="bgr8", z16="z16"),
        camera_info=NS(serial_number="serial", name="name"),
    )
    monkeypatch.setattr(camera, "_import_realsense_sdk", lambda: rs)
    state.bgr, state.raw = bgr, raw
    return state


def test_realsense_selection_alignment_units_and_owned_buffers(sdk):
    with camera.RealSenseCameraStream("SDK-2") as stream:
        bgr, depth, meta = stream.read_frame()
        assert sdk.config.serial == "SDK-2"
        assert sdk.config.streams == [
            ("color", 1280, 720, "bgr8", 30),
            ("depth", 1280, 720, "z16", 30),
        ]
        assert sdk.pulls == camera._REALSENSE_WARMUP_FRAMES + camera._FLUSH_FRAMES + 1
        assert sdk.aligns == sdk.pulls
        assert sdk.starts == 1
        assert depth.dtype == np.float32 and bgr.dtype == np.uint8
        assert depth.shape == bgr.shape[:2] == (720, 1280)
        assert depth[0, 0] == 0 and depth[1, 1] == pytest.approx(0.4)
        assert bgr[0, 0].tolist() == [10, 20, 30]
        assert meta["serial"] == "SDK-2" and meta["depth_scale_m"] == 0.002
        assert meta["distortion_model"] == "inverse_brown_conrady"
        assert meta["streams"]["depth"]["format"] == "z16"
        sdk.bgr.fill(0)
        sdk.raw.fill(0)
        assert bgr[0, 0].tolist() == [10, 20, 30] and depth[1, 1] == pytest.approx(0.4)
        _, _, newer = stream.read_frame()
        assert newer["frame_number"] > meta["frame_number"]
    stream.close()
    assert sdk.stops == 1 and stream._ctx is None


@pytest.mark.parametrize("problem", ["missing", "timeout", "start_error", "align_error"])
def test_realsense_start_failure_releases_resources(sdk, problem):
    setattr(sdk, problem, True)
    stream = camera.RealSenseCameraStream()
    with pytest.raises(camera.CameraError):
        stream.start()
    assert sdk.stops == 1 and stream._pipeline is None and stream._ctx is None


@pytest.mark.parametrize("problem", ["missing", "timeout", "align_error", "bad_shape"])
def test_realsense_read_failure_releases_resources(sdk, problem):
    stream = camera.RealSenseCameraStream()
    stream.start()
    setattr(sdk, problem, True)
    with pytest.raises(camera.CameraError):
        stream.read_frame()
    assert sdk.stops == 1 and stream._pipeline is None


def test_device_not_found_and_restart(sdk):
    with pytest.raises(camera.CameraError, match="SDK-1, SDK-2"):
        camera.RealSenseCameraStream("USB-path-is-not-a-serial").start()
    sdk.devices = []
    with pytest.raises(camera.CameraError, match="No RealSense"):
        camera.RealSenseCameraStream().start()
    assert sdk.starts == 0


def test_one_shot_and_persistent_capture(sdk):
    _, _, meta = camera.capture_rgbd_frame()
    assert meta["serial"] == "SDK-1" and sdk.stops == 1
    with camera.create_camera_stream() as stream:
        camera.capture_rgbd_frame(stream=stream)
        assert sdk.stops == 1
    assert sdk.stops == 2


def test_factory_defaults_and_lazy_sdks(monkeypatch):
    def forbidden():
        raise AssertionError("SDK imported before stream.start()")

    monkeypatch.setattr(camera, "_import_sdk", forbidden)
    monkeypatch.setattr(camera, "_import_realsense_sdk", forbidden)
    rs = camera.create_camera_stream()
    ob = camera.create_camera_stream("orbbec", "SDK-SN")
    assert isinstance(rs, camera.RealSenseCameraStream) and (rs.width, rs.height) == (1280, 720)
    assert isinstance(ob, camera.OrbbecCameraStream) and (ob.width, ob.height) == (1280, 800)
    assert ob.serial == "SDK-SN" and ob.fps == rs.fps == 30
    with pytest.raises(ValueError, match="Unsupported camera"):
        camera.create_camera_stream("unknown")


@pytest.mark.parametrize(
    "module,loader", [("pyrealsense2", "_import_realsense_sdk"), ("pyorbbecsdk", "_import_sdk")]
)
def test_missing_sdk_message(monkeypatch, module, loader):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(camera.CameraError, match=module + " is not installed"):
        getattr(camera, loader)()


def test_orbbec_depth_units_and_copy():
    raw = np.array([0, 100, 200, 300], dtype=np.uint16)
    frame = NS(
        get_width=lambda: 2, get_height=lambda: 2, get_data=lambda: raw, get_depth_scale=lambda: 2.0
    )
    depth = camera._depth_to_meters(frame)
    raw.fill(0)
    assert depth.dtype == np.float32
    assert np.allclose(depth, [[0, 0.2], [0.4, 0.6]])


def test_orbbec_stream_metadata_alignment_and_failure_cleanup(monkeypatch):
    state = NS(stops=0, missing=False)
    bgr = np.full((800, 1280, 3), 128, dtype=np.uint8)
    raw = np.full((800, 1280), 400, dtype=np.uint16)
    color = NS(
        get_width=lambda: 1280,
        get_height=lambda: 800,
        get_format=lambda: "BGR",
        get_data=lambda: bgr,
        get_index=lambda: 15,
        get_timestamp=lambda: 500.0,
    )
    depth = NS(
        get_width=lambda: 1280,
        get_height=lambda: 800,
        get_data=lambda: raw,
        get_depth_scale=lambda: 1.0,
    )
    info = NS(get_name=lambda: "Gemini 305", get_serial_number=lambda: "OB-SDK")
    device = NS(get_device_info=lambda: info)
    devices = NS(get_count=lambda: 1, get_device_by_index=lambda i: device)
    dist = NS(
        **dict.fromkeys(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"], 0.0),
        model="OBCameraDistortionModel.BROWN_CONRADY_K6",
    )
    profiles = {
        name: NS(
            get_width=lambda: 1280,
            get_height=lambda: 800,
            get_fps=lambda: 30,
            get_format=lambda name=name: name,
            get_intrinsic=lambda: NS(fx=612.0, fy=613.0, cx=633.0, cy=400.0),
            get_distortion=lambda: dist,
        )
        for name in ("COLOR", "DEPTH")
    }
    frames = NS(
        get_color_frame=lambda: color, get_depth_frame=lambda: None if state.missing else depth
    )
    frames.as_frame_set = lambda: frames

    class Context:
        set_logger_to_file = staticmethod(lambda *a: None)
        set_logger_to_console = staticmethod(lambda *a: None)

        def query_devices(self):
            return devices

    def stop():
        state.stops += 1

    pipeline = NS(
        get_stream_profile_list=lambda sensor: NS(
            get_video_stream_profile=lambda *a: profiles[sensor]
        ),
        enable_frame_sync=lambda: None,
        start=lambda c: None,
        wait_for_frames=lambda ms: frames,
        stop=stop,
    )
    ob = NS(
        Context=Context,
        OBLogLevel=NS(NONE=0, ERROR=1),
        Pipeline=lambda d: pipeline,
        Config=lambda: NS(
            enable_stream=lambda p: None, set_frame_aggregate_output_mode=lambda m: None
        ),
        OBFormat=NS(
            RGB="RGB", BGR="BGR", MJPG="MJPG", YUYV="YUYV", YUY2="YUY2", Y16="Y16", Z16="Z16"
        ),
        OBSensorType=NS(COLOR_SENSOR="COLOR", DEPTH_SENSOR="DEPTH"),
        OBStreamType=NS(COLOR_STREAM="COLOR"),
        OBFrameAggregateOutputMode=NS(FULL_FRAME_REQUIRE=1),
        AlignFilter=lambda **kw: NS(process=lambda f: f),
    )
    monkeypatch.setattr(camera, "_import_sdk", lambda: ob)
    with pytest.raises(camera.CameraError, match="not found"):
        camera.OrbbecCameraStream("wrong-serial").start()
    stream = camera.create_camera_stream("orbbec", "OB-SDK")
    image, meters, meta = stream.read_frame()
    assert meta["serial"] == "OB-SDK" and meta["depth_aligned_to"] == "color"
    assert meta["distortion_model"] == "brown_conrady_k6" and len(meta["distortion"]) == 8
    assert meta["width"] == 1280 and meta["depth_scale_m"] == 0.001
    assert image.shape == (800, 1280, 3) and meters.dtype == np.float32
    bgr.fill(0)
    assert image[0, 0, 0] == 128
    state.missing = True
    with pytest.raises(camera.CameraError, match="Timed out"):
        stream.read_frame()
    assert state.stops == 1 and stream._ctx is None and stream._pipeline is None
