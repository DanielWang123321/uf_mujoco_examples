# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Orbbec RGB-D capture (tested with the Gemini 305) with depth aligned to the color image.

``pyorbbecsdk`` (PyPI: ``pyorbbecsdk2``) is imported lazily, so simulation-only installs do not
need it. Install it with ``pip install -e ".[real]"``.
"""

from __future__ import annotations

from typing import Any, Self

import cv2
import numpy as np

DEFAULT_CAMERA_SIZE = (1280, 800)
DEFAULT_CAMERA_FPS = 30
_WARMUP_FRAMES = 10
_FLUSH_FRAMES = 2
_FRAME_TIMEOUT_MS = 1000


class CameraError(RuntimeError):
    """Raised when the Orbbec camera cannot be opened or returns invalid frames."""


def _import_sdk() -> Any:
    try:
        import pyorbbecsdk as ob
    except ImportError as exc:
        raise CameraError(
            'pyorbbecsdk is not installed; run: pip install -e ".[real]" (package pyorbbecsdk2)'
        ) from exc
    return ob


def _color_to_bgr(ob: Any, frame: Any) -> np.ndarray:
    """Convert an Orbbec color frame to a BGR ``uint8`` image."""
    width, height = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == ob.OBFormat.MJPG:
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    elif fmt in (ob.OBFormat.RGB, ob.OBFormat.BGR):
        image = raw[: height * width * 3].reshape(height, width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if fmt == ob.OBFormat.RGB else image.copy()
    elif fmt in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
        image = cv2.cvtColor(
            raw[: height * width * 2].reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUYV
        )
    else:
        raise CameraError(f"Unsupported color format: {fmt}")
    if image is None or image.shape[:2] != (height, width):
        raise CameraError("Failed to decode the color frame")
    return image


def _depth_to_meters(frame: Any) -> np.ndarray:
    """Convert an Orbbec 16-bit depth frame to float32 meters."""
    width, height = frame.get_width(), frame.get_height()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
    if raw.size < width * height:
        raise CameraError("Truncated depth frame")
    scale_m = float(frame.get_depth_scale()) / 1000.0
    return raw[: width * height].reshape(height, width).astype(np.float32) * scale_m


class OrbbecCameraStream:
    """Open Orbbec RGB-D stream; ``read_frame`` returns the latest aligned frame.

    Use as a context manager, or call ``start`` / ``close`` explicitly.
    """

    def __init__(
        self,
        serial: str = "",
        width: int = DEFAULT_CAMERA_SIZE[0],
        height: int = DEFAULT_CAMERA_SIZE[1],
        fps: int = DEFAULT_CAMERA_FPS,
    ) -> None:
        self.serial = serial or ""
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.meta: dict[str, Any] = {}
        self._ob: Any = None
        self._ctx: Any = None
        self._pipeline: Any = None
        self._align: Any = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def _open_device(self) -> Any:
        ob = self._ob
        ob.Context.set_logger_to_file(ob.OBLogLevel.NONE, "")
        ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
        # The context must outlive the device handles it creates.
        self._ctx = ob.Context()
        devices = self._ctx.query_devices()
        if devices.get_count() == 0:
            raise CameraError("No Orbbec camera found; check the USB connection and udev rules")
        found = []
        for idx in range(devices.get_count()):
            device = devices.get_device_by_index(idx)
            sn = device.get_device_info().get_serial_number()
            if not self.serial or sn == self.serial:
                return device
            found.append(sn)
        raise CameraError(
            f"Orbbec camera {self.serial!r} not found (connected: {', '.join(found)})"
        )

    def _video_profile(self, sensor: Any, formats: tuple[Any, ...]) -> Any:
        profiles = self._pipeline.get_stream_profile_list(sensor)
        for fmt in formats:
            try:
                return profiles.get_video_stream_profile(self.width, self.height, fmt, self.fps)
            except Exception:  # noqa: BLE001 - try the next format
                continue
        raise CameraError(f"No {self.width}x{self.height}@{self.fps} profile for {sensor}")

    def start(self) -> dict[str, Any]:
        """Open the camera, warm up the stream, and return its metadata."""
        if self._pipeline is not None:
            return self.meta
        ob = self._ob = _import_sdk()
        device = self._open_device()
        self._pipeline = ob.Pipeline(device)
        try:
            color_profile = self._video_profile(
                ob.OBSensorType.COLOR_SENSOR,
                (ob.OBFormat.RGB, ob.OBFormat.BGR, ob.OBFormat.MJPG, ob.OBFormat.YUYV),
            )
            depth_profile = self._video_profile(
                ob.OBSensorType.DEPTH_SENSOR, (ob.OBFormat.Y16, ob.OBFormat.Z16)
            )
            config = ob.Config()
            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            self._pipeline.enable_frame_sync()
            self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            self._pipeline.start(config)
            for _ in range(_WARMUP_FRAMES):
                self._pull()
            info = device.get_device_info()
            intr = color_profile.get_intrinsic()
            self.meta = {
                "name": info.get_name(),
                "serial": info.get_serial_number(),
                "width": color_profile.get_width(),
                "height": color_profile.get_height(),
                "fps": color_profile.get_fps(),
                "K": [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
            }
        except BaseException:
            self.close()
            raise
        return self.meta

    def _pull(self) -> tuple[Any, Any]:
        for _ in range(5):
            frames = self._pipeline.wait_for_frames(_FRAME_TIMEOUT_MS)
            if frames is None:
                continue
            aligned = self._align.process(frames)
            if aligned is None:
                continue
            aligned = aligned.as_frame_set()
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if color is not None and depth is not None:
                return color, depth
        raise CameraError("Timed out waiting for an aligned color + depth frame")

    def read_frame(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return a fresh ``(color_bgr, depth_m, metadata)`` frame (depth aligned to color)."""
        self.start()
        try:
            for _ in range(_FLUSH_FRAMES):
                self._pull()
            color, depth = self._pull()
            return _color_to_bgr(self._ob, color), _depth_to_meters(depth), self.meta
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
        self._pipeline = None
        self._align = None
        self._ctx = None


def capture_rgbd_frame(
    serial: str = "",
    width: int = DEFAULT_CAMERA_SIZE[0],
    height: int = DEFAULT_CAMERA_SIZE[1],
    fps: int = DEFAULT_CAMERA_FPS,
    stream: OrbbecCameraStream | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Capture one aligned ``(color_bgr, depth_m, metadata)`` frame.

    Uses ``stream`` if given, otherwise opens the camera for a single frame.
    """
    if stream is not None:
        return stream.read_frame()
    with OrbbecCameraStream(serial, width, height, fps) as one_shot:
        return one_shot.read_frame()
