# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""RGB-D streams for RealSense D435/D435i and Orbbec Gemini 305.

Both SDKs are imported lazily. Frames are copied BGR uint8 and aligned float32 meters.
"""

from __future__ import annotations

from typing import Any, Protocol, Self

import cv2
import numpy as np

DEFAULT_CAMERA_TYPE = "realsense"
CAMERA_TYPES = ("realsense", "orbbec")
# Maximum synchronized RGB + Z16 depth resolution on D435/D435i (native 16:9 full sensor).
DEFAULT_CAMERA_SIZE = (1280, 720)
ORBBEC_CAMERA_SIZE = (1280, 800)
DEFAULT_CAMERA_FPS = 30
_WARMUP_FRAMES = 10
# D435 auto white balance initially tints a white cup green on the table mat.
_REALSENSE_WARMUP_FRAMES = 30
_FLUSH_FRAMES = 2
_FRAME_TIMEOUT_MS = 1000


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or returns invalid frames."""


class CameraStream(Protocol):
    """Common lifecycle and frame contract for persistent RGB-D streams."""

    meta: dict[str, Any]

    def start(self) -> dict[str, Any]: ...

    def read_frame(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


def _import_realsense_sdk() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraError('pyrealsense2 is not installed; run: pip install -e ".[real]"') from exc
    return rs


class RealSenseCameraStream:
    """RealSense color + Z16 depth aligned to the color optical frame."""

    def __init__(
        self,
        serial: str = "",
        width: int = DEFAULT_CAMERA_SIZE[0],
        height: int = DEFAULT_CAMERA_SIZE[1],
        fps: int = DEFAULT_CAMERA_FPS,
    ) -> None:
        self.serial = serial or ""
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.meta: dict[str, Any] = {}
        self._rs: Any = None
        self._ctx: Any = None
        self._pipeline: Any = None
        self._align: Any = None
        self._depth_scale_m = 0.0

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def start(self) -> dict[str, Any]:
        if self._pipeline is not None:
            return self.meta
        rs = self._rs = _import_realsense_sdk()
        try:
            self._ctx = rs.context()
            devices = list(self._ctx.query_devices())
            serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
            if not serials:
                raise CameraError(
                    "No RealSense camera found; check the USB connection and udev rules"
                )
            if self.serial and self.serial not in serials:
                raise CameraError(
                    f"RealSense camera {self.serial!r} not found (connected: {', '.join(serials)})"
                )
            config = rs.config()
            config.enable_device(self.serial or serials[0])
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self._pipeline = rs.pipeline(self._ctx)
            profile = self._pipeline.start(config)
            device = profile.get_device()
            self._depth_scale_m = float(device.first_depth_sensor().get_depth_scale())
            if not np.isfinite(self._depth_scale_m) or self._depth_scale_m <= 0:
                raise CameraError("Invalid RealSense depth scale")
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            self.meta = {
                "type": "realsense",
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "width": intr.width,
                "height": intr.height,
                "fps": color_profile.fps(),
                "K": [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                "distortion": list(intr.coeffs),
                "distortion_model": str(intr.model).removeprefix("distortion."),
                "depth_scale_m": self._depth_scale_m,
                "depth_aligned_to": "color",
                "streams": {
                    "color": self._stream_metadata(color_profile),
                    "depth": self._stream_metadata(depth_profile),
                },
            }
            self._align = rs.align(rs.stream.color)
            for _ in range(_REALSENSE_WARMUP_FRAMES):
                self._pull()
        except BaseException as exc:
            self.close()
            if isinstance(exc, Exception) and not isinstance(exc, CameraError):
                raise CameraError(f"Failed to start RealSense camera: {exc}") from exc
            raise
        return dict(self.meta)

    @staticmethod
    def _stream_metadata(profile: Any) -> dict[str, Any]:
        return {
            "width": profile.width(),
            "height": profile.height(),
            "fps": profile.fps(),
            "format": str(profile.format()),
        }

    def _pull(self) -> tuple[Any, Any]:
        last_error: Exception | None = None
        for _ in range(5):
            try:
                frames = self._pipeline.wait_for_frames(_FRAME_TIMEOUT_MS)
            except RuntimeError as exc:
                last_error = exc
                continue
            if not frames:
                continue
            aligned = self._align.process(frames)
            if not aligned:
                continue
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if color and depth:
                return color, depth
        raise CameraError("Timed out waiting for an aligned color + depth frame") from last_error

    def read_frame(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self.start()
        try:
            for _ in range(_FLUSH_FRAMES):
                self._pull()
            color, depth = self._pull()
            bgr = np.asanyarray(color.get_data()).copy()
            raw_depth = np.asanyarray(depth.get_data())
            if (
                bgr.dtype != np.uint8
                or bgr.shape != (self.meta["height"], self.meta["width"], 3)
                or raw_depth.dtype != np.uint16
                or raw_depth.shape != bgr.shape[:2]
            ):
                raise CameraError("Invalid or misaligned RealSense RGB-D frame sizes/formats")
            depth_m = raw_depth.astype(np.float32) * np.float32(self._depth_scale_m)
            self.meta.update(
                frame_number=color.get_frame_number(), timestamp_ms=color.get_timestamp()
            )
            return bgr, depth_m, dict(self.meta)
        except BaseException as exc:
            self.close()
            if isinstance(exc, Exception) and not isinstance(exc, CameraError):
                raise CameraError(f"Failed to read RealSense frame: {exc}") from exc
            raise

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 - start may have failed before streaming
                pass
        self._pipeline = self._align = self._ctx = None


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
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise CameraError("Invalid Orbbec depth scale")
    return raw[: width * height].reshape(height, width).astype(np.float32) * scale_m


class OrbbecCameraStream:
    """Open Orbbec RGB-D stream; ``read_frame`` returns the latest aligned frame.

    Use as a context manager, or call ``start`` / ``close`` explicitly.
    """

    def __init__(
        self,
        serial: str = "",
        width: int = ORBBEC_CAMERA_SIZE[0],
        height: int = ORBBEC_CAMERA_SIZE[1],
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
        try:
            device = self._open_device()
            self._pipeline = ob.Pipeline(device)
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
            depth = None
            for _ in range(_WARMUP_FRAMES):
                _, depth = self._pull()
            info = device.get_device_info()
            intr = color_profile.get_intrinsic()
            dist = color_profile.get_distortion()
            self.meta = {
                "type": "orbbec",
                "name": info.get_name(),
                "serial": info.get_serial_number(),
                "width": color_profile.get_width(),
                "height": color_profile.get_height(),
                "fps": color_profile.get_fps(),
                "K": [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
                "distortion": [
                    getattr(dist, k) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
                ],
                "distortion_model": str(dist.model).split(".")[-1].lower(),
                "depth_scale_m": float(depth.get_depth_scale()) / 1000.0,
                "depth_aligned_to": "color",
                "streams": {
                    name: {
                        "width": p.get_width(),
                        "height": p.get_height(),
                        "fps": p.get_fps(),
                        "format": str(p.get_format()),
                    }
                    for name, p in (("color", color_profile), ("depth", depth_profile))
                },
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
            bgr, depth_m = _color_to_bgr(self._ob, color), _depth_to_meters(depth)
            if bgr.shape[:2] != depth_m.shape:
                raise CameraError("Invalid or misaligned Orbbec RGB-D frame sizes")
            self.meta.update(
                depth_scale_m=float(depth.get_depth_scale()) / 1000.0,
                frame_number=color.get_index(),
                timestamp_ms=color.get_timestamp(),
            )
            return bgr, depth_m, dict(self.meta)
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


def create_camera_stream(
    camera_type: str = DEFAULT_CAMERA_TYPE,
    serial: str = "",
    width: int | None = None,
    height: int | None = None,
    fps: int = DEFAULT_CAMERA_FPS,
) -> CameraStream:
    """Create the selected stream without importing its SDK until ``start``."""
    if camera_type not in CAMERA_TYPES:
        raise ValueError(f"Unsupported camera type: {camera_type!r}; choose from {CAMERA_TYPES}")
    size = DEFAULT_CAMERA_SIZE if camera_type == "realsense" else ORBBEC_CAMERA_SIZE
    cls = RealSenseCameraStream if camera_type == "realsense" else OrbbecCameraStream
    return cls(
        serial, size[0] if width is None else width, size[1] if height is None else height, fps
    )


def capture_rgbd_frame(
    serial: str = "",
    width: int | None = None,
    height: int | None = None,
    fps: int = DEFAULT_CAMERA_FPS,
    stream: CameraStream | None = None,
    *,
    camera_type: str = DEFAULT_CAMERA_TYPE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Capture one aligned ``(color_bgr, depth_m, metadata)`` frame.

    Uses ``stream`` if given, otherwise opens the camera for a single frame.
    """
    if stream is not None:
        return stream.read_frame()
    with create_camera_stream(camera_type, serial, width, height, fps) as one_shot:
        return one_shot.read_frame()
