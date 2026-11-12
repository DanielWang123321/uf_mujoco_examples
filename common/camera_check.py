# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Read-only RGB-D health check: python -m common.camera_check --duration 30."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.camera import CAMERA_TYPES, RealSenseCameraStream, create_camera_stream
from common.sim2real import write_json_report


class _CheckedRealSenseStream(RealSenseCameraStream):
    """Keep the selected aligned SDK depth frame for independent distance checks."""

    def _pull(self) -> tuple[Any, Any]:
        color, self.sdk_depth = super()._pull()
        return color, self.sdk_depth


def check_camera(
    *,
    duration: float = 30.0,
    camera_type: str = "realsense",
    serial: str = "",
    output: Path = Path("build/camera_check"),
) -> dict[str, Any]:
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    output.mkdir(parents=True, exist_ok=True)
    stream = (
        _CheckedRealSenseStream(serial)
        if camera_type == "realsense"
        else create_camera_stream(camera_type, serial)
    )
    report: dict[str, Any] = {"ok": False, "captured_frames": 0}
    try:
        with stream:
            start = time.monotonic()
            previous_number = -1
            previous_timestamp = -1.0
            max_distance_error = 0.0
            valid_fractions = []
            while report["captured_frames"] == 0 or time.monotonic() - start < duration:
                bgr, depth, meta = stream.read_frame()
                assert bgr.dtype == np.uint8 and depth.dtype == np.float32
                assert bgr.shape == (meta["height"], meta["width"], 3)
                assert depth.shape == bgr.shape[:2]
                if (
                    meta["frame_number"] <= previous_number
                    or meta["timestamp_ms"] <= previous_timestamp
                ):
                    raise RuntimeError("Camera returned a stale frame")
                previous_number, previous_timestamp = meta["frame_number"], meta["timestamp_ms"]
                if isinstance(stream, _CheckedRealSenseStream):
                    h, w = depth.shape
                    for y in np.linspace(0, h - 1, 5, dtype=int):
                        for x in np.linspace(0, w - 1, 5, dtype=int):
                            error = abs(
                                float(depth[y, x]) - stream.sdk_depth.get_distance(int(x), int(y))
                            )
                            max_distance_error = max(max_distance_error, error)
                valid_fractions.append(float(np.mean(depth > 0)))
                report["captured_frames"] += 1
                report["camera"] = meta
            report.update(
                duration_s=time.monotonic() - start,
                valid_depth_fraction_min=min(valid_fractions),
                valid_depth_fraction_mean=float(np.mean(valid_fractions)),
                sdk_distance_max_error_m=max_distance_error if camera_type == "realsense" else None,
            )
            if max_distance_error > 1e-6:
                raise RuntimeError(
                    f"Depth conversion differs from SDK distance: {max_distance_error} m"
                )
            if not cv2.imwrite(str(output / "color.png"), bgr):
                raise RuntimeError("Failed to write color image")
            np.savez_compressed(output / "frame.npz", color_bgr=bgr, depth_m=depth)
            report["ok"] = True
    except Exception as exc:
        report.setdefault("camera", dict(stream.meta))
        report["error"] = str(exc)
        raise
    finally:
        write_json_report(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-type", choices=CAMERA_TYPES, default="realsense")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("build/camera_check"))
    args = parser.parse_args()
    report = check_camera(
        duration=args.duration,
        camera_type=args.camera_type,
        serial=args.camera_serial,
        output=args.output,
    )
    print(
        f"PASS: {report['captured_frames']} frames in {report['duration_s']:.1f}s; {args.output}/report.json"
    )


if __name__ == "__main__":
    main()
