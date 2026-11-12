# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""YAML config loading and command-line options shared by every example.

Priority for each option: command-line flag > ``config.yaml`` > built-in default.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from common.camera import CAMERA_TYPES, DEFAULT_CAMERA_TYPE
from common.paths import (
    DEFAULT_CALIBRATION_YAML,
    DEFAULT_ROBOT_IP,
    default_handeye_path,
    resolve_repo_path,
)

MODES = ("sim", "dry-run", "sdk-sim", "real")
HARDWARE_MODES = ("dry-run", "sdk-sim", "real")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping; returns ``{}`` if the file is missing or not a mapping."""
    path = Path(config_path)
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def build_parser(description: str | None, *, default_config: Path) -> argparse.ArgumentParser:
    """Create a parser pre-populated with the options shared by all examples."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(default_config),
        help=f"YAML config file (default: {default_config.name})",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="sim: MuJoCo only; dry-run: real camera + digital-twin preflight (no motion); "
        "sdk-sim: xArm controller simulation (no motion); real: execute on hardware",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run MuJoCo without the viewer window",
    )
    parser.add_argument("--ip", default=None, help="xArm controller IP")
    parser.add_argument("--calibration", default=None, help="xArm kinematic calibration YAML")
    parser.add_argument("--handeye", default=None, help="Eye-in-Hand calibration YAML")
    parser.add_argument(
        "--camera-type",
        choices=CAMERA_TYPES,
        default=None,
        help="RGB-D camera backend (default: realsense / D435)",
    )
    parser.add_argument(
        "--camera-serial", default=None, help="SDK camera serial number (default: first found)"
    )
    parser.add_argument(
        "--confirm-real", action="store_true", help="Authorize one real motion cycle (--mode real)"
    )
    parser.add_argument(
        "--cycles", type=int, default=1, help="Number of cycles (hardware modes: 1)"
    )
    parser.add_argument(
        "--post-verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Re-capture the scene after a real run and verify the result",
    )
    parser.add_argument(
        "--save-vis", default=None, help="Save the pre-grasp detection overlay (PNG)"
    )
    parser.add_argument(
        "--save-after-vis", default=None, help="Save the post-task verification overlay (PNG)"
    )
    parser.add_argument("--report", default=None, help="Save a structured run report (JSON)")
    return parser


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name)
    return value if isinstance(value, dict) else {}


def _pick(cli_value: Any, cfg_value: Any) -> Any:
    return cli_value if cli_value is not None else cfg_value


@dataclass
class RunOptions:
    """Options resolved from the CLI and ``config.yaml``."""

    mode: str
    headless: bool
    ip: str
    calibration_path: str
    handeye_path: str
    camera_type: str
    camera_serial: str
    confirm_real: bool
    cycles: int
    post_verify: bool
    save_vis: str | None
    save_after_vis: str | None
    report_path: str | None
    task: dict[str, Any] = field(default_factory=dict)
    """The raw ``task:`` section for example-specific keys."""


def resolve_run_options(args: argparse.Namespace) -> RunOptions:
    """Merge parsed CLI arguments with the YAML file given by ``--config``."""
    cfg = load_config(args.config)
    robot_cfg = _section(cfg, "robot")
    cam_cfg = _section(cfg, "camera")
    task_cfg = _section(cfg, "task")
    camera_type = str(_pick(args.camera_type, cam_cfg.get("type", DEFAULT_CAMERA_TYPE)))
    handeye_default = default_handeye_path(camera_type)
    return RunOptions(
        mode=str(_pick(args.mode, cfg.get("mode", "sim"))),
        headless=bool(_pick(args.headless, cfg.get("headless", False))),
        ip=str(_pick(args.ip, robot_cfg.get("ip", DEFAULT_ROBOT_IP))),
        calibration_path=str(
            resolve_repo_path(
                _pick(args.calibration, robot_cfg.get("calibration", DEFAULT_CALIBRATION_YAML))
            )
        ),
        handeye_path=str(
            resolve_repo_path(_pick(args.handeye, cam_cfg.get("handeye")) or handeye_default)
        ),
        camera_type=camera_type,
        camera_serial=str(_pick(args.camera_serial, cam_cfg.get("serial")) or ""),
        confirm_real=bool(args.confirm_real),
        cycles=int(args.cycles),
        post_verify=bool(_pick(args.post_verify, task_cfg.get("post_verify", True))),
        save_vis=_pick(args.save_vis, task_cfg.get("save_vis")),
        save_after_vis=_pick(args.save_after_vis, task_cfg.get("save_after_vis")),
        report_path=_pick(args.report, task_cfg.get("report")),
        task=task_cfg,
    )


def run_cli(body: Callable[[], int | None]) -> int:
    """Run an example entry point; print any failure to stderr and return a process exit code."""
    try:
        result = body()
    except Exception as exc:  # noqa: BLE001 - the CLI must report any failure and return nonzero
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0 if result is None else int(result)
