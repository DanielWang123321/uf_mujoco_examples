"""Shared Eye-in-Hand Orbbec Gemini 305 camera stream, projection, and look-down IK utilities."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Self

import cv2
import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent / "ufactory_mujoco"))

from ufactory_mujoco.calibration import (
    KinematicsCalibration,
    apply_calibration_to_model,
)
from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.ik import solve_ik
from ufactory_mujoco.safety import SafetyGateError
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

DEFAULT_HANDEYE_YAML = REPO_ROOT.parent / "camera_tool" / "result" / "eye_in_hand.yaml"
CAMERA_TOOL_SRC = REPO_ROOT.parent / "camera_tool" / "src"
PY312_PYTHON = Path("/opt/conda_envs/py312/bin/python")

# Safe clear workspace on the green anti-static mat (x in [0.20, 0.60] m, y in [-0.30, 0.22] m)
WORKSPACE_BOUNDS_M: dict[str, float] = {
    "x_min": 0.20,
    "x_max": 0.60,
    "y_min": -0.30,
    "y_max": 0.22,
}


class PerceptionError(RuntimeError):
    """Raised when camera capture, hand-eye calibration, or object segmentation fails."""


def validate_xy_in_workspace(
    xy_m: np.ndarray | tuple[float, float],
    label: str,
    bounds: dict[str, float] = WORKSPACE_BOUNDS_M,
) -> None:
    """Raise PerceptionError if a 2D point in base frame falls outside safe workspace bounds."""
    x, y = float(xy_m[0]), float(xy_m[1])
    if not (bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]):
        raise PerceptionError(
            f"{label} ({x:.3f}, {y:.3f}) m 超出安全净空工作区 "
            f"X=[{bounds['x_min']:.2f}, {bounds['x_max']:.2f}], "
            f"Y=[{bounds['y_min']:.2f}, {bounds['y_max']:.2f}] m"
        )


def load_handeye_calibration(yaml_path: str | Path = DEFAULT_HANDEYE_YAML) -> dict[str, Any]:
    """Load and validate Eye-in-Hand calibration YAML."""
    raw = Path(yaml_path).expanduser()
    if not raw.is_absolute() and not raw.is_file() and (REPO_ROOT / raw).is_file():
        path = (REPO_ROOT / raw).resolve()
    else:
        path = raw.resolve()
    if not path.is_file():
        raise PerceptionError(f"手眼标定文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "camera" not in data or "extrinsics" not in data:
        raise PerceptionError(f"手眼标定文件格式无效: {path}")
    x_fc = np.asarray(data["extrinsics"]["X_flange_cam"], dtype=np.float64)
    if x_fc.shape != (4, 4) or not np.all(np.isfinite(x_fc)):
        raise PerceptionError(f"X_flange_cam 矩阵无效: {path}")
    return data


def deproject_pixels_to_cam(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    camera_cfg: dict[str, Any],
) -> np.ndarray:
    """Deproject distorted pixel coordinates (N, 2) and optical depth (N,) to camera 3D points."""
    pts_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depth_m, dtype=np.float64).reshape(-1, 1)
    k_mat = np.asarray(camera_cfg["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    undist = cv2.undistortPoints(pts_uv.reshape(-1, 1, 2), k_mat, dist).reshape(-1, 2)
    return np.column_stack([undist, np.ones(len(undist), dtype=np.float64)]) * z


def project_cam_to_pixels(points_cam: np.ndarray, camera_cfg: dict[str, Any]) -> np.ndarray:
    """Project 3D points in camera frame (N, 3) to 2D image pixels (N, 2)."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    k_mat = np.asarray(camera_cfg["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k_mat, dist)
    return proj.reshape(-1, 2)


def compute_flange_and_cam_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q6: np.ndarray,
    x_flange_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (T_base_flange, T_base_cam) 4x4 matrices for arm configuration q6."""
    data.qpos[:6] = np.asarray(q6, dtype=np.float64).reshape(6)
    mujoco.mj_forward(model, data)
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    t_bf = np.eye(4, dtype=np.float64)
    t_bf[:3, :3] = data.xmat[link6_id].reshape(3, 3)
    t_bf[:3, 3] = data.xpos[link6_id]
    t_bc = t_bf @ np.asarray(x_flange_cam, dtype=np.float64).reshape(4, 4)
    return t_bf, t_bc


class OrbbecCameraStream:
    """Persistent Orbbec Gemini 305 subprocess stream for fast (<0.15s) on-demand frame capture."""

    def __init__(
        self,
        serial: str = "CV2C861000RV",
        width: int = 1280,
        height: int = 800,
        fps: int = 30,
    ) -> None:
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._proc: subprocess.Popen[str] | None = None
        self.meta: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def start(self) -> dict[str, Any]:
        if self._proc is not None:
            return self.meta
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="orbbec_stream_")
        out_dir = self._tmp_dir.name
        worker_code = (
            "import json, sys, numpy as np\n"
            f"sys.path.insert(0, {str(CAMERA_TOOL_SRC)!r})\n"
            "from camera_tool.devices.orbbec import OrbbecSession\n"
            "cfg = json.loads(sys.argv[1])\n"
            "out_dir = sys.argv[2]\n"
            "s = OrbbecSession(cfg)\n"
            "meta = s.start()\n"
            "try:\n"
            "    for _ in range(10):\n"
            "        s.read()\n"
            "    with open(out_dir + '/meta.json', 'w', encoding='utf-8') as f:\n"
            "        json.dump(meta, f)\n"
            "    sys.stdout.write('READY\\n')\n"
            "    sys.stdout.flush()\n"
            "    for line in sys.stdin:\n"
            "        cmd = line.strip()\n"
            "        if cmd == 'QUIT':\n"
            "            break\n"
            "        if cmd == 'SNAP':\n"
            "            for _ in range(2):\n"
            "                s.read()\n"
            "            frame = s.read()\n"
            "            np.save(out_dir + '/color.npy', frame['color_bgr'])\n"
            "            np.save(out_dir + '/depth.npy', frame['depth_m'])\n"
            "            sys.stdout.write('OK\\n')\n"
            "            sys.stdout.flush()\n"
            "finally:\n"
            "    s.stop()\n"
        )
        python_exe = str(PY312_PYTHON) if PY312_PYTHON.is_file() else "python3"
        cfg_json = json.dumps(
            {
                "serial": self.serial,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "enable_depth": True,
            }
        )
        self._proc = subprocess.Popen(
            [python_exe, "-c", worker_code, cfg_json, out_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self._proc.stdout is not None
        ready = ""
        while True:
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                self.close()
                raise PerceptionError(f"Orbbec 常驻相机进程启动失败: {err.strip()}")
            if line.strip() == "READY":
                ready = "READY"
                break
        if ready != "READY":
            self.close()
            raise PerceptionError("Orbbec 常驻相机进程未能进入 READY 状态")
        self.meta = json.loads((Path(out_dir) / "meta.json").read_text(encoding="utf-8"))
        return self.meta

    def read_frame(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self._proc is None or self._tmp_dir is None:
            self.start()
        assert self._proc is not None and self._tmp_dir is not None
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write("SNAP\n")
            self._proc.stdin.flush()
            while True:
                resp = self._proc.stdout.readline()
                if not resp:
                    err = self._proc.stderr.read() if self._proc.stderr else ""
                    raise PerceptionError(f"Orbbec 常驻相机取帧中断: {err.strip()}")
                if resp.strip() == "OK":
                    break
            out_dir = Path(self._tmp_dir.name)
            color_bgr = np.load(out_dir / "color.npy")
            depth_m = np.load(out_dir / "depth.npy")
            return color_bgr, depth_m, self.meta
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write("QUIT\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except (OSError, subprocess.SubprocessError):
                self._proc.kill()
            self._proc = None
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None


def capture_rgbd_frame(
    serial: str = "CV2C861000RV",
    width: int = 1280,
    height: int = 800,
    fps: int = 30,
    stream: OrbbecCameraStream | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Capture an aligned (color_bgr, depth_m, metadata) frame from Orbbec Gemini 305."""
    if stream is not None:
        return stream.read_frame()
    helper_code = (
        "import json, sys, numpy as np\n"
        f"sys.path.insert(0, {str(CAMERA_TOOL_SRC)!r})\n"
        "from camera_tool.devices.orbbec import OrbbecSession\n"
        "cfg = json.loads(sys.argv[1])\n"
        "out_dir = sys.argv[2]\n"
        "s = OrbbecSession(cfg)\n"
        "meta = s.start()\n"
        "try:\n"
        "    for _ in range(10):\n"
        "        s.read()\n"
        "    frame = s.read()\n"
        "finally:\n"
        "    s.stop()\n"
        "np.save(out_dir + '/color.npy', frame['color_bgr'])\n"
        "np.save(out_dir + '/depth.npy', frame['depth_m'])\n"
        "with open(out_dir + '/meta.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(meta, f)\n"
    )
    python_exe = str(PY312_PYTHON) if PY312_PYTHON.is_file() else "python3"
    cfg_json = json.dumps(
        {
            "serial": serial,
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "enable_depth": True,
        }
    )
    with tempfile.TemporaryDirectory(prefix="orbbec_capture_") as tmp_dir:
        proc = subprocess.run(
            [python_exe, "-c", helper_code, cfg_json, tmp_dir],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "未知错误").strip()
            raise PerceptionError(f"Orbbec Gemini 305 相机取流失败: {detail}")
        color_bgr = np.load(Path(tmp_dir) / "color.npy")
        depth_m = np.load(Path(tmp_dir) / "depth.npy")
        meta = json.loads((Path(tmp_dir) / "meta.json").read_text(encoding="utf-8"))
    return color_bgr, depth_m, meta


def solve_camera_lookdown_q(
    target_xy_m: tuple[float, float] | np.ndarray,
    *,
    z_cam_m: float = 0.280,
    q_seed: np.ndarray,
    scene_path: str | Path,
    calibration: KinematicsCalibration | None = None,
    handeye_yaml: str | Path = DEFAULT_HANDEYE_YAML,
    ignore_bodies: tuple[str, ...] = (),
) -> np.ndarray:
    """Solve collision-free joint angles that place the Eye-in-Hand camera directly over target_xy_m."""
    handeye = load_handeye_calibration(handeye_yaml)
    x_flange_cam = np.asarray(handeye["extrinsics"]["X_flange_cam"], dtype=np.float64)
    checker = CollisionChecker(
        scene_path,
        safety_margin=0.0,
        query_margin=0.01,
        ignore_pairs=[("world", "link_base"), ("world", "link1")],
        ignore_bodies=ignore_bodies,
    )
    if calibration is not None:
        apply_calibration_to_model(checker.model, calibration, checker.data)

    c_xy = np.asarray(target_xy_m, dtype=np.float64).reshape(2)
    model = checker.model
    data = mujoco.MjData(model)
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
    site_name = model.site(site_id).name

    quat = GRIP_DOWN_QUAT.copy()
    q_full = data.qpos.copy()
    q_full[:6] = np.asarray(q_seed, dtype=np.float64).reshape(6)
    q_tmp, ok = solve_ik(
        model,
        data,
        site_name,
        np.array([0.40, 0.0, 0.25], dtype=np.float64),
        quat,
        q_init=q_full,
    )
    if not ok:
        raise SafetyGateError("相机正视观测位参考姿态逆解失败")

    data.qpos[:] = q_tmp
    mujoco.mj_forward(model, data)
    t_bf = np.eye(4, dtype=np.float64)
    t_bf[:3, :3] = data.xmat[link6_id].reshape(3, 3)
    t_bf[:3, 3] = data.xpos[link6_id]
    t_bc = t_bf @ x_flange_cam
    tcp_tmp = data.site_xpos[site_id].copy()
    offset_cam_minus_tcp = t_bc[:3, 3] - tcp_tmp
    target_tcp = (
        np.array([c_xy[0], c_xy[1], float(z_cam_m)], dtype=np.float64)
        - offset_cam_minus_tcp
    )
    q_sol, ok = solve_ik(model, data, site_name, target_tcp, quat, q_init=q_tmp)
    if not ok:
        raise SafetyGateError(
            f"无法求解目标 ({c_xy[0]:.3f}, {c_xy[1]:.3f}) 正上方相机观测逆解"
        )
    q6 = np.asarray(q_sol[:6], dtype=np.float64).copy()
    if checker.in_collision(q6):
        raise SafetyGateError("相机正上方观测位姿检测到碰撞风险")
    return q6

