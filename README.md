# uf_mujoco_examples

[English](README.md) | [中文](README_zh.md)

MuJoCo manipulation and motion-planning examples for the **UFACTORY xArm 6** with **Gripper G2**. All examples run in simulation. The cup and nut tasks also support execution on a real robot with a **RealSense D435/D435i** (default) or **Orbbec Gemini 305** wrist camera (eye-in-hand).

Objects are moved only through simulated contacts (no welds or attachments). Every hardware motion is first rehearsed in a MuJoCo digital twin that uses the kinematic calibration of your robot.

## Examples

| Example | Task | Docs |
| :--- | :--- | :--- |
| [`pick_cup`](examples/pick_cup) | Pick a paper cup lying on its side, flip it in the air, place it upright. | [EN](examples/pick_cup/README.md) / [中文](examples/pick_cup/README_zh.md) |
| [`stack_cups`](examples/stack_cups) | Pick 2–3 paper cups and nest them into one stack. | [EN](examples/stack_cups/README.md) / [中文](examples/stack_cups/README_zh.md) |
| [`screw_nut`](examples/screw_nut) | Pick a hex flange nut and thread it onto an M12 screw fixed on the table. | [EN](examples/screw_nut/README.md) / [中文](examples/screw_nut/README_zh.md) |
| [`obstacle_avoidance`](examples/obstacle_avoidance) | Grasp a block, carry it around a table-mounted cylinder, and place it; **sim only**. | [EN](examples/obstacle_avoidance/README.md) / [中文](examples/obstacle_avoidance/README_zh.md) |

`pick_cup`, `stack_cups`, and `screw_nut` support these run modes. `obstacle_avoidance` supports only `sim`:

| Mode | What runs | Robot moves? |
| :--- | :--- | :--- |
| `sim` (default) | MuJoCo simulation only | No |
| `dry-run` | Camera detection + digital-twin preflight | No |
| `sdk-sim` | The preflight joint stream is replayed in the xArm controller's simulation mode | No |
| `real` | Full task on the robot; requires `--confirm-real` | Yes |

## Requirements

- Linux, Python 3.12–3.14
- Simulation: no hardware needed
- Hardware modes: xArm 6 + Gripper G2, D435/D435i or Gemini 305 mounted on the gripper, and the setup in [Hardware setup](#hardware-setup)

## Installation

This repository uses robot models, IK, collision checking, and safety gates from [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco). Clone both repositories into the same folder:

```bash
mkdir workspace && cd workspace
git clone https://github.com/DanielWang123321/ufactory_mujoco.git
git clone https://github.com/DanielWang123321/uf_mujoco_examples.git

cd uf_mujoco_examples
pip install -e ".[dev]"          # simulation + tests
pip install -e ".[dev,real]"     # + xArm, RealSense, and Orbbec SDKs (camera imports are lazy)
```

`ufactory_mujoco` does not need to be installed: the scenes include its models by relative path, and `common/paths.py` adds it to `sys.path`.

## Usage

Run from the repository root:

```bash
python -m examples.pick_cup                      # MuJoCo viewer
python -m examples.pick_cup --headless           # no window
python -m examples.pick_cup --help               # all options
python -m examples.obstacle_avoidance --headless # static obstacle planning, sim only

# On hardware, in this order:
python -m examples.pick_cup --mode dry-run
python -m examples.pick_cup --mode sdk-sim
python -m examples.pick_cup --mode real --confirm-real
```

Option priority: command line > `examples/<task>/config.yaml` > built-in defaults. `--report <file>.json` saves a run report.

`camera.type` / `--camera-type` defaults to `realsense` (D435/D435i, 1280×720@30). Use `--camera-type orbbec` for Gemini 305 (1280×800@30). `--camera-serial` selects the serial returned by that camera's SDK. An empty `camera.handeye` selects the default for the backend; an explicit YAML path or `--handeye` overrides it.

```bash
python -m common.camera_check --duration 30                    # read-only RGB-D check
python -m examples.pick_cup --mode dry-run --camera-type orbbec
```

Live detection, near measurements, and post-task verification use the SDK's color intrinsics, distortion, and image size. Offline perception calls can use YAML intrinsics, or pass `camera_metadata` from capture. Depth is aligned to color and expressed in meters. Zero distortion uses pinhole geometry; unsupported nonzero models raise an error. Hardware reports include camera model, SDK serial, actual stream profiles, intrinsics, depth scale, per-capture metadata, the synchronized flange pose, and the hand-eye file SHA-256. Requested overlays also save the raw `.input.png` and replayable `.rgbd.npz` input. See the [RealSense SDK geometry conventions](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0).

For acceptance, run single cup, two cups (`--num-cups 2`), three cups (`--num-cups 3`), and screw/nut separately in `dry-run → sdk-sim → real --confirm-real` order, with `--post-verify`. Use a separate `build/<scene>/<mode>/` directory for `--save-vis`, `--save-after-vis`, `--report`, and redirected terminal logs. Only real runs produce post-task images.

## Hardware setup

1. **Calibration files** in [`calibration/`](calibration/README.md):
   - `xarm6_kinematics.yaml`: kinematic calibration read from your xArm controller with `ufactory_mujoco/scripts/gen_kinematics_params.py`;
   - D435: [`eye_in_hand.d435.yaml`](calibration/eye_in_hand.d435.yaml), 20-pose Bundle Adjustment flange-to-color-optical extrinsics for the local D435i mounting (verified at startup with Gripper G2 SDK `tcp_offset = [0, 0, 172, 0, 0, 0]`); see [calibration notes](calibration/README.md) before changing the bracket.
   - Gemini: `eye_in_hand.yaml` (format: [`eye_in_hand.example.yaml`](calibration/eye_in_hand.example.yaml)).
2. **Robot IP**: `robot.ip` in `config.yaml` or `--ip` (default `192.168.1.60`).
3. **Camera**: connect the selected camera to USB 3. Install RealSense udev rules using the [official Linux instructions](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md) if required. For Gemini 305, install the rules shipped with `pyorbbecsdk2`:
   ```bash
   sudo bash "$(python -c 'import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))')/shared/install_udev_rules.sh"
   ```
   With several cameras, set `camera.serial` in `config.yaml` or use `--camera-serial`.
4. **Scene and initial pose**: place the robot at the initial hover TCP pose `[300, 0, 400, 180, 0, 0]` for D435 (`[300, 0, 300, 180, 0, 0]` for Gemini) (`[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]`) so the wrist camera overlooks the workspace; each run returns to this starting pose. Keep every cup fully inside the detection view; a silhouette touching its boundary is rejected. Adjust the observation height or object positions if needed. The perception is tuned for white paper cups and metal parts on a green table mat inside X ∈ [0.20, 0.60] m, Y ∈ [−0.30, 0.22] m (robot base frame).

> [!CAUTION]
> `real` mode moves the robot. Keep the emergency stop within reach, run `dry-run` and `sdk-sim` first, and check that the workspace is clear. `--confirm-real` authorizes one cycle only.

## Repository layout

```text
uf_mujoco_examples/
├── common/                 # shared code (no task-specific logic)
│   ├── paths.py            # repository paths, default IP and calibration files
│   ├── cli.py              # shared CLI options and config.yaml loading
│   ├── sim.py              # TaskSimRobot base class, contact helpers
│   ├── geometry.py         # quaternion helpers
│   ├── motion.py           # joint-stream smoothing, soft landing, real-robot streaming
│   ├── camera.py           # RealSense / Orbbec RGB-D capture
│   ├── camera_check.py     # read-only RGB-D stream and depth health check
│   ├── vision.py           # hand-eye calibration, projection, camera look-down IK
│   ├── sim2real.py         # collision checker, config hash, JSON report
│   └── objects/            # shared object models and perception helpers (paper cup)
├── examples/<task>/
│   ├── __main__.py         # python -m examples.<task>
│   ├── <task>.py           # simulation task + CLI
│   ├── perception.py       # camera detection and post-task verification
│   ├── sim2real.py         # digital-twin preflight, sdk-sim, real execution
│   ├── scene.xml, config.yaml, assets/ (symlinks)
│   └── test_<task>.py
├── assets/                 # meshes and generators (paper cup, screw/nut, Gripper G2)
├── calibration/            # hand-eye defaults/examples and local robot calibration
└── tests/                  # cross-task camera, perception, and digital-twin tests
```

### Adding an example

1. Create `examples/<task>/` with the files above (copy `pick_cup` as a template).
2. Subclass `common.sim.TaskSimRobot` for the scene; record joint waypoints through `active_waypoint_sink` so the same motion can be replayed on hardware.
3. Build the CLI with `common.cli.build_parser` / `resolve_run_options` / `run_cli`.
4. Put code needed by more than one example in `common/`, not in another example.
5. Add `test_<task>.py`; `pytest` discovers it automatically.

## Tests

```bash
pytest              # simulation and synthetic-image tests, no hardware needed
ruff check . && ruff format --check .
```

## License

[BSD 3-Clause](LICENSE). Third-party material and upstream licenses: [NOTICE](NOTICE).
