# uf_mujoco_examples

[English](README.md) | [中文](README_zh.md)

MuJoCo manipulation examples for the **UFACTORY xArm 6** with **Gripper G2**. Each example runs in simulation and can be executed on a real robot with an **Orbbec Gemini 305** wrist camera (eye-in-hand).

Objects are moved only through simulated contacts (no welds or attachments). Every hardware motion is first rehearsed in a MuJoCo digital twin that uses the kinematic calibration of your robot.

## Examples

| Example | Task | Docs |
| :--- | :--- | :--- |
| [`pick_cup`](examples/pick_cup) | Pick a paper cup lying on its side, flip it in the air, place it upright. | [EN](examples/pick_cup/README.md) / [中文](examples/pick_cup/README_zh.md) |
| [`stack_cups`](examples/stack_cups) | Pick 2–3 paper cups and nest them into one stack. | [EN](examples/stack_cups/README.md) / [中文](examples/stack_cups/README_zh.md) |
| [`screw_nut`](examples/screw_nut) | Pick a hex flange nut and thread it onto an M12 screw fixed on the table. | [EN](examples/screw_nut/README.md) / [中文](examples/screw_nut/README_zh.md) |

All examples support the same run modes:

| Mode | What runs | Robot moves? |
| :--- | :--- | :--- |
| `sim` (default) | MuJoCo simulation only | No |
| `dry-run` | Camera detection + digital-twin preflight | No |
| `sdk-sim` | The preflight joint stream is replayed in the xArm controller's simulation mode | No |
| `real` | Full task on the robot; requires `--confirm-real` | Yes |

## Requirements

- Linux, Python 3.12–3.14
- Simulation: no hardware needed
- Hardware modes: xArm 6 + Gripper G2, Orbbec Gemini 305 mounted on the gripper, and the setup in [Hardware setup](#hardware-setup)

## Installation

This repository uses robot models, IK, collision checking, and safety gates from [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco). Clone both repositories into the same folder:

```bash
mkdir workspace && cd workspace
git clone https://github.com/DanielWang123321/ufactory_mujoco.git
git clone https://github.com/DanielWang123321/uf_mujoco_examples.git

cd uf_mujoco_examples
pip install -e ".[dev]"          # simulation + tests
pip install -e ".[dev,real]"     # + xArm SDK and Orbbec SDK for the hardware modes
```

`ufactory_mujoco` does not need to be installed: the scenes include its models by relative path, and `common/paths.py` adds it to `sys.path`.

## Usage

Run from the repository root:

```bash
python -m examples.pick_cup                      # MuJoCo viewer
python -m examples.pick_cup --headless           # no window
python -m examples.pick_cup --help               # all options

# On hardware, in this order:
python -m examples.pick_cup --mode dry-run
python -m examples.pick_cup --mode sdk-sim
python -m examples.pick_cup --mode real --confirm-real
```

Option priority: command line > `examples/<task>/config.yaml` > built-in defaults. `--report <file>.json` saves a run report.

## Hardware setup

1. **Calibration files** in [`calibration/`](calibration/README.md):
   - `xarm6_kinematics.yaml`: kinematic calibration read from your xArm controller with `ufactory_mujoco/scripts/gen_kinematics_params.py`;
   - `eye_in_hand.yaml`: camera intrinsics and camera pose in the flange frame (format: [`eye_in_hand.example.yaml`](calibration/eye_in_hand.example.yaml)).
2. **Robot IP**: `robot.ip` in `config.yaml` or `--ip` (default `192.168.1.60`).
3. **Camera**: connect the Gemini 305 to a USB 3 port. If it is not found, install the udev rules shipped with `pyorbbecsdk2`:
   ```bash
   sudo bash "$(python -c 'import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))')/shared/install_udev_rules.sh"
   ```
   With several cameras, set `camera.serial` in `config.yaml` or use `--camera-serial`.
4. **Scene and initial pose**: place the robot at the initial hover TCP pose `[300, 0, 300, 180, 0, 0]` (`[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]`) so the wrist camera overlooks the workspace; each run returns to this starting pose. The perception is tuned for white paper cups and metal parts on a green table mat inside X ∈ [0.20, 0.60] m, Y ∈ [−0.30, 0.22] m (robot base frame).

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
│   ├── camera.py           # Orbbec RGB-D capture
│   ├── vision.py           # hand-eye calibration, projection, camera look-down IK
│   ├── sim2real.py         # collision checker, config hash, JSON report
│   └── objects/            # object models shared by several examples (paper cup)
├── examples/<task>/
│   ├── __main__.py         # python -m examples.<task>
│   ├── <task>.py           # simulation task + CLI
│   ├── perception.py       # camera detection and post-task verification
│   ├── sim2real.py         # digital-twin preflight, sdk-sim, real execution
│   ├── scene.xml, config.yaml, assets/ (symlinks)
│   └── test_<task>.py
├── assets/                 # meshes and generators (paper cup, screw/nut, Gripper G2)
└── calibration/            # per-robot calibration files (not tracked by git)
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
