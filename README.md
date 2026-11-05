# uf_mujoco_examples

[English](README.md) | [中文](README_zh.md)

Contact-rich MuJoCo manipulation and closed-loop **Sim-to-Real** examples for the **UFACTORY xArm 6** robot arm, **Gripper G2** 4-bar parallel gripper, and **Orbbec Gemini 305** Eye-in-Hand RGB-D camera.

Built on top of [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco), this repository demonstrates high-precision deformable/thin-wall grasping, multi-object coaxial nesting, and 3D helical screw-nut threading with tactile Z-axis closed-loop adaptation.

---

## Highlights

- **Contact-Rich Physics Modeling**:
  - **Hollow Paper Cup (`5.74 g`)**: 24-segment tilted frustum collision wall (`1 mm` thickness) paired with decoupled 3-tier contact priorities ($\mu_{\text{cup-cup}} = 0.08 < \tan\alpha = 0.125$) for self-centering coaxial nesting without jamming.
  - **M12 Helical Screw & Hex Flange Nut (`56.5 g` / `29.0 g`)**: 12-turn continuous right-hand 3D capsule thread ridge (`2.40 mm` pitch, `192` segments) engaged by 8 internal helical follower sphere teeth, supporting mid-shaft mechanical self-locking and bright chrome-steel visual rendering.
- **Closed-Loop Eye-in-Hand Perception (`common/vision.py`)**:
  - Automated Orbbec Gemini 305 RGB-D capture and base-frame point-cloud deprojection.
  - **2-Stage Overhead Refinement**: Uses `solve_camera_lookdown_q` (`z_cam = 250 mm`) directly above target objects to eliminate oblique parallax before grasping or insertion.
  - **Post-Task Closed-Loop Verification**: Automatically returns to the observation pose after physical execution to verify upright placement, stack coaxiality, or threaded nut height.
- **Tactile Z-Axis Closed-Loop Adaptation & Soft-Landing Motion (`common/motion.py`)**:
  - Monotonic soft-landing velocity profiling (`smooth_progress_profile` and curvature-bounded `interpolate_waypoint_chain` with terminal deceleration).
  - Real-time Gripper G2 jaw-opening feedback (`get_measured_gripper_gap`) detects empty grasps, bottom-flange collisions (`> 21.2 mm` on a `Ø20 mm` hex body), and thread-pitch lag, automatically executing upward Z-backoff (`+3.5 mm`) and adaptive extra ratchet strokes.
- **3-Stage Sim-to-Real Safety Gate**:
  1. `dry-run`: Real camera perception + calibrated MuJoCo digital-twin preflight (zero hardware motion).
  2. `sdk-sim`: xArm controller internal simulation (`mode=4, state=5`, physical arm stationary).
  3. `real --confirm-real`: Single-cycle physical execution gated by digital-twin clearance (`>= 5 mm`), camera table clearance (`>= 88–95 mm`), and viewer abort protection.

---

## Examples Matrix

| ID | Task | Directory | Core Capabilities | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Disposable Paper Cup Pick & Upright Place** | [`examples/pick_cup`](examples/pick_cup) | Sideways frustum 6D pose estimation, compliant `15%`-force waist grasp, collision-free `90°` in-air flip, upright placement & post-place verification | [English](examples/pick_cup/README.md) \| [中文](examples/pick_cup/README_zh.md) |
| **02** | **Multi-Cup Coaxial Nesting Stack** | [`examples/stack_cups`](examples/stack_cups) | Multi-cup point-cloud segmentation, right-to-left grasp ordering, rim-inset anti-slip release (`4 mm` below lower rim), 2- or 3-cup coaxial stacking | [English](examples/stack_cups/README.md) \| [中文](examples/stack_cups/README_zh.md) |
| **03** | **M12 Helical Screw-Nut Threading** | [`examples/screw_nut`](examples/screw_nut) | 12-turn 3D helical thread engagement, multi-stroke ratcheting within `J6 ∈ [-360°, +360°]`, overhead re-measurement, tactile Z-axis closed-loop adaptation | [English](examples/screw_nut/README.md) \| [中文](examples/screw_nut/README_zh.md) |

---

## Installation & Workspace Layout

### 1. Clone Sibling Repositories

This repository references robot models and core kinematics/safety libraries from [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco) placed in the same parent directory:

```bash
mkdir -p ~/workspace && cd ~/workspace
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git
git clone https://github.com/DanielWang123321/uf_mujoco_examples.git
```

Expected workspace directory layout:

```text
workspace/
├── ufactory_mujoco/          # Core UFACTORY MuJoCo models, IK, trajectory & safety library
├── uf_mujoco_examples/       # This repository (contact-rich tasks & Sim-to-Real pipelines)
└── camera_tool/              # (Optional, for real hardware only) Eye-in-Hand calibration & Orbbec wrapper
```

### 2. Python Environment Setup

Supports **Python 3.12, 3.13, and 3.14** on Linux (Ubuntu 22.04 / 24.04 recommended):

```bash
cd ~/workspace/uf_mujoco_examples

# Install core ufactory_mujoco package and example dependencies
pip install -e ../ufactory_mujoco
pip install -e ".[dev]"
```

---

## Quick Start

All examples can be launched either as direct scripts (`python examples/<task>/<task>.py`) or as Python modules (`python -m examples.<task>.run`).

### 1. Pure MuJoCo Simulation (`--mode sim`)

```bash
# Example 01: Pick up a sideways paper cup, flip 90°, and place upright
python -m examples.pick_cup.run --mode sim

# Example 02: Pick and coaxially stack multiple paper cups (2 or 3 cups)
python -m examples.stack_cups.run --mode sim --num-cups 2
python -m examples.stack_cups.run --mode sim --num-cups 3 --headless

# Example 03: Pick up a hex flange nut and ratchet-thread 10 turns onto an M12 screw
python -m examples.screw_nut.run --mode sim
python -m examples.screw_nut.run --mode sim --headless --thread-turns 3.0
```

> Pass `--headless` to any command to run offscreen without opening a GUI window, and `--report build/<task>/sim_report.json` to export structured metrics.

### 2. Three-Stage Sim-to-Real Execution

Before running on physical hardware, configure `robot.ip`, `robot.calibration`, and `camera.handeye` in `examples/<task>/config.yaml` (or pass `--ip`, `--calibration`, `--handeye` on the CLI):

```bash
# Stage 1: Real Gemini 305 perception + MuJoCo digital-twin preflight (zero hardware motion)
python -m examples.screw_nut.run --mode dry-run

# Stage 2: xArm controller internal simulation (mode=4, state=5; physical arm stays still)
python -m examples.screw_nut.run --mode sdk-sim

# Stage 3: Full closed-loop physical execution (requires explicit --confirm-real flag)
python -m examples.screw_nut.run --mode real --confirm-real
```

---

## Repository Structure

```text
uf_mujoco_examples/
├── assets/
│   ├── cup/                  # Procedural hollow paper cup generator, OBJ/STL meshes & standalone MJCF
│   ├── gripper_g2/           # Double-sided smooth-normal Gripper G2 visual OBJs & collision STLs
│   └── screw_nut/            # M12 helical screw & hex flange nut generator, chrome texture & asset tests
├── common/
│   ├── motion.py             # Soft-landing progress curves, curvature-bounded interpolation & real streaming
│   └── vision.py             # Orbbec Gemini 305 stream, Eye-in-Hand projection & overhead look-down IK
├── examples/
│   ├── pick_cup/             # 01: Single paper cup pick, 90° flip & upright placement (Sim + Sim-to-Real)
│   ├── stack_cups/           # 02: Multi-cup pick, flip & coaxial nesting stack (Sim + Sim-to-Real)
│   └── screw_nut/            # 03: M12 screw-nut threading & tactile Z-axis closed-loop control
├── LICENSE                   # MIT License
├── NOTICE                    # Third-party asset & upstream license notices
├── pyproject.toml            # Package metadata, dependencies, pytest & ruff configuration
└── requirements.txt          # Core runtime dependencies
```

---

## Testing & Code Quality

Run the automated test suite (covering 3D asset geometry, contact stability, multi-cup stacking, helical thread self-locking, tactile Z-axis closed-loop adaptation, and Sim-to-Real safety gates) and linter from the repository root:

```bash
# Run all unit and physics simulation tests
pytest -v

# Run Ruff linter
ruff check assets/ common/ examples/
```

---

## License

The source code and procedural generators in this repository are licensed under the [MIT License](LICENSE). Derived robot/gripper mesh assets are covered by the upstream licenses documented in [NOTICE](NOTICE).

