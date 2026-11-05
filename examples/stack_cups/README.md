# 02 — Multi-Cup Pick & Coaxial Nesting Stack (Simulation & Sim-to-Real)

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation and closed-loop Sim-to-Real example for **UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305 (Eye-in-Hand)**: sequentially picks up multiple sideways disposable paper cups (`5.74 g` mass, `53 mm` bottom diameter, `75 mm` top diameter, `88 mm` height, `1 mm` wall thickness), flips each cup `90°` in mid-air so the rim faces up, places the base cup upright at the target coordinates, and coaxially nests subsequent cups inside the lower cups.

## Configuration (`config.yaml`)

Default configuration is located at `examples/stack_cups/config.yaml`:

```yaml
mode: sim
headless: false  # false: open 3D MuJoCo viewer; true: run offscreen

robot:
  ip: "192.168.1.60"
  calibration: "../ufactory_mujoco/calibrations/xarm6_kinematics_L56A01.yaml"

camera:
  handeye: "../camera_tool/result/eye_in_hand.yaml"

task:
  num_cups: 2              # Number of cups to pick and stack (supports 2 or 3, default 2)
  target_xy: [0.30, 0.20]  # Target stack horizontal coordinates [x_m, y_m]
  post_verify: true        # Automatically capture a post-stack RGB-D verification frame on real robot
  save_vis: "build/stack_cups/real_detection.png"
  save_after_vis: "build/stack_cups/real_after_stack.png"
  report: "build/stack_cups/sim_report.json"
```

## Quick Start Commands

Run from the repository root:

```bash
# 1. Pure MuJoCo simulation (default 2-cup coaxial stacking with 3D interactive viewer)
python -m examples.stack_cups.run --mode sim
python -m examples.stack_cups.run --mode sim --headless --num-cups 2 --report build/stack_cups/sim_2cups.json
python -m examples.stack_cups.run --mode sim --headless --num-cups 3 --report build/stack_cups/sim_3cups.json

# 2. Sim-to-Real Stage 1: Gemini 305 multi-cup perception + MuJoCo digital-twin preflight (zero hardware motion)
python -m examples.stack_cups.run --mode dry-run --num-cups 2 --report build/stack_cups/dry_run.json

# 3. Sim-to-Real Stage 2: xArm controller internal simulation (mode=4, state=5; physical arm stationary)
python -m examples.stack_cups.run --mode sdk-sim --num-cups 2 --report build/stack_cups/sdk_sim.json

# 4. Sim-to-Real Stage 3: Multi-cup perception -> digital-twin preflight -> physical stacking -> post-stack verification
python -m examples.stack_cups.run --mode real --confirm-real --num-cups 2 --report build/stack_cups/real.json

# Optional: Manually specify multiple sideways cup poses (X1 Y1 YAW1 X2 Y2 YAW2 ...)
python -m examples.stack_cups.run --mode dry-run --num-cups 2 \
  --cup-poses 0.568 -0.168 -142.2 0.503 0.081 -61.0 \
  --target-xy 0.36 -0.06
```

## Physics & Control Architecture

- **Eye-in-Hand Multi-Cup 6D Pose Estimation & Post-Stack Verification** (`examples/stack_cups/perception.py`):
  - Segments multiple white paper cup clusters in the safe workspace (`x ∈ [0.20, 0.60] m, y ∈ [-0.26, 0.12] m`), estimates each cup's 2D PCA axis and bottom-to-top orientation, and sorts grasp order from right (`-Y`) to left (`+Y`).
  - After physical stacking, captures a second RGB-D frame (`verify_stacked_cups`) to verify stack center alignment, top-rim height, and source area clearance.
- **3-Tier Contact Priority Decoupling**:
  - **Gripper G2 finger pads** (`priority=2, friction=4.0`): High tangential friction for slip-free grasping and `90°` flipping.
  - **Tabletop** (`priority=2, friction=1.5`): Prevents sideways cups from rolling and anchors the base cup during nesting.
  - **Cup walls** (`priority=1, friction=0.08`): Satisfies $\mu_{\text{cup-cup}} = 0.08 < \tan\alpha = 0.125$ (frustum half-cone angle $\alpha \approx 7.125^\circ$), allowing upper cups to self-center and slide smoothly into the lower cup upon release.
- **Rim-Inset Anti-Slip Release**:
  - For cup $k \ge 2$, the arm aligns coaxially `25 mm` above the lower cup's rim and descends until the upper cup's bottom reaches $z_{\text{bot}, k} = z_{\text{rim}, k-1} - 4\text{ mm}$ before opening the gripper. The `53 mm` cup base is enclosed `4 mm` inside the `73 mm` inner mouth while the gripper fingers remain `~22 mm` above the rim.

## Running Tests

```bash
pytest examples/stack_cups/test_stack_cups.py -v
```
