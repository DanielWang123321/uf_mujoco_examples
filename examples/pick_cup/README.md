# 01 — Disposable Paper Cup Pick & Upright Place (Simulation & Sim-to-Real)

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation and closed-loop Sim-to-Real example for **UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305 (Eye-in-Hand)**: picks up a sideways disposable paper cup (`5.74 g` mass, `53 mm` bottom diameter, `75 mm` top diameter, `88 mm` height), flips it `90°` in mid-air, and places it upright on the tabletop with the bottom down and mouth facing up.

## Configuration (`config.yaml`)

Default configuration is located at `examples/pick_cup/config.yaml`:

```yaml
mode: sim
headless: false  # false: open 3D MuJoCo viewer; true: run offscreen

robot:
  ip: "192.168.1.60"
  calibration: "../ufactory_mujoco/calibrations/xarm6_kinematics_L56A01.yaml"

camera:
  handeye: "../camera_tool/result/eye_in_hand.yaml"

task:
  post_verify: true  # Capture a post-place RGB-D frame from observation pose for closed-loop verification
  save_vis: "build/pick_cup/real_detection.png"
  save_after_vis: "build/pick_cup/real_after_place.png"
```

## Quick Start Commands

Run from the repository root:

```bash
# 1. Pure MuJoCo simulation (opens 3D interactive viewer by default)
python -m examples.pick_cup.run --mode sim
python -m examples.pick_cup.run --mode sim --headless

# 2. Sim-to-Real Stage 1: Real perception + MuJoCo digital-twin preflight (zero hardware motion)
python -m examples.pick_cup.run --mode dry-run --report build/pick_cup/dry_run.json

# 3. Sim-to-Real Stage 2: xArm controller internal simulation (mode=4, state=5; physical arm stationary)
python -m examples.pick_cup.run --mode sdk-sim --report build/pick_cup/sdk_sim.json

# 4. Sim-to-Real Stage 3: Gemini 305 perception -> digital-twin preflight -> physical execution -> post-place verification
python -m examples.pick_cup.run --mode real --confirm-real --report build/pick_cup/real.json

# Optional: Manually specify sideways cup pose (X_M Y_M YAW_DEG, bypassing camera perception)
python -m examples.pick_cup.run --mode dry-run --cup-pose 0.44 -0.05 135.0
```

## Architecture & Control Details

- **Hollow Paper Cup Asset** (`assets/cup/generate_paper_cup.py`):
  - Generates a 1:1 visual mesh (`paper_cup.obj`) and hollow collision geometry (bottom cylinder `0.001180 kg` + 24 tilted frustum wall boxes `24 × 0.000190 kg`, total mass `0.00574 kg`).
- **Eye-in-Hand Perception & Post-Place Verification** (`examples/pick_cup/perception.py`):
  - Captures aligned RGB-D frames from the Orbbec Gemini 305 and deprojects points into the robot base frame using the Eye-in-Hand calibration YAML.
  - Segments the white paper cup within the safe workspace (`x ∈ [0.20, 0.60] m, y ∈ [-0.26, 0.12] m`), estimates the 2D PCA principal axis, disambiguates the narrow bottom (`53 mm`) from the wide rim (`75 mm`), and computes the grasp center `40 mm` from the cup base.
  - After physical placement, automatically captures a second RGB-D frame (`verify_upright_cup`) to verify horizontal placement error, rim height, and upright stability.
- **Digital-Twin Preflight & Safety Gate** (`examples/pick_cup/sim2real.py`):
  - Rehearses the entire trajectory inside a calibrated MuJoCo digital twin before streaming any command to the controller.
  - Enforces minimum collision clearance `>= 5 mm`, wrist-camera tabletop clearance `>= 95 mm`, joint step `<= 0.08 rad`, and 1.0 s unassisted upright stability.
  - Uses compliant Gripper G2 force control (`55 mm` target opening, `20%` speed, `15%` force) to hold the thin-wall cup without crushing it.

## Running Tests

```bash
pytest examples/pick_cup/test_pick_cup.py -v
```
