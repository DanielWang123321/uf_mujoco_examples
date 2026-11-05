# 03 — M12 Helical Screw-Nut Threading & Tactile Closed-Loop Control

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation and closed-loop Sim-to-Real example for **UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305 (Eye-in-Hand)**: an inverted M12 screw (`56.5 g` mass, `45.0 mm` total height, `Ø20.0 × 14.0 mm` lower head, `Ø12.0 × 31.0 mm` upper threaded shaft with **12 continuous 3D helical thread turns** at `2.40 mm` pitch) is glued upright to the tabletop. The robot picks up a free hex flange nut (`29.0 g` mass, `Ø29.0 × 3.8 mm` bottom flange, `Ø20.0 × 10.7 mm` upper hex body, `14.5 mm` total height, with **8 internal helical follower teeth**), aligns it coaxially onto the screw tip, and executes **multi-stroke clockwise ratcheting** (default `10.0 turns = 3600°` across 7 strokes) with **tactile Z-axis closed-loop adaptation**.

## Configuration (`config.yaml`)

Default configuration is located at `examples/screw_nut/config.yaml`:

```yaml
mode: sim
headless: false  # false: open 3D MuJoCo viewer; true: run offscreen

robot:
  ip: "192.168.1.60"
  calibration: "../ufactory_mujoco/calibrations/xarm6_kinematics_L56A01.yaml"

camera:
  handeye: "../camera_tool/result/eye_in_hand.yaml"

task:
  nut_xy: [0.3538, -0.0114]    # Initial tabletop coordinates of the free nut [x_m, y_m]
  nut_yaw_deg: 0.0             # Initial Z-axis yaw angle of the nut (deg)
  screw_xy: [0.3560, -0.1028]  # Tabletop coordinates of the glued upright screw [x_m, y_m]
  thread_turns: 10.0           # Clockwise threading turns (default 10.0 turns = 3600° across 7 strokes)
  post_verify: true            # Perform overhead Gemini 305 visual verification after threading
  save_vis: "build/screw_nut/real_detection.png"
  save_after_vis: "build/screw_nut/real_after_thread.png"
  report: "build/screw_nut/sim_report.json"
```

## Quick Start Commands

Run from the repository root:

```bash
# 1. Default interactive simulation (opens 3D viewer, threads 10.0 turns across 7 ratchet strokes)
python -m examples.screw_nut.run --mode sim

# 2. Headless simulation with JSON report output
python -m examples.screw_nut.run --mode sim --headless --report build/screw_nut/sim_report.json

# 3. Custom nut/screw poses and partial threading (e.g., 3.0 turns = 1080°, self-locking mid-shaft)
python -m examples.screw_nut.run --mode sim --headless \
  --nut-pose 0.33 0.05 25.0 \
  --screw-xy 0.38 -0.08 \
  --thread-turns 3.0 \
  --report build/screw_nut/sim_custom.json

# 4. Sim-to-Real 3-stage pipeline (dry-run / sdk-sim / real)
python -m examples.screw_nut.run --mode dry-run
python -m examples.screw_nut.run --mode sdk-sim
python -m examples.screw_nut.run --mode real --confirm-real
```

## Physics & Closed-Loop Control Architecture

1. **12-Turn 3D Helical Thread Collision & Mid-Shaft Self-Locking**:
   - **Screw Shaft Collision**: Core cylinder (`r = 4.90 mm`) + **12 right-hand helical capsule turns** (`192` segments, `2.40 mm` pitch, `12.0 mm` outer diameter).
   - **Nut Internal Bore Collision**: Bottom `7.2 mm` coaxial guide skirt + **8 internal helical sphere follower teeth** (`5.40 mm` tip radius) that mesh directly into the screw's helical groove.
   - Releasing the gripper mid-shaft leaves the nut mechanically self-locked on the thread (`< 0.15 mm` settle drop).
2. **Multi-Stroke Ratcheting within `J6 ∈ [-360°, +360°]` Limits**:
   - Automatically decomposes total turns (e.g. `10.0 turns = 3600°`) into bounded strokes (first 6 strokes of `1.5 turns = 540°`, 7th stroke of `1.0 turn = 360°`).
   - During each stroke, the arm rotates `J6` clockwise while descending synchronously by $\Delta z = p \cdot N_{\text{stroke}}$ (`3.60 mm` per `1.5` turns), opens the gripper to `50.0 mm`, unwinds `J6` counter-clockwise back to `-346°`, and re-grasps the upper hex body.
3. **Tactile Z-Axis Closed-Loop Adaptation (`MAX_HEX_FLAT_GAP_MM = 21.2 mm`)**:
   - Gripper grasping force is strictly capped at `25.0%`.
   - Because rotational slip during thread engagement can cause the nut's actual descent $\Delta z_{\text{actual}}$ to lag behind the nominal pitch descent $\Delta z_{\text{nominal}}$, re-grasping at the nominal Z height would collide with the `Ø29 mm` bottom flange (`meas_gap > 21.2 mm`) instead of the `Ø20 mm` upper hex flats (`~18.5–19.8 mm`).
   - Whenever `meas_gap > 21.2 mm` is detected on re-grasp, the controller automatically opens the gripper, backs off upward by `+3.5 mm` per retry (`z_backoff_m`), re-grasps the `Ø20 mm` hex body, and appends up to `3` adaptive extra ratchet strokes until the nut seats on the `z = 14.0 mm` screw head shoulder.
4. **2-Stage Overhead Visual Refinement & Post-Task Verification (`common/vision.py`)**:
   - Uses `solve_camera_lookdown_q` (`z_cam = 250 mm`) directly above the nut and screw before grasping/insertion to eliminate oblique camera parallax, and verifies final coaxially seated nut height upon completion.

## Running Tests

```bash
pytest assets/screw_nut/test_screw_nut.py examples/screw_nut/test_screw_nut_sim.py -v
```
