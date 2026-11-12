# pick_cup — pick, flip, and place a paper cup upright

[English](README.md) | [中文](README_zh.md)

A disposable paper cup (5.74 g, Ø53 mm bottom, Ø75 mm rim, 88 mm tall) lies on its side. The robot grasps it, flips it 90° in the air, and places it upright at a target point.

## Run

```bash
python -m examples.pick_cup                                   # simulation with viewer
python -m examples.pick_cup --headless
python -m examples.pick_cup --cup-pose 0.46 -0.07 140 --target-xy 0.38 -0.18

python -m examples.pick_cup --mode dry-run                    # camera + digital twin, no motion
python -m examples.pick_cup --mode sdk-sim                    # controller simulation, no motion
python -m examples.pick_cup --mode real --confirm-real        # hardware
```

| Option | Meaning |
| :--- | :--- |
| `--cup-pose X Y YAW_DEG` | Cup bottom position (m) and bottom→rim yaw. In hardware modes it replaces camera detection. |
| `--target-xy X Y` | Upright placement point (m). Sim default `(0.30, 0.15)`; hardware default `(0.38, -0.18)`, falling back to `(0.38, 0.04)` if it is within 110 mm of the cup. |

Shared options (`--mode`, `--headless`, `--ip`, `--calibration`, `--handeye`, `--camera-type`, `--camera-serial`, `--post-verify`, `--save-vis`, `--save-after-vis`, `--report`) and their defaults are in [`config.yaml`](config.yaml).

Default camera: D435/D435i, 1280×720@30. For Gemini 305 use `--camera-type orbbec` (1280×800@30). Empty `camera.handeye` selects the backend default; `--handeye` overrides it. Live captures supply color intrinsics to all detection and verification steps. See [camera setup](../../README.md#hardware-setup).

## How it works

**Simulation** (`pick_cup.py`)

1. Grasp 40 mm from the cup bottom with the gripper vertical; close to 53 mm, back off 1 mm, then keep ~2 N contact force with a 20 Hz feedback loop.
2. Lift, flip the cup axis to vertical, carry it while correcting with the live cup position, and lower it until the bottom touches the table.
3. Release, retreat, and require 1 s of free standing (tilt ≤ 10°, placement error ≤ 30 mm, drift ≤ 2 mm).

During the grasp the task aborts if bilateral finger contact is lost for > 0.10 s or the cup moves > 25 mm / 25° relative to the TCP.

**Sim-to-Real** (`sim2real.py`, `perception.py`)

1. Read the joint state and capture one RGB-D frame. The cup is segmented (white pixels + depth above the table); its axis comes from PCA and the bottom/rim ends from width and height.
2. Run the task in the calibrated digital twin and record the joint waypoints. Every sample of the resulting 50 Hz stream is checked for joint limits, collisions, wrist-camera height ≥ 95 mm above the table, and joint step ≤ 0.15 rad.
3. `sdk-sim` replays the stream in the controller's simulation mode. Before a camera-guided real grasp, move the camera above the cup (`360 mm` for D435, `280 mm` for Gemini), re-measure its pose, and rerun the complete calibrated twin. A missing cup or failed refined preflight returns to the observation pose; invalid gripper readings or gaps below 30 mm abort before lifting. Execute with a 58 mm grasp gap, 10 % speed, 5 % force.
4. After returning to the observation pose, a second frame verifies the result (≥ 1200 cup pixels, XY error ≤ 45 mm, rim height ≥ 45 mm).

## Files

| File | Content |
| :--- | :--- |
| `pick_cup.py` | `CupRobot`, `run_pick_cup`, CLI |
| `perception.py` | `detect_sideways_cup`, shared `detect_single_cup_near`, `verify_upright_cup`, overlays |
| `sim2real.py` | digital-twin preflight, sdk-sim, real execution |
| `scene.xml` | xArm 6 + G2, table, paper cup |
| `test_pick_cup.py` | `pytest examples/pick_cup` |
