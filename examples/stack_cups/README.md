# stack_cups — pick several paper cups and nest them into one stack

[English](README.md) | [中文](README_zh.md)

Two or three disposable paper cups (5.74 g, Ø53 mm bottom, Ø75 mm rim, 88 mm tall) lie on their sides. The robot picks each cup, flips it upright in the air, places the first cup at the target point, and nests each following cup into the previous one.

## Run

```bash
python -m examples.stack_cups                                 # simulation with viewer
python -m examples.stack_cups --headless --num-cups 3
python -m examples.stack_cups --headless --cup-poses 0.47 -0.15 -60 0.50 0.08 30 --target-xy 0.36 -0.06

python -m examples.stack_cups --mode dry-run                  # camera + digital twin, no motion
python -m examples.stack_cups --mode sdk-sim                  # controller simulation, no motion
python -m examples.stack_cups --mode real --confirm-real      # hardware
```

| Option | Meaning |
| :--- | :--- |
| `--num-cups {2,3}` | Number of cups (default `task.num_cups` in `config.yaml`). |
| `--target-xy X Y` | Stack position (m), default `task.target_xy`. In hardware modes, if no target is given, a free spot ≥ 110 mm away from all cups is chosen. |
| `--cup-poses X1 Y1 YAW1 ...` | Cup bottom positions (m) and bottom→rim yaws. In hardware modes it replaces camera detection. |

Shared options (`--mode`, `--headless`, `--ip`, `--calibration`, `--handeye`, `--camera-serial`, `--post-verify`, `--save-vis`, `--save-after-vis`, `--report`) and their defaults are in [`config.yaml`](config.yaml).

## How it works

**Simulation** (`stack_cups.py`)

1. For each cup: grasp it sideways, lift, and flip the cup axis to vertical (same grasp as [`pick_cup`](../pick_cup)).
2. Move the cup bottom 25 mm above the target (first cup) or above the rim of the previous cup, correcting with the live cup positions.
3. First cup: lower until it touches the table. Following cups: lower until the bottom is 4 mm below the previous rim (the fingers stay ~22 mm above it), then release; the cup slides into the previous one.
4. After the last cup, require 1 s of rest: stack XY error ≤ 30 mm, adjacent cups coaxial within 8 mm, tilt ≤ 10°.

Cup-to-cup contacts use low friction (0.08 < tan 7.1°, the cone half-angle), so a released cup centers itself in the cup below. Finger pads use high friction (4.0).

**Sim-to-Real** (`sim2real.py`, `perception.py`)

1. Capture one RGB-D frame and detect all white sideways cups (axis by PCA, bottom/rim end by width and height); cups are picked from right (−Y) to left (+Y).
2. Run all cycles in the calibrated digital twin. Every sample of the 50 Hz joint stream is checked for joint limits, collisions, wrist-camera height ≥ 95 mm above the table, and joint step ≤ 0.15 rad.
3. `sdk-sim` replays the stream in the controller's simulation mode. `real`:
   - before each grasp, moves the camera 280 mm directly above the cup, re-measures it, and replans the grasp; returns home if the cup is gone;
   - aborts and returns home if the measured grasp gap is < 30 mm (empty grasp).
4. After returning, a second frame verifies the stack (≥ 1200 cup pixels, XY error ≤ 45 mm, rim height ≥ 50 mm, initial cup positions empty).

## Files

| File | Content |
| :--- | :--- |
| `stack_cups.py` | `StackCupRobot`, `run_stack_cups`, CLI |
| `perception.py` | `detect_sideways_cups`, `detect_single_cup_near`, `verify_stacked_cups`, overlays |
| `sim2real.py` | digital-twin preflight, sdk-sim, real execution |
| `scene.xml` | xArm 6 + G2, table, three paper cups |
| `test_stack_cups.py` | `pytest examples/stack_cups` |
