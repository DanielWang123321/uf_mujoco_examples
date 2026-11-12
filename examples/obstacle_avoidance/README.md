# obstacle_avoidance — pick a block and carry it around a cylinder

[English](README.md) | [中文](README_zh.md)

The robot grasps a block (50 × 50 × 50 mm, 80 g) on the table, plans a collision-free path around a fixed cylinder (Ø90 × 450 mm) in the center of the table, and places the block at the target area. **Only `sim` mode is supported.**

## Run

```bash
python -m examples.obstacle_avoidance                                   # simulation with viewer
python -m examples.obstacle_avoidance --headless                        # headless simulation
python -m examples.obstacle_avoidance --headless --seed 1
python -m examples.obstacle_avoidance --start-pose 350 -200 300 180 0 0 --goal-pose 350 200 300 180 0 0
```

| Option | Meaning |
| :--- | :--- |
| `--headless` / `--no-headless` | Run without or with the viewer window (opens window by default). |
| `--start-pose X Y Z R P Y` | Initial hover pose after lifting the block (mm, extrinsic XYZ angles), default `[350, -200, 300, 180, 0, 0]`. |
| `--goal-pose X Y Z R P Y` | Goal hover pose after clearing the obstacle, default `[350, 200, 300, 180, 0, 0]` (its XY is the placement target). |
| `--seed N` | Random seed for IK initial guesses, RRT search, and path shortcutting (default `0`). |
| `--config PATH` | Configuration file path (default [`config.yaml`](config.yaml)). |
| `--report PATH` | JSON run report path (default `build/obstacle_avoidance/sim_report.json`). |

Planning parameters (RRT step size, iteration count, planning clearance `clearance_m`) and motion limits (execution clearance `execution_clearance_m`, joint velocity/acceleration limits) are configured in [`config.yaml`](config.yaml). Priority: command line > `config.yaml` > built-in defaults.

## How it works

1. **Grasp and lift**: descend from the start hover pose, close the gripper on the block (initial center `[350, -200, 25]` mm), confirm bilateral finger-pad contact, and lift back to the start hover pose.
2. **Loaded obstacle-avoidance planning**: build a carried-payload collision model from the current joint angles and the measured block pose relative to the TCP, search a path around the cylinder (center `[350, 0, 0]` mm) with RRT-Connect, then shortcut and time-parameterize the trajectory. In the viewer, the **red line** shows the colliding direct joint-interpolation baseline (not executed) and the **green line** shows the planned TCP path.
3. **Carry and place**: follow the planned trajectory around the cylinder to the goal hover pose, lower the block until it touches the table, release, retreat, and verify 1 s of independent rest (placement error ≤ 5 mm, drift ≤ 1 mm, tilt ≤ 5°).

Throughout execution, collision clearance is checked for the arm, gripper, and carried block, and grasp stability is monitored (the task aborts if relative shift > 5 mm, relative rotation > 5°, or bilateral contact is lost for > 0.1 s).

## Files

| File | Content |
| :--- | :--- |
| `obstacle_avoidance.py` | `ObstacleRobot`, task pipeline, CLI entry point |
| `planning.py` | IK solving, collision checking, RRT loaded planning, trajectory time parameterization |
| `manipulation.py` | Grasp, lift, placement actions, and grasp-slip monitoring |
| `scene.xml` | xArm 6 + G2, table, cylinder obstacle, and block |
| `config.yaml` | Default hover poses, planning parameters, and motion limits |
| `test_obstacle_avoidance.py` | `pytest examples/obstacle_avoidance` |
