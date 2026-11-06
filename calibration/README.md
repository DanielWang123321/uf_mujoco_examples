# Calibration files

[English](README.md) | [中文](README_zh.md)

The hardware modes (`dry-run`, `sdk-sim`, `real`) read two per-robot files from this folder.
They are ignored by git. Paths can be changed in `examples/<task>/config.yaml` or with
`--calibration` / `--handeye`.

| File | Content | How to get it |
| :--- | :--- | :--- |
| `xarm6_kinematics.yaml` | Kinematic calibration of your xArm 6, bound to its serial number | `python ../ufactory_mujoco/scripts/gen_kinematics_params.py <robot_ip> <suffix> calibration/`, then rename the output to `xarm6_kinematics.yaml` |
| `eye_in_hand.yaml` | Camera intrinsics and the camera pose in the flange frame | Any eye-in-hand calibration tool; write the result in the format of [`eye_in_hand.example.yaml`](eye_in_hand.example.yaml) |

`sdk-sim` and `real` refuse to run if the serial number in `xarm6_kinematics.yaml` does not match
the connected robot.

The tests use `eye_in_hand.example.yaml` and the sample calibration shipped with `ufactory_mujoco`,
so no hardware or calibration is needed for `pytest`.
