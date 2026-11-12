# Calibration files

[English](README.md) | [中文](README_zh.md)

The hardware modes (`dry-run`, `sdk-sim`, `real`) read kinematic and hand-eye calibration here.
Local per-robot files are ignored by git; the migrated D435 default and examples are tracked.
Paths can be changed in `examples/<task>/config.yaml` or with
`--calibration` / `--handeye`.

| File | Content | How to get it |
| :--- | :--- | :--- |
| `xarm6_kinematics.yaml` | Kinematic calibration of your xArm 6, bound to its serial number | `python ../ufactory_mujoco/scripts/gen_kinematics_params.py <robot_ip> <suffix> calibration/`, then rename the output to `xarm6_kinematics.yaml` |
| `eye_in_hand.yaml` | Camera intrinsics and the camera pose in the flange frame | Any eye-in-hand calibration tool; write the result in the format of [`eye_in_hand.example.yaml`](eye_in_hand.example.yaml) |

`sdk-sim` and `real` refuse to run if the serial number in `xarm6_kinematics.yaml` does not match
the connected robot.

With `camera.handeye: null`, `realsense` selects [`eye_in_hand.d435.yaml`](eye_in_hand.d435.yaml),
and `orbbec` selects `eye_in_hand.yaml`. An explicit config path or `--handeye` overrides this.

The default D435 file [`eye_in_hand.d435.yaml`](eye_in_hand.d435.yaml) is calibrated on the local D435i (`109622071268`, `1280×720@30`) via 20-pose Bundle Adjustment in `camera_tool` (table plane `Z = -8 mm`).
`X_flange_cam` uses `Rz @ Ry @ Rx` and maps the color optical frame into flange/`link6`, while `X_tcp_cam` reflects Gripper G2's `tcp_offset = [0, 0, 172, 0, 0, 0]` (`mm, rad`).
At startup, the hardware pipelines query SDK `tcp_offset` and forward kinematics to verify Gripper G2 TCP settings against calibrated MuJoCo `link6`.
[`eye_in_hand.d435.example.yaml`](eye_in_hand.d435.example.yaml) retains a reference example format for another mounting.

Turning the camera body in place so the stereo lenses exchange sides is different from rotating
the whole bracket around the flange. Do not negate the flange XY translation based on that description;
recalibrate the optical pose. Record the actual table height relative to the robot mounting plane.

Live captures always supply their own color intrinsics, distortion, and image size. The D435 YAML
intrinsics are offline examples measured on the local D435i (`1280×720` in `eye_in_hand.d435.yaml`, `640×480` in `eye_in_hand.d435.example.yaml`); they are not universal.
Zero SDK distortion coefficients use the pinhole model, including D435i's zero inverse Brown-Conrady;
unsupported nonzero models are rejected. The depth scale comes from the SDK, not the hand-eye file.
After changing the mounting, check the table in the base frame (height error target ≤5 mm) before task execution.

Tests use the tracked hand-eye examples and the sample kinematic calibration shipped with
`ufactory_mujoco`; `pytest` needs no camera SDK, hardware, or local calibration.
