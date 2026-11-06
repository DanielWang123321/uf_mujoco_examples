# 标定文件

[English](README.md) | [中文](README_zh.md)

硬件模式（`dry-run`、`sdk-sim`、`real`）从本目录读取两个与具体机器相关的文件，这些文件不纳入 git。
路径可在 `examples/<task>/config.yaml` 中修改，或通过 `--calibration` / `--handeye` 指定。

| 文件 | 内容 | 获取方式 |
| :--- | :--- | :--- |
| `xarm6_kinematics.yaml` | xArm 6 运动学标定参数，绑定机械臂序列号 | `python ../ufactory_mujoco/scripts/gen_kinematics_params.py <机械臂IP> <后缀> calibration/`，将输出文件重命名为 `xarm6_kinematics.yaml` |
| `eye_in_hand.yaml` | 相机内参及相机在法兰坐标系下的位姿 | 任意眼在手上标定工具，结果按 [`eye_in_hand.example.yaml`](eye_in_hand.example.yaml) 的格式填写 |

若 `xarm6_kinematics.yaml` 中的序列号与所连接的机械臂不一致，`sdk-sim` 和 `real` 会拒绝运行。

测试使用 `eye_in_hand.example.yaml` 和 `ufactory_mujoco` 自带的样例标定，运行 `pytest` 无需硬件和标定文件。
