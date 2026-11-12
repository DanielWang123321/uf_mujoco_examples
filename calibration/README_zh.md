# 标定文件

[English](README.md) | [中文](README_zh.md)

硬件模式（`dry-run`、`sdk-sim`、`real`）从本目录读取运动学和手眼标定。本机标定不纳入 git；迁移的 D435 默认文件与示例纳入版本控制。
路径可在 `examples/<task>/config.yaml` 中修改，或通过 `--calibration` / `--handeye` 指定。

| 文件 | 内容 | 获取方式 |
| :--- | :--- | :--- |
| `xarm6_kinematics.yaml` | xArm 6 运动学标定参数，绑定机械臂序列号 | `python ../ufactory_mujoco/scripts/gen_kinematics_params.py <机械臂IP> <后缀> calibration/`，将输出文件重命名为 `xarm6_kinematics.yaml` |
| `eye_in_hand.yaml` | 相机内参及相机在法兰坐标系下的位姿 | 任意眼在手上标定工具，结果按 [`eye_in_hand.example.yaml`](eye_in_hand.example.yaml) 的格式填写 |

若 `xarm6_kinematics.yaml` 中的序列号与所连接的机械臂不一致，`sdk-sim` 和 `real` 会拒绝运行。

`camera.handeye: null` 时，`realsense` 选择 [`eye_in_hand.d435.yaml`](eye_in_hand.d435.yaml)，
`orbbec` 选择 `eye_in_hand.yaml`。显式配置路径或 `--handeye` 优先。

D435 默认外参 [`eye_in_hand.d435.yaml`](eye_in_hand.d435.yaml) 由 `camera_tool` 对本机 D435i（`109622071268`，`1280×720@30`）执行 20 点 Bundle Adjustment 眼在手上标定导出（桌面基准 `Z = -8 mm`），
按 `Rz @ Ry @ Rx` 转换为 `X_flange_cam`，将彩色光学坐标映射至法兰/`link6`；并按 Gripper G2 的 `tcp_offset = [0, 0, 172, 0, 0, 0]`（`mm, rad`）同步 `X_tcp_cam`。
硬件管线启动时会通过 SDK 读取控制器 `tcp_offset` 与正运动学，校验其与 G2 偏移及标定后 MuJoCo `link6` 位姿一致。
其他安装关系可参照 [`eye_in_hand.d435.example.yaml`](eye_in_hand.d435.example.yaml) 的格式重新标定。

相机机身原地旋转使双目左右互换，不等于整套支架绕法兰旋转，不能据此直接取反法兰 XY 平移；应重新标定光学位姿。
同时记录实际桌面相对机械臂基座安装平面的高度。

实时采集始终提供自身的彩色内参、畸变与图像尺寸；D435 YAML 内参仅为本机 D435i 的离线示例（`eye_in_hand.d435.yaml` 为 `1280×720`，`eye_in_hand.d435.example.yaml` 为 `640×480`），不能用于所有设备。
SDK 畸变系数全零时采用针孔模型，包括本机 D435i 的零逆 Brown-Conrady；不支持的非零模型会报错。
深度比例来自 SDK，不从手眼文件读取。安装关系改变后，先检查基座坐标系中的桌面高度，误差目标 ≤5 mm，再执行任务。

测试使用纳入版本控制的手眼示例及 `ufactory_mujoco` 自带的运动学样例；`pytest` 无需相机 SDK、硬件或本机标定文件。
