# uf_mujoco_examples

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的 **UFACTORY xArm 6 + Gripper G2** 操作示例。每个示例都可以在仿真中运行，也可以配合腕部 **Orbbec Gemini 305** 相机（眼在手上）在真机上执行。

物体只通过仿真接触被移动（不使用 weld 等附着约束）。所有真机动作都先在加载本机运动学标定参数的 MuJoCo 数字孪生中预演。

## 示例

| 示例 | 任务 | 文档 |
| :--- | :--- | :--- |
| [`pick_cup`](examples/pick_cup) | 夹取侧卧纸杯，空中翻转后竖直放置。 | [EN](examples/pick_cup/README.md) / [中文](examples/pick_cup/README_zh.md) |
| [`stack_cups`](examples/stack_cups) | 夹取 2–3 个纸杯并套叠成一摞。 | [EN](examples/stack_cups/README.md) / [中文](examples/stack_cups/README_zh.md) |
| [`screw_nut`](examples/screw_nut) | 夹取六角法兰螺母并拧到固定在桌面的 M12 螺丝上。 | [EN](examples/screw_nut/README.md) / [中文](examples/screw_nut/README_zh.md) |

所有示例支持相同的运行模式：

| 模式 | 运行内容 | 机械臂是否运动 |
| :--- | :--- | :--- |
| `sim`（默认） | 仅 MuJoCo 仿真 | 否 |
| `dry-run` | 相机识别 + 数字孪生预演 | 否 |
| `sdk-sim` | 预演得到的关节轨迹在 xArm 控制器仿真模式中回放 | 否 |
| `real` | 真机执行完整任务，必须加 `--confirm-real` | 是 |

## 环境要求

- Linux，Python 3.12–3.14
- 仿真：无需硬件
- 硬件模式：xArm 6 + Gripper G2，Orbbec Gemini 305 安装在夹爪上，并完成[硬件配置](#硬件配置)

## 安装

本仓库使用 [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco) 中的机器人模型、逆解、碰撞检测和安全门。将两个仓库克隆到同一目录：

```bash
mkdir workspace && cd workspace
git clone https://github.com/DanielWang123321/ufactory_mujoco.git
git clone https://github.com/DanielWang123321/uf_mujoco_examples.git

cd uf_mujoco_examples
pip install -e ".[dev]"          # 仿真 + 测试
pip install -e ".[dev,real]"     # + 硬件模式所需的 xArm SDK 与 Orbbec SDK
```

`ufactory_mujoco` 无需安装：场景文件按相对路径引用其模型，`common/paths.py` 会将其加入 `sys.path`。

## 使用

在仓库根目录运行：

```bash
python -m examples.pick_cup                      # 打开 MuJoCo 窗口
python -m examples.pick_cup --headless           # 无窗口
python -m examples.pick_cup --help               # 全部参数

# 真机按以下顺序执行：
python -m examples.pick_cup --mode dry-run
python -m examples.pick_cup --mode sdk-sim
python -m examples.pick_cup --mode real --confirm-real
```

参数优先级：命令行 > `examples/<task>/config.yaml` > 代码默认值。`--report <file>.json` 保存运行报告。

## 硬件配置

1. **标定文件**，放在 [`calibration/`](calibration/README_zh.md)：
   - `xarm6_kinematics.yaml`：用 `ufactory_mujoco/scripts/gen_kinematics_params.py` 从 xArm 控制器读取的运动学标定；
   - `eye_in_hand.yaml`：相机内参及相机在法兰坐标系下的位姿（格式见 [`eye_in_hand.example.yaml`](calibration/eye_in_hand.example.yaml)）。
2. **机械臂 IP**：`config.yaml` 中的 `robot.ip` 或 `--ip`（默认 `192.168.1.60`）。
3. **相机**：Gemini 305 接 USB 3 接口。若无法识别，安装 `pyorbbecsdk2` 自带的 udev 规则：
   ```bash
   sudo bash "$(python -c 'import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))')/shared/install_udev_rules.sh"
   ```
   连接多台相机时，在 `config.yaml` 中设置 `camera.serial` 或使用 `--camera-serial`。
4. **场景与初始悬停位**：运行前将机械臂置于初始悬停 TCP 位姿 `[300, 0, 300, 180, 0, 0]`（`[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]`）以俯视工作区，每次任务结束后会自动退回该起始位姿。感知算法针对绿色桌垫上的白色纸杯和金属零件调试，工作区为 X ∈ [0.20, 0.60] m、Y ∈ [−0.30, 0.22] m（机械臂基座坐标系）。

> [!CAUTION]
> `real` 模式会驱动机械臂运动。请确保急停按钮触手可及，先运行 `dry-run` 和 `sdk-sim`，并确认工作区无障碍。`--confirm-real` 只授权执行一个周期。

## 目录结构

```text
uf_mujoco_examples/
├── common/                 # 公共代码（不含具体任务逻辑）
│   ├── paths.py            # 仓库路径，默认 IP 与标定文件
│   ├── cli.py              # 公共命令行参数与 config.yaml 读取
│   ├── sim.py              # TaskSimRobot 基类、接触判断
│   ├── geometry.py         # 四元数工具
│   ├── motion.py           # 关节轨迹平滑、软着陆、真机流式下发
│   ├── camera.py           # Orbbec RGB-D 采集
│   ├── vision.py           # 手眼标定、投影、相机俯视逆解
│   ├── sim2real.py         # 碰撞检测器、配置哈希、JSON 报告
│   └── objects/            # 多个示例共用的物体模型（纸杯）
├── examples/<task>/
│   ├── __main__.py         # python -m examples.<task>
│   ├── <task>.py           # 仿真任务 + 命令行入口
│   ├── perception.py       # 相机识别与任务后视觉确认
│   ├── sim2real.py         # 数字孪生预演、sdk-sim、真机执行
│   ├── scene.xml, config.yaml, assets/（软链接）
│   └── test_<task>.py
├── assets/                 # 网格与生成脚本（纸杯、螺丝螺母、Gripper G2）
└── calibration/            # 各机器的标定文件（不纳入 git）
```

### 新增示例

1. 新建 `examples/<task>/`，包含上述文件（可复制 `pick_cup` 作为模板）。
2. 继承 `common.sim.TaskSimRobot` 实现场景机器人；通过 `active_waypoint_sink` 记录关节路点，以便在真机上回放同一动作。
3. 使用 `common.cli.build_parser` / `resolve_run_options` / `run_cli` 构建命令行。
4. 被多个示例使用的代码放入 `common/`，示例之间不互相引用。
5. 添加 `test_<task>.py`，`pytest` 会自动发现。

## 测试

```bash
pytest              # 仿真与合成图像测试，无需硬件
ruff check . && ruff format --check .
```

## 许可

[BSD 3-Clause](LICENSE)。第三方资源与上游许可：[NOTICE](NOTICE)。
