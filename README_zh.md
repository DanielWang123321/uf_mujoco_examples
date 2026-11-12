# uf_mujoco_examples

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的 **UFACTORY xArm 6 + Gripper G2** 操作与轨迹规划示例。所有示例均可在仿真中运行；纸杯与螺母任务也可配合腕部 **RealSense D435/D435i**（默认）或 **Orbbec Gemini 305** 相机（眼在手上）在真机上执行。

物体只通过仿真接触被移动（不使用 weld 等附着约束）。所有真机动作都先在加载本机运动学标定参数的 MuJoCo 数字孪生中预演。

## 示例

| 示例 | 任务 | 文档 |
| :--- | :--- | :--- |
| [`pick_cup`](examples/pick_cup) | 夹取侧卧纸杯，空中翻转后竖直放置。 | [EN](examples/pick_cup/README.md) / [中文](examples/pick_cup/README_zh.md) |
| [`stack_cups`](examples/stack_cups) | 夹取 2–3 个纸杯并套叠成一摞。 | [EN](examples/stack_cups/README.md) / [中文](examples/stack_cups/README_zh.md) |
| [`screw_nut`](examples/screw_nut) | 夹取六角法兰螺母并拧到固定在桌面的 M12 螺丝上。 | [EN](examples/screw_nut/README.md) / [中文](examples/screw_nut/README_zh.md) |
| [`obstacle_avoidance`](examples/obstacle_avoidance) | 抓取方块，携物绕过固定于桌面的圆柱后放置；**仅 sim**。 | [EN](examples/obstacle_avoidance/README.md) / [中文](examples/obstacle_avoidance/README_zh.md) |

`pick_cup`、`stack_cups`、`screw_nut` 支持以下模式；`obstacle_avoidance` 仅支持 `sim`：

| 模式 | 运行内容 | 机械臂是否运动 |
| :--- | :--- | :--- |
| `sim`（默认） | 仅 MuJoCo 仿真 | 否 |
| `dry-run` | 相机识别 + 数字孪生预演 | 否 |
| `sdk-sim` | 预演得到的关节轨迹在 xArm 控制器仿真模式中回放 | 否 |
| `real` | 真机执行完整任务，必须加 `--confirm-real` | 是 |

## 环境要求

- Linux，Python 3.12–3.14
- 仿真：无需硬件
- 硬件模式：xArm 6 + Gripper G2，D435/D435i 或 Gemini 305 安装在夹爪上，并完成[硬件配置](#硬件配置)

## 安装

本仓库使用 [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco) 中的机器人模型、逆解、碰撞检测和安全门。将两个仓库克隆到同一目录：

```bash
mkdir workspace && cd workspace
git clone https://github.com/DanielWang123321/ufactory_mujoco.git
git clone https://github.com/DanielWang123321/uf_mujoco_examples.git

cd uf_mujoco_examples
pip install -e ".[dev]"          # 仿真 + 测试
pip install -e ".[dev,real]"     # + xArm、RealSense 和 Orbbec SDK（相机 SDK 延迟导入）
```

`ufactory_mujoco` 无需安装：场景文件按相对路径引用其模型，`common/paths.py` 会将其加入 `sys.path`。

## 使用

在仓库根目录运行：

```bash
python -m examples.pick_cup                      # 打开 MuJoCo 窗口
python -m examples.pick_cup --headless           # 无窗口
python -m examples.pick_cup --help               # 全部参数
python -m examples.obstacle_avoidance --headless # 静态绕障规划，仅 sim

# 真机按以下顺序执行：
python -m examples.pick_cup --mode dry-run
python -m examples.pick_cup --mode sdk-sim
python -m examples.pick_cup --mode real --confirm-real
```

参数优先级：命令行 > `examples/<task>/config.yaml` > 代码默认值。`--report <file>.json` 保存运行报告。

`camera.type` / `--camera-type` 默认为 `realsense`（D435/D435i，1280×720@30）；Gemini 305 使用 `--camera-type orbbec`（1280×800@30）。`--camera-serial` 使用所选相机 SDK 返回的序列号。`camera.handeye` 留空时按相机类型选择默认文件；显式 YAML 路径或 `--handeye` 优先。

```bash
python -m common.camera_check --duration 30                    # RGB-D 只读检查
python -m examples.pick_cup --mode dry-run --camera-type orbbec
```

实时检测、近距复测和任务后验证均采用 SDK 的彩色内参、畸变与图像尺寸。离线感知可使用 YAML 内参，也可传入采集返回的 `camera_metadata`。深度对齐至彩色图，单位为米；零畸变采用针孔模型，不支持的非零畸变模型会明确报错。硬件报告包含相机型号、SDK 序列号、实际流配置、内参、深度比例、每次采集元数据、同步法兰位姿及手眼文件 SHA-256。指定检测图路径时，还保存原始 `.input.png` 与可重放的 `.rgbd.npz` 输入。几何约定见 [RealSense SDK 文档](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)。

验收分别运行单杯、两杯（`--num-cups 2`）、三杯（`--num-cups 3`）和螺母场景，依次通过 `dry-run → sdk-sim → real --confirm-real`，开启 `--post-verify`。各场景、模式分别使用 `build/<场景>/<模式>/` 保存 `--save-vis`、`--save-after-vis`、`--report` 与重定向的终端日志；任务后图片仅由 real 生成。

## 硬件配置

1. **标定文件**，放在 [`calibration/`](calibration/README_zh.md)：
   - `xarm6_kinematics.yaml`：用 `ufactory_mujoco/scripts/gen_kinematics_params.py` 从 xArm 控制器读取的运动学标定；
   - D435：[`eye_in_hand.d435.yaml`](calibration/eye_in_hand.d435.yaml)，本机 D435i 经 20 点 Bundle Adjustment 实测的法兰到彩色光学坐标系外参（配合控制器 Gripper G2 `tcp_offset = [0, 0, 172, 0, 0, 0]` 门禁校验）；更改支架前参阅[标定说明](calibration/README_zh.md)。
   - Gemini：`eye_in_hand.yaml`（格式见 [`eye_in_hand.example.yaml`](calibration/eye_in_hand.example.yaml)）。
2. **机械臂 IP**：`config.yaml` 中的 `robot.ip` 或 `--ip`（默认 `192.168.1.60`）。
3. **相机**：所选相机接 USB 3 接口。RealSense 按[官方 Linux 说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md)配置 udev；Gemini 305 使用 `pyorbbecsdk2` 自带的规则：
   ```bash
   sudo bash "$(python -c 'import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))')/shared/install_udev_rules.sh"
   ```
   连接多台相机时，在 `config.yaml` 中设置 `camera.serial` 或使用 `--camera-serial`。
4. **场景与初始悬停位**：运行前将机械臂置于初始悬停 TCP 位姿：D435 推荐 `[300, 0, 400, 180, 0, 0]`，Gemini 为 `[300, 0, 300, 180, 0, 0]`（`[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]`）以俯视工作区，每次任务结束后会自动退回该起始位姿。纸杯轮廓须完整进入感知视野，触及边界会拒绝检测；可提高观察位或调整杯位。感知算法针对绿色桌垫上的白色纸杯和金属零件调试，工作区为 X ∈ [0.20, 0.60] m、Y ∈ [−0.30, 0.22] m（机械臂基座坐标系）。

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
│   ├── camera.py           # RealSense / Orbbec RGB-D 采集
│   ├── camera_check.py     # RGB-D 视频流与深度只读健康检查
│   ├── vision.py           # 手眼标定、投影、相机俯视逆解
│   ├── sim2real.py         # 碰撞检测器、配置哈希、JSON 报告
│   └── objects/            # 多示例共用的物体模型与感知辅助（纸杯）
├── examples/<task>/
│   ├── __main__.py         # python -m examples.<task>
│   ├── <task>.py           # 仿真任务 + 命令行入口
│   ├── perception.py       # 相机识别与任务后视觉确认
│   ├── sim2real.py         # 数字孪生预演、sdk-sim、真机执行
│   ├── scene.xml, config.yaml, assets/（软链接）
│   └── test_<task>.py
├── assets/                 # 网格与生成脚本（纸杯、螺丝螺母、Gripper G2）
├── calibration/            # 默认/示例手眼标定与本机机器人标定
└── tests/                  # 跨示例相机、感知与数字孪生回归测试
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
