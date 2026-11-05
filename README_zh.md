# uf_mujoco_examples

[English](README.md) | [中文](README_zh.md)

基于 **MuJoCo** 与 **UFACTORY xArm 6** 机械臂、**Gripper G2** 四连杆平行夹爪及 **Orbbec Gemini 305**（Eye-in-Hand 眼在手上）RGB-D 相机的接触丰富操作（Contact-Rich Manipulation）与 **Sim-to-Real** 闭环控制示例集。

本仓库构建于 [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco) 基础之上，展示了薄壁轻质物体柔顺抓取翻转、多物体同心套叠以及带触觉 Z 轴闭环自适应的 3D 螺旋螺纹旋紧等高精度操作任务。

---

## 核心特性

- **高保真接触物理建模**：
  - **中空一次性纸杯（`5.74 g`）**：采用 24 段倾角圆台薄壁碰撞几何（壁厚 `1 mm`）与三级接触优先级解耦设计（$\mu_{\text{cup-cup}} = 0.08 < \tan\alpha = 0.125$），实现上层纸杯释放后沿内壁自定心顺滑嵌套而不卡滞。
  - **M12 螺旋螺丝与六角法兰螺母（`56.5 g` / `29.0 g`）**：螺杆具备 12 圈连续右旋 3D 胶囊体螺旋凸起（螺距 `2.40 mm`，共 `192` 段），配合螺母内孔 8 颗内螺旋啮合球齿，支持螺杆半途物理自锁与亮光金属（Chrome Steel）高光反射渲染。
- **Eye-in-Hand 闭环视觉感知（`common/vision.py`）**：
  - 支持 Orbbec Gemini 305 对齐 RGB-D 采集与基座坐标系点云反投影。
  - **正上方二次精测（2-Stage Look-Down Refinement）**：通过 `solve_camera_lookdown_q`（`z_cam = 250 mm`）将腕部相机移动至目标正上方垂直俯视复测，消除斜视视差并拦截目标移位/缺失。
  - **任务完成闭环确认**：真机作业完成后自动退回观测位二次拍照，对纸杯竖放倾角、叠杯同心度及螺母旋紧高度进行闭环验收。
- **触觉 Z 轴闭环自适应与软着陆轨迹规划（`common/motion.py`）**：
  - 提供单调软着陆进度曲线（`smooth_progress_profile`）与曲率限速多路点平滑插值（`interpolate_waypoint_chain`），确保起止零速与末端柔和减速。
  - 实时读取 Gripper G2 夹爪开度反馈（`get_measured_gripper_gap`），自动识别空抓、误夹螺母底法兰（开度 `> 21.2 mm`）及真机轴向滞后，自主触发 Z 轴上抬重抓（`+3.5 mm`）与自适应补拧行程。
- **三阶段 Sim-to-Real 安全门禁**：
  1. `dry-run`：实机相机感知 + 加载单机运动学标定的 MuJoCo 数字孪生预演（零硬件运动）。
  2. `sdk-sim`：xArm 控制器内部仿真演练（`mode=4, state=5`，物理机械臂不动）。
  3. `real --confirm-real`：单周期真机闭环执行，受数字孪生碰撞净空（`>= 5 mm`）、腕部相机离桌净空（`>= 88–95 mm`）及预演窗口提前关闭熔断保护。

---

## 示例导航

| 编号 | 示例任务 | 目录 | 核心能力概览 | 说明文档 |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **一次性纸杯夹取与翻转竖放** | [`examples/pick_cup`](examples/pick_cup) | 侧卧圆台纸杯 6D 位姿感知、`15%` 柔顺力控夹持、空中 `90°` 无碰撞翻转、竖直放置与视觉闭环确认 | [中文](examples/pick_cup/README_zh.md) \| [English](examples/pick_cup/README.md) |
| **02** | **多个纸杯夹取与同心套杯堆叠** | [`examples/stack_cups`](examples/stack_cups) | 多纸杯连通域分割、从右至左抓取排序、杯口内嵌 `4 mm` 防滑出释放、2/3 杯同心套叠与静置验收 | [中文](examples/stack_cups/README_zh.md) \| [English](examples/stack_cups/README.md) |
| **03** | **M12 螺丝螺母多把倒手旋紧** | [`examples/screw_nut`](examples/screw_nut) | 12 圈 3D 螺旋啮合与半途自锁、`J6 ∈ [-360°, +360°]` 多把倒手（Ratchet）、正上方二次精测、触觉 Z 轴闭环自适应 | [中文](examples/screw_nut/README_zh.md) \| [English](examples/screw_nut/README.md) |

---

## 环境安装与工作区布局

### 1. 同级目录克隆仓库

本仓库通过相对路径引用同级目录 [`ufactory_mujoco`](https://github.com/DanielWang123321/ufactory_mujoco) 中的机械臂模型及核心运动学/安全门禁库：

```bash
mkdir -p ~/workspace && cd ~/workspace
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git
git clone https://github.com/DanielWang123321/uf_mujoco_examples.git
```

标准工作区目录结构如下：

```text
workspace/
├── ufactory_mujoco/          # UFACTORY 机器人核心 MuJoCo 模型、IK、轨迹规划与安全接口库
├── uf_mujoco_examples/       # 本仓库（接触丰富操作仿真与 Sim-to-Real 闭环控制示例集）
└── camera_tool/              # （可选，仅真机相机模式使用）Eye-in-Hand 手眼标定结果与 Orbbec 采集工具
```

### 2. 安装 Python 依赖

支持 **Python 3.12、3.13、3.14**（推荐 Ubuntu 22.04 / 24.04）：

```bash
cd ~/workspace/uf_mujoco_examples

# 以可编辑模式安装 ufactory_mujoco 核心包与本仓库依赖
pip install -e ../ufactory_mujoco
pip install -e ".[dev]"
```

---

## 快速上手

所有示例均支持通过模块入口（`python -m examples.<task>.run`）或脚本路径（`python examples/<task>/<task>.py`）启动。

### 1. 纯 MuJoCo 仿真模式（`--mode sim`）

```bash
# 示例 01：夹取侧卧一次性纸杯，空中翻转 90° 并竖直放置
python -m examples.pick_cup.run --mode sim

# 示例 02：多个一次性纸杯连续夹取与同心套叠（支持 2 杯或 3 杯）
python -m examples.stack_cups.run --mode sim --num-cups 2
python -m examples.stack_cups.run --mode sim --num-cups 3 --headless

# 示例 03：夹取六角法兰螺母并多把倒手旋紧至桌面 M12 螺丝（默认 10 圈）
python -m examples.screw_nut.run --mode sim
python -m examples.screw_nut.run --mode sim --headless --thread-turns 3.0
```

> 传入 `--headless` 可在后台无头运行物理仿真，传入 `--report build/<task>/sim_report.json` 可输出结构化评测报告。

### 2. Sim-to-Real 三阶段安全执行

连接真机前，可在 `examples/<task>/config.yaml` 中配置 `robot.ip`、`robot.calibration` 和 `camera.handeye`（或通过命令行参数覆盖）：

```bash
# 阶段一 (dry-run)：Gemini 305 实机感知 + MuJoCo 数字孪生可视化预演（零硬件运动）
python -m examples.screw_nut.run --mode dry-run

# 阶段二 (sdk-sim)：xArm 控制器内部仿真演练（mode=4, state=5，物理机械臂不动）
python -m examples.screw_nut.run --mode sdk-sim

# 阶段三 (real)：完整闭环真机执行（必须显式传入 --confirm-real 授权标志）
python -m examples.screw_nut.run --mode real --confirm-real
```

---

## 项目目录结构

```text
uf_mujoco_examples/
├── assets/
│   ├── cup/                  # 一次性纸杯参数化生成脚本、视觉/碰撞网格与独立 MJCF
│   ├── gripper_g2/           # 双面平滑法线 Gripper G2 高精度视觉 OBJ 与碰撞 STL 网格
│   └── screw_nut/            # M12 螺旋螺丝与六角螺母生成脚本、亮光金属贴图及资产单元测试
├── common/
│   ├── motion.py             # 软着陆进度曲线、曲率限速轨迹平滑插值与真机流式下发工具
│   └── vision.py             # Orbbec Gemini 305 采集流、手眼坐标变换与正上方俯视 IK 求解
├── examples/
│   ├── pick_cup/             # 01: 单个纸杯夹取、空中翻转与竖放（仿真 + Sim-to-Real）
│   ├── stack_cups/           # 02: 多个纸杯连续夹取与同心套杯堆叠（仿真 + Sim-to-Real）
│   └── screw_nut/            # 03: M12 螺丝螺母旋紧与触觉 Z 轴闭环自适应控制
├── LICENSE                   # MIT 开源许可证
├── NOTICE                    # 第三方资产与上游开源协议声明
├── pyproject.toml            # Python 包配置、依赖声明及 pytest / ruff 规则
└── requirements.txt          # 核心运行时依赖清单
```

---

## 测试与代码规范检查

在仓库根目录下运行全量单元测试（覆盖 3D 资产几何、接触稳定性、多杯同心堆叠、螺旋螺纹自锁、触觉 Z 轴闭环自适应及 Sim-to-Real 安全门禁）与代码静态检查：

```bash
# 运行全部单元测试与物理仿真回归测试
pytest -v

# 运行 Ruff 代码规范检查
ruff check assets/ common/ examples/
```

---

## 开源许可

本仓库源码与参数化资产生成工具遵循 [MIT License](LICENSE) 开源协议；衍生机器人与夹爪网格资产遵循 [NOTICE](NOTICE) 中列出的上游开源协议。

