# 01 — xArm 6 一次性纸杯抓取与翻转竖放（仿真与 Sim-to-Real）

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 与 UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305（Eye-in-Hand）的一次性纸杯抓取与翻转竖放示例：从桌面夹起侧卧纸杯（质量 `5.74 g`，底径 `53 mm`，顶径 `75 mm`，高 `88 mm`），在空中翻转 `90°` 并竖直放置到目标位置，使杯底着桌、杯口朝上。

## 配置文件 (`config.yaml`)

默认配置位于 `examples/pick_cup/config.yaml`，支持配置运行模式、MuJoCo 预演是否无头运行以及放置后闭环拍照确认：

```yaml
mode: sim
headless: false  # false: 默认弹出 MuJoCo 3D 可视化窗口预演；true: 无头模式后台运行

robot:
  ip: "192.168.1.60"
  calibration: "../ufactory_mujoco/calibrations/xarm6_kinematics_L56A01.yaml"

camera:
  handeye: "../camera_tool/result/eye_in_hand.yaml"

task:
  post_verify: true  # 真机完成竖放回到观测位后，自动通过 Gemini 305 拍照进行闭环确认
  save_vis: "build/pick_cup/real_detection.png"
  save_after_vis: "build/pick_cup/real_after_place.png"
```

## 运行模式

在项目根目录下执行：

```bash
# 1. 纯 MuJoCo 仿真模式（默认读取 config.yaml，弹出 3D 可视化窗口）
python -m examples.pick_cup.run --mode sim
python -m examples.pick_cup.run --mode sim --headless

# 2. Sim-to-Real 阶段一：实机感知 + MuJoCo 可视化数字孪生预演（零硬件运动）
python -m examples.pick_cup.run --mode dry-run --report build/pick_cup/dry_run.json

# 3. Sim-to-Real 阶段二：xArm 控制器内部仿真演练（mode=4, state=5，物理机械臂不动）
python -m examples.pick_cup.run --mode sdk-sim --report build/pick_cup/sdk_sim.json

# 4. Sim-to-Real 阶段三：Gemini 305 感知 -> MuJoCo 可视化预演 -> 真机执行 -> 拍照闭环确认
python -m examples.pick_cup.run --mode real --confirm-real --report build/pick_cup/real.json

# 可选：手动指定桌面纸杯位姿（绕过相机感知，X/Y 单位 m，YAW_DEG 为杯底指向杯口的平面角）
python -m examples.pick_cup.run --mode dry-run --cup-pose 0.44 -0.05 135.0
```

## 架构与控制说明

- **纸杯资产**（`assets/cup/generate_paper_cup.py`）：生成 1:1 视觉网格（`paper_cup.obj`）与中空碰撞几何（底面圆柱 `0.001180 kg` + 24 段圆台倾角盒体 `24 × 0.000190 kg`，总质量 `0.00574 kg`）。
- **Eye-in-Hand 感知与放置后闭环验证**（`examples/pick_cup/perception.py`）：
  - 采集 Gemini 305 对齐 RGB-D 帧，并结合手眼标定 YAML 反投影至基座坐标系。
  - 在安全工作空间范围（`x ∈ [0.20, 0.60] m, y ∈ [-0.26, 0.12] m`）内提取白色纸杯点云，基于 2D PCA 主轴与圆台两端正交宽度/离桌高度自动判别杯底（`53 mm`）与杯口（`75 mm`）朝向，计算距杯底 `40 mm` 抓取中心点。
  - 真机放置完成后，自动在观测位二次采集 RGB-D 帧（`verify_upright_cup`），对目标区域内竖直纸杯的水平位置偏差、杯口高度与前景像素进行闭环确认，并输出标注图（`save_post_place_overlay`）。
- **数字孪生可视化预演与安全门禁**（`examples/pick_cup/sim2real.py`）：
  - 在任何实机或控制器轨迹下发前，先在加载实机运动学标定的 MuJoCo 数字孪生环境中执行完整物理预演。
  - 全程校验：自碰撞与桌面碰撞最小间距 `>= 5 mm`、腕部相机离桌高度 `>= 95 mm`、关节步进 `<= 0.08 rad`、仿真竖放稳定无倾倒；若预演窗口被提前关闭则立即阻断真机执行。
  - 真机 G2 夹爪采用柔顺力控闭合（目标开度 `55 mm`、速度 `20%`、力 `15%`），夹住纸杯腰部微弹性形变约 `52.5 mm` 而不捏瘪杯身。

## 测试

```bash
pytest examples/pick_cup/test_pick_cup.py -v
```

