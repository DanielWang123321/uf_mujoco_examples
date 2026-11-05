# 02 — xArm 6 多个一次性纸杯夹取与同心套杯堆叠（仿真与 Sim-to-Real）

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 与 UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305（Eye-in-Hand）的多个一次性纸杯连续抓取、空中翻转与同心套杯堆叠示例：依次从桌面夹起侧卧纸杯（单杯质量 `5.74 g`，底径 `53 mm`，顶径 `75 mm`，高 `88 mm`，壁厚 `1 mm`），在空中翻转 `90°` 使杯口朝上，先将底层基准杯竖直放置于目标位置，再将后续纸杯同心对准并套入下层纸杯内部完成叠杯。

## 配置文件 (`config.yaml`)

默认配置位于 `examples/stack_cups/config.yaml`，支持配置运行模式、纸杯数量、MuJoCo 预演是否无头运行以及叠杯后闭环拍照确认：

```yaml
mode: sim
headless: false  # false: 默认弹出 MuJoCo 3D 可视化窗口；true: 无头模式后台运行

robot:
  ip: "192.168.1.60"
  calibration: "../ufactory_mujoco/calibrations/xarm6_kinematics_L56A01.yaml"

camera:
  handeye: "../camera_tool/result/eye_in_hand.yaml"

task:
  num_cups: 2              # 参与夹取-堆叠的纸杯数量 (支持 2 或 3，默认 2)
  target_xy: [0.30, 0.20]  # 目标堆叠中心水平坐标 [x_m, y_m]
  post_verify: true        # 真机完成叠杯回到观测位后，自动通过 Gemini 305 拍照进行闭环确认
  save_vis: "build/stack_cups/real_detection.png"
  save_after_vis: "build/stack_cups/real_after_stack.png"
  report: "build/stack_cups/sim_report.json"
```

## 运行模式

在项目根目录下执行：

```bash
# 1. 纯 MuJoCo 仿真模式（默认 2 杯同心套杯堆叠，弹出 3D 可视化窗口）
python -m examples.stack_cups.run --mode sim
python -m examples.stack_cups.run --mode sim --headless --num-cups 2 \
  --report build/stack_cups/sim_2cups.json
python -m examples.stack_cups.run --mode sim --headless --num-cups 3 \
  --report build/stack_cups/sim_3cups.json

# 2. Sim-to-Real 阶段一：Gemini 305 多纸杯感知 + MuJoCo 数字孪生预演（零硬件运动）
python -m examples.stack_cups.run --mode dry-run --num-cups 2 \
  --report build/stack_cups/dry_run.json

# 3. Sim-to-Real 阶段二：xArm 控制器内部仿真演练（mode=4, state=5，物理机械臂不动）
python -m examples.stack_cups.run --mode sdk-sim --num-cups 2 \
  --report build/stack_cups/sdk_sim.json

# 4. Sim-to-Real 阶段三：Gemini 305 多杯感知 -> MuJoCo 预演 -> 真机双杯同心堆叠 -> 拍照闭环确认
python -m examples.stack_cups.run --mode real --confirm-real --num-cups 2 \
  --report build/stack_cups/real.json

# 可选：手动指定多个侧卧纸杯位姿 (X1 Y1 YAW1 X2 Y2 YAW2 ...)
python -m examples.stack_cups.run --mode dry-run --num-cups 2 \
  --cup-poses 0.568 -0.168 -142.2 0.503 0.081 -61.0 \
  --target-xy 0.36 -0.06
```

## 物理机制与 Sim-to-Real 控制架构

- **Eye-in-Hand 多纸杯 6D 位姿感知与叠杯后闭环验证**（`examples/stack_cups/perception.py`）：
  - 在安全工作区（`x ∈ [0.20, 0.60] m, y ∈ [-0.26, 0.12] m`）内分割多个白色纸杯连通域，通过点云规模与圆台尺寸门限滤除背景边缘伪影，基于 2D PCA 主轴与两端正交宽度/离桌高度自动判别每个纸杯的杯底（`53 mm`）与杯口（`75 mm`）朝向，并按从右（`-Y`）到左（`+Y`）顺序排列抓取次序。
  - 真机完成所有纸杯堆叠回到观测位后，自动二次采集 RGB-D 帧（`verify_stacked_cups`），对目标堆叠区域的纸杯像素、水平中心偏差、顶层杯口离桌高度及原位侧卧纸杯清空状态进行闭环确认。
- **三级接触优先级（Contact Priority）解耦**：
  - **Gripper G2 指垫**（`priority=2, friction=4.0`）：提供高切向摩擦，确保 `5.74 g` 纸杯在腰部柔顺夹持与 `90°` 空中翻转中零滑脱。
  - **桌面**（`priority=2, friction=1.5`）：防止待抓侧卧纸杯滚动，并确保底层基准杯在套杯过程中稳固不移位。
  - **纸杯壁面**（`priority=1, friction=0.08`）：满足 $\mu_{\text{cup-cup}} = 0.08 < \tan\alpha = 0.125$（圆台半锥角 $\alpha \approx 7.125^\circ$），使上层纸杯在释放后沿下层纸杯圆台内壁顺滑自定心滑入嵌套位。
- **杯口内嵌防滑出释放与防夹爪干涉**：
  - 堆叠第 $k$（$k \ge 2$）个纸杯时，先悬停于下层纸杯杯口上方 `25 mm` 同心对齐，再垂直下探至 $z_{\text{bot}, k} = z_{\text{rim}, k-1} - 4\text{ mm}$ 处缓速张开夹爪。此时上杯底面（外径 `53 mm`）已深入下杯内腔口（内径 `73 mm`）下方 `4 mm`，受下杯口内壁 $360^\circ$ 环绕限位，而夹爪指垫整体仍位于下杯口上方约 `22 mm` 处。

## 单元测试

```bash
pytest examples/stack_cups/test_stack_cups.py -v
```

