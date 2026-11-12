# pick_cup — 纸杯夹取、翻转与竖放

[English](README.md) | [中文](README_zh.md)

一次性纸杯（5.74 g，底部 Ø53 mm，杯口 Ø75 mm，高 88 mm）侧卧在桌面上。机械臂夹取纸杯，空中翻转 90° 后竖直放到目标点。

## 运行

```bash
python -m examples.pick_cup                                   # 仿真（打开窗口）
python -m examples.pick_cup --headless
python -m examples.pick_cup --cup-pose 0.46 -0.07 140 --target-xy 0.38 -0.18

python -m examples.pick_cup --mode dry-run                    # 相机 + 数字孪生，不运动
python -m examples.pick_cup --mode sdk-sim                    # 控制器仿真，不运动
python -m examples.pick_cup --mode real --confirm-real        # 真机
```

| 参数 | 含义 |
| :--- | :--- |
| `--cup-pose X Y YAW_DEG` | 杯底位置（m）与杯底→杯口偏航角。硬件模式下替代相机检测。 |
| `--target-xy X Y` | 竖放目标点（m）。仿真默认 `(0.30, 0.15)`；硬件默认 `(0.38, -0.18)`，与纸杯距离小于 110 mm 时改用 `(0.38, 0.04)`。 |

公共参数（`--mode`、`--headless`、`--ip`、`--calibration`、`--handeye`、`--camera-type`、`--camera-serial`、`--post-verify`、`--save-vis`、`--save-after-vis`、`--report`）及默认值见 [`config.yaml`](config.yaml)。

默认相机为 D435/D435i，1280×720@30。Gemini 305 使用 `--camera-type orbbec`（1280×800@30）。`camera.handeye` 留空时按相机类型选择；`--handeye` 可覆盖。实时采集为检测、复测和验证传递彩色内参，详见[相机配置](../../README_zh.md#硬件配置)。

## 流程

**仿真**（`pick_cup.py`）

1. 夹爪竖直，在距杯底 40 mm 处夹持；先闭合到 53 mm，回退 1 mm，再以 20 Hz 力反馈保持约 2 N 接触力。
2. 抬起，将杯轴翻转至竖直，搬运过程中按纸杯实时位置修正，下降至杯底接触桌面。
3. 松开、撤离，并要求纸杯独立站立 1 s（倾角 ≤ 10°，落点误差 ≤ 30 mm，漂移 ≤ 2 mm）。

夹持期间，若双侧指垫接触中断超过 0.10 s，或纸杯相对 TCP 位移超过 25 mm / 转角超过 25°，任务中止。

**Sim-to-Real**（`sim2real.py`、`perception.py`）

1. 读取关节状态并拍摄一帧 RGB-D。通过白色像素 + 高于桌面的深度分割纸杯，PCA 求杯轴，按两端宽度与高度区分杯底和杯口。
2. 在加载标定的数字孪生中执行任务并记录关节路点。对生成的 50 Hz 轨迹逐点检查关节限位、碰撞、腕部相机离桌高度 ≥ 95 mm、相邻步关节变化 ≤ 0.15 rad。
3. `sdk-sim` 在控制器仿真模式中回放轨迹。相机引导的 `real` 抓取前，先将相机移到杯子正上方（D435 为 `360 mm`，Gemini 为 `280 mm`）复测，再运行完整标定数字孪生；杯子缺失或复测预演失败则退回观察位。开度读数无效或低于 30 mm 时在抬升前中止。夹持开度 58 mm，速度 10 %，力 5 %。
4. 回到观测位后再拍一帧验证结果（纸杯像素 ≥ 1200，水平误差 ≤ 45 mm，杯口高度 ≥ 45 mm）。

## 文件

| 文件 | 内容 |
| :--- | :--- |
| `pick_cup.py` | `CupRobot`、`run_pick_cup`、命令行 |
| `perception.py` | `detect_sideways_cup`、共享 `detect_single_cup_near`、`verify_upright_cup`、可视化 |
| `sim2real.py` | 数字孪生预演、sdk-sim、真机执行 |
| `scene.xml` | xArm 6 + G2、桌面、纸杯 |
| `test_pick_cup.py` | `pytest examples/pick_cup` |
