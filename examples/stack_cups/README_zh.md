# stack_cups — 夹取多个纸杯并套叠成一摞

[English](README.md) | [中文](README_zh.md)

桌面上侧卧 2 或 3 个一次性纸杯（5.74 g，底径 53 mm，口径 75 mm，高 88 mm）。机械臂依次夹取每个纸杯，在空中翻转竖直，第一个放到目标点，之后的纸杯依次套入前一个纸杯。

## 运行

```bash
python -m examples.stack_cups                                 # 仿真（带窗口）
python -m examples.stack_cups --headless --num-cups 3
python -m examples.stack_cups --headless --cup-poses 0.47 -0.15 -60 0.50 0.08 30 --target-xy 0.36 -0.06

python -m examples.stack_cups --mode dry-run                  # 相机 + 数字孪生，机械臂不动
python -m examples.stack_cups --mode sdk-sim                  # 控制器仿真，机械臂不动
python -m examples.stack_cups --mode real --confirm-real      # 真机
```

| 参数 | 含义 |
| :--- | :--- |
| `--num-cups {2,3}` | 纸杯数量（默认为 `config.yaml` 中的 `task.num_cups`）。 |
| `--target-xy X Y` | 堆叠位置（m），默认 `task.target_xy`。硬件模式下若未指定，则自动选择距所有纸杯 ≥ 110 mm 的空位。 |
| `--cup-poses X1 Y1 YAW1 ...` | 各纸杯杯底位置（m）及杯底→杯口朝向。硬件模式下指定后不使用相机识别。 |

公共参数（`--mode`、`--headless`、`--ip`、`--calibration`、`--handeye`、`--camera-serial`、`--post-verify`、`--save-vis`、`--save-after-vis`、`--report`）及默认值见 [`config.yaml`](config.yaml)。

## 原理

**仿真**（`stack_cups.py`）

1. 对每个纸杯：侧向夹取、抬起，将杯轴翻转为竖直（夹取方式同 [`pick_cup`](../pick_cup)）。
2. 将杯底移动到目标点（第一个杯）或前一个杯口上方 25 mm 处，并按纸杯实时位置修正。
3. 第一个杯：下降至接触桌面。后续纸杯：下降至杯底低于前一杯口 4 mm（手指仍高于杯口约 22 mm）后松开，纸杯滑入下方纸杯。
4. 最后一个纸杯放完后要求静置 1 s：堆叠水平误差 ≤ 30 mm，相邻纸杯同轴偏差 ≤ 8 mm，倾斜 ≤ 10°。

纸杯之间使用低摩擦（0.08 < tan 7.1°，即锥面半角），松开后上层纸杯在下层纸杯内自动对中。指垫使用高摩擦（4.0）。

**Sim-to-Real**（`sim2real.py`、`perception.py`）

1. 采集一帧 RGB-D，识别所有白色侧卧纸杯（PCA 求杯轴，按宽度和高度区分杯底/杯口），按从右（−Y）到左（+Y）的顺序夹取。
2. 在加载标定参数的数字孪生中运行全部流程。对 50 Hz 关节轨迹的每个采样点检查关节限位、碰撞、腕部相机距桌面 ≥ 95 mm、关节步长 ≤ 0.15 rad。
3. `sdk-sim` 在控制器仿真模式中回放轨迹。`real` 模式：
   - 每次夹取前，相机移动到纸杯正上方 280 mm 处重新测量并重新规划夹取；纸杯不见则回到初始位姿；
   - 实测夹持开度 < 30 mm（空抓）则中止并回到初始位姿。
4. 返回后再拍一帧确认堆叠结果（纸杯像素 ≥ 1200，水平误差 ≤ 45 mm，杯口高度 ≥ 50 mm，纸杯原位置已空）。

## 文件

| 文件 | 内容 |
| :--- | :--- |
| `stack_cups.py` | `StackCupRobot`、`run_stack_cups`、命令行 |
| `perception.py` | `detect_sideways_cups`、`detect_single_cup_near`、`verify_stacked_cups`、叠加图 |
| `sim2real.py` | 数字孪生预演、sdk-sim、真机执行 |
| `scene.xml` | xArm 6 + G2、桌面、三个纸杯 |
| `test_stack_cups.py` | `pytest examples/stack_cups` |
