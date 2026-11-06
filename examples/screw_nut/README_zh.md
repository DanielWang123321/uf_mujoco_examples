# screw_nut — 将六角法兰螺母拧到固定螺丝上

[English](README.md) | [中文](README_zh.md)

M12 螺丝（56.5 g，长 45 mm，头部 Ø20 × 14 mm）竖直粘在桌面上。机械臂夹取六角法兰螺母（29 g，法兰 Ø29 × 3.8 mm，六角体 Ø20 mm，高 14.5 mm），放到螺丝顶端，分多次顺时针旋转把螺母拧到螺丝头部。

## 运行

```bash
python -m examples.screw_nut                                  # 仿真（带窗口）
python -m examples.screw_nut --headless
python -m examples.screw_nut --headless --nut-pose 0.33 0.05 25 --screw-xy 0.38 -0.08 --thread-turns 3

python -m examples.screw_nut --mode dry-run                   # 相机 + 数字孪生，机械臂不动
python -m examples.screw_nut --mode sdk-sim                   # 控制器仿真，机械臂不动
python -m examples.screw_nut --mode real --confirm-real       # 真机
```

| 参数 | 含义 |
| :--- | :--- |
| `--nut-pose X Y YAW_DEG` | 螺母位置（m）和六角朝向。仿真默认值为 `config.yaml` 中的 `task.nut_xy` / `task.nut_yaw_deg`。 |
| `--screw-xy X Y` | 螺丝位置（m）。仿真默认值为 `task.screw_xy`。硬件模式下同时给出 `--nut-pose` 和 `--screw-xy` 时不使用相机识别。 |
| `--thread-turns N` | 螺母套上螺丝顶端后的顺时针圈数，`0 < N ≤ 12`（默认 10）。 |

公共参数（`--mode`、`--headless`、`--ip`、`--calibration`、`--handeye`、`--camera-serial`、`--post-verify`、`--save-vis`、`--save-after-vis`、`--report`）及默认值见 [`config.yaml`](config.yaml)。

## 原理

**仿真**（`screw_nut.py`）

1. TCP 距桌面 9 mm 夹持六角体（指尖高于法兰），实测开度须在 12–24.5 mm 之间。
2. 抬起螺母，移到螺丝上方，下降套入螺丝顶端（螺母与螺丝偏差 ≤ 6 mm）。
3. 拧紧：每一次旋转中 J6 顺时针转动，TCP 每圈下降一个螺距。总圈数按 J6 ∈ [−360°, +360°] 拆分：10 圈 = 6 × 1.5 + 1 × 1.0（共 7 次）。两次旋转之间夹爪张开到 50 mm，J6 反转回位后重新夹紧。
4. 松开、撤离，并要求静置 1 s：螺母落在螺丝头上（z ≈ 14 mm），水平误差 ≤ 1.5 mm，倾斜 ≤ 5°。

螺丝碰撞模型有 12 圈螺旋凸棱（螺距 2.4 mm），螺母内孔有 8 个内齿，松开后螺母停留在螺纹上不下滑。指垫使用高摩擦（4.0），螺母与螺丝之间使用低摩擦（0.10）。

**Sim-to-Real**（`sim2real.py`、`perception.py`）

1. 采集一帧 RGB-D，识别螺母和螺丝（相对绿色桌垫做前景分割 + 深度）。
2. 在加载标定参数的数字孪生中运行任务。对 50 Hz 关节轨迹的每个采样点检查关节限位、碰撞、腕部相机距桌面 ≥ 88 mm、关节步长 ≤ 0.15 rad。
3. `sdk-sim` 在控制器仿真模式中回放轨迹。`real` 模式：
   - 相机分别移动到螺母和螺丝正上方 250 mm 处重新测量，并重新规划；
   - 以 25 % 夹持力夹取；实测开度 < 12 mm（空抓）则中止；
   - 每次 J6 反转前要求夹爪张开 ≥ 42 mm（避开法兰）；
   - 每次重新夹紧后，若开度 > 21.2 mm，说明手指滑到了 Ø29 mm 法兰上：夹爪张开，TCP 上抬 3.5–5.0 mm 后重新夹紧（每次旋转最多 3 次，结束时最多追加 3 次旋转）；
   - 每次旋转记录到 `build/screw_nut/real_execution.log`。
4. 返回后在螺丝正上方拍照确认结果（螺母在螺丝上，水平误差 ≤ 15 mm，螺母顶面高度与预期相差 ≤ 9 mm，螺母原位置已空）。

> [!NOTE]
> 仿真螺距（2.4 mm）大于真实 M12 螺距。真机上由上述重新夹紧开度检查补偿两者差异。

## 文件

| 文件 | 内容 |
| :--- | :--- |
| `screw_nut.py` | `ScrewNutRobot`、`run_screw_nut`、命令行 |
| `perception.py` | `detect_nut_and_screw_overview`、`detect_nut_near`、`detect_screw_near`、`verify_threaded_nut` |
| `sim2real.py` | 数字孪生预演、sdk-sim、真机执行 |
| `scene.xml` | xArm 6 + G2、桌面、螺丝、螺母 |
| `test_screw_nut_sim.py` | `pytest examples/screw_nut` |

物体网格和碰撞模型由 [`assets/screw_nut/generate_screw_nut.py`](../../assets/screw_nut/generate_screw_nut.py) 生成。
