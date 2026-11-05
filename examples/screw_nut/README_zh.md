# 03 — xArm 6 机械臂拧螺母（螺丝胶水固定于桌面）仿真与闭环控制示例

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 与 UFACTORY xArm 6 + Gripper G2 + Orbbec Gemini 305（Eye-in-Hand）的拧螺母仿真及 Sim-to-Real 闭环控制示例：桌面已用胶水垂直固定一枚倒置的 M12 螺丝（质量 `56.5 g`，总长 `45.0 mm`，下部圆头 `Ø20.0 × 14.0 mm`，上部螺纹杆 `Ø12.0 × 31.0 mm`，含 **12 圈真实 3D 螺旋碰撞凸起**，螺距 `2.40 mm`），机械臂从桌面夹起一枚自由静置的六角法兰螺母（质量 `29.0 g`，底法兰 `Ø29.0 × 3.8 mm`，上部六角体 `Ø20.0 × 10.7 mm`，总高 `14.5 mm`，中空内孔含 **8 颗内螺旋啮合球齿**），同心套入螺丝顶端导引段并执行右旋螺纹**多把倒手（Ratchet）顺时针螺旋旋紧**（默认旋转 `10.0 圈 = 3600°`，分 7 把执行），最后释放螺母平坐于 `z = 14.0 mm` 螺丝头肩部台阶面。

本示例集成了公共模块 `common/`（`common.motion` 与 `common.vision`）的**停止软着陆加减速**、**抓取前正上方二次精测**以及**触觉 Z 轴闭环自适应**能力。

## 配置文件 (`config.yaml`)

默认配置位于 `examples/screw_nut/config.yaml`：

```yaml
mode: sim
headless: false  # false: 默认弹出 MuJoCo 3D 可视化窗口；true: 无头模式后台运行

robot:
  ip: "192.168.1.60"
  calibration: "../ufactory_mujoco/calibrations/xarm6_kinematics_L56A01.yaml"

camera:
  handeye: "../camera_tool/result/eye_in_hand.yaml"

task:
  nut_xy: [0.3538, -0.0114]    # 螺母初始桌面坐标 [x_m, y_m]
  nut_yaw_deg: 0.0             # 螺母初始绕 Z 轴偏航角 (deg)
  screw_xy: [0.3560, -0.1028]  # 胶水固定螺丝的桌面中心坐标 [x_m, y_m]
  thread_turns: 10.0           # 螺母套入螺杆后的顺时针旋紧圈数 (默认 10.0 圈 = 3600°，分 7 把倒手旋紧至底部肩台)
  post_verify: true            # 完成拧螺母后是否通过 Gemini 305 正上方拍照进行闭环确认
  save_vis: "build/screw_nut/real_detection.png"
  save_after_vis: "build/screw_nut/real_after_thread.png"
  report: "build/screw_nut/sim_report.json"
```

## 运行命令

在项目根目录下执行：

```bash
# 1. 默认纯仿真运行（弹出 MuJoCo 3D 可视化窗口，7 把倒手旋紧 10.0 圈）
python -m examples.screw_nut.run --mode sim

# 2. 无头模式运行并输出 JSON 评估报告
python -m examples.screw_nut.run --mode sim --headless \
  --report build/screw_nut/sim_report.json

# 3. 自定义螺母/螺丝位置与旋紧圈数（如 3.0 圈 = 1080°，2 把倒手旋紧后自锁于螺杆中部）
python -m examples.screw_nut.run --mode sim --headless \
  --nut-pose 0.33 0.05 25.0 \
  --screw-xy 0.38 -0.08 \
  --thread-turns 3.0 \
  --report build/screw_nut/sim_custom.json

# 4. Sim-to-Real 三阶段（dry-run / sdk-sim / real）
python -m examples.screw_nut.run --mode dry-run
python -m examples.screw_nut.run --mode sdk-sim
python -m examples.screw_nut.run --mode real --confirm-real
```

## 物理与闭环控制机制

1. **12 圈真实 3D 螺旋螺纹啮合与中途自锁**：
   - **螺杆物理碰撞**：由芯部圆柱（`r = 4.90 mm`）与 **12 圈右旋螺旋胶囊体（Capsule）凸起**（共 `192` 段，螺距 `p = 2.40 mm`，外径 `12.0 mm`）构成；
   - **螺母内孔物理碰撞**：底部 `7.2 mm` 同轴导引裙边（确保套入螺杆顶端时零倾角同心对准）+ 上部 **8 颗内螺旋球形啮合齿**（齿尖半径 `5.40 mm`，嵌入螺杆螺旋槽）；
   - 当夹爪在螺杆半途松开时，螺母依靠 3D 螺旋凸起物理自锁停留在当前螺纹高度（实测松爪下沉 `< 0.15 mm`）。
2. **多把倒手（Ratchet）旋紧与 `J6` `[-360°, +360°]` 限位保护**：
   - 将总旋紧圈数（如 `10.0 圈 = 3600°`）自动拆解为多个单把行程（前 6 把各 `1.5 圈 = 540° = 3 × 180°`，第 7 把 `1.0 圈 = 360°`）；
   - 每把顺时针旋紧同步下降 $\Delta z = p \cdot N_{\text{stroke}}$（`1.5 圈` 下降 `3.60 mm`）后，张开夹爪至 `50.0 mm`（螺母自锁停留于螺纹上），`J6` 逆时针回转倒手至起始角（`-346°`）并跟随螺母当前高度重新夹持上层六角体，全程严格保持在 `J6 ∈ [-360°, +360°]` 限位内。
3. **触觉 Z 轴闭环自适应（`MAX_HEX_FLAT_GAP_MM = 21.2 mm`）**：
   - 夹爪夹持力严格保持 `25.0%` 不上调；
   - 当螺母因入扣或螺纹摩擦出现周向打滑（实际下降量滞后于理论螺距下降量）时，机械臂若按开环高度重抓会撞到螺母底部 `Ø29 mm` 法兰或六角对角顶点（实测夹持开度 `> 21.2 mm`）；
   - 控制器在每把重抓后实时检测夹爪开度：若 `meas_gap > 21.2 mm`，自动张开夹爪并沿 Z 轴向上回退 `+3.5 mm`（`z_backoff_m`）重新夹持上部 `Ø20 mm` 六角对边（`18.5–19.8 mm`），并在原定行程结束后自动追加最多 `3` 把自适应补拧行程直至螺母到底。
4. **抓取前正上方二次精测与任务完成闭环确认（`common/vision.py` + `perception.py`）**：
   - 通过 `solve_camera_lookdown_q`（`z_cam = 250 mm`）消除斜视视差，在抓取前复测螺母圆心与偏航角（`detect_nut_near`）、在套入前复测固定螺丝中心与高度（`detect_screw_near`），并在旋紧完成后闭环确认螺母同轴就位高度（`verify_threaded_nut`）。

## 单元测试

```bash
pytest assets/screw_nut/test_screw_nut.py examples/screw_nut/test_screw_nut_sim.py -v
```

