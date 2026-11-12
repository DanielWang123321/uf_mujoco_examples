# obstacle_avoidance — 抓取方块并绕行圆柱

[English](README.md) | [中文](README_zh.md)

机械臂夹取桌面上的方块（50 × 50 × 50 mm，80 g），规划无碰撞路径绕过桌面中央的固定圆柱（Ø90 × 450 mm），将方块放置到目标区域。本示例**仅支持 `sim` 模式**。

## 运行

```bash
python -m examples.obstacle_avoidance                                   # 仿真（打开窗口）
python -m examples.obstacle_avoidance --headless                        # 无窗口仿真
python -m examples.obstacle_avoidance --headless --seed 1
python -m examples.obstacle_avoidance --start-pose 350 -200 300 180 0 0 --goal-pose 350 200 300 180 0 0
```

| 参数 | 含义 |
| :--- | :--- |
| `--headless` / `--no-headless` | 是否使用无窗口模式（默认打开窗口）。 |
| `--start-pose X Y Z R P Y` | 抓取抬升后的起始悬停位姿（mm，外旋 XYZ 角度），默认 `[350, -200, 300, 180, 0, 0]`。 |
| `--goal-pose X Y Z R P Y` | 绕障后的目标悬停位姿，默认 `[350, 200, 300, 180, 0, 0]`（其 XY 即放置点）。 |
| `--seed N` | 随机种子（控制逆解初值、RRT 搜索与路径简化，默认 `0`）。 |
| `--config PATH` | 配置文件路径（默认 [`config.yaml`](config.yaml)）。 |
| `--report PATH` | JSON 运行报告路径（默认 `build/obstacle_avoidance/sim_report.json`）。 |

规划参数（RRT 步长、迭代次数、规划净空 `clearance_m`）与运动限制（执行净空 `execution_clearance_m`、关节最大速度/加速度）见 [`config.yaml`](config.yaml)。参数优先级：命令行 > `config.yaml` > 默认值。

## 流程

1. **抓取与抬升**：从起始悬停位下探，闭合夹爪夹取方块（初始中心 `[350, -200, 25]` mm），确认双侧指垫接触后抬升回起始悬停位。
2. **携物绕障规划**：根据当前关节角及方块相对 TCP 的实测位姿建立携物碰撞模型，使用 RRT-Connect 搜索避开圆柱（中心 `[350, 0, 0]` mm）的路径，并完成路径简化与时间参数化。窗口中**红线**为会碰撞的关节直连基线（不执行），**绿线**为规划出的 TCP 绕障路径。
3. **搬运与放置**：沿规划轨迹绕过圆柱到达目标悬停位，下降至方块接触桌面后松爪撤离，并验证方块独立静置 1 s（落点误差 ≤ 5 mm，漂移 ≤ 1 mm，倾角 ≤ 5°）。

搬运期间全程检查机械臂、夹爪与方块的碰撞净空，并监测夹持状态（相对位移 > 5 mm、相对转角 > 5° 或双侧接触中断 > 0.1 s 时中止任务）。

## 文件

| 文件 | 内容 |
| :--- | :--- |
| `obstacle_avoidance.py` | `ObstacleRobot`、任务主流程与命令行入口 |
| `planning.py` | 逆解求解、碰撞检测、RRT 携物规划与轨迹时间参数化 |
| `manipulation.py` | 抓取、抬升、放置动作与夹持滑脱监测 |
| `scene.xml` | xArm 6 + G2、桌面、圆柱障碍物与方块 |
| `config.yaml` | 默认悬停位姿、规划参数与运动限制 |
| `test_obstacle_avoidance.py` | `pytest examples/obstacle_avoidance` |
