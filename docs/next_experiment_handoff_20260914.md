# Go2-X5 locomotion 下一轮实验交接

更新时间：2026-09-14 13:21 CST

目标是用最短时间得到遥控响应快、base z 平稳、可迁移到真机的 locomotion policy。
下一轮 session 应先复现并定位问题，再启动小规模并行筛选；不要直接延长当前训练或扩大随机搜索。

## 当前用户反馈与证据边界

用户在 `rl_sar` MuJoCo 图形遥控中观察到：

1. 角速度跟踪明显偏慢，转向手感迟钝。
2. base 的 z 方向平稳度仍需改善，尤其需要检查它是否和转向或机械臂运动耦合。

这是重要的人工闭环反馈，但目前没有对应的固定输入 trace，不能从“手感慢”直接判断是手柄映射、
MuJoCo 执行器、策略响应还是训练奖励造成的。现有 12 s 自动 MuJoCo smoke 只有静止、直行和停止，
没有 yaw 阶跃；其 2.5--9.5 s 直行稳定段结果如下：

| 场景 | height mean | height std | height peak-to-peak | RMSE to 0.33 m |
|---|---:|---:|---:|---:|
| nominal | 0.33217 m | 0.00165 m | 0.00785 m | 0.00273 m |
| moving arm | 0.33223 m | 0.00171 m | 0.00860 m | 0.00281 m |

因此自动测试尚未复现 z 问题。不能用这组短测否定用户反馈，也不能把“12 s 未跌倒”解释为遥控性能通过。

## 可复现基线

### 源码与策略

- `robot_lab`：分支 `feat/vel-pose-track-simon`，提交 `8149832`
- `rl_sar`：分支 `fix/stand-gait-consistency`，提交 `1c131e0`
- 冻结基线 checkpoint：
  `logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-14_04-20-58_go2x5_mild_s2r_s42_selected_8200/model_8200.pt`
- checkpoint SHA256：`c0b15fc8018f692d6587d9e8131f763288da43ca3597733312897182aa19168b`
- RL-SAR bundle：
  `/home/simon/Projects/Simon/wbc_rl_mpc/rl_sar/policy/go2_x5/robot_lab_mild_s2r_s42_8200/`
- TorchScript SHA256：`73621cb766c80039a4333cec48187ec1e57864e6c61c138b91cffe1d23f01b84`

该策略是单帧 64D 输入、12D 腿动作、50 Hz policy。`model_8200.pt` 是当前经过匹配 IsaacLab
评估和 MuJoCo 导出的基线；不要因为训练目录出现更晚 checkpoint 就自动替换它。

### 已有定量结果

在固定 `seed=202`、`domain_rand=benchmark`、`robustness_iteration=8000`、三个相同 arm mode、
每个模式 1200 steps 的 IsaacLab 匹配评估中：

| checkpoint | overall failure | hard failure | 结论 |
|---|---:|---:|---|
| 385/model_8200 | 1.223% | 5.54% | 当前基线 |
| 387/model_8200 | 1.954% | 8.72% | 不晋级 |

原始指标位于服务器快照：

```text
/home/simon/Projects/Locomotion/robot_lab_e4_20260914_041541/
  cluster_runs/evaluation_8h_matched1200/p385-{hold,random,structured}/evaluation/locomotion_metrics.jsonl
  cluster_runs/evaluation_8h_matched1200/p387-{hold,random,structured}/evaluation/locomotion_metrics.jsonl
```

RL-SAR 的固定输入 MuJoCo smoke 在 MuJoCo 3.2.7 中完成 nominal、moving-arm、push 三个 12 s
场景，未触发跌倒判据。它只证明导出、64D 观测、12D 动作和闭环执行可运行，不证明 yaw 响应或真机效果。

## 服务器训练状态快照

以下状态只对应 2026-09-14 13:21 CST，下一轮开始必须重新查询：

| Job | 配方 | 状态 | 最后 checkpoint | 真实判断 |
|---:|---|---|---|---|
| 385 | mild sim2real seed42 | RUNNING，约 8:58 | model_9100 | 唯一仍在运行；未评估 9100 |
| 386 | mild sim2real seed73 | Slurm COMPLETED | model_1400 | Python 因 negative std 异常退出，不是完整训练 |
| 387 | mild practical seed42 | Slurm COMPLETED | model_9000 | Python 因 negative std 异常退出，不是完整训练 |
| 388 | flat sim2real seed42 | Slurm COMPLETED | model_1400 | Python 因 negative std 异常退出，不是完整训练 |

386/388 在退出前出现极大的 `action_rate_l2`，387 的最后打印均值没有同样的大值，但终止异常相同。
Slurm 和 `completion.env` 都记录 exit 0，因此以后必须同时检查 stderr traceback、最后 checkpoint、
fault dump 和成功标志，不能用 Slurm `COMPLETED` 宣称训练完成。

## 已确认的结构性问题

### P0：训练数值保护不足

- RSL-RL 当前使用 `noise_std_type: scalar`，`std` 是无约束参数；三个 job 最终都可能走到负 std。
- `agent.clip_actions` 当前为 `null`。虽然 action term 会裁剪执行目标，但 action observation 和
  `action_rate_l2` 可能先看到未约束 raw action，巨大的有限值足以污染 reward/value target。
- 下一轮任何新训练之前必须先改为 `noise_std_type: log`、在 wrapper 入口裁剪到 `[-3, 3]`，
  并确认 action-rate 使用的也是有界动作。
- 从 scalar-std checkpoint 启动 log-std actor 时需要显式迁移 `log_std=log(clamp(std, min))`；
  只加载有限的 actor/critic 权重，不沿用未经审计的 optimizer state。

### P0：IsaacLab 与 MuJoCo 执行器限幅不一致

- IsaacLab 当前腿 actuator 使用统一 `effort_limit=33.5 Nm`。
- RL-SAR Go2-X5 MJCF 使用 hip/thigh `23.7 Nm`、calf `45.43 Nm`。
- X5 在 IsaacLab 中为统一 `20 Nm`，MJCF 中前三轴 `27 Nm`、后三轴 `7 Nm`。
- 导出 YAML 无法覆盖 MJCF `<motor ctrlrange>` 的实际裁剪。转向依赖 hip/thigh 力矩，较低的 MuJoCo
  限幅可能直接造成 yaw 迟缓。下一轮应做 training-limit 与 hardware-limit 的 A/B 诊断；面向真机的
  最终方案应让训练使用硬件实际分轴限幅，而不是把 MuJoCo 放宽后就宣称问题解决。

### P1：yaw 奖励对大误差的梯度偏弱

`track_ang_vel_z_exp` 当前 weight 为 6、std 为 0.5 rad/s。误差达到 1 rad/s 时指数奖励已接近零，
对“从几乎不转到开始转”的改善提供的梯度较弱。现有训练末尾打印的 `error_vel_yaw` 约
0.25--0.26 rad/s，也明显高于 XY 跟踪误差的量级。只有在输入映射和执行器 A/B 后两侧仿真都慢时，
才进入 yaw reward 实验。

### P1：策略没有 base height 闭环观测

64D actor 看到 height command 和 base linear velocity z，但没有当前 terrain-relative height 或 height error。
`track_height_exp(std=0.06 m)` 能提供训练信号，却不能让部署策略根据实际高度做闭环修正。若固定 yaw
复现表明 IsaacLab 和 MuJoCo 都存在 z 波动，优先增加 1D terrain-relative height error，而不是先堆更大
height reward。真机使用该版本前，必须明确状态估计器提供的 base z 与局部地面参考。

## 下一轮必须先做的无训练诊断

### 1. 固定命令记录

扩展同一套 evaluator，使 IsaacLab 和 MuJoCo 都执行以下 50 Hz 命令序列：

1. 2 s：`[vx, vy, wz] = [0, 0, 0]`
2. 4 s：`wz = +0.4 rad/s`
3. 2 s：归零
4. 4 s：`wz = +0.8 rad/s`
5. 2 s：归零
6. 4 s：`wz = -0.8 rad/s`
7. 4 s：`vx = 0.4 m/s, wz = +0.8 rad/s`
8. 2 s：归零

固定 `seed=[202, 303, 404]`，分别运行 arm hold 和 structured。逐步记录：

- clamp 后实际送入 actor 的 `vx/vy/wz/height/roll/pitch/yaw`
- world-z 与 body-z angular velocity
- terrain-relative base height、world-z linear velocity
- 12D raw/clipped action、position target、实际 torque 与 saturation fraction
- roll/pitch、足端接触和失败状态

报告 yaw 的 10--90% rise time、1 s 尾窗稳态误差、overshoot、IAE 和正负方向对称性；报告 z 的
均值误差、去均值 RMS、peak-to-peak、vertical-velocity RMS，以及转向开始后 0--2 s 和 2--4 s 窗口。
所有对比使用相同 command trace 和 common seeds。

### 2. 遥控输入门槛

直接 joydev 当前将 axis 3 映射到 `control.yaw`，范围应为 `[-1, 1]`，deadzone 为 0.05。记录一次完整
右摇杆扫描。如果 clamp 后命令达不到 ±0.8 rad/s，先修输入映射或手柄轴选择，不启动策略训练。

### 3. 定位决策

| 结果 | 下一步 |
|---|---|
| actor 输入命令本身偏小/错误 | 只修 RL-SAR 输入映射并复测 |
| IsaacLab yaw 正常、当前 MuJoCo yaw 慢 | 做执行器限幅 A/B；按硬件限幅改训练模型 |
| 两边 yaw 都慢 | 进入 Y1 yaw reward 实验 |
| z 只在 MuJoCo 或只在 yaw 时波动 | 查执行器、接触和 yaw-z 耦合，不先改 height reward |
| 两边都有 z 波动 | 进入 Z1 height-observation 实验 |

## 条件满足后启动的四路筛选

四路均从冻结且已评估的 `385/model_8200` 做 weight-only 初始化，使用 mild terrain、sim2real DR、
`seed=42`、4096 envs。先跑 1500 iterations，在 500/1000/1500 做相同固定评估；不要先做多 seed。

| ID | 改动 | 目的 |
|---|---|---|
| R0 | log std + wrapper action clip + bounded action-rate；奖励和 64D obs 不变 | 数值安全对照 |
| Y1 | R0 + yaw exp std `0.5 -> 0.3`，weight 保持 6 | 提高 yaw 误差敏感度 |
| Z1 | R0 + 1D terrain-relative height error，形成 65D contract | 验证 z 的可观测性瓶颈 |
| YZ1 | Y1 + Z1 | 检查两项改动能否兼容 |

65D warm start 必须保持旧 64D 各项顺序不变，在 projected gravity 后插入 height error。Actor 和 critic
第一层在新列处初始化为 0，其余列逐项复制；新 contract 命名为 `go2_x5_locomotion_v3_65`，同步修改
checkpoint 检查、导出器和 RL-SAR observation。必须用固定 probe 验证：当 height-error 新列为 0 时，
迁移模型与原模型输出逐元素一致。

如果无训练诊断已证明 hardware-limit 模型是 yaw 主因，则用“按硬件分轴 torque limit 训练”替换 Y1，
不要同时改变 torque limit 和 yaw reward。这样才能判断因果。

## 晋级门槛

任何候选先过以下门槛：

1. 1500 iterations 无 traceback、fault dump、非有限张量或 std/action/value 异常；Slurm exit 不是充分证据。
2. 固定 yaw trace 相对 R0：稳态绝对误差下降至少 30%，rise time 不增加，正负方向无明显偏置。
3. 复现 z 问题的窗口相对 R0：height RMS 或 peak-to-peak 至少下降 25%。
4. XY velocity RMSE 不恶化超过 10%；整体 failure 不高于 R0 0.5 个百分点。
5. 在已有 benchmark 协议中，hard failure、foot slip、EE induced vz 和 torque saturation 不出现明显退化。

只给第一名补 `seed=73`，然后重新导出 RL-SAR，在相同 MuJoCo trace 和图形遥控下复核。固定输入指标
通过后才评价遥控手感；MuJoCo 通过仍不能直接推出真机效果。

## 下一轮启动检查表

```bash
cd /home/simon/Projects/Locomotion/robot_lab
git status --short --branch
git rev-parse HEAD                    # expected: 8149832...

ssh login01.e48f.corelab.wan
squeue -j 385,386,387,388
sacct -j 385,386,387,388 --format=JobID,State,Elapsed,ExitCode
```

随后按顺序执行：

1. 保存 385 的实时状态、最新 checkpoint、stderr、completion/fault receipt；不自动续训。
2. 对 385 的 9000/最新有限 checkpoint 做与 model_8200 相同的固定评估，仅在明确胜出时更新基线。
3. 实现固定 yaw/z trace 和数值保护，先做短 smoke。
4. 完成无训练的 IsaacLab/MuJoCo/actuator-limit 定位。
5. 根据定位结果提交四路 1500-iteration 筛选。
6. 使用 raw sums/counts、common seeds 和多个尾窗生成决策表，再决定续训或 seed 验证。

当前最高优先级不是继续堆训练时长，而是得到能复现遥控问题的固定 trace，并修复会让训练假完成的
negative-std/action-rate 数值路径。
