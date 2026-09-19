# GO2-X5 locomotion optimization handoff

更新时间：2026-09-19  
任务：`RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0`

## 2026-09-19 session continuation

本 session 没有使用或修改其他项目的 Slurm 队列。工作区当前有未提交改动，涉及伸展诊断、USD 末端运动学、固定场景 smoke、成对 checkpoint 评估和续训脚本。

已经完成：

- 伸展诊断改为使用 curriculum 实际发出的 `extension_goal`，并记录目标误差、伸展力矩饱和、首次到位时间、末端前向误差和 command tracking error；
- 增加从当前 Go2-X5 USD 运动链计算 `link6` 位置的 `ArmKinematics`，模拟器实测位置对照最大误差约 `2.1e-6 m`；
- 增加固定零速度、100% 完全伸展、最终 curriculum 的 `--arm_full_extension_eval` smoke 模式；
- 增加 `scripts/reinforcement_learning/rsl_rl/evaluate_stability.py`，覆盖伸展站立、0.3/0.6 m/s 前进减速、左右横移和 push；
- IsaacLab 2.3.0 下 CPU 回归测试 **16/16 通过**；8 env、30 steps、`domain_rand=none` smoke 通过；policy/command/action contract 仍为 `63 / 6 / 12`。

第一组真实固定评估（5577，seed 42，16 个并行环境，20 s）没有跌倒，但平均机械臂目标误差约 `0.39 rad`，末端前向误差约 `0.056 m`，四脚接触率很低。轨迹中机械臂实际力矩达到约 `20 Nm`，而 saturation 指标仍报 `0`；这说明当前 PhysX effort limit 口径仍不能代表显式 PD actuator 的 `20 Nm` clip limit，不能据此宣称没有饱和。

评估输出：

```text
outputs/stability_20260919/paired_eval/
outputs/stability_20260919/paired_eval.log
```

5575、5577、5578 的 seed 42/73 均已完成，配对初始状态校验通过；没有遗留评估进程，不要重复运行这组评估。

成对评估摘要（每个场景 16 个 replica，结果文件未纳入 git）如下：

- 机械臂完全伸展场景的到位率为 `0`，平均实际目标误差约 `0.36–0.44 rad`，末端前向误差约 `0.054–0.063 m`；因此当前 12000-iteration checkpoint 尚未证明可以完成伸展保持。
- 5577 在运动场景整体最稳：两 seed 下前进/横移未出现系统性跌倒，0.6 m/s 前进四脚接触率约 `0.15`；但右移仍有 `0–6.25%` 跌倒，不能直接上机。
- 5575 的 0.6 m/s 前进接触率约 `0.15`，但 push 场景有 `12.5%` 跌倒；5578 的右移最差（最高 `18.75%` 跌倒），暂不推荐。
- 5577 的 `FR / median(other legs)` 速度 p95 仍约 `11–23`，说明需继续核对接触/动作映射，不能仅凭训练 rollout 判定右前腿问题已解决。

当前结论：5577 可作为下一轮训练和运动回归的候选，5575 可作为站立/接触对照；机械臂伸展指标和 actuator saturation 口径仍是 P0，长训脚本暂不应直接提交到集群。

尚未提交新的长训练。`scripts/cluster/continue_stability_e48f.sh` 是从 5577 的 `model_11999.pt` 继续 6000 次、目标到 18000 的草稿；它依赖外部生成的 `SHA256SUMS` 和已完成的评估 receipt，本次仅提交脚本供后续审核，不启动长训。不要直接提交原五路 stability sweep.

## 当前代码状态

基线提交是 `d393709 feat: stabilize quadruped stance and leg speed`。本 session 在其上有未提交的诊断、评估和续训辅助改动；不要把这些改动误认为已经提交到训练快照。

```text
d393709 feat: stabilize quadruped stance and leg speed
```

这个提交包含以下行为变化：

- command 默认删除 pose yaw，GO2-X5 使用 6 维 command：`[vx, vy, wz, height, roll, pitch]`。
- 默认观测布局为 `go2_x5_locomotion_v4_63`，policy 输入为 63 维。
- 训练中加入机械臂完全伸展样本，目标关节角为：
  `(0.0, 3.14159, 2.91688, 0.22471, 0.0, 0.0)` rad。
- 增加四脚接触奖励，减少站立时单腿悬空。
- 增加四腿关节速度均衡惩罚，抑制右前腿速度异常偏快。
- 增加四脚接触率、平均接触脚数、各腿关节速度和机械臂伸展诊断指标。
- `rl_sar` 的运行时 command 已支持 6 维格式；导出时要使用同一 observation contract。

本地验证结果：

- `python -m unittest scripts/tests/test_locomotion_robustness.py scripts/tests/test_robustness_cli.py`：16/16 通过。
- IsaacLab 2.3.0 headless smoke：8 env、30 steps、`domain_rand=none` 通过；另完成 USD FK 对照，最大误差约 `2.1e-6 m`。
- 训练使用 `WANDB_MODE=offline` 和 TensorBoard，没有 WandB 在线同步。

## 集群实验记录

远端共享目录：

```text
/home/simon/Projects/Locomotion/robot_lab
```

TensorBoard 根目录：

```text
/home/simon/Projects/Locomotion/robot_lab/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild
```

5 个 Slurm 任务都正常完成，退出码均为 `0:0`，每个任务训练到 `model_11999.pt`：

| Job | 节点 | 实验目录后缀 | 关键参数 | 失败率 | 四脚接触率 | 平均接触脚数 |
|---:|---|---|---|---:|---:|---:|
| 5574 | cn001 | `stability_contact2_balance08_ext25` | contact 2.0 / balance 0.08 / ext 25% | 10.38% | 0.389 | 3.15 |
| 5575 | cn001 | `stability_contact4_balance12_ext25` | contact 4.0 / balance 0.12 / ext 25% | 4.76% | **0.557** | **3.42** |
| 5576 | cn002 | `stability_contact2_balance08_slowarm` | 慢速机械臂伸展 | 8.00% | 0.382 | 3.15 |
| 5577 | cn002 | `stability_contact3_balance15_ext35` | contact 3.0 / balance 0.15 / ext 35% | **1.10%** | 0.485 | 3.32 |
| 5578 | pro6000-01 | `stability_contact3_balance10_ext35_hold` | ext 35% / 长保持 | 7.48% | 0.488 | 3.33 |

各模型完整路径为：

```text
/home/simon/Projects/Locomotion/robot_lab/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-18_18-25-41_stability_contact2_balance08_ext25/model_11999.pt
/home/simon/Projects/Locomotion/robot_lab/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-18_18-25-44_stability_contact4_balance12_ext25/model_11999.pt
/home/simon/Projects/Locomotion/robot_lab/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-18_18-25-43_stability_contact2_balance08_slowarm/model_11999.pt
/home/simon/Projects/Locomotion/robot_lab/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-18_18-25-41_stability_contact3_balance15_ext35/model_11999.pt
/home/simon/Projects/Locomotion/robot_lab/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-18_18-25-42_stability_contact3_balance10_ext35_hold/model_11999.pt
```

推荐的后续评估顺序：

1. `5577`：失败率最低，适合作为下一轮默认 locomotion 候选。
2. `5575`：四脚接触最好，适合作为站立稳定性候选。
3. `5578`：伸展保持时间更长，可用于机械臂负载场景对比。

右前腿速度问题在 5 个实验中没有继续表现为最高速度。5577 的四腿速度 MAE 为：
`FL 2.982 / FR 3.073 / RL 3.147 / RR 3.231 rad/s`；5575 为：
`FL 2.931 / FR 2.908 / RL 3.044 / RR 3.067 rad/s`。

## 当前指标的解释和限制

`arm_full_extension_case` 表示当前样本处于完全伸展 episode，`arm_full_extension_at_target` 表示实测六关节与最终目标的最大误差小于 `0.15 rad`。

本轮 12000 iterations 时，机械臂 curriculum 还没有走完：

```text
arm_start = 2000
arm_end   = 16000
```

所以训练结束时目标仍是从当前姿态向最终目标靠近的中间目标，而诊断指标比较的是最终目标。5 个实验的 `arm_full_extension_at_target` 为 0，不能直接说明机械臂一定无法到达；当前指标口径需要先修正为比较实际的 `extension_goal`，或者把训练延长到至少 16000 iterations 后再判断。

另外，当前接触指标是训练 rollout 的聚合统计，不是独立测试集，也不是实机数据。上真机前必须做固定种子、固定 command 的回放测试，并检查实际关节误差、足端力、关节力矩和跌倒事件。

## 下一轮优化优先级

### P0：先修机械臂伸展指标和目标跟踪

修改 `LocomotionRuntime.measure_and_check_failure()`：

- 用 `extension_goal` 计算 `arm_full_extension_at_target`；
- 单独记录 `extension_target_error_rad` 和末端前向距离误差；
- 记录伸展期间的 arm torque saturation、最大关节误差和到位时间；
- 增加一个固定的 `arm_full_extension_eval` 场景，避免只看训练随机样本。

建议先用 16000 或 18000 iterations 重跑一组，确认机械臂确实能到位，再做真机迁移。

### P1：独立评估站立和运动

对 5575、5577、5578 做相同评估：

- 零速度、机械臂完全伸展，连续站立 10 s；
- `vx=0.3/0.6 m/s` 前进和减速；
- 左右横移；
- 随机 push；
- 记录每条腿的关节速度峰值、速度比 `FR / median(other legs)`、接触丢失次数和跌倒率。

选择标准优先使用跌倒率和接触丢失次数，再看平均 reward。当前仅从训练统计看，5577 适合运动，5575 适合站立。

### P2：确认硬件相关参数

真机复现时重点核对：

- 右前腿的 joint order 是否仍为 `FR_hip, FR_thigh, FR_calf`；
- 真实 actuator 的 stiffness、damping、command scale 和延迟；
- 四条腿的零位、编码器方向和关节限位；
- 机械臂伸展时的实际 payload 和质心偏移。

如果仿真中 FR 速度已均衡而真机仍偏快，优先检查 action 映射、PD 参数和编码器标定，而不是继续提高速度均衡 reward。

### P3：导出和真机前检查

导出前确认 observation contract 为 `go2_x5_locomotion_v4_63`，command 为 6 维；运行时使用同一版本的 `rl_sar`。至少做一次：

```text
policy input width = 63
command width      = 6
action width       = 12
```

然后回放 5577 和 5575，优先选一套做低速真机测试，机械臂伸展保持时间从 1–2 s 开始。

## 常用命令

查看训练曲线：

```bash
tensorboard --logdir /home/simon/Projects/Locomotion/robot_lab/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild --port 6006
```

查看 Slurm 状态：

```bash
ssh login01.e48f.corelab.wan
squeue -u $USER
sacct -j 5574,5575,5576,5577,5578 --format=JobID,State,Elapsed,NodeList,ExitCode -X
```

训练脚本：

```text
scripts/cluster/train_stability_e48f.sh
```

下一轮代码修改完成后，先在本地运行 14 个 CPU 回归测试和 IsaacLab headless smoke，再提交短的固定场景评估，最后才提交长训练。
