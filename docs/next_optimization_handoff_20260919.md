# GO2-X5 locomotion optimization handoff

更新时间：2026-09-19  
任务：`RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0`

## 当前代码状态

当前仓库没有未提交改动。用于本轮训练的提交是：

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

- `python -m unittest scripts/tests/test_locomotion_robustness.py scripts/tests/test_robustness_cli.py`：14/14 通过。
- IsaacLab headless smoke：8 env、30 steps、`domain_rand=none` 通过。
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
