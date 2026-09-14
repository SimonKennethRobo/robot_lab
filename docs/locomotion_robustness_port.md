# GO2 + X5 locomotion：RoboDuet 首批移植

来源是 `/home/simon/Projects/WBC/RoboDuet` 的 `v3-stage2`，提交 `97397ad`。
移植时读取该提交的快照，没有使用或修改 RoboDuet 工作区里的未提交改动。
目标分支的原始提交为 `3a87de9`。

## 任务与观测契约

- 平地：`RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0`
- **首批推荐配方**：`RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0`
- 原有复杂地形：`RobotLab-Isaac-VelocityPose-Rough-Unitree-Go2-X5-v0`

策略和 critic 均为 **64D**，动作是 12D 腿关节目标偏移。
观测依次为：机身线速度 3、角速度 3、重力方向 3、命令 7、上一动作 12、
腿关节位置 12、腿关节速度 12、机械臂关节位置 6、机械臂关节速度 6。
腿关节的观测与动作顺序统一为 **FR、FL、RR、RL**，每腿 hip、thigh、calf。
删除 14D 零占位、世界系末端相对位置和按假定质量计算的 COM 特征；机械臂速度缩放为 0.1。
actor 不读取地形真值或随机化质量。机身线速度在部署时仍需状态估计器提供。

命令为 `[vx, vy, wz, height, roll, pitch, yaw]`；XY 速度使用随机身 yaw 转动的水平坐标系，
角速度使用世界竖直轴，height 是相对当地地面的根链接高度。yaw 姿态命令保持 0。
线速度沿用 Isaac Lab 根刚体 COM 速度定义；actor 中以机身坐标表达，跟踪时转到 yaw 对齐坐标。
EE 扰动指标使用根链接原点速度和真实 EE 杠杆臂，避免把 COM 速度当作链接原点速度。
地形射线只服务于训练奖励、终止与诊断；一个环境的无效射线不会影响其它环境。

checkpoint 内保存观测契约、关节顺序、配方与实际 policy-step 计数。
**旧 84D checkpoint 不能直接恢复**，加载器会明确拒绝；此变更需要重新训练，未做权重迁移。
课程恢复使用已执行的 policy steps，不乘环境数，也不依赖 RSL-RL 滞后一拍的迭代变量。
恢复课程进度不等于逐位恢复仿真状态和 RNG。

## 连续课程

默认每迭代 24 policy steps，每步 20 ms。下表为 PPO 迭代数，和环境数量无关。
范围可以通过 `env.robustness.*` 的 Hydra override 修改。
训练和播放入口会把 robustness 的整数 float override 自动转换为 float，例如
`env.robustness.hard_fraction=0` 自动成为 `0.0`；其余 IsaacLab 任务字段仍保持原有严格类型检查。

| 项目 | 起点 → 满强度 | 满强度配置 |
|---|---:|---|
| reset 分组 | 2,000 → 8,000 | 固定约 80% easy / 20% hard；easy roll/pitch ±10°，hard 从 ±10° 增至 ±45° |
| 机械臂 | 1,000 → 8,000 | random / structured / hold = 70% / 20% / 10%；目标速度 ≤2.5 rad/s，加速度 ≤8 rad/s² |
| 推扰 | 1,000 → 8,000 | 每 4–8 s，XY 速度增量每轴 ±0.6 m/s，三轴角速度增量每轴 ±0.6 rad/s |
| 末端载荷 | 跟随机械臂课程 | 每次 reset 采样 0–1.5 kg，EE 局部偏心 ±(0.10, 0.05, 0.05) m |
| 高度 / 姿态命令 | 1,000 → 6,000 | 高度 0.28–0.38 m，roll ±15°，pitch ±10° |

固定 reset 分组用独立 CPU RNG 生成，不消耗仿真随机数，不随 episode 重分组。
reset 时腿关节偏移 ±0.05 rad，机械臂初始偏移随课程增至 ±0.25 rad，并受关节工作区限制。
机械臂生成器从真实 reset 关节位置开始，轨迹限位前提前减速；六个臂关节与夹爪分开处理。
左右夹爪均由同一组 PD actuator 保持默认开度，策略并不输出夹爪动作。
目标运动边界由测试验证，实际速度、加速度与力矩饱和率另行记录。

载荷通过每个 physics step 更新的世界系重力和偏心力矩实现，施力位置以真实 EE 链接姿态计算，
并修正链接原点与 COM 的差异。**这是重力 wrench 近似，不增加载荷惯性，也不模拟抓取接触。**

轻度地形按列固定分配：50% 平面、25% ±2 cm、25% ±4 cm 连续起伏；水平采样间隔 0.4 m。
对三次插值的过冲再次限幅。这里 2/4 cm 是相对零平面的最大正负高度，不是峰峰值。
没有台阶、陡坡或地形等级升级；保持同一组小网格射线作为高度参考。

## Domain randomization 与奖励精简

| 模式 | 内容 |
|---|---|
| `none` | 关闭物理参数 DR、推扰、额外载荷、观测噪声、动作延迟；保留任务命令、reset 课程和机械臂任务运动 |
| `benchmark` | 摩擦、质量、COM、载荷和推扰；关闭观测噪声、PD 随机化与动作延迟 |
| `sim2real`（训练默认） | benchmark 内容加观测噪声、分腿/臂 PD 范围、0–2 physics ticks 动作延迟 |

摩擦范围 static 0.5–1.25、dynamic 0.4–1.0，并保证 dynamic ≤ static；恢复系数 0–0.1。
base 质量增加 -1～1 kg，其他链接质量乘 0.85–1.15，质量改变同步更新惯性。
base COM 各轴 ±2 cm。腿 Kp/Kd 乘 0.8–1.2；臂 Kp 乘 0.8–1.2、Kd 乘 0.7–1.3。
PD 随机化写入实际使用的显式 actuator，不只修改仿真中未使用的隐式 PD 参数。

修复或移除的内容：

- 运动/静止判断只检查前三维速度命令；静止触地奖励持续奖励触地，不奖励反复抬落脚。
- 脚滑使用触地脚相对静止地面的世界系 XY 速度，不减去运动机身的速度。
- 常量非零高度、姿态范围能正确采样；移除重复命令采样实现。
- 使用重力方向比较目标倾角，避免 roll/pitch 耦合时的小角度近似；命令指标与奖励使用相同坐标系。
- 腿策略只承担腿关节正则项；合并站立姿态惩罚，去掉镜像、air-time 方差、固定机身系脚高目标。
- 静止 yaw 惩罚有界，最多贡献 -0.5；触地掩码只看脚，移除隐藏的角速度积分状态。
- 取消持续整 episode 的随机 base wrench、20k 静臂阶段、九模式状态机和阶段奖励跳变。
- 持续低于 0.12 m 或倾角超过 75° 达 2 s 后重置，允许短暂恢复；可用
  `env.robustness.terminate_unrecovered=false` 保留整 episode 恢复尝试。
- 默认关闭命令可视化，显式开启时仅显示前 8 个环境；删除 play 的重复可视化循环。
- 合并重复 IK 文件为兼容导入，训练不加载它们。实验 IK 任务含占位 Jacobian，现明确拒绝启动，
  避免把它误用成可训练配方；未实施 Stage 2 IK。

默认预算 20,000 iterations，满强度后仍有 12,000 iterations；按跟踪误差和失败率决定是否续训。
保留网络 512/256/128、PPO 每次更新 5 epochs。传感延迟、历史/adaptation、mount 几何随机化、
复杂地形、真实新增惯性和完整 IK 不在这批移植中。

## 运行与验证

环境准备（不安装项目、不修改 IsaacLab）：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab230
export PYTHONPATH="$PWD/source/robot_lab:$PYTHONPATH"
python scripts/tests/test_locomotion_robustness.py
python -u scripts/tests/smoke_locomotion_robustness.py --headless --device cuda:0 \
  --steps 144 --ppo_iterations 2 --output outputs/locomotion_validation/mild_sim2real
```

正式训练入口（本次工作不启动正式训练）：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0 --headless \
  --num_envs 4096 --logger tensorboard
```

播放新 checkpoint 时默认 `--domain_rand benchmark`，恢复 checkpoint 的课程进度；
可用 `--robustness_iteration 8000 --arm_mode structured` 指定固定挑战等级。
旧的 `--curriculum_stage` 只用于普通 GO2 旧配方；GO2-X5 使用连续课程。

`locomotion_metrics.jsonl` 每个 rollout 写入真实样本数，以及 all/easy/hard × all/early/late
的速度、高度、倾角误差、脚滑、臂速度/加速度、力矩饱和和机身诱发 EE 世界系 z 速度。
early 指 reset 后实际 2 s 内，独立于 PPO 随机 episode-length 初始化；统计在自动 reset 之前采样。
reset 后首帧不计入臂加速度分母。失败比例以完成的 episode 为分母，计数 0 时不要解释为成功率。
无效扫描有单独计数；非有限动作/状态/观测/奖励触发异常并保存 fault dump，停止污染 PPO。

小规模 smoke 的通过只证明接口、数值和更新流程可运行；没有证明收敛收益、最终 locomotion
性能或实机效果。`expected_fault/` 是测试主动注入 NaN 的诊断产物，不是物理仿真故障。

下一轮 yaw 响应与 base-z 稳定性实验从
[`next_experiment_handoff_20260914.md`](next_experiment_handoff_20260914.md) 开始，先完成固定输入复现和
数值保护，再按交接中的条件矩阵提交训练。
