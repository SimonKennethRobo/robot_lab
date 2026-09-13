# 坐标系更新说明 - VelocityPose 任务扩展到 7D 指令

## 日期
2026-01-15

## 概述
将 VelocityPose 任务从 6D 指令扩展到 7D 指令，添加了 yaw 姿态控制，并明确定义了三个坐标系的关系。

## 三个坐标系定义

### 1. 世界坐标系 A (World Frame A)
- **特征**: 固定不动的全局参考系
- **Z 轴**: 竖直向上（重力方向）
- **XY 平面**: 固定方向（例如：X指向东，Y指向北）
- **用途**: 全局定位参考

### 2. 机器人质点坐标系 B (Robot Point Frame B / Yaw-Aligned Frame)
- **特征**: Z 轴始终竖直向上（与世界坐标系 A 的 Z 轴平行）
- **XY 平面**: 跟随机器人运动朝向旋转
- **物理意义**: 将机器人视为质点，只考虑其在水平面的运动方向
- **旋转自由度**: 只绕 Z 轴旋转（yaw 运动）
- **用途**: 
  - 定义运动指令（lin_vel_x, lin_vel_y, ang_vel_z）
  - 作为姿态控制的参考系

### 3. 机器狗 Base 坐标系 C (Robot Base Frame C / Body Frame)
- **特征**: 完全跟随机器人 base link 的姿态
- **全部自由度**: 包括 roll, pitch, yaw 三个旋转
- **用途**: 机器人的实际姿态

### 坐标系关系
```
World Frame A (固定)
     ↓ (ang_vel_z 旋转)
Point Frame B (Z轴竖直，XY跟随运动方向)
     ↓ (roll, pitch, yaw 旋转)
Base Frame C (完全跟随机器人姿态)
```

## 指令结构变化

### 之前 (6D)
```python
[lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch]
```

### 现在 (7D)
```python
[lin_vel_x, lin_vel_y, ang_vel_z, height, roll, pitch, yaw]
```

### 指令含义

#### 运动指令（在质点坐标系 B 中定义）
1. **lin_vel_x** (m/s): 质点坐标系 B 的 X 方向线速度（前进/后退）
2. **lin_vel_y** (m/s): 质点坐标系 B 的 Y 方向线速度（左移/右移）
3. **ang_vel_z** (rad/s): 绕质点坐标系 B 的 Z 轴角速度（竖直轴）
   - **作用**: 改变机器人运动朝向（旋转 Point Frame B 相对于 World Frame A）
   - **不改变**: Base 相对于 Point Frame B 的姿态

#### 姿态指令（Base Frame C 相对于 Point Frame B）
4. **height** (m): 机器人 base 的目标高度
5. **roll** (rad): Base Frame C 相对于 Point Frame B 的 roll 角
6. **pitch** (rad): Base Frame C 相对于 Point Frame B 的 pitch 角
7. **yaw** (rad): Base Frame C 相对于 Point Frame B 的 yaw 角

#### 重要说明
- 当 `[roll=0, pitch=0, yaw=0]` 时，Base Frame C 和 Point Frame B 完全对齐
- `yaw` 姿态指令改变的是 **Base 姿态**，不改变 **运动朝向**
- `ang_vel_z` 运动指令改变的是 **运动朝向**（Point Frame B 相对于 World Frame A），间接影响 Base 的世界朝向

## 代码修改清单

### 1. 指令生成器 (commands.py)
**文件**: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp/commands.py`

**修改内容**:
- ✅ `UniformVelocityPoseCommand.pose_command`: 从 `(num_envs, 2)` 改为 `(num_envs, 3)`
- ✅ `command` 属性: 从 6D 改为 7D
- ✅ `_resample_command`: 添加 yaw 采样逻辑
- ✅ `UniformVelocityPoseCommandCfg.Ranges`: 添加 `yaw: tuple[float, float]`
- ✅ 更新文档字符串，详细说明三个坐标系

### 2. 奖励函数 (rewards.py)
**文件**: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp/rewards.py`

**修改内容**:
- ✅ `track_orientation_exp`: 
  - 从 `command[:, 4:6]` 改为 `command[:, 4:7]`
  - 添加 `target_yaw = command[:, 6]`
  - 更新四元数计算以包含 yaw
  - 更新调试输出
- ✅ `stand_still_full_cmd`: `pose_cmd = command[:, 4:7]`
- ✅ `joint_pos_penalty_full_cmd`: `pose_cmd = command[:, 4:7]`
- ✅ `is_moving_full_cmd`: `pose_cmd_norm = torch.norm(command[:, 4:7], dim=1)`

### 3. 观测函数 (observations.py)
**文件**: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/mdp/observations.py`

**修改内容**:
- ✅ `base_orientation_command`: 
  - 返回值从 `(num_envs, 2)` 改为 `(num_envs, 3)`
  - `return command[:, 4:7]`  # 包含 roll, pitch, yaw

### 4. 环境配置 (velocity_pose_env_cfg.py)
**文件**: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/velocity_pose_env_cfg.py`

**修改内容**:
- ✅ 修改 `track_ang_vel_z_exp` 使用**世界坐标系 Z 轴**跟踪
  - **原因**: `ang_vel_z` 在质点坐标系 B (Yaw-Aligned Frame) 中定义
  - 质点坐标系 B 的 Z 轴 = 世界坐标系 Z 轴（都是竖直向上）
  - 使用 `root_ang_vel_w[:, 2]` 可以**严格精确**地跟踪质点朝向变化
  - **对比**: 使用 `root_ang_vel_b[:, 2]` 在机器人倾斜时会有误差
- ✅ 实现方式: 
  ```python
  # 覆盖奖励函数为世界坐标系版本
  from robot_lab.tasks.manager_based.locomotion.velocity.mdp import rewards as velocity_rewards
  self.rewards.track_ang_vel_z_exp.func = velocity_rewards.track_ang_vel_z_world_exp
  ```

## 物理意义示例

### 示例 1: 原地转圈 + 向左倾斜
```python
指令: [vx=0, vy=0, ωz=1.0, h=0.35, roll=0.3, pitch=0, yaw=0]

效果:
- 机器人原地不动（vx=vy=0）
- 机器人绕自己的竖直轴以 1.0 rad/s 旋转（ωz=1.0）
  → Point Frame B 相对于 World Frame A 旋转
  → 机器人在原地转圈
- 机器人向左倾斜 0.3 rad（roll=0.3）
  → Base Frame C 相对于 Point Frame B 倾斜
- Base 的 yaw 跟随运动朝向旋转（由于 pose yaw=0，Base 始终对齐 Point Frame B）
```

### 示例 2: 前进 + 机头向左偏转
```python
指令: [vx=1.0, vy=0, ωz=0, h=0.35, roll=0, pitch=0, yaw=0.5]

效果:
- 机器人沿质点坐标系 B 的 X 轴以 1.0 m/s 前进（vx=1.0）
  → 沿着当前运动方向移动
- 运动朝向不变（ωz=0）
  → Point Frame B 保持方向
- 机器人机头向左偏转 0.5 rad（yaw=0.5）
  → Base Frame C 相对于 Point Frame B 有 yaw 偏差
  → 注意：这不改变运动方向！机器人仍然沿着 Point Frame B 的 X 轴前进
  → 视觉效果：机器人"侧身"前进，机头指向左前方
```

### 示例 3: 对比 ang_vel_z vs yaw 姿态
```python
情况 A - 使用 ang_vel_z 改变运动朝向:
指令: [vx=1.0, vy=0, ωz=0.5, h=0.35, roll=0, pitch=0, yaw=0]
→ 机器人沿圆弧路径移动（Point Frame B 旋转）
→ Base Frame C 始终对齐 Point Frame B（因为 yaw=0）
→ 机器人朝向和运动方向一致，沿圆弧转弯

情况 B - 使用 yaw 姿态产生侧身效果:
指令: [vx=1.0, vy=0, ωz=0, h=0.35, roll=0, pitch=0, yaw=0.5]
→ 机器人沿直线移动（Point Frame B 不旋转）
→ Base Frame C 相对于运动方向偏转 0.5 rad（yaw=0.5）
→ 机器人朝向和运动方向不一致，产生"侧滑/漂移"效果
```

## 训练注意事项

### Curriculum Learning
由于添加了新维度，建议分阶段训练：

#### Stage 1 (0-5K iterations)
```python
ranges.yaw = (0.0, 0.0)  # yaw 保持 0
```

#### Stage 2 (5K-15K iterations)  
```python
ranges.yaw = (-0.2, 0.2)  # ±11.5°
```

#### Stage 3 (15K-25K iterations)
```python
ranges.yaw = (-0.5, 0.5)  # ±28.6°
```

### 奖励权重建议
```python
track_orientation_exp.weight = 2.0  # 包含 yaw 后可能需要调整
track_ang_vel_z_exp.weight = 1.5    # 世界坐标系跟踪
```

## 验证清单

- [x] 指令生成器支持 7D 指令
- [x] 奖励函数正确读取 7D 指令
- [x] 观测函数返回 3D 姿态指令
- [x] 角速度使用世界坐标系跟踪
- [x] 姿态跟踪在 yaw-aligned frame 中计算
- [ ] 测试新指令在实际环境中的效果
- [ ] 验证 yaw 姿态控制不影响运动朝向
- [ ] 验证 ang_vel_z 改变运动朝向但不直接改变 base yaw

## 后续工作

1. **测试运行**: 使用少量环境测试新指令结构
2. **可视化**: 添加调试可视化，显示三个坐标系
3. **Curriculum**: 设计 yaw 姿态的课程学习策略
4. **文档**: 更新用户文档，说明 7D 指令的使用方法

## 参考
- 修改日期: 2026-01-15
- 修改者: GitHub Copilot
- 相关 Issue/PR: N/A
