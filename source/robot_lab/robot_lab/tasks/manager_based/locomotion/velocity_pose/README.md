# VelocityPose on Isaac Lab 3.0

Go2-X5 tracks six base commands:
`[vx, vy, yaw_rate, height, roll, pitch]`. Go2-X5 uses 63 policy/critic
observations and 12 policy leg actions, with six arm joints controlled separately
by bounded arm motion or position-only IK. Default height is 0.33 m.
Go2 retains its seven-command layout, including the legacy pose-yaw column.

## Registered tasks

| Robot | IDs (replace `{variant}`) | Backend |
|---|---|---|
| Go2 | `RobotLab-Isaac-VelocityPose-{Flat,Rough}-Unitree-Go2-v0` | PhysX |
| Go2-X5 | `RobotLab-Isaac-VelocityPose-{Flat,Rough,Mild,IK}-Unitree-Go2-X5-v0` | PhysX / Newton MJWarp |

PhysX is the default. X5 supports the legacy typed selector `physics=newton`;
use `presets=newton_mjwarp` with the current Lab preset spelling. Ordinary Go2
VelocityPose tasks do not declare Newton presets.

## Train and play

Run from the repository root with Isaac Lab 3.0 and RSL-RL 5.5.1 Python.

```bash
PYTHON_BIN=${ISAACLAB_PYTHON:-python}
TASK=RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0
ROBOT_LAB_INVALID_WORLD_WINDOW_RATE_LIMIT=10 "$PYTHON_BIN" \
  scripts/reinforcement_learning/rsl_rl/train.py --task "$TASK" --viz none \
  --num_envs 4096 --seed 42 --max_iterations 25000 presets=newton_mjwarp \
  agent.clip_actions=10 'env.actions.joint_pos.clip={".*": [-10.0, 10.0]}' \
  env.rewards.joint_pos_limits.weight=-10

"$PYTHON_BIN" scripts/reinforcement_learning/rsl_rl/play.py --task "$TASK" \
  --checkpoint "$CHECKPOINT" --num_envs 8 --viz kit presets=newton_mjwarp \
  agent.clip_actions=10 'env.actions.joint_pos.clip={".*": [-10.0, 10.0]}' \
  env.rewards.joint_pos_limits.weight=-10
```

The final `stageg_clip10` recipe uses **±10 action clipping** and
`joint_pos_limits` reward weight −10. These are explicit run overrides; task
defaults remain unchanged. Clip 3 showed action-std blow-up in the long run.
Use the checkpoint's saved recipe when playing or comparing it. X5 play restores
continuous curriculum progress; `--domain_rand none|benchmark|sim2real` selects
randomization and `--arm_mode random|structured|hold` selects arm motion.

Newton recovery only masks attributed invalid worlds, preserves healthy rows
and isolates invalid-reset RNG consumption. The window-gate env var above stops
training when a disjoint 100-update window after update 500 exceeds 10 resets per
million environment-steps. Raw-action magnitude>3 audit logging stays enabled.

## Export and verify

```bash
"$PYTHON_BIN" scripts/reinforcement_learning/rsl_rl/export_rl_sar.py \
  --checkpoint "$CHECKPOINT" --rl-sar-root "$RL_SAR_ROOT" \
  --config-name stageg_clip10 --output-dir "$BUNDLE"
"$PYTHON_BIN" scripts/tests/smoke_velocity_pose.py --task "$TASK" \
  --num-envs 8 --steps 30 --preset newton_mjwarp --viz none --output "$SMOKE_OUT"
"$PYTHON_BIN" -m unittest discover -s scripts/tests -p 'test_*.py' -v
```

Exporter bounds come from saved `params/env.yaml` and `agent.yaml`; inconsistent
clips are rejected. A compatible rl_sar SDK must support the six-wide velocity
pose command. Keep new bundle/output paths and verify the deployment controller
contract before hardware use.

MDP terms live in `mdp/shared`, `mdp/low_level` and `mdp/high_level`; native IK
uses base-frame link-origin poses/Jacobians and arm gravity feedforward. The
ordinary locomotion task has no arm gravity feedforward.
See [the final port report](../../../../../../../docs/vel_pose_v2_isaaclab3_port.md)
for migration evidence and limitations.

## Curriculum and robustness

Go2 uses the reward/command curriculum in `command_curriculum_height_pose`.
Its stages progressively introduce height, orientation and broader velocity ranges;
the curriculum also changes penalty weights, so zero initialization is not proof
that a reward is dead. X5 uses continuous schedules measured in PPO updates
(policy steps divided by `steps_per_iteration`, normally 24):

| Schedule | Start | Full intensity |
|---|---:|---:|
| Push | 1000 | 8000 |
| Height / roll / pitch | 1000 | 6000 |
| Hard resets | 2000 | 8000 |
| Arm motion / payload | 2000 | 16000 |

The fixed hard cohort is 20%, selected with cohort seed 1234 independently of
the rollout RNG. Default easy/hard reset limits are 10/45 degrees. Random arm
acceleration is bounded by the configured joint limits, velocity and acceleration;
25% of episodes use reach/hold/retract/hold. IK uses these reachable trajectory
targets through native position IK. Resume restores the policy-step schedule.
`sim2real` includes observation corruption, randomized gains and actuator delays;
`benchmark` disables those; `none` also disables material/mass/COM/push DR.
Metrics retain all/easy/hard and all/early/late denominators in
`locomotion_metrics.jsonl`. Recovery grace and failure height/tilt belong to
`RobustnessCfg`; numerical Newton recovery is a separate mechanism.

```bash
"$PYTHON_BIN" scripts/sim2sim/evaluate_go2_x5_mujoco.py --xml "$MJCF" \
  --isaac-packets "$PACKET" --bundle "$BUNDLE" --replicas 16 --output "$EVAL_OUT"
# Optional --arm-recipe "$ARM_RECIPE" uses saved soft limits and generator kwargs.
"$PYTHON_BIN" scripts/monitoring/monitor_newton_training.py --events "$EVENT_DIR" \
  --state "$MONITOR_STATE" --num-envs 4096
```

MuJoCo evaluation first requires identical-state observation packets to pass
1e-4 absolute error, then reports seeded first-episode falls and post-4s tracking.
Use fresh output directories and one monitor state per event directory/protocol.
