# GO2-X5 SOTA continuation plan (2026-09-19)

## Objective

Continue from the real-robot-tested `gait_W30_formal_s42_1000u/model_10698.pt` and select one overall policy using velocity, yaw, height and pose tracking, disturbance robustness, foot width, leg-speed balance and gait symmetry. The policy contract for this run is `go2_x5_locomotion_v4_63`; all starts are weight-only migrations from the W30 `go2_x5_locomotion_v2_64` checkpoint.

Parent SHA256:

```text
909c731e5079290a6e3139edf7b06a96dcf146e4d038f4958bbf7bbe44c975eb
```

Each run adds 24,000 PPO updates and targets iteration `34,698`. The run uses `iteration_override=-1`, `sim2real` domain randomization, height range `(0.28, 0.38)`, and pose curriculum driven by the training clock. The yaw-aware leg velocity balance gate is enabled for `[vx, vy, wz]`.

## Recipes

| branch | standing contact | leg speed balance | stance reward | extra gait terms | seeds |
|---|---:|---:|---:|---|---|
| main | 3.0 | 0.15 | all 0.50 + rear 1.00 | none | 42, 73 |
| contact | 4.0 | 0.12 | all 0.50 + rear 1.00 | none | 42, 73 |
| nominal | 2.0 | 0.08 | all 1.00 + rear 1.00 | none | 42, 73 |
| gait | 3.0 | 0.18 | all 0.50 + rear 1.00 | timing 0.25, swing-height 0.50, contact-sync 0.75 | 42, 73 |

The gait branch is a controlled exploration because the earlier timing and height-only continuations improved individual symmetry metrics while regressing the combined tracking score. It will only be promoted if the fixed evaluation confirms an overall gain.

## Active placement

The original TensorBoard-only launch was stopped on 2026-09-19 and restarted with WandB online under project `robotlab`. The active E4A tag is `sota_20260919_e4a_wandb`; the E4 fallback tag is `sota_20260919_e4_fallback_wandb`. The current effective run names and links are recorded in [next_optimization_handoff_20260919.md](next_optimization_handoff_20260919.md).


| run | device | envs | log location | status at restart |
|---|---|---:|---|---|
| main-s42 | E4A RTX 5090 | 4096 | `robot_lab_sota_20260919/cluster_runs/sota_20260919_e4a_wandb/logs/sota_20260919_e4a_wandb_main_s42.log` | PPO running with WandB online |
| main-s73 | E4A RTX 4080S-0 | 2048 | same E4A log directory | PPO running with WandB online |
| contact-s42 | E4A RTX 4080S-1 | 2048 | same E4A log directory | PPO running with WandB online |
| contact-s73 | E4A RTX 5080-1 | 2048 | same E4A log directory | PPO running with WandB online |
| nominal-s42 | E4A RTX 5080-2 | 2048 | `.../sota_20260919_e4a_wandb_nominal_s42_retry4.log` | PPO running with WandB online |
| nominal-s73 | E4 RTX 4090-0 (`cn002`) | 2048 | `cluster_runs/sota_20260919_e4_fallback_wandb/logs/nominal_s73.log` | PPO running with WandB online |
| gait-s42 | E4 RTX 4090-1 (`cn002`) | 2048 | `cluster_runs/sota_20260919_e4_fallback_wandb/logs/gait_s42.log` | PPO running with WandB online |
| gait-s73 | E4 RTX Pro6000 | 4096 | `cluster_runs/sota_20260919_e4_fallback_wandb/logs/gait_s73.log` | PPO running with WandB online |

The three E4A RTX 3080 nodes had pre-existing RoboDuet jobs using GPU memory. The initial 1024-env starts on those nodes hit PhysX out-of-memory and were stopped; no checkpoint from those attempts is used.

## Acceptance and selection

At checkpoints around 20k, 28k and 34,698, run the same deterministic evaluation for every branch and both seeds: stand, forward `0.3/0.6`, lateral left/right, yaw-only, forward+yaw, height commands `0.28/0.33/0.38`, arm full-extension hold, and push disturbances. Record velocity/yaw/height/tilt MAE or RMSE, failure and recovery fractions, foot width, four-foot contact, contact losses, per-leg speed p95 and `FR / median(other legs)`, swing-height and air-time spreads.

Promotion is based on a normalized multi-metric score with hard gates: no systematic yaw-only failure, no single-leg speed spike, no height command collapse, no regression in W30 forward/lateral tracking, and no unacceptable push failure rate. Training reward alone does not select the SOTA model.

## Reproducibility

The E4A launcher is [launch_sota_e4a_20260919.sh](../scripts/cluster/launch_sota_e4a_20260919.sh). The E4A submission record is stored at `/home/simon-nfs/Projects/Locomotion/robot_lab_sota_20260919/cluster_runs/sota_20260919_e4a/submissions.tsv`; failed-resource and failed-import records are kept alongside it. The E4 fallback processes use the same parent hash and overrides.

Local regression tests remain `16/16` passing. A 5090 one-update warm-start preflight completed and produced `training_success.json` with the migrated 63D contract.
