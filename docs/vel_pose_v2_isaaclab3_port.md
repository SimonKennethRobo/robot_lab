# VelocityPose: Isaac Lab 3.0 / Newton port

Final report, 2026-10-07. The port is basically successful. The seed-42
`stageg_clip10` policy completed 25,000 Newton updates and passed the user's
MuJoCo and rl_sar teleop review. This establishes the tested simulation recipe,
not hardware readiness or backend equivalence.

The full lab notebook, rejected probes and original receipts remain in history:
`git show 2f7826e:docs/vel_pose_v2_isaaclab3_port.md`.
The final training recipe was authored at `1e99cb5`; cleanup changes no task config.

## Condensed API migration

Paths below are relative to `source/robot_lab/robot_lab/`.

| Area | Isaac Lab 3.0 implementation |
|---|---|
| Registry / MDP exports | Lazy imports with explicit `.pyi` exports; defer USD import until simulator selection |
| Launch / CLI | `launch_simulation`, preset resolution, native video recorders, `--viz none` for headless |
| Actuators | Separate actuator effort/velocity limits from solver limits; `resolve_joint_parameter`, `_clip_effort(effort, joint_vel)` |
| State / events | `root_view`, indexed write APIs, ProxyArray `.torch`, explicit unsupported-capability errors |
| Composite action | `target_command.set_position_index(value=...)`; unchanged leg processing and separate gripper |
| Noise / scanners | `UniformNoiseCfg`; Go2 scanner path `/Robot/Geometry/base` |
| IK | Built-in position-only DLS; native link-origin pose and Jacobian in base frame |
| RSL config | Separate `RslRlMLPModelCfg` actor/critic, observation groups, Gaussian `distribution_cfg` |
| Runner | Native `PPO.actor` / `PPO.critic`, logger/save/load and `load_cfg` |

### Quaternions

Lab 3 state and math use **XYZW**. USD imaginary XYZ plus real W is converted at
that boundary. MuJoCo uses **WXYZ**, so deployment replays explicitly reorder
`[x,y,z,w]` to `[w,x,y,z]`. Yaw extraction and frame rotations follow the native
Lab convention. CPU tests check 256 seeded quaternions (rotated-basis max error
0) and 256 seeded FK samples (position max error `7.47e-8 m`). Native Jacobian
finite-difference smoke validation uses link-origin, base-frame quantities.

### RSL-RL 5.5.1 checkpoints

| Legacy combined state | Native separate state |
|---|---|
| `actor.N.weight/bias` | `actor_state_dict.mlp.N.weight/bias` |
| `critic.N.weight/bias` | `critic_state_dict.mlp.N.weight/bias` |
| `std` / `log_std` | `distribution.std_param` / `distribution.log_std_param` |

The runner strictly rejects unknown, missing, nonfinite and wrong-shaped tensors.
Legacy combined optimizer resume is rejected; native optimizer resume is supported. Official `handle_deprecated_rsl_rl_cfg`
removes inherited deprecated config fields at construction boundaries.
For 1024×63 inputs, actual old/new runtime actor FP32 max difference was
`0.0009765625` (approved tolerance `1e-3`); FP64 max difference was `1.819e-12`.
Mapping within the same Torch runtime was exact. These checks establish mapping,
not closed-loop behavior.

## Decisions retained

| Decision | Rationale and evidence |
|---|---|
| Legacy TGS external-force iteration setting = false | Lab 3's changed default caused a major first-step discrepancy; false restores legacy semantics, though deprecated upstream |
| Lab PD path | Retain sampled explicit leg PD and delays; bypass probes isolated remaining self-collision effects |
| 19-body contacts | X5 has 28 physical bodies but 19 sensed bodies; preserve measured contact/public ordering |
| Newton implicit arm drive | Default `kd·dt/I` is 2.090 and 10.899 on joints 5/6, outside sampled explicit stability bounds; implicit drive retains original gains, 20 Nm limit and delay |
| Newton invalid-world recovery | Initial explicit-arm run failed at update 134; row-local recovery masks only attributed invalid worlds before reward accumulation and resets them with RNG isolation |
| Native base-frame IK + gravity feedforward | Preserve position-only DLS, lambda 0.05, 0.2 rad correction cap and soft limits; gravity load explains the PD hold residual; feedforward is confined to IK |
| Clip ±10 and joint-position-limit reward −10 | Clip-3 long run showed action-std blow-up; final clip10 recipe completed 25k. This recipe comparison does not isolate either change causally |

Recovery leaves PhysX unchanged and raises for unattributed nonfinite values.
`ROBOT_LAB_INVALID_WORLD_WINDOW_RATE_LIMIT=10` stops training if a **disjoint
100-update window after update 500** exceeds 10 invalid resets per million
aggregate environment-steps. Denominator is `100 × num_envs × rollout_steps`.
The monitor additionally reports rolling 100-update breaches. Set the env var in
the training process; monitoring does not itself stop training.
`Audit/raw_action_abs_gt3_fraction` remains the fraction of sampled raw scalar
policy actions above magnitude 3 before clipping, with an explicit denominator.
It is an audit threshold even when execution clip is 10.

USD composition and SHA256 are pinned in `velocity/newton_ordering_profiles.json`
(X-006). `go2_x5.usd` resolves to v3, whose payloads use go2 and x5 USDs. All pinned
v1/v2/v3 layers are retained, including apparently unused versions.

## Validation and boundaries

These are historical research results from the notebook at `2f7826e`, unless
explicitly labeled final training data. Evaluations retain their own denominators.

### Stage B behavior gate

Same legacy policy on the 2.3 and 3.0 PhysX plants: shared reset arrays and
observation ordering agree; quaternion and DR checks pass. The initial strict
gate passed 6/12 cells, with five contact-rate cells and one fall-count cell
failing. The user subsequently accepted Stage B with absolute contact-rate
difference tolerance 0.01. This is an accepted migration gate, not trajectory
identity or PhysX/Newton equivalence.

### Stage C: paired 2000-update training and cross-evaluation

Final implicit-arm Newton and explicit-arm PhysX runs: seed 42, 1024 worlds,
24 steps/update, 49,152,000 environment-steps/backend. Newton invalid resets: 0.
Both final checkpoints are finite. Earlier damping-only/explicit-arm retries
with invalid resets were rejected as the final recipe.

| Final training statistic | Newton | PhysX |
|---|---:|---:|
| Mean episode reward | 277.372 | 254.930 |
| Mean episode length (policy steps) | 996.87 | 984.99 |
| Action std | 1.6071 | 1.4555 |
| Raw abs(action)>3 | 35.173% | 38.278% |

Cross-eval pair names mean train/eval physics. Nominal cells use seeds 42/73,
16 replicas/seed and 20 s first episodes; tracking uses surviving samples at t≥4 s.

| Pair | Nominal falls /32 | Normalized tracking error | Stress falls /192 |
|---|---:|---:|---:|
| NN | 0 | 0.049139 | 192 |
| NP | 26 | 0.193093 | 192 |
| PN | 1 | 0.075455 | 188 |
| PP | 6 | 0.110837 | 191 |

Stress aggregates are not a backend ranking; exposure/valid-sample counts differ.

### Stage D: deployment and IK

Flat-floor MuJoCo, 0.005 s physics / 0.02 s policy, seeds 42/73,
seven scenarios ×32 first episodes, default arm: Newton falls **0/224**,
PhysX falls **32/224**. At yaw +0.8 rad/s, requested-axis MAE was
0.40724/0.41159 rad/s (N/P; PhysX censored after falls).
The identical-state observation check used 384 packets and passed `1e-4`.
Native IK frame validation rejected a fixed-root Newton probe with 0.709 m
FK/body discrepancy; it is not valid accuracy evidence. Gravity compensation
then improved position tracking, without proving learned stance reliability.

### Final 25k run and Stage H

Recipe: `1e99cb5`, seed 42, 4096 worlds, 24 rollout steps/update, 25,000 updates
(2,457,600,000 aggregate environment-steps), final iteration 24999. Checkpoint
SHA256: `cc99eef56211c8b5609a875464efa3209bc84311318ec0d45d9aaf4eb8d53c66`.
Final TensorBoard event 24999: reward 289.7581, episode length 997.7100 steps,
std 0.423349, vx/vy/yaw MAE 0.056429 m/s /0.055954 m/s /0.120134 rad/s,
raw abs(action)>3 fraction 0.0890393 (8.90393%), arm/reset intensity 1.
Final 100-update invalid-reset rate is 0/million; monitor records no disjoint or
rolling gate breaches. These training metrics are not deployment measurements.

MuJoCo Stage H keeps Stage D's seeds, delays, thresholds and sample masking.
Each row has 32 episodes; final moving-arm intensity is 1.

| Scenario | 25k default / moving falls | Default / moving requested-axis MAE |
|---|---:|---:|
| Stand | 0 /0 | Planar drift 0.000894 /0.012329 m/s |
| vx 0.3 | 0 /1 | 0.06137 /0.08361 m/s |
| vx 0.6 | 0 /1 | 0.06095 /0.07450 m/s |
| vy +0.3 | 0 /1 | 0.06772 /0.07160 m/s |
| vy −0.3 | 0 /1 | 0.06856 /0.08671 m/s |
| yaw +0.8 | 0 /2 | 0.31083 /0.27298 rad/s |
| yaw −0.8 | 0 /1 | 0.30637 /0.28558 rad/s |
| Total falls /224 | 0 /7 | 2000-update moving baseline: 224/224 |

Standing IK: same 256 base-frame targets, settle 2 s then reach 10 s; every sample
in the final 1 s must meet ≤1 cm error, height>0.20 m and up-cosine>0.8.

| Backend | 2000 /25k simultaneous holds /256 | 25k position p50 /p95 (m) |
|---|---:|---:|
| Newton | 33 /256 | 5.606e-6 /1.774e-5 |
| PhysX | 1 /51 | 0.044677 /0.251354 |

Newton25k combined PD+FF torque clip binds on 99/3,072,000 samples; PhysX25k on
371,840/3,072,000. Applied arm torques remain ≤20.001 Nm.
Five 10 s seed-42 Newton closed-loop videos had zero fall-threshold samples.
They use MuJoCo rasterization of recorded Newton poses, with **no MuJoCo physics
steps**. Native Kit stalled and GL blank-frame/crash attempts were rejected.

Known limitations: PhysX IK only 51/256; moving-arm MuJoCo falls 7/224; yaw MAE
about 0.3 rad/s at 0.8 rad/s; one training seed. Finished-recipe comparisons
combine duration, reward and clip changes. They do not establish a unique cause.
Changed targets or unbounded arm torque do not explain the backend IK gap;
255/256 PhysX bases stand, so stance loss alone is also insufficient.
Next experiment: replay the seven failed moving-arm replicas at identical seeds
and delays, retaining arm/action/fall traces before selecting a controller change.

## Usage

See [the task README](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity_pose/README.md)
for task IDs, train/play/export, smoke, monitoring and MuJoCo commands.
The historical diagnostic scripts and old recipe migrations remain in Git history.
