#!/usr/bin/env bash
# Eight-way SOTA continuation from the real-robot-tested W30 policy.
# Submit from login01.e48f.corelab.lan after committing the source tree.
set -euo pipefail

SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-/home/simon/Projects/Locomotion/robot_lab}
PARENT=${PARENT:-$SNAPSHOT_ROOT/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-14_20-00-39_gait_W30_formal_s42_1000u/model_10698.pt}
NUM_ENVS=${NUM_ENVS:-4096}
ADDITIONAL_ITERATIONS=${ADDITIONAL_ITERATIONS:-24000}
CONDA_ENV=${CONDA_ENV:-isaac230}
TASK=RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0

expected_sha=909c731e5079290a6e3139edf7b06a96dcf146e4d038f4958bbf7bbe44c975eb
test -s "$PARENT"
actual_sha=$(sha256sum "$PARENT" | awk '{print $1}')
[[ "$actual_sha" == "$expected_sha" ]] || { echo "Unexpected parent SHA256: $actual_sha" >&2; exit 2; }

if git -C "$SNAPSHOT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$SNAPSHOT_ROOT" diff --quiet || { echo "Working tree must be clean before submission" >&2; exit 2; }
fi

record="${SUBMISSION_RECORD:-$SNAPSHOT_ROOT/cluster_runs/sota_20260919_submissions.tsv}"
subset=",${SUBSET:-},"
if [[ -e "$record" ]]; then
  echo "Submission record already exists: $record" >&2
  exit 2
fi
mkdir -p "$SNAPSHOT_ROOT/cluster_runs/slurm" "$SNAPSHOT_ROOT/cluster_runs/jobs"

# All branches retain the real-robot W30 command/DR envelope, but use the
# current v4_63 contract and the newly yaw-aware leg-balance implementation.
submit_one() {
  local name=$1 seed=$2 overrides=$3
  sbatch --parsable \
    --gres=shard:1 --constraint= --cpus-per-task=4 --mem=16G \
    --job-name="$name" \
    --output="$SNAPSHOT_ROOT/cluster_runs/slurm/%x-%j.out" \
    --error="$SNAPSHOT_ROOT/cluster_runs/slurm/%x-%j.err" \
    --export="ALL,SNAPSHOT_ROOT=$SNAPSHOT_ROOT,TASK=$TASK,RUN_NAME=$name,SEED=$seed,NUM_ENVS=$NUM_ENVS,MAX_ITERATIONS=$ADDITIONAL_ITERATIONS,CONDA_ENV=$CONDA_ENV,WARM_START=$PARENT,LOGGER=tensorboard,EXTRA_OVERRIDES=$overrides" \
    "$SNAPSHOT_ROOT/tmp/cluster/train_go2_x5_slurm.sh"
}

common="env.robustness.recovery_recipe=none env.robustness.iteration_override=-1 env.robustness.observation_layout=go2_x5_locomotion_v4_63 env.robustness.domain_rand=sim2real env.robustness.recovery_metrics=true env.robustness.payload_max_kg=0 env.robustness.stance_width_target_m=0.30 env.robustness.stance_width_std_m=0.10 env.robustness.stance_width_reward_weight=0.50 env.robustness.rear_stance_width_target_m=0.30 env.robustness.rear_stance_width_std_m=0.08 env.robustness.rear_stance_width_reward_weight=1.00"

# Two seeds per recipe reduce seed-selection risk. The four recipes trade off
# contact robustness, nominal W30 gait preservation, and gait regularity.
declare -a specs=(
  "main|42|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.15"
  "main|73|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.15"
  "contact|42|env.robustness.standing_contact_reward_weight=4.0 env.robustness.leg_velocity_balance_cost_weight=0.12"
  "contact|73|env.robustness.standing_contact_reward_weight=4.0 env.robustness.leg_velocity_balance_cost_weight=0.12"
  "nominal|42|env.robustness.standing_contact_reward_weight=2.0 env.robustness.leg_velocity_balance_cost_weight=0.08 env.robustness.stance_width_reward_weight=1.00"
  "nominal|73|env.robustness.standing_contact_reward_weight=2.0 env.robustness.leg_velocity_balance_cost_weight=0.08 env.robustness.stance_width_reward_weight=1.00"
  "gait|42|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.18 env.robustness.gait_timing_variance_cost_weight=0.25 env.robustness.gait_swing_height_cost_weight=0.50 env.robustness.gait_contact_sync_reward_weight=0.75"
  "gait|73|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.18 env.robustness.gait_timing_variance_cost_weight=0.25 env.robustness.gait_swing_height_cost_weight=0.50 env.robustness.gait_contact_sync_reward_weight=0.75"
)

printf 'name\tseed\tjob_id\tparent_sha\tadditional_iterations\toverrides\n' > "$record"
for spec in "${specs[@]}"; do
  IFS='|' read -r branch seed branch_overrides <<< "$spec"
  [[ "$subset" == ",," || "$subset" == *",$branch,"* ]] || continue
  name="sota_${branch}_s${seed}_to34698"
  overrides="$common $branch_overrides"
  job=$(submit_one "$name" "$seed" "$overrides")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$seed" "$job" "$actual_sha" "$ADDITIONAL_ITERATIONS" "$overrides" | tee -a "$record"
done

echo "Submitted 8 SOTA jobs; record: $record"
