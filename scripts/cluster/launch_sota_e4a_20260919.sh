#!/usr/bin/env bash
# Launch the eight physical E4A GPUs through the shared NFS login host.
set -euo pipefail

ROOT=${ROOT:-/home/simon-nfs/Projects/Locomotion/robot_lab_sota_20260919}
PARENT=${PARENT:-$ROOT/logs/rsl_rl/unitree_go2_x5_velocity_pose_mild/2026-09-14_20-00-39_gait_W30_formal_s42_1000u/model_10698.pt}
PORT=${PORT:-65522}
REMOTE_USER=${REMOTE_USER:-simon-nfs}
ADDITIONAL_ITERATIONS=${ADDITIONAL_ITERATIONS:-24000}
TAG=${TAG:-sota_20260919_e4a}
RUN_FILTER=${RUN_FILTER:-}

expected_sha=909c731e5079290a6e3139edf7b06a96dcf146e4d038f4958bbf7bbe44c975eb
actual_sha=$(sha256sum "$PARENT" | awk '{print $1}')
[[ "$actual_sha" == "$expected_sha" ]] || { echo "Unexpected parent SHA256: $actual_sha" >&2; exit 2; }

record="$ROOT/cluster_runs/$TAG/submissions.tsv"
mkdir -p "$ROOT/cluster_runs/$TAG/logs"
if [[ -e "$record" ]]; then
  echo "Submission record already exists: $record" >&2
  exit 2
fi
printf 'branch\tseed\thost\tgpu\tnum_envs\tpid\tlog\tparent_sha\toverrides\n' > "$record"

common="env.robustness.recovery_recipe=none env.robustness.iteration_override=-1 env.robustness.observation_layout=go2_x5_locomotion_v4_63 env.robustness.domain_rand=sim2real env.robustness.recovery_metrics=true env.robustness.payload_max_kg=0 env.robustness.stance_width_target_m=0.30 env.robustness.stance_width_std_m=0.10 env.robustness.stance_width_reward_weight=0.50 env.robustness.rear_stance_width_target_m=0.30 env.robustness.rear_stance_width_std_m=0.08 env.robustness.rear_stance_width_reward_weight=1.00"

declare -a specs=(
  "main|42|simon-5090.lan|0|4096|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.15"
  "main|73|corelab-4080s.lan|0|2048|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.15"
  "contact|42|corelab-4080s.lan|1|2048|env.robustness.standing_contact_reward_weight=4.0 env.robustness.leg_velocity_balance_cost_weight=0.12"
  "contact|73|corelab-5080-1.lan|0|2048|env.robustness.standing_contact_reward_weight=4.0 env.robustness.leg_velocity_balance_cost_weight=0.12"
  "nominal|42|corelab-5080-2.lan|0|2048|env.robustness.standing_contact_reward_weight=2.0 env.robustness.leg_velocity_balance_cost_weight=0.08 env.robustness.stance_width_reward_weight=1.00"
  "nominal|73|actlab-a4.lan|0|1024|env.robustness.standing_contact_reward_weight=2.0 env.robustness.leg_velocity_balance_cost_weight=0.08 env.robustness.stance_width_reward_weight=1.00"
  "gait|42|actlab-a5.lan|0|1024|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.18 env.robustness.gait_timing_variance_cost_weight=0.25 env.robustness.gait_swing_height_cost_weight=0.50 env.robustness.gait_contact_sync_reward_weight=0.75"
  "gait|73|actlab-a6.lan|0|1024|env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.18 env.robustness.gait_timing_variance_cost_weight=0.25 env.robustness.gait_swing_height_cost_weight=0.50 env.robustness.gait_contact_sync_reward_weight=0.75"
)

for spec in "${specs[@]}"; do
  IFS='|' read -r branch seed host gpu num_envs branch_overrides <<< "$spec"
  run_key="${branch}_s${seed}"
  [[ -z "$RUN_FILTER" || ",${RUN_FILTER}," == *",${run_key},"* ]] || continue
  name="${TAG}_${branch}_s${seed}"
  log="$ROOT/cluster_runs/$TAG/logs/${name}.log"
  overrides="$common $branch_overrides"
  remote_script=$(cat <<EOF
set -euo pipefail
set +u
source /home/simon-nfs/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab230
set -u
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=online WANDB_PROJECT=robotlab
export PYTHONPATH="$ROOT/source/robot_lab:/home/simon-nfs/Apps/Simulator/Isaac/isaac-lab-2.3.0/source/isaaclab:/home/simon-nfs/Apps/Simulator/Isaac/isaac-lab-2.3.0/source/isaaclab_tasks:/home/simon-nfs/Apps/Simulator/Isaac/isaac-lab-2.3.0/source/isaaclab_rl:/opt/isaac/isaacsim-5.0/extscache/omni.kit.pip_archive-0.0.0+8131b85d.lx64.cp311/pip_prebundle:/home/simon-nfs/Apps/Simulator/Isaac/isaac-lab-2.3.0/_isaac_sim/extscache/omni.services.pip_archive-0.16.0+107.0.3.lx64.cp311/pip_prebundle:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="/local_ssd/\${USER}/.cache/robot_lab_sota_20260919"
export TORCH_EXTENSIONS_DIR="\$XDG_CACHE_HOME/torch_extensions"
mkdir -p "\$XDG_CACHE_HOME" "\$TORCH_EXTENSIONS_DIR"
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES=$gpu python -u scripts/reinforcement_learning/rsl_rl/train.py \\
  --task RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0 --headless \\
  --device cuda:0 --num_envs $num_envs --max_iterations $ADDITIONAL_ITERATIONS \\
  --seed $seed --run_name "$name" --logger wandb --log_project_name robotlab --warm_start "$PARENT" \\
  $overrides >"$log" 2>&1 < /dev/null &
echo \$!
EOF
)
  pid=$(ssh -p "$PORT" "$REMOTE_USER@$host" "bash -s" <<< "$remote_script")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$branch" "$seed" "$host" "$gpu" "$num_envs" "$pid" "$log" "$actual_sha" "$overrides" | tee -a "$record"
done
