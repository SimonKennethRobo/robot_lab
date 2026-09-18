#!/usr/bin/env bash
# Five-way stability sweep for the GO2-X5 velocity-pose policy.
# Submit with: sbatch --export=ALL,STABILITY_ID=1,SNAPSHOT_ROOT=/path/to/robot_lab ...
# The checkout is shared through NFS; each task gets one GPU.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

set -euo pipefail
: "${SNAPSHOT_ROOT:?set SNAPSHOT_ROOT to the shared robot_lab checkout}"
: "${STABILITY_ID:?set STABILITY_ID to 1..5}"

TASK=RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0
SEED=$((4200 + STABILITY_ID))
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-12000}
CONDA_ENV=${CONDA_ENV:-isaac230}
case "$STABILITY_ID" in
  1) NAME=stability_contact2_balance08_ext25
     OVERRIDES="env.robustness.standing_contact_reward_weight=2.0 env.robustness.leg_velocity_balance_cost_weight=0.08 env.robustness.arm_full_extension_fraction=0.25 env.robustness.arm_full_extension_standing_fraction=0.50" ;;
  2) NAME=stability_contact4_balance12_ext25
     OVERRIDES="env.robustness.standing_contact_reward_weight=4.0 env.robustness.leg_velocity_balance_cost_weight=0.12 env.robustness.arm_full_extension_fraction=0.25 env.robustness.arm_full_extension_standing_fraction=0.75" ;;
  3) NAME=stability_contact2_balance08_slowarm
     OVERRIDES="env.robustness.standing_contact_reward_weight=2.0 env.robustness.leg_velocity_balance_cost_weight=0.08 env.robustness.arm_full_extension_max_velocity=1.0 env.robustness.arm_full_extension_max_acceleration=2.0" ;;
  4) NAME=stability_contact3_balance15_ext35
     OVERRIDES="env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.15 env.robustness.arm_full_extension_fraction=0.35 env.robustness.arm_full_extension_standing_fraction=0.50" ;;
  5) NAME=stability_contact3_balance10_ext35_hold
     OVERRIDES="env.robustness.standing_contact_reward_weight=3.0 env.robustness.leg_velocity_balance_cost_weight=0.10 env.robustness.arm_full_extension_fraction=0.35 env.robustness.arm_full_extension_standing_fraction=0.80 env.robustness.arm_full_extension_hold_s=4.0" ;;
  *) echo "STABILITY_ID must be 1..5" >&2; exit 2 ;;
esac

set +u
source /home/simon/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=offline
export PYTHONPATH="$SNAPSHOT_ROOT/source/robot_lab:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/scratch/$USER/robot_lab_cache"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"
cd "$SNAPSHOT_ROOT"

RUN_ROOT="$SNAPSHOT_ROOT/cluster_runs/stability_20260918"
mkdir -p "$RUN_ROOT"
printf 'id=%s\nname=%s\nhost=%s\nseed=%s\nnum_envs=%s\nmax_iterations=%s\noverrides=%s\n' \
  "$STABILITY_ID" "$NAME" "$(hostname -s)" "$SEED" "$NUM_ENVS" "$MAX_ITERATIONS" "$OVERRIDES" \
  > "$RUN_ROOT/${NAME}-${SLURM_JOB_ID}.env"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
  > "$RUN_ROOT/${NAME}-${SLURM_JOB_ID}.gpu.csv"

srun python -u scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" --headless --device cuda:0 --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" --seed "$SEED" --run_name "$NAME" \
  --logger tensorboard $OVERRIDES
