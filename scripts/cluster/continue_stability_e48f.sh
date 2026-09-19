#!/usr/bin/env bash
# Extend the 5577 recipe from 12000 to 18000 completed PPO updates.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${SNAPSHOT_ROOT:?set the immutable robot_lab snapshot path}"
: "${EXPERIMENT_ROOT:?set the experiment output path}"
SOURCE_RUN=2026-09-18_18-25-41_stability_contact3_balance15_ext35
set +u
source /home/simon/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-isaac230}"
set -u
export PYTHONPATH="$SNAPSHOT_ROOT/source/robot_lab:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=offline
export XDG_CACHE_HOME="/scratch/$USER/robot_lab_cache"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
mkdir -p "$EXPERIMENT_ROOT" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"
cd "$SNAPSHOT_ROOT"
# A completed evaluator receipt is required before starting a long run.
test -f "$EXPERIMENT_ROOT/paired_eval/success.json"
sha256sum -c "$SNAPSHOT_ROOT/SHA256SUMS"
export TRAINING_RECEIPT_PATH="$EXPERIMENT_ROOT/training_success_${SLURM_JOB_ID:-local}.json"
# --max_iterations is ADDITIONAL updates in RSL-RL, not the final index.
# The runner restores optimizer state and the 288000-step curriculum clock.
srun python -u scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0 \
  --headless --device cuda:0 --num_envs "${NUM_ENVS:-4096}" \
  --resume --load_run "$SOURCE_RUN" --checkpoint model_11999.pt \
  --max_iterations "${ADDITIONAL_ITERATIONS:-6000}" --seed 4204 \
  --run_name "${RUN_NAME:-stability_p0_5577_resume18k}" --logger tensorboard \
  env.robustness.standing_contact_reward_weight=3.0 \
  env.robustness.leg_velocity_balance_cost_weight=0.15 \
  env.robustness.arm_full_extension_fraction=0.35 \
  env.robustness.arm_full_extension_standing_fraction=0.50
