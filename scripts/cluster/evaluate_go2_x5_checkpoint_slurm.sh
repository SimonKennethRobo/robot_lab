#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --constraint=4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

: "${SNAPSHOT_ROOT:?SNAPSHOT_ROOT is required}"
: "${EVALUATION_RUN_DIR:?EVALUATION_RUN_DIR is required}"
: "${ARM_MODE:?ARM_MODE is required}"

set +u
source /home/simon/miniconda3/etc/profile.d/conda.sh
conda activate isaac230
set -u

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export PYTHONPATH="$SNAPSHOT_ROOT/source/robot_lab:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/scratch/$USER/robot_lab_cache"
export TORCH_EXTENSIONS_DIR="/scratch/$USER/robot_lab_cache/torch_extensions"

cd "$SNAPSHOT_ROOT"
srun python -u scripts/reinforcement_learning/rsl_rl/play.py \
    --task RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0 \
    --headless \
    --device cuda:0 \
    --num_envs 1024 \
    --seed 202 \
    --checkpoint "$EVALUATION_RUN_DIR/model_8200.pt" \
    --domain_rand benchmark \
    --robustness_iteration 8000 \
    --arm_mode "$ARM_MODE" \
    --max_steps 1200
