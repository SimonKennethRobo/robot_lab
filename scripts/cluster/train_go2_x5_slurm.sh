#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --constraint=4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00

set -euo pipefail

: "${SNAPSHOT_ROOT:?SNAPSHOT_ROOT must point to the immutable NFS source snapshot}"
: "${TASK:?TASK is required}"
: "${RUN_NAME:?RUN_NAME is required}"

SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
CONDA_ENV="${CONDA_ENV:-isaac230}"

# IsaacLab's activation hook probes shell-specific variables that may be unset.
set +u
source /home/simon/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export PYTHONPATH="$SNAPSHOT_ROOT/source/robot_lab:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/scratch/$USER/robot_lab_cache"
export TORCH_EXTENSIONS_DIR="/scratch/$USER/robot_lab_cache/torch_extensions"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"

cd "$SNAPSHOT_ROOT"

JOB_RECORD_DIR="$SNAPSHOT_ROOT/cluster_runs/jobs/$SLURM_JOB_ID"
mkdir -p "$JOB_RECORD_DIR"
{
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=$SLURM_JOB_NAME"
    echo "host=$(hostname -s)"
    echo "task=$TASK"
    echo "run_name=$RUN_NAME"
    echo "seed=$SEED"
    echo "num_envs=$NUM_ENVS"
    echo "max_iterations=$MAX_ITERATIONS"
    echo "extra_overrides=$EXTRA_OVERRIDES"
    echo "conda_env=$CONDA_ENV"
    echo "python=$(command -v python)"
    echo "started_at=$(date --iso-8601=seconds)"
} > "$JOB_RECORD_DIR/runtime.env"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
    > "$JOB_RECORD_DIR/gpu.csv"

override_args=()
if [[ -n "$EXTRA_OVERRIDES" ]]; then
    read -r -a override_args <<< "$EXTRA_OVERRIDES"
fi

set +e
srun python -u scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --device cuda:0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    --logger tensorboard \
    "${override_args[@]}"
status=$?
set -e

{
    echo "exit_code=$status"
    echo "finished_at=$(date --iso-8601=seconds)"
} > "$JOB_RECORD_DIR/completion.env"
exit "$status"
