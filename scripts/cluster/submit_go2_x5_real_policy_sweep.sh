#!/bin/bash

set -euo pipefail

: "${SNAPSHOT_ROOT:?Export SNAPSHOT_ROOT before submitting the sweep}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SBATCH_SCRIPT="$SCRIPT_DIR/train_go2_x5_slurm.sh"
MILD_TASK="RobotLab-Isaac-VelocityPose-Mild-Unitree-Go2-X5-v0"
FLAT_TASK="RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-X5-v0"

mkdir -p "$SNAPSHOT_ROOT/cluster_runs/slurm" "$SNAPSHOT_ROOT/cluster_runs/jobs"

submit() {
    local job_name=$1
    local task=$2
    local seed=$3
    local overrides=$4

    sbatch --parsable \
        --job-name="$job_name" \
        --output="$SNAPSHOT_ROOT/cluster_runs/slurm/%x-%j.out" \
        --error="$SNAPSHOT_ROOT/cluster_runs/slurm/%x-%j.err" \
        --export="ALL,SNAPSHOT_ROOT=$SNAPSHOT_ROOT,TASK=$task,RUN_NAME=$job_name,SEED=$seed,NUM_ENVS=4096,MAX_ITERATIONS=20000,CONDA_ENV=isaac230,EXTRA_OVERRIDES=$overrides" \
        "$SBATCH_SCRIPT"
}

# Two seeds of the most likely deployment recipe reduce seed-selection risk.
submit "go2x5_mild_s2r_s42" "$MILD_TASK" 42 \
    "env.robustness.domain_rand=sim2real"
submit "go2x5_mild_s2r_s73" "$MILD_TASK" 73 \
    "env.robustness.domain_rand=sim2real"

# A less aggressive arm/load envelope can preserve gait quality for typical hardware use.
submit "go2x5_mild_practical_s42" "$MILD_TASK" 42 \
    "env.robustness.domain_rand=sim2real env.robustness.hard_fraction=0.1 env.robustness.hard_reset_degrees=35 env.robustness.arm_max_velocity=2 env.robustness.arm_max_acceleration=6 env.robustness.payload_max_kg=1"

# Flat-floor candidate gives a nominal-performance upper bound for early hardware trials.
submit "go2x5_flat_s2r_s42" "$FLAT_TASK" 42 \
    "env.robustness.domain_rand=sim2real"
