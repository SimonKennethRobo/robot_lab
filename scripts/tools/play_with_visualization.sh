#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_path="${ISAACLAB_PYTHON:-python}"
exec "$python_path" scripts/reinforcement_learning/rsl_rl/play.py \
    --task RobotLab-Isaac-VelocityPose-Flat-Unitree-Go2-v0 --num_envs 4 --viz kit "$@"
