#!/usr/bin/env bash
# Small/short local HFC run for catching bugs fast, without waiting on a full SLURM job.
# HFC trains via train_ftr.py (PPO) — see marv_config_hfc.yaml's comment for why.
# Few envs, short horizon, no W&B.
# Usage: bash scripts/train_hfc_debug.sh [extra train_ftr.py / OmegaConf dotlist args]
# Override any default below via env vars, e.g.: NUM_ENVS=16 bash scripts/train_hfc_debug.sh
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/scripts/lib/debug_train.sh"

# TOTAL_FRAMES default is ~4 PPO iterations at NUM_ENVS=16, time_steps_per_batch=128.
debug_train baselines/marv_config_hfc.yaml train_ftr.py \
    --num_envs "${NUM_ENVS:-16}" \
    total_frames="${TOTAL_FRAMES:-8192}" \
    eval_and_save_every="${EVAL_AND_SAVE_EVERY:-2}" \
    -- "$@"
