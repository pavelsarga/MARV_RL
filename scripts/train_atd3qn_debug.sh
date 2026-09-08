#!/usr/bin/env bash
# Small/short local AT-D3QN run for catching bugs fast, without waiting on a full SLURM job.
# Few envs, short horizon, replay updates triggered almost immediately, no W&B.
# Usage: bash scripts/train_atd3qn_debug.sh [extra train_d3qn.py / OmegaConf dotlist args]
# Override any default below via env vars, e.g.: NUM_ENVS=16 bash scripts/train_atd3qn_debug.sh
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/scripts/lib/debug_train.sh"

debug_train baselines/marv_config_atd3qn.yaml train_d3qn.py \
    --num_envs "${NUM_ENVS:-32}" \
    total_frames="${TOTAL_FRAMES:-20480}" \
    min_replay_size="${MIN_REPLAY_SIZE:-1000}" \
    updates_per_batch="${UPDATES_PER_BATCH:-4}" \
    eval_and_save_every="${EVAL_AND_SAVE_EVERY:-2}" \
    eval_repeats=1 \
    eval_repeats_after_training=1 \
    -- "$@"
