#!/usr/bin/env bash
# Small/short local AT-D3QN run for catching bugs fast, without waiting on a full SLURM
# job. Few envs, short horizon, replay updates triggered almost immediately, no W&B.
# Usage: bash scripts/train_atd3qn_debug.sh [extra train_d3qn.py / OmegaConf dotlist args]
# Override any default below via env vars, e.g.: NUM_ENVS=16 bash scripts/train_atd3qn_debug.sh
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"

NUM_ENVS=${NUM_ENVS:-32}
TOTAL_FRAMES=${TOTAL_FRAMES:-20480}          # ~5 collector iterations at NUM_ENVS=32
MIN_REPLAY_SIZE=${MIN_REPLAY_SIZE:-1000}     # well under one iteration's worth of transitions
UPDATES_PER_BATCH=${UPDATES_PER_BATCH:-4}
EVAL_AND_SAVE_EVERY=${EVAL_AND_SAVE_EVERY:-2}

CONFIG=baselines/marv_config_atd3qn.yaml \
TRAIN_SCRIPT=train_d3qn.py \
bash "$WS/scripts/train.sh" \
    --num_envs "$NUM_ENVS" \
    total_frames="$TOTAL_FRAMES" \
    min_replay_size="$MIN_REPLAY_SIZE" \
    updates_per_batch="$UPDATES_PER_BATCH" \
    eval_and_save_every="$EVAL_AND_SAVE_EVERY" \
    eval_repeats=1 \
    eval_repeats_after_training=1 \
    use_wandb=false \
    use_tensorboard=false \
    "$@"
