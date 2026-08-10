#!/usr/bin/env bash
# Small/short local HFC run for catching bugs fast, without waiting on a full SLURM job.
# HFC now trains via train_ftr.py (PPO) — see marv_config_hfc.yaml's comment for why.
# Few envs, short horizon, no W&B.
# Usage: bash scripts/train_hfc_debug.sh [extra train_ftr.py / OmegaConf dotlist args]
# Override any default below via env vars, e.g.: NUM_ENVS=16 bash scripts/train_hfc_debug.sh
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"

NUM_ENVS=${NUM_ENVS:-16}
TOTAL_FRAMES=${TOTAL_FRAMES:-8192}            # ~4 PPO iterations at NUM_ENVS=16, time_steps_per_batch=128
EVAL_AND_SAVE_EVERY=${EVAL_AND_SAVE_EVERY:-2}

CONFIG=baselines/marv_config_hfc.yaml \
TRAIN_SCRIPT=train_ftr.py \
bash "$WS/scripts/train.sh" \
    --num_envs "$NUM_ENVS" \
    total_frames="$TOTAL_FRAMES" \
    eval_and_save_every="$EVAL_AND_SAVE_EVERY" \
    eval_repeats=1 \
    eval_repeats_after_training=1 \
    use_wandb=false \
    use_tensorboard=false \
    "$@"
