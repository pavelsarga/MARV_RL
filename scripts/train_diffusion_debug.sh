#!/usr/bin/env bash
# Small/short local run of the receding-horizon (Phase 1) policy, for catching bugs fast
# without waiting on a full SLURM job. Few envs, a handful of iterations, no W&B.
# Usage: bash scripts/train_diffusion_debug.sh [extra train_diffusion.py / OmegaConf dotlist args]
# Override any default below via env vars, e.g.: NUM_ENVS=16 bash scripts/train_diffusion_debug.sh
#
# The most useful variant is T_a=1 with T_p=1, which reduces the whole thing to ordinary
# single-step PPO and is the cleanest correctness check available:
#   EXECUTION_HORIZON=1 PREDICTION_HORIZON=1 bash scripts/train_diffusion_debug.sh
# (T_p=1 needs a single-level U-Net, hence the down_dims override below.)
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"

NUM_ENVS=${NUM_ENVS:-32}
TIME_STEPS_PER_BATCH=${TIME_STEPS_PER_BATCH:-16}
PREDICTION_HORIZON=${PREDICTION_HORIZON:-16}
EXECUTION_HORIZON=${EXECUTION_HORIZON:-4}
# total_frames counts CONTROL steps: iterations = it / (steps * envs * T_a).
TOTAL_FRAMES=${TOTAL_FRAMES:-$(( NUM_ENVS * TIME_STEPS_PER_BATCH * EXECUTION_HORIZON * 5 ))}
FRAMES_PER_SUB_BATCH=${FRAMES_PER_SUB_BATCH:-128}
EVAL_AND_SAVE_EVERY=${EVAL_AND_SAVE_EVERY:-2}
DOWN_DIMS=${DOWN_DIMS:-[64,128]}

CONFIG=diffusion/marv_config_diffusion_p1.yaml \
TRAIN_SCRIPT=train_diffusion.py \
bash "$WS/scripts/train.sh" \
    --num_envs "$NUM_ENVS" \
    total_frames="$TOTAL_FRAMES" \
    time_steps_per_batch="$TIME_STEPS_PER_BATCH" \
    frames_per_sub_batch="$FRAMES_PER_SUB_BATCH" \
    prediction_horizon="$PREDICTION_HORIZON" \
    execution_horizon="$EXECUTION_HORIZON" \
    policy_opts.down_dims="$DOWN_DIMS" \
    eval_and_save_every="$EVAL_AND_SAVE_EVERY" \
    eval_repeats=1 \
    eval_repeats_after_training=1 \
    use_wandb=false \
    use_tensorboard=false \
    "$@"
