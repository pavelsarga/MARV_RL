#!/usr/bin/env bash
# Small/short local CREPS run for catching bugs fast, without waiting on a full SLURM job.
# Few envs (rollouts/iteration), few outer iterations, no W&B.
# Usage: bash scripts/train_creps_debug.sh [extra train_creps.py / OmegaConf dotlist args]
# Override any default below via env vars, e.g.: NUM_ENVS=15 bash scripts/train_creps_debug.sh
# NUM_ENVS must be evenly divisible by NUM_EXECUTIONS (default 3 each, matching the
# reference implementation's num_executions_with_same_omega).
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/scripts/lib/debug_train.sh"

debug_train baselines/marv_config_creps.yaml train_creps.py \
    --num_envs "${NUM_ENVS:-15}" \
    num_executions_with_same_omega="${NUM_EXECUTIONS:-3}" \
    num_iterations="${NUM_ITERATIONS:-3}" \
    eval_and_save_every="${EVAL_AND_SAVE_EVERY:-1}" \
    -- "$@"
