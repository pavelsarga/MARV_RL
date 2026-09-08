#!/usr/bin/env bash
# Small/short local C-TRAC (SAC + C-VAE) run for catching bugs fast, without waiting on a
# full SLURM job. Few envs, short horizon, replay updates triggered almost immediately, no
# W&B. Cold-starts the C-VAE (cvae_weights_path stays whatever the config has — set it to
# null for a pure smoke test, or a real Stage I checkpoint to test warm-starting).
# Usage: bash scripts/train_ctrac_debug.sh [extra train_sac.py / OmegaConf dotlist args]
# Override any default below via env vars, e.g.: NUM_ENVS=16 bash scripts/train_ctrac_debug.sh
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/scripts/lib/debug_train.sh"

debug_train baselines/marv_config_ctrac.yaml train_sac.py \
    --num_envs "${NUM_ENVS:-16}" \
    total_frames="${TOTAL_FRAMES:-8192}" \
    min_replay_size="${MIN_REPLAY_SIZE:-512}" \
    updates_per_batch="${UPDATES_PER_BATCH:-2}" \
    cvae_updates_per_sac_step="${CVAE_UPDATES_PER_SAC_STEP:-1}" \
    eval_and_save_every="${EVAL_AND_SAVE_EVERY:-2}" \
    -- "$@"
