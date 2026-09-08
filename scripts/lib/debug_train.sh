#!/bin/bash
# Shared body of the scripts/train_*_debug.sh smoke tests.
#
# Each of those runs the real trainer on the real config, shrunk to a few envs and a handful
# of iterations, with logging off — enough to catch a crash or a shape bug in a minute
# instead of finding it an hour into a SLURM job. Only the config, the trainer and the
# per-method size knobs differ; the rest (no W&B, no TensorBoard, single-repeat eval) is the
# same everywhere and lives here.
#
# Only `use_wandb=false use_tensorboard=false` is genuinely universal and lives here — the
# eval-repeat knobs are NOT, since FtrCREPSConfig has no eval_repeats field and rejects it
# at parse time. Each wrapper passes the overrides its own config accepts.
#
# Usage from a wrapper:
#
#     WS="$(cd "$(dirname "$0")/.." && pwd)"
#     source "$WS/scripts/lib/debug_train.sh"
#     debug_train baselines/marv_config_atd3qn.yaml train_d3qn.py \
#         --num_envs "${NUM_ENVS:-32}" total_frames="${TOTAL_FRAMES:-20480}" -- "$@"
#
# Everything between the trainer name and `--` is a per-method override; everything after
# `--` is the caller's own arguments, appended last so they win.
debug_train() {
    local config="$1" train_script="$2"; shift 2

    local overrides=() passthrough=() seen_sep=0 arg
    for arg in "$@"; do
        if [ "$seen_sep" -eq 0 ] && [ "$arg" = "--" ]; then seen_sep=1; continue; fi
        if [ "$seen_sep" -eq 1 ]; then passthrough+=("$arg"); else overrides+=("$arg"); fi
    done

    CONFIG="$config" TRAIN_SCRIPT="$train_script" \
    bash "$WS/scripts/train.sh" \
        "${overrides[@]}" \
        use_wandb=false \
        use_tensorboard=false \
        "${passthrough[@]}"
}
