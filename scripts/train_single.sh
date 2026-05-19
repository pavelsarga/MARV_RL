#!/usr/bin/env bash
# Run a single training instance on one GPU inside the isaaclab apptainer container.
# Usage: CONFIG=config.yaml ./scripts/train_single.sh [--gpu 1]
set -e

CONFIG=${CONFIG:?"Set CONFIG"}
GPU=${GPU:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        *) break ;;
    esac
done

WS="$(cd "$(dirname "$0")/.." && pwd)"
SIF="$WS/containers/isaaclab.sif"
LOGDIR="$WS/logs/run_gpu${GPU}_$$"
mkdir -p "$LOGDIR"

echo "Starting on GPU $GPU → config: $CONFIG  logs: $LOGDIR"

apptainer exec --nv \
    --bind "$WS":/ws \
    --bind "$WS/src/FTR-benchmark":/local/flipper_training/src/FTR-benchmark \
    --bind "$WS/src/flipper_training":/local/flipper_training/src/flipper_training \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_DIR=/ws/logs/run_gpu${GPU}_$$ \
    --env WANDB_API_KEY=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 \
    --env CUDA_LAUNCH_BLOCKING=1 \
    "$SIF" \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/train_ftr.py \
    --config "/ws/configs/$CONFIG" \
    --headless --gpu "$GPU" "$@" \
    >"$LOGDIR/output.log" 2>&1 &

echo "PID $! → logs: $LOGDIR/output.log"
