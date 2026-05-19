#!/usr/bin/env bash
# Run two independent training instances in parallel, one per GPU, inside the isaaclab apptainer container.
# Usage: CONFIG0=config_run0.yaml CONFIG1=config_run1.yaml ./scripts/train_dual.sh
# CONFIG1 defaults to CONFIG0 if not set.
set -e

CONFIG0=${CONFIG0:-${CONFIG:?"Set CONFIG or CONFIG0"}}
CONFIG1=${CONFIG1:-$CONFIG0}

WS="$(cd "$(dirname "$0")/.." && pwd)"
SIF="$WS/containers/isaaclab.sif"
LOGDIR0="$WS/logs/run_0_$$"
LOGDIR1="$WS/logs/run_1_$$"
mkdir -p "$LOGDIR0" "$LOGDIR1"

echo "Starting GPU 0 → config: $CONFIG0  logs: $LOGDIR0"
apptainer exec --nv \
    --bind "$WS":/ws \
    --bind "$WS/src/FTR-benchmark":/local/flipper_training/src/FTR-benchmark \
    --bind "$WS/src/flipper_training":/local/flipper_training/src/flipper_training \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_DIR=/ws/logs/run_0_$$ \
    --env WANDB_API_KEY=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 \
    --env CUDA_LAUNCH_BLOCKING=1 \
    "$SIF" \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/train_ftr.py \
    --config "/ws/configs/$CONFIG0" \
    --headless --gpu 0 "$@" \
    > "$LOGDIR0/output.log" 2>&1 &
PID0=$!

echo "Starting GPU 1 → config: $CONFIG1  logs: $LOGDIR1"
apptainer exec --nv \
    --bind "$WS":/ws \
    --bind "$WS/src/FTR-benchmark":/local/flipper_training/src/FTR-benchmark \
    --bind "$WS/src/flipper_training":/local/flipper_training/src/flipper_training \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_DIR=/ws/logs/run_1_$$ \
    --env WANDB_API_KEY=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 \
    --env CUDA_LAUNCH_BLOCKING=1 \
    "$SIF" \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/train_ftr.py \
    --config "/ws/configs/$CONFIG1" \
    --headless --gpu 1 "$@" \
    > "$LOGDIR1/output.log" 2>&1 &
PID1=$!

echo "Both runs launched (PIDs: $PID0 $PID1). Waiting..."

EXIT=0
wait $PID0 || { echo "GPU 0 run failed (PID $PID0)"; EXIT=1; }
wait $PID1 || { echo "GPU 1 run failed (PID $PID1)"; EXIT=1; }
exit $EXIT
