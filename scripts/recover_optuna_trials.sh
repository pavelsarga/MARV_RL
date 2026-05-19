#!/bin/bash
# Recover a cancelled Optuna trial by running eval on its saved weights
# and submitting the result to the Optuna DB.
#
# Usage:
#   ./scripts/recover_optuna_trials.sh <trial_num> [extra args...]
#
# <trial_num> is the Optuna trial number to recover (e.g. 39).
#
# Examples:
#   ./scripts/recover_optuna_trials.sh 39
#   ./scripts/recover_optuna_trials.sh 39 --headless

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <trial_num> [extra args...]"
    exit 1
fi

TRIAL_NUM="$1"
shift

WS="$(cd "$(dirname "$0")/.." && pwd)"
SIF=$WS/containers/isaaclab_optuna.sif
OPTUNA_CONFIG=/ws/configs/optuna_ftr.yaml

mkdir -p $WS/logs $WS/logs/wandb $WS/logs/isaac_cache $WS/logs/isaac_logs $WS/logs/isaac_data

rm -rf $WS/logs/isaac_cache/Kit 2>/dev/null || true

HOST_LIBS=$WS/logs/host_libs
mkdir -p "$HOST_LIBS"
cp -u /usr/lib/x86_64-linux-gnu/libGLU.so.1 "$HOST_LIBS/" 2>/dev/null || true
cp -u /usr/lib/x86_64-linux-gnu/libXt.so.6  "$HOST_LIBS/" 2>/dev/null || true

cd "$WS" || { echo "Failed to cd into $WS"; exit 1; }

SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
else
    echo "WARNING: ${SECRETS_FILE} not found — W&B disabled (fine for recovery)."
    export WANDB_API_KEY=""
    export WANDB_PROJECT=""
fi

echo "========================================================================"
echo "recover_optuna_trials.sh — recovering trial ${TRIAL_NUM}"
echo "========================================================================"

apptainer exec --nv \
    --bind $WS:/ws \
    --bind "$WS/src/FTR-benchmark":/local/flipper_training/src/FTR-benchmark \
    --bind "$WS/src/flipper_training":/local/flipper_training/src/flipper_training \
    --bind "$WS/logs/isaac_cache":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/cache \
    --bind "$WS/logs/isaac_logs":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/logs \
    --bind "$WS/logs/isaac_data":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/data \
    --bind "$HOST_LIBS":/host_libs \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_PROJECT="${WANDB_PROJECT}" \
    --env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    --env LD_LIBRARY_PATH=/host_libs:\$LD_LIBRARY_PATH \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python -m flipper_training.experiments.ppo.recover_optuna_trials \
    --trial_num "$TRIAL_NUM" \
    --optuna_config "$OPTUNA_CONFIG" \
    --headless \
    "$@"

EXIT_STATUS=$?
echo "========================================================================"
echo "recover_optuna_trials.sh — trial ${TRIAL_NUM} finished (exit ${EXIT_STATUS})"
echo "========================================================================"
exit $EXIT_STATUS
