#!/bin/bash
# Evaluate top-K trials from an Optuna study using eval_optuna_top.py.
#
# Usage:
#   ./scripts/eval_optuna_top.sh --study <study_name> [extra eval_optuna_top.py args...]
#
# Examples:
#   ./scripts/eval_optuna_top.sh --study ftr_potential_optuna_v3 --top 15 --repeats 10 --num_envs 128
#   ./scripts/eval_optuna_top.sh --study ftr_potential_optuna_v3 --top 5 --output /ws/logs/results.csv

set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"
SIF=$WS/containers/isaaclab_optuna.sif

mkdir -p $WS/logs $WS/logs/wandb $WS/logs/isaac_cache $WS/logs/isaac_logs $WS/logs/isaac_data $WS/logs/eval_optuna_top

OUTFILE="$WS/logs/eval_optuna_top/eval_optuna_top_$(date +%Y-%m-%d_%H-%M-%S).out"
exec > >(tee -a "$OUTFILE") 2>&1
echo "Output logged to: $OUTFILE"

rm -rf $WS/logs/isaac_cache/Kit 2>/dev/null || true

HOST_LIBS=$WS/logs/host_libs
mkdir -p "$HOST_LIBS"
cp -u /usr/lib/x86_64-linux-gnu/libGLU.so.1 "$HOST_LIBS/" 2>/dev/null || true
cp -u /usr/lib/x86_64-linux-gnu/libXt.so.6  "$HOST_LIBS/" 2>/dev/null || true

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
else
    echo "ERROR: ${SECRETS_FILE} not found."
    exit 1
fi

apptainer exec --nv \
    --bind $WS:/ws \
    --bind "$WS/logs/isaac_cache":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/cache \
    --bind "$WS/logs/isaac_logs":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/logs \
    --bind "$WS/logs/isaac_data":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/data \
    --bind "$HOST_LIBS":/host_libs \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_MODE=disabled \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env LD_LIBRARY_PATH=/host_libs:\$LD_LIBRARY_PATH \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/eval_optuna_top.py \
    --headless \
    "$@"

EXIT_STATUS=$?
echo "========================================================================"
echo "eval_optuna_top.sh finished with exit status: $EXIT_STATUS"
echo "========================================================================"
exit $EXIT_STATUS
