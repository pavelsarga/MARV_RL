#!/bin/bash
# Evaluate a trained SAC policy (currently: C-TRAC) using eval_sac.py.
#
# Generic SAC counterpart of eval_d3qn.sh — eval_sac.py always parses the config into
# FtrSACConfig, same reasoning as why eval_d3qn.py/eval_ftr.py each need their own script
# (the config dataclasses aren't interchangeable).
#
# Usage:
#   ./scripts/eval_sac.sh <run_dir> [extra eval_sac.py args...]
#
# <run_dir> is the path to the run directory. Host paths under the workspace root are
# automatically rewritten to the container mount point /ws/.
#
# Examples:
#   ./scripts/eval_sac.sh logs/train_ctrac_.../attempt_0 --num_envs 16 --repeats 10 --headless
#   ./scripts/eval_sac.sh "..." --weights latest --num_envs 256 --repeats 30 --headless
#   ./scripts/eval_sac.sh "..." --output_dir logs/policy_eval --eval_id ctrac_a0 --headless

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [extra eval_sac.py args...]"
    echo ""
    echo "  run_dir  Path to the run directory containing config.yaml + weights/."
    echo "           Either a container path (/ws/...) or a host-relative path."
    exit 1
fi

RUN_DIR="$1"
shift

WS="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$RUN_DIR" != /ws/* ]]; then
    RUN_DIR="${RUN_DIR#${WS}/}"
    RUN_DIR="/ws/${RUN_DIR#/}"
fi

_rewrite_ws() {
    local p="$1"
    if [[ "$p" != /ws/* && "$p" == "${WS}"* ]]; then
        p="${p#${WS}/}"
        p="/ws/${p#/}"
    fi
    echo "$p"
}
NEW_ARGS_PATHS=()
_next_is_output_dir=0
_next_is_env_names=0
for arg in "$@"; do
    if [ "$_next_is_output_dir" -eq 1 ]; then
        NEW_ARGS_PATHS+=("$(_rewrite_ws "$arg")")
        _next_is_output_dir=0
        continue
    fi
    if [ "$_next_is_env_names" -eq 1 ]; then
        NEW_ARGS_PATHS+=("$(_rewrite_ws "$arg")")
        _next_is_env_names=0
        continue
    fi
    [[ "$arg" == "--output_dir" ]]   && _next_is_output_dir=1
    [[ "$arg" == "--env_names_yaml" ]] && _next_is_env_names=1
    NEW_ARGS_PATHS+=("$arg")
done
set -- "${NEW_ARGS_PATHS[@]}"

SIF=$WS/containers/isaaclab_optuna.sif

mkdir -p $WS/logs $WS/logs/wandb $WS/logs/isaac_cache $WS/logs/isaac_logs $WS/logs/isaac_data
rm -rf $WS/logs/isaac_cache/Kit 2>/dev/null || true

HOST_LIBS=$WS/logs/host_libs
mkdir -p "$HOST_LIBS"
cp -u /usr/lib/x86_64-linux-gnu/libGLU.so.1 "$HOST_LIBS/" 2>/dev/null || true
cp -u /usr/lib/x86_64-linux-gnu/libXt.so.6 "$HOST_LIBS/" 2>/dev/null || true

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

HOST_WEIGHTS_DIR="$WS/${RUN_DIR#/ws/}/weights"

WEIGHTS_STEP=""
NEW_ARGS=()
_skip=0
for arg in "$@"; do
    if [ "$_skip" -eq 1 ]; then
        WEIGHTS_STEP="$arg"
        _skip=0
        continue
    fi
    if [ "$arg" = "--weights" ]; then
        _skip=1
        continue
    fi
    NEW_ARGS+=("$arg")
done
set -- "${NEW_ARGS[@]}"

if [ -n "$WEIGHTS_STEP" ]; then
    if [ "$WEIGHTS_STEP" = "final" ]; then
        POLICY_FILE="policy_final.pth"
        VECNORM_FILE="vecnorm_final.pth"
    elif [ "$WEIGHTS_STEP" = "latest" ] || [ "$WEIGHTS_STEP" = "auto" ]; then
        POLICY_FILE=$(basename "$(ls -v "$HOST_WEIGHTS_DIR"/policy_step_*.pth 2>/dev/null | tail -1)")
        VECNORM_FILE=$(basename "$(ls -v "$HOST_WEIGHTS_DIR"/vecnorm_step_*.pth 2>/dev/null | tail -1)")
        [ -z "$POLICY_FILE" ] && { echo "ERROR: no policy_step_*.pth found in $HOST_WEIGHTS_DIR"; exit 1; }
    else
        POLICY_FILE="policy_step_${WEIGHTS_STEP}.pth"
        VECNORM_FILE="vecnorm_step_${WEIGHTS_STEP}.pth"
    fi
    echo "Weights: $POLICY_FILE  +  $VECNORM_FILE"
    [ ! -f "$HOST_WEIGHTS_DIR/$POLICY_FILE" ] && echo "WARNING: $POLICY_FILE not found in $HOST_WEIGHTS_DIR"
    [ ! -f "$HOST_WEIGHTS_DIR/$VECNORM_FILE" ] && echo "WARNING: $VECNORM_FILE not found in $HOST_WEIGHTS_DIR"
    set -- "--policy" "$POLICY_FILE" "--vecnorm" "$VECNORM_FILE" "$@"
else
    if [ -f "$HOST_WEIGHTS_DIR/policy_final.pth" ]; then
        echo "Weights: policy_final.pth  +  vecnorm_final.pth"
    else
        LATEST_POLICY=$(ls -v "$HOST_WEIGHTS_DIR"/policy_step_*.pth 2>/dev/null | tail -1)
        LATEST_VECNORM=$(ls -v "$HOST_WEIGHTS_DIR"/vecnorm_step_*.pth 2>/dev/null | tail -1)
        if [ -n "$LATEST_POLICY" ] && [ -n "$LATEST_VECNORM" ]; then
            POLICY_FILE=$(basename "$LATEST_POLICY")
            VECNORM_FILE=$(basename "$LATEST_VECNORM")
            echo "No policy_final.pth found — auto-selecting latest checkpoint: $POLICY_FILE"
            set -- "--policy" "$POLICY_FILE" "--vecnorm" "$VECNORM_FILE" "$@"
        else
            echo "WARNING: no weights found in $HOST_WEIGHTS_DIR"
        fi
    fi
fi

echo "========================================================================"
echo "eval_sac.sh — SAC (C-TRAC) policy evaluation"
echo "Run directory (container path): $RUN_DIR"
echo "Extra args: $@"
echo "========================================================================"
SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
    echo "W&B project : ${WANDB_PROJECT}"
else
    echo "ERROR: ${SECRETS_FILE} not found — upload it to the cluster before submitting."
    exit 1
fi
apptainer exec --nv \
    --bind $WS:/ws \
    --bind "$WS/src/FTR-Benchmark":/local/flipper_training/src/FTR-Benchmark \
    --bind "$WS/src/flipper_training":/local/flipper_training/src/flipper_training \
    --bind "$WS/logs/isaac_cache":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/cache \
    --bind "$WS/logs/isaac_logs":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/logs \
    --bind "$WS/logs/isaac_data":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/data \
    --bind "$HOST_LIBS":/host_libs \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env WANDB_PROJECT=${WANDB_PROJECT} \
    --env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
    --env LD_LIBRARY_PATH=/host_libs:\$LD_LIBRARY_PATH \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
    python -m marv_rl_training.training.eval_sac \
    --rundir "$RUN_DIR" \
    --max_steps 2000 \
    "$@"

EXIT_STATUS=$?
echo "========================================================================"
echo "eval_sac.sh finished with exit status: $EXIT_STATUS"
echo "========================================================================"
exit $EXIT_STATUS
