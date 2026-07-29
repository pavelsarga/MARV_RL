#!/bin/bash
# Evaluate a trained CREPS (Pecka et al. 2016) policy using eval_creps.py.
#
# This is the CREPS counterpart of eval.sh (PPO-only) / eval_d3qn.sh (D3QN-only):
# eval_creps.py parses the run's config into FtrCREPSConfig and builds a deterministic
# CREPSLowerLevelPolicy from the checkpoint's omega_mean (broadcast to every env — no
# sampling from omega_cov, that's only for training-time exploration).
#
# Usage:
#   ./scripts/eval_creps.sh <run_dir> [extra eval_creps.py args...]
#
# <run_dir> is the path to the run directory. Host paths under the workspace root are
# automatically rewritten to the container mount point /ws/.
#
# Examples:
#   ./scripts/eval_creps.sh experiments/baselines/creps/train_marv_creps_11271127 --num_envs 64 --repeats 10 --headless
#   ./scripts/eval_creps.sh "..." --weights latest --num_envs 192 --repeats 30 --headless
#   ./scripts/eval_creps.sh "..." --weights 29 --output_dir logs/policy_eval --eval_id creps_it29 --headless
#   ./scripts/eval_creps.sh "..." --weights final --map pecka_pallet --headless
#
# --weights {step|final|latest}:
#   final              → creps_state_final.pth
#   step number (e.g. 29) → creps_state_step_29.pth
#   latest / (omitted, no creps_state_final.pth) → highest-numbered creps_state_step_*.pth
#   CREPS SLURM runs that timed out often have no creps_state_final.pth — omitting the
#   flag then auto-selects the latest step checkpoint.
#
# With --output_dir, the run's terrain (config `terrain:`) is recorded in
# <output_dir>/eval_terrain.json and its generator config + preview plot copied to
# <output_dir>/terrain/, so notebooks/eval_analysis.ipynb can label and group by terrain.

set -e

# ---------------------------------------------------------------------------
# Argument handling — first positional arg is the run directory, the rest are
# forwarded verbatim to eval_creps.py.
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [extra eval_creps.py args...]"
    echo ""
    echo "  run_dir  Path to the run directory containing config.yaml + weights/."
    echo "           Either a container path (/ws/...) or a host-relative path."
    exit 1
fi

RUN_DIR="$1"
shift

# Rewrite host-relative paths to the container mount point
WS="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$RUN_DIR" != /ws/* ]]; then
    RUN_DIR="${RUN_DIR#${WS}/}"
    RUN_DIR="/ws/${RUN_DIR#/}"
fi

# Rewrite --output_dir and --env_names_yaml paths to container mount if under WS.
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

# ---------------------------------------------------------------------------
# Environment — mirrors eval_d3qn.sh / train_org.sbatch exactly
# ---------------------------------------------------------------------------
SIF=$WS/containers/isaaclab_optuna.sif

mkdir -p $WS/logs $WS/logs/wandb $WS/logs/isaac_cache $WS/logs/isaac_logs $WS/logs/isaac_data

# Clean stale lock files from any previous crashed run
rm -rf $WS/logs/isaac_cache/Kit 2>/dev/null || true

# Copy host GL libraries needed by Isaac Sim renderer
HOST_LIBS=$WS/logs/host_libs
mkdir -p "$HOST_LIBS"
cp -u /usr/lib/x86_64-linux-gnu/libGLU.so.1 "$HOST_LIBS/" 2>/dev/null || true
cp -u /usr/lib/x86_64-linux-gnu/libXt.so.6 "$HOST_LIBS/" 2>/dev/null || true

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

# ---------------------------------------------------------------------------
# Resolve --weights <step|final|latest> into --policy for eval_creps.py.
#
# --weights 29    →  creps_state_step_29.pth
# --weights final →  creps_state_final.pth
# --weights latest →  highest-numbered step checkpoint (same as omitting the flag)
# (no flag)       →  creps_state_final.pth if it exists, else highest step checkpoint
# ---------------------------------------------------------------------------
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
        POLICY_FILE="creps_state_final.pth"
    elif [ "$WEIGHTS_STEP" = "latest" ] || [ "$WEIGHTS_STEP" = "auto" ]; then
        POLICY_FILE=$(basename "$(ls -v "$HOST_WEIGHTS_DIR"/creps_state_step_*.pth 2>/dev/null | tail -1)")
        [ -z "$POLICY_FILE" ] && { echo "ERROR: no creps_state_step_*.pth found in $HOST_WEIGHTS_DIR"; exit 1; }
    else
        POLICY_FILE="creps_state_step_${WEIGHTS_STEP}.pth"
    fi
    echo "Weights: $POLICY_FILE"
    [ ! -f "$HOST_WEIGHTS_DIR/$POLICY_FILE" ] && echo "WARNING: $POLICY_FILE not found in $HOST_WEIGHTS_DIR"
    set -- "--policy" "$POLICY_FILE" "$@"
else
    # No --weights flag: use final if it exists, otherwise auto-select latest step checkpoint.
    if [ -f "$HOST_WEIGHTS_DIR/creps_state_final.pth" ]; then
        echo "Weights: creps_state_final.pth"
    else
        LATEST_POLICY=$(ls -v "$HOST_WEIGHTS_DIR"/creps_state_step_*.pth 2>/dev/null | tail -1)
        if [ -n "$LATEST_POLICY" ]; then
            POLICY_FILE=$(basename "$LATEST_POLICY")
            echo "No creps_state_final.pth found — auto-selecting latest checkpoint: $POLICY_FILE"
            set -- "--policy" "$POLICY_FILE" "$@"
        else
            echo "WARNING: no weights found in $HOST_WEIGHTS_DIR"
        fi
    fi
fi

echo "========================================================================"
echo "eval_creps.sh — CREPS policy evaluation"
echo "Run directory (container path): $RUN_DIR"
echo "Extra args: $@"
echo "========================================================================"
# --- W&B credentials (sourced for parity with other eval scripts; eval_creps.py disables wandb) ---
SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
    echo "W&B project : ${WANDB_PROJECT}"
else
    echo "WARNING: ${SECRETS_FILE} not found — continuing without it (eval doesn't use W&B)."
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
    python -m marv_rl_training.training.eval_creps \
    --rundir "$RUN_DIR" \
    --max_steps 2000 \
    "$@"

EXIT_STATUS=$?
echo "========================================================================"
echo "eval_creps.sh finished with exit status: $EXIT_STATUS"
echo "========================================================================"
exit $EXIT_STATUS
