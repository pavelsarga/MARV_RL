#!/bin/bash
# Evaluate a trained AT-D3QN / ICM-D3QN policy using eval_d3qn.py.
#
# This is the D3QN-family counterpart of eval.sh (which is PPO-only: eval_ftr.py parses
# the config into FtrPPOConfig and rejects D3QN fields like replay_buffer_capacity).
# eval_d3qn.py auto-detects atd3qn vs icmd3qn from env_cfg_overrides.module_name and
# builds the matching FtrD3QNConfig/FtrICMD3QNConfig + greedy Q-network policy.
#
# Usage:
#   ./scripts/eval_d3qn.sh <run_dir> [extra eval_d3qn.py args...]
#
# <run_dir> is the path to the run directory. Host paths under the workspace root are
# automatically rewritten to the container mount point /ws/.
#
# Examples:
#   ./scripts/eval_d3qn.sh logs/train_marv_atd3qn_11196914/attempt_4 --num_envs 16 --repeats 10 --headless
#   ./scripts/eval_d3qn.sh "..." --weights latest --num_envs 256 --repeats 30 --headless
#   ./scripts/eval_d3qn.sh "..." --weights 50462720 --output_dir logs/policy_eval --eval_id atd3qn_a4 --headless
#   ./scripts/eval_d3qn.sh "..." --weights final --map cur_stairs_up --headless
#
# --weights {step|final|latest}:
#   latest / (omitted, no policy_final.pth) → highest-numbered policy_step_*.pth checkpoint.
#   D3QN SLURM runs that timed out often have no policy_final.pth — omitting the flag then
#   auto-selects the latest step checkpoint.
#
# With --output_dir, the run's terrain (config `terrain:`) is recorded in
# <output_dir>/eval_terrain.json and its generator config + preview plot copied to
# <output_dir>/terrain/, so notebooks/eval_analysis.ipynb can label and group by terrain.
#
# Available terrains: ground  cur_mixed  cur_stairs_up  exp_stair33_up  and everything under
# ftr_envs/assets/terrain/gen_config/ (custom_mixed, pan_symmetric, mitriakov_stairs, ...)

set -e

# ---------------------------------------------------------------------------
# Argument handling — first positional arg is the run directory, the rest are
# forwarded verbatim to eval_d3qn.py.
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [extra eval_d3qn.py args...]"
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
    # Strip leading workspace root if the user passed an absolute host path
    RUN_DIR="${RUN_DIR#${WS}/}"
    # Now prepend the container mount
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
# Environment — mirrors train_org.sbatch / eval.sh exactly
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
# Resolve --weights <step|final|latest> into --policy / --vecnorm for eval_d3qn.py.
#
# --weights 50462720  →  policy_step_50462720.pth + vecnorm_step_50462720.pth
# --weights final     →  policy_final.pth          + vecnorm_final.pth
# --weights latest    →  highest-numbered step checkpoint (same as omitting the flag)
# (no flag)           →  policy_final.pth if it exists, else highest step checkpoint
# ---------------------------------------------------------------------------
HOST_WEIGHTS_DIR="$WS/${RUN_DIR#/ws/}/weights"

# Extract --weights value from args (remove the flag+value from $@ afterwards)
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
    # No --weights flag: use final if it exists, otherwise auto-select latest step checkpoint.
    # D3QN SLURM runs that were killed before finishing have no policy_final.pth.
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
echo "eval_d3qn.sh — AT-D3QN / ICM-D3QN policy evaluation"
echo "Run directory (container path): $RUN_DIR"
echo "Extra args: $@"
echo "========================================================================"
# --- W&B credentials (sourced for parity with eval.sh; eval_d3qn.py disables wandb) ---
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
    python -m marv_rl_training.training.eval_d3qn \
    --rundir "$RUN_DIR" \
    --max_steps 2000 \
    "$@"

EXIT_STATUS=$?
echo "========================================================================"
echo "eval_d3qn.sh finished with exit status: $EXIT_STATUS"
echo "========================================================================"
exit $EXIT_STATUS
