#!/bin/bash
# Evaluate a trained FTR PPO policy using eval_ftr.py.
#
# Usage:
#   ./scripts/eval.sh <run_dir> [extra eval_ftr.py args...]
#
# <run_dir> is the path to the run directory. Host paths under the workspace
# root are automatically rewritten to the container mount point /ws/.
# IMPORTANT: Quote the path or pass it on one line — backslash line
# continuation is unreliable when terminal wrapping adds trailing spaces.
#
# Examples:
#   ./scripts/eval.sh "/home/robot/workspaces/robot_rodeo_gym_ws/runs/ppo/ftr_ppo_..."
#                                          # auto: policy_final.pth or latest step checkpoint
#   ./scripts/eval.sh "..." --weights 5963776      # policy_step_5963776.pth + vecnorm_step_5963776.pth
#   ./scripts/eval.sh "..." --weights final        # policy_final.pth + vecnorm_final.pth
#   ./scripts/eval.sh "..." --weights latest       # highest-numbered step checkpoint
#   ./scripts/eval.sh "..." --weights 5963776 --num_envs 32 --repeats 3
#   ./scripts/eval.sh "..." --weights 5963776 --map cur_stairs_up
#   ./scripts/eval.sh "..." --plot_heightmap
#   ./scripts/eval.sh "..." --plot_heightmap --plot_interval 5
#   Heightmap PNGs (and optionally a GIF) are saved to /tmp/ftr_eval_<timestamp>/ on the host.
#
# Per-env-type CSV output (eval_summary.csv, eval_per_env.csv, eval_episodes.csv):
#   ./scripts/eval.sh "..." --output_dir /tmp/eval_out
#   ./scripts/eval.sh "..." --output_dir /tmp/eval_out --num_env_types 16 --repeats 5
#   ./scripts/eval.sh "..." --output_dir /tmp/eval_out --env_names_yaml /ws/configs/env_names.yaml
#   ./scripts/eval.sh "..." --output_dir /tmp/eval_out --eval_id my_run_label
#   Host paths under the workspace root for --output_dir and --env_names_yaml are
#   automatically rewritten to the container mount point /ws/.
#
# Available terrains: ground  cur_mixed  cur_stairs_up  exp_stair33_up

set -e

# ---------------------------------------------------------------------------
# Argument handling — first positional arg is the run directory, the rest are
# forwarded verbatim to eval_ftr.py.
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [extra eval_ftr.py args...]"
    echo ""
    echo "  run_dir  Path to the run directory containing config.yaml + weights/."
    echo "           Either a container path (/ws/...) or a host-relative path."
    exit 1
fi

RUN_DIR="$1"
shift

# Detect --plot_heightmap in remaining args and force --num_envs 1
PLOT_HEIGHTMAP=0
for arg in "$@"; do
    [[ "$arg" == "--plot_heightmap" ]] && PLOT_HEIGHTMAP=1
done
if [ "$PLOT_HEIGHTMAP" -eq 1 ]; then
    # Inject --num_envs 1 unless the user already passed --num_envs
    if ! echo "$@" | grep -q "\-\-num_envs"; then
        set -- "--num_envs" "1" "$@"
    fi
    echo "Heightmap mode: forcing num_envs=1. Plots will be saved to /tmp/ftr_eval_<timestamp>/ on the host."
fi

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
# Environment — mirrors train_org.sbatch exactly
# ---------------------------------------------------------------------------
SIF=$WS/containers/isaaclab.sif

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
# Resolve --weights <step|final|latest> into --policy / --vecnorm for eval_ftr.py.
#
# --weights 500000  →  policy_step_500000.pth + vecnorm_step_500000.pth
# --weights final   →  policy_final.pth        + vecnorm_final.pth
# --weights latest  →  highest-numbered step checkpoint (same as omitting the flag)
# (no flag)         →  policy_final.pth if it exists, else highest step checkpoint
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
echo "eval.sh — FTR policy evaluation"
echo "Run directory (container path): $RUN_DIR"
echo "Extra args: $@"
echo "========================================================================"
# --- W&B credentials ---
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
    --bind "$WS/src/FTR-benchmark":/local/flipper_training/src/FTR-benchmark \
    --bind "$WS/src/flipper_training":/local/flipper_training/src/flipper_training \
    --bind "$WS/logs/isaac_cache":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/cache \
    --bind "$WS/logs/isaac_logs":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/logs \
    --bind "$WS/logs/isaac_data":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/data \
    --bind "$HOST_LIBS":/host_libs \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env WANDB_PROJECT=${WANDB_PROJECT} \
    --env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    --env LD_LIBRARY_PATH=/host_libs:\$LD_LIBRARY_PATH \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python -m flipper_training.experiments.ppo.eval_ftr \
    --rundir "$RUN_DIR" \
    --max_steps 2000 \
    --num_env_types 16 \
    "$@"

EXIT_STATUS=$?
echo "========================================================================"
echo "eval.sh finished with exit status: $EXIT_STATUS"
echo "========================================================================"
exit $EXIT_STATUS
