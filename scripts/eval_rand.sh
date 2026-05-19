#!/bin/bash
# Evaluate the random policy baseline using eval_ftr_rand.py.
#
# Usage:
#   ./scripts/eval_rand.sh <config.yaml> [extra eval_ftr_rand.py args...]
#
# <config.yaml> is the path to the random policy eval config. Host paths under
# the workspace root are automatically rewritten to the container mount /ws/.
# IMPORTANT: Quote the path or pass it on one line.
#
# Examples:
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml --num_envs 64 --repeats 5
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml --plot_heightmap
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml --plot_heightmap --plot_interval 5
#   Heightmap PNGs (and optionally a GIF) are saved to /tmp/ftr_eval_<timestamp>/ on the host.
#
# Per-env-type CSV output (eval_summary.csv, eval_per_env.csv, eval_episodes.csv):
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml --output_dir /tmp/eval_out
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml --output_dir /tmp/eval_out --num_env_types 16 --repeats 5
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml --output_dir /tmp/eval_out --env_names_yaml /ws/configs/env_names.yaml
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml --output_dir /tmp/eval_out --eval_id rand_baseline
#   Host paths under the workspace root for --output_dir and --env_names_yaml are
#   automatically rewritten to the container mount point /ws/.
#
# To change policy parameters without editing the YAML, pass OmegaConf overrides:
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml policy_opts.linear_speed=0.5
#   ./scripts/eval_rand.sh configs/rand_policy_eval.yaml policy_opts.freq_max=0.02 num_robots=128

set -e

# ---------------------------------------------------------------------------
# Argument handling — first positional arg is the config YAML, the rest are
# forwarded verbatim to eval_ftr_rand.py.
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [extra eval_ftr_rand.py args...]"
    echo ""
    echo "  config.yaml  Path to the random policy eval config YAML."
    echo "               Either a container path (/ws/...) or a host-relative path."
    exit 1
fi

CONFIG="$1"
shift

# Detect --plot_heightmap in remaining args and force --num_envs 1
PLOT_HEIGHTMAP=0
for arg in "$@"; do
    [[ "$arg" == "--plot_heightmap" ]] && PLOT_HEIGHTMAP=1
done
if [ "$PLOT_HEIGHTMAP" -eq 1 ]; then
    if ! echo "$@" | grep -q "\-\-num_envs"; then
        set -- "--num_envs" "1" "$@"
    fi
    echo "Heightmap mode: forcing num_envs=1. Plots will be saved to /tmp/ftr_eval_<timestamp>/ on the host."
fi

# Rewrite host-relative paths to the container mount point
WS="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$CONFIG" != /ws/* ]]; then
    CONFIG="${CONFIG#${WS}/}"
    CONFIG="/ws/${CONFIG#/}"
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
    [[ "$arg" == "--output_dir" ]]     && _next_is_output_dir=1
    [[ "$arg" == "--env_names_yaml" ]] && _next_is_env_names=1
    NEW_ARGS_PATHS+=("$arg")
done
set -- "${NEW_ARGS_PATHS[@]}"

# ---------------------------------------------------------------------------
# Environment — mirrors eval.sh exactly
# ---------------------------------------------------------------------------
SIF=$WS/containers/isaaclab_optuna.sif

mkdir -p $WS/logs $WS/logs/wandb $WS/logs/isaac_cache $WS/logs/isaac_logs $WS/logs/isaac_data

rm -rf $WS/logs/isaac_cache/Kit 2>/dev/null || true

HOST_LIBS=$WS/logs/host_libs
mkdir -p "$HOST_LIBS"
cp -u /usr/lib/x86_64-linux-gnu/libGLU.so.1 "$HOST_LIBS/" 2>/dev/null || true
cp -u /usr/lib/x86_64-linux-gnu/libXt.so.6 "$HOST_LIBS/" 2>/dev/null || true

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

echo "========================================================================"
echo "eval_rand.sh — random policy baseline evaluation"
echo "Config (container path): $CONFIG"
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
    python /ws/src/flipper_training/flipper_training/experiments/ppo/eval_ftr_rand.py \
    --config "$CONFIG" \
    --max_steps 2000 \
    --num_env_types 16 \
    "$@"

EXIT_STATUS=$?
echo "========================================================================"
echo "eval_rand.sh finished with exit status: $EXIT_STATUS"
echo "========================================================================"
exit $EXIT_STATUS
