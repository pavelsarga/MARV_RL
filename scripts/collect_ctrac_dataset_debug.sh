#!/usr/bin/env bash
# Small/short local C-TRAC dataset-collection run (collect_ctrac_dataset.py) for catching
# bugs fast. Requires marv_config_ctrac_collect_dataset.yaml's marv_rl_weights_path to
# already point at a real, trained marv_rl checkpoint — this script does not train one.
# Usage: bash scripts/collect_ctrac_dataset_debug.sh [extra collect_ctrac_dataset.py args]
# Override defaults via env vars, e.g.: NUM_ENVS=8 bash scripts/collect_ctrac_dataset_debug.sh
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=${CONFIG:-baselines/marv_config_ctrac_collect_dataset.yaml}
NUM_ENVS=${NUM_ENVS:-8}
EPISODE_REPEATS=${EPISODE_REPEATS:-2}
OUTPUT=${OUTPUT:-$WS/logs/ctrac_dataset_debug/ctrac_stage1_shards_debug}

mkdir -p "$OUTPUT"

cd $WS || { echo "Failed to cd into $WS"; exit 1; }
echo "nvidia-smi output:"
nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU may be unavailable. Aborting."; exit 1; }

OMNI_KIT_ACCEPT_EULA=Y \
CUDA_VISIBLE_DEVICES=0 \
conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=$WS/src/FTR-Benchmark:$WS/src/flipper_training \
    python $WS/src/flipper_training/marv_rl_training/training/collect_ctrac_dataset.py \
    --config $WS/configs/$CONFIG \
    --num_envs "$NUM_ENVS" \
    --episode_repeats "$EPISODE_REPEATS" \
    --output "$OUTPUT" \
    log_every_n_steps=1 shard_size_steps=100 \
    --headless "$@"

EXIT_STATUS=$?
echo "Dataset collection finished with exit status: $EXIT_STATUS"
echo "Output: $OUTPUT"
exit $EXIT_STATUS
