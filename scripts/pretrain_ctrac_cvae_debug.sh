#!/usr/bin/env bash
# Local C-VAE pretraining smoke test (pretrain_ctrac_cvae.py) on a dataset file
# collect_ctrac_dataset_debug.sh already produced. Unlike every other *_debug.sh in this
# project, this needs no GPU/Isaac Sim/--headless — it's pure PyTorch.
# Usage: bash scripts/pretrain_ctrac_cvae_debug.sh [extra pretrain_ctrac_cvae.py dotlist args]
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=${CONFIG:-baselines/marv_config_ctrac_cvae_pretrain.yaml}
DATASET=${DATASET:-$WS/logs/ctrac_dataset_debug/ctrac_stage1_shards_debug}
OUTPUT=${OUTPUT:-$WS/logs/ctrac_dataset_debug/ctrac_cvae_stage1_debug.pth}
TOTAL_STEPS=${TOTAL_STEPS:-50}

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=$WS/src/FTR-Benchmark:$WS/src/flipper_training \
    python -m marv_rl_training.training.pretrain_ctrac_cvae \
    --config $WS/configs/$CONFIG \
    dataset_path="$DATASET" \
    output_path="$OUTPUT" \
    total_steps="$TOTAL_STEPS" \
    device=cpu \
    "$@"

EXIT_STATUS=$?
echo "C-VAE pretraining finished with exit status: $EXIT_STATUS"
echo "Output: $OUTPUT"
exit $EXIT_STATUS
