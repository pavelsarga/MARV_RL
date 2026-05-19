#!/usr/bin/env bash
set -e

echo "########################################################################"
echo "Local training run"
echo "Script arguments: $@"
echo "########################################################################"

# --- Environment Setup ---
echo "Setting up environment..."
WS="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=${CONFIG:-ftr_config_new_v2.yaml}
RUN_NAME="train_ftr_$(date +%Y-%m-%d_%H-%M-%S)"
LOGDIR=$WS/logs/$RUN_NAME
mkdir -p $LOGDIR $WS/logs/wandb
cp $WS/configs/$CONFIG $LOGDIR/

OUTFILE=$LOGDIR/$RUN_NAME.out
exec > >(tee -a "$OUTFILE") 2>&1
echo "Output logged to: $OUTFILE"

cd $WS || { echo "Failed to cd into $WS"; exit 1; }
echo "Changed directory to $(pwd)"
echo "------------------------------------------------------------------------"

# --- GPU Check ---
echo "nvidia-smi output:"
nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU may be unavailable. Aborting."; exit 1; }

GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: GPU already has $GPU_PROCS active CUDA context(s)."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
fi
echo "------------------------------------------------------------------------"

# --- W&B credentials ---
SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
    echo "W&B project: flipper_training"
else
    echo "WARNING: ${SECRETS_FILE} not found — W&B logging may not work."
fi
export WANDB_MODE=offline
# --- Execution ---
echo "Running training..."
echo "------------------------------------------------------------------------"

OMNI_KIT_ACCEPT_EULA=Y \
CUDA_VISIBLE_DEVICES=0 \
WANDB_API_KEY=${WANDB_API_KEY} \
WANDB_PROJECT=flipper_training \
WANDB_DIR=$WS/logs/wandb \
conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=$WS/src/FTR-benchmark:$WS/src/flipper_training \
    python $WS/src/flipper_training/flipper_training/experiments/ppo/train_ftr.py \
    --config $WS/configs/$CONFIG \
    --headless $@

EXIT_STATUS=$?
echo "########################################################################"
echo "Training finished with exit status: $EXIT_STATUS"
echo "Log: $OUTFILE"
echo "########################################################################"
exit $EXIT_STATUS
