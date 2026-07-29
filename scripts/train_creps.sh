#!/usr/bin/env bash
# Full local CREPS training run — marv_config_creps.yaml's real settings (not the tiny
# smoke-test overrides in train_creps_debug.sh). CREPS has no replay buffer/optimizer/
# neural net to speak of (see rl_modules/creps/creps_algorithm.py) and num_robots=192 is
# well within a single GPU's budget, so this runs directly against the local isaaclab
# conda env — no apptainer container, no SLURM respawn loop, no W&B (all of that is
# cluster-specific ceremony this doesn't need to run locally).
#
# Usage: bash scripts/train_creps.sh [extra train_creps.py / OmegaConf dotlist args]
# Examples:
#   bash scripts/train_creps.sh
#   bash scripts/train_creps.sh num_iterations=10
#   bash scripts/train_creps.sh --num_envs 96 kl_epsilon=0.3
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=marv_config_creps.yaml

RUN_NAME="train_creps_$(date +%Y-%m-%d_%H-%M-%S)"
LOGDIR=$WS/logs/$RUN_NAME
mkdir -p "$LOGDIR"
cp "$WS/configs/$CONFIG" "$LOGDIR/"

OUTFILE=$LOGDIR/$RUN_NAME.out
exec > >(tee -a "$OUTFILE") 2>&1
echo "Output logged to: $OUTFILE"

cd "$WS" || { echo "Failed to cd into $WS"; exit 1; }
echo "------------------------------------------------------------------------"

# --- GPU check ---
echo "nvidia-smi output:"
nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU may be unavailable. Aborting."; exit 1; }

GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "WARNING: GPU already has $GPU_PROCS active CUDA context(s)."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
fi
echo "------------------------------------------------------------------------"

# --- Execution ---
echo "Running CREPS training..."
echo "------------------------------------------------------------------------"

OMNI_KIT_ACCEPT_EULA=Y \
CUDA_VISIBLE_DEVICES=0 \
conda run -n isaaclab --no-capture-output \
    env PYTHONPATH="$WS/src/FTR-Benchmark:$WS/src/flipper_training" \
    python "$WS/src/flipper_training/marv_rl_training/training/train_creps.py" \
    --config "$WS/configs/$CONFIG" \
    --headless use_wandb=false use_tensorboard=false "$@"

EXIT_STATUS=$?
echo "########################################################################"
echo "Training finished with exit status: $EXIT_STATUS"
echo "Log: $OUTFILE"
echo "########################################################################"
exit $EXIT_STATUS
