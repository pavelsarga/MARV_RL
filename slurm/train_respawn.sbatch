#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpulong
#SBATCH --time=42:00:00
#SBATCH --job-name=train_ftr
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.err

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Script arguments received by sbatch script ($0): $@"
echo "Number of arguments: $#"
echo "########################################################################"

# --- Environment Setup ---
echo "Setting up environment..."
WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif
CONFIG=ftr_config_new_v3.yaml
LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p $LOGDIR $WS/logs/wandb
cp $WS/configs/$CONFIG $LOGDIR/

export WANDB_INIT_TIMEOUT=1000
export WANDB_HTTP_TIMEOUT=1000

cd $WS || {
    echo "Failed to cd into $WS"
    exit 1
}
echo "Changed directory to $(pwd)"
echo "------------------------------------------------------------------------"

# --- GPU Check ---
echo "nvidia-smi output:"
nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU may be unavailable on this node. Aborting."; exit 1; }

# Check for existing CUDA contexts (processes holding a GPU context without memory)
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "ERROR: GPU already has $GPU_PROCS active CUDA context(s) — another job may share this GPU. Aborting."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
    exit 1
fi
echo "GPU is clear (no active CUDA contexts)."
echo "------------------------------------------------------------------------"

# --- W&B credentials ---
SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
    echo "W&B project : flipper_training"
else
    echo "ERROR: ${SECRETS_FILE} not found — upload it to the cluster before submitting."
    exit 1
fi
if ! getent hosts api.wandb.ai > /dev/null 2>&1; then
    echo "WARNING: Cannot resolve api.wandb.ai — running W&B in offline mode."
    export WANDB_MODE=offline
fi

# --- Respawn Loop ---
MAX_RESPAWNS=3
RESPAWN_COUNT=0

while true; do
    COMMAND="apptainer exec --nv \
        --bind $WS:/ws \
        --env OMNI_KIT_ACCEPT_EULA=Y \
        --env CUDA_VISIBLE_DEVICES=0 \
        --env WANDB_API_KEY=${WANDB_API_KEY} \
        --env WANDB_PROJECT=flipper_training \
        --env WANDB_DIR=/ws/logs/wandb \
        $SIF \
        conda run -n isaaclab --no-capture-output \
        env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
        python /ws/src/flipper_training/flipper_training/experiments/ppo/train_ftr.py \
        --config /ws/configs/$CONFIG \
        --headless $@"

    echo "########################################################################"
    echo "Starting training (respawn #$RESPAWN_COUNT)..."
    echo "########################################################################"

    srun $COMMAND
    EXIT_STATUS=$?

    echo "########################################################################"
    echo "Python script finished with exit status: $EXIT_STATUS"
    echo "########################################################################"

    # Exit code 75 = CUDA/W&B error, should respawn
    if [ "$EXIT_STATUS" -eq 75 ]; then
        RESPAWN_COUNT=$((RESPAWN_COUNT + 1))
        if [ "$RESPAWN_COUNT" -lt "$MAX_RESPAWNS" ]; then
            echo "Exit code 75 detected (CUDA/W&B error). Respawning training..."
            sleep 5  # Brief pause before respawn
            continue
        else
            echo "Max respawns ($MAX_RESPAWNS) reached. Exiting."
            exit 75
        fi
    else
        # Normal exit (0 or other non-75 code)
        echo "Training finished with exit code $EXIT_STATUS. Not respawning."
        exit $EXIT_STATUS
    fi
done
