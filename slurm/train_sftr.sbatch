#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH --time=10:00:00
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
SIF=$WS/containers/isaaclab.sif

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p $LOGDIR $WS/logs/wandb
cp $WS/configs/ftr_config_marv_potential_1.yaml $LOGDIR/

cd $WS || {
    echo "Failed to cd into $WS"
    exit 1
}
echo "Changed directory to $(pwd)"
echo "------------------------------------------------------------------------"

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

# --- Execution ---
COMMAND="apptainer exec --nv \
    --bind $WS:/ws \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env WANDB_PROJECT=${WANDB_PROJECT} \
    --env WANDB_DIR=/ws/logs/wandb \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/train_ftr.py \
    --config /ws/configs/ftr_config_marv_potential_1.yaml \
    --num_envs 256 \
    --terrain cur_mixed \
    --task Marv-Crossing-Potential-v0 \
    --headless $@"

echo "Executing command via srun:"
echo "------------------------------------------------------------------------"

srun $COMMAND

EXIT_STATUS=$?
echo "########################################################################"
echo "Python script finished with exit status: $EXIT_STATUS"
echo "Job finished."
echo "########################################################################"
exit $EXIT_STATUS