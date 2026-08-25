#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
#SBATCH --job-name=train_mitriakov
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
CONFIG=baselines/marv_config_mitriakov.yaml
LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
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
    echo "W&B project : MARV_RL"
else
    echo "ERROR: ${SECRETS_FILE} not found — upload it to the cluster before submitting."
    exit 1
fi
if ! getent hosts api.wandb.ai > /dev/null 2>&1; then                                     
    echo "WARNING: Cannot resolve api.wandb.ai — running W&B in offline mode."                
    export WANDB_MODE=offline
fi

# --- per-job Isaac Sim shader cache ---
# Isolate this job's Omniverse cache to a node-local /tmp directory.
# Prevents NFS cache corruption from a crashed job affecting subsequent runs.
LOCAL_CACHE="/tmp/omni_cache_${SLURM_JOB_ID}"
echo "Copying Omniverse cache to node-local ${LOCAL_CACHE} ..."
rsync -a --no-perms "${HOME}/.cache/ov/" "${LOCAL_CACHE}/" 2>/dev/null \
    || mkdir -p "${LOCAL_CACHE}"
trap "echo 'Cleaning up node-local cache...'; rm -rf '${LOCAL_CACHE}'" EXIT

# --- Respawn Configuration ---
# Respawns are bounded by the config's frame budget, not by a fixed count: after a crash the run
# is restarted with only the frames still missing, until total_frames has actually been collected.
# STALL_LIMIT is the safety net for a failure that recurs before any progress is made — without it
# an unrecoverable crash would respawn for the rest of the SLURM allocation.
source $WS/slurm/lib/respawn_common.sh
frame_budget_init "$WS/configs/$CONFIG"
STALL_LIMIT=5
STALLED=0
RESPAWN_COUNT=0

# --- Execution Loop ---
while true; do
    REMAINING=$(frames_remaining)
    if [ -n "$REMAINING" ] && [ "$REMAINING" -le 0 ]; then
        echo "########################################################################"
        echo "Frame budget reached: $FRAMES_DONE / $FRAME_BUDGET frames collected. Done."
        echo "########################################################################"
        exit 0
    fi
    FRAME_ARGS=$(frame_overrides "$REMAINING")

    COMMAND="apptainer exec --nv \
        --bind $WS:/ws \
        --bind ${LOCAL_CACHE}:${HOME}/.cache/ov \
        --env OMNI_KIT_ACCEPT_EULA=Y \
        --env CUDA_VISIBLE_DEVICES=0 \
        --env WANDB_API_KEY=${WANDB_API_KEY} \
        --env WANDB_PROJECT=MARV_RL \
        --env WANDB_DIR=/ws/logs/wandb \
        --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        $SIF \
        conda run -n isaaclab --no-capture-output \
        env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
        python /ws/src/flipper_training/marv_rl_training/training/train_ftr.py \
        --config /ws/configs/$CONFIG \
        --headless $FRAME_ARGS $@"

    echo "########################################################################"
    if [ -n "$REMAINING" ]; then
        echo "Starting training (respawn #$RESPAWN_COUNT — $FRAMES_DONE/$FRAME_BUDGET frames done, $REMAINING to go)..."
    else
        echo "Starting training (respawn #$RESPAWN_COUNT)..."
    fi
    echo "########################################################################"

    srun $COMMAND
    EXIT_STATUS=$?

    echo "########################################################################"
    echo "Python script finished with exit status: $EXIT_STATUS"
    echo "########################################################################"

    attempt_record_progress "$LOGDIR"
    echo "Attempt collected $ATTEMPT_PROGRESS frames (job total: $FRAMES_DONE${FRAME_BUDGET:+ / $FRAME_BUDGET})."

    # Exit code 75 = CUDA/W&B error, should respawn and auto-resume from crash checkpoint
    if [ "$EXIT_STATUS" -eq 75 ]; then
        if [ "$ATTEMPT_PROGRESS" -gt 0 ] && attempt_wrote_weights "$ATTEMPT_DIR"; then
            STALLED=0
        else
            STALLED=$((STALLED + 1))
            echo "This attempt made no progress ($STALLED in a row)."
        fi
        if [ "$STALLED" -ge "$STALL_LIMIT" ]; then
            echo "Aborting: $STALLED consecutive respawns made no progress — the failure is not transient."
            exit 75
        fi
        RESPAWN_COUNT=$((RESPAWN_COUNT + 1))
        echo "Exit code 75 detected (CUDA/W&B error). Respawning (attempt $((RESPAWN_COUNT + 1)))..."
        sleep 5  # Brief pause before respawn to allow cleanup
        continue
    else
        # Normal exit (0 or other non-75 code). Exit 0 means the trainer collected everything it
        # was asked for, so the budget is met; anything else is a failure respawning cannot fix.
        echo "Training finished with exit code $EXIT_STATUS. Not respawning."
        exit $EXIT_STATUS
    fi
done
