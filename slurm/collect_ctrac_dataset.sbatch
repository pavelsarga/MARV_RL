#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --time=4:00:00
#SBATCH --job-name=collect_ctrac_dataset
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.err

# Stage 0 for C-TRAC: collects the Stage I C-VAE pretraining dataset by rolling out an
# already-trained marv_rl policy through the CTRAC-configured env (collect_ctrac_dataset.py).
# marv_config_ctrac_collect_dataset.yaml's marv_rl_weights_path must point at a real,
# trained marv_rl checkpoint before submitting this.

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "########################################################################"

WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif
CONFIG=marv_config_ctrac_collect_dataset.yaml
LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
cp $WS/configs/$CONFIG $LOGDIR/

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

nvidia-smi || { echo "nvidia-smi failed — GPU driver unreachable"; exit 1; }

COMMAND="apptainer exec --nv \
    --bind $WS:/ws \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/marv_rl_training/training/collect_ctrac_dataset.py \
    --config /ws/configs/$CONFIG \
    --output /ws/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/ctrac_stage1_dataset.pt \
    --headless $@"

echo "Executing:"
echo "$COMMAND"
echo "------------------------------------------------------------------------"

srun $COMMAND
EXIT_STATUS=$?
echo "########################################################################"
echo "Dataset collection finished with exit status: $EXIT_STATUS"
echo "########################################################################"
exit $EXIT_STATUS
