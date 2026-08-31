#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
# 16 h, raised from 8. Measured on job 11445067: 6052 of 30000 control steps in 1:07:53,
# i.e. ~5.6 h for the 30000 steps that episode_repeats=100 produces (~11.2 min per 51200-row
# shard) plus ~12 min of Isaac startup. episode_repeats=200 doubles that to ~11.4 h, which
# the old 8 h limit would have cut off around shard 42 of 60.
#SBATCH --time=16:00:00
#SBATCH --job-name=collect_ctrac_dataset
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# Stage 0 for C-TRAC: collects the Stage I C-VAE pretraining dataset by rolling out an
# already-trained marv_rl policy through the CTRAC-configured env (collect_ctrac_dataset.py).
# marv_config_ctrac_collect_dataset.yaml's marv_rl_weights_path must point at a real,
# trained marv_rl checkpoint before submitting this.

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "########################################################################"

WS=/mnt/personal/sargapav/MARV_RL_ws
SIF=$WS/containers/isaaclab_optuna.sif
CONFIG=baselines/marv_config_ctrac_collect_dataset.yaml
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
    --headless $@"
# No --output override here on purpose: the dataset location is the config's own
# output_path, so it stays in one place across the collect -> pretrain -> train chain
# (marv_config_ctrac_cvae_pretrain.yaml's dataset_path has to point at it). An --output
# pointing into this job's logs/ dir used to silently win over the config, which is how a
# collection run ended up somewhere the pretrain config wasn't looking. Relative config
# paths resolve against the workspace root, not cwd — see collect_ctrac_dataset.py's
# _resolve_ws_path. Pass --output explicitly on the sbatch command line to override.

echo "Executing:"
echo "$COMMAND"
echo "------------------------------------------------------------------------"

srun $COMMAND
EXIT_STATUS=$?
echo "########################################################################"
echo "Dataset collection finished with exit status: $EXIT_STATUS"
echo "########################################################################"
exit $EXIT_STATUS
