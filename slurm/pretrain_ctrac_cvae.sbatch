#!/bin/bash
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=cpufast
#SBATCH --time=8:00:00
#SBATCH --job-name=pretrain_ctrac_cvae
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.err

# Stage I for C-TRAC: supervised-only C-VAE pretraining (pretrain_ctrac_cvae.py). Unlike
# every other train_*.py/*.sbatch in this project, this needs NO GPU / Isaac Sim / --nv —
# it's pure PyTorch on a dataset file collect_ctrac_dataset.sbatch already produced, so a
# CPU partition is correct (the earlier OOM kill here was a RAM sizing bug, not a
# missing-GPU issue — see pretrain_ctrac_cvae.py's _load_dataset docstring: loading the
# whole dataset (num_robots x episode_repeats scale, e.g. ~26 GB for a real collection
# run) used to require holding both every shard AND the final concatenated tensor in RAM
# simultaneously. --mem is sized with headroom above that; if you raise
# max_dataset_rows well beyond ~2M transitions, raise this too.

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "########################################################################"

WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif
CONFIG=baselines/marv_config_ctrac_cvae_pretrain.yaml
LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
cp $WS/configs/$CONFIG $LOGDIR/

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

COMMAND="apptainer exec \
    --bind $WS:/ws \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
    python -m marv_rl_training.training.pretrain_ctrac_cvae \
    --config /ws/configs/$CONFIG \
    $@"
# No output_path override here on purpose (same reasoning as collect_ctrac_dataset.sbatch):
# the checkpoint goes where the config says, so marv_config_ctrac.yaml's
# policy_opts.cvae_weights_path can name a stable path instead of a job-id'd logs/ dir.
# Append output_path=... on the sbatch command line to override.

echo "Executing:"
echo "$COMMAND"
echo "------------------------------------------------------------------------"

srun $COMMAND
EXIT_STATUS=$?
echo "########################################################################"
echo "C-VAE pretraining finished with exit status: $EXIT_STATUS"
echo "########################################################################"
exit $EXIT_STATUS
