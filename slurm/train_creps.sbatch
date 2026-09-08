#!/bin/bash
#SBATCH --mem=30G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --job-name=train_marv_creps
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# The job body — argument parsing, trainer dispatch, GPU/W&B checks and the frame-budget
# respawn loop — lives in slurm/lib/train_common.sh, shared by every train_*.sbatch.
#
# Run a different config of the same method without editing this file:
#   sbatch slurm/train_creps.sbatch --config baselines/<other>.yaml
#   sbatch --job-name=my_variant slurm/train_creps.sbatch --config ... num_robots=64
# Anything that is not --config is forwarded to the trainer as an OmegaConf override.

WS=/mnt/personal/sargapav/MARV_RL_ws
DEFAULT_CONFIG=baselines/marv_config_creps.yaml
EXPECT_MODULE=creps

source $WS/slurm/lib/train_common.sh
run_training_job "$@"
