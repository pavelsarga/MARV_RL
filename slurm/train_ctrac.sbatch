#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
#SBATCH --job-name=train_ctrac
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# Stage II: joint asymmetric SAC + C-VAE training for C-TRAC (Pan et al. 2025).
# Requires Stage I (pretrain_ctrac_cvae.sbatch) to have already produced a checkpoint at
# marv_config_ctrac.yaml's policy_opts.cvae_weights_path — otherwise this cold-starts the
# C-VAE jointly from scratch (matches the paper's own "C-TRAC w/o Pretrained" ablation).
# The job body — argument parsing, trainer dispatch, GPU/W&B checks and the frame-budget
# respawn loop — lives in slurm/lib/train_common.sh, shared by every train_*.sbatch.
#
# Run a different config of the same method without editing this file:
#   sbatch slurm/train_ctrac.sbatch --config baselines/<other>.yaml
#   sbatch --job-name=my_variant slurm/train_ctrac.sbatch --config ... num_robots=64
# Anything that is not --config is forwarded to the trainer as an OmegaConf override.

WS=${MARV_WS:-/mnt/personal/sargapav/MARV_RL_ws}
DEFAULT_CONFIG=baselines/marv_config_ctrac.yaml
EXPECT_MODULE=ctrac

source $WS/slurm/lib/train_common.sh
run_training_job "$@"
