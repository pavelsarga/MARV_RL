#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
#SBATCH --job-name=train_auto
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# Auto-dispatching training job: reads the config given with --config and runs the matching
# train_*.py, so one submit command works for every method — no need to pick
# train_marv_rl.sbatch vs train_atd3qn.sbatch vs train_ctrac.sbatch by hand. Dispatch rules
# (and why they are not just module_name) are in scripts/lib/config_detect.sh; the job body
# is slurm/lib/train_common.sh, shared with the per-method scripts.
#
# Naming: the job renames itself to train_<module_name> once the module is known, so the
# SLURM job name, logs/<name>_<job_id>/ directory and the W&B run all read
# train_<module>_<job_id>. Passing your own --job-name at submit time disables the rename.
#
# Usage:
#   sbatch slurm/train_auto.sbatch --config configs/baselines/marv_config_atd3qn.yaml
#
# --config accepts a path relative to the workspace, relative to $WS/configs, or absolute.
# Everything that is not --config is passed through to the trainer unchanged (OmegaConf
# dotlist overrides etc.). Partition is a submit-time property, so override it there when a
# method needs a different queue than the default amdgpu:
#
#   sbatch --partition=amdgpulong slurm/train_auto.sbatch --config baselines/marv_config_marv_rl.yaml
#   sbatch --partition=amdgpufast slurm/train_auto.sbatch --config baselines/marv_config_creps.yaml
#   sbatch slurm/train_auto.sbatch --config baselines/marv_config_icmd3qn_paper.yaml num_robots=64

WS=/mnt/personal/sargapav/MARV_RL_ws
AUTO_JOB_RENAME=1

source $WS/slurm/lib/train_common.sh
run_training_job "$@"
