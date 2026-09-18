#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_auto
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# Auto-dispatching eval job: reads the run's saved config.yaml and runs the matching
# eval_*.py, so the same submit command works for any trained module — no need to pick
# eval_ftr.sbatch vs eval_d3qn.sbatch vs eval_creps.sbatch vs eval_sac.sbatch by hand.
# Dispatch rules (and why they are not just module_name) are in scripts/lib/config_detect.sh;
# the job body is slurm/lib/eval_common.sh, shared with the per-method eval scripts.
# scripts/eval_auto.sh is the same dispatch as a local run instead of a SLURM job.
#
# Usage:
#   sbatch slurm/eval_auto.sbatch --rundir <run_dir> [eval_*.py args...]
#
# <run_dir> must be an absolute host path or a /ws/... container path. Paths under $WS are
# rewritten to /ws/... automatically.
#
# Examples:
#   sbatch slurm/eval_auto.sbatch --rundir /ws/experiments/my_ppo_run
#   sbatch slurm/eval_auto.sbatch --rundir /ws/logs/train_marv_atd3qn_.../attempt_0 --num_envs 256 --repeats 30
#   sbatch slurm/eval_auto.sbatch --rundir /ws/experiments/baselines/creps/train_marv_creps_.../attempt_0
#   sbatch slurm/eval_auto.sbatch --rundir /ws/logs/train_ctrac_.../attempt_0 --output_dir /ws/logs/policy_eval

WS=${MARV_WS:-/mnt/personal/sargapav/MARV_RL_ws}

source $WS/slurm/lib/eval_common.sh
run_eval_job "$@"
