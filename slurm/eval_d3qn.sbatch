#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_d3qn
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# Evaluate a trained AT-D3QN / ICM-D3QN policy from a run directory.
#
# Usage:
#   sbatch slurm/eval_d3qn.sbatch --rundir <run_dir> [eval args...]
#
# <run_dir> is an absolute host path or a /ws/... container path; paths under $WS are
# rewritten to /ws/... automatically. Weights default to the run's *_final.pth checkpoint,
# falling back to the highest-numbered *_step_*.pth — SLURM runs that timed out often have
# no final checkpoint. --output_dir defaults to this job's log directory.
#
# Alongside the CSVs the job writes eval_terrain.json (which terrain this eval_id ran on, its
# env-type row names and depth-column count, from the run config's `terrain:`) and copies
# that terrain's generator config + preview plot out of ftr_envs/assets/terrain/ into
# <output_dir>/terrain/. notebooks/eval_analysis.ipynb reads those to label and group its
# plots, so pull the whole output directory back — not just the CSVs.
#
# Examples:
#   sbatch slurm/eval_d3qn.sbatch --rundir /ws/experiments/my_run --repeats 20
#   sbatch slurm/eval_d3qn.sbatch --rundir /ws/experiments/my_run --num_envs 256 --map cur_mixed
#   sbatch slurm/eval_d3qn.sbatch --rundir /ws/experiments/my_run --output_dir /ws/experiments --eval_id label
#
# eval_d3qn.py auto-detects atd3qn vs icmd3qn from env_cfg_overrides.module_name and builds
# the matching FtrD3QNConfig/FtrICMD3QNConfig + greedy Q-network policy. eval_ftr.py cannot
# be used: it parses the config into FtrPPOConfig, which rejects fields like
# replay_buffer_capacity.
# The job body is slurm/lib/eval_common.sh, shared by every eval_*.sbatch. Use
# slurm/eval_auto.sbatch to dispatch on the run's own config instead of pinning the method.

WS=/mnt/personal/sargapav/MARV_RL_ws
FORCE_EVAL_KIND=d3qn

source $WS/slurm/lib/eval_common.sh
run_eval_job "$@"
