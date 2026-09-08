#!/usr/bin/env bash
# Auto-dispatching eval wrapper: reads the run's saved config.yaml and runs the matching
# eval_*.py, so the same command works for any trained run without picking a script by hand.
#
# Dispatch rules — and why they are not just env_cfg_overrides.module_name — are in
# scripts/lib/config_detect.sh; the job body is scripts/lib/eval_run.sh, shared with the
# per-method wrappers. slurm/eval_auto.sbatch is the same dispatch as a SLURM job.
#
# Usage:
#   ./scripts/eval_auto.sh <run_dir> [extra eval args...]
#
# Examples:
#   ./scripts/eval_auto.sh logs/train_marv_atd3qn_.../attempt_0 --num_envs 16 --repeats 10 --headless
#   ./scripts/eval_auto.sh experiments/baselines/creps --weights latest --headless
#   ./scripts/eval_auto.sh logs/train_ctrac_.../attempt_0 --output_dir logs/policy_eval --eval_id ctrac_a0 --headless

WS="$(cd "$(dirname "$0")/.." && pwd)"

source "$WS/scripts/lib/eval_run.sh"
run_local_eval "$@"
