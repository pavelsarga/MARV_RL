#!/usr/bin/env bash
# Evaluate a trained AT-D3QN / ICM-D3QN policy (eval_d3qn.py).
#
# Usage:
#   ./scripts/eval_d3qn.sh <run_dir> [extra eval_d3qn.py args...]
#
# Not eval.sh: that runs eval_ftr.py, which parses the config into FtrPPOConfig and rejects
# D3QN fields like replay_buffer_capacity. eval_d3qn.py auto-detects atd3qn vs icmd3qn from
# env_cfg_overrides.module_name and builds the matching config + greedy Q-network policy.
#
# --weights {step|final|latest}: latest, or omitting the flag when there is no
# policy_final.pth, picks the highest-numbered policy_step_*.pth — D3QN SLURM runs that
# timed out often have no final checkpoint.
#
# Examples:
#   ./scripts/eval_d3qn.sh logs/train_marv_atd3qn_11196914/attempt_4 --num_envs 16 --repeats 10 --headless
#   ./scripts/eval_d3qn.sh "..." --weights latest --num_envs 256 --repeats 30 --headless
#   ./scripts/eval_d3qn.sh "..." --output_dir logs/policy_eval --eval_id atd3qn_a4 --headless
#
# See scripts/eval.sh for the shared flags; the job body is scripts/lib/eval_run.sh.

WS="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_KIND=d3qn

source "$WS/scripts/lib/eval_run.sh"
run_local_eval "$@"
