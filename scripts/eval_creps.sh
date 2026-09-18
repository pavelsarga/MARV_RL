#!/usr/bin/env bash
# Evaluate a trained CREPS policy (eval_creps.py).
#
# Usage:
#   ./scripts/eval_creps.sh <run_dir> [extra eval_creps.py args...]
#
# CREPS keeps its whole policy in one creps_state_*.pth file and writes no vecnorm
# checkpoint, so --weights resolves to creps_state_{final,step_<n>}.pth.
#
# Examples:
#   ./scripts/eval_creps.sh experiments/baselines/creps --num_envs 256 --repeats 30 --headless
#   ./scripts/eval_creps.sh "..." --weights latest --output_dir logs/policy_eval --eval_id creps --headless
#
# See scripts/eval.sh for the shared flags; the job body is scripts/lib/eval_run.sh.

WS="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_KIND=creps
WANDB_OPTIONAL=1   # CREPS eval never needed secrets/wandb.env; keep that

source "$WS/scripts/lib/eval_run.sh"
run_local_eval "$@"
