#!/usr/bin/env bash
# Evaluate a trained C-TRAC policy — asymmetric SAC + C-VAE, Pan et al. 2025 (eval_sac.py).
#
# Usage:
#   ./scripts/eval_sac.sh <run_dir> [extra eval_sac.py args...]
#
# Examples:
#   ./scripts/eval_sac.sh logs/train_ctrac_11362192/attempt_14 --num_envs 256 --repeats 30 --headless
#   ./scripts/eval_sac.sh "..." --weights latest --output_dir logs/policy_eval --eval_id ctrac --headless
#
# See scripts/eval.sh for the shared flags; the job body is scripts/lib/eval_run.sh.

WS="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_KIND=sac

source "$WS/scripts/lib/eval_run.sh"
run_local_eval "$@"
