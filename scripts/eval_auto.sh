#!/bin/bash
# Auto-dispatching eval wrapper: reads the run's saved config.yaml and forwards to the
# matching eval_*.sh script, so the same command works for any trained run without you
# having to remember/change which script it needs.
#
# Detection lives in scripts/lib/eval_target.sh (shared with slurm/eval_auto.sbatch):
#   top-level prediction_horizon + execution_horizon  -> eval_diffusion.sh (eval_diffusion.py)
#   else by env_cfg_overrides.module_name (rl_modules/registry.py):
#     marv_rl, hfc, mitriakov  -> eval.sh       (eval_ftr.py, PPO)
#     atd3qn, icmd3qn          -> eval_d3qn.sh  (eval_d3qn.py)
#     creps                    -> eval_creps.sh (eval_creps.py)
#     ctrac                    -> eval_sac.sh   (eval_sac.py)
#
# The horizon check comes first and wins: receding-horizon runs use module_name: marv_rl on
# purpose (same obs/rewards as the baseline, so the comparison means something), but they are
# trained by train_diffusion.py and parse into FtrDiffusionConfig. Routing them on module_name
# alone sends them to eval_ftr.py, which dies on prediction_horizon at parse time.
#
# Usage:
#   ./scripts/eval_auto.sh <run_dir> [extra eval args...]
#
# <run_dir> is the path to the run directory (host-relative, absolute host, or /ws/...
# container path — same convention every eval_*.sh already uses). All other args are
# forwarded verbatim to the dispatched script.
#
# Examples:
#   ./scripts/eval_auto.sh logs/train_marv_atd3qn_.../attempt_0 --num_envs 16 --repeats 10 --headless
#   ./scripts/eval_auto.sh experiments/baselines/creps/train_marv_creps_.../attempt_0 --weights latest --headless
#   ./scripts/eval_auto.sh logs/train_ctrac_.../attempt_0 --output_dir logs/policy_eval --eval_id ctrac_a0 --headless

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [extra eval args...]"
    echo ""
    echo "  run_dir  Path to the run directory containing config.yaml + weights/."
    exit 1
fi

RUN_DIR="$1"
shift

WS="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve a HOST filesystem path to config.yaml, regardless of whether RUN_DIR was passed
# as a container path (/ws/...), an absolute host path, or a workspace-relative path — this
# is only used here to locate+read config.yaml; the dispatched script does its own separate
# rewriting of RUN_DIR for the container.
HOST_RUN_DIR="$RUN_DIR"
if [[ "$HOST_RUN_DIR" == /ws/* ]]; then
    HOST_RUN_DIR="$WS/${HOST_RUN_DIR#/ws/}"
elif [[ "$HOST_RUN_DIR" != /* ]]; then
    HOST_RUN_DIR="$WS/$HOST_RUN_DIR"
fi
CONFIG_FILE="$HOST_RUN_DIR/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: no config.yaml found at $CONFIG_FILE"
    exit 1
fi

source "$WS/scripts/lib/eval_target.sh"
detect_eval_target "$CONFIG_FILE" || exit 1

echo "Detected module_name: $MODULE_NAME  (eval kind: $EVAL_KIND)"

case "$EVAL_KIND" in
    ppo)       TARGET_SCRIPT="$WS/scripts/eval.sh" ;;
    d3qn)      TARGET_SCRIPT="$WS/scripts/eval_d3qn.sh" ;;
    creps)     TARGET_SCRIPT="$WS/scripts/eval_creps.sh" ;;
    sac)       TARGET_SCRIPT="$WS/scripts/eval_sac.sh" ;;
    diffusion) TARGET_SCRIPT="$WS/scripts/eval_diffusion.sh" ;;
    *)
        echo "ERROR: detect_eval_target returned unknown kind '$EVAL_KIND'"
        exit 1
        ;;
esac

echo "Dispatching to: $TARGET_SCRIPT"
exec "$TARGET_SCRIPT" "$RUN_DIR" "$@"
