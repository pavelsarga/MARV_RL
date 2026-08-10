#!/bin/bash
# Auto-dispatching eval wrapper: reads env_cfg_overrides.module_name from the run's saved
# config.yaml and forwards to the matching eval_*.sh script, so the same command works for
# any trained module without you having to remember/change which script it needs.
#
# Module -> script mapping (rl_modules/registry.py):
#   marv_rl, hfc, mitriakov  -> eval.sh       (eval_ftr.py, PPO)
#   atd3qn, icmd3qn          -> eval_d3qn.sh  (eval_d3qn.py)
#   creps                    -> eval_creps.sh (eval_creps.py)
#   ctrac                    -> eval_sac.sh   (eval_sac.py)
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

MODULE_NAME=""
if command -v python3 >/dev/null 2>&1 && python3 -c "import yaml" >/dev/null 2>&1; then
    MODULE_NAME=$(python3 - "$CONFIG_FILE" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
print((cfg.get("env_cfg_overrides") or {}).get("module_name", ""))
PYEOF
)
else
    # Fallback if python3/PyYAML isn't available on this host: grep the first non-comment
    # "module_name: <value>" line (every config in this project indents it 2 spaces under
    # env_cfg_overrides:).
    MODULE_NAME=$(grep -E "^[[:space:]]*module_name:[[:space:]]*" "$CONFIG_FILE" \
        | grep -v "^[[:space:]]*#" | head -1 \
        | sed -E 's/^[[:space:]]*module_name:[[:space:]]*//' | tr -d '"'"'" | awk '{print $1}')
fi

if [ -z "$MODULE_NAME" ]; then
    echo "ERROR: could not determine env_cfg_overrides.module_name from $CONFIG_FILE"
    exit 1
fi

echo "Detected module_name: $MODULE_NAME"

case "$MODULE_NAME" in
    marv_rl|hfc|mitriakov)
        TARGET_SCRIPT="$WS/scripts/eval.sh"
        ;;
    atd3qn|icmd3qn)
        TARGET_SCRIPT="$WS/scripts/eval_d3qn.sh"
        ;;
    creps)
        TARGET_SCRIPT="$WS/scripts/eval_creps.sh"
        ;;
    ctrac)
        TARGET_SCRIPT="$WS/scripts/eval_sac.sh"
        ;;
    *)
        echo "ERROR: unrecognized module_name '$MODULE_NAME' — no eval script mapping known for it."
        echo "       Known: marv_rl, hfc, mitriakov -> eval.sh | atd3qn, icmd3qn -> eval_d3qn.sh | creps -> eval_creps.sh | ctrac -> eval_sac.sh"
        exit 1
        ;;
esac

echo "Dispatching to: $TARGET_SCRIPT"
exec "$TARGET_SCRIPT" "$RUN_DIR" "$@"
