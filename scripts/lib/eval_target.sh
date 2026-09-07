#!/bin/bash
# Shared eval-dispatch detection for scripts/eval_auto.sh and slurm/eval_auto.sbatch.
#
#   detect_eval_target <path/to/config.yaml>
#     -> sets MODULE_NAME (env_cfg_overrides.module_name) and EVAL_KIND, one of:
#        ppo | d3qn | creps | sac | diffusion
#     -> returns non-zero and prints an error if it cannot decide.
#
# WHY EVAL_KIND IS NOT JUST module_name
# -------------------------------------
# module_name selects the observation+reward implementation (rl_modules/registry.py); it does
# NOT say which trainer produced the run, and the two are independent. The receding-horizon /
# diffusion runs use module_name: marv_rl — same observations, same rewards, deliberately, so
# they stay comparable with the PPO baseline — but they are trained by train_diffusion.py and
# parse into FtrDiffusionConfig, not FtrPPOConfig. Dispatching those on module_name alone
# sends them to eval_ftr.py, which dies at parse time on prediction_horizon.
#
# So the trainer-distinguishing config fields are checked FIRST and win over the module map.
# Same rule as CLAUDE.md's "TRAIN_SCRIPT must match CONFIG's trainer": the dataclasses are not
# interchangeable, and the config's own fields are the only honest evidence of which one it is.
detect_eval_target() {
    local config_file="$1"
    MODULE_NAME=""
    EVAL_KIND=""

    if [ ! -f "$config_file" ]; then
        echo "ERROR: no config.yaml found at $config_file" >&2
        return 1
    fi

    local detected=""
    if command -v python3 >/dev/null 2>&1 && python3 -c "import yaml" >/dev/null 2>&1; then
        detected=$(python3 - "$config_file" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
module = (cfg.get("env_cfg_overrides") or {}).get("module_name", "")
# Top-level only. policy_opts carries interpolated copies (prediction_horizon:
# ${prediction_horizon}), which are strings in an unresolved config and must not count.
chunked = all(isinstance(cfg.get(k), int) for k in ("prediction_horizon", "execution_horizon"))
print(module)
print("chunked" if chunked else "")
PYEOF
)
    else
        # Fallback for a host without python3/PyYAML: grep. module_name is indented 2 spaces
        # under env_cfg_overrides:; the horizons we care about are at column 0 (the indented
        # occurrences are the policy_opts interpolations).
        local module
        module=$(grep -E "^[[:space:]]*module_name:[[:space:]]*" "$config_file" \
            | grep -v "^[[:space:]]*#" | head -1 \
            | sed -E 's/^[[:space:]]*module_name:[[:space:]]*//' | tr -d '\042\047' | awk '{print $1}')
        local chunked=""
        if grep -qE "^execution_horizon:[[:space:]]*[0-9]+" "$config_file" \
           && grep -qE "^prediction_horizon:[[:space:]]*[0-9]+" "$config_file"; then
            chunked="chunked"
        fi
        detected=$(printf '%s\n%s\n' "$module" "$chunked")
    fi

    MODULE_NAME=$(printf '%s\n' "$detected" | sed -n '1p')
    local chunked_flag
    chunked_flag=$(printf '%s\n' "$detected" | sed -n '2p')

    if [ -z "$MODULE_NAME" ]; then
        echo "ERROR: could not determine env_cfg_overrides.module_name from $config_file" >&2
        return 1
    fi

    if [ -n "$chunked_flag" ]; then
        # Trainer fields win over the module map — see the comment above.
        EVAL_KIND=diffusion
        return 0
    fi

    case "$MODULE_NAME" in
        marv_rl|hfc|mitriakov) EVAL_KIND=ppo ;;
        atd3qn|icmd3qn)        EVAL_KIND=d3qn ;;
        creps)                 EVAL_KIND=creps ;;
        ctrac)                 EVAL_KIND=sac ;;
        *)
            echo "ERROR: unrecognized module_name '$MODULE_NAME' — no eval script mapping known for it." >&2
            echo "       Known: marv_rl, hfc, mitriakov -> eval_ftr.py | atd3qn, icmd3qn -> eval_d3qn.py |" >&2
            echo "              creps -> eval_creps.py | ctrac -> eval_sac.py |" >&2
            echo "              (any config with top-level prediction_horizon+execution_horizon -> eval_diffusion.py)" >&2
            return 1
            ;;
    esac
    return 0
}
