#!/bin/bash
# Shared config introspection for the scripts/ and slurm/ entry points.
#
#   config_module_name <config.yaml>       -> echoes env_cfg_overrides.module_name
#   config_is_chunked  <config.yaml>       -> exit 0 when the config is a receding-horizon one
#   detect_train_target <config.yaml>      -> sets MODULE_NAME, TRAIN_SCRIPT
#   detect_eval_target  <config.yaml>      -> sets MODULE_NAME, EVAL_KIND, EVAL_SCRIPT
#
# WHY THE TRAINER IS NOT JUST module_name
# ---------------------------------------
# module_name selects the observation+reward implementation (rl_modules/registry.py); it does
# NOT say which trainer produced the run, and the two are independent. The receding-horizon /
# diffusion runs use module_name: marv_rl — same observations, same rewards, deliberately, so
# they stay comparable with the PPO baseline — but they are trained by train_diffusion.py and
# parse into FtrDiffusionConfig, not FtrPPOConfig. Dispatching those on module_name alone
# sends them to train_ftr.py/eval_ftr.py, which die at parse time on prediction_horizon.
#
# So the trainer-distinguishing config fields are checked FIRST and win over the module map.
# Same rule as CLAUDE.md's "TRAIN_SCRIPT must match CONFIG's trainer": the dataclasses are not
# interchangeable, and the config's own fields are the only honest evidence of which one it is.

# Reads both facts in one pass. Sets MODULE_NAME and _CFG_CHUNKED ("chunked" or "").
_config_probe() {
    local config_file="$1"
    MODULE_NAME=""
    _CFG_CHUNKED=""

    if [ ! -f "$config_file" ]; then
        echo "ERROR: no config found at $config_file" >&2
        return 1
    fi

    local detected
    if command -v python3 >/dev/null 2>&1 && python3 -c "import yaml" >/dev/null 2>&1; then
        detected=$(python3 - "$config_file" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
# env_cfg_overrides.module_name specifically: several configs also mention module_name in
# their top-of-file `comment:` block, and a plain grep picks that up first.
print((cfg.get("env_cfg_overrides") or {}).get("module_name", ""))
# Top-level only. policy_opts carries interpolated copies (prediction_horizon:
# ${prediction_horizon}), which are strings in an unresolved config and must not count.
print("chunked" if all(isinstance(cfg.get(k), int)
                       for k in ("prediction_horizon", "execution_horizon")) else "")
PYEOF
)
    else
        # Fallback for a host without python3/PyYAML. module_name is indented under
        # env_cfg_overrides:; the horizons we care about are at column 0 (the indented
        # occurrences are the policy_opts interpolations).
        local module chunked=""
        module=$(awk '
            index($0, "env_cfg_overrides:") == 1 { inb = 1; next }
            inb && /^[^[:space:]#]/ { inb = 0 }
            inb && $0 ~ "^[[:space:]]+module_name:" {
                sub(/^[[:space:]]+module_name:[[:space:]]*/, ""); sub(/#.*$/, "");
                gsub(/["'"'"'[:space:]]/, ""); print; exit
            }
        ' "$config_file")
        if grep -qE "^execution_horizon:[[:space:]]*[0-9]+" "$config_file" \
           && grep -qE "^prediction_horizon:[[:space:]]*[0-9]+" "$config_file"; then
            chunked="chunked"
        fi
        detected=$(printf '%s\n%s\n' "$module" "$chunked")
    fi

    MODULE_NAME=$(printf '%s\n' "$detected" | sed -n '1p')
    _CFG_CHUNKED=$(printf '%s\n' "$detected" | sed -n '2p')
    return 0
}

config_module_name() {
    _config_probe "$1" || return 1
    printf '%s\n' "$MODULE_NAME"
}

config_is_chunked() {
    _config_probe "$1" || return 1
    [ -n "$_CFG_CHUNKED" ]
}

# detect_train_target <config.yaml> -> MODULE_NAME, TRAIN_SCRIPT (basename)
detect_train_target() {
    _config_probe "$1" || return 1
    if [ -z "$MODULE_NAME" ]; then
        echo "ERROR: could not determine env_cfg_overrides.module_name from $1" >&2
        echo "       Add it to the config, or set TRAIN_SCRIPT explicitly." >&2
        return 1
    fi
    if [ -n "$_CFG_CHUNKED" ]; then
        TRAIN_SCRIPT=train_diffusion.py   # trainer fields win over the module map
        return 0
    fi
    case "$MODULE_NAME" in
        marv_rl|hfc|mitriakov) TRAIN_SCRIPT=train_ftr.py ;;
        atd3qn)                TRAIN_SCRIPT=train_d3qn.py ;;
        icmd3qn)               TRAIN_SCRIPT=train_icmd3qn.py ;;
        ctrac)                 TRAIN_SCRIPT=train_sac.py ;;
        creps)                 TRAIN_SCRIPT=train_creps.py ;;
        *)
            echo "ERROR: unrecognized module_name '$MODULE_NAME' — no trainer mapping known." >&2
            echo "       Known: marv_rl, hfc, mitriakov -> train_ftr.py | atd3qn -> train_d3qn.py |" >&2
            echo "              icmd3qn -> train_icmd3qn.py | ctrac -> train_sac.py | creps -> train_creps.py |" >&2
            echo "              (any config with top-level prediction_horizon+execution_horizon -> train_diffusion.py)" >&2
            return 1 ;;
    esac
    return 0
}

# detect_eval_target <config.yaml> -> MODULE_NAME, EVAL_KIND, EVAL_SCRIPT (basename)
detect_eval_target() {
    _config_probe "$1" || return 1
    if [ -z "$MODULE_NAME" ]; then
        echo "ERROR: could not determine env_cfg_overrides.module_name from $1" >&2
        return 1
    fi
    if [ -n "$_CFG_CHUNKED" ]; then
        EVAL_KIND=diffusion
        EVAL_SCRIPT=eval_diffusion.py
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
            return 1 ;;
    esac
    eval_script_for_kind "$EVAL_KIND"
}

# eval_script_for_kind <eval_kind> -> sets EVAL_SCRIPT
eval_script_for_kind() {
    case "$1" in
        ppo)       EVAL_SCRIPT=eval_ftr.py ;;
        d3qn)      EVAL_SCRIPT=eval_d3qn.py ;;
        creps)     EVAL_SCRIPT=eval_creps.py ;;
        sac)       EVAL_SCRIPT=eval_sac.py ;;
        diffusion) EVAL_SCRIPT=eval_diffusion.py ;;
        *) echo "ERROR: unknown eval kind '$1'" >&2; return 1 ;;
    esac
    return 0
}

# eval_weight_scheme <eval_kind>
#   Sets the checkpoint-naming convention each trainer writes:
#     FINAL_POLICY / FINAL_VECNORM   the *_final.pth names
#     STEP_POLICY_GLOB / STEP_VECNORM_GLOB   the periodic-checkpoint globs
#     HAS_VECNORM                    0 for CREPS, which stores everything in one state file
# Everything but CREPS writes a policy_/vecnorm_ pair — train_diffusion.py included, since it
# is a fork of train_ftr.py and only the entry point differs.
eval_weight_scheme() {
    FINAL_POLICY=policy_final.pth
    FINAL_VECNORM=vecnorm_final.pth
    STEP_POLICY_GLOB="policy_step_*.pth"
    STEP_VECNORM_GLOB="vecnorm_step_*.pth"
    HAS_VECNORM=1
    case "$1" in
        ppo|d3qn|sac|diffusion) ;;
        creps)
            FINAL_POLICY=creps_state_final.pth
            FINAL_VECNORM=""
            STEP_POLICY_GLOB="creps_state_step_*.pth"
            STEP_VECNORM_GLOB=""
            HAS_VECNORM=0 ;;
        *)
            echo "ERROR: unknown eval kind '$1'" >&2
            return 1 ;;
    esac
    return 0
}
