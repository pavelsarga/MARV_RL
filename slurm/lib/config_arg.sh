#!/bin/bash
# Shared `--config` handling for the train_*.sbatch scripts.
#
# Lets one sbatch file run any of its method's config variants without editing it:
#
#     sbatch slurm/train_atd3qn.sbatch --config baselines/marv_config_atd3qn_paper.yaml
#     sbatch --job-name=at_flipperonly slurm/train_atd3qn.sbatch \
#            --config baselines/marv_config_atd3qn_flipperonly.yaml
#
# Paths are relative to $WS/configs. Anything that is NOT --config is collected into
# EXTRA_ARGS and passed through to the trainer unchanged (OmegaConf dotlist overrides etc.),
# which is what the scripts previously did with a bare "$@".
#
# Job outputs land in logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}, so pass --job-name when
# launching several variants of the same method or they will only differ by job id.

# parse_config_arg <default-config> <expected-module-name> "$@"
#   Sets CONFIG (string) and EXTRA_ARGS (array) in the caller's scope.
parse_config_arg() {
    CONFIG="$1"; shift
    local expect_module="$1"; shift

    EXTRA_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --config)
                if [ -z "$2" ]; then
                    echo "ERROR: --config needs a value, e.g. --config baselines/marv_config_atd3qn_paper.yaml" >&2
                    exit 1
                fi
                CONFIG="$2"; shift 2 ;;
            --config=*)
                CONFIG="${1#*=}"; shift ;;
            *)
                EXTRA_ARGS+=("$1"); shift ;;
        esac
    done

    if [ ! -f "$WS/configs/$CONFIG" ]; then
        echo "ERROR: config not found: $WS/configs/$CONFIG" >&2
        echo "Available ${expect_module} configs (paths are relative to \$WS/configs):" >&2
        ( cd "$WS/configs" && ls baselines/*"${expect_module}"*.yaml 2>/dev/null | sed 's/^/  /' ) >&2
        exit 1
    fi

    # Fail fast on a config/trainer mismatch. train_ftr.py / train_d3qn.py / train_icmd3qn.py
    # parse the YAML into three non-interchangeable dataclasses, so pointing the wrong sbatch
    # at a config dies deep inside Python with e.g. "FtrD3QNConfig.__init__() got an
    # unexpected keyword argument 'icm_opts'" — after the container and Isaac Sim have
    # already started. Catch it here instead.
    local cfg_module
    cfg_module=$(grep -E '^[[:space:]]*module_name:' "$WS/configs/$CONFIG" | head -1 | awk '{print $2}' | tr -d '"'"'"'')
    if [ -n "$cfg_module" ] && [ "$cfg_module" != "$expect_module" ]; then
        echo "ERROR: $CONFIG has module_name: $cfg_module, but this sbatch runs the $expect_module trainer." >&2
        echo "       Use slurm/train_${cfg_module}.sbatch for that config." >&2
        exit 1
    fi

    echo "Config      : $CONFIG (module_name: ${cfg_module:-unset})"
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        echo "Extra args  : ${EXTRA_ARGS[*]}"
    fi
}
