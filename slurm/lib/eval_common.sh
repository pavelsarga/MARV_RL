#!/bin/bash
# Shared body of every slurm/eval_*.sbatch that evaluates a saved run directory.
#
# Usage from an sbatch file:
#
#     WS=/mnt/personal/sargapav/MARV_RL_ws
#     FORCE_EVAL_KIND=d3qn        # optional; omit to dispatch on the run's own config
#     source $WS/slurm/lib/eval_common.sh
#     run_eval_job "$@"
#
# Takes --rundir <run_dir> plus any eval_*.py arguments. <run_dir> may be an absolute host
# path or a /ws/... container path; --rundir, --output_dir and --env_names_yaml values under
# $WS are rewritten to /ws/... automatically. --output_dir defaults to the SLURM log dir.
#
# Which eval_*.py runs is read from <run_dir>/config.yaml, on the host, before the container
# starts — see scripts/lib/config_detect.sh for the mapping and why it is not just
# module_name. FORCE_EVAL_KIND pins it instead, which is all the per-method eval_*.sbatch
# files do: they exist for their #SBATCH defaults and for failing loudly when pointed at a
# run of another method.

source "$WS/scripts/lib/config_detect.sh"

# Container mounts $WS as /ws; rewrite any path under $WS to /ws/...
_rewrite_ws() {
    local p="$1"
    if [[ "$p" != /ws/* && "$p" == "${WS}"* ]]; then
        p="${p#${WS}/}"
        p="/ws/${p#/}"
    fi
    echo "$p"
}

# Scans the argument list: rewrites --rundir / --output_dir / --env_names_yaml values to
# container paths and records whether the caller set --output_dir / --policy explicitly.
# Sets RUN_DIR, CALLER_SET_OUTPUT_DIR, CALLER_SET_POLICY and REWRITTEN_ARGS.
_rewrite_eval_args() {
    CALLER_SET_OUTPUT_DIR=0
    CALLER_SET_POLICY=0
    RUN_DIR=""
    REWRITTEN_ARGS=()
    local expect=""
    for arg in "$@"; do
        case "$expect" in
            rundir)    RUN_DIR="$(_rewrite_ws "$arg")"; REWRITTEN_ARGS+=("$RUN_DIR"); expect=""; continue ;;
            path)      REWRITTEN_ARGS+=("$(_rewrite_ws "$arg")"); expect=""; continue ;;
        esac
        case "$arg" in
            --rundir)         expect=rundir ;;
            --output_dir)     expect=path; CALLER_SET_OUTPUT_DIR=1 ;;
            --env_names_yaml) expect=path ;;
            --policy)         CALLER_SET_POLICY=1 ;;
        esac
        REWRITTEN_ARGS+=("$arg")
    done
}

# Prefer the *_final.pth checkpoint; fall back to the highest-numbered *_step_*.pth. SLURM
# runs killed before finishing often have no final checkpoint at all. Skipped when the caller
# passed --policy. Prepends the resolved flags to the positional arguments.
_resolve_weights() {
    [ "$CALLER_SET_POLICY" -eq 0 ] || return 0
    local weights_dir="$WS/${RUN_DIR#/ws/}/weights"
    if [ -f "$weights_dir/$FINAL_POLICY" ]; then
        echo "Weights: $FINAL_POLICY$([ "$HAS_VECNORM" -eq 1 ] && echo "  +  $FINAL_VECNORM")"
        return 0
    fi
    local latest_policy latest_vecnorm policy_file vecnorm_file
    latest_policy=$(ls -v "$weights_dir"/$STEP_POLICY_GLOB 2>/dev/null | tail -1)
    if [ -z "$latest_policy" ]; then
        echo "WARNING: no weights found in $weights_dir"
        return 0
    fi
    policy_file=$(basename "$latest_policy")
    echo "No $FINAL_POLICY found — auto-selecting latest checkpoint: $policy_file"
    RESOLVED_WEIGHT_ARGS=(--policy "$policy_file")
    if [ "$HAS_VECNORM" -eq 1 ]; then
        latest_vecnorm=$(ls -v "$weights_dir"/$STEP_VECNORM_GLOB 2>/dev/null | tail -1)
        if [ -n "$latest_vecnorm" ]; then
            vecnorm_file=$(basename "$latest_vecnorm")
            RESOLVED_WEIGHT_ARGS+=(--vecnorm "$vecnorm_file")
        fi
    fi
}

run_eval_job() {
    echo "########################################################################"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node: $SLURMD_NODENAME"
    echo "Script arguments: $@"
    echo "########################################################################"

    local SIF=$WS/containers/isaaclab_optuna.sif
    local LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
    mkdir -p "$LOGDIR"
    cd $WS || { echo "Failed to cd into $WS"; exit 1; }

    nvidia-smi || { echo "nvidia-smi failed — GPU driver unreachable"; exit 1; }

    _rewrite_eval_args "$@"
    set -- "${REWRITTEN_ARGS[@]}"
    [ -n "$RUN_DIR" ] || { echo "ERROR: --rundir is required."; exit 1; }

    local config_file="$WS/${RUN_DIR#/ws/}/config.yaml"
    [ -f "$config_file" ] || { echo "ERROR: no config.yaml found at $config_file"; exit 1; }
    detect_eval_target "$config_file" || exit 1
    echo "Detected module_name: $MODULE_NAME  (eval kind: $EVAL_KIND)"
    if [ -n "${FORCE_EVAL_KIND:-}" ] && [ "$EVAL_KIND" != "$FORCE_EVAL_KIND" ]; then
        echo "ERROR: this sbatch runs the $FORCE_EVAL_KIND eval, but $RUN_DIR is a $EVAL_KIND run." >&2
        echo "       Use slurm/eval_auto.sbatch, which dispatches on the run's own config." >&2
        exit 1
    fi
    eval_weight_scheme "$EVAL_KIND" || exit 1
    local target_py=/ws/src/flipper_training/marv_rl_training/training/$EVAL_SCRIPT
    echo "Dispatching to: $target_py"

    RESOLVED_WEIGHT_ARGS=()
    _resolve_weights
    set -- "${RESOLVED_WEIGHT_ARGS[@]}" "$@"

    if [ "$CALLER_SET_OUTPUT_DIR" -eq 0 ]; then
        local logdir_container="/ws/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
        set -- "--output_dir" "$logdir_container" "$@"
        echo "Auto output_dir: $logdir_container"
    fi

    local COMMAND="apptainer exec --nv \
        --bind $WS:/ws \
        --env OMNI_KIT_ACCEPT_EULA=Y \
        --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        $SIF \
        conda run -n isaaclab --no-capture-output \
        env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
        python $target_py \
        --headless --max_steps 2000 $@"

    echo "Executing:"
    echo "$COMMAND"
    echo "------------------------------------------------------------------------"

    srun $COMMAND
    local exit_status=$?
    echo "########################################################################"
    echo "Eval finished with exit status: $exit_status"
    echo "########################################################################"
    exit $exit_status
}
