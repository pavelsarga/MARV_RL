#!/bin/bash
# Shared body of scripts/eval.sh, eval_d3qn.sh, eval_sac.sh, eval_creps.sh and eval_auto.sh.
#
# Runs one of the eval_*.py entry points against a saved run directory, inside the Apptainer
# container, on this machine. The per-method wrappers exist only to pin which entry point
# runs and to document its flags; everything below — path rewriting, --weights resolution,
# the Isaac cache/host-GL setup and the apptainer invocation — is the same for all of them.
#
# Usage from a wrapper:
#
#     WS="$(cd "$(dirname "$0")/.." && pwd)"
#     EVAL_KIND=ppo                      # or leave unset to dispatch on the run's config
#     source "$WS/scripts/lib/eval_run.sh"
#     run_local_eval "$@"
#
# The first positional argument is the run directory; everything else is forwarded to the
# eval script verbatim. Host paths under the workspace root are rewritten to the container
# mount point /ws/ — for the run directory itself and for --output_dir / --env_names_yaml.
#
# EVAL_SCRIPT=<name>.py in the environment overrides the entry point outright. That is how
# scripts/eval_diffusion.sh reuses this: a receding-horizon run has module_name: marv_rl but
# is trained by train_diffusion.py, so it needs eval_diffusion.py rather than eval_ftr.py.

source "$WS/scripts/lib/config_detect.sh"

# --weights {step|final|latest} -> the matching --policy / --vecnorm pair under
# <run_dir>/weights/.
#
#   --weights 500000  ->  policy_step_500000.pth + vecnorm_step_500000.pth
#   --weights final   ->  policy_final.pth       + vecnorm_final.pth
#   --weights latest  ->  highest-numbered step checkpoint (same as omitting the flag)
#   (no flag)         ->  policy_final.pth if it exists, else the highest step checkpoint
#
# CREPS names its single state file creps_state_*.pth and has no vecnorm — see
# eval_weight_scheme in config_detect.sh. Runs killed by a SLURM timeout often have no
# final checkpoint, which is why omitting the flag falls back rather than failing.
_resolve_weight_args() {
    local weights_dir="$1" step="$2"
    RESOLVED_WEIGHT_ARGS=()

    if [ -n "$step" ]; then
        local policy_file vecnorm_file
        case "$step" in
            final)
                policy_file="$FINAL_POLICY"; vecnorm_file="$FINAL_VECNORM" ;;
            latest|auto)
                policy_file=$(basename "$(ls -v "$weights_dir"/$STEP_POLICY_GLOB 2>/dev/null | tail -1)")
                [ -n "$policy_file" ] || { echo "ERROR: no $STEP_POLICY_GLOB found in $weights_dir"; exit 1; }
                [ "$HAS_VECNORM" -eq 1 ] && vecnorm_file=$(basename "$(ls -v "$weights_dir"/$STEP_VECNORM_GLOB 2>/dev/null | tail -1)") ;;
            *)
                policy_file="${STEP_POLICY_GLOB/\*/$step}"
                [ "$HAS_VECNORM" -eq 1 ] && vecnorm_file="${STEP_VECNORM_GLOB/\*/$step}" ;;
        esac
        echo "Weights: $policy_file${vecnorm_file:+  +  $vecnorm_file}"
        [ -f "$weights_dir/$policy_file" ] || echo "WARNING: $policy_file not found in $weights_dir"
        RESOLVED_WEIGHT_ARGS=(--policy "$policy_file")
        if [ -n "$vecnorm_file" ]; then
            [ -f "$weights_dir/$vecnorm_file" ] || echo "WARNING: $vecnorm_file not found in $weights_dir"
            RESOLVED_WEIGHT_ARGS+=(--vecnorm "$vecnorm_file")
        fi
        return 0
    fi

    if [ -f "$weights_dir/$FINAL_POLICY" ]; then
        # The eval scripts already default to the final checkpoint; nothing to pass.
        echo "Weights: $FINAL_POLICY$([ "$HAS_VECNORM" -eq 1 ] && echo "  +  $FINAL_VECNORM")"
        return 0
    fi

    local latest_policy latest_vecnorm
    latest_policy=$(ls -v "$weights_dir"/$STEP_POLICY_GLOB 2>/dev/null | tail -1)
    if [ -z "$latest_policy" ]; then
        echo "WARNING: no weights found in $weights_dir"
        return 0
    fi
    echo "No $FINAL_POLICY found — auto-selecting latest checkpoint: $(basename "$latest_policy")"
    RESOLVED_WEIGHT_ARGS=(--policy "$(basename "$latest_policy")")
    if [ "$HAS_VECNORM" -eq 1 ]; then
        latest_vecnorm=$(ls -v "$weights_dir"/$STEP_VECNORM_GLOB 2>/dev/null | tail -1)
        [ -n "$latest_vecnorm" ] && RESOLVED_WEIGHT_ARGS+=(--vecnorm "$(basename "$latest_vecnorm")")
    fi
}

_rewrite_ws() {
    local p="$1"
    if [[ "$p" != /ws/* && "$p" == "${WS}"* ]]; then
        p="${p#${WS}/}"
        p="/ws/${p#/}"
    fi
    echo "$p"
}

run_local_eval() {
    set -e

    if [ $# -lt 1 ]; then
        echo "Usage: $0 <run_dir> [extra eval args...]"
        echo ""
        echo "  run_dir  Path to the run directory containing config.yaml + weights/."
        echo "           Either a container path (/ws/...) or a host path."
        exit 1
    fi

    local RUN_DIR="$1"; shift

    # --plot_heightmap / --print_actions render one robot; more than one env makes the
    # output meaningless, so force --num_envs 1 unless the caller set it.
    local arg
    for arg in "$@"; do
        case "$arg" in
            --plot_heightmap|--print_actions)
                if ! printf '%s\n' "$@" | grep -qx -- "--num_envs"; then
                    set -- "--num_envs" "1" "$@"
                    echo "${arg#--} mode: forcing num_envs=1."
                fi
                break ;;
        esac
    done

    # Resolve the run directory on both sides: HOST_RUN_DIR to read config.yaml and the
    # weights, RUN_DIR as the container path handed to the eval script.
    local HOST_RUN_DIR="$RUN_DIR"
    if [[ "$HOST_RUN_DIR" == /ws/* ]]; then
        HOST_RUN_DIR="$WS/${HOST_RUN_DIR#/ws/}"
    elif [[ "$HOST_RUN_DIR" != /* ]]; then
        HOST_RUN_DIR="$WS/$HOST_RUN_DIR"
    fi
    RUN_DIR="/ws/${HOST_RUN_DIR#$WS/}"

    # Entry point: an explicit EVAL_SCRIPT wins, then the wrapper's EVAL_KIND, then the
    # run's own config. See config_detect.sh for why this is not just module_name.
    local config_file="$HOST_RUN_DIR/config.yaml"
    local override_script="${EVAL_SCRIPT:-}"
    if [ -z "${EVAL_KIND:-}" ]; then
        detect_eval_target "$config_file" || exit 1
        echo "Detected module_name: $MODULE_NAME  (eval kind: $EVAL_KIND)"
    else
        local pinned="$EVAL_KIND"
        # A pinned kind is authoritative, but say so when the run looks like another method:
        # the eval scripts parse the saved config into their own non-interchangeable dataclass,
        # so a mismatch dies deep inside Python instead of here.
        if [ -z "$override_script" ] && [ -f "$config_file" ] \
           && detect_eval_target "$config_file" 2>/dev/null && [ "$EVAL_KIND" != "$pinned" ]; then
            echo "WARNING: $RUN_DIR looks like a $EVAL_KIND run, but this script runs the $pinned eval."
            echo "         Use scripts/eval_auto.sh to dispatch on the run's own config."
        fi
        EVAL_KIND="$pinned"
        eval_script_for_kind "$EVAL_KIND" || exit 1
    fi
    eval_weight_scheme "$EVAL_KIND" || exit 1
    [ -n "$override_script" ] && EVAL_SCRIPT="$override_script"
    local eval_module=${EVAL_SCRIPT%.py}

    # Rewrite output paths that point into the workspace to their container equivalents.
    local new_args=() expect=""
    for arg in "$@"; do
        if [ -n "$expect" ]; then
            new_args+=("$(_rewrite_ws "$arg")"); expect=""; continue
        fi
        case "$arg" in
            --output_dir|--env_names_yaml) expect=path ;;
        esac
        new_args+=("$arg")
    done
    set -- "${new_args[@]}"

    # Pull --weights <value> out of the argument list; the eval scripts take --policy/--vecnorm.
    local weights_step="" skip=0
    new_args=()
    for arg in "$@"; do
        if [ "$skip" -eq 1 ]; then weights_step="$arg"; skip=0; continue; fi
        if [ "$arg" = "--weights" ]; then skip=1; continue; fi
        new_args+=("$arg")
    done
    set -- "${new_args[@]}"

    local SIF=$WS/containers/isaaclab_optuna.sif
    mkdir -p $WS/logs $WS/logs/wandb $WS/logs/isaac_cache $WS/logs/isaac_logs $WS/logs/isaac_data
    # Stale Kit lock files from a previously crashed run keep Isaac Sim from starting.
    rm -rf $WS/logs/isaac_cache/Kit 2>/dev/null || true

    # Isaac Sim's renderer needs GL libraries the container does not ship.
    local HOST_LIBS=$WS/logs/host_libs
    mkdir -p "$HOST_LIBS"
    cp -u /usr/lib/x86_64-linux-gnu/libGLU.so.1 "$HOST_LIBS/" 2>/dev/null || true
    cp -u /usr/lib/x86_64-linux-gnu/libXt.so.6 "$HOST_LIBS/" 2>/dev/null || true

    cd $WS || { echo "Failed to cd into $WS"; exit 1; }

    RESOLVED_WEIGHT_ARGS=()
    _resolve_weight_args "$HOST_RUN_DIR/weights" "$weights_step"
    set -- "${RESOLVED_WEIGHT_ARGS[@]}" "$@"

    echo "========================================================================"
    echo "$(basename "$0") — $eval_module"
    echo "Run directory (container path): $RUN_DIR"
    echo "Extra args: $@"
    echo "========================================================================"

    local SECRETS_FILE="${WS}/secrets/wandb.env"
    if [[ -f "$SECRETS_FILE" ]]; then
        chmod 600 "$SECRETS_FILE"
        source "$SECRETS_FILE"
        echo "W&B project : ${WANDB_PROJECT}"
    else
        echo "ERROR: ${SECRETS_FILE} not found — create it with WANDB_API_KEY=..."
        exit 1
    fi

    set +e
    apptainer exec --nv \
        --bind $WS:/ws \
        --bind "$WS/src/FTR-Benchmark":/local/flipper_training/src/FTR-Benchmark \
        --bind "$WS/src/flipper_training":/local/flipper_training/src/flipper_training \
        --bind "$WS/logs/isaac_cache":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/cache \
        --bind "$WS/logs/isaac_logs":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/logs \
        --bind "$WS/logs/isaac_data":/opt/conda/envs/isaaclab/lib/python3.10/site-packages/omni/data \
        --bind "$HOST_LIBS":/host_libs \
        --env OMNI_KIT_ACCEPT_EULA=Y \
        --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env WANDB_API_KEY=${WANDB_API_KEY} \
        --env WANDB_PROJECT=${WANDB_PROJECT} \
        --env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
        --env LD_LIBRARY_PATH=/host_libs:\$LD_LIBRARY_PATH \
        $SIF \
        conda run -n isaaclab --no-capture-output \
        env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
        python -m marv_rl_training.training.${eval_module} \
        --rundir "$RUN_DIR" \
        --max_steps 2000 \
        "$@"

    local exit_status=$?
    echo "========================================================================"
    echo "$(basename "$0") finished with exit status: $exit_status"
    echo "========================================================================"
    exit $exit_status
}
