#!/bin/bash
# Shared body of every slurm/train_*.sbatch.
#
# The per-method sbatch files exist only for their #SBATCH defaults (partition, memory, job
# name) and a default config; everything below the header — argument parsing, config/trainer
# resolution, GPU health check, W&B credentials, the node-local Omniverse cache and the
# frame-budget respawn loop — is identical for every method and lives here.
#
# Usage from an sbatch file:
#
#     WS=/mnt/personal/sargapav/MARV_RL_ws
#     DEFAULT_CONFIG=baselines/marv_config_marv_rl.yaml
#     source $WS/slurm/lib/train_common.sh
#     run_training_job "$@"
#
# Variables the caller may set before sourcing:
#   DEFAULT_CONFIG   config path relative to $WS/configs, used when --config is not given.
#                    Leave unset to make --config mandatory (train_auto.sbatch).
#   TRAIN_SCRIPT     basename of the trainer to run. Omit to derive it from the config —
#                    see scripts/lib/config_detect.sh for why that is not just module_name.
#   EXPECT_MODULE    abort when the config's module_name differs from this. Catches a
#                    config/trainer mismatch here instead of deep inside Python, after the
#                    container and Isaac Sim have already started.
#   AUTO_JOB_RENAME  1 to rename the job to train_<module_name> (train_auto.sbatch only).
#   STALL_LIMIT      consecutive no-progress respawns before giving up (default 5).
#
# Anything on the command line that is not --config is forwarded to the trainer unchanged
# (OmegaConf dotlist overrides etc.).
#
# RESUME_FROM=<old job logdir> continues a run that ended in a different SLURM job with its
# optimizer, schedules and frame counter intact — see seed_resume_from in respawn_common.sh.

source "$WS/scripts/lib/config_detect.sh"
source "$WS/slurm/lib/respawn_common.sh"

# ── --config parsing ─────────────────────────────────────────────────────────
# Sets CONFIG (relative to $WS/configs), CONFIG_HOST and EXTRA_ARGS.
_parse_train_args() {
    CONFIG="${DEFAULT_CONFIG:-}"
    EXTRA_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --config)
                [ -n "$2" ] || { echo "ERROR: --config needs a value" >&2; exit 1; }
                CONFIG="$2"; shift 2 ;;
            --config=*)
                CONFIG="${1#*=}"; shift ;;
            *)
                EXTRA_ARGS+=("$1"); shift ;;
        esac
    done

    if [ -z "$CONFIG" ]; then
        echo "ERROR: --config is required, e.g." >&2
        echo "  sbatch $0 --config baselines/marv_config_atd3qn.yaml" >&2
        exit 1
    fi

    # Accept an absolute host path, a workspace-relative one, or a configs-relative one.
    CONFIG="${CONFIG#$WS/}"
    CONFIG="${CONFIG#configs/}"
    CONFIG_HOST="$WS/configs/$CONFIG"
    if [ ! -f "$CONFIG_HOST" ]; then
        echo "ERROR: config not found: $CONFIG_HOST" >&2
        echo "Available configs (paths are relative to \$WS/configs):" >&2
        ( cd "$WS/configs" && ls baselines/*"${EXPECT_MODULE:-}"*.yaml 2>/dev/null | sed 's/^/  /' ) >&2
        exit 1
    fi
}

# ── Trainer resolution ───────────────────────────────────────────────────────
_resolve_trainer() {
    if [ -n "${TRAIN_SCRIPT:-}" ]; then
        _config_probe "$CONFIG_HOST" || exit 1
    else
        detect_train_target "$CONFIG_HOST" || exit 1
    fi

    if [ -n "${EXPECT_MODULE:-}" ] && [ -n "$MODULE_NAME" ] && [ "$MODULE_NAME" != "$EXPECT_MODULE" ]; then
        echo "ERROR: $CONFIG has module_name: $MODULE_NAME, but this sbatch runs the $EXPECT_MODULE trainer." >&2
        echo "       Use slurm/train_${MODULE_NAME}.sbatch, or slurm/train_auto.sbatch for any config." >&2
        exit 1
    fi

    TARGET_PY=/ws/src/flipper_training/marv_rl_training/training/$TRAIN_SCRIPT
    echo "Config      : $CONFIG (module_name: ${MODULE_NAME:-unset})"
    echo "Trainer     : $TRAIN_SCRIPT"
    [ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "Extra args  : ${EXTRA_ARGS[*]}"
}

# ── Job naming ───────────────────────────────────────────────────────────────
# RunLogger derives both logs/<job_name>_<job_id>/attempt_N and the W&B run name from
# $SLURM_JOB_NAME, so exporting it here is what actually places the run; `scontrol update`
# is only cosmetic (squeue/sacct). SLURM has already opened .out/.err under the submit-time
# name, so those files are moved across and the old directory becomes a symlink — moving an
# open file keeps slurmstepd's descriptor valid, so the job keeps writing into the new path.
_setup_logdir() {
    local orig_name="$SLURM_JOB_NAME"
    local orig_logdir=$WS/logs/${orig_name}_${SLURM_JOB_ID}

    if [ "${AUTO_JOB_RENAME:-0}" = "1" ] && [ -n "$MODULE_NAME" ]; then
        if [ "$orig_name" = "train_auto" ]; then
            export SLURM_JOB_NAME="train_${MODULE_NAME}"
            scontrol update JobId="$SLURM_JOB_ID" JobName="$SLURM_JOB_NAME" >/dev/null 2>&1 \
                || echo "NOTE: scontrol could not rename the job (cosmetic only)."
        else
            echo "NOTE: submitted with --job-name=$orig_name — keeping it instead of train_${MODULE_NAME}."
        fi
    fi

    LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
    mkdir -p "$LOGDIR"
    if [ "$LOGDIR" != "$orig_logdir" ] && [ -d "$orig_logdir" ] && [ ! -L "$orig_logdir" ]; then
        for f in "$orig_logdir"/*.out "$orig_logdir"/*.err; do
            [ -e "$f" ] && mv "$f" "$LOGDIR/" 2>/dev/null
        done
        if rmdir "$orig_logdir" 2>/dev/null; then
            ln -s "$LOGDIR" "$orig_logdir" 2>/dev/null
        else
            echo "NOTE: $orig_logdir was not empty — left in place."
        fi
    fi
    echo "Log directory: $LOGDIR"
    cp "$CONFIG_HOST" "$LOGDIR/"
}

# ── Node health ──────────────────────────────────────────────────────────────
_check_gpu() {
    echo "nvidia-smi output:"
    nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU may be unavailable on this node. Aborting."; exit 1; }

    # A process holding a CUDA context without memory means the GPU is shared or was left
    # dirty by a previous job; either way this run would be unreliable.
    local gpu_procs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        echo "ERROR: GPU already has $gpu_procs active CUDA context(s) — another job may share this GPU. Aborting."
        nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
        exit 1
    fi
    echo "GPU is clear (no active CUDA contexts)."
    echo "------------------------------------------------------------------------"
}

_load_wandb_credentials() {
    local secrets_file="${WS}/secrets/wandb.env"
    if [[ -f "$secrets_file" ]]; then
        chmod 600 "$secrets_file"
        source "$secrets_file"
        echo "W&B project : MARV_RL"
    else
        echo "ERROR: ${secrets_file} not found — upload it to the cluster before submitting."
        exit 1
    fi
    if ! getent hosts api.wandb.ai > /dev/null 2>&1; then
        echo "WARNING: Cannot resolve api.wandb.ai — running W&B in offline mode."
        export WANDB_MODE=offline
    fi
}

# Isolate this job's Omniverse shader cache to a node-local /tmp directory, so a crashed job
# cannot corrupt the shared NFS cache for subsequent runs.
_setup_local_cache() {
    LOCAL_CACHE="/tmp/omni_cache_${SLURM_JOB_ID}"
    echo "Copying Omniverse cache to node-local ${LOCAL_CACHE} ..."
    rsync -a --no-perms "${HOME}/.cache/ov/" "${LOCAL_CACHE}/" 2>/dev/null || mkdir -p "${LOCAL_CACHE}"
    trap "echo 'Cleaning up node-local cache...'; rm -rf '${LOCAL_CACHE}'" EXIT
}

# ── Respawn loop ─────────────────────────────────────────────────────────────
# Respawns are bounded by the config's frame budget rather than by a fixed count: after a
# crash the run restarts with only the frames still missing, until total_frames has actually
# been collected. Configs without total_frames (CREPS, which counts num_iterations) fall
# through to "respawn until a clean exit" — frame_budget_init reports which mode applies.
# STALL_LIMIT is the safety net for a failure that recurs before any progress is made;
# without it an unrecoverable crash would respawn for the rest of the SLURM allocation.
_respawn_loop() {
    frame_budget_init "$CONFIG_HOST"
    # Seed FRAMES_DONE from any attempts linked in by RESUME_FROM, so the budget accounts for
    # work already done and the trainer is asked only for what is missing. No-op on a fresh run.
    attempt_record_progress "$LOGDIR"
    [ "${FRAMES_DONE:-0}" -gt 0 ] && echo "Resuming with $FRAMES_DONE frames already collected."

    local stalled=0 respawn_count=0 remaining frame_args exit_status
    local stall_limit=${STALL_LIMIT:-5}

    while true; do
        remaining=$(frames_remaining)
        if [ -n "$remaining" ] && [ "$remaining" -le 0 ]; then
            echo "########################################################################"
            echo "Frame budget reached: $FRAMES_DONE / $FRAME_BUDGET frames collected. Done."
            echo "########################################################################"
            exit 0
        fi
        frame_args=$(frame_overrides "$remaining")

        COMMAND="apptainer exec --nv \
            --bind $WS:/ws \
            --bind ${LOCAL_CACHE}:${HOME}/.cache/ov \
            --env OMNI_KIT_ACCEPT_EULA=Y \
            --env CUDA_VISIBLE_DEVICES=0 \
            --env SLURM_JOB_NAME=${SLURM_JOB_NAME} \
            --env WANDB_API_KEY=${WANDB_API_KEY} \
            --env WANDB_PROJECT=MARV_RL \
            --env WANDB_DIR=/ws/logs/wandb \
            --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
            --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
            --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
            $SIF \
            conda run -n isaaclab --no-capture-output \
            env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
            python $TARGET_PY \
            --config /ws/configs/$CONFIG \
            --headless $frame_args ${EXTRA_ARGS[*]}"

        echo "########################################################################"
        if [ -n "$remaining" ]; then
            echo "Starting ${MODULE_NAME:-} training (respawn #$respawn_count — $FRAMES_DONE/$FRAME_BUDGET frames done, $remaining to go)..."
        else
            echo "Starting ${MODULE_NAME:-} training (respawn #$respawn_count)..."
        fi
        echo "########################################################################"

        srun $COMMAND
        exit_status=$?

        echo "########################################################################"
        echo "Python script finished with exit status: $exit_status"
        echo "########################################################################"

        attempt_record_progress "$LOGDIR"
        echo "Attempt collected $ATTEMPT_PROGRESS frames (job total: $FRAMES_DONE${FRAME_BUDGET:+ / $FRAME_BUDGET})."

        # Exit code 75 = CUDA/W&B error: transient, respawn and auto-resume from the checkpoint.
        if [ "$exit_status" -ne 75 ]; then
            # Exit 0 means the trainer collected everything it was asked for, so the budget is
            # met; anything else is a failure respawning cannot fix.
            echo "Training finished with exit code $exit_status. Not respawning."
            exit $exit_status
        fi

        if [ "$ATTEMPT_PROGRESS" -gt 0 ] && attempt_wrote_weights "$ATTEMPT_DIR"; then
            stalled=0
        else
            stalled=$((stalled + 1))
            echo "This attempt made no progress ($stalled in a row)."
        fi
        if [ "$stalled" -ge "$stall_limit" ]; then
            echo "Aborting: $stalled consecutive respawns made no progress — the failure is not transient."
            exit 75
        fi
        respawn_count=$((respawn_count + 1))
        echo "Exit code 75 detected (CUDA/W&B error). Respawning (attempt $((respawn_count + 1)))..."
        sleep 5  # brief pause before respawn to allow cleanup
    done
}

run_training_job() {
    echo "########################################################################"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node: $SLURMD_NODENAME"
    echo "Script arguments: $@"
    echo "########################################################################"

    SIF=$WS/containers/isaaclab_optuna.sif
    _parse_train_args "$@"
    _resolve_trainer
    _setup_logdir

    export WANDB_INIT_TIMEOUT=1000
    export WANDB_HTTP_TIMEOUT=1000

    cd $WS || { echo "Failed to cd into $WS"; exit 1; }
    echo "Changed directory to $(pwd)"
    echo "------------------------------------------------------------------------"

    _check_gpu
    _load_wandb_credentials
    _setup_local_cache

    [ -n "${RESUME_FROM:-}" ] && { seed_resume_from "$RESUME_FROM" "$LOGDIR" || exit 1; }

    _respawn_loop
}
