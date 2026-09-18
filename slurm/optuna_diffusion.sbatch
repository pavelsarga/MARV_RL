#!/bin/bash
# Receding-horizon grid (T_o x T_a x T_p) for the Phase 1 policy — one SLURM array task per
# grid point, each a full training run with crash respawn.
#
# Submit (from the git clone; the #SBATCH --output lines below cannot expand variables):
#   MARV_WS=/mnt/personal/sargapav/MARV_RL_ws_git sbatch --array=0-44%4 \
#       --output=/mnt/personal/sargapav/MARV_RL_ws_git/logs/%x_%A/%x_%A_%a/%x_%A_%a.out \
#       --error=/mnt/personal/sargapav/MARV_RL_ws_git/logs/%x_%A/%x_%A_%a/%x_%A_%a.err \
#       slurm/optuna_diffusion.sbatch
# Re-run one point whose task gave up:  the same line with --array=7
# List the points:  python -m marv_rl_training.training.horizon_grid configs/optuna/optuna_diffusion_horizons.yaml
#
# How this differs from optuna_ftr.sbatch, and why:
#   * The grid point is a pure function of the array index (horizon_grid.py), forced onto the
#     trial with PartialFixedSampler — not sampled by Optuna. See horizon_grid.py for why
#     GridSampler cannot be trusted with trials that crash.
#   * Each task RESPAWNS on the exit-75 PhysX crash, exactly like train_diffusion.sbatch, and
#     resumes from its own checkpoints. optuna_ftr.sbatch does not respawn — a failed trial
#     is simply re-sampled by another task — which is fine for 6-hour PPO trials and fatal for
#     day-long diffusion runs that crash several times each.
#   * To make the trainer's single-job resume machinery (attempt_N dirs, latest_attempt_config,
#     candidate_weight_dirs) apply, the python process is given a single-job identity: the
#     array variables are hidden from it and SLURM_JOB_NAME is rewritten so RunLogger writes
#     logs/<job>_<A>/<job>_<A>_<a>_<jobid>/attempt_N — under this task's own --output dir.
#   * Frame accounting needs this task's execution_horizon, which is a grid axis, not a
#     config value: BUDGET_CHUNK_OVERRIDE feeds it to slurm/lib/respawn_common.sh.
#
# Prerequisite once, from the login node: bash scripts/setup_optuna_db.sh --write-yaml
#
#SBATCH --array=0-44%4
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpulong
#SBATCH --job-name=optuna_diffusion
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%A/%x_%A_%a/%x_%A_%a.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%A/%x_%A_%a/%x_%A_%a.err

echo "########################################################################"
echo "Job array ID : $SLURM_ARRAY_JOB_ID"
echo "Array task   : $SLURM_ARRAY_TASK_ID"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Extra args   : $@"
echo "########################################################################"

WS=${MARV_WS:-/mnt/personal/sargapav/MARV_RL_ws}
SIF=$WS/containers/isaaclab_optuna.sif
# Base training config = the best Phase 1 arm. Point this at the harder-terrain config when
# that exists; the grid axes override its horizons per task.
TRAIN_CONFIG_REL=${TRAIN_CONFIG:-diffusion/marv_config_diffusion_p1_Ta1.yaml}
OPTUNA_CONFIG_REL=${OPTUNA_CONFIG:-optuna/optuna_diffusion_horizons.yaml}
TRAIN_CONFIG_HOST=$WS/configs/$TRAIN_CONFIG_REL
OPTUNA_CONFIG_HOST=$WS/configs/$OPTUNA_CONFIG_REL
PG_DATA_DIR="${HOME}/postgres_optuna"
PG_LOG="${PG_DATA_DIR}/postgres.log"

# This task's single-job identity for the trainer (see header). LOGDIR must be exactly the
# directory RunLogger derives from these, since respawn_common reads attempt_N/ under it.
TASK_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PY_JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/${TASK_NAME}
LOGDIR=$WS/logs/${PY_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$LOGDIR" "$WS/logs/wandb"
cp "$TRAIN_CONFIG_HOST" "$OPTUNA_CONFIG_HOST" "$LOGDIR/" 2>/dev/null || true

cd "$WS" || { echo "Failed to cd into $WS"; exit 1; }

# ── GPU health check ──────────────────────────────────────────────────────────
nvidia-smi || { echo "nvidia-smi failed — GPU driver unreachable"; exit 1; }
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
    echo "GPU has ${GPU_PROCS} active CUDA context(s) — node not clean"
    exit 1
fi
echo "GPU is clean."

if ! getent hosts api.wandb.ai > /dev/null 2>&1; then
    echo "WARNING: Cannot resolve api.wandb.ai — running W&B in offline mode."
    export WANDB_MODE=offline
fi

# ── per-job Isaac Sim shader cache (node-local, so a crash cannot corrupt the NFS copy) ──
LOCAL_CACHE="/tmp/omni_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
rsync -a --no-perms "${HOME}/.cache/ov/" "${LOCAL_CACHE}/" 2>/dev/null || mkdir -p "${LOCAL_CACHE}"
trap "echo 'Cleaning up node-local cache...'; rm -rf '${LOCAL_CACHE}'" EXIT

# ── DB backend from optuna_db.yaml ───────────────────────────────────────────
DB_URL=$(grep '^url:' "$WS/optuna_db.yaml" | awk '{print $2}')
DB_HOST=$(grep '^db_host:' "$WS/optuna_db.yaml" | awk '{print $2}')
DB_PORT=$(grep '^db_port:' "$WS/optuna_db.yaml" | awk '{print $2}')
if [[ "$DB_URL" == sqlite://* ]]; then
    SQLITE_PATH="${DB_URL#sqlite:///}"
    SQLITE_HOST_PATH="${WS}${SQLITE_PATH#/ws}"
    mkdir -p "$(dirname "$SQLITE_HOST_PATH")"
    echo "SQLite database: ${SQLITE_HOST_PATH}"
    if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
        apptainer exec --nv --bind $WS:/ws $SIF conda run -n isaaclab --no-capture-output \
            env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training python -c "
import pathlib
from optuna.storages import RDBStorage
pathlib.Path('${SQLITE_HOST_PATH}').parent.mkdir(parents=True, exist_ok=True)
RDBStorage('${DB_URL}', engine_kwargs={'connect_args': {'timeout': 300}})
print('Schema ready.')
" && echo "Schema initialised." || echo "WARNING: schema pre-init failed (may already exist — continuing)."
    fi
    sleep $((SLURM_ARRAY_TASK_ID % 10 + 2))
else
    if [[ -z "$DB_HOST" || -z "$DB_PORT" ]]; then
        echo "ERROR: optuna_db.yaml must have either 'url' (SQLite) or 'db_host'+'db_port' (PostgreSQL)"; exit 1
    fi
    if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
        ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no "${SLURM_SUBMIT_HOST}" \
            "source /etc/profile 2>/dev/null; module load PostgreSQL/17.5-GCCcore-14.3.0 2>/dev/null;
             pg_ctl -D '${PG_DATA_DIR}' status > /dev/null 2>&1 || pg_ctl -D '${PG_DATA_DIR}' -l '${PG_LOG}' start; sleep 2" \
            && echo "PostgreSQL running on ${SLURM_SUBMIT_HOST}." \
            || echo "WARNING: could not verify/start PostgreSQL."
    fi
    for i in $(seq 1 12); do
        if bash -c "exec 3<>/dev/tcp/${DB_HOST}/${DB_PORT}" 2>/dev/null; then exec 3>&-; echo "PostgreSQL is reachable."; break; fi
        sleep 5
        [[ $i -eq 12 ]] && { echo "ERROR: PostgreSQL at ${DB_HOST}:${DB_PORT} not ready within 60 s."; exit 1; }
    done
fi

# ── W&B credentials ───────────────────────────────────────────────────────────
SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"; source "$SECRETS_FILE"; echo "W&B project : ${WANDB_PROJECT}"
else
    echo "ERROR: ${SECRETS_FILE} not found — upload it to the cluster before submitting."; exit 1
fi

# ── frame budget, sized with THIS task's execution_horizon ───────────────────
source $WS/slurm/lib/respawn_common.sh
CHUNK=$(apptainer exec --bind $WS:/ws $SIF conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/flipper_training python -m marv_rl_training.training.horizon_grid \
    /ws/configs/$OPTUNA_CONFIG_REL "$SLURM_ARRAY_TASK_ID" --key execution_horizon 2>/dev/null | tail -1)
case "$CHUNK" in
    ''|*[!0-9]*) echo "ERROR: could not read execution_horizon for task $SLURM_ARRAY_TASK_ID (got '$CHUNK')"; exit 1 ;;
esac
export BUDGET_CHUNK_OVERRIDE=$CHUNK
echo "Grid point $SLURM_ARRAY_TASK_ID: execution_horizon=$CHUNK"
frame_budget_init "$TRAIN_CONFIG_HOST"
STALL_LIMIT=5
STALLED=0
RESPAWN_COUNT=0

# ── execution loop (mirrors train_diffusion.sbatch) ──────────────────────────
while true; do
    REMAINING=$(frames_remaining)
    if [ -n "$REMAINING" ] && [ "$REMAINING" -le 0 ]; then
        echo "Frame budget reached: $FRAMES_DONE / $FRAME_BUDGET frames collected. Done."
        EXIT_STATUS=0
        break
    fi
    FRAME_ARGS=$(frame_overrides "$REMAINING")

    COMMAND="apptainer exec --nv \
        --bind $WS:/ws \
        --bind ${LOCAL_CACHE}:${HOME}/.cache/ov \
        --env OMNI_KIT_ACCEPT_EULA=Y \
        --env CUDA_VISIBLE_DEVICES=0 \
        --env WANDB_API_KEY=${WANDB_API_KEY} \
        --env WANDB_PROJECT=${WANDB_PROJECT} \
        --env WANDB_DIR=/ws/logs/wandb \
        --env WANDB_INIT_TIMEOUT=60 \
        --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        $SIF \
        conda run -n isaaclab --no-capture-output \
        env -u SLURM_ARRAY_JOB_ID -u SLURM_ARRAY_TASK_ID SLURM_JOB_NAME=${PY_JOB_NAME} \
            PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
        python /ws/src/flipper_training/marv_rl_training/training/optuna_train_diffusion.py \
        --train_config  /ws/configs/$TRAIN_CONFIG_REL \
        --optuna_config /ws/configs/$OPTUNA_CONFIG_REL \
        --task_id $SLURM_ARRAY_TASK_ID \
        --headless $FRAME_ARGS $@"

    echo "########################################################################"
    echo "Starting trial (respawn #$RESPAWN_COUNT — ${FRAMES_DONE}/${FRAME_BUDGET:-?} frames done, ${REMAINING:-?} to go)..."
    echo "########################################################################"
    srun $COMMAND
    EXIT_STATUS=$?
    echo "Python finished with exit status: $EXIT_STATUS"

    attempt_record_progress "$LOGDIR"
    echo "Attempt collected $ATTEMPT_PROGRESS frames (task total: $FRAMES_DONE${FRAME_BUDGET:+ / $FRAME_BUDGET})."

    if [ "$EXIT_STATUS" -eq 75 ]; then
        if [ "$ATTEMPT_PROGRESS" -gt 0 ] && attempt_wrote_weights "$ATTEMPT_DIR"; then
            STALLED=0
        else
            STALLED=$((STALLED + 1)); echo "This attempt made no progress ($STALLED in a row)."
        fi
        if [ "$STALLED" -ge "$STALL_LIMIT" ]; then
            echo "Aborting: $STALLED consecutive respawns made no progress. Resubmit --array=$SLURM_ARRAY_TASK_ID later."
            break
        fi
        RESPAWN_COUNT=$((RESPAWN_COUNT + 1))
        echo "Exit 75 (CUDA/PhysX). Respawning (attempt $((RESPAWN_COUNT + 1))) — same grid point, resuming from checkpoint..."
        sleep 5
        continue
    fi
    # 0: the trial completed and Optuna recorded it. Anything else is not a crash respawning can fix.
    break
done

# ── post-run: a C-level crash can skip Python's own FAIL marking ──────────────
if [ "$EXIT_STATUS" -ne 0 ] && [[ "$DB_URL" == sqlite://* ]] && [ -f "$SQLITE_HOST_PATH" ]; then
    LOGPATH_PAT="%${TASK_NAME}%"
    N=$(sqlite3 "$SQLITE_HOST_PATH" \
        "UPDATE trials SET state='FAIL', datetime_complete='$(date -u +"%Y-%m-%d %H:%M:%S")' \
         WHERE state='RUNNING' AND trial_id IN ( \
           SELECT trial_id FROM trial_user_attributes WHERE key='logpath' AND value_json LIKE '${LOGPATH_PAT}'); \
         SELECT changes();" 2>/dev/null || echo 0)
    echo "Post-run DB cleanup: ${N} trial(s) marked FAIL (exit=${EXIT_STATUS})"
fi
exit $EXIT_STATUS
