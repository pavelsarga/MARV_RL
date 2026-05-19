#!/bin/bash

# SLURM array job — one task = one Optuna trial.
# Submit:  sbatch --array=0-150%10 slurm/optuna_ftr.sbatch
#   or:    sbatch slurm/optuna_ftr.sbatch   (uses the default array range below)
#
# The array range should be >= num_trials in configs/optuna_ftr.yaml.
# Extra tasks exit cleanly once the study's trial budget is exhausted.
#
# Prerequisites (run once from the login node before submitting):
#   bash scripts/setup_optuna_db.sh --write-yaml
#   # Ensure db_host in optuna_db.yaml (workspace root) is the login node hostname.
#
#SBATCH --array=0-149%8
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
#SBATCH --time=24:00:00
#SBATCH --job-name=optuna_ftr
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%A/%x_%A_%a/%x_%A_%a.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%A/%x_%A_%a/%x_%A_%a.err

echo "########################################################################"
echo "Job array ID : $SLURM_ARRAY_JOB_ID"
echo "Array task   : $SLURM_ARRAY_TASK_ID"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Submit host  : $SLURM_SUBMIT_HOST"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "########################################################################"

# GPU infrastructure failures just exit — failed trials don't count toward
# the Optuna budget, so remaining/new array tasks fill in naturally.

WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif

TRAIN_CONFIG=$WS/configs/ftr_config_A_new.yaml
OPTUNA_CONFIG=$WS/configs/optuna_ftr_smooth.yaml

PG_DATA_DIR="${HOME}/postgres_optuna"
PG_LOG="${PG_DATA_DIR}/postgres.log"

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
TASKDIR=$LOGDIR/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TASKDIR" "$WS/logs/wandb"
cp "$TRAIN_CONFIG"                              "$LOGDIR/" 2>/dev/null || true
cp "$OPTUNA_CONFIG"                             "$LOGDIR/" 2>/dev/null || true

cd "$WS" || { echo "Failed to cd into $WS"; exit 1; }

# ── GPU health check ──────────────────────────────────────────────────────────
# Catches two failure modes:
#   1. GPU invisible to the OS (previous crashed job left it in bad state)
#   2. Active CUDA contexts still running (node not fully cleaned up)
nvidia-smi || { echo "nvidia-smi failed — GPU driver unreachable"; exit 1; }
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
    echo "GPU has ${GPU_PROCS} active CUDA context(s) — node not clean"
    exit 1
fi
echo "GPU is clean."

# ── network check (W&B hangs forever if DNS is broken) ───────────────────────
if ! getent hosts api.wandb.ai > /dev/null 2>&1; then
    echo "WARNING: Cannot resolve api.wandb.ai — running W&B in offline mode."
    export WANDB_MODE=offline
fi

# ── per-job Isaac Sim shader cache ────────────────────────────────────────────
# Isolate each trial's Omniverse cache to a node-local /tmp directory.
# Prevents NFS cache corruption from a crashed job affecting subsequent trials
# on any node (all nodes share the same ~/ via NFS).
LOCAL_CACHE="/tmp/omni_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Copying Omniverse cache to node-local ${LOCAL_CACHE} ..."
rsync -a --no-perms "${HOME}/.cache/ov/" "${LOCAL_CACHE}/" 2>/dev/null \
    || mkdir -p "${LOCAL_CACHE}"
trap "echo 'Cleaning up node-local cache...'; rm -rf '${LOCAL_CACHE}'" EXIT

# ── detect DB backend from optuna_db.yaml ────────────────────────────────────
DB_URL=$(grep '^url:' "$WS/optuna_db.yaml" | awk '{print $2}')
DB_HOST=$(grep '^db_host:' "$WS/optuna_db.yaml" | awk '{print $2}')
DB_PORT=$(grep '^db_port:' "$WS/optuna_db.yaml" | awk '{print $2}')

if [[ "$DB_URL" == sqlite://* ]]; then
    echo "Using SQLite database: ${DB_URL}"
    SQLITE_PATH="${DB_URL#sqlite:///}"           # strip sqlite:/// prefix
    SQLITE_HOST_PATH="${WS}${SQLITE_PATH#/ws}"   # /ws/... -> $WS/...
    mkdir -p "$(dirname "$SQLITE_HOST_PATH")"
    echo "SQLite file will be at: ${SQLITE_HOST_PATH}"

    # Pre-initialise the schema from a single process (task 0 only) so that all
    # array tasks find the tables already created and don't race on CREATE TABLE.
    if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
        echo "Task 0: pre-initialising Optuna SQLite schema..."
        apptainer exec --nv \
            --bind $WS:/ws \
            $SIF \
            conda run -n isaaclab --no-capture-output \
            env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
            python -c "
import optuna, pathlib
from optuna.storages import RDBStorage
pathlib.Path('${SQLITE_HOST_PATH}').parent.mkdir(parents=True, exist_ok=True)
RDBStorage('${DB_URL}', engine_kwargs={'connect_args': {'timeout': 300}})
print('Schema ready.')
" && echo "Schema initialised." || echo "WARNING: schema pre-init failed (may already exist — continuing)."
    fi
    # All tasks: give task 0 time to finish schema creation before proceeding.
    sleep $((SLURM_ARRAY_TASK_ID % 10 + 2))
else
    # PostgreSQL mode — start server and wait for it to be reachable
    if [[ -z "$DB_HOST" || -z "$DB_PORT" ]]; then
        echo "ERROR: optuna_db.yaml must have either 'url' (SQLite) or 'db_host'+'db_port' (PostgreSQL)"
        exit 1
    fi
    echo "PostgreSQL target: ${DB_HOST}:${DB_PORT}"

    # task 0 only: ensure postgres is running on the submit (login) node
    if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
        echo "Task 0: checking/starting PostgreSQL on ${SLURM_SUBMIT_HOST} ..."
        ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
            "${SLURM_SUBMIT_HOST}" \
            "source /etc/profile 2>/dev/null; module load PostgreSQL/17.5-GCCcore-14.3.0 2>/dev/null;
             pg_ctl -D '${PG_DATA_DIR}' status > /dev/null 2>&1 \
                 || pg_ctl -D '${PG_DATA_DIR}' -l '${PG_LOG}' start;
             sleep 2" \
            && echo "PostgreSQL running on ${SLURM_SUBMIT_HOST}." \
            || echo "WARNING: could not verify/start PostgreSQL — tasks may fail to connect."
    fi

    # all tasks: wait until postgres is reachable (up to 60 s)
    for i in $(seq 1 12); do
        if bash -c "exec 3<>/dev/tcp/${DB_HOST}/${DB_PORT}" 2>/dev/null; then
            exec 3>&-
            echo "PostgreSQL is reachable."
            break
        fi
        echo "  Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT} (attempt ${i}/12) ..."
        sleep 5
        if [[ $i -eq 12 ]]; then
            echo "ERROR: PostgreSQL at ${DB_HOST}:${DB_PORT} did not become ready within 60 s."
            exit 1
        fi
    done
fi

# ── W&B credentials ───────────────────────────────────────────────────────────
# Edit secrets/wandb.env to change project or rotate the API key.
# chmod 600 is set automatically below.
SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
    echo "W&B project : ${WANDB_PROJECT}"
else
    echo "ERROR: ${SECRETS_FILE} not found — upload it to the cluster before submitting."
    exit 1
fi

# ── run ───────────────────────────────────────────────────────────────────────
COMMAND="apptainer exec --nv \
    --bind $WS:/ws \
    --bind ${LOCAL_CACHE}:${HOME}/.cache/ov \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env WANDB_PROJECT=${WANDB_PROJECT} \
    --env WANDB_DIR=/ws/logs/wandb \
    --env WANDB_INIT_TIMEOUT=60 \
    --env WANDB__NETWORK_BUFFER=60 \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/optuna_train_ftr.py \
    --train_config  $TRAIN_CONFIG \
    --optuna_config $OPTUNA_CONFIG \
    --headless $@"

echo "Executing:"
echo "$COMMAND"
echo "------------------------------------------------------------------------"

srun $COMMAND
EXIT_STATUS=$?

echo "########################################################################"
echo "Task $SLURM_ARRAY_TASK_ID finished with exit status: $EXIT_STATUS"
echo "########################################################################"

# ── post-run: mark trial as FAIL if the process crashed ──────────────────────
# Handles cases where Isaac Sim crashes at the C level before Python can update
# the DB.  Matches by logpath user attribute so only this task's trial is touched.
if [ "$EXIT_STATUS" -ne 0 ]; then
    SQLITE_HOST_PATH="${WS}/optuna/optuna.db"
    LOGPATH_PAT="%optuna_ftr_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}%"
    if [ -f "$SQLITE_HOST_PATH" ]; then
        N=$(sqlite3 "$SQLITE_HOST_PATH" \
            "UPDATE trials \
             SET state='FAIL', datetime_complete='$(date -u +"%Y-%m-%d %H:%M:%S")' \
             WHERE state='RUNNING' \
               AND trial_id IN ( \
                 SELECT trial_id FROM trial_user_attributes \
                 WHERE key='logpath' AND value_json LIKE '${LOGPATH_PAT}' \
               ); \
             SELECT changes();" 2>/dev/null || echo 0)
        echo "Post-run DB cleanup: ${N} trial(s) marked FAIL (exit=${EXIT_STATUS})"
    fi
fi

exit $EXIT_STATUS
