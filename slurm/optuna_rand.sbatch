#!/bin/bash

# SLURM array job for random-policy Optuna search — one task = one trial.
# Submit:  sbatch --array=0-79%8 slurm/optuna_rand.sbatch
#   or:    sbatch slurm/optuna_rand.sbatch  (uses the default array range below)
#
# The array range should be >= num_trials in configs/optuna_rand_policy.yaml.
# Extra tasks exit cleanly once the trial budget is exhausted.
#
#SBATCH --array=0-79%8
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
#SBATCH --time=02:00:00
#SBATCH --job-name=optuna_rand
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

WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif

EVAL_CONFIG=$WS/configs/rand_policy_eval.yaml
OPTUNA_CONFIG=$WS/configs/optuna_rand_policy.yaml

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
TASKDIR=$LOGDIR/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TASKDIR"
cp "$EVAL_CONFIG"    "$LOGDIR/" 2>/dev/null || true
cp "$OPTUNA_CONFIG"  "$LOGDIR/" 2>/dev/null || true

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

# ── per-job Isaac Sim shader cache ────────────────────────────────────────────
LOCAL_CACHE="/tmp/omni_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Copying Omniverse cache to node-local ${LOCAL_CACHE} ..."
rsync -a --no-perms "${HOME}/.cache/ov/" "${LOCAL_CACHE}/" 2>/dev/null \
    || mkdir -p "${LOCAL_CACHE}"
trap "echo 'Cleaning up node-local cache...'; rm -rf '${LOCAL_CACHE}'" EXIT

# ── read DB URL from optuna config ────────────────────────────────────────────
DB_URL=$(grep '^db_url:' "$OPTUNA_CONFIG" | awk '{print $2}' | tr -d '"')

if [[ "$DB_URL" == sqlite://* ]]; then
    echo "Using SQLite database: ${DB_URL}"
    # Resolve /ws/... prefix to the actual workspace path on the host
    SQLITE_PATH="${DB_URL#sqlite:///}"
    SQLITE_HOST_PATH="${WS}${SQLITE_PATH#/ws}"
    mkdir -p "$(dirname "$SQLITE_HOST_PATH")"
    echo "SQLite file: ${SQLITE_HOST_PATH}"

    # Task 0 pre-initialises the schema to prevent concurrent CREATE TABLE races
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
    # Stagger task starts so task 0 can finish schema init
    sleep $((SLURM_ARRAY_TASK_ID % 10 + 2))
else
    echo "ERROR: only SQLite is supported for the random-policy study (db_url must start with sqlite:///)"
    exit 1
fi

# ── run ───────────────────────────────────────────────────────────────────────
COMMAND="apptainer exec --nv \
    --bind $WS:/ws \
    --bind ${LOCAL_CACHE}:${HOME}/.cache/ov \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/optuna_eval_rand.py \
    --config  $EVAL_CONFIG \
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

# ── post-run: mark stuck RUNNING trial as FAIL if process crashed ─────────────
if [ "$EXIT_STATUS" -ne 0 ]; then
    if [ -f "$SQLITE_HOST_PATH" ]; then
        N=$(sqlite3 "$SQLITE_HOST_PATH" \
            "UPDATE trials \
             SET state='FAIL', datetime_complete='$(date -u +"%Y-%m-%d %H:%M:%S")' \
             WHERE state='RUNNING'; \
             SELECT changes();" 2>/dev/null || echo 0)
        echo "Post-run DB cleanup: ${N} trial(s) marked FAIL (exit=${EXIT_STATUS})"
    fi
fi

exit $EXIT_STATUS
