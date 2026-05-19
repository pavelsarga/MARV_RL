#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH --time=10:00:00
#SBATCH --job-name=eval_optuna
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.err

# Evaluate top-K Optuna trials and compare their performance.
#
# Usage:
#   sbatch slurm/eval_optuna_top.sbatch --study ftr_v3_stage1_rewards
#   sbatch slurm/eval_optuna_top.sbatch --study ftr_v3_stage1_rewards --top 5 --repeats 50
#
# All arguments after the script name are forwarded to eval_optuna_top.py.

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Script arguments: $@"
echo "########################################################################"

WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

# ── GPU health check ────────────────────────────────────────────────────────
nvidia-smi || { echo "nvidia-smi failed — GPU driver unreachable"; exit 1; }

# ── DB connectivity ─────────────────────────────────────────────────────────
DB_HOST=$(grep '^db_host:' "$WS/optuna_db.yaml" | awk '{print $2}')
DB_PORT=$(grep '^db_port:' "$WS/optuna_db.yaml" | awk '{print $2}')
DB_URL=$(grep '^url:' "$WS/optuna_db.yaml" | awk '{print $2}')

if [[ -n "$DB_HOST" && -n "$DB_PORT" ]]; then
    echo "Checking PostgreSQL at ${DB_HOST}:${DB_PORT} ..."
    for i in $(seq 1 6); do
        if bash -c "exec 3<>/dev/tcp/${DB_HOST}/${DB_PORT}" 2>/dev/null; then
            exec 3>&-
            echo "PostgreSQL is reachable."
            break
        fi
        echo "  Waiting (attempt ${i}/6) ..."
        sleep 5
        if [[ $i -eq 6 ]]; then
            echo "ERROR: PostgreSQL not reachable within 30 s."
            exit 1
        fi
    done
elif [[ "$DB_URL" == sqlite://* ]]; then
    echo "Using SQLite: ${DB_URL}"
fi

# ── Execution ───────────────────────────────────────────────────────────────
COMMAND="apptainer exec --nv \
    --bind $WS:/ws \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/eval_optuna_top.py \
    --num_envs 512 \
    --headless $@"

echo "Executing:"
echo "$COMMAND"
echo "------------------------------------------------------------------------"

srun $COMMAND

EXIT_STATUS=$?
echo "########################################################################"
echo "Eval finished with exit status: $EXIT_STATUS"
echo "########################################################################"
exit $EXIT_STATUS
