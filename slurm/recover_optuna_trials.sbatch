#!/bin/bash
# Recover cancelled Optuna trials by running eval on their saved weights
# and submitting results to the Optuna DB.
#
# Each array task recovers ONE trial.  The task index maps to the trial number:
#   TRIAL_NUM = TRIAL_OFFSET + SLURM_ARRAY_TASK_ID
#
# Submit:
#   sbatch slurm/recover_optuna_trials.sbatch          # uses default array below
#   sbatch --array=0-27 slurm/recover_optuna_trials.sbatch
#
# Adjust TRIAL_OFFSET below to match the first trial number to recover.
#
#SBATCH --array=0-27
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
#SBATCH --time=10:00:00
#SBATCH --job-name=recover_optuna
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%A/%x_%A_%a/%x_%A_%a.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%A/%x_%A_%a/%x_%A_%a.err

echo "########################################################################"
echo "Job array ID : $SLURM_ARRAY_JOB_ID"
echo "Array task   : $SLURM_ARRAY_TASK_ID"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "########################################################################"

# First Optuna trial number to recover.  Task 0 → trial TRIAL_OFFSET, etc.
# Override without editing this file:  sbatch --export=ALL,TRIAL_OFFSET=75 ...
TRIAL_OFFSET=${TRIAL_OFFSET:-39}
TRIAL_NUM=$((TRIAL_OFFSET + SLURM_ARRAY_TASK_ID))

WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif
# Override without editing:  sbatch --export=ALL,OPTUNA_CONFIG=/ws/configs/optuna_ftr_smooth.yaml ...
OPTUNA_CONFIG=${OPTUNA_CONFIG:-$WS/configs/optuna_ftr.yaml}

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
TASKDIR=$LOGDIR/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TASKDIR" "$WS/logs/wandb"

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

# ── run ───────────────────────────────────────────────────────────────────────
echo "Recovering trial ${TRIAL_NUM} (array task ${SLURM_ARRAY_TASK_ID})"

srun apptainer exec --nv \
    --bind $WS:/ws \
    --bind ${LOCAL_CACHE}:${HOME}/.cache/ov \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python -m flipper_training.experiments.ppo.recover_optuna_trials \
    --trial_num "$TRIAL_NUM" \
    --optuna_config "$OPTUNA_CONFIG" \
    --headless

EXIT_STATUS=$?
echo "########################################################################"
echo "Trial ${TRIAL_NUM} (task ${SLURM_ARRAY_TASK_ID}) finished — exit ${EXIT_STATUS}"
echo "########################################################################"
exit $EXIT_STATUS
