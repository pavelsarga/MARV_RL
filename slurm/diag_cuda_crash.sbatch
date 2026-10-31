#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --job-name=diag_cuda
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err
#
# Reproduce and diagnose the PhysX "CUDA error: an illegal memory access" crash.
#
# Resumed attempts crash ~5x sooner than fresh ones (median 2.1M vs 10.5M frames), so this
# resumes a run that crashes often (RESUME_FROM, required) and runs DIAG_ATTEMPTS attempts of
# DIAG_FRAMES extra frames each, regardless of that run's own frame budget. W&B is offline.
#
#   DIAG_BLOCKING=1       CUDA_LAUNCH_BLOCKING=1 — PhysX checks each kernel launch, so the
#                         faulting kernel is named ("GPU <kernel> fail to launch kernel!!")
#                         instead of being reported later at GpuArticulationView.cpp:749
#   DIAG_RECORDER=64      FTR_CRASH_RECORDER: control steps of state dumped to
#                         $LOGDIR/crash_recorder/ when the step raises (0 = off)
#   DIAG_ENV="K=V ..."    extra environment for the trainer
#
#   WS=/mnt/personal/sargapav/MARV_RL_ws_git
#   MARV_WS=$WS RESUME_FROM=$WS/logs/<run> sbatch --output=$WS/logs/%x_%j/%x_%j.out \
#       --error=$WS/logs/%x_%j/%x_%j.err slurm/diag_cuda_crash.sbatch --config <cfg> [overrides]

WS=${MARV_WS:-/mnt/personal/sargapav/MARV_RL_ws}
[ -n "${RESUME_FROM:-}" ] || { echo "ERROR: RESUME_FROM=<run logdir> is required"; exit 1; }
source $WS/slurm/lib/train_common.sh

_respawn_loop() {
    local attempts=${DIAG_ATTEMPTS:-4} frames=${DIAG_FRAMES:-20000000} i exit_status
    export WANDB_MODE=offline
    export CUDA_LAUNCH_BLOCKING=${DIAG_BLOCKING:-1}
    export FTR_CRASH_RECORDER=${DIAG_RECORDER:-64}
    export FTR_CRASH_RECORDER_DIR=/ws/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/crash_recorder
    local k
    for k in ${DIAG_ENV:-}; do export "$k"; done
    echo "Diag env    : CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING FTR_CRASH_RECORDER=$FTR_CRASH_RECORDER ${DIAG_ENV:-}"

    for i in $(seq 1 "$attempts"); do
        echo "########################################################################"
        echo "Diag attempt $i/$attempts ($frames frames) started $(date -Is)"
        echo "########################################################################"
        srun apptainer exec --nv \
            --bind $WS:/ws \
            --bind ${LOCAL_CACHE}:${HOME}/.cache/ov \
            --env OMNI_KIT_ACCEPT_EULA=Y \
            --env CUDA_VISIBLE_DEVICES=0 \
            --env SLURM_JOB_NAME=${SLURM_JOB_NAME} \
            --env WANDB_API_KEY=${WANDB_API_KEY} \
            --env WANDB_PROJECT=MARV_RL \
            --env WANDB_DIR=/ws/logs/wandb \
            $SIF \
            conda run -n isaaclab --no-capture-output \
            env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
            python $TARGET_PY --config /ws/configs/$CONFIG --headless total_frames=$frames ${EXTRA_ARGS[*]}
        exit_status=$?
        echo "Diag attempt $i exited $exit_status at $(date -Is)"
        nvidia-smi --query-gpu=name,driver_version,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total,retired_pages.pending --format=csv
        [ "$exit_status" -eq 75 ] || break
        sleep 5
    done
}

run_training_job "$@"
