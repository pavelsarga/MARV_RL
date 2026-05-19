#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_ftr
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.err

# Evaluate a trained FTR policy from a run directory.
#
# Usage:
#   sbatch slurm/eval_ftr.sbatch --rundir <run_dir> [eval_ftr.py args...]
#
# <run_dir> must be an absolute host path or /ws/... container path.
# Paths under $WS are automatically rewritten to /ws/... inside the container.
#
# Examples:
#   sbatch slurm/eval_ftr.sbatch --rundir /home/sargapav/robot_rodeo_gym_ws/experiments/my_run
#   sbatch slurm/eval_ftr.sbatch --rundir /ws/experiments/my_run --repeats 20
#   sbatch slurm/eval_ftr.sbatch --rundir /ws/experiments/my_run --num_envs 256 --map cur_mixed
#   sbatch slurm/eval_ftr.sbatch --rundir /ws/experiments/my_run log_raw_accel=true
#
# Per-env-type CSV output is saved automatically to the SLURM log directory.
# Override with --output_dir (host or /ws/ path, auto-rewritten):
#   sbatch slurm/eval_ftr.sbatch --rundir /ws/experiments/my_run --repeats 10
#   sbatch slurm/eval_ftr.sbatch --rundir /ws/experiments/my_run --output_dir /ws/experiments --eval_id label

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

# ── Rewrite host paths to container mount and auto-inject --output_dir ──────
# Container mounts $WS as /ws; rewrite any path under $WS to /ws/...
_rewrite_ws() {
    local p="$1"
    if [[ "$p" != /ws/* && "$p" == "${WS}"* ]]; then
        p="${p#${WS}/}"
        p="/ws/${p#/}"
    fi
    echo "$p"
}

# Scan args: rewrite --rundir, --output_dir, --env_names_yaml values to
# container paths; detect if --output_dir was explicitly set by the caller.
CALLER_SET_OUTPUT_DIR=0
NEW_ARGS=()
_next_is_rundir=0
_next_is_output_dir=0
_next_is_env_names=0
for arg in "$@"; do
    if [ "$_next_is_rundir" -eq 1 ]; then
        NEW_ARGS+=("$(_rewrite_ws "$arg")")
        _next_is_rundir=0
        continue
    fi
    if [ "$_next_is_output_dir" -eq 1 ]; then
        NEW_ARGS+=("$(_rewrite_ws "$arg")")
        _next_is_output_dir=0
        continue
    fi
    if [ "$_next_is_env_names" -eq 1 ]; then
        NEW_ARGS+=("$(_rewrite_ws "$arg")")
        _next_is_env_names=0
        continue
    fi
    [[ "$arg" == "--rundir" ]]       && _next_is_rundir=1
    if [[ "$arg" == "--output_dir" ]]; then
        CALLER_SET_OUTPUT_DIR=1
        _next_is_output_dir=1
    fi
    [[ "$arg" == "--env_names_yaml" ]] && _next_is_env_names=1
    NEW_ARGS+=("$arg")
done
set -- "${NEW_ARGS[@]}"

# Default --output_dir to the SLURM log directory (container path)
if [ "$CALLER_SET_OUTPUT_DIR" -eq 0 ]; then
    LOGDIR_CONTAINER="/ws/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
    set -- "--output_dir" "$LOGDIR_CONTAINER" "$@"
    echo "Auto output_dir: $LOGDIR_CONTAINER"
fi

# ── Execution ───────────────────────────────────────────────────────────────
COMMAND="apptainer exec --nv \
    --bind $WS:/ws \
    --bind $WS/experiments/eval_top3_averaged/weights:/ws/experiments/eval_top3_averaged/weights \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/flipper_training/experiments/ppo/eval_ftr.py \
    --headless --num_env_types 16 --max_steps 2000 $@"

echo "Executing:"
echo "$COMMAND"
echo "------------------------------------------------------------------------"

srun $COMMAND

EXIT_STATUS=$?
echo "########################################################################"
echo "Eval finished with exit status: $EXIT_STATUS"
echo "########################################################################"
exit $EXIT_STATUS
