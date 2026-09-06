#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_creps
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# Evaluate a trained CREPS (Pecka et al. 2016) policy from a run directory.
#
# CREPS counterpart of eval_ftr.sbatch (PPO-only) / eval_d3qn.sbatch (D3QN-only):
# eval_creps.py parses the run's config into FtrCREPSConfig and builds a deterministic
# CREPSLowerLevelPolicy from the checkpoint's omega_mean (broadcast to every env — no
# sampling from omega_cov, that's only for training-time exploration).
#
# Usage:
#   sbatch slurm/eval_creps.sbatch --rundir <run_dir> [eval_creps.py args...]
#
# <run_dir> must be an absolute host path or /ws/... container path.
# Paths under $WS are automatically rewritten to /ws/... inside the container.
#
# Examples:
#   sbatch slurm/eval_creps.sbatch --rundir /mnt/personal/sargapav/MARV_RL_ws/experiments/baselines/creps/train_marv_creps_11271127
#   sbatch slurm/eval_creps.sbatch --rundir /ws/experiments/baselines/creps/train_marv_creps_11271127 --num_envs 192 --repeats 30
#   sbatch slurm/eval_creps.sbatch --rundir /ws/.../train_marv_creps_.../attempt_0 --policy creps_state_step_29.pth
#   sbatch slurm/eval_creps.sbatch --rundir /ws/.../attempt_0 --output_dir /ws/logs/policy_eval --eval_id creps_a0
#
# CREPS SLURM runs that timed out often have no creps_state_final.pth — unless the
# caller passes --policy explicitly, this script auto-selects the highest-numbered
# creps_state_step_*.pth checkpoint instead (mirrors scripts/eval_creps.sh).
#
# Per-env-type CSV output is saved automatically to the SLURM log directory.
# Override with --output_dir (host or /ws/ path, auto-rewritten).

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Script arguments: $@"
echo "########################################################################"

WS=/mnt/personal/sargapav/MARV_RL_ws
SIF=$WS/containers/isaaclab_optuna.sif

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

cd $WS || { echo "Failed to cd into $WS"; exit 1; }

# ── GPU health check ────────────────────────────────────────────────────────
nvidia-smi || { echo "nvidia-smi failed — GPU driver unreachable"; exit 1; }

# ── Rewrite host paths to container mount ───────────────────────────────────
_rewrite_ws() {
    local p="$1"
    if [[ "$p" != /ws/* && "$p" == "${WS}"* ]]; then
        p="${p#${WS}/}"
        p="/ws/${p#/}"
    fi
    echo "$p"
}

# Scan args: rewrite --rundir, --output_dir, --env_names_yaml values to
# container paths; detect if --policy/--output_dir were explicitly set by the caller.
CALLER_SET_OUTPUT_DIR=0
CALLER_SET_POLICY=0
RUN_DIR=""
NEW_ARGS=()
_next_is_rundir=0
_next_is_output_dir=0
_next_is_env_names=0
for arg in "$@"; do
    if [ "$_next_is_rundir" -eq 1 ]; then
        RUN_DIR="$(_rewrite_ws "$arg")"
        NEW_ARGS+=("$RUN_DIR")
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
    [[ "$arg" == "--rundir" ]]         && _next_is_rundir=1
    if [[ "$arg" == "--output_dir" ]]; then
        CALLER_SET_OUTPUT_DIR=1
        _next_is_output_dir=1
    fi
    [[ "$arg" == "--env_names_yaml" ]] && _next_is_env_names=1
    [[ "$arg" == "--policy" ]] && CALLER_SET_POLICY=1
    NEW_ARGS+=("$arg")
done
set -- "${NEW_ARGS[@]}"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: --rundir is required."
    exit 1
fi

# ── Auto-detect weights: use creps_state_final.pth if present, else the
# highest-numbered creps_state_step_*.pth checkpoint (mirrors scripts/eval_creps.sh).
# Skipped if the caller already passed --policy explicitly. ──
if [ "$CALLER_SET_POLICY" -eq 0 ]; then
    HOST_WEIGHTS_DIR="$WS/${RUN_DIR#/ws/}/weights"
    if [ -f "$HOST_WEIGHTS_DIR/creps_state_final.pth" ]; then
        echo "Weights: creps_state_final.pth"
    else
        LATEST_POLICY=$(ls -v "$HOST_WEIGHTS_DIR"/creps_state_step_*.pth 2>/dev/null | tail -1)
        if [ -n "$LATEST_POLICY" ]; then
            POLICY_FILE=$(basename "$LATEST_POLICY")
            echo "No creps_state_final.pth found — auto-selecting latest checkpoint: $POLICY_FILE"
            set -- "--policy" "$POLICY_FILE" "$@"
        else
            echo "WARNING: no weights found in $HOST_WEIGHTS_DIR"
        fi
    fi
fi

# Default --output_dir to the SLURM log directory (container path)
if [ "$CALLER_SET_OUTPUT_DIR" -eq 0 ]; then
    LOGDIR_CONTAINER="/ws/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
    set -- "--output_dir" "$LOGDIR_CONTAINER" "$@"
    echo "Auto output_dir: $LOGDIR_CONTAINER"
fi

# ── Execution ───────────────────────────────────────────────────────────────
COMMAND="apptainer exec --nv \
    --bind $WS:/ws \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
    python /ws/src/flipper_training/marv_rl_training/training/eval_creps.py \
    --headless --max_steps 2000 $@"

echo "Executing:"
echo "$COMMAND"
echo "------------------------------------------------------------------------"

srun $COMMAND

EXIT_STATUS=$?
echo "########################################################################"
echo "Eval finished with exit status: $EXIT_STATUS"
echo "########################################################################"
exit $EXIT_STATUS
