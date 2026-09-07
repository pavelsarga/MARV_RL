#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_auto
#SBATCH --output=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/mnt/personal/sargapav/MARV_RL_ws/logs/%x_%j/%x_%j.err

# Auto-dispatching eval job: reads the run's saved config.yaml (detection shared with
# scripts/eval_auto.sh via scripts/lib/eval_target.sh) and dispatches from
# config.yaml and runs the matching eval_*.py, so the same submit command works for any
# trained module — no need to remember/pick eval_ftr.sbatch vs eval_d3qn.sbatch vs
# eval_creps.sbatch vs eval_sac.sbatch by hand. Same underlying logic as scripts/eval_auto.sh
# (path rewriting, weights auto-detection, --output_dir default), just as a single SLURM job
# instead of chaining sbatch calls.
#
# Module -> eval script mapping (rl_modules/registry.py):
#   top-level prediction_horizon + execution_horizon -> eval_diffusion.py
#     (wins over module_name: these runs use module_name: marv_rl but parse into
#      FtrDiffusionConfig, and eval_ftr.py dies on prediction_horizon at parse time)
#   marv_rl, hfc, mitriakov  -> eval_ftr.py   (policy_final.pth / vecnorm_final.pth)
#   atd3qn, icmd3qn          -> eval_d3qn.py  (policy_final.pth / vecnorm_final.pth)
#   creps                    -> eval_creps.py (creps_state_final.pth, no vecnorm)
#   ctrac                    -> eval_sac.py   (policy_final.pth / vecnorm_final.pth)
#
# Usage:
#   sbatch slurm/eval_auto.sbatch --rundir <run_dir> [eval_*.py args...]
#
# <run_dir> must be an absolute host path or /ws/... container path.
# Paths under $WS are automatically rewritten to /ws/... inside the container.
#
# Examples:
#   sbatch slurm/eval_auto.sbatch --rundir /ws/experiments/my_ppo_run
#   sbatch slurm/eval_auto.sbatch --rundir /ws/logs/train_marv_atd3qn_.../attempt_0 --num_envs 256 --repeats 30
#   sbatch slurm/eval_auto.sbatch --rundir /ws/experiments/baselines/creps/train_marv_creps_.../attempt_0
#   sbatch slurm/eval_auto.sbatch --rundir /ws/logs/train_ctrac_.../attempt_0 --output_dir /ws/logs/policy_eval

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
    [[ "$arg" == "--rundir" ]] && _next_is_rundir=1
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

# ── Detect module_name from the run's saved config.yaml (read on the host, before
# entering the container) ────────────────────────────────────────────────────
HOST_RUN_DIR="$WS/${RUN_DIR#/ws/}"
CONFIG_FILE="$HOST_RUN_DIR/config.yaml"
[ -f "$CONFIG_FILE" ] || { echo "ERROR: no config.yaml found at $CONFIG_FILE"; exit 1; }

source "$WS/scripts/lib/eval_target.sh"
detect_eval_target "$CONFIG_FILE" || exit 1
echo "Detected module_name: $MODULE_NAME  (eval kind: $EVAL_KIND)"

# ── Map the detected kind -> target eval script + checkpoint naming convention ────
HAS_VECNORM=1
TRAINING=/ws/src/flipper_training/marv_rl_training/training
case "$EVAL_KIND" in
    ppo)
        TARGET_PY=$TRAINING/eval_ftr.py
        FINAL_POLICY=policy_final.pth; FINAL_VECNORM=vecnorm_final.pth
        STEP_POLICY_GLOB="policy_step_*.pth"; STEP_VECNORM_GLOB="vecnorm_step_*.pth"
        ;;
    d3qn)
        TARGET_PY=$TRAINING/eval_d3qn.py
        FINAL_POLICY=policy_final.pth; FINAL_VECNORM=vecnorm_final.pth
        STEP_POLICY_GLOB="policy_step_*.pth"; STEP_VECNORM_GLOB="vecnorm_step_*.pth"
        ;;
    creps)
        TARGET_PY=$TRAINING/eval_creps.py
        FINAL_POLICY=creps_state_final.pth
        STEP_POLICY_GLOB="creps_state_step_*.pth"
        HAS_VECNORM=0
        ;;
    sac)
        TARGET_PY=$TRAINING/eval_sac.py
        FINAL_POLICY=policy_final.pth; FINAL_VECNORM=vecnorm_final.pth
        STEP_POLICY_GLOB="policy_step_*.pth"; STEP_VECNORM_GLOB="vecnorm_step_*.pth"
        ;;
    diffusion)
        # Same checkpoint naming as PPO — train_diffusion.py is a fork of train_ftr.py and
        # writes policy_/vecnorm_ pairs identically; only the entry point differs.
        TARGET_PY=$TRAINING/eval_diffusion.py
        FINAL_POLICY=policy_final.pth; FINAL_VECNORM=vecnorm_final.pth
        STEP_POLICY_GLOB="policy_step_*.pth"; STEP_VECNORM_GLOB="vecnorm_step_*.pth"
        ;;
    *)
        echo "ERROR: detect_eval_target returned unknown kind '$EVAL_KIND'"
        exit 1
        ;;
esac
echo "Dispatching to: $TARGET_PY"

# ── Auto-detect weights: use the *_final.pth checkpoint(s) if present, else the
# highest-numbered *_step_*.pth (mirrors each module's own eval_*.sh). Skipped if the
# caller already passed --policy explicitly. ──
if [ "$CALLER_SET_POLICY" -eq 0 ]; then
    HOST_WEIGHTS_DIR="$WS/${RUN_DIR#/ws/}/weights"
    if [ -f "$HOST_WEIGHTS_DIR/$FINAL_POLICY" ]; then
        echo "Weights: $FINAL_POLICY$([ "$HAS_VECNORM" -eq 1 ] && echo "  +  $FINAL_VECNORM")"
    else
        LATEST_POLICY=$(ls -v "$HOST_WEIGHTS_DIR"/$STEP_POLICY_GLOB 2>/dev/null | tail -1)
        if [ -n "$LATEST_POLICY" ]; then
            POLICY_FILE=$(basename "$LATEST_POLICY")
            echo "No $FINAL_POLICY found — auto-selecting latest checkpoint: $POLICY_FILE"
            if [ "$HAS_VECNORM" -eq 1 ]; then
                LATEST_VECNORM=$(ls -v "$HOST_WEIGHTS_DIR"/$STEP_VECNORM_GLOB 2>/dev/null | tail -1)
                if [ -n "$LATEST_VECNORM" ]; then
                    VECNORM_FILE=$(basename "$LATEST_VECNORM")
                    set -- "--policy" "$POLICY_FILE" "--vecnorm" "$VECNORM_FILE" "$@"
                else
                    set -- "--policy" "$POLICY_FILE" "$@"
                fi
            else
                set -- "--policy" "$POLICY_FILE" "$@"
            fi
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
    --bind $WS/experiments/eval_top3_averaged/weights:/ws/experiments/eval_top3_averaged/weights \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    $SIF \
    conda run -n isaaclab --no-capture-output \
    env PYTHONPATH=/ws/src/FTR-Benchmark:/ws/src/flipper_training \
    python $TARGET_PY \
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
