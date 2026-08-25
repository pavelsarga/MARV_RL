#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu
#SBATCH --job-name=train_auto
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.err

# Auto-dispatching training job: reads env_cfg_overrides.module_name from the config given
# with --config and runs the matching train_*.py, so one submit command works for every
# method — no need to pick train_marv_rl.sbatch vs train_atd3qn.sbatch vs train_ctrac.sbatch
# by hand. Same idea as eval_auto.sbatch, on the training side.
#
# Module -> trainer mapping (rl_modules/registry.py):
#   marv_rl, hfc, mitriakov  -> train_ftr.py      (PPO,      FtrPPOConfig)
#   atd3qn                   -> train_d3qn.py     (D3QN,     FtrD3QNConfig)
#   icmd3qn                  -> train_icmd3qn.py  (D3QN+ICM, FtrICMD3QNConfig)
#   ctrac                    -> train_sac.py      (SAC+C-VAE Stage II)
#   creps                    -> train_creps.py    (CREPS, iteration-counted)
#
# Naming: the job renames itself to train_<module_name> once the module is known, so the
# SLURM job name, logs/<name>_<job_id>/ directory and the W&B run all read
# train_<module>_<job_id> (RunLogger builds both from $SLURM_JOB_NAME/$SLURM_JOB_ID, see
# marv_rl_training/utils/logutils.py::_slurm_log_dir). The .out/.err files SLURM opened
# under the submit-time name are moved into that directory and the old path is left behind
# as a symlink. Passing your own --job-name at submit time disables the rename.
#
# Usage:
#   sbatch slurm/train_auto.sbatch --config configs/baselines/marv_config_atd3qn.yaml
#
# --config accepts a path relative to the workspace, relative to $WS/configs, or absolute.
# Everything that is not --config is passed through to the trainer unchanged (OmegaConf
# dotlist overrides etc.). Partition is a submit-time property, so override it there when a
# method needs a different queue than the default amdgpu:
#
#   sbatch --partition=amdgpulong slurm/train_auto.sbatch --config baselines/marv_config_marv_rl.yaml
#   sbatch --partition=amdgpufast slurm/train_auto.sbatch --config baselines/marv_config_creps.yaml
#   sbatch slurm/train_auto.sbatch --config baselines/marv_config_icmd3qn_paper.yaml num_robots=64

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Script arguments received by sbatch script ($0): $@"
echo "Number of arguments: $#"
echo "########################################################################"

# --- Environment Setup ---
echo "Setting up environment..."
WS=/home/sargapav/robot_rodeo_gym_ws
SIF=$WS/containers/isaaclab_optuna.sif

# --- Argument parsing: --config is ours, everything else goes to the trainer ---
CONFIG=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            if [ -z "$2" ]; then
                echo "ERROR: --config needs a value, e.g. --config configs/baselines/marv_config_atd3qn.yaml"
                exit 1
            fi
            CONFIG="$2"; shift 2 ;;
        --config=*)
            CONFIG="${1#*=}"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required, e.g."
    echo "  sbatch slurm/train_auto.sbatch --config configs/baselines/marv_config_atd3qn.yaml"
    exit 1
fi

# Normalise CONFIG to a path relative to $WS/configs (which is what /ws/configs/$CONFIG needs).
CONFIG="${CONFIG#$WS/}"          # absolute host path -> workspace-relative
CONFIG="${CONFIG#configs/}"      # workspace-relative -> configs-relative
CONFIG_HOST="$WS/configs/$CONFIG"
if [ ! -f "$CONFIG_HOST" ]; then
    echo "ERROR: config not found: $CONFIG_HOST"
    echo "Available configs (paths are relative to \$WS/configs):"
    ( cd "$WS/configs" && ls baselines/*.yaml 2>/dev/null | sed 's/^/  /' )
    exit 1
fi

# --- Detect module_name from the config (on the host, before the container starts) ---
# Must read env_cfg_overrides.module_name specifically: several configs mention module_name
# in their top-of-file `comment:` block, and a plain grep picks that up first.
MODULE_NAME=""
if command -v python3 >/dev/null 2>&1 && python3 -c "import yaml" >/dev/null 2>&1; then
    MODULE_NAME=$(python3 - "$CONFIG_HOST" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
print((cfg.get("env_cfg_overrides") or {}).get("module_name", ""))
PYEOF
)
else
    MODULE_NAME=$(awk '
        index($0, "env_cfg_overrides:") == 1 { inb = 1; next }
        inb && /^[^[:space:]#]/ { inb = 0 }
        inb && $0 ~ "^[[:space:]]+module_name:" {
            sub(/^[[:space:]]+module_name:[[:space:]]*/, ""); sub(/#.*$/, "");
            gsub(/["'"'"'[:space:]]/, ""); print; exit
        }
    ' "$CONFIG_HOST")
fi
if [ -z "$MODULE_NAME" ]; then
    echo "ERROR: could not determine env_cfg_overrides.module_name from $CONFIG_HOST"
    echo "       Add it to the config, or use the method's own train_*.sbatch."
    exit 1
fi
echo "Config       : $CONFIG"
echo "Detected module_name: $MODULE_NAME"

# --- Map module_name -> trainer entry point ---
case "$MODULE_NAME" in
    marv_rl|hfc|mitriakov)
        TARGET_PY=/ws/src/flipper_training/marv_rl_training/training/train_ftr.py ;;
    atd3qn)
        TARGET_PY=/ws/src/flipper_training/marv_rl_training/training/train_d3qn.py ;;
    icmd3qn)
        TARGET_PY=/ws/src/flipper_training/marv_rl_training/training/train_icmd3qn.py ;;
    ctrac)
        TARGET_PY=/ws/src/flipper_training/marv_rl_training/training/train_sac.py ;;
    creps)
        TARGET_PY=/ws/src/flipper_training/marv_rl_training/training/train_creps.py ;;
    *)
        echo "ERROR: unrecognized module_name '$MODULE_NAME' — no trainer mapping known for it."
        echo "       Known: marv_rl, hfc, mitriakov -> train_ftr.py | atd3qn -> train_d3qn.py |"
        echo "              icmd3qn -> train_icmd3qn.py | ctrac -> train_sac.py | creps -> train_creps.py"
        exit 1
        ;;
esac
echo "Dispatching to: $TARGET_PY"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra args   : ${EXTRA_ARGS[*]}"
fi

# --- Rename the job to train_<module_name> ---
# RunLogger derives both logs/<job_name>_<job_id>/attempt_N and the W&B run name from
# $SLURM_JOB_NAME, so exporting it here is what actually places the run. `scontrol update`
# is only cosmetic (squeue/sacct). SLURM already opened .out/.err under the submit-time
# name, so those files are moved across and the old directory becomes a symlink — moving an
# open file keeps slurmstepd's descriptor valid, so the job keeps writing into the new path.
ORIG_JOB_NAME="$SLURM_JOB_NAME"
ORIG_LOGDIR=$WS/logs/${ORIG_JOB_NAME}_${SLURM_JOB_ID}
if [ "$ORIG_JOB_NAME" = "train_auto" ]; then
    export SLURM_JOB_NAME="train_${MODULE_NAME}"
    scontrol update JobId="$SLURM_JOB_ID" JobName="$SLURM_JOB_NAME" >/dev/null 2>&1 \
        || echo "NOTE: scontrol could not rename the job (cosmetic only)."
else
    echo "NOTE: submitted with --job-name=$ORIG_JOB_NAME — keeping it instead of train_${MODULE_NAME}."
fi

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
if [ "$LOGDIR" != "$ORIG_LOGDIR" ] && [ -d "$ORIG_LOGDIR" ] && [ ! -L "$ORIG_LOGDIR" ]; then
    for f in "$ORIG_LOGDIR"/*.out "$ORIG_LOGDIR"/*.err; do
        [ -e "$f" ] && mv "$f" "$LOGDIR/" 2>/dev/null
    done
    if rmdir "$ORIG_LOGDIR" 2>/dev/null; then
        ln -s "$LOGDIR" "$ORIG_LOGDIR" 2>/dev/null
    else
        echo "NOTE: $ORIG_LOGDIR was not empty — left in place."
    fi
fi
echo "Log directory: $LOGDIR"
cp "$CONFIG_HOST" "$LOGDIR/"

export WANDB_INIT_TIMEOUT=1000
export WANDB_HTTP_TIMEOUT=1000

cd $WS || {
    echo "Failed to cd into $WS"
    exit 1
}
echo "Changed directory to $(pwd)"
echo "------------------------------------------------------------------------"

# --- GPU Check ---
echo "nvidia-smi output:"
nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU may be unavailable on this node. Aborting."; exit 1; }

# Check for existing CUDA contexts (processes holding a GPU context without memory)
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "ERROR: GPU already has $GPU_PROCS active CUDA context(s) — another job may share this GPU. Aborting."
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
    exit 1
fi
echo "GPU is clear (no active CUDA contexts)."
echo "------------------------------------------------------------------------"

# --- W&B credentials ---
SECRETS_FILE="${WS}/secrets/wandb.env"
if [[ -f "$SECRETS_FILE" ]]; then
    chmod 600 "$SECRETS_FILE"
    source "$SECRETS_FILE"
    echo "W&B project : MARV_RL"
else
    echo "ERROR: ${SECRETS_FILE} not found — upload it to the cluster before submitting."
    exit 1
fi
if ! getent hosts api.wandb.ai > /dev/null 2>&1; then
    echo "WARNING: Cannot resolve api.wandb.ai — running W&B in offline mode."
    export WANDB_MODE=offline
fi

# --- per-job Isaac Sim shader cache ---
# Isolate this job's Omniverse cache to a node-local /tmp directory.
# Prevents NFS cache corruption from a crashed job affecting subsequent runs.
LOCAL_CACHE="/tmp/omni_cache_${SLURM_JOB_ID}"
echo "Copying Omniverse cache to node-local ${LOCAL_CACHE} ..."
rsync -a --no-perms "${HOME}/.cache/ov/" "${LOCAL_CACHE}/" 2>/dev/null \
    || mkdir -p "${LOCAL_CACHE}"
trap "echo 'Cleaning up node-local cache...'; rm -rf '${LOCAL_CACHE}'" EXIT

# --- Respawn Configuration ---
# Identical to the per-method train_*.sbatch scripts, and module-agnostic by construction:
# respawns are bounded by the config's frame budget rather than by a fixed count, so after a
# crash the run is restarted with only the frames still missing until total_frames has
# actually been collected. Configs without total_frames (CREPS, which counts num_iterations)
# fall through to "respawn until a clean exit" — frame_budget_init reports which mode applies.
# STALL_LIMIT is the safety net for a failure that recurs before any progress is made —
# without it an unrecoverable crash would respawn for the rest of the SLURM allocation.
source $WS/slurm/lib/respawn_common.sh
frame_budget_init "$CONFIG_HOST"
STALL_LIMIT=5
STALLED=0
RESPAWN_COUNT=0

# --- Execution Loop ---
while true; do
    REMAINING=$(frames_remaining)
    if [ -n "$REMAINING" ] && [ "$REMAINING" -le 0 ]; then
        echo "########################################################################"
        echo "Frame budget reached: $FRAMES_DONE / $FRAME_BUDGET frames collected. Done."
        echo "########################################################################"
        exit 0
    fi
    FRAME_ARGS=$(frame_overrides "$REMAINING")

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
        --headless $FRAME_ARGS ${EXTRA_ARGS[*]}"

    echo "########################################################################"
    if [ -n "$REMAINING" ]; then
        echo "Starting $MODULE_NAME training (respawn #$RESPAWN_COUNT — $FRAMES_DONE/$FRAME_BUDGET frames done, $REMAINING to go)..."
    else
        echo "Starting $MODULE_NAME training (respawn #$RESPAWN_COUNT)..."
    fi
    echo "########################################################################"

    srun $COMMAND
    EXIT_STATUS=$?

    echo "########################################################################"
    echo "Python script finished with exit status: $EXIT_STATUS"
    echo "########################################################################"

    attempt_record_progress "$LOGDIR"
    echo "Attempt collected $ATTEMPT_PROGRESS frames (job total: $FRAMES_DONE${FRAME_BUDGET:+ / $FRAME_BUDGET})."

    # Exit code 75 = CUDA/W&B error, should respawn and auto-resume from crash checkpoint
    if [ "$EXIT_STATUS" -eq 75 ]; then
        if [ "$ATTEMPT_PROGRESS" -gt 0 ] && attempt_wrote_weights "$ATTEMPT_DIR"; then
            STALLED=0
        else
            STALLED=$((STALLED + 1))
            echo "This attempt made no progress ($STALLED in a row)."
        fi
        if [ "$STALLED" -ge "$STALL_LIMIT" ]; then
            echo "Aborting: $STALLED consecutive respawns made no progress — the failure is not transient."
            exit 75
        fi
        RESPAWN_COUNT=$((RESPAWN_COUNT + 1))
        echo "Exit code 75 detected (CUDA/W&B error). Respawning (attempt $((RESPAWN_COUNT + 1)))..."
        sleep 5  # Brief pause before respawn to allow cleanup
        continue
    else
        # Normal exit (0 or other non-75 code). Exit 0 means the trainer collected everything it
        # was asked for, so the budget is met; anything else is a failure respawning cannot fix.
        echo "Training finished with exit code $EXIT_STATUS. Not respawning."
        exit $EXIT_STATUS
    fi
done
