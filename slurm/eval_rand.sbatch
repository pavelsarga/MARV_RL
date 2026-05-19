#!/bin/bash
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_rand
#SBATCH --output=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.out
#SBATCH --error=/home/sargapav/robot_rodeo_gym_ws/logs/%x_%j/%x_%j.err

# Evaluate the random policy baseline on the cluster.
#
# Usage:
#   sbatch slurm/eval_rand.sbatch <config.yaml> [eval_ftr_rand.py args...]
#
# Examples:
#   sbatch slurm/eval_rand.sbatch configs/rand_policy_eval.yaml
#   sbatch slurm/eval_rand.sbatch configs/rand_policy_eval.yaml --num_envs 64 --repeats 5
#   sbatch slurm/eval_rand.sbatch configs/rand_policy_eval.yaml policy_opts.linear_speed=0.5
#
# Per-env-type CSV output is saved automatically to the SLURM log directory.
# Override with --output_dir <path> (host path; auto-rewritten by eval_rand.sh):
#   sbatch slurm/eval_rand.sbatch configs/rand_policy_eval.yaml --num_env_types 16 --repeats 10
#   sbatch slurm/eval_rand.sbatch configs/rand_policy_eval.yaml --output_dir /custom/path --eval_id label
#   sbatch slurm/eval_rand.sbatch configs/rand_policy_eval.yaml --env_names_yaml configs/env_names.yaml

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Script arguments: $@"
echo "########################################################################"

WS=/home/sargapav/robot_rodeo_gym_ws

LOGDIR=$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

cd $WS || { echo "Failed to cd into $WS"; exit 1; }
# ── GPU health check ────────────────────────────────────────────────────────
nvidia-smi || { echo "nvidia-smi failed — GPU driver unreachable"; exit 1; }

# ── Execution ───────────────────────────────────────────────────────────────
# Inject --accel_out into the job log directory unless the caller already set it.
CONFIG="$1"; 
cp $WS/$CONFIG $LOGDIR/

if echo "$@" | grep -q "\-\-accel_out"; then
    ACCEL_ARGS=()
else
    ACCEL_ARGS=(--accel_out "$LOGDIR/raw_accel.npz")
fi

# Auto-inject --output_dir to the SLURM log directory unless the caller set it.
if echo "$@" | grep -q "\-\-output_dir"; then
    OUTPUT_ARGS=()
else
    OUTPUT_ARGS=(--output_dir "$LOGDIR")
    echo "Auto output_dir: $LOGDIR"
fi

echo "Executing: scripts/eval_rand.sh $CONFIG --headless ${ACCEL_ARGS[*]} ${OUTPUT_ARGS[*]} $@"
echo "------------------------------------------------------------------------"

srun bash $WS/scripts/eval_rand.sh "$@" "${ACCEL_ARGS[@]}" "${OUTPUT_ARGS[@]}" --headless

EXIT_STATUS=$?
echo "########################################################################"
echo "eval_rand finished with exit status: $EXIT_STATUS"
echo "########################################################################"
exit $EXIT_STATUS
