#!/usr/bin/env bash
# Local training with terrain rotation. The trainer swaps courses in-process (FtrEnv.swap_terrain);
# this loop only relaunches when it exits 76 (in-process swap impossible) — on the course named in
# <job dir>/terrain_rotation/next_course.txt — until the frame budget is met.
#
#   CONFIG=terrain_rotation/<config>.yaml TRAIN_SCRIPT=train_diffusion.py bash scripts/train_rotation.sh [train args]
#
# The config needs `terrain_rotation:` and `env_cfg_overrides.spawn_sampler:`. Runs use the SLURM
# attempt layout (logs/<job name>_<job id>/attempt_N) so that every relaunch resumes the previous
# one's checkpoint — set ROTATION_JOB_NAME / ROTATION_JOB_ID to continue an earlier local run.
# Each relaunch asks only for the frames still missing (slurm/lib/respawn_common.sh), with the
# LR / penalty schedules pinned to the original horizon.
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/slurm/lib/respawn_common.sh"

CONFIG=${CONFIG:?set CONFIG=<path under configs/>}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_diffusion.py}
export SLURM_JOB_NAME=${ROTATION_JOB_NAME:-rotation_local}
export SLURM_JOB_ID=${ROTATION_JOB_ID:-$(date +%Y%m%d%H%M%S)}
JOB_DIR="$WS/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
MAX_LAUNCHES=${MAX_LAUNCHES:-1000}
echo "terrain rotation run: $JOB_DIR (config $CONFIG, trainer $TRAIN_SCRIPT)"

frame_budget_init "$WS/configs/$CONFIG"
# Smoke runs with fewer robots than the config: TOTAL_FRAMES replaces the budget and ITER_FRAMES
# (time_steps_per_batch x robots x execution_horizon of THIS run) the iteration size the pinned
# schedule horizons are computed from — otherwise they would come from the config's num_robots.
if [ -n "${TOTAL_FRAMES:-}" ]; then FRAME_BUDGET=$TOTAL_FRAMES; fi
if [ -n "${ITER_FRAMES:-}" ]; then BUDGET_ITER_SIZE=$ITER_FRAMES; fi
if [ -n "${TOTAL_FRAMES:-}${ITER_FRAMES:-}" ] && [ -n "$FRAME_BUDGET" ] && [ -n "${BUDGET_ITER_SIZE:-}" ]; then
    BUDGET_TOTAL_ITERS=$((FRAME_BUDGET / BUDGET_ITER_SIZE))
    echo "budget override: $FRAME_BUDGET frames, iteration $BUDGET_ITER_SIZE -> $BUDGET_TOTAL_ITERS iterations"
fi
terrain_arg=()
for ((launch = 0; launch < MAX_LAUNCHES; launch++)); do
    attempt_record_progress "$JOB_DIR"
    remaining=$(frames_remaining)
    if [ -n "$remaining" ] && [ "$remaining" -le 0 ]; then
        echo "Frame budget reached: $FRAMES_DONE / $FRAME_BUDGET. Done."
        exit 0
    fi
    # the course a relaunch must start on (rewritten by every in-process swap too)
    course_file="$JOB_DIR/terrain_rotation/next_course.txt"
    [ -s "$course_file" ] && terrain_arg=(--terrain "$(tr -d '[:space:]' < "$course_file")")
    # shellcheck disable=SC2207
    frame_args=($(frame_overrides "$remaining"))
    echo "=== launch $launch: ${terrain_arg[*]:-config terrain}, ${FRAMES_DONE:-0}/${FRAME_BUDGET:-?} frames done"
    CONFIG="$CONFIG" TRAIN_SCRIPT="$TRAIN_SCRIPT" bash "$WS/scripts/train.sh" "$@" "${frame_args[@]}" "${terrain_arg[@]}"
    rc=$?
    if [ "$rc" -eq 76 ]; then
        next_file="$JOB_DIR/terrain_rotation/next_course.txt"
        [ -s "$next_file" ] || { echo "ERROR: exit 76 but $next_file is missing"; exit 1; }
        terrain_arg=(--terrain "$(tr -d '[:space:]' < "$next_file")")
        echo "=== terrain rotation: next course ${terrain_arg[1]}"
        continue
    fi
    echo "=== trainer exited $rc — stopping"
    exit "$rc"
done
echo "=== MAX_LAUNCHES ($MAX_LAUNCHES) reached"
