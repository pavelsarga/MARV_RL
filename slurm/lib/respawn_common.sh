# Shared respawn helpers for the slurm/train_*.sbatch scripts.
#
# A training job is respawned after a CUDA/W&B crash (exit 75) until the frame budget in its
# config is met, rather than a fixed number of times. Each respawn is launched with only the
# frames that are still missing, so the job stops as soon as `total_frames` has actually been
# collected instead of restarting the full budget every time.
#
# Source from an sbatch script, then use:
#   frame_budget_init  <config_path>            # reads total_frames / iteration size
#   frames_remaining                            # frames still to collect (empty = no budget)
#   frame_overrides    <remaining>              # OmegaConf CLI overrides for the next attempt
#   attempt_record_progress                     # call after each attempt; updates FRAMES_DONE
#
# Frame accounting comes from the checkpoint filenames the trainers write
# (`weights/policy_step_<frames>.pth`), so it needs no cooperation from the Python side.

# ── YAML reading ─────────────────────────────────────────────────────────────
# Only what the training configs actually use: top-level int scalars, optionally written as one
# of OmegaConf's arithmetic resolvers (`${mul:a,b}` and friends), which are still unresolved in
# the file on disk.
_yaml_top_scalar() {
    awk -v k="$2" 'index($0, k ":") == 1 { sub("^" k ":", ""); print; exit }' "$1"
}

_resolve_int() {
    local v
    v=$(sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$1")
    if [[ "$v" =~ ^\$\{(mul|add|sub|intdiv):([0-9]+),([0-9]+)\}$ ]]; then
        local op="${BASH_REMATCH[1]}" a="${BASH_REMATCH[2]}" b="${BASH_REMATCH[3]}"
        case "$op" in
            mul) echo $((a * b)) ;;
            add) echo $((a + b)) ;;
            sub) echo $((a - b)) ;;
            intdiv) [ "$b" -ne 0 ] && echo $((a / b)) ;;
        esac
        return 0
    fi
    [[ "$v" =~ ^[0-9]+$ ]] && { echo "$v"; return 0; }
    return 1
}

_yaml_top_int() {
    local raw
    raw=$(_yaml_top_scalar "$1" "$2") || return 1
    [ -n "$raw" ] || return 1
    _resolve_int "$raw"
}

# True when <key>: exists at the top level, whatever its value (including null).
_yaml_has_top_key() {
    awk -v k="$2" 'index($0, k ":") == 1 { found = 1; exit } END { exit(found ? 0 : 1) }' "$1"
}

# True when <block>: exists at the top level and contains a <key>: entry.
_yaml_block_has_key() {
    awk -v b="$2" -v k="$3" '
        index($0, b ":") == 1 { inb = 1; next }
        inb && /^[^[:space:]#]/ { inb = 0 }
        inb && $0 ~ "^[[:space:]]+" k ":" { found = 1; exit }
        END { exit(found ? 0 : 1) }
    ' "$1"
}

# ── Frame budget ─────────────────────────────────────────────────────────────
# FRAME_BUDGET      total frames the config asks for ("" when the trainer is not frame-based,
#                   e.g. CREPS, which counts iterations instead)
# FRAMES_DONE       frames collected by attempts of this job so far
# BUDGET_CONFIG     config the budget was read from
frame_budget_init() {
    BUDGET_CONFIG="$1"
    FRAMES_DONE=0
    FRAME_BUDGET=$(_yaml_top_int "$BUDGET_CONFIG" total_frames) || FRAME_BUDGET=""
    local steps robots
    steps=$(_yaml_top_int "$BUDGET_CONFIG" time_steps_per_batch) || steps=""
    robots=$(_yaml_top_int "$BUDGET_CONFIG" num_robots) || robots=""
    if [ -n "$FRAME_BUDGET" ] && [ -n "$steps" ] && [ -n "$robots" ] && [ "$((steps * robots))" -gt 0 ]; then
        BUDGET_ITER_SIZE=$((steps * robots))
        BUDGET_TOTAL_ITERS=$((FRAME_BUDGET / BUDGET_ITER_SIZE))
    else
        BUDGET_ITER_SIZE=""
        BUDGET_TOTAL_ITERS=""
    fi

    if [ -n "$FRAME_BUDGET" ]; then
        echo "Frame budget: ${FRAME_BUDGET} frames from ${BUDGET_CONFIG}" \
             "(iteration size ${BUDGET_ITER_SIZE:-?}, ${BUDGET_TOTAL_ITERS:-?} iterations)"
    else
        echo "No total_frames in ${BUDGET_CONFIG} — respawning until the trainer exits cleanly" \
             "instead of until a frame budget is met."
    fi
}

frames_remaining() {
    [ -n "$FRAME_BUDGET" ] || return 0
    echo $((FRAME_BUDGET - FRAMES_DONE))
}

# OmegaConf overrides for the next attempt: shrink total_frames to what is left, but keep every
# schedule (LR, step penalty, action bonus) on its ORIGINAL horizon. Those configs derive
# `total_iters` from `${total_frames}`, so without pinning them a shortened respawn would compress
# the whole schedule into the leftover frames and silently change the hyperparameters the
# checkpoint was trained with.
frame_overrides() {
    local remaining="$1"
    [ -n "$FRAME_BUDGET" ] || return 0
    local out="total_frames=${remaining}"
    if [ -n "$BUDGET_TOTAL_ITERS" ]; then
        local blk
        for blk in scheduler_opts step_penalty_scheduler action_bonus_coef_scheduler; do
            if _yaml_block_has_key "$BUDGET_CONFIG" "$blk" total_iters; then
                out="${out} ${blk}.total_iters=${BUDGET_TOTAL_ITERS}"
            fi
        done
        # The D3QN trainers fall back to total_frames // iteration_size for the epsilon-greedy decay
        # when epsilon_decay_iters is null, so that one has to be pinned explicitly too.
        if _yaml_has_top_key "$BUDGET_CONFIG" epsilon_decay_iters; then
            out="${out} epsilon_decay_iters=${BUDGET_TOTAL_ITERS}"
        fi
    fi
    echo "$out"
}

# ── Progress tracking ────────────────────────────────────────────────────────
_latest_attempt_dir() {
    ls -d "$1"/attempt_* 2>/dev/null \
        | sed 's#.*/attempt_##' | sort -n | tail -1 \
        | awk -v d="$1" 'NF { print d "/attempt_" $0 }'
}

# Highest frame count among the checkpoints an attempt wrote (0 when it wrote none).
_attempt_step_frames() {
    ls "$1"/weights/policy_step_*.pth 2>/dev/null \
        | sed 's/.*policy_step_\([0-9]*\)\.pth/\1/' | sort -n | tail -1
}

# Call after an attempt exits. Sets ATTEMPT_PROGRESS to the frames it added (0 if it produced
# nothing) and folds them into FRAMES_DONE.
#
# The trainers normally restart their frame counter at 0 on a respawn, so an attempt's own
# checkpoints count what it alone collected and the totals add up. Should the trainer manage to
# restore its training_state.pth (which also carries the cumulative frame count), the counter
# continues instead — recognisable by the attempt reporting more frames than it was asked for —
# and the number is already absolute.
#
# Attempts that die on an unhandled exception resume from policy_crash.pth, which carries no frame
# count in its name; that progress is not counted here, so the job may overshoot the budget by up
# to one save interval. Overshooting is the safe direction.
attempt_record_progress() {
    local logdir="$1" asked="$2"
    local dir step
    dir=$(_latest_attempt_dir "$logdir")
    ATTEMPT_DIR="$dir"
    ATTEMPT_PROGRESS=0
    if [ -z "$dir" ]; then
        return 0
    fi
    step=$(_attempt_step_frames "$dir")
    step=${step:-0}
    if [ -n "$asked" ] && [ "$step" -gt "$asked" ]; then
        ATTEMPT_PROGRESS=$((step - FRAMES_DONE))
        FRAMES_DONE=$step
    else
        ATTEMPT_PROGRESS=$step
        FRAMES_DONE=$((FRAMES_DONE + step))
    fi
    [ "$ATTEMPT_PROGRESS" -lt 0 ] && ATTEMPT_PROGRESS=0
    return 0
}

# An attempt that produced no weights at all never got past start-up; used to stop a respawn loop
# that would otherwise spin on an unrecoverable failure for the rest of the SLURM allocation.
attempt_wrote_weights() {
    [ -n "$1" ] && compgen -G "$1/weights/*.pth" > /dev/null 2>&1
}
