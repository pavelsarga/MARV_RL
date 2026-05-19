#!/usr/bin/env bash
# optuna_fix_stale_running.sh
#
# Finds Optuna trials stuck in RUNNING state whose SLURM task is no longer
# active, then marks them FAIL. Run this after a batch of jobs finishes or
# whenever the DB looks like it has phantom RUNNING entries.
#
# Usage:
#   bash scripts/optuna_fix_stale_running.sh [--dry-run] [--user USERNAME] [--db PATH]
#
# Options:
#   --dry-run        Print what would change without modifying the DB
#   --user USERNAME  SLURM user to query (default: sargapav)
#   --db PATH        Path to optuna SQLite DB (default: <ws>/optuna/optuna.db)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
WS=$(cd "$(dirname "$0")/.." && pwd)
DB="$WS/optuna/optuna.db"
SLURM_USER="sargapav"
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true ;;
        --user)     SLURM_USER="$2"; shift ;;
        --db)       DB="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ ! -f "$DB" ]]; then
    echo "ERROR: DB not found at $DB" >&2
    exit 1
fi

echo "DB       : $DB"
echo "SLURM user: $SLURM_USER"
echo "Dry run  : $DRY_RUN"
echo ""

# ── Step 1: collect active SLURM job+task IDs from squeue ────────────────────
declare -A active_tasks

while IFS= read -r spec; do
    [[ -z "$spec" ]] && continue

    if [[ "$spec" =~ ^([0-9]+)_([0-9]+)$ ]]; then
        # Individual running task: 10972867_127
        active_tasks["${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"]=1

    elif [[ "$spec" =~ ^([0-9]+)_\[([0-9]+)-([0-9]+) ]]; then
        # Pending array range: 10972867_[128-199%8]  or  10972867_[5-10]
        job_id="${BASH_REMATCH[1]}"
        range_start="${BASH_REMATCH[2]}"
        range_end="${BASH_REMATCH[3]}"
        for ((t = range_start; t <= range_end; t++)); do
            active_tasks["${job_id}_${t}"]=1
        done

    elif [[ "$spec" =~ ^([0-9]+)_\[([0-9,]+)\] ]]; then
        # Sparse list: 10972867_[1,3,7]
        job_id="${BASH_REMATCH[1]}"
        IFS=',' read -ra ids <<< "${BASH_REMATCH[2]}"
        for t in "${ids[@]}"; do
            active_tasks["${job_id}_${t}"]=1
        done
        # (bare job IDs with no underscore are non-array jobs — skip)
    fi
done < <(squeue -u "$SLURM_USER" -h -o "%i" 2>/dev/null)

echo "Active SLURM tasks : ${#active_tasks[@]}"

# ── Step 2: fetch RUNNING trials with logpath from DB ────────────────────────
mapfile -t db_rows < <(sqlite3 "$DB" \
    "SELECT t.trial_id, ua.value_json
     FROM trials t
     JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
     WHERE t.state = 'RUNNING' AND ua.key = 'logpath'
     ORDER BY t.trial_id;")

total_running=$(sqlite3 "$DB" \
    "SELECT COUNT(*) FROM trials WHERE state='RUNNING';")

echo "RUNNING trials in DB: $total_running  (${#db_rows[@]} with logpath)"
echo ""

# ── Step 3: cross-reference ───────────────────────────────────────────────────
stale_ids=()

for row in "${db_rows[@]}"; do
    trial_id="${row%%|*}"
    logpath="${row#*|}"

    # Strip surrounding JSON quotes: "/ws/.../optuna_ftr_JOBID_TASKID" → ...
    logpath="${logpath#\"}"
    logpath="${logpath%\"}"

    # basename → optuna_ftr_JOBID_TASKID
    dir_name=$(basename "$logpath")

    # Last segment after final underscore = task_id; second-to-last = job_id
    task_id="${dir_name##*_}"
    tmp="${dir_name%_*}"
    job_id="${tmp##*_}"

    if ! [[ "$job_id" =~ ^[0-9]+$ && "$task_id" =~ ^[0-9]+$ ]]; then
        echo "  WARN  trial $trial_id — cannot parse job/task from logpath: $logpath"
        continue
    fi

    key="${job_id}_${task_id}"

    if [[ -n "${active_tasks[$key]+_}" ]]; then
        printf "  OK    trial %-6s  task %s  (active in squeue)\n" "$trial_id" "$key"
    else
        printf "  STALE trial %-6s  task %s  (not in squeue)\n" "$trial_id" "$key"
        stale_ids+=("$trial_id")
    fi
done

# Warn about any RUNNING trials that had no logpath at all
no_logpath=$((total_running - ${#db_rows[@]}))
if [[ $no_logpath -gt 0 ]]; then
    echo ""
    echo "  WARN  $no_logpath RUNNING trial(s) have no logpath attribute — cannot check, skipping."
fi

# ── Step 4: mark stale trials FAIL ───────────────────────────────────────────
echo ""
if [[ ${#stale_ids[@]} -eq 0 ]]; then
    echo "No stale RUNNING trials found. DB is clean."
    exit 0
fi

id_list=$(IFS=,; echo "${stale_ids[*]}")
echo "Stale trial(s) to mark FAIL: $id_list"

if $DRY_RUN; then
    echo "(dry-run — no changes made)"
    exit 0
fi

NOW=$(date -u +"%Y-%m-%d %H:%M:%S")
result=$(sqlite3 "$DB" \
    "UPDATE trials
     SET state = 'FAIL', datetime_complete = '${NOW}'
     WHERE trial_id IN (${id_list}) AND state = 'RUNNING';
     SELECT changes() || ' trial(s) marked FAIL.';")

echo "$result"
