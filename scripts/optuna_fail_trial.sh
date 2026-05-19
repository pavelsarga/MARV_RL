#!/bin/bash
# Change the state of one or more Optuna trials in the local SQLite DB.
#
# Usage:
#   scripts/optuna_fail_trial.sh --study_id 3 --trial_id 10
#   scripts/optuna_fail_trial.sh --study_id 3 --trial_id 10 --trial_id 11 --trial_id 12
#   scripts/optuna_fail_trial.sh --study_id 3 --trial_id 10-15 --state FAIL
#   scripts/optuna_fail_trial.sh --study_id 3 --trial_id 10 --state COMPLETE

set -euo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
DB="$WS/optuna/optuna.db"
VALID_STATES="RUNNING COMPLETE PRUNED FAIL WAITING"

usage() {
    echo "Usage: $0 --study_id <id> --trial_id <id|range> [--trial_id ...] [--state STATE] [--db PATH]"
    echo "  --study_id   Optuna study_id (integer)"
    echo "  --trial_id   Trial number(s): single int or range (e.g. 10-15)"
    echo "  --state      Target state: RUNNING, COMPLETE, PRUNED, FAIL, WAITING (default: FAIL)"
    echo "  --db         Path to SQLite DB (default: optuna/optuna.db)"
    exit 1
}

STUDY_ID=""
TRIAL_IDS=()
NEW_STATE="FAIL"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --study_id) STUDY_ID="$2"; shift 2 ;;
        --state)    NEW_STATE="${2^^}"; shift 2 ;;
        --trial_id)
            ARG="$2"; shift 2
            if [[ "$ARG" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                for i in $(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"); do
                    TRIAL_IDS+=("$i")
                done
            else
                TRIAL_IDS+=("$ARG")
            fi
            ;;
        --db) DB="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$STUDY_ID" ]] && { echo "ERROR: --study_id is required"; usage; }
[[ ${#TRIAL_IDS[@]} -eq 0 ]] && { echo "ERROR: at least one --trial_id is required"; usage; }
[[ -f "$DB" ]] || { echo "ERROR: DB not found at $DB"; exit 1; }
[[ " $VALID_STATES " == *" $NEW_STATE "* ]] || { echo "ERROR: invalid state '$NEW_STATE'. Valid: $VALID_STATES"; exit 1; }

echo "DB:         $DB"
echo "Study ID:   $STUDY_ID"
echo "New state:  $NEW_STATE"
echo ""

for TRIAL_NUM in "${TRIAL_IDS[@]}"; do
    ROW=$(sqlite3 "$DB" "
        SELECT trial_id, state FROM trials
        WHERE study_id = $STUDY_ID AND number = $TRIAL_NUM;
    ")
    if [[ -z "$ROW" ]]; then
        echo "  Trial #$TRIAL_NUM — NOT FOUND in study $STUDY_ID, skipping."
        continue
    fi
    TRIAL_ID=$(echo "$ROW" | cut -d'|' -f1)
    CURRENT_STATE=$(echo "$ROW" | cut -d'|' -f2)

    if [[ "$CURRENT_STATE" == "$NEW_STATE" ]]; then
        echo "  Trial #$TRIAL_NUM (id=$TRIAL_ID) — already $NEW_STATE, skipping."
        continue
    fi

    sqlite3 "$DB" "UPDATE trials SET state = '$NEW_STATE' WHERE trial_id = $TRIAL_ID;"
    echo "  Trial #$TRIAL_NUM (id=$TRIAL_ID): $CURRENT_STATE → $NEW_STATE"
done

echo ""
echo "Done."
