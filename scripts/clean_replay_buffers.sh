#!/usr/bin/env bash
# Delete C-TRAC replay-buffer memmap directories to reclaim disk.
#
# Two layouts exist and they are NOT equivalent:
#   <run>/attempt_N/replay_buffer   legacy, per-attempt. Never reused by anything —
#                                   always safe to delete once the attempt is over.
#   <run>/replay_buffer             current, shared by every attempt of a run. This is
#                                   what a respawn restores from; deleting it under a
#                                   live run makes that run refill from empty.
#
# Default is a DRY RUN over the legacy layout only. Deleting is opt-in (--delete), and
# run-root buffers are opt-in on top of that (--include-run-root). Anything touched
# within --min-age minutes is skipped as probably-live regardless of flags.
#
# Usage:
#   bash scripts/clean_replay_buffers.sh                        # dry run, legacy only
#   bash scripts/clean_replay_buffers.sh --delete               # delete legacy
#   bash scripts/clean_replay_buffers.sh --delete --include-run-root
#   bash scripts/clean_replay_buffers.sh --delete --min-age 0   # ignore the liveness guard
#   bash scripts/clean_replay_buffers.sh --roots logs runs      # search elsewhere
set -euo pipefail

DELETE=0; INCLUDE_RUN_ROOT=0; MIN_AGE=60; ROOTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete)           DELETE=1; shift ;;
        --include-run-root) INCLUDE_RUN_ROOT=1; shift ;;
        --min-age)          MIN_AGE="$2"; shift 2 ;;
        --roots)            shift; while [[ $# -gt 0 && "$1" != --* ]]; do ROOTS+=("$1"); shift; done ;;
        -h|--help)          sed -n '2,22p' "$0"; exit 0 ;;
        *)                  echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
cd "$(dirname "$0")/.."
[[ ${#ROOTS[@]} -eq 0 ]] && ROOTS=(logs runs)

EXISTING=()
for r in "${ROOTS[@]}"; do [[ -d "$r" ]] && EXISTING+=("$r"); done
if [[ ${#EXISTING[@]} -eq 0 ]]; then echo "none of the search roots exist: ${ROOTS[*]}"; exit 0; fi

mapfile -t ALL < <(find "${EXISTING[@]}" -type d -name replay_buffer -prune | sort)

TOTAL_KB=0; SKIPPED=0; TARGETS=()
printf '%-10s %10s  %s\n' KIND SIZE PATH
for d in "${ALL[@]}"; do
    if [[ "$(basename "$(dirname "$d")")" =~ ^attempt_[0-9]+$ ]]; then kind=legacy; else kind=run-root; fi
    [[ $kind == run-root && $INCLUDE_RUN_ROOT -eq 0 ]] && continue
    # Liveness guard: anything modified recently may belong to a running job.
    if [[ "$MIN_AGE" -gt 0 ]] && [[ -n "$(find "$d" -newermt "-${MIN_AGE} minutes" -print -quit 2>/dev/null)" ]]; then
        printf '%-10s %10s  %s  <-- SKIPPED (modified <%s min ago)\n' "$kind" "$(du -sh "$d" | cut -f1)" "$d" "$MIN_AGE"
        SKIPPED=$((SKIPPED + 1)); continue
    fi
    kb=$(du -sk "$d" | cut -f1); TOTAL_KB=$((TOTAL_KB + kb))
    printf '%-10s %10s  %s\n' "$kind" "$(du -sh "$d" | cut -f1)" "$d"
    TARGETS+=("$d")
done

echo
echo "${#TARGETS[@]} directories, $(numfmt --to=iec --from-unit=1024 "$TOTAL_KB" 2>/dev/null || echo "${TOTAL_KB}K") reclaimable; $SKIPPED skipped as possibly live."
if [[ ${#TARGETS[@]} -eq 0 ]]; then exit 0; fi
if [[ $DELETE -eq 0 ]]; then echo "Dry run — re-run with --delete to remove them."; exit 0; fi

for d in "${TARGETS[@]}"; do rm -rf -- "$d"; done
echo "Deleted ${#TARGETS[@]} directories."
