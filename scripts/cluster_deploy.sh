#!/usr/bin/env bash
# Deploy this workspace to the RCI git clone.
#
#   bash scripts/cluster_deploy.sh              # push current branch, then deploy it
#   bash scripts/cluster_deploy.sh --status     # show local vs cluster state, change nothing
#   bash scripts/cluster_deploy.sh --no-push    # deploy whatever is already on the remote
#   bash scripts/cluster_deploy.sh -b main      # deploy a specific branch
#
# Env overrides: CLUSTER_HOST, CLUSTER_WS, SSH_KEY.
#
# Why this exists rather than a bare `git pull` on the cluster: submodule pointers. The
# superrepo records a commit in flipper_training, and pushing the superrepo without first
# pushing that submodule commit leaves the cluster fetching a pointer to an object the
# remote does not have — `git submodule update` then fails with a confusing
# "reference is not a tree". This script always pushes submodules first, and verifies the
# deployed commits afterwards instead of trusting exit codes.
set -euo pipefail

HOST="${CLUSTER_HOST:-sargapav@login4.rci.cvut.cz}"
WS="${CLUSTER_WS:-/mnt/personal/sargapav/MARV_RL_ws_git}"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SUBMODULES="src/flipper_training src/FTR-Benchmark"   # FTR-Bench-terrain-gen has no deploy key

BRANCH=""; PUSH=1; STATUS_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        -b|--branch) BRANCH="$2"; shift 2 ;;
        --no-push)   PUSH=0; shift ;;
        --status)    STATUS_ONLY=1; PUSH=0; shift ;;
        -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
BRANCH="${BRANCH:-$(git branch --show-current)}"
SSH="ssh -n -o BatchMode=yes -i $KEY $HOST"

say() { printf '\033[1m%s\033[0m\n' "$*"; }

say "workspace : $WS on $HOST"
say "branch    : $BRANCH"

if [ "$STATUS_ONLY" -eq 1 ]; then
    echo
    say "-- local --"
    printf '  superrepo        %s\n' "$(git log --oneline -1)"
    for s in $SUBMODULES; do printf '  %-16s %s\n' "$(basename "$s")" "$(git -C "$s" log --oneline -1)"; done
    git status --porcelain | grep -q . && echo "  (uncommitted changes present — they will NOT be deployed)"
    echo
    say "-- cluster --"
    $SSH "cd $WS 2>/dev/null || { echo '  clone not found'; exit 0; }
          printf '  superrepo        %s\n' \"\$(git log --oneline -1)\"
          for s in $SUBMODULES; do printf '  %-16s %s\n' \"\$(basename \$s)\" \"\$(git -C \$s log --oneline -1)\"; done"
    exit 0
fi

# Uncommitted work is invisible to git deploy — say so rather than silently shipping stale code.
DIRTY=$(git status --porcelain | wc -l)
if [ "$DIRTY" -gt 0 ]; then
    echo
    echo "WARNING: $DIRTY uncommitted change(s) will NOT be deployed. Commit them first if" >&2
    echo "         they matter — with git, unlike rsync, what is on disk is not what ships." >&2
    git status --porcelain | head -8 | sed 's/^/  /' >&2
    [ "$DIRTY" -gt 8 ] && echo "  ... and $((DIRTY - 8)) more (run --status, or git status)" >&2
    echo
fi

if [ "$PUSH" -eq 1 ]; then
    for s in $SUBMODULES; do
        # Push only if this submodule's HEAD is not already reachable on the remote.
        # `submodule update` resolves the recorded SHA, not a branch, so a submodule whose
        # commit is already on origin (e.g. still on main) needs no branch of its own.
        if ! git -C "$s" branch -r --contains HEAD 2>/dev/null | grep -q .; then
            say "pushing submodule $(basename "$s") ($BRANCH)"
            git -C "$s" push -q origin "HEAD:$BRANCH"
        fi
    done
    say "pushing superrepo ($BRANCH)"
    git push -q origin "HEAD:$BRANCH"
fi

say "deploying on cluster"
$SSH "set -e
      cd $WS
      git fetch -q origin
      git checkout -q -B $BRANCH origin/$BRANCH
      git submodule update --init --quiet $SUBMODULES
      echo \"  superrepo        \$(git log --oneline -1)\"
      for s in $SUBMODULES; do echo \"  \$(basename \$s)  \$(git -C \$s log --oneline -1)\"; done"

# Verify rather than trust: compare the deployed commits against local HEADs.
say "verifying"
LOCAL_SUPER=$(git rev-parse HEAD)
REMOTE_SUPER=$($SSH "git -C $WS rev-parse HEAD")
if [ "$LOCAL_SUPER" = "$REMOTE_SUPER" ]; then
    echo "  superrepo matches local HEAD"
else
    echo "  MISMATCH: local $LOCAL_SUPER != cluster $REMOTE_SUPER" >&2
    echo "  (expected if you used --no-push, or if local has unpushed commits)" >&2
fi
for s in $SUBMODULES; do
    L=$(git -C "$s" rev-parse HEAD); R=$($SSH "git -C $WS/$s rev-parse HEAD")
    [ "$L" = "$R" ] && echo "  $(basename "$s") matches local HEAD" \
                    || echo "  MISMATCH in $(basename "$s"): local $L != cluster $R" >&2
done
