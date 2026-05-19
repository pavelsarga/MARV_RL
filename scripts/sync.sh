#!/usr/bin/env bash
# Sync local workspace to RCI cluster, mirroring .vscode/sftp.json ignore rules.
# Usage:
#   ./scripts/sync.sh               # dry-run (shows what would change)
#   ./scripts/sync.sh --push        # push local → remote
#   ./scripts/sync.sh --pull        # pull remote → local
#   ./scripts/sync.sh --pull-logs   # pull remote logs/ and runs/ → local

LOCAL="/home/robot/workspaces/robot_rodeo_gym_ws/"
REMOTE="sargapav@147.32.84.192:/local/flipper_training/"

RSYNC_OPTS=(
    -avz
    --progress
    --exclude=".vscode/"
    --exclude=".git/"
    --exclude=".DS_Store"
    --exclude="__pycache__/"
    --exclude="*.pt"
    --exclude="*.pth"
    --exclude="/log/"
    --exclude="/build/"
    --exclude="install/"
    --exclude="wandb/"
    --exclude="CLAUDE.md"
    --exclude=".claude/"
    --exclude=".idea/"
    --exclude="*.sif"
    --exclude="FTR-Benchmark/"
    --exclude="logs/"
    --exclude="runs/"
)

case "${1:-}" in
    --push)
        echo "Pushing local → remote..."
        rsync "${RSYNC_OPTS[@]}" "$LOCAL" "$REMOTE"
        ;;
    --pull)
        echo "Pulling remote → local..."
        rsync "${RSYNC_OPTS[@]}" "$REMOTE" "$LOCAL"
        ;;
    --pull-logs)
        echo "Pulling remote logs/ and runs/ → local..."
        rsync -avz --progress --mkpath "$REMOTE/logs/" "$LOCAL/logs/"
        rsync -avz --progress --mkpath "$REMOTE/runs/" "$LOCAL/runs/"
        ;;
    *)
        echo "Dry-run (pass --push, --pull, or --pull-logs to sync):"
        rsync "${RSYNC_OPTS[@]}" --dry-run "$LOCAL" "$REMOTE"
        ;;
esac
