#!/usr/bin/env bash
# Sync local workspace to RCI cluster, mirroring .vscode/sftp.json ignore rules.
# Usage:
#   ./scripts/sync.sh               # dry-run (shows what would change)
#   ./scripts/sync.sh --push        # push local → remote
#   ./scripts/sync.sh --pull        # pull remote → local
#   ./scripts/sync.sh --pull-logs   # pull remote logs/ and runs/ → local

LOCAL="/home/robot/workspaces/robot_rodeo_gym_ws/src/flipper_training"
REMOTE="marv-robot-jetson:/home/robot/rodeo_rl_ws/src/flipper_training"

RSYNC_OPTS=(
    -avz
    --progress
    --exclude="*.pt"
    --exclude="*.pth"
    --exclude="*.csv"
    --exclude=".idea"
    --exclude=".pytest_cache"
    --exclude=".rodeo_cache"
    --exclude="cli_tools"
    --exclude="cross_eval_configs"
    --exclude="cross_eval_results"
    --exclude="final_training_configs"
    --exclude="heightmap_files"
    --exclude="meshes"
    --exclude="notebooks"
    --exclude="runs"
    --exclude="misc_benchmarking"
    --exclude="report"
    --exclude="modified_networks"
    --exclude="sota_configs"
    --exclude="sota_configs"
    --exclude="test_configs"
    --exclude="tests"
    --exclude="slurm"
    --exclude=".git"
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
