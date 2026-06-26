#!/usr/bin/env bash
# Config-driven gamepad teleop for FTR/MARV (runs scripts/teleop.py directly via conda,
# not apptainer, so it can access /dev/input/jsX and the GUI viewport).
#
# Usage:
#   CONFIG=teleop_marv.yaml bash scripts/teleop.sh
#   CONFIG=teleop_ftr.yaml  bash scripts/teleop.sh
#   bash scripts/teleop.sh --config configs/teleop_marv.yaml --js /dev/input/js1
#
# CONFIG (or --config) selects a YAML file under configs/ with all teleop settings
# (robot_type, terrain, sensitivities, episode_length_s override, wheel1 glitch
# tracking). Any extra CLI args are forwarded to teleop.py and override the config.
set -e

WS="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$WS/logs"

cd "$WS" || { echo "Failed to cd into $WS"; exit 1; }

# Only inject a --config from $CONFIG/default if the caller didn't already pass one
# explicitly on the command line (e.g. `teleop.sh --config configs/teleop_ftr.yaml`).
EXTRA_ARGS=()
if [[ " $* " != *" --config "* ]]; then
    CONFIG=${CONFIG:-teleop_marv.yaml}
    EXTRA_ARGS=(--config "$WS/configs/$CONFIG")
    echo "========================================================================"
    echo "teleop.sh — CONFIG=$CONFIG"
    echo "========================================================================"
else
    echo "========================================================================"
    echo "teleop.sh — using --config from command line"
    echo "========================================================================"
fi

OMNI_KIT_ACCEPT_EULA=Y \
conda run -n isaaclab --no-capture-output \
    env PYTHONPATH="$WS/src/FTR-Benchmark:$WS/src/flipper_training" \
    python "$WS/scripts/teleop.py" \
    "${EXTRA_ARGS[@]}" \
    "$@"
