#!/usr/bin/env bash
# MARV yaw-authority (friction turn) test: hold a fixed flipper posture on flat ground and
# command a full turn (w = +1, i.e. track_ang_vel_max) open-loop, then read the achieved
# yaw rate from the env's own state/yaw_rate* stats. Answers "can this robot pivot at all,
# and in which posture" — the prerequisite for any terrain that requires turning
# (mixed_v2_full's offset_gate / chicane / flat_turn rows). Background in CLAUDE.md
# ("MARV cannot skid-steer under load").
#
# Usage:  bash scripts/turn_test.sh <ppo_run_dir> [out_file] [--num_envs N] [--max_steps N]
#
# Position control mode maps action -1 -> the "up" bound for the FRONT pair (-60 deg) but the
# "down" bound for the REAR pair (-80 deg), so the scripted vectors below are asymmetric.
# Flat = action -0.143 (front) / +0.143 (rear) with the marv_rl asymmetric limits (60 up, 80 down).

set -euo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${1:?run dir with config.yaml + weights/}"; shift || true
OUT="${1:-$WS/logs/turn_test/summary.txt}"; [[ "${1:-}" == --* ]] || shift || true
mkdir -p "$(dirname "$OUT")"
NUM_ENVS=16; MAX_STEPS=600
while [[ $# -gt 0 ]]; do case "$1" in
  --num_envs) NUM_ENVS="$2"; shift 2;; --max_steps) MAX_STEPS="$2"; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done

declare -A POSTURES=(
  [flat]="-0.143,-0.143,0.143,0.143"
  [front_up]="-1,-1,0.143,0.143"
  [rear_up]="-0.143,-0.143,1,1"
  [both_up]="-1,-1,1,1"
  [all_down_tiptoe]="1,1,-1,-1"
)
: > "$OUT"
for drive in "0,1" "0.5,1"; do
  for name in flat front_up rear_up both_up all_down_tiptoe; do
    log="$(dirname "$OUT")/${name}_v${drive%%,*}.log"
    echo "=== posture=$name v,w=$drive -> $log"
    bash "$WS/scripts/eval_auto.sh" "$RUN_DIR" --map ground --num_envs "$NUM_ENVS" --max_steps "$MAX_STEPS" \
        --scripted_action "${drive},${POSTURES[$name]}" env_cfg_overrides.flipper_control_mode=position \
        --headless > "$log" 2>&1 || echo "  (eval exited non-zero, see $log)"
    printf '%-18s v,w=%-6s ' "$name" "$drive" >> "$OUT"
    grep -E "state/(yaw_rate|ang_velocity|lin_velocity|heading_err_deg)( |_max|_min|_mean)" "$log" | tail -12 | tr -s ' ' | paste -sd' ' >> "$OUT"
  done
done
echo; cat "$OUT"
