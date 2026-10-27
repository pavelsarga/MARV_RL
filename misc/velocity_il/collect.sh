#!/usr/bin/env bash
# Velocity-IL collection: several crossings in ONE Gazebo session (no restart between rides).
#
#   misc/velocity_il/collect.sh [-n REPEATS] [-w WORLD] POLICY [POLICY ...]
#   misc/velocity_il/collect.sh -n 3 rl:smooth_ramp0_temporal sas     # 6 rides
#   misc/velocity_il/collect.sh -n 3 manual                            # manual runs alone
#
# Runs eval_run.py (speed_source joy: operator throttle on the left stick, GUI, throttle_ui) for
# POLICY x REPEATS rides, policy by policy. Before each ride the robot is reset and the log shows
#   OPERATOR: READY -- ... release the left stick, then push it to start
# -- the bag and the ride start on that push, so take a break there whenever needed; GO / STOP
# banners follow as before. Ctrl-C ends the session (the current bag is closed).
#
# With sas in the list, the reactive controller's anti-stuck dig is slowed for the session
# (SAS_PARAMS below; set live, reactive_sim.yaml and the SAS eval baseline stay untouched).
set -euo pipefail

WS=/home/robot/workspaces/flipper_eval_marv
REPEATS=1
WORLD=straight_arena
# dig 2.0 rad/s (4x the package 0.5) levers the body off the ground on occasion -> 1.0; the
# stabilization is the bang-bang + P hybrid (constant_rate_add_p): sign(e) * 1.0 + 2.0 * e outside the
# 3 deg deadzone -- 1.24 rad/s at 10 deg of tilt, 1.59 at 20, 1.94 at 30; never weaker than the plain
# bang-bang. (The P law alone is weaker below ~26 deg: its deadzone is subtracted from the error.)
SAS_PARAMS='{"control.antistuck.increment": 1.0, "control.stabilization.constant_rate_mode": true,
             "control.stabilization.constant_rate": 1.0, "control.stabilization.constant_rate_add_p": true,
             "control.stabilization.pitch_p": 2.0, "control.stabilization.roll_p": 2.0}'

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while getopts "n:w:h" opt; do
    case $opt in
        n) REPEATS=$OPTARG ;;
        w) WORLD=$OPTARG ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || usage 1
if [[ " $* " == *" manual "* && $# -gt 1 ]]; then
    echo "manual runs alone (mode_teleop holds /flipper_override): collect.sh -n $REPEATS manual" >&2
    exit 1
fi

PARAMS='{}'
[[ " $* " == *" sas "* ]] && PARAMS=$SAS_PARAMS

echo "collecting $REPEATS x [$*] on $WORLD -- $(( REPEATS * $# )) rides in one session"
cd "$WS"
exec python3 src/robot_rodeo_gym_ros2/sim/scripts/eval_run.py \
    --world "$WORLD" --gt 0 --speed_source joy --policies "$@" --repeats "$REPEATS" \
    --controller_params "$PARAMS"
