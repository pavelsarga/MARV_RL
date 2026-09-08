#!/usr/bin/env bash
# Evaluate a trained FTR PPO policy (eval_ftr.py) in the Apptainer container, on this machine.
#
# Usage:
#   ./scripts/eval.sh <run_dir> [extra eval_ftr.py args...]
#
# <run_dir> is the run directory. Host paths under the workspace root are rewritten to the
# container mount point /ws/ automatically — for <run_dir>, --output_dir and --env_names_yaml.
# Quote the path or pass it on one line: backslash continuation is unreliable when terminal
# wrapping adds trailing spaces.
#
# Examples:
#   ./scripts/eval.sh experiments/baselines/marv_rl          # policy_final.pth, else latest step
#   ./scripts/eval.sh "..." --weights 5963776                # policy/vecnorm_step_5963776.pth
#   ./scripts/eval.sh "..." --weights final                  # policy/vecnorm_final.pth
#   ./scripts/eval.sh "..." --weights latest                 # highest-numbered step checkpoint
#   ./scripts/eval.sh "..." --weights latest --num_envs 32 --repeats 3 --map cur_stairs_up
#   ./scripts/eval.sh "..." --plot_heightmap [--plot_interval 5]   # forces num_envs=1
#   Heightmap PNGs (and optionally a GIF) go to /tmp/ftr_eval_<timestamp>/ on the host.
#
# Per-env-type CSV output (eval_summary.csv, eval_per_env.csv, eval_episodes.csv):
#   ./scripts/eval.sh "..." --output_dir /tmp/eval_out [--eval_id label] [--repeats 5]
#   Env-type count/names default to the terrain's layout, read from
#   ftr_envs/assets/terrain/gen_config/<terrain>.yaml (see env_type_registry.py);
#   --num_env_types / --env_names_yaml override. The run's terrain is recorded in
#   <output_dir>/eval_terrain.json and its gen_config + preview plot copied to
#   <output_dir>/terrain/ before the first rollout, so notebooks/eval_analysis.ipynb can
#   label and group the results by terrain even if the job dies.
#
# Available terrains: ground, cur_mixed, cur_stairs_up, exp_stair33_up and everything under
# ftr_envs/assets/terrain/gen_config/ (custom_mixed, pan_symmetric, mitriakov_stairs, ...).
#
# EVAL_SCRIPT=<name>.py overrides the entry point — that is how scripts/eval_diffusion.sh
# reuses this script. The job body is scripts/lib/eval_run.sh, shared by every eval_*.sh.

WS="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_KIND=${EVAL_KIND:-ppo}

source "$WS/scripts/lib/eval_run.sh"
run_local_eval "$@"
