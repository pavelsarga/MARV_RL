#!/usr/bin/env bash
# Evaluate a trained receding-horizon / diffusion policy.
#
#   bash scripts/eval_diffusion.sh <run_dir> [--num_envs 256] [--repeats 30] [--headless] ...
#
# NOT eval.sh / eval_auto.sh. Those route on env_cfg_overrides.module_name, which is
# marv_rl for these runs too — but that names the observation/reward module, not the
# trainer. eval_ftr.py parses the config into FtrPPOConfig and dies on
# prediction_horizon/execution_horizon/history_len/control_gamma, and even past that it
# builds the env without ActionChunkEnv or CatFrames, so the action spec is 6-D instead of
# T_p x 6 and nothing writes obs_history.
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_SCRIPT=eval_diffusion.py bash "$WS/scripts/eval.sh" "$@"
