#!/usr/bin/env bash
# Short local PPO smoke run on a mixed_v2 course (default: straight): few envs, three
# collection batches, an eval after each — enough to prove every row spawns (both
# directions), the observation window is sane on the mesh obstacles, and
# eval_per_env.csv / eval_per_spot.csv come out with one column per row. Not a
# training run. TERRAIN=mixed_v2_full for the turning course. The PhysX GPU buffers
# are shrunk from the config's cluster sizes (2 GB heap + 1 GB temp) to the trainer
# defaults, since on a laptop GPU the cluster sizes make PhysX refuse to create the
# scene ("Physics::createScene: desc.isValid() is false") and nothing simulates.
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/scripts/lib/debug_train.sh"
TERRAIN="${TERRAIN:-straight}"
NUM_ENVS="${NUM_ENVS:-64}"
debug_train "terrain_v2/marv_config_marv_rl_v2_${TERRAIN}.yaml" train_ftr.py \
    --num_envs "$NUM_ENVS" \
    total_frames="${TOTAL_FRAMES:-$((128 * NUM_ENVS * 3))}" \
    eval_and_save_every="${EVAL_AND_SAVE_EVERY:-1}" \
    eval_repeats=1 \
    eval_repeats_after_training=1 \
    physx_gpu_heap_capacity="${PHYSX_HEAP:-268435456}" \
    physx_gpu_temp_buffer_capacity="${PHYSX_TEMP:-67108864}" \
    -- "$@"
