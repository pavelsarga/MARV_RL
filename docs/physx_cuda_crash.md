# PhysX `illegal memory access` training crash — cause and fix

```
[Error] [omni.physx.tensors.plugin] CUDA error: an illegal memory access was encountered:
  ../../../extensions/runtime/source/omni.physx.tensors/plugins/gpu/GpuArticulationView.cpp: 749
```

## Symptom

Census of every training log (705 attempts, 242 jobs, 2026-10-31): 85 % of training attempts
ended in this crash, about one per 10M simulated frames, flat across trainer (PPO, D3QN,
ICM-D3QN, SAC, diffusion), terrain, solver iterations, friction, `num_robots`, node and driver.
Resumed attempts were ~6x more likely to crash in their first 1M frames. Standalone evals never
crashed. The traceback always came out of the torchrl collector's rollout, at the next PhysX
tensor-API call after the fault (usually `set_dof_position_targets` → `as_contiguous_uint32`).

## Cause

torchrl's `SyncDataCollector` with its default `no_cuda_sync=False` creates a high-priority side
CUDA stream in `iterator()` and keeps it current across every `yield`. `env.step` — and every
IsaacLab torch op that builds the index and target tensors handed to the PhysX tensor API — ran
on that stream. `omni.physics.tensors` orders its GPU work against the default stream only, so
PhysX could read those tensors before the side stream had written them.

## Fix

`marv_rl_training.training.trainer_common.collector_opts()` defaults `no_cuda_sync=True` for the
collector of every trainer (`train_ftr`, `train_d3qn`/`train_icmd3qn`, `train_sac`,
`train_diffusion`); an explicit `data_collector_opts.no_cuda_sync` in a config still wins. On a
single device the collector's explicit synchronisations it skips are not needed, and training
speed is unchanged.

Do not construct a `SyncDataCollector` without it, and do not step the env under any
`torch.cuda.stream(...)` context.

## Evidence

Reproduced by resuming a crash-prone checkpoint (`diff_p1_smooth_ramp0_fwdmap_v3_11824337`,
67.4M frames) — on the RCI A100s and on a laptop RTX 2000 Ada alike.

| condition | result |
|---|---|
| default (side stream) | crashes within minutes to tens of minutes, every run |
| PhysX buffers shrunk 10x | overflow errors print loudly — and never appear in production logs |
| pre-fault state (crash recorder, 6 dumps) | benign: no NaN, < 1 m/s, no reset burst needed |
| `CUDA_LAUNCH_BLOCKING=1` | no crash |
| device sync around every ArticulationView call | no crash |
| device sync after each `sim.step()` | crashes |
| converted index temporaries kept alive | crashes |
| `no_cuda_sync=True` (`env.step` on `cuda_stream=0x0`) | no crash |
| interleaved control, same machine (`cuda_stream=0x8173…`) | crashed after 5 min |

Ruled out on the way: node hardware (no node correlation, ECC clean, reproduces on a different
GPU and driver), PhysX GPU buffer overflow, exploding/NaN articulations, the >64-link GPU
articulation partition bug (MARV has 50 links), and the articulation sleep / staging-record bug
(PhysX issue #551 — the direct GPU API disables sleeping and the env makes no runtime
articulation edits). No PhysX release after 5.4.2 lists a fix for this signature.

## Diagnostics kept (all off by default)

| switch | effect |
|---|---|
| `FTR_CRASH_RECORDER=<steps>` | CPU ring buffer of root/joint state, targets, actions and resets, dumped to `FTR_CRASH_RECORDER_DIR` when `env.step` raises; prints the CUDA stream `env.step` runs on |
| `FTR_PHYSX_SYNC=views\|set\|get\|step` | `torch.cuda.synchronize()` around ArticulationView calls / after `sim.step()` |
| `FTR_PHYSX_KEEPALIVE=<n>` | keep the last n tensor-API conversion temporaries referenced |
| `FTR_PHYSX_OVERRIDES="field=value,…"` | set any `PhysxCfg` field on an unchanged config |
| `slurm/diag_cuda_crash.sbatch` | resume a run with the above, `DIAG_BLOCKING`, or `DIAG_SANITIZER=memcheck` |

`CUDA_LAUNCH_BLOCKING` and compute-sanitizer both serialise kernel execution, so they hide a
stream race rather than locate it — a crash that disappears under them is a race.
