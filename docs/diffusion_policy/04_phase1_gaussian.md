# 04 — Phase 1: Gaussian chunk actor

The point of Phase 1 is to build and validate **everything except the diffusion sampler**,
under a loss we already trust.

## Head

The `ConditionalUnet1D` from `03_network.md`, conditioned on `global_cond` only (no
denoising-step embedding), takes a learned constant input sequence of shape
`[N, A, T_p]` and emits `2*A` channels -> `NormalParamExtractor` ->
`ProbabilisticActor(TanhNormal, low=-1, high=1, return_log_prob=True)` over the flattened
`T_p * A` action.

`ClipPPOLoss` and `GAE` are used **unchanged**.

Using the U-Net rather than an MLP here is deliberate: it means Phase 2 changes the
head's *interface* (add the `k` embedding, add a noisy input sequence) but not its
architecture, so a Phase 1 checkpoint is a usable initialisation for the Phase 2 U-Net.

## Trainer

`src/flipper_training/marv_rl_training/training/train_diffusion.py`, with
`FtrDiffusionConfig`. A **fork** of `train_ftr.py`, not an extension, because:

- the chunked env, macro-gamma and frame accounting are invasive, and `train_ftr.py`
  carries the tuned baseline we must not regress;
- Phase 2 needs a different loss;
- the project's convention is one dataclass per trainer, and pointing the wrong
  `TRAIN_SCRIPT` at a config must fail loudly at parse time (see CLAUDE.md).

Deltas from `train_ftr.py`:

- wrap the env in `ActionChunkEnv` before `make_transformed_env`;
- `post_vecnorm_transforms=[CatFrames(...)]`;
- new config fields `T_o`, `T_p`, `T_a`, `control_gamma`;
- transition-vs-frame accounting (`01_action_chunking.md`);
- action logging. `train_ftr.py:683-694` hardcodes `actions[:, 0]` and `actions[:, 2+fi]`,
  which is meaningless on a flattened chunk. Reshape to `[N, T_p, A]` and log per-dimension
  mean/std over the executed prefix, plus **`action/chunk_step_delta`** = mean
  `|a_{t+1} - a_t|` within a chunk. That last one is the direct measurement of the temporal
  smoothness this whole exercise is after — log it from the first run.

## ⚠ A crash must force-exit, or the SLURM job hangs

`train_ftr.py` and every trainer forked from it call `trainer.train()` unguarded at module
level, and reach `os._exit(0)` only on the success path. `train()` itself force-exits with
`os._exit(75)` on CUDA/W&B errors, but **any other exception is re-raised** — and an
uncaught exception means Python runs a normal interpreter shutdown, which triggers Isaac
Sim's atexit handlers, which deadlock. That is the same deadlock the `os._exit(0)` at the
bottom of the file exists to avoid; the generic error path just never got the same
treatment.

The job then holds its node until walltime instead of failing. On `amdgpulong` that is up
to 24 h of a GPU per crash.

Observed, not theorised: job `diff_smoke_11493997` crashed on a tensor-shape bug at
00:10:20, logged `Training failed`, wrote its crash checkpoint and closed the RunLogger —
and was still `RUNNING` 15 minutes later, doing nothing, until it was cancelled by hand.

`train_diffusion.py` now wraps the top-level call:

```python
try:
    trainer.train()
except BaseException:
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)          # 1, not 75: 75 means "transient, respawn me"
```

**Where the fix already existed.** `optuna_train_ftr.py` has had exactly this guard all
along (`except Exception: traceback.print_exception(_e); os._exit(1)`), and
`recover_optuna_trials.py` ends with `os._exit(0 if ok else 1)` for the same reason. The
pattern was written for the Optuna runner and never propagated to the `train_*.py` family.
Nothing reverted it and no other mechanism covers it — there are no signal handlers, no
`atexit` registration, no `srun --kill-on-bad-exit`, and no timeout wrapper. The respawn
loop cannot help either: it only runs *after* `srun` returns, and `srun` never returns
while the process is hung.

The guard is now applied to all six trainers (`train_ftr.py`, `train_d3qn.py`,
`train_icmd3qn.py`, `train_sac.py`, `train_creps.py`, `train_diffusion.py`). It catches
`BaseException`, not `Exception`, because a `KeyboardInterrupt` or `SystemExit` escaping
here would hang exactly the same way; a `SystemExit`'s own exit code is preserved.

## Config

`configs/diffusion/marv_config_diffusion_p1.yaml`, copied from
`configs/baselines/marv_config_marv_rl.yaml` and changed in:

- `policy_config` -> `DiffusionPolicyConfig`
- `T_o`, `T_p`, `T_a`, `control_gamma`, `gae_opts.gamma: ${pow:...}`
- `env_cfg_overrides.shaping_gamma: ${control_gamma}`
- `env_cfg_overrides.flipper_control_mode: position`

Everything else — terrain, reward coefficients, VecNorm, physics — is held fixed so the
comparison against the baseline means something.

## The position-control side effect

`flipper_control_mode: position` bypasses the integration and clamp at
`ftr_env.py:688-712`, which is the only temporal smoothing the flippers currently have
(5 deg per control step = 50 deg/s). Under position control the policy can command a full
swing in one step.

Watch for it. If the flippers snap, either re-enable `joint_vel_variance_coef` (which is
`null` in the baseline) or add an explicit within-chunk rate penalty. `action/chunk_step_delta`
is the instrument. Open question 5.

## ⚠ 160 PPO updates per iteration diverges — use `epochs_per_batch: 3`

The shakedown's finding, and the least obvious one. With `T_a=4` the baseline's
`epochs_per_batch: 5` (× 32 sub-batches = 160 updates per iteration) drove `clip_fraction`
to 0.66 and KL to 0.09, both climbing monotonically, with reward flat and the policy never
concentrating. Measured at iteration 6-7, all else equal:

| config | `clip_fraction` | KL | trend |
|---|---|---|---|
| `ep=5`, lr ×1 | 0.662 | 0.089 | rising ✗ |
| `ep=5`, **lr /4** | 0.549 | 0.046 | rising ✗ |
| `ep=5`, **lr /2** | 0.643 | 0.074 | rising ✗ |
| `ep=2`, lr ×1 | 0.131 | 0.010 | falling ✓ |
| **`ep=3`, lr ×1** | **0.192** | **0.014** | **falling ✓** |

**Lowering the learning rate made it worse, and that is the instructive part.** PPO's
clipped objective has zero gradient for samples outside the trust region, so at full LR the
policy reaches the clip boundary quickly and drift *self-limits* — clipped samples stop
contributing. At a smaller LR more samples stay *inside* the region for longer, gradients
keep flowing across all 160 updates, and cumulative drift ends up as large or larger. For
this failure mode the lever is the update **count**; the step **size** regulates itself.

`ep=3` is chosen over the safer `ep=2` because gradient steps are scarce here: see below.

## Fewer gradient steps for the same frame budget

`total_frames` is held at the baseline's 73.4M **control** steps so the simulation budget
is identical and the comparison is fair. But one iteration now spans `T_a` times more
control steps, so the run gets 140 iterations where the baseline gets 560 — about a
quarter of the gradient steps.

That is inherent to chunking (the policy makes fewer decisions per frame), not a bug. It
does mean Phase 1 could lose to the baseline by underfitting rather than by chunking being
a bad idea. Distinguish the two by looking at whether the reward curve has plateaued at the
end; if it is still rising, raise `epochs_per_batch` before touching `total_frames`. Open
question 14.

## Supporting files

- `scripts/train_diffusion_debug.sh` — few envs, a handful of iterations, no W&B. Pattern
  of `scripts/train_atd3qn_debug.sh`. Its most useful mode is the `T_a=1, T_p=1` reduction
  to ordinary single-step PPO:
  `EXECUTION_HORIZON=1 PREDICTION_HORIZON=1 DOWN_DIMS=[64] bash scripts/train_diffusion_debug.sh`
- `slurm/train_diffusion.sbatch` — pattern of `slurm/train_marv_rl.sbatch` plus
  `config_arg.sh` for config/override selection.
- `scripts/bench_diffusion_head.py` — the sizing benchmark (see `03_network.md`).
- `training/test_action_chunk_env.py` — Isaac-free unit test for the wrapper and the
  observation window (24 checks).
- `training/test_diffusion_policy_shapes.py` — Isaac-free rank/shape test for the networks,
  ending in a real GAE call over an `[envs, time]` batch. Needs both `src/flipper_training`
  and `src/FTR-Benchmark` on `PYTHONPATH`. See the vmap warning in `03_network.md`.

**One change was needed in shared code**: `frame_budget_init` in
`slurm/lib/respawn_common.sh` computed the iteration count as
`total_frames / (time_steps_per_batch * num_robots)`, which is `T_a` times too large under
chunking. It pins `scheduler_opts.total_iters`, `step_penalty_scheduler.total_iters` and
`action_bonus_coef_scheduler.total_iters` on every respawn, so without the fix any respawn
would have stretched the LR, step-penalty and action-bonus schedules by 4x — silently, and
only on respawned runs. It now reads an optional top-level `execution_horizon` and divides
by it; configs without the key are unaffected (verified against
`configs/baselines/marv_config_marv_rl.yaml`, still 560 iterations).
