# 01 — Action chunking / receding horizon

## The wrapper

`src/flipper_training/marv_rl_training/environment/chunked_env.py` defines
`ActionChunkEnv(EnvBase)`, which wraps `FtrTorchRLEnv`
(`environment/ftr_env_adapter.py:30`) and turns `T_a` control steps into one RL step.

- `action_spec` becomes `Bounded(-1, 1, shape=(num_envs, T_p * A))` — **flattened**, not
  `(num_envs, T_p, A)`. That keeps `NormalParamExtractor`, `TanhNormal`,
  `ProbabilisticActor` and `ClipPPOLoss` on their well-trodden paths and makes
  `sample_log_prob` one scalar per macro-step. The wrapper reshapes internally.
- `_step` reshapes to `[N, T_p, A]`, takes `chunk[:, :T_a]`, and calls the inner env's
  `_step` `T_a` times. The remaining `T_p - T_a` steps are discarded — that is the
  receding-horizon part; they exist only to give the network a longer context to shape
  the executed prefix.
- The returned observation is the last inner observation.

## Reward aggregation

```
R_macro = sum_{i < T_a} gamma_ctrl^i * r_i      (accumulated only while the env is alive)
```

so that the macro-MDP with discount `gamma_ctrl^{T_a}` has the same discounted return as
the underlying control-step MDP. See "Discounting" below.

## Done aggregation

`terminated`, `truncated` and `explosion` are OR-ed across the sub-steps. `done` is
`terminated | truncated`. The `explosion` key must be preserved: `train_ftr.py` drops
whole env trajectories that contain any explosion step before computing GAE, and that
filter reads `("next", "explosion")`.

## The auto-reset hazard — the main compromise in this design

`FtrEnv` (IsaacLab `DirectRLEnv`) **auto-resets terminated envs inside `step()`**, and
the env is `_batch_locked`, so we cannot step a subset. Once an env terminates part-way
through a chunk, the remaining sub-steps would drive a *freshly respawned* robot with
stale chunk actions meant for the previous episode.

Mitigation: from the sub-step after an env goes done, we feed it a **neutral action** for
the rest of the chunk — zero track velocity, and a flipper command that holds the current
angle (in position mode: the normalised current angle; in velocity mode: zero). Its
reward stops accumulating at that point.

Residual cost: a fresh episode loses up to `T_a - 1` control steps to idling, i.e. up to
0.3 s of a 30 s episode at `T_a=4`. It is *not* zero — the new episode's first few steps
are wasted and its `shaping` term is computed from a stationary robot. Open question 9
tracks whether this is measurable; the cheap test is `T_a=1` (where the artifact cannot
occur) against `T_a=4`.

Rejected alternatives:

- *Stop the chunk for everyone at the first done.* Correct, but at 1024 envs some env
  terminates almost every step, so chunks would almost always be truncated to length 1
  and the whole exercise would be pointless.
- *Disable auto-reset.* It is IsaacLab `DirectRLEnv` behaviour and the adapter's
  `_reset` is built around it (`ftr_env_adapter.py:129-152`). Changing it is a much
  larger, riskier change to shared code.

## Pass-through API

`run_tracked_rollout` (`training/eval_data.py:369`) and all `rew/*` logging call methods
on the *inner* env. `ActionChunkEnv` forwards them unchanged:

`peek_reward_series`, `pop_reward_info`, `pop_state_stats`, `pop_termination_info`,
`enable_per_env_tracking`, `pop_per_env_termination`, `disable_per_env_tracking`,
`observations`, `ftr_env`.

Because the inner env accumulates per control step, every `rew/*` and `state/*` number
keeps its **per-control-step** meaning. Only `train/mean_reward` (the macro reward) is on
the new scale. Say so in any comparison against baseline runs.

## Discounting

`gae_opts.gamma` must become the **macro** discount `gamma_ctrl^{T_a}`, while the
potential-based shaping term stays on the per-control-step discount. The baseline config
ties them together (`shaping_gamma: ${gae_opts.gamma}`), which would be silently wrong
here: shaping would be discounted as if one macro-step were one control step, breaking
the potential-based-shaping invariance that makes it unbiased.

A `pow` OmegaConf resolver is registered alongside `add/mul/div/intdiv/cls/dtype/tensor`
in `marv_rl_training/__init__.py`, so the config can derive one from the other:

```yaml
control_gamma: 0.9956304305372804          # unchanged per-control-step value
T_a: 4
gae_opts:
  gamma: ${pow:${control_gamma},${T_a}}    # macro discount
env_cfg_overrides:
  shaping_gamma: ${control_gamma}          # NOT gae_opts.gamma
```

## Frame accounting

`total_frames` keeps counting **control steps**. Two reasons: the SLURM respawn helper
`slurm/lib/respawn_common.sh` reads `total_frames` out of the config and tracks progress
from `weights/policy_step_<frames>.pth`, and we want the frame budget to mean the same
thing as in the baseline runs.

That makes two different quantities per training iteration:

```
transitions_per_iter = time_steps_per_batch * num_robots          # what PPO optimises over
frames_per_iter      = transitions_per_iter * T_a                 # what total_frames counts
```

Every site that currently conflates them has to pick the right one:

| site | quantity |
|---|---|
| `SyncDataCollector(frames_per_batch=...)` | transitions (the collector counts its own steps) |
| replay buffer `max_size` | transitions |
| `frames_per_sub_batch` divisibility check | transitions |
| `total_collected_frames`, the progress bar, `policy_step_<n>.pth` names | control-step frames |
| `SyncDataCollector(total_frames=...)` | transitions — `total_frames // T_a` |
| `scheduler_opts.total_iters` in the config | iterations — `total_frames // (time_steps_per_batch * num_robots * T_a)` |

Getting this wrong corrupts respawn budgeting silently: the helper would decrement the
budget by the wrong amount and either stop early or spin. This is the same class of
failure CLAUDE.md documents for the C-TRAC replay buffer.
