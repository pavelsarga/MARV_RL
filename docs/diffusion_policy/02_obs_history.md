# 02 — Observation history (`T_o`)

## Why a stored key and not a ring buffer

PPO flattens the rollout (`flat = tensordict_data.reshape(-1)`, `train_ftr.py:665`) and
then samples **shuffled** minibatches from a replay buffer. Any per-env ring buffer held
inside the actor module is indexed by env and is only valid during live collection; on a
shuffled minibatch it is wrong.

This is not hypothetical. `rl_modules/ctrac/ctrac_policy.py:107-121` documents exactly
this bug being found and fixed in the C-TRAC actor: a 256-row minibatch came back as
`[99,99,99,99,99,99,99,99]` where the live rollout gives `[0,1,...,7]`, so the C-VAE was
trained on an input distribution the actor never encountered. Their fix was to emit the
window as an extra output key so it lands in the collected tensordict.

We get the same guarantee more cheaply from a transform.

## The transform

```python
CatFrames(N=T_o, dim=-1, in_keys=[OBS_KEY], out_keys=["obs_history"], padding="same")
```

- `OBS_KEY` is `"MarvRLFlatObservation"` (966-D). It is **left in place** — `eval_data.py`'s
  `_OBS_SLICES` observation statistics are gated on `obs_dim == 966`, and the encoder
  lookup in `MLPPolicyConfig` keys off the observation class name. We add a key, we do
  not replace one.
- `padding="same"` repeats the first frame to fill the window at episode start, which is
  what the paper does.
- Output is `[N, T_o * 966]`; the encoder views it as `[N, T_o, 966]`.

## Ordering: after VecNorm

`make_transformed_env` (`training/common.py:17`) currently builds

```
StepCounter -> RawRewardSaveTransform -> *extra_env_transforms -> *policy_transforms -> VecNorm
```

with `VecNorm` appended last. The history window must stack **already-normalised** frames,
otherwise each of the `T_o` slots carries a different, un-normalised scale and VecNorm's
running statistics are computed over a key that no longer means one observation.

So `make_transformed_env` gains an explicit `post_vecnorm_transforms` argument appended
after `VecNorm`, rather than reordering `extra_env_transforms` (other configs rely on
those running before VecNorm).

## Reset semantics

Our env auto-resets internally and reports `done`; TorchRL then issues `_reset` carrying
a `"_reset"` mask, and `CatFrames` clears exactly those rows. This is the standard
frame-stacking path, but it is worth a unit test because the auto-reset makes our env
unusual — `test_action_chunk_env.py` asserts that after a done on env `i`, the window for
`i` is the new observation repeated, while other envs' windows are untouched.

## Cost

`T_o * 966 * 4 B = 7.7 KB` per transition at `T_o=2`, on top of the 966-D observation
stored twice (root and `next`). At `time_steps_per_batch=128, num_robots=1024` that is
about 1 GB per iteration in the CPU replay buffer. Acceptable, but it is the reason `T_o`
should not be raised casually — and the paper found `T_o > 2` actively harmful for the
CNN variant anyway.
