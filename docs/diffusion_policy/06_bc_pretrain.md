# 06 — BC pretraining (warm start)

## Why

DPPO fine-tunes a BC-pretrained diffusion model; it does not train one from scratch. A
randomly initialised `eps_theta` emits noise chunks, and PPO gradients through an 8-step
denoising chain from that starting point are high-variance. The warm start is not
imitation learning as the method — it is initialisation.

## Two scripts

Both follow the shape of the existing C-TRAC dataset pipeline, which does exactly this
job already: `training/collect_ctrac_dataset.py` (roll out a trained `marv_rl` policy,
shard observations to `.npz`) and `training/pretrain_ctrac_cvae.py` (train a module on
those shards, write a checkpoint the policy config loads).

### 1. `training/collect_chunk_dataset.py`

Rolls out a trained `marv_rl` policy from `experiments/<name>/attempt_N` and writes
sharded `(obs_history[T_o, 966], action_chunk[T_p, A])` pairs.

Details that matter:

- Build `obs_history` and `action_chunk` with the **same** padding convention the
  `CatFrames` transform uses (`padding="same"`), or the pretrained head sees a different
  input distribution than the RL fine-tuning does. This is the same class of mismatch that
  bit C-TRAC.
- Do not let a chunk straddle an episode boundary. Drop chunks whose window crosses a
  reset, the way `cvae_skip_reset_frames` does.
- Record the VecNorm state used during collection and reuse it, or record raw
  observations and normalise at training time. Do not mix.

### 2. `training/pretrain_diffusion_bc.py`

Epsilon-MSE objective (paper Eq. 5):

```
L = MSE( eps^k , eps_theta(A^0 + eps^k, cond, k) ),  k ~ U{1..K_train}
```

Paper's optimiser settings: AdamW `betas=(0.95, 0.999)`, `eps=1e-8`, lr 1e-4, weight
decay 1e-6 for the CNN variant, cosine LR with 500 warmup steps, EMA `power=0.75`,
`max_value=0.9999`.

Writes a checkpoint that `DiffusionPolicyConfig(weights_path=...)` loads, so switching
between cold and warm start is one config field.

## The velocity-to-position label conversion is free

The demonstrator was trained with `flipper_control_mode: velocity`, and we are switching
to `position`. That looks like it should make the demonstrations unusable — it does not.

What we record is not the demonstrator's *command* but the **achieved flipper angle
trajectory**, which is exactly the position-mode action after normalising:

```
unit = 2 * (theta - low) / (high - low) - 1
```

with `low, high` from `FtrEnv.flipper_angle_bounds()` (the asymmetric MARV limits:
front up 60 / down 80, back up 60 / down 80 degrees). `v` and `w` carry over unchanged.

So a velocity-mode demonstrator produces perfectly valid position-mode labels, and the
position-control switch costs nothing at the BC stage. This is a concrete reason the
switch was cheap to take.

One caveat: the demonstrator's angle trajectory is inherently rate-limited to 5 deg/step,
so the BC-pretrained policy will start out smooth. That is a good initialisation but it
means the RL stage is where any snapping would appear — do not conclude from a smooth BC
policy that open question 5 is settled.
