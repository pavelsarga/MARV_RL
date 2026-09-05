# 05 — Phase 2: diffusion head + DPPO

## The head

The same `ConditionalUnet1D`, now predicting noise: `eps_theta(A^k, cond, k)` where
`cond = concat(global_cond, SinusoidalPosEmb(k))`.

## Schedule and sampler

`policies/diffusion_schedule.py` — a minimal squared-cosine DDPM schedule (iDDPM, Nichol
& Dhariwal, which the paper found worked best) plus a DDIM sampler. ~120 lines.

We write it rather than adding `diffusers` as a dependency: the apptainer image
`containers/isaaclab_optuna.sif` is fixed, and rebuilding it to add a pip package is a
separate job with its own risk. The subset we need — betas, alphas_cumprod, the DDIM
update, epsilon-prediction, and clipping the predicted `A^0` to `[-1, 1]` — is small and
easy to unit-test.

- `K_train = 100`, `K_infer = 8` (strided subset of the 100 training steps).
- Epsilon prediction, per the paper's Eq. 5.
- Actions normalised to `[-1, 1]` per dimension, which the env's action spec already is.

## RL objective — DPPO

PPO needs `log pi(a|s)`, which a diffusion sampler does not give directly. DPPO (Ren et
al. 2024) resolves this by treating the denoising chain as an MDP: each DDIM step is a
Gaussian

```
pi(A^{k-1} | A^k, s) = N( mu_theta(A^k, s, k), sigma_k^2 I )
```

so the chain has a tractable likelihood.

**Storage.** The actor stores the full chain `A^K ... A^0` in the rollout tensordict under
`denoise_chain`, shape `[N, K_infer+1, T_p, A]`. At `N=1024, K=8, T_p=16, A=6` that is
~3.5 MB per macro-step, ~450 MB per iteration of 128 macro-steps, on the CPU replay
buffer. Fine. Log-probs are recomputed from the chain at update time so the gradient
flows.

**2A (start here).** Compute the joint chain-level log-prob, write it to
`sample_log_prob`, and reuse **`ClipPPOLoss` verbatim**. One ratio per macro-step, exactly
the shape PPO expects.

**2B (only if 2A is unstable).** Per-denoising-step clipping, which is what DPPO actually
advocates: treat each denoising transition as its own PPO sample with zero reward except
at `k=0`, and clip each ratio separately. Needs a custom loss module. The trigger to
escalate is `train/mean_kl_approx` blowing up or `mean_clip_fraction` pinning near 1.

**`K_ft`.** DPPO fine-tunes only the last few denoising steps to cut gradient variance and
memory. Start with all 8; make it a config field. Open question 8.

## Exploration and entropy

Diffusion entropy is not tractable, so:

- Set `ppo_opts.entropy_bonus: false`. `ClipPPOLoss` would otherwise call the
  distribution's `entropy()`, which we cannot provide.
- Exploration comes from a **floor on the sampling sigma** (`min_sampling_std`, DPPO's
  `min_sampling_denoising_std`). This is the knob that replaces `entropy_coef`.

## Health diagnostics — have these in from the first run

C-TRAC's lesson is in CLAUDE.md: a posterior-collapsed C-VAE latent looked fine on the
success curve for 22M frames while the entire contact-estimation architecture contributed
nothing. The analogous silent failure here is a chain that has collapsed to a
deterministic map, or one whose later steps do nothing.

Log every iteration:

- `diff/sigma_k` for each `k` — the actual sampling std used.
- `diff/chain_delta_k` = mean `|A^{k-1} - A^k|` per step. If the late steps are ~0, the
  chain is doing nothing and `K_infer` can be cut; if the early steps are ~0, the schedule
  is wrong.
- `diff/chain_logprob_std` — collapse shows up as this going to zero.
- `train/mean_kl_approx`, `train/mean_clip_fraction` as usual.

## What stays the same

Critic, GAE, the chunked env, the observation history, the reward, the terrain, and the
frame accounting are all unchanged from Phase 1. Phase 2 is a head swap plus a loss
change, nothing else.
