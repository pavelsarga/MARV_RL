# 05 — Phase 2: diffusion head + DPPO

## The head

The same `ConditionalUnet1D`, now predicting noise: `eps_theta(A^k, cond, k)` where
`cond = concat(global_cond, SinusoidalPosEmb(k))`.

## Schedule and sampler

`policies/diffusion_schedule.py` — **built and tested**. Squared-cosine DDPM schedule
(iDDPM, Nichol & Dhariwal, which the paper found worked best) plus a strided DDIM sampler.
`ddim_step` returns the transition's **(mean, std)**, not just the next sample, because DPPO
scores each denoising step as a Gaussian. That also forces `eta > 0` — a deterministic DDIM
has sigma=0 and no log-prob — and makes `min_sampling_std` the exploration knob replacing
`entropy_coef`. Verified with an oracle epsilon: the chain reconstructs `x0` to 0.015.

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

**2A (start here).** ⚠ An earlier draft of this plan said `ClipPPOLoss` could be reused
*verbatim* here. That is wrong. `ClipPPOLoss` recomputes log-probs by calling
`actor(tensordict)` and asking the resulting distribution for `log_prob(action)` — and a
diffusion actor cannot answer that: the quantity to score is the stored denoising chain,
not the final action, and calling the actor again merely draws a fresh chain. There is no
distribution object that fixes this without smuggling the network inside it.

The correct minimal design is a small `ClipPPOLoss` **subclass** overriding the log-weight
computation: read `denoise_chain` `[N, K+1, T_p, A]` from the tensordict, re-run `eps_theta`
at each `k` on the **stored** `A^k` to get `mu_k`, sum `log N(A^{k-1}; mu_k, sigma_k)` over
the chain, and ratio that against the stored `sample_log_prob`. Roughly 80 lines. GAE, the
critic, the chunked env and the observation window are all untouched.

**⚠ Measured before running anything: 2A will very likely saturate.** The chain log-prob
sums `K x T_p x A = 8 x 4 x 6 = 192` Gaussian terms, so its sensitivity to eps_theta
compounds. Perturbing eps_theta and re-scoring a stored chain
(`training/test_dppo.py`, section 5):

| perturbation | mean change in chain log-prob | vs clip threshold `log(1.2) = 0.182` |
|---|---|---|
| 1e-4 | 0.108 nats | inside the trust region |
| 1e-3 | 3.22 nats | clips immediately |
| 1e-2 | 12.17 nats | clips hard |

The usable parameter-step budget is therefore around 1e-4, while Adam at `lr = 5e-4` takes
per-parameter steps of roughly `lr` — already past it. Once every sample clips, `clamp` has
zero gradient outside its bounds and the objective goes flat: observed directly, with
eps_theta receiving a gradient norm of 8e-17 while the critic absorbed 12.5.

(Caveat: the sweep perturbs with random noise, whereas a real gradient step is correlated,
so treat this as an order of magnitude rather than an exact threshold.)

**So start at 2B, not 2A.** Per-denoising-step clipping keeps each ratio over `T_p x A = 24`
dims instead of 192, which is what DPPO advocates and why. 2A remains implemented and
tested (`DPPOClipLoss`) and is worth one run as the ablation that demonstrates the problem,
but it should not be the default. The alternative levers are the Phase 1 lesson applied
again — fewer updates per iteration — and a much lower actor LR.

**2B.** Per-denoising-step clipping, which is what DPPO actually
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
