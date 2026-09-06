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

**Measured ratio sensitivity (`training/test_dppo.py`).** The chain log-prob sums
`K x T_p x A = 192` Gaussian terms, so it was worth checking whether the chain-level ratio
is usable at all before committing to it. Perturbing eps_theta and re-scoring a stored
chain, against the clip threshold `log(1.2) = 0.182`:

| perturbation of eps_theta | 2A chain-level | 2B per-step |
|---|---|---|
| 1e-4 | 0.016 nats | — |
| **3e-4** (about one Adam step at lr 3e-4) | **0.039 nats, 0% clip** | **0.010 nats, 0% clip** |
| 1e-3 | 0.232 nats — clips | — |
| 1e-2 | 5.06 nats — clips hard | — |

So **2A is usable**: at realistic step sizes it sits well inside the trust region, and only
saturates once parameter steps exceed ~1e-3. 2B is a further ~4x less sensitive and remains
the safety valve if the chain-level KL misbehaves in practice. Both are implemented
(`DPPOClipLoss`, `DPPOPerStepClipLoss`) and tested; start at 2A as originally planned.

⚠ An earlier revision of this document claimed the opposite — that 2A would "very likely
saturate", citing 3.2 and 10.5 nats. Those figures were wrong. The test perturbed eps_theta
and never restored it, so every later measurement compared against a log-prob captured
under different parameters and reported the *setup* perturbation rather than the one under
study. The test now restores state, and asserts that per-step log-probs sum to the chain
log-prob, which is the invariant that would have caught it.

**A hypothesis that did not survive either:** `min_sampling_std` was expected to be a strong
lever, since `log N(x; mu, sigma)` sensitivity to a shift in `mu` scales as `1/sigma^2`.
Sweeping it 0.02 -> 0.2 moved per-step sensitivity only 0.007 -> 0.004 nats, because DDIM's
own `sigma_k` is above the floor for most of the chain and the floor rarely binds. Treat it
as an exploration knob only, which is what DPPO uses it for.

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


## First real run (job 11500855, 24 iterations, 64 envs)

Phase 2 runs end to end against real physics: `DPPOClipLoss` selected, exit 0, mid-training
and final eval both fine, actor 1.69M parameters.

**The chain is doing real work.** `diff/chain_delta_k*` at the last iteration:

```
k0 1.032  k1 0.692  k2 0.556  k3 0.463  k4 0.370  k5 0.283  k6 0.194  k7 0.073
```

Monotonically decaying — large early corrections tapering to fine ones, which is what a
well-formed denoising schedule should look like. No dead steps at either end, so `K_infer=8`
is neither wasted compute nor too short. This is the diagnostic that rules out the C-TRAC
failure mode (a component that contributes nothing while the success curve looks fine), and
it is worth reading on every run.

**`target_kl` binds**: `train/epochs_run` alternates 1 and 2, so the guard is cutting
iterations short.

**Open concern.** KL sits at 0.05-0.08 and `clip_fraction` at 0.44-0.49, higher than Phase 1
at the same point and above the 0.03 target even with early stopping — which can only stop
*after* an epoch exceeds, not prevent it within one. So the instinct behind the retracted
"start at 2B" claim may have been directionally right even though the measurement supporting
it was wrong. Test `per_step_clipping: true` and a lower actor LR before any long Phase 2
run; both are one config line.

Note the diagnostics land in `diff.csv`, not `action.csv` — RunLogger splits CSVs by the
topic prefix before the slash.


## Settled: use per-denoising-step clipping (2B)

Head-to-head, 24-iteration smoke runs identical apart from the flag:

| iteration | chain-level (2A) | per-step (2B) |
|---|---|---|
| 1 | kl 0.062, clip 0.44 | **kl 0.010, clip 0.06** |
| 12 | kl 0.053, clip 0.42 | **kl 0.019, clip 0.08** |
| 20-24 | kl 0.082, clip 0.49 | **kl 0.024, clip 0.12** |

Per-step stays under `target_kl` throughout and runs its full epoch budget; chain-level sits
2-3x over and is repeatedly cut short. `per_step_clipping: true` is now the default.

**Why the earlier measurement pointed the other way, twice.** The static sweep in
`test_dppo.py` perturbs `eps_theta` with random noise and reports the chain log-prob moving
only 0.039 nats under a realistic step — comfortably inside the trust region. A real
optimiser step is *correlated* across parameters, and correlated perturbations move a
192-term log-prob sum far more than random ones of the same norm. The sweep is a lower
bound, not an estimate. The lesson generalises: a perturbation study substitutes a
distribution of your choosing for the one the optimiser actually produces, and when those
differ the study is measuring the wrong thing.
