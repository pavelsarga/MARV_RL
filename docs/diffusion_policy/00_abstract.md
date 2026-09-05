# Receding-horizon control for MARV — abstract plan

## Why

The current MARV policy (`marv_rl` module, `MLPPolicyConfig`, PPO in `train_ftr.py`) is a
memoryless single-step controller: one 966-D observation in, one 6-D action
`[v, w, fl, fr, rl, rr]` out, at 10 Hz (`sim_dt 0.005 x decimation 20`). Every control
step is decided from scratch. Flipper commands are angular velocities integrated onto the
current angle at 5 deg/step.

We want the **receding-horizon formulation** of Chi et al., *Diffusion Policy*
(`paper_repos/diffusion_policy.pdf`): the network sees the last `T_o` observations,
predicts a `T_p`-step action trajectory, and executes the first `T_a` steps open-loop
before re-planning.

What we expect from it:

- **Temporal consistency.** A flipper manoeuvre (mount a step, bridge a gap) is a
  multi-step commitment. A single-step policy can dither; a chunk cannot.
- **Latency tolerance.** The paper measures graceful degradation up to 4 steps of
  latency under position control. That matters for real deployment.
- **Multimodality.** Two valid ways over an obstacle average badly in a unimodal
  Gaussian policy. A diffusion head represents both.

## What we keep, what we drop

**Keep** — the `marv_rl` observation (966-D: 945 heightmap 45x21 + 21 proprio), the tuned
`marv_rl` reward, the actor-critic structure, PPO, and `MarvRLCNNFlatEncoder` as the
per-frame observation encoder.

**Keep from the paper** — action chunking / receding horizon, the CNN backbone (1-D
temporal U-Net over the horizon axis), FiLM conditioning, position control, DDIM
sampling.

**Drop** — the time-series diffusion transformer. The paper's own recommendation is to
"start with the CNN-based Diffusion Policy implementation"; the transformer needs
per-task tuning of attention dropout and weight decay, degrades when you add layers, and
lost on real Push-T (0.53 vs 0.80 IoU). Its contribution is not clearly established.

**Drop** — imitation learning as the training signal. We train with RL. BC is used only
as a warm start (see `06_bc_pretrain.md`).

## Notation

| symbol | meaning | default | source |
|---|---|---|---|
| `T_o` | observation history length | 2 | paper (>2 hurt their CNN variant) |
| `T_p` | prediction horizon | 16 | paper |
| `T_a` | execution horizon | 4 | paper uses 8 at 10 Hz; MARV is on obstacles, start shorter |
| `K_train` | training diffusion steps | 100 | paper |
| `K_infer` | DDIM inference steps | 8 | paper quotes 10 -> 0.1 s on a 3080 |
| `A` | action dim | 6 | `[v, w, fl, fr, rl, rr]` |
| control dt | | 0.1 s | unchanged - do NOT touch `decimation` |

One episode is 300 control steps (`episode_length_s 30`), so ~75 macro-steps at `T_a=4`.

## Staging

The project is deliberately split so that "does chunking help?" and "does diffusion
help?" are separable questions, and so a failure in the second does not sink the first.

**Phase 1 — Gaussian chunk actor.** Build the entire receding-horizon scaffold: the
`T_o` observation history, the FiLM-conditioned Conv1d U-Net, the chunked environment
wrapper, macro-step discounting. The head emits a Gaussian over the flattened `T_p x A`
chunk, so `ClipPPOLoss` and `GAE` are used unchanged. This is a complete, evaluable
result on its own.

**Phase 2 — Diffusion head.** The same U-Net becomes `eps_theta(A^k, cond, k)`. Sampling
is DDIM. Training uses the DPPO formulation (Ren et al. 2024): each denoising step is a
Gaussian, so the chain has a tractable likelihood and PPO still applies.

**Warm start.** Phase 2 is BC-pretrained on chunks harvested from an already-trained
`marv_rl` policy, then RL fine-tuned. DPPO itself always fine-tunes a BC-pretrained
diffusion model; a randomly initialised `eps_theta` emits noise chunks and the gradients
through the chain are high-variance.

## Sub-plans

| doc | scope |
|---|---|
| `01_action_chunking.md` | chunked env wrapper, discounting, done/auto-reset semantics, frame accounting |
| `02_obs_history.md` | `T_o` rolling window, transform ordering, VecNorm interaction |
| `03_network.md` | per-frame encoder -> FiLM Conv1d U-Net, sizing, latency budget |
| `04_phase1_gaussian.md` | Gaussian chunk actor under existing PPO |
| `05_phase2_diffusion.md` | DDIM sampler + DPPO objective |
| `06_bc_pretrain.md` | chunk dataset collection + epsilon-MSE pretraining |
| `07_deployment.md` | ROS2 receding-horizon inference (later) |

## Open questions register

Nothing here is settled. Update the status column as each is resolved.

| # | question | current guess | resolve when | status |
|---|---|---|---|---|
| 1 | `T_a` — how long dare we go open-loop on an obstacle? | 4 (0.4 s) | sweep in Phase 1 | open |
| 2 | `T_p` vs `T_a` ratio | 16 / 4 | after Phase 1 baseline | open |
| 3 | `T_o` — does history help at all, given the heightmap is already exteroceptive? | 2 | Phase 1 ablation vs `T_o=1` | open |
| 4 | U-Net `down_dims` | `[64,128]` — 1.9x baseline actor cost; `[64,128,256]` was 4.4x | measured on a laptop GPU; re-run `scripts/bench_diffusion_head.py` on the A100 | **provisionally resolved** |
| 5 | Does removing the 5 deg/step slew limit need a replacement rate penalty? | probably yes | first Phase 1 run | open |
| 6 | Reward coefficients were tuned for 10 Hz single-step — re-tune under macro-steps? | assume no; `step_penalty` is the suspect | after first full run | open |
| 7 | DPPO chain-level (2A) vs per-denoising-step (2B) clipping | start 2A | Phase 2 | open |
| 8 | `K_ft` — fine-tune all 8 denoising steps or only the last few? | all 8 | Phase 2 | open |
| 9 | Is the mid-chunk-termination idling artifact measurable in success rate? | assume negligible | compare `T_a=1` vs `T_a=4` | open |
| 10 | Does the BC warm start bias the policy to the demonstrator's local optimum? | monitor | Phase 2 | open |
| 11 | Should the critic see the whole chunk, or only `s_t`? | only `s_t` (V, not Q) | if advantages look badly biased | open |
| 12 | Is `TanhNormal` over a 96-D flattened chunk well-behaved? | log-prob stable at ~-64, no NaNs on the smoke run | re-check over a full run | **provisionally resolved** |
| 13 | Phase 2's actor cost is ~12x the baseline per control step (K=8 U-Net passes). Does that matter once PhysX is counted? | probably not dominant | profile a real iteration | open |
| 14 | Chunking gives ~T_a times fewer gradient steps for the same frame budget. Underfit? | watch for a still-rising curve at the end | first full Phase 1 run | open |
| 15 | `entropy_coef` rescaled by 1/T_p because entropy is now summed over 96 dims, not 6. Is per-dimension parity the right target? | yes, by arithmetic; but `clip_fraction` hit 0.54 on the smoke run | first full Phase 1 run | open |
| 16 | Actor LR was tuned for a 0.25M MLP head and now drives a 1.3M U-Net. Re-tune? | untouched so far | after the entropy fix, if `clip_fraction` is still high | open |
| 17 | Should the crash force-exit be backported to the other five trainers? | it is the same four lines and they are all affected | user's call — they carry tuned baselines | open |

## Validated on the cluster (RCI, 2026-09-06)

Phase 1 runs end to end. Job `diff_smoke2_11494004` on `gpufast`: 64 envs, 3 iterations,
12288 control frames, exit 0 in 1m34s, including a mid-training eval and a final eval.

Confirmed working: the chunked env wrapper in real Isaac, the CatFrames window, the FiLM
U-Net actor and critic, GAE over macro steps, the PPO update, checkpointing
(`policy_step_4096.pth` / `policy_step_12288.pth` — correctly named in **control** frames),
and `run_tracked_rollout` through the chunked env (`eval/rollout_steps: 20`, in macro
steps, as designed).

Two things the run taught us:

1. **The encoder rank bug** (see the warning in `03_network.md`) — found only because the
   smoke test went all the way to GAE.
2. **`entropy_coef` needed rescaling by 1/T_p.** `train/mean_entropy` came back at 64.2
   nats — 0.67 per dimension across the 96-D chunk, essentially log(2), the maximum for a
   TanhNormal on [-1,1] — where the baseline's 6-D action gives about 4 nats. The same
   coefficient would therefore apply ~16x the entropy pressure. `train/mean_clip_fraction`
   climbed 0.38 → 0.45 → 0.54 over the three iterations, well above the healthy 0.1-0.3
   band, which is the symptom. Three iterations at 64 envs cannot establish causation, so
   treat the fix as arithmetic (which is solid) and the diagnosis as provisional.

3. **A crashed job does not terminate.** The failing run held its GPU node for 15 minutes
   after the Python process was finished, until cancelled by hand — Isaac Sim's atexit
   handlers deadlock on interpreter shutdown, and only the success path had an `os._exit`.
   Fixed in `train_diffusion.py`; still present in the five other trainers. See the warning
   in `04_phase1_gaussian.md`.

`action/chunk_step_delta` read 0.73 with a max of 1.99 on a [-1, 1] action — i.e. the
predicted chunk is near-white-noise across the horizon. That is exactly right for an
untrained policy whose sampling scale is 1.0, and it is the number that should fall as
training proceeds. If it does not, chunking is buying nothing.

## Decisions already taken

1. Staged Gaussian-first, diffusion-second (above).
2. BC warm start before RL fine-tuning.
3. **Flipper position control** (`flipper_control_mode: position`, already implemented at
   `ftr_env.py:688`). The paper is explicit that temporal convolutions have a
   low-frequency bias and handle velocity-command sequences badly, and that position
   control consistently beats velocity control for chunked prediction.
4. ROS2 deployment designed for now, implemented later.
5. Control rate stays 10 Hz. Changing `decimation` would break comparability with the
   `marv_rl` baseline, which is the only reference point we have.
