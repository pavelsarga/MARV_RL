# 03 — Network

## Shape

```
obs_history [N, T_o, 966]
   |
   +-- MarvRLCNNFlatEncoder, weights SHARED across the T_o frames, applied per frame
   |     945-D heightmap -> view(1,45,21) -> CNN -> 128
   |      21-D proprio   -> concat        -> fusion MLP -> 128
   |
   -> [N, T_o, 128] -> flatten -> global_cond [N, T_o*128 = 256]

Phase 2 only:
   k -> SinusoidalPosEmb(64) -> Linear -> Mish -> Linear -> [N, 64]
   cond = concat(global_cond, k_emb)                  # 256 (P1) / 320 (P2)

ConditionalUnet1D over the action sequence [N, A=6, T_p=16]
   Conv1dBlock   = Conv1d(k=5, p=2) -> GroupNorm(8) -> Mish
   CondResBlock  = Conv1dBlock -> FiLM(gamma,beta = Linear(cond, 2*C)) -> Conv1dBlock
                   + Conv1d(in,out,1) residual shortcut
   down_dims [64, 128, 256]; 2 res blocks per level
   downsample Conv1d(k=3, s=2); upsample ConvTranspose1d(k=4, s=2)
   2 res blocks in the mid block
   head: Conv1dBlock -> Conv1d(-> out_channels, 1)
```

## Why the observation encoder does not change

`MarvRLCNNFlatEncoder` (`rl_modules/marv_rl/marv_rl_cnn_flat_encoder.py:56`) already does
the whole perception job: a 3-layer CNN over the 45x21 heightmap to 128 dims, concatenated
with the raw 21-D proprioceptive state, through a fusion MLP configured by
`ftr_obs_encoder_opts`. We reuse it verbatim, with **shared weights across the `T_o`
frames** (the paper encodes each observation timestep independently and concatenates).

This answers the "will the network get too deep?" worry directly: the added depth is the
**1-D U-Net along the horizon axis**, not a second spatial CNN. FiLM conditions that
U-Net; it does not sit between the observation encoder and anything else. There is no
extra CNN after the observation encoder.

## FiLM

Per Perez et al., applied channel-wise at every conditional residual block:

```
[gamma, beta] = Linear(cond)                 # -> 2 * C_out
h             = gamma * GroupNorm(Conv1d(h)) + beta
```

with `gamma, beta` broadcast across the temporal (horizon) axis. This is the mechanism
the paper uses to inject the observation *and* the denoising step into a network whose
input is only the action sequence — which is what makes the encoder run once per chunk
rather than once per denoising step.

## Sizing — measured, not guessed

`scripts/bench_diffusion_head.py`, RTX 2000 Ada laptop GPU, `T_o=2 T_p=16 T_a=4 K_infer=8`,
batch 1024. "x" is the actor-forward cost per **control** step relative to the current
marv_rl actor (293k params, 1.49 ms at this batch):

| `down_dims` | U-Net params | P1 ms/macro | P1 x | P2 ms/macro | P2 x |
|---|---|---|---|---|---|
| `[64, 128]` | 1.30M | 11.4 | **1.9** | 73.5 | 12.3 |
| `[64, 128, 256]` | 4.75M | 26.0 | 4.4 | 190.1 | 31.9 |
| `[128, 256, 512]` | 17.1M | 60.2 | 10.1 | 463.1 | 77.7 |
| paper (real-world) `[256,512,1024]` | 67M | — | — | — | — |

**`[64, 128]` is the default**: it is the only size that meets the <=2x gate. The
observation encoder adds 259k parameters on top (unchanged from the baseline).

Two caveats before treating this as final:

- These are actor-forward times **in isolation**. The PhysX step at 512-1024 envs still
  dominates total wall clock, so a 4x actor cost may be a much smaller fraction of a
  training iteration. Re-run the benchmark on the A100 before ruling a larger head out.
- The baseline actor is small enough to be partly kernel-launch-bound, which flatters the
  ratios on a small GPU.

`down_dims`, `kernel_size`, `n_groups` and `diffusion_step_embed_dim` are all config
fields. Do not hardcode.

## Latency budget

At 10 Hz with `T_a=4`, a chunk covers **400 ms**, so `K_infer=8` denoising steps get
**50 ms each** on the deployment GPU. That is comfortable for a <=2.5M-param U-Net over a
16-step sequence. The paper reports 0.1 s for 10 DDIM steps of a 67M-param network on a
3080.

The encoder runs **once per chunk**, not once per denoising step. That is the paper's key
speed trick (excluding the observation from the denoising output) and it is why the
`K`-loop only touches the U-Net.

## The real risk is training throughput, not deployment latency

Per control step the actor cost goes from

```
enc + small MLP          ->        (enc + K * unet) / T_a
```

At `K=8, T_a=4` that is 2 U-Net passes per control step plus a quarter of an encoder pass.
Whether that matters depends on how it compares to the PhysX step at 512-1024 envs, which
currently dominates.

**Measure before choosing.** `scripts/bench_diffusion_head.py` is a standalone,
Isaac-free benchmark that times the encoder and each candidate U-Net at
`num_robots in {512, 1024}` and reports ms per macro-step plus the projected slowdown
against the current actor. Gate for the default size:

- projected training slowdown <= ~2x the current actor cost, and
- `K_infer` forward passes <= `T_a * 100 ms` at batch 1 (deployment).

**Measured deployment latency (batch 1, 8 DDIM steps, laptop RTX 2000 Ada):** 15 ms for
`[64,128]`, 23 ms for `[64,128,256]`, 27 ms for `[128,256,512]` — against a 400 ms budget
at `T_a=4`. Deployment latency is a non-issue at every size considered; roughly 25x
headroom on hardware weaker than anything this would deploy on. Phase 2's cost lands
entirely on training throughput.

## Critic

A plain `MLP` on its own copy of the obs-history embedding, emitting one value per
macro-step. `share_encoder: false`, as in the baseline. No diffusion in the critic — it
runs once per macro-step and is cheap. Open question 11 tracks whether V(s_t) is enough
or whether the critic should see the chunk.
