#!/usr/bin/env python3
"""Benchmark the receding-horizon policy head against the current marv_rl actor.

Standalone: no Isaac Sim, no env, no config — just the networks. Run it BEFORE committing
to a U-Net size, because the binding constraint on this architecture is training
throughput, not deployment latency.

    PYTHONPATH=src/flipper_training:src/FTR-Benchmark python scripts/bench_diffusion_head.py

What it reports, per candidate ``down_dims``:

  * ms per macro step at the training batch size (Phase 1: one encoder pass over the T_o
    window plus one U-Net pass; Phase 2: the same encoder pass plus K_infer U-Net passes,
    since the encoder runs once per chunk, not once per denoising step).
  * The same figure amortised per CONTROL step (divided by T_a) and the resulting slowdown
    against the baseline actor, which is the number to judge a size by.
  * Deployment latency at batch 1, to check K_infer passes fit inside T_a control steps.

Gates from docs/diffusion_policy/03_network.md:
  * projected training slowdown <= ~2x the current actor cost;
  * batch-1 Phase 2 latency <= T_a * control_dt.
"""

from __future__ import annotations

import argparse
import time

import torch

from marv_rl_training.policies import MLP
from marv_rl_training.policies.diffusion_policy import ConditionalUnet1D, ObsHistoryEncoder
from rl_modules.marv_rl.marv_rl_cnn_flat_encoder import MarvRLCNNFlatEncoder

OBS_DIM = 966
ACTION_DIM = 6
# Matches ftr_obs_encoder_opts in configs/baselines/marv_config_marv_rl.yaml.
ENCODER_OPTS = dict(num_hidden=3, hidden_dim=256, output_dim=128, layernorm=True)


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@torch.no_grad()
def timeit(fn, warmup: int, iters: int, device: torch.device) -> float:
    """Mean wall-clock milliseconds per call."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--T_o", type=int, default=2, help="observation history length")
    ap.add_argument("--T_p", type=int, default=16, help="prediction horizon")
    ap.add_argument("--T_a", type=int, default=4, help="execution horizon")
    ap.add_argument("--K_infer", type=int, default=8, help="DDIM inference steps (Phase 2)")
    ap.add_argument("--control-dt", type=float, default=0.1, help="seconds per control step (10 Hz)")
    ap.add_argument("--down-dims", type=str, nargs="+", default=["64,128", "64,128,256", "128,256,512"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  T_o={args.T_o} T_p={args.T_p} T_a={args.T_a} K_infer={args.K_infer}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print()

    # ---- baseline: current marv_rl actor (single-step encoder + 2x128 MLP head) ----
    base_encoder = MarvRLCNNFlatEncoder(input_dim=OBS_DIM, **ENCODER_OPTS).to(device).eval()
    base_head = MLP(in_dim=128, out_dim=2 * ACTION_DIM, hidden_dim=128, num_hidden=2, layernorm=False).to(device).eval()
    print(f"baseline actor: {count(base_encoder) + count(base_head):,} params")

    # ---- candidate heads ----
    cand = {}
    for spec in args.down_dims:
        dims = [int(x) for x in spec.split(",")]
        if args.T_p % (2 ** (len(dims) - 1)) != 0:
            print(f"  skipping down_dims={dims}: T_p={args.T_p} not divisible by {2 ** (len(dims) - 1)}")
            continue
        enc = ObsHistoryEncoder(MarvRLCNNFlatEncoder(input_dim=OBS_DIM, **ENCODER_OPTS), OBS_DIM, args.T_o).to(device).eval()
        # Phase 1 emits 2*A channels (loc, scale); Phase 2 emits A (predicted noise).
        unet_p1 = ConditionalUnet1D(ACTION_DIM, 2 * ACTION_DIM, args.T_p, enc.output_dim, dims).to(device).eval()
        unet_p2 = ConditionalUnet1D(ACTION_DIM, ACTION_DIM, args.T_p, enc.output_dim, dims).to(device).eval()
        cand[spec] = (dims, enc, unet_p1, unet_p2)
        print(f"  down_dims={dims}: encoder {count(enc):,} + unet(P1) {count(unet_p1):,} params")
    print()

    budget_ms = args.T_a * args.control_dt * 1000.0

    for batch in args.batch_sizes:
        obs1 = torch.randn(batch, OBS_DIM, device=device)
        obsT = torch.randn(batch, args.T_o * OBS_DIM, device=device)

        base_ms = timeit(lambda: base_head(base_encoder(obs1)), args.warmup, args.iters, device)
        print(f"=== batch {batch} ===")
        print(f"  baseline actor            {base_ms:8.3f} ms / control step")
        header = (
            f"  {'down_dims':<16}{'P1 ms/macro':>13}{'P1 ms/ctrl':>12}{'P1 x':>7}"
            f"{'P2 ms/macro':>13}{'P2 ms/ctrl':>12}{'P2 x':>7}"
        )
        print(header)
        for spec, (dims, enc, unet_p1, unet_p2) in cand.items():
            seq = torch.randn(batch, ACTION_DIM, args.T_p, device=device)

            def p1():
                unet_p1(seq, enc(obsT))

            def p2():
                # The encoder runs ONCE per chunk; only the U-Net is inside the K loop.
                cond = enc(obsT)
                for _ in range(args.K_infer):
                    unet_p2(seq, cond)

            p1_ms = timeit(p1, args.warmup, args.iters, device)
            p2_ms = timeit(p2, args.warmup, args.iters, device)
            print(
                f"  {str(dims):<16}{p1_ms:13.3f}{p1_ms / args.T_a:12.3f}{p1_ms / args.T_a / base_ms:7.2f}"
                f"{p2_ms:13.3f}{p2_ms / args.T_a:12.3f}{p2_ms / args.T_a / base_ms:7.2f}"
            )
        print()

    # ---- deployment: batch 1, one robot ----
    print(f"=== deployment (batch 1) — budget is T_a * control_dt = {budget_ms:.0f} ms ===")
    obsT1 = torch.randn(1, args.T_o * OBS_DIM, device=device)
    for spec, (dims, enc, _unet_p1, unet_p2) in cand.items():
        seq = torch.randn(1, ACTION_DIM, args.T_p, device=device)

        def infer():
            cond = enc(obsT1)
            for _ in range(args.K_infer):
                unet_p2(seq, cond)

        ms = timeit(infer, args.warmup, max(args.iters, 50), device)
        verdict = "OK" if ms <= budget_ms else "OVER BUDGET"
        print(f"  {str(dims):<16}{ms:9.3f} ms for {args.K_infer} DDIM steps   {verdict}")


if __name__ == "__main__":
    main()
