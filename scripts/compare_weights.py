#!/usr/bin/env python3
"""Compare two .pth weight files and report per-layer and summary statistics."""

import argparse
import sys
from pathlib import Path

import torch


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def short_key(key: str, max_len: int = 60) -> str:
    return key if len(key) <= max_len else "…" + key[-(max_len - 1):]


def compare(a: dict, b: dict) -> None:
    keys_a, keys_b = set(a), set(b)

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a:
        print(f"\n[!] Keys only in A ({len(only_a)}): {sorted(only_a)[:5]}")
    if only_b:
        print(f"[!] Keys only in B ({len(only_b)}): {sorted(only_b)[:5]}")

    common = sorted(keys_a & keys_b)
    print(f"\n{'Layer':<62} {'shapeA':>14}  {'normA':>9} {'normB':>9} {'Δnorm':>9} {'maxΔ':>9} {'cos_sim':>8}")
    print("-" * 120)

    total_params = 0
    total_delta_l2 = 0.0
    cos_sims = []

    for key in common:
        va, vb = a[key].float(), b[key].float()
        if va.shape != vb.shape:
            print(f"  {short_key(key):<62} shape mismatch: {va.shape} vs {vb.shape}")
            continue

        delta = vb - va
        norm_a = va.norm().item()
        norm_b = vb.norm().item()
        delta_norm = delta.norm().item()
        max_delta = delta.abs().max().item()

        flat_a, flat_b = va.flatten(), vb.flatten()
        denom = flat_a.norm() * flat_b.norm()
        cos_sim = (flat_a @ flat_b / denom).item() if denom > 0 else float("nan")

        total_params += va.numel()
        total_delta_l2 += delta_norm ** 2
        cos_sims.append(cos_sim)

        print(
            f"  {short_key(key):<62} {str(tuple(va.shape)):>14}  "
            f"{norm_a:>9.4f} {norm_b:>9.4f} {delta_norm:>9.4f} {max_delta:>9.4f} {cos_sim:>8.5f}"
        )

    print("-" * 120)

    if total_params > 0:
        rms_delta = (total_delta_l2 / total_params) ** 0.5
        mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else float("nan")
        print(f"\nSummary over {len(common)} shared tensors ({total_params:,} params):")
        print(f"  RMS weight change (per element) : {rms_delta:.6f}")
        print(f"  Mean cosine similarity          : {mean_cos:.6f}")
        print(f"  Layers with cos_sim < 0.99      : {sum(1 for c in cos_sims if c < 0.99)}")
        print(f"  Layers with cos_sim < 0.90      : {sum(1 for c in cos_sims if c < 0.90)}")
        print(f"  Layers with cos_sim < 0.50      : {sum(1 for c in cos_sims if c < 0.50)}")
        print(f"  Layers with cos_sim < 0.30      : {sum(1 for c in cos_sims if c < 0.30)}")
        print(f"  Layers with cos_sim < 0.20      : {sum(1 for c in cos_sims if c < 0.20)}")
        print(f"  Layers with cos_sim < 0.10      : {sum(1 for c in cos_sims if c < 0.10)}")
        most_changed = sorted(
            [(a[k].float() - b[k].float()).norm().item() / max(a[k].float().norm().item(), 1e-9) for k in common
             if a[k].shape == b[k].shape],
        )
        print(f"  Max relative Δ (norm(Δ)/norm(A)): {most_changed[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare two .pth weight files.")
    parser.add_argument("a", help="First .pth file (baseline)")
    parser.add_argument("b", help="Second .pth file (comparison)")
    args = parser.parse_args()

    path_a, path_b = Path(args.a), Path(args.b)
    for p in (path_a, path_b):
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    print(f"A: {path_a}")
    print(f"B: {path_b}")

    a, b = load(str(path_a)), load(str(path_b))
    print(f"   {len(a)} tensors          {len(b)} tensors")

    compare(a, b)


if __name__ == "__main__":
    main()
