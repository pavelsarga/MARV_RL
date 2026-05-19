#!/usr/bin/env python3
"""Average the weights of two or more .pth policy files and save the result."""

import argparse
import sys
from pathlib import Path

import torch


def load(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=True)


def average_weights(state_dicts: list[dict]) -> dict:
    reference = state_dicts[0]
    averaged = {}
    skipped = []

    for key in reference:
        tensors = []
        for sd in state_dicts:
            if key not in sd:
                break
            if sd[key].shape != reference[key].shape:
                break
            tensors.append(sd[key].float())
        else:
            averaged[key] = torch.stack(tensors).mean(dim=0)
            # preserve original dtype
            averaged[key] = averaged[key].to(reference[key].dtype)
            continue
        skipped.append(key)

    if skipped:
        print(f"[!] Skipped {len(skipped)} keys (missing or shape mismatch): {skipped[:5]}")

    return averaged


def main():
    parser = argparse.ArgumentParser(
        description="Average weights of two or more .pth files."
    )
    parser.add_argument("inputs", nargs="+", help="Input .pth files (at least 2)")
    parser.add_argument("-o", "--output", required=True, help="Output .pth file path")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Optional per-file weights for weighted average (must match number of inputs)",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("Error: at least two input files are required.", file=sys.stderr)
        sys.exit(1)

    if args.weights is not None and len(args.weights) != len(args.inputs):
        print("Error: --weights count must match number of input files.", file=sys.stderr)
        sys.exit(1)

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {len(paths)} checkpoints...")
    state_dicts = [load(p) for p in paths]
    print(f"  {[len(sd) for sd in state_dicts]} tensors per file")

    if args.weights is not None:
        total = sum(args.weights)
        w = [x / total for x in args.weights]
        print(f"Weighted average with normalised weights: {[f'{x:.3f}' for x in w]}")
        # scale each state dict by its weight then sum
        reference = state_dicts[0]
        averaged = {}
        skipped = []
        for key in reference:
            tensors = []
            for sd, wi in zip(state_dicts, w):
                if key not in sd or sd[key].shape != reference[key].shape:
                    break
                tensors.append(sd[key].float() * wi)
            else:
                averaged[key] = torch.stack(tensors).sum(dim=0).to(reference[key].dtype)
                continue
            skipped.append(key)
        if skipped:
            print(f"[!] Skipped {len(skipped)} keys: {skipped[:5]}")
    else:
        print("Uniform average...")
        averaged = average_weights(state_dicts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, str(out))
    print(f"Saved {len(averaged)} tensors → {out}")


if __name__ == "__main__":
    main()
