#!/usr/bin/env python3
"""Print an eval's success-rate table (env type x difficulty column) from eval_per_spot.csv.

    python scripts/eval_spot_table.py logs/policy_eval_v2 [--eval_id calib_marv_rl_2] [--metric mean_steps]

Episodes are pooled over repeats (weighted by episodes_total), so the numbers are the
per-spot success over everything that eval collected. With --by-direction the forward (-X)
and reverse (+X) legs of a `bidirectional` course are shown separately, from
eval_episodes.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def read_grown_csv(path: Path) -> pd.DataFrame:
    """read_csv that tolerates rows with MORE columns than the header — what an
    eval_per_spot.csv looks like when PerSpotRow gained fields while evals kept
    appending to the same file. Extra values get generic names."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    width = max(len(r) for r in body) if body else len(header)
    header = header + [f"extra_{k}" for k in range(len(header), width)]
    body = [r + [""] * (width - len(r)) for r in body]
    df = pd.DataFrame(body, columns=header)
    for c in df.columns:
        if c in ("eval_id", "policy", "terrain", "env_type_name", "outcome", "direction"):
            continue
        df[c] = pd.to_numeric(df[c].replace("", "nan"), errors="coerce")
    return df


def spot_table(ps: pd.DataFrame, metric: str = "success_rate") -> pd.DataFrame:
    def pooled(x):
        w = x.episodes_total
        return (x[metric] * w).sum() / w.sum() if w.sum() else float("nan")

    g = ps.groupby(["env_type_idx", "env_type_name", "depth_col"]).apply(pooled).rename(metric).reset_index()
    pv = g.pivot_table(index="env_type_name", columns="depth_col", values=metric, dropna=False)
    order = g.drop_duplicates("env_type_name").sort_values("env_type_idx").env_type_name
    return pv.reindex(order)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--eval_id", default=None, help="one eval_id (default: every eval_id in the file, in turn)")
    ap.add_argument("--metric", default="success_rate", help="column of eval_per_spot.csv to pool (success_rate, failure_rate, mean_steps, ...)")
    ap.add_argument("--by-direction", action="store_true", help="split forward/reverse legs (needs eval_episodes.csv with a direction column)")
    args = ap.parse_args()

    ps = read_grown_csv(args.output_dir / "eval_per_spot.csv")
    ids = [args.eval_id] if args.eval_id else list(dict.fromkeys(ps.eval_id))
    pd.set_option("display.width", 200)
    for eid in ids:
        sub = ps[ps.eval_id == eid]
        print(f"\n=== {eid}  terrain={sub.terrain.iloc[0]}  repeats={sub.repeat.nunique()}  "
              f"episodes={int(sub.episodes_total.sum())}  metric={args.metric}")
        tbl = spot_table(sub, args.metric)
        print(tbl.round(2).to_string())
        if args.metric == "success_rate":
            print(f"overall: {(sub.success_rate * sub.episodes_total).sum() / sub.episodes_total.sum():.3f}")
        if args.by_direction:
            ep = read_grown_csv(args.output_dir / "eval_episodes.csv")
            ep = ep[ep.eval_id == eid]
            if "direction" not in ep.columns:
                print("(no direction column in eval_episodes.csv)")
                continue
            for d, e in ep.groupby("direction"):
                agg = e.assign(s=(e.outcome == "success").astype(float)).groupby(["env_type_idx", "env_type_name", "depth_col"]).s.mean().reset_index()
                pv = agg.pivot_table(index="env_type_name", columns="depth_col", values="s", dropna=False)
                order = agg.drop_duplicates("env_type_name").sort_values("env_type_idx").env_type_name
                print(f"--- direction={d}  ({len(e)} episodes)")
                print(pv.reindex(order).round(2).to_string())


if __name__ == "__main__":
    main()
