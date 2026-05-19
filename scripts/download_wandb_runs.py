"""Download W&B run history as CSVs.

Each run is saved under:
    <output_dir>/<run_name>/train.csv
    <output_dir>/<run_name>/rew.csv
    <output_dir>/<run_name>/env.csv
    <output_dir>/<run_name>/eval.csv
    <output_dir>/<run_name>/config.yaml
    ...

The topic split (train/, rew/, etc.) mirrors how RunLogger writes local CSVs.

Usage examples:
    # Download all runs
    python scripts/download_wandb_runs.py

    # Only runs whose name contains a substring
    python scripts/download_wandb_runs.py --filter crossing_potential

    # Only runs with a specific tag
    python scripts/download_wandb_runs.py --tag ppo

    # Limit how many rows are sampled per run (default: all rows)
    python scripts/download_wandb_runs.py --samples 500

    # Choose output directory (default: runs/wandb_export/)
    python scripts/download_wandb_runs.py --out runs/my_export
"""

import argparse
import csv
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import wandb
from omegaconf import OmegaConf

PROJECT = "flipper_training"
STEP_COL = "log_step"


def split_into_topics(rows: list[dict]) -> dict[str, list[dict]]:
    """Group columns by their topic prefix (everything before the last '/')."""
    if not rows:
        return {}

    # Collect all column names across all rows
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys.discard(STEP_COL)
    all_keys.discard("_step")
    all_keys.discard("_timestamp")

    topics: dict[str, set] = defaultdict(set)
    for key in all_keys:
        if "/" in key:
            topic = key.rsplit("/", maxsplit=1)[0]
        else:
            topic = "misc"
        topics[topic].add(key)

    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        step = row.get(STEP_COL) or row.get("_step")
        for topic, keys in topics.items():
            topic_row = {k: row[k] for k in keys if k in row}
            if topic_row:
                topic_row[STEP_COL] = step
                result[topic].append(topic_row)

    return result


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    # Collect all fieldnames, step column first
    fieldnames = [STEP_COL] + [k for k in rows[0] if k != STEP_COL]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def download_run(run, out_dir: Path, samples: int | None):
    run_dir = out_dir / run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        OmegaConf.save(OmegaConf.create(dict(run.config)), config_path)

    # history — scan_history gives every logged row; history(samples=N) down-samples
    if samples is None:
        rows = list(run.scan_history())
    else:
        rows = run.history(samples=samples, pandas=False)

    if not rows:
        print(f"  [skip] {run.name} — no history rows")
        return

    topics = split_into_topics(rows)
    for topic, topic_rows in topics.items():
        write_csv(run_dir / f"{topic}.csv", topic_rows)

    print(f"  [ok]   {run.name}  ({len(rows)} rows, topics: {sorted(topics)})")


def main():
    parser = argparse.ArgumentParser(description="Download W&B runs as CSVs")
    parser.add_argument("--filter", "-f", default=None, help="Only runs whose name contains this string")
    parser.add_argument("--tag", "-t", default=None, help="Only runs with this tag")
    parser.add_argument("--state", default=None, choices=["finished", "running", "crashed", "failed"], help="Filter by run state")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Max rows per run (default: all)")
    parser.add_argument("--out", "-o", default="runs/wandb_export", help="Output directory")
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()

    api = wandb.Api()
    filters: dict = {}
    if args.tag:
        filters["tags"] = args.tag
    if args.state:
        filters["state"] = args.state

    runs = api.runs(args.project, filters=filters or None)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for run in runs:
        if args.filter and args.filter not in run.name:
            continue
        download_run(run, out_dir, args.samples)
        downloaded += 1

    print(f"\nDone. {downloaded} run(s) → {out_dir}")


if __name__ == "__main__":
    main()
