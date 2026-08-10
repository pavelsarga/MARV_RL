#!/usr/bin/env python3
"""Turn a raw logs/<job_name>_<job_id>/ training run into a curated experiments/<name>/ dir.

A logs/ run has one attempt_N/ per SLURM resubmission of the *same* config, each with its own
full set of per-step CSVs, its own copy of config.yaml, and every checkpoint pth the trainer ever
wrote. That's convenient while a job is running but noisy to keep around long-term: the per-step
CSVs are identical in schema across attempts (just different runs of the same config), the
config.yaml copies are identical byte-for-byte, and only the last checkpoint is ever loaded for
eval. This script produces the smaller, eval-ready shape used under experiments/ (see
experiments/baselines/atd3qn for a hand-made example):

  - Every *.csv is merged across all attempts into one file at the top level with a leading
    "attempt" column, instead of N separate copies of the same columns.
  - Weights, config.yaml, and any other per-attempt output (e.g. an eval_id/ dir) come from a
    single representative attempt only -- the last one (highest attempt_N) -- flattened straight
    into the output dir. No attempt_N/ nesting is kept; the rest of the attempts only contribute
    their csv rows.
  - The representative attempt's weights/ is pruned to just the final checkpoint
    (policy_final.pth + vecnorm_final.pth, renamed up from the highest
    policy_step_*.pth/vecnorm_step_*.pth if no *_final.pth was ever written) plus
    training_state.pth, dropping the many intermediate step checkpoints.

The source logs/ dir is only ever read, never modified.

Usage:
    python scripts/logs_to_experiment.py <log_run_dir> <experiment_out_dir> [--force] [--dry-run]

Examples:
    python scripts/logs_to_experiment.py logs/train_marv_atd3qn_11278667 experiments/baselines/atd3qn
    python scripts/logs_to_experiment.py logs/train_marv_mitriakov_11279578 experiments/thesis/mitriakov --dry-run
"""

import argparse
import csv
import filecmp
import re
import shutil
import sys
from pathlib import Path

ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")
STEP_RE = re.compile(r"^(policy|vecnorm)_step_(\d+)\.pth$")


def find_attempts(log_dir: Path) -> list[Path]:
    attempts = [d for d in log_dir.iterdir() if d.is_dir() and ATTEMPT_RE.match(d.name)]
    return sorted(attempts, key=lambda d: int(ATTEMPT_RE.match(d.name).group(1)))


def copy_pruned_weights(src: Path, dst: Path, dry_run: bool, log) -> None:
    if not src.is_dir():
        return
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    final_policy, final_vecnorm = src / "policy_final.pth", src / "vecnorm_final.pth"
    if final_policy.is_file() and final_vecnorm.is_file():
        chosen = {"policy_final.pth": final_policy, "vecnorm_final.pth": final_vecnorm}
    else:
        policies, vecnorms = {}, {}
        for f in src.glob("policy_step_*.pth"):
            if (m := STEP_RE.match(f.name)):
                policies[int(m.group(2))] = f
        for f in src.glob("vecnorm_step_*.pth"):
            if (m := STEP_RE.match(f.name)):
                vecnorms[int(m.group(2))] = f
        common = sorted(set(policies) & set(vecnorms))
        if not common:
            log(f"  WARNING: no weights found in {src}")
            return
        n = common[-1]
        log(f"  no *_final.pth in {src}, using highest step {n} -> policy_final.pth/vecnorm_final.pth")
        chosen = {"policy_final.pth": policies[n], "vecnorm_final.pth": vecnorms[n]}

    training_state = src / "training_state.pth"
    if training_state.is_file():
        chosen["training_state.pth"] = training_state

    for name, path in chosen.items():
        log(f"  copy {path} -> {dst / name}")
        if not dry_run:
            shutil.copy2(path, dst / name)


def merge_csvs(attempts: list[Path], out_dir: Path, dry_run: bool, log) -> None:
    csv_names = sorted({f.name for a in attempts for f in a.glob("*.csv")})
    for name in csv_names:
        fieldnames: list[str] = []
        rows: list[dict] = []
        for a in attempts:
            csv_path = a / name
            if not csv_path.is_file():
                continue
            attempt_idx = int(ATTEMPT_RE.match(a.name).group(1))
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for field in reader.fieldnames or []:
                    if field not in fieldnames:
                        fieldnames.append(field)
                for row in reader:
                    row["attempt"] = attempt_idx
                    rows.append(row)

        dest = out_dir / name
        log(f"  merge {name}: {len(rows)} rows from {sum(1 for a in attempts if (a / name).is_file())} attempts -> {dest}")
        if dry_run:
            continue
        with open(dest, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["attempt", *fieldnames], restval="")
            writer.writeheader()
            writer.writerows(rows)


def copy_extra_entries(src: Path, dst: Path, skip_names: set, dry_run: bool, log) -> None:
    """Copy everything in src except CSVs, weights/, and config.yaml (handled separately)."""
    for entry in src.iterdir():
        if entry.name in skip_names or entry.suffix == ".csv":
            continue
        dest = dst / entry.name
        log(f"  copy {entry} -> {dest}")
        if dry_run:
            continue
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest)


def check_configs_match(attempts: list[Path], log) -> None:
    configs = [a / "config.yaml" for a in attempts if (a / "config.yaml").is_file()]
    if len(configs) < 2:
        return
    reference = configs[0]
    for other in configs[1:]:
        if not filecmp.cmp(reference, other, shallow=False):
            log(f"  WARNING: {other} differs from {reference} -- attempts are not identical configs; "
                f"keeping {reference} as the single merged config.yaml anyway")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", type=Path, help="source logs/<job_name>_<job_id>/ directory")
    parser.add_argument("out_dir", type=Path, help="destination experiments/... directory to create")
    parser.add_argument("--force", action="store_true", help="overwrite out_dir if it already exists")
    parser.add_argument("--dry-run", action="store_true", help="print planned actions without writing anything")
    args = parser.parse_args()

    def log(msg):
        print(msg)

    log_dir: Path = args.log_dir
    out_dir: Path = args.out_dir

    if not log_dir.is_dir():
        print(f"error: {log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if out_dir.exists():
        if not args.force:
            print(f"error: {out_dir} already exists (pass --force to overwrite)", file=sys.stderr)
            sys.exit(1)
        log(f"removing existing {out_dir}")
        if not args.dry_run:
            shutil.rmtree(out_dir)

    attempts = find_attempts(log_dir)
    if not attempts:
        print(f"error: no attempt_N/ subdirs found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True)

    check_configs_match(attempts, log)

    # Top-level files that sit next to the attempt_N/ dirs (launch config yaml, .out/.err).
    for f in log_dir.iterdir():
        if f.is_file():
            log(f"copy {f} -> {out_dir / f.name}")
            if not args.dry_run:
                shutil.copy2(f, out_dir / f.name)

    primary = attempts[-1]
    log(f"representative attempt: {primary.name} (weights/config/extras) -- {len(attempts)} attempt(s) total")

    reference_config = primary / "config.yaml"
    if reference_config.is_file():
        log(f"copy {reference_config} -> {out_dir / 'config.yaml'}")
        if not args.dry_run:
            shutil.copy2(reference_config, out_dir / "config.yaml")

    if len(attempts) == 1:
        for csv_f in sorted(primary.glob("*.csv")):
            dest = out_dir / csv_f.name
            log(f"copy {csv_f} -> {dest}")
            if not args.dry_run:
                shutil.copy2(csv_f, dest)
    else:
        merge_csvs(attempts, out_dir, args.dry_run, log)

    copy_pruned_weights(primary / "weights", out_dir / "weights", args.dry_run, log)
    copy_extra_entries(primary, out_dir, skip_names={"weights", "config.yaml"}, dry_run=args.dry_run, log=log)

    log(f"\n{'would produce' if args.dry_run else 'produced'}: {out_dir}")


if __name__ == "__main__":
    main()
