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
  - The representative attempt's weights/ is pruned to ONE consistent final snapshot --
    every weight component the trainer wrote, renamed up to <component>_final.pth from the
    highest step all components share if no *_final.pth exists -- plus training_state.pth,
    dropping the many intermediate step checkpoints. Components are discovered from the
    filenames, so this covers policy/vecnorm (PPO, D3QN), qvalue and cvae (SAC / C-TRAC),
    icm (ICM-D3QN) and creps_state (CREPS) without naming any of them. Crash dumps
    (*_crash.pth) and a persisted replay buffer are skipped: neither is needed to evaluate
    or resume, and the buffer is tens of GB.

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
# Any "<component>_step_<frames>.pth" / "<component>_final.pth", NOT a fixed component list.
# The trainers in this project write different sets: PPO/D3QN write policy+vecnorm, SAC/C-TRAC
# adds qvalue+cvae, ICM-D3QN adds icm, CREPS adds creps_state. Hardcoding (policy|vecnorm) --
# which this did -- silently dropped every non-PPO component on the way into experiments/,
# so a curated C-TRAC dir had no C-VAE and could not be evaluated or resumed at all.
STEP_RE = re.compile(r"^(?P<component>.+)_step_(?P<frames>\d+)\.pth$")
FINAL_RE = re.compile(r"^(?P<component>.+)_final\.pth$")
# Written next to the weights but deliberately not carried over: crash dumps are debugging
# artifacts, and a persisted replay buffer is tens of GB of training-only state.
SKIP_WEIGHT_NAMES = {"replay_buffer.pt"}
SKIP_EXTRA_NAMES = {"replay_buffer"}


def find_attempts(log_dir: Path) -> list[Path]:
    attempts = [d for d in log_dir.iterdir() if d.is_dir() and ATTEMPT_RE.match(d.name)]
    return sorted(attempts, key=lambda d: int(ATTEMPT_RE.match(d.name).group(1)))


def copy_pruned_weights(src: Path, dst: Path, dry_run: bool, log) -> None:
    """Keep one consistent snapshot of EVERY weight component, plus training_state.pth.

    Components are discovered from the filenames rather than hardcoded, so a trainer that
    starts writing a new one is picked up with no change here.

    "Consistent" is the load-bearing word: the components of a snapshot must all come from
    the same moment. Pairing, say, a policy from 52.4M frames with a C-VAE from 51.9M gives
    an actor whose embedded encoder is not the one it was trained against. So the selection
    is, in order:
      1. every component has a *_final.pth  -> use those
      2. otherwise the highest step number present for ALL step-based components
      3. otherwise (no shared step) fall back per component, loudly
    """
    if not src.is_dir():
        return
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    steps: dict[str, dict[int, Path]] = {}
    finals: dict[str, Path] = {}
    for f in sorted(src.glob("*.pth")):
        if f.name in SKIP_WEIGHT_NAMES or f.name == "training_state.pth" or f.name.endswith("_crash.pth"):
            continue
        if (m := STEP_RE.match(f.name)):
            steps.setdefault(m.group("component"), {})[int(m.group("frames"))] = f
        elif (m := FINAL_RE.match(f.name)):
            finals[m.group("component")] = f

    components = sorted(set(steps) | set(finals))
    if not components:
        log(f"  WARNING: no weight components found in {src}")
        return

    chosen: dict[str, Path] = {}
    if components and all(c in finals for c in components):
        log(f"  components {components}: using *_final.pth for all")
        chosen = {f"{c}_final.pth": finals[c] for c in components}
    else:
        shared = None
        for c in components:
            if c not in steps:
                continue
            shared = set(steps[c]) if shared is None else shared & set(steps[c])
        if shared:
            n = max(shared)
            log(f"  components {components}: using shared step {n} -> *_final.pth")
            for c in components:
                if c in steps and n in steps[c]:
                    chosen[f"{c}_final.pth"] = steps[c][n]
                elif c in finals:
                    chosen[f"{c}_final.pth"] = finals[c]
        else:
            log(f"  WARNING: components {components} share no common step in {src}; "
                f"falling back per component -- the snapshot may be INCONSISTENT")
            for c in components:
                if c in finals:
                    chosen[f"{c}_final.pth"] = finals[c]
                elif steps.get(c):
                    n = max(steps[c])
                    log(f"    {c}: highest step {n}")
                    chosen[f"{c}_final.pth"] = steps[c][n]

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
        if entry.name in skip_names or entry.name in SKIP_EXTRA_NAMES or entry.suffix == ".csv":
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
