#!/usr/bin/env python3
"""
Backfill trial_user_attributes in an Optuna SQLite DB from SLURM .out log files.

Parses lines of the form:
  Trial 5 done — score=0.0484  components={'eval/success_rate': 0.074, 'eval/shock_normalised': 0.0002}

and inserts eval/success_rate, eval/shock_normalised, and score as user attrs
for any trial that is missing them.

Usage:
  python scripts/backfill_optuna_user_attrs.py \
      --db optuna/optuna.db \
      --logs logs/optuna_ftr_10891096 \
      --study ftr_optuna_smooth_v1 \
      [--dry-run]
"""

import argparse
import ast
import re
import sqlite3
from pathlib import Path

LOG_PATTERN = re.compile(
    r"Trial\s+(\d+)\s+done\s+[—-]+\s+score=([0-9eE+.\-]+)\s+components=(\{.*\})"
)


def parse_logs(log_dir: Path) -> dict[int, dict]:
    """Return {trial_number: {key: value}} from all .out files under log_dir."""
    parsed: dict[int, dict] = {}
    for out_file in sorted(log_dir.rglob("*.out")):
        for line in out_file.read_text(errors="replace").splitlines():
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            trial_num = int(m.group(1))
            score = float(m.group(2))
            try:
                components: dict = ast.literal_eval(m.group(3))
            except (ValueError, SyntaxError):
                print(f"  WARNING: could not parse components on line: {line.strip()}")
                continue
            if trial_num in parsed:
                # keep last occurrence (most recent run wins)
                pass
            parsed[trial_num] = {"score": score, **components}
    return parsed


def backfill(db_path: Path, log_dir: Path, study_name: str, dry_run: bool) -> None:
    parsed = parse_logs(log_dir)
    if not parsed:
        print("No 'Trial X done' lines found in logs — nothing to backfill.")
        return
    print(f"Parsed {len(parsed)} trial result(s) from logs.")

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Resolve study_id
    row = cur.execute(
        "SELECT study_id FROM studies WHERE study_name = ?", (study_name,)
    ).fetchone()
    if row is None:
        print(f"ERROR: study '{study_name}' not found in DB.")
        con.close()
        return
    study_id = row[0]

    # Build map: trial_number -> trial_id for COMPLETE trials in this study
    db_trials = {
        num: tid
        for tid, num in cur.execute(
            "SELECT trial_id, number FROM trials WHERE study_id = ? AND state = 'COMPLETE'",
            (study_id,),
        ).fetchall()
    }
    print(f"Found {len(db_trials)} COMPLETE trial(s) in DB for study '{study_name}'.")

    inserted = skipped = 0
    for trial_num, attrs in sorted(parsed.items()):
        if trial_num not in db_trials:
            print(f"  Trial {trial_num}: not in DB as COMPLETE — skipping.")
            continue
        trial_id = db_trials[trial_num]

        for key, value in attrs.items():
            existing = cur.execute(
                "SELECT 1 FROM trial_user_attributes WHERE trial_id = ? AND key = ?",
                (trial_id, key),
            ).fetchone()
            if existing:
                skipped += 1
                continue
            value_json = str(value)
            if not dry_run:
                cur.execute(
                    "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, ?, ?)",
                    (trial_id, key, value_json),
                )
            print(f"  {'[DRY] ' if dry_run else ''}Trial {trial_num}: INSERT {key} = {value_json}")
            inserted += 1

    if not dry_run:
        con.commit()
    con.close()

    print(f"\nDone: {inserted} attr(s) inserted, {skipped} already present (skipped).")
    if dry_run:
        print("(dry-run — no changes written)")


def main():
    ap = argparse.ArgumentParser(description="Backfill Optuna user attrs from SLURM logs.")
    ap.add_argument("--db", default="optuna/optuna.db", help="Path to Optuna SQLite DB")
    ap.add_argument("--logs", default="logs/optuna_ftr_10891096", help="Log directory to scan")
    ap.add_argument("--study", default="ftr_optuna_smooth_v1", help="Optuna study name")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be inserted without writing")
    args = ap.parse_args()

    ws = Path("/home/robot/workspaces/robot_rodeo_gym_ws")
    db_path  = Path(args.db)  if Path(args.db).is_absolute()   else ws / args.db
    log_dir  = Path(args.logs) if Path(args.logs).is_absolute() else ws / args.logs

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        return
    if not log_dir.exists():
        print(f"ERROR: Log directory not found at {log_dir}")
        return

    print(f"DB  : {db_path}")
    print(f"Logs: {log_dir}")
    print(f"Study: {args.study}")
    print()
    backfill(db_path, log_dir, args.study, args.dry_run)


if __name__ == "__main__":
    main()
