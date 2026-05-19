#!/usr/bin/env python3
"""Generate a deployable training config from the best Optuna trial.

Loads the best trial from an Optuna SQLite study, merges its hyperparameters
into a base training config, and writes the result as a complete YAML file
ready for production training.

Usage:
    python scripts/deploy_best_config.py [options]

Options:
    --study NAME        Optuna study name (default: most recent study in DB)
    --db PATH           SQLite DB path (default: <ws>/optuna/optuna.db)
    --base-config PATH  Base config YAML (default: <ws>/configs/ftr_config_A_new.yaml)
    --out PATH          Output YAML path (default: auto-versioned in configs/)
    --top K             Print top K trials, write only the best (default: 1)
    --dry-run           Print config to stdout without writing any file

Examples:
    python scripts/deploy_best_config.py --dry-run
    python scripts/deploy_best_config.py --study ftr_optuna_smooth_v2 --top 3
    python scripts/deploy_best_config.py --out configs/my_best.yaml
"""
import argparse
import re
import sys
from pathlib import Path

import optuna
from omegaconf import OmegaConf

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent


def _latest_study_name(db_path: Path) -> str:
    import sqlite3

    con = sqlite3.connect(db_path)
    row = con.execute("SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 1").fetchone()
    con.close()
    if row is None:
        raise RuntimeError(f"No studies found in {db_path}")
    return row[0]


def _next_version_path(configs_dir: Path) -> Path:
    existing = list(configs_dir.glob("ftr_config_optuna_best_v*.yaml"))
    nums = []
    for p in existing:
        m = re.search(r"_v(\d+)", p.stem)
        if m:
            nums.append(int(m.group(1)))
    next_v = (max(nums) + 1) if nums else 1
    return configs_dir / f"ftr_config_optuna_best_v{next_v}.yaml"


def _best_trials(study: optuna.Study, top_k: int) -> list[optuna.Trial]:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"Study '{study.study_name}' has no completed trials.")
    return sorted(complete, key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)[:top_k]


def _build_config(base_cfg, trial: optuna.Trial) -> OmegaConf:
    dotlist = [f"{k}={v}" for k, v in trial.params.items()]
    trial_overrides = OmegaConf.from_dotlist(dotlist)
    return OmegaConf.merge(base_cfg, trial_overrides)


def _metadata_comment(study: optuna.Study, trial: optuna.Trial) -> str:
    lines = [
        f"# Best Optuna trial — study: {study.study_name}, trial #{trial.number}",
        f"# Objective value : {trial.value:.6f}  (0.65*success_rate - 0.35*shock_p99_norm)",
    ]
    attrs = trial.user_attrs
    sr = attrs.get("eval/success_rate") or attrs.get("success_rate")
    shock = attrs.get("eval/shock_p99_normalised") or attrs.get("shock_p99_normalised")
    if sr is not None:
        lines.append(f"# success_rate    : {float(sr):.4f}")
    if shock is not None:
        lines.append(f"# shock_p99_norm  : {float(shock):.4f}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default=None, help="Optuna study name (default: most recent)")
    parser.add_argument("--db", type=Path, default=ROOT / "optuna" / "optuna.db")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "ftr_config_A_new.yaml")
    parser.add_argument("--out", type=Path, default=None, help="Output YAML path (default: auto-versioned)")
    parser.add_argument("--top", type=int, default=1, help="Print top K trials (write only the best)")
    parser.add_argument("--dry-run", action="store_true", help="Print config without writing")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found at {args.db}", file=sys.stderr)
        sys.exit(1)
    if not args.base_config.exists():
        print(f"ERROR: base config not found at {args.base_config}", file=sys.stderr)
        sys.exit(1)

    study_name = args.study or _latest_study_name(args.db)
    storage = f"sqlite:///{args.db.resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    trials = _best_trials(study, args.top)

    print(f"Study : {study.study_name}")
    print(f"Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} complete")
    print()

    if args.top > 1:
        print(f"Top {args.top} trials:")
        for rank, t in enumerate(trials, 1):
            sr = t.user_attrs.get("eval/success_rate") or t.user_attrs.get("success_rate", "?")
            print(f"  #{rank:2d}  trial={t.number:4d}  value={t.value:.4f}  success_rate={sr}")
        print()

    best = trials[0]
    sr = best.user_attrs.get("eval/success_rate") or best.user_attrs.get("success_rate", "?")
    print(f"Best → trial #{best.number}  value={best.value:.4f}  success_rate={sr}")
    print()

    base_cfg = OmegaConf.load(args.base_config)
    cfg = _build_config(base_cfg, best)

    comment = _metadata_comment(study, best)
    yaml_text = comment + OmegaConf.to_yaml(cfg, resolve=False)

    out_path = args.out or _next_version_path(ROOT / "configs")

    if args.dry_run:
        print("─" * 60)
        print(yaml_text)
        print("─" * 60)
        print("(dry-run — no file written)")
    else:
        out_path.write_text(yaml_text)
        print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
