#!/usr/bin/env python3
"""Print the best trial's parameters from an Optuna study as YAML frozen_params.

Usage:
    python scripts/optuna_best_params.py <study_name> [--top K]

Examples:
    python scripts/optuna_best_params.py ftr_v3_stage1_rewards
    python scripts/optuna_best_params.py ftr_v3_stage1_rewards --top 5
"""
import argparse
import sys
from pathlib import Path

import optuna
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent


def get_storage():
    db_path = ROOT / "optuna_db.yaml"
    if not db_path.exists():
        print(f"ERROR: {db_path} not found.", file=sys.stderr)
        sys.exit(1)
    db_secret = OmegaConf.load(db_path)
    if "url" in db_secret:
        conn_str = db_secret["url"]
    else:
        sslmode = db_secret.get("sslmode", "require")
        conn_str = (
            f"postgresql+psycopg2://{db_secret['db_user']}:{db_secret['db_password']}"
            f"@{db_secret['db_host']}:{db_secret['db_port']}/{db_secret['db_name']}?sslmode={sslmode}"
        )
    return optuna.storages.RDBStorage(conn_str)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study_name", help="Optuna study name")
    parser.add_argument("--top", type=int, default=1, help="Show top K trials (default: 1)")
    args = parser.parse_args()

    storage = get_storage()
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    )

    if not trials:
        print("No completed trials found.", file=sys.stderr)
        sys.exit(1)

    for i, trial in enumerate(trials[: args.top]):
        print(f"# Trial #{trial.number}  value={trial.value:.4f}")
        if i == 0:
            print("# Copy the block below into stage 2 config under frozen_params:")
            print("frozen_params:")
        for k, v in sorted(trial.params.items()):
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
