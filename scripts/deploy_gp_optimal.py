#!/usr/bin/env python3
"""Generate a deployable training config from GP-predicted optimal hyperparameters.

Loads completed trials from an Optuna SQLite study, fits a Gaussian Process
surrogate on the scalarised score, finds the GP posterior optimum via multi-start
L-BFGS-B, and writes a complete training config YAML.

Two GPs are fitted:
  Main GP    — all params present in ≥50% of trials, using all those trials.
  Secondary  — params present in <50% of trials (added mid-study), on the
               subset of trials that have them. Their predicted values are
               merged on top of the main GP result.

Usage:
    python scripts/deploy_gp_optimal.py [options]

Options:
    --study NAME        Optuna study name (default: most recent in DB)
    --db PATH           SQLite DB path (default: <ws>/optuna/optuna.db)
    --base-config PATH  Base config YAML (default: <ws>/configs/ftr_config_A_new.yaml)
    --out PATH          Output YAML path (default: auto-versioned in configs/)
    --n-restarts N      GP optimisation restarts (default: 500)
    --dry-run           Print config to stdout without writing any file

Examples:
    python scripts/deploy_gp_optimal.py --dry-run
    python scripts/deploy_gp_optimal.py --study ftr_optuna_smooth_v2 --n-restarts 1000
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from omegaconf import OmegaConf
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent

# Metric weights — must match optuna_ftr_smooth.yaml
METRIC_WEIGHTS = {
    "eval/success_rate":         0.65,
    "eval/shock_p99_normalised": -0.35,
}

# Params whose last name fragment implies log-scale (must be positive)
LOG_SCALE_KEYWORDS = {"lr", "entropy_coef", "shock_coef"}

# Params present in fewer than this fraction of trials are treated as mid-study
MID_STUDY_FRAC = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _latest_study_name(db_path: Path) -> str:
    import sqlite3
    con = sqlite3.connect(db_path)
    row = con.execute("SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 1").fetchone()
    con.close()
    if row is None:
        raise RuntimeError(f"No studies found in {db_path}")
    return row[0]


def _next_version_path(configs_dir: Path) -> Path:
    existing = list(configs_dir.glob("ftr_config_gp_optimal_v*.yaml"))
    nums = []
    for p in existing:
        m = re.search(r"_v(\d+)", p.stem)
        if m:
            nums.append(int(m.group(1)))
    next_v = (max(nums) + 1) if nums else 1
    return configs_dir / f"ftr_config_gp_optimal_v{next_v}.yaml"


def _build_dataframe(trials: list, component_keys: list) -> pd.DataFrame:
    records = []
    for t in trials:
        score = t.values[0] if t.values else t.user_attrs.get("score", float("nan"))
        row = {"trial": t.number, "score": score}
        for k in component_keys:
            row[k] = t.user_attrs.get(k, float("nan"))
        row.update(t.params)
        records.append(row)
    return pd.DataFrame(records).sort_values("trial").reset_index(drop=True)


def _classify_params(df: pd.DataFrame, non_param_cols: set):
    param_names = [c for c in df.columns if c not in non_param_cols]
    categorical = set()
    for name in param_names:
        col = df[name]
        if not np.issubdtype(col.dtype, np.number) or col.nunique() <= 6:
            categorical.add(name)
    continuous = [p for p in param_names if p not in categorical]
    categorical = [p for p in param_names if p in categorical]
    log_params = {
        p for p in continuous
        if p.rsplit(".", 1)[-1] in LOG_SCALE_KEYWORDS and df[p].dropna().min() > 0
    }
    return param_names, continuous, categorical, log_params


def _fit_gp(param_cols: list, data_df: pd.DataFrame,
            categorical_params: list, log_params: set,
            n_restarts: int, tag: str = "") -> dict:
    """Fit GP and find posterior optimum. Returns a result dict."""
    X_raw = data_df[param_cols].values.astype(float)
    y     = data_df["score"].values

    log_mask = np.array([p in log_params for p in param_cols])
    X_gp = X_raw.copy()
    X_gp[:, log_mask] = np.log(np.clip(X_gp[:, log_mask], 1e-10, None))

    X_min   = X_gp.min(axis=0)
    X_max   = X_gp.max(axis=0)
    X_range = np.where(X_max - X_min > 0, X_max - X_min, 1.0)
    X_norm  = (X_gp - X_min) / X_range

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X_norm.shape[1])) + WhiteKernel(0.01)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                   alpha=1e-6, normalize_y=True)
    gp.fit(X_norm, y)
    print(f"  {tag}GP fitted ({len(data_df)} trials × {len(param_cols)} params) "
          f"— log-ML: {gp.log_marginal_likelihood_value_:.2f}")

    rng = np.random.default_rng(42)
    best_val, best_xn = -np.inf, None
    for _ in range(n_restarts):
        x0  = rng.random(X_norm.shape[1])
        res = minimize(lambda x: -gp.predict(x.reshape(1, -1))[0],
                       x0, bounds=[(0, 1)] * X_norm.shape[1], method="L-BFGS-B")
        if -res.fun > best_val:
            best_val, best_xn = -res.fun, res.x.copy()

    best_xo = best_xn * X_range + X_min
    best_xo[log_mask] = np.exp(best_xo[log_mask])
    for i, p in enumerate(param_cols):
        if p in categorical_params:
            cats = sorted(data_df[p].dropna().unique())
            best_xo[i] = min(cats, key=lambda c: abs(c - best_xo[i]))

    gp_mean, gp_std = gp.predict(best_xn.reshape(1, -1), return_std=True)
    return {
        "gp": gp,
        "param_cols": param_cols,
        "best_params": dict(zip(param_cols, best_xo)),
        "predicted_score": float(gp_mean[0]),
        "predicted_std":   float(gp_std[0]),
        "observed_best":   float(y.max()),
        "n_trials":        len(data_df),
    }


def _metadata_comment(study_name: str, result_main: dict,
                       result_secondary: dict | None) -> str:
    lines = [
        f"# GP-optimal config — study: {study_name}",
        f"# Main GP     : {result_main['n_trials']} trials × "
        f"{len(result_main['param_cols'])} params",
        f"#   predicted score : {result_main['predicted_score']:.4f} "
        f"± {result_main['predicted_std']:.4f}",
        f"#   observed best   : {result_main['observed_best']:.4f}",
    ]
    if result_secondary:
        lines += [
            f"# Secondary GP: {result_secondary['n_trials']} trials × "
            f"{len(result_secondary['param_cols'])} params "
            f"({', '.join(p.rsplit('.', 1)[-1] for p in result_secondary['param_cols'])})",
            f"#   predicted score : {result_secondary['predicted_score']:.4f} "
            f"± {result_secondary['predicted_std']:.4f}",
        ]
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study",       default=None)
    parser.add_argument("--db",          type=Path, default=ROOT / "optuna" / "optuna.db")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "ftr_config_A_new.yaml")
    parser.add_argument("--out",         type=Path, default=None)
    parser.add_argument("--n-restarts",  type=int,  default=500)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found at {args.db}", file=sys.stderr); sys.exit(1)
    if not args.base_config.exists():
        print(f"ERROR: base config not found at {args.base_config}", file=sys.stderr); sys.exit(1)

    study_name = args.study or _latest_study_name(args.db)
    storage    = f"sqlite:///{args.db.resolve()}"
    study      = optuna.load_study(study_name=study_name, storage=storage)

    component_keys = list(METRIC_WEIGHTS.keys())
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("ERROR: no completed trials.", file=sys.stderr); sys.exit(1)

    print(f"Study : {study_name}")
    print(f"Trials: {len(trials)} complete")
    print()

    df = _build_dataframe(trials, component_keys)

    non_param_cols = {"trial", "score"} | set(component_keys)
    param_names, continuous, categorical, log_params = _classify_params(df, non_param_cols)

    all_param_cols = continuous + categorical
    mid_study      = [p for p in all_param_cols if df[p].notna().mean() < MID_STUDY_FRAC]
    main_cols      = [p for p in all_param_cols if p not in set(mid_study)]

    if mid_study:
        print(f"Mid-study params (secondary GP): "
              f"{[p.rsplit('.', 1)[-1] for p in mid_study]}")
    print()

    # Main GP
    print("Fitting main GP…")
    gp_df        = df[main_cols + ["score"]].dropna()
    result_main  = _fit_gp(main_cols, gp_df, categorical, log_params,
                            args.n_restarts, tag="Main ")

    # Secondary GP
    result_secondary = None
    if mid_study:
        gp2_df = df[mid_study + ["score"]].dropna()
        if len(gp2_df) >= 5:
            print("Fitting secondary GP…")
            result_secondary = _fit_gp(mid_study, gp2_df, categorical, log_params,
                                        args.n_restarts, tag="Secondary ")
        else:
            print(f"Secondary GP skipped — only {len(gp2_df)} trials with mid-study params.")

    print()
    print(f"Main GP predicted optimum  : {result_main['predicted_score']:.4f} "
          f"± {result_main['predicted_std']:.4f}")
    print(f"Main GP observed best      : {result_main['observed_best']:.4f}")
    if result_secondary:
        print(f"Secondary GP predicted opt : {result_secondary['predicted_score']:.4f} "
              f"± {result_secondary['predicted_std']:.4f}")

    # Merge: main GP params, then secondary GP params on top
    optimal_params = {**result_main["best_params"]}
    if result_secondary:
        optimal_params.update(result_secondary["best_params"])

    print()
    print("GP-optimal parameters:")
    for k, v in optimal_params.items():
        print(f"  {k:<55s} = {v:.6g}")

    # Build config
    base_cfg = OmegaConf.load(args.base_config)
    dotlist  = [f"{k}={v}" for k, v in optimal_params.items()]
    cfg      = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(dotlist))

    comment  = _metadata_comment(study_name, result_main, result_secondary)
    yaml_text = comment + OmegaConf.to_yaml(cfg, resolve=False)

    out_path = args.out or _next_version_path(ROOT / "configs")

    print()
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