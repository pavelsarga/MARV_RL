"""
Plot figure: empirical acceleration distribution + deadzone normalisation curve.

Prefers raw_accel.npz files (per-robot values) when available, falls back to
step-mean/step-max from shock.csv files.

Output: images/shock_distribution.pdf  (and .png for quick preview)

Usage:
    python scripts/plot_shock_distribution.py
    python scripts/plot_shock_distribution.py --out /path/to/images/
    python scripts/plot_shock_distribution.py --runs runs/my_run
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

# ── Deadzone parameters (must match crossing_env.py) ─────────────────────────
THRESHOLD = 11.0   # m/s²
SCALE     = 35.0   # m/s²

def shock_norm(a):
    return np.clip((a - THRESHOLD) / SCALE, 0.0, 1.0)


def load_raw_data(search_dirs: list[Path]) -> np.ndarray | None:
    """Return concatenated raw per-robot accel values from raw_accel.npz files, or None."""
    arrays = []
    for d in search_dirs:
        for f in sorted(d.rglob("raw_accel.npz")):
            try:
                arrays.append(np.load(f)["accel"])
            except Exception:
                pass
    if not arrays:
        return None
    data = np.concatenate(arrays)
    print(f"Loaded {len(data):,} raw per-robot samples from {len(arrays)} raw_accel.npz file(s).")
    return data


def load_csv_data(search_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Return (step_means, step_maxes) from shock.csv files (fallback)."""
    means, maxes = [], []
    n_files = 0
    for d in search_dirs:
        for csv in sorted(d.rglob("shock.csv")):
            try:
                df = pd.read_csv(csv)
                if "shock/accel_magnitude" in df.columns:
                    means.append(df["shock/accel_magnitude"].dropna().values)
                if "shock/accel_magnitude_max" in df.columns:
                    maxes.append(df["shock/accel_magnitude_max"].dropna().values)
                n_files += 1
            except Exception:
                pass
    means = np.concatenate(means) if means else np.array([])
    maxes = np.concatenate(maxes) if maxes else np.array([])
    print(f"Loaded {len(means):,} step-mean and {len(maxes):,} step-max samples from {n_files} shock.csv file(s).")
    return means, maxes


def make_figure_raw(raw: np.ndarray, out_dir: Path) -> None:
    raw = raw[(raw >= 0) & (raw < 150)]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0, 80, 160)

    ax1.hist(raw, bins=bins, density=True, alpha=0.55,
             color="#1f77b4", label="per-robot acceleration (raw)", zorder=2)
    kde = gaussian_kde(raw[raw < 80], bw_method=0.06)
    xs = np.linspace(0, 80, 500)
    ax1.plot(xs, kde(xs), color="#1f77b4", linewidth=1.8, zorder=3)

    ax1.set_xlabel(r"Linear acceleration magnitude $\|\Delta\mathbf{v}/\Delta t\|_2$ (m/s²)", fontsize=11)
    ax1.set_ylabel("Probability density", fontsize=11)
    ax1.set_xlim(0, 80)
    ax1.set_ylim(bottom=0)
    ax1.axvline(THRESHOLD, color="#2ca02c", linestyle="--", linewidth=1.6,
                label=rf"$\tau_\mathrm{{shock}} = {THRESHOLD:.0f}$ m/s² (dead-zone)", zorder=5)

    ax2 = ax1.twinx()
    a_grid = np.linspace(0, 80, 500)
    ax2.plot(a_grid, shock_norm(a_grid), color="#d62728", linewidth=2.2,
             linestyle="-.", label=r"$\sigma_\mathrm{shock}(a)$ (right axis)", zorder=4)
    ax2.set_ylabel(r"Normalised shock $\sigma_\mathrm{shock}$  (−)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.label.set_color("#d62728")
    ax2.tick_params(axis="y", colors="#d62728")

    sat_x = THRESHOLD + SCALE
    ax2.annotate(
        rf"$\tau + \xi = {sat_x:.0f}$ m/s²" + "\n(saturates at 1.0)",
        xy=(sat_x - 1, 0.99), xytext=(sat_x - 20, 0.78),
        fontsize=8, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="right", framealpha=0.9)
    ax1.grid(True, alpha=0.25, zorder=0)
    fig.tight_layout()
    _save(fig, out_dir)


def make_figure_csv(means: np.ndarray, maxes: np.ndarray, out_dir: Path) -> None:
    means = means[means < 150]
    maxes = maxes[maxes < 150]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0, 80, 120)
    xs = np.linspace(0, 80, 500)

    if len(means):
        ax1.hist(means, bins=bins, density=True, alpha=0.55,
                 color="#1f77b4", label=r"step mean $\bar{a}$ (1024 robots)", zorder=2)
        kde_m = gaussian_kde(means[means < 80], bw_method=0.15)
        ax1.plot(xs, kde_m(xs), color="#1f77b4", linewidth=1.8, zorder=3)

    if len(maxes):
        ax1.hist(maxes, bins=bins, density=True, alpha=0.40,
                 color="#ff7f0e", label=r"step max $a_\mathrm{max}$ (impact peaks)", zorder=2)
        kde_M = gaussian_kde(maxes[maxes < 80], bw_method=0.15)
        ax1.plot(xs, kde_M(xs), color="#ff7f0e", linewidth=1.8, zorder=3)

    ax1.set_xlabel(r"Linear acceleration magnitude $\|\Delta\mathbf{v}/\Delta t\|_2$ (m/s²)", fontsize=11)
    ax1.set_ylabel("Probability density", fontsize=11)
    ax1.set_xlim(0, 80)
    ax1.set_ylim(bottom=0)
    ax1.axvline(THRESHOLD, color="#2ca02c", linestyle="--", linewidth=1.6,
                label=rf"$\tau_\mathrm{{shock}} = {THRESHOLD:.0f}$ m/s² (dead-zone)", zorder=5)

    ax2 = ax1.twinx()
    a_grid = np.linspace(0, 80, 500)
    ax2.plot(a_grid, shock_norm(a_grid), color="#d62728", linewidth=2.2,
             linestyle="-.", label=r"$\sigma_\mathrm{shock}(a)$ (right axis)", zorder=4)
    ax2.set_ylabel(r"Normalised shock $\sigma_\mathrm{shock}$  (−)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.label.set_color("#d62728")
    ax2.tick_params(axis="y", colors="#d62728")

    sat_x = THRESHOLD + SCALE
    ax2.annotate(
        rf"$\tau + \xi = {sat_x:.0f}$ m/s²" + "\n(saturates at 1.0)",
        xy=(sat_x - 1, 0.99), xytext=(sat_x - 20, 0.85),
        fontsize=8, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="right", framealpha=0.9)
    ax1.grid(True, alpha=0.25, zorder=0)
    fig.tight_layout()
    _save(fig, out_dir)


def _save(fig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "shock_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "shock_distribution.png", dpi=150, bbox_inches="tight")
    print(f"Saved:\n  {out_dir / 'shock_distribution.pdf'}\n  {out_dir / 'shock_distribution.png'}")
    plt.close(fig)


def main():
    ws = Path("/home/robot/workspaces/robot_rodeo_gym_ws")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ws / "images"), help="Output directory for PDF/PNG")
    ap.add_argument("--runs", nargs="*", default=[str(ws / "runs"), str(ws / "logs")],
                    help="Directories to search for raw_accel.npz or shock.csv files")
    args = ap.parse_args()

    search_dirs = [Path(p) for p in args.runs if Path(p).exists()]

    raw = load_raw_data(search_dirs)
    if raw is not None:
        make_figure_raw(raw, Path(args.out))
    else:
        print("No raw_accel.npz found — falling back to shock.csv step-mean/max data.")
        means, maxes = load_csv_data(search_dirs)
        if len(means) == 0 and len(maxes) == 0:
            print("No data found — check --runs paths.")
            return
        make_figure_csv(means, maxes, Path(args.out))


if __name__ == "__main__":
    main()
