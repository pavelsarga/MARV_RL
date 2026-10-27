#!/usr/bin/env python3
"""Fit ElevationMapEmulator (FTR-Benchmark) to paired Gazebo windows from extract_pairs.py.

    fit_emulator.py [--data data/] [--out fit/]

The true-geometry windows are re-cut from the saved GT poses with the emulator's context
margin, so the filters see real terrain past the window edge exactly as they will in Isaac.
The score is the MSE over EDGE cells -- cells within 3 cells of a true height step of more
than 2 cm -- because on flat ground every candidate is right and that would drown the
difference. Cells the recorded map does not cover (NaN) and anything taller than 0.9 m (shelf
props off the course, cut by the mapping height crop) are left out.

Writes fit/fit.json (best parameters, the full grid) and fit/profiles.png (centre-column height
profiles across obstacles: true geometry vs Gazebo map vs emulator).
"""
import argparse
import glob
import importlib.util
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "velocity_il"))
sys.path.insert(0, HERE)
import bagio  # noqa: E402
from extract_pairs import gt_gridmap  # noqa: E402

# imported by path: the ftr_envs package __init__ pulls in Isaac
_spec = importlib.util.spec_from_file_location("elevation_map_emulator", os.path.join(
    HERE, "..", "..", "src", "FTR-Benchmark", "ftr_envs", "tasks", "crossing", "elevation_map_emulator.py"))
emu_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emu_mod)

KS = (1, 3, 5, 7, 9, 11)
SIGMAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
MAX_MARGIN = max(KS) // 2 * 2 + int(np.ceil(3 * max(SIGMAS)))


def load(data_dir):
    maps, gts, vs = [], [], []
    worlds = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(f)
        world = str(d["world"])
        if world not in worlds:
            worlds[world] = gt_gridmap(world)
        G, gc, gres = worlds[world]
        gts.append(np.stack([bagio.robot_window(G, gc, gres, p, margin=MAX_MARGIN) for p in d["pose"]]))
        maps.append(d["map"])
        vs.append(d["v"])
        print(f"{os.path.basename(f)}: {len(d['map'])} pairs", flush=True)
    return np.concatenate(maps), np.concatenate(gts), np.concatenate(vs)


def edge_mask(gt_core, radius=3, step=0.02):
    t = torch.from_numpy(gt_core).unsqueeze(1)
    gy = (t[:, :, 1:, :] - t[:, :, :-1, :]).abs() > step
    gx = (t[:, :, :, 1:] - t[:, :, :, :-1]).abs() > step
    e = torch.zeros_like(t, dtype=torch.bool)
    e[:, :, 1:, :] |= gy
    e[:, :, :-1, :] |= gy
    e[:, :, :, 1:] |= gx
    e[:, :, :, :-1] |= gx
    k = 2 * radius + 1
    return (torch.nn.functional.max_pool2d(e.float(), k, stride=1, padding=radius) > 0).squeeze(1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "fit"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    win_map, gt_big, v = load(a.data)
    M = MAX_MARGIN
    gt_core = gt_big[:, M:-M, M:-M]
    valid = ~np.isnan(win_map) & (np.abs(gt_core) < 0.9) & ~np.isnan(gt_core)
    edges = edge_mask(np.nan_to_num(gt_core)) & valid
    flat = valid & ~edges
    print(f"{len(win_map)} windows, {edges.sum()} edge cells, {flat.sum()} flat cells", flush=True)

    target = torch.from_numpy(np.nan_to_num(win_map))
    src = torch.from_numpy(np.nan_to_num(gt_big))
    E, Fl = torch.from_numpy(edges), torch.from_numpy(flat)
    grid = []
    for kd, ke, sg in itertools.product(KS, KS, SIGMAS):
        emu = emu_mod.ElevationMapEmulator(kd, ke, sg)
        m = emu.margin
        out = emu(src[:, M - m: src.shape[1] - (M - m), M - m: src.shape[2] - (M - m)])
        # the best constant offset is closed-form: the mean residual over the scored (edge) cells
        offset = float((target - out)[E].mean())
        out = out + offset
        err = (out - target) ** 2
        grid.append(dict(k_dilate=kd, k_erode=ke, blur_sigma=sg, height_offset=round(offset, 4),
                         rmse_edge=float(err[E].mean().sqrt()), rmse_flat=float(err[Fl].mean().sqrt()),
                         bias_edge=float((out - target)[E].mean())))
    grid.sort(key=lambda r: r["rmse_edge"])
    ident = next(r for r in grid if r["k_dilate"] == 1 and r["k_erode"] == 1 and r["blur_sigma"] == 0)
    best = grid[0]
    print("identity:", ident)
    for r in grid[:8]:
        print(r)
    json.dump(dict(best=best, identity=ident, grid=grid, n_windows=int(len(win_map)),
                   edge_cells=int(edges.sum())), open(os.path.join(a.out, "fit.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    emu = emu_mod.ElevationMapEmulator(best["k_dilate"], best["k_erode"], best["blur_sigma"], best["height_offset"])
    m = emu.margin
    out = emu(src[:, M - m: src.shape[1] - (M - m), M - m: src.shape[2] - (M - m)]).numpy()
    # windows with the most edge cells in the centre column: obstacles straight ahead
    score = edges[:, :, 10].sum(1)
    picks = np.argsort(-score)[: 600: 75]
    fig, axs = plt.subplots(2, 4, figsize=(18, 7))
    rows_m = (bagio.HM_ROWS // 2 - np.arange(bagio.HM_ROWS)) * bagio.HM_RES
    for ax, i in zip(axs.ravel(), picks):
        ax.plot(rows_m, gt_core[i, :, 10], "k-", label="true geometry (Isaac)")
        ax.plot(rows_m, win_map[i, :, 10], "C0-", lw=2, label="Gazebo elevation map")
        ax.plot(rows_m, out[i, :, 10], "C3--", lw=2, label="emulator")
        ax.set_title(f"window {i}, v = {v[i]:.2f} m/s")
        ax.set_xlabel("ahead of robot [m]")
        ax.invert_xaxis()
    axs[0, 0].set_ylabel("height rel. base [m]")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"centre column; best k_dilate={best['k_dilate']} k_erode={best['k_erode']} "
                 f"sigma={best['blur_sigma']}: edge RMSE {best['rmse_edge']:.3f} m "
                 f"(identity {ident['rmse_edge']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "profiles.png"), dpi=70)


if __name__ == "__main__":
    main()
