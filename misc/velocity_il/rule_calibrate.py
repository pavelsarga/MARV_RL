"""Heightmap metrics of the rule speed controller over recorded rides -- a tuning aid, not part of it.

Runs the map criteria of a velocity_rule config (step_up / step_down / roughness / map_mismatch)
over the extracted 20 Hz shards (extract_velocity_il.py; driving ticks only) and prints, per area
(-2 = gap strip), each metric's median and 95th percentile and the share of ticks it would act at
the config's thresholds (linear: above lo, bang: above trip). Flat stretches should stay below
lo / release; each obstacle should exceed its own criterion. The reactive thresholds are physical
units and need no data.

The shards hold no roll / pitch, so map_mismatch is computed against a level body plane here:
it reads high on ramps -- judge it on flat areas only.

    python misc/velocity_il/rule_calibrate.py [--config configs/velocity_rule/default.yaml] [shard.npz ...]
"""
import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("velocity_rules", ROOT / "src/FTR-Benchmark/ftr_envs/tasks/crossing/velocity_rules.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shards", nargs="*", type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/velocity_rule/default.yaml")
    a = ap.parse_args()
    shards = a.shards or sorted((ROOT / "misc/velocity_il/data").glob("straight_r*/*.npz"))
    cfg = R.load_rule_config(a.config)
    cfg["criteria"] = {k: c for k, c in cfg["criteria"].items() if R.KINDS[k][2]}
    criteria = R.RuleSpeed(cfg, num_envs=1).criteria

    per_area = defaultdict(lambda: defaultdict(list))         # area -> criterion -> metric arrays
    for path in shards:
        d = np.load(path)
        keep = d["driving"]
        hm = torch.from_numpy(d["hm"][keep].astype(np.float32))
        n = len(hm)
        x = R.Inputs(hm=hm, v_meas=torch.zeros(n), flip_vel=torch.zeros(n, 4), rpy=torch.zeros(n, 3),
                     ang_vel=torch.zeros(n, 3), w_cmd=torch.zeros(n))
        area = d["obstacle"][keep]
        for c in criteria:
            m = c.metric(x).numpy()
            for k in np.unique(area):
                per_area[int(k)][c.name].append(m[area == k])
    print(f"{len(shards)} rides, config {a.config}")
    print("area   " + "".join(f"{c.name + ' p50/p95 act':>30}" for c in criteria))
    for k in sorted(per_area):
        cells = []
        for c in criteria:
            m = np.concatenate(per_area[k][c.name])
            act = np.mean(m > (c.lo if c.mode == "linear" else c.trip))
            cells.append(f"{np.median(m):.3f}/{np.percentile(m, 95):.3f} {act:4.0%}")
        print(f"{k:>4}   " + "".join(f"{s:>30}" for s in cells))


if __name__ == "__main__":
    main()
