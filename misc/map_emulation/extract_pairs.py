#!/usr/bin/env python3
"""Paired heightmap windows for fitting the Isaac elevation-map emulator.

For every /elevation_map_gt_filtered message in a Gazebo sweep bag, the robot-centric 45 x 21
FTR window is cut twice at the GT pose of that map's stamp:
    map  -- from the recorded elevation map (layer elevation_inpainted), what a policy sees
    gt   -- from the world's true geometry (worldgeom), what Isaac's numpy map would give
Both relative to (base z - wheel radius), exactly like the FTR observation.

    extract_pairs.py --world straight_arena --out data/ <bag_dir> [<bag_dir> ...] [-j 8]

One .npz per bag: map, gt [N, 45, 21] float32, pose [N, 4], v [N] (GT forward speed), t [N].
The true geometry is rasterised at 0.05 m, Isaac's terrain-map resolution, so the GT window
carries the same quantisation the emulator will be fed in training.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "velocity_il"))
import bagio  # noqa: E402
import worldgeom  # noqa: E402

ARENAS = "/home/robot/workspaces/flipper_eval_marv/src/robot_rodeo_gym_ros2/arenas"
WORLD_SDF = {a: f"{ARENAS}/{a}/world/robot_rodeo_gym_world.sdf"
             for a in ("straight_arena", "long_arena", "short_arena")}
GT_RES = 0.05


def gt_gridmap(world, margin=6.0):
    boxes, skipped = worldgeom.load_world(WORLD_SDF[world])
    if skipped:
        print(f"[{world}] {len(skipped)} collision(s) not rasterised: {[s[1] for s in skipped[:5]]}", flush=True)
    cx = np.array([c[0] for c, *_ in boxes])
    cy = np.array([c[1] for c, *_ in boxes])
    H, xs, ys = worldgeom.height_raster(boxes, (cx.min() - margin, cx.max() + margin),
                                        (cy.min() - margin, cy.max() + margin), GT_RES)
    return worldgeom.raster_as_gridmap(H, xs, ys)


def one_bag(job):
    bag, world, out_dir = job
    out = os.path.join(out_dir, os.path.basename(bag.rstrip("/")) + ".npz")
    if os.path.exists(out):
        return f"{out} exists, skipped"
    G, gc, gres = gt_gridmap(world)
    d = bagio.read_topics(bag, ["/elevation_map_gt_filtered", "/ground_truth_odom"])
    tm, maps = d["/elevation_map_gt_filtered"]
    to, odom = d["/ground_truth_odom"]
    if not maps or not odom:
        return f"{bag}: no maps or no GT odometry"
    pose_all, v_all = bagio.odom_arrays(odom)
    keep = (tm >= to[0]) & (tm <= to[-1])
    tm = tm[keep]
    maps = [m for m, k in zip(maps, keep) if k]
    pose = bagio.interp_pose(tm, to, pose_all)
    v = np.interp(tm, to, v_all)
    win_map = np.stack([bagio.robot_window(m.H, m.center, m.res, p) for m, p in zip(maps, pose)])
    win_gt = np.stack([bagio.robot_window(G, gc, gres, p) for p in pose])
    np.savez_compressed(out, map=win_map.astype(np.float32), gt=win_gt.astype(np.float32),
                        pose=pose, v=v, t=tm, world=world)
    return f"{out}: {len(tm)} pairs, map NaN {np.isnan(win_map).mean():.2%}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--world", default="straight_arena", choices=sorted(WORLD_SDF))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("-j", "--jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(b, a.world, a.out) for b in a.bags]
    with Pool(min(a.jobs, len(jobs))) as pool:
        for line in pool.imap_unordered(one_bag, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
