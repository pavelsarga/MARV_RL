#!/usr/bin/env python3
"""Velocity-IL dataset: Gazebo shared-control bags -> 20 Hz .npz shards, no ROS.

    extract_velocity_il.py --out data/ <bag_dir> [...] [-j 8]

A collection bag (eval_run.py speed_source joy, or policy_sweep --speed-source joy) has auto_ride
steering, a flipper controller on the flippers and the operator's throttle on
/auto_ride/speed_cmd -- the label. Every series is resampled onto one 20 Hz sim-time grid (the
control rate of the velocity controller in Isaac and in Gazebo):

  hm           [T, 45, 21] float16  robot-centric FTR window of the latest map (zero-order hold),
                                    cropped at the pose of THAT tick, relative to (z - 0.1165);
                                    unmapped cells -> 0 (flat at the robot's level)
  hm_nan       [T]       fraction of the window that was unmapped
  v            [T]       measured forward speed (GT odometry, body frame)
  v_label      [T]       operator throttle /auto_ride/speed_cmd (NaN when the bag has none)
  v_cmd, w_cmd [T]       what auto_ride actually commanded on /cmd_vel
  flip_angle   [T, 4]    measured flipper angles, FTR convention
  flip_target  [T, 4]    commanded flipper target, FTR convention (see below)
  flip_cmd_vel [T, 4]    commanded flipper velocity (velocity-mode channels only, else 0), FTR conv.
  override, sas, driving [T] bool   /flipper_override, /override_sas, auto_ride state "driving"
  obstacle     [T]       /obstacle_id (zero-order hold)
  pose         [T, 4]    x, y, z, yaw

FTR flipper convention = ROS joint angle x [-1, -1, 1, 1] (FtrWheelArticulation): positive is UP
on all four flippers, radians, order fl, fr, rl, rr.

Flipper target. Controllers command flippers in different modes, so every mode is turned into
one absolute target per flipper, updated event by event:
    /flippers_cmd_pos/*      target = value
    /flippers_cmd_pos_rel/*  target = measured + value
    /flippers_cmd_vel/*      velocity u held (until the next message, or 0.25 s), target += u dt
                             each tick, kept within +/-0.35 rad of the measured angle so a
                             saturated joint does not wind the target up
and finally clamped to the Isaac configs' logical flipper limits (velocity_il_features.TARGET_LOW / TARGET_HIGH).
While /flipper_override is on, only the manual channel (/flippers_cmd_vel_manual/*: the
operator or the SAS rescue) counts -- the relay gates the policy out exactly then.
"""
import argparse
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bagio  # noqa: E402

from marv_rl_training.training.velocity_il_features import (  # noqa: E402  (bagio put it on the path)
    CORNERS, FTR_SIGN, JOINTS, RATE, FlipperTargetTracker)

CMD_KINDS = ("pos", "pos_rel", "vel", "vel_manual")
TOPICS = (["/elevation_map_gt_filtered", "/ground_truth_odom", "/joint_states", "/cmd_vel",
           "/auto_ride/speed_cmd", "/auto_ride/state", "/flipper_override", "/override_sas", "/obstacle_id",
           "/active_policy"]
          + [f"/flippers_cmd_{k}/{c}" for k in CMD_KINDS for c in CORNERS])


def zoh(t_query, t_src, values, default=np.nan):
    idx = bagio.zoh_index(t_query, t_src)
    values = np.asarray(values)
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)] if len(values) else default, default)
    return out


def flipper_targets(tg, angle, d):
    """[T, 4] targets, [T, 4] held command velocities (logical convention) and [T] override, by
    replaying every flipper command through the same FlipperTargetTracker the live node uses."""
    t_ov, ov_msgs = d["/flipper_override"]
    override = zoh(tg, t_ov, [m.data for m in ov_msgs], False).astype(bool) if ov_msgs else np.zeros(len(tg), bool)
    events = []
    for kind in FlipperTargetTracker.KINDS:
        for k, corner in enumerate(CORNERS):
            t, msgs = d[f"/flippers_cmd_{kind}/{corner}"]
            events += [(ti, kind, k, m.data) for ti, m in zip(t, msgs)]
    events.sort(key=lambda e: e[0])
    tracker = FlipperTargetTracker(angle[0])
    target = np.zeros((len(tg), 4))
    cmd_vel = np.zeros((len(tg), 4))
    e_i = 0
    for i, ti in enumerate(tg):
        while e_i < len(events) and events[e_i][0] <= ti:
            tracker.command(*events[e_i])
            e_i += 1
        target[i], cmd_vel[i] = tracker.step(ti, angle[i], bool(override[i]))
    return target, cmd_vel, override


def one_bag(job):
    bag, out_dir = job
    name = os.path.basename(bag.rstrip("/"))
    out = os.path.join(out_dir, name + ".npz")
    if os.path.exists(out):
        return f"{out} exists, skipped"
    d = bagio.read_topics(bag, TOPICS)
    tm, maps = d["/elevation_map_gt_filtered"]
    to, odom = d["/ground_truth_odom"]
    tj, joints = d["/joint_states"]
    if not maps or not odom or not joints:
        return f"{bag}: missing maps / odometry / joint states"
    t0, t1 = max(tm[0], to[0], tj[0]), min(to[-1], tj[-1])
    tg = np.arange(t0, t1, 1.0 / RATE)
    pose_all, v_all = bagio.odom_arrays(odom)
    pose = bagio.interp_pose(tg, to, pose_all)
    v = np.interp(tg, to, v_all)

    name_idx = {n: i for i, n in enumerate(joints[0].name)}
    ja = np.array([[m.position[name_idx[j]] for j in JOINTS] for m in joints]) * FTR_SIGN
    angle = np.stack([np.interp(tg, tj, ja[:, k]) for k in range(4)], axis=1)
    target, cmd_vel, override = flipper_targets(tg, angle, d)

    mi = bagio.zoh_index(tg, tm)
    hm = np.zeros((len(tg), bagio.HM_ROWS, bagio.HM_COLS), np.float32)
    for i, (k, p) in enumerate(zip(mi, pose)):
        m = maps[max(k, 0)]
        hm[i] = bagio.robot_window(m.H, m.center, m.res, p)
    hm_nan = np.isnan(hm).mean(axis=(1, 2))
    hm = np.nan_to_num(hm, nan=0.0)

    ts, sp = d["/auto_ride/speed_cmd"]
    v_label = zoh(tg, ts, [m.data for m in sp]) if sp else np.full(len(tg), np.nan)
    tc, cmds = d["/cmd_vel"]
    v_cmd = zoh(tg, tc, [m.linear.x for m in cmds], 0.0) if cmds else np.zeros(len(tg))
    w_cmd = zoh(tg, tc, [m.angular.z for m in cmds], 0.0) if cmds else np.zeros(len(tg))
    tr, rs = d["/auto_ride/state"]
    driving = zoh(tg, tr, [m.data.startswith("driving") for m in rs], False).astype(bool) if rs else np.zeros(len(tg), bool)
    tsas, sas = d["/override_sas"]
    sas = zoh(tg, tsas, [m.data for m in sas], False).astype(bool) if sas else np.zeros(len(tg), bool)
    tob, ob = d["/obstacle_id"]
    obstacle = zoh(tg, tob, [m.data for m in ob], -1).astype(np.int16) if ob else np.full(len(tg), -1, np.int16)
    _, pol = d["/active_policy"]
    # mode_teleop (manual rides) announces an empty /active_policy
    policy = pol[-1].data if pol and pol[-1].data else re.sub(r"_sweep_.*$", "", name)

    np.savez_compressed(out, t=tg, hm=hm.astype(np.float16), hm_nan=hm_nan.astype(np.float32), v=v, v_label=v_label,
                        v_cmd=v_cmd, w_cmd=w_cmd, flip_angle=angle.astype(np.float32),
                        flip_target=target.astype(np.float32), flip_cmd_vel=cmd_vel.astype(np.float32),
                        override=override, sas=sas, driving=driving, obstacle=obstacle, pose=pose,
                        policy=policy, bag=name)
    return (f"{out}: {len(tg)} ticks ({len(tg) / RATE:.0f} s sim), policy {policy}, "
            f"label {'yes' if sp else 'NO'}, driving {driving.mean():.0%}, map NaN {hm_nan.mean():.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("-j", "--jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(b, a.out) for b in a.bags]
    with Pool(min(a.jobs, len(jobs))) as pool:
        for line in pool.imap_unordered(one_bag, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
