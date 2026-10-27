"""Gazebo rosbag (mcap) reading for the velocity-IL dataset and the map-emulator fit -- no ROS.

Messages are decoded with mcap + mcap_ros2 from the schemas embedded in the bag, so this runs
in the isaaclab env (or any python with `pip install mcap mcap-ros2-support`).

Time base is SIM time everywhere: header stamps where a message has one; for header-less
topics (std_msgs/Float64 commands, ...) the bag's wall-clock log_time is mapped to sim time
through the /clock messages recorded in the same bag. The sim runs well below real time
(~0.3x), so mixing the two clocks would put every command seconds away from the pose it
belongs to.

The heightmap window (robot_window, re-exported from marv_rl_training.training.velocity_il_features)
follows FTR's observation exactly: 45 x 21 cells at 0.05 m, row 0 = front, col 0 = robot's right,
rotated with the robot's yaw, heights relative to (base z - track wheel radius 0.1165).
"""
import glob
import os
from collections import namedtuple

import sys

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# the window crop is shared with the live velocity node: one implementation in the package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "flipper_training"))
from marv_rl_training.training.velocity_il_features import (  # noqa: E402,F401  (re-exported)
    HM_COLS, HM_RES, HM_ROWS, TRACK_WHEEL_RADIUS, gridmap_layer, robot_window, sample_world, window_offsets)

# One elevation-map layer as compact float32 + its georeference (H in grid_map index order).
GridLayer = namedtuple("GridLayer", "H center res t")
# nav_msgs/Odometry reduced to the pose and the body-frame forward speed.
Odom = namedtuple("Odom", "t x y z yaw v")


def mcap_path(bag_dir):
    files = sorted(glob.glob(os.path.join(bag_dir, "*.mcap")))
    if not files:
        raise FileNotFoundError(f"no .mcap in {bag_dir} (offloaded to the cluster?)")
    return files[0]


def stamp_s(header):
    return header.stamp.sec + 1e-9 * header.stamp.nanosec


def read_topics(bag_dir, topics, convert=None, layer="elevation_inpainted"):
    """{topic: (t_sim [N], [item, ...])}, sorted by time. Topics absent from the bag map to
    empty lists. /clock is always read for the log-time -> sim-time mapping.

    Each message is reduced AS IT IS DECODED: `convert` maps topic -> fn(msg) -> item, and
    every grid_map_msgs/GridMap without one becomes GridLayer(H, center, res, t) of `layer`
    only. Keeping decoded GridMaps instead is not an option: one sweep bag holds ~2000 of them
    x 12 layers x 25600 cells as Python floats, tens of GB -- which is how this once took the
    whole machine down."""
    convert = dict(convert or {})
    raw = {t: [] for t in topics}
    clock_log, clock_sim = [], []
    with open(mcap_path(bag_dir), "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages(topics=list(set(topics) | {"/clock"})):
            if channel.topic == "/clock":
                clock_log.append(message.log_time * 1e-9)
                clock_sim.append(msg.clock.sec + 1e-9 * msg.clock.nanosec)
                if "/clock" not in raw:
                    continue
            fn = convert.get(channel.topic)
            if fn is None and schema.name == "grid_map_msgs/msg/GridMap":
                fn = convert[channel.topic] = lambda m: GridLayer(*gridmap_layer(m, layer), stamp_s(m.header))
            elif fn is None and schema.name == "nav_msgs/msg/Odometry":
                fn = convert[channel.topic] = odom_sample
            raw[channel.topic].append((message.log_time * 1e-9, fn(msg) if fn else msg))
    clock_log, clock_sim = np.asarray(clock_log), np.asarray(clock_sim)
    if clock_log.size < 2:
        raise RuntimeError(f"{bag_dir}: no /clock recorded -- cannot map header-less topics to sim time")

    def to_sim(log_t):
        return np.interp(log_t, clock_log, clock_sim)

    out = {}
    for topic, items in raw.items():
        if not items:
            out[topic] = (np.zeros(0), [])
            continue
        msgs = [m for _, m in items]
        if isinstance(msgs[0], (GridLayer, Odom)):
            t = np.array([m.t for m in msgs])
        elif hasattr(msgs[0], "header") and msgs[0].header.stamp.sec + msgs[0].header.stamp.nanosec > 0:
            t = np.array([stamp_s(m.header) for m in msgs])
        else:
            t = to_sim(np.array([lt for lt, _ in items]))
        order = np.argsort(t, kind="stable")
        out[topic] = (t[order], [msgs[i] for i in order])
    return out


# ---------------------------------------------------------------------------- odometry --
def yaw_of(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def odom_sample(m):
    """nav_msgs/Odometry -> Odom. Gazebo's ground_truth_odom twist is in the child (body) frame,
    so twist.linear.x is already the forward speed -- what FTR observes as
    robot_lin_velocities[:, 0]."""
    p = m.pose.pose.position
    return Odom(stamp_s(m.header), p.x, p.y, p.z, yaw_of(m.pose.pose.orientation), m.twist.twist.linear.x)


def odom_arrays(samples):
    """[N, 4] pose (x, y, z, yaw) and [N] body-frame forward speed from Odom samples."""
    a = np.array([s[1:] for s in samples]).reshape(-1, 5)
    return a[:, :4], a[:, 4]


def interp_pose(t_query, t, pose):
    """Pose at arbitrary times; yaw interpolated on the unwrapped angle."""
    yaw = np.unwrap(pose[:, 3])
    cols = [np.interp(t_query, t, pose[:, k]) for k in range(3)] + [np.interp(t_query, t, yaw)]
    out = np.stack(cols, axis=-1)
    out[..., 3] = (out[..., 3] + np.pi) % (2 * np.pi) - np.pi
    return out


def zoh_index(t_query, t_src):
    """Index of the latest source sample at or before each query time (-1 before the first)."""
    return np.searchsorted(t_src, t_query, side="right") - 1
