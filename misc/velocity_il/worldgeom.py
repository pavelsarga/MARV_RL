"""True terrain height of a Gazebo world, rasterised from its SDF collision geometry -- no ROS, no sim.

Every collision <box> of the world's own models, plus those of <include>d Fuel models resolved
from the local Fuel cache (~/.gz/fuel), becomes an oriented box; a <mesh> collision becomes the
oriented box of its vertex AABB (the Euro pallet is the only mesh on the arenas, and its top
deck is what a lidar sees). The height at (x, y) is the highest point where a vertical line
leaves any box, with the ground plane at z = 0 -- the same construction as
marv_flipper_eval/gt_elevation_publisher.py's height_grid, without its rclpy dependency.

Unresolvable geometry (a Fuel model missing from the cache, an unsupported shape) is listed in
`skipped` with its world footprint centre, so a fit can mask those cells out instead of
treating them as flat ground.
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

FUEL_CACHE = os.path.expanduser("~/.gz/fuel")


def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _pose(el):
    p = el.find("pose") if el is not None else None
    v = [float(t) for t in p.text.split()] if p is not None and p.text and p.text.strip() else [0.0] * 6
    return np.array(v[:3]), _rpy_R(*v[3:6])


def _compose(t1, R1, t2, R2):
    return t1 + R1 @ t2, R1 @ R2


def _fuel_dir(uri):
    """Local cache dir (latest version) of a Fuel model URI, or None."""
    m = re.match(r"https?://([^/]+)/[^/]+/([^/]+)/models/(.+)$", uri.strip())
    if not m:
        return None
    host, owner, name = m.groups()
    base = os.path.join(FUEL_CACHE, host, owner.lower(), "models", name.lower())
    if not os.path.isdir(base):
        return None
    versions = sorted((v for v in os.listdir(base) if v.isdigit()), key=int)
    return os.path.join(base, versions[-1]) if versions else None


def _mesh_aabb(path, scale):
    """(centre, half extents) of a COLLADA/OBJ mesh's vertices, scaled.

    COLLADA vertices are placed through the visual-scene node transforms (and a Y_UP asset is
    turned Z-up), as Gazebo does: the Euro pallet's geometry is authored off-centre and only
    its node matrix brings it back -- skipping that put the pallet 0.6 m to the side."""
    if path.lower().endswith(".dae"):
        root = ET.parse(path).getroot()
        tag = lambda el: el.tag.rsplit("}", 1)[-1]
        geoms = {}
        for g in root.iter():
            if tag(g) != "geometry":
                continue
            arrs = [np.array(fa.text.split(), dtype=float).reshape(-1, 3) for fa in g.iter()
                    if tag(fa) == "float_array" and "position" in (fa.get("id") or "").lower()]
            if arrs:
                geoms[g.get("id")] = np.concatenate(arrs)
        if not geoms:
            raise ValueError(f"no position arrays in {path}")
        pts = []

        def walk(node, T):
            for child in node:
                k = tag(child)
                if k == "matrix":
                    T = T @ np.array(child.text.split(), dtype=float).reshape(4, 4)
                elif k == "translate":
                    M = np.eye(4)
                    M[:3, 3] = np.array(child.text.split(), dtype=float)
                    T = T @ M
                elif k == "scale":
                    T = T @ np.diag(list(np.array(child.text.split(), dtype=float)) + [1.0])
            for child in node:
                k = tag(child)
                if k == "instance_geometry" and child.get("url", "").lstrip("#") in geoms:
                    v = geoms[child.get("url").lstrip("#")]
                    pts.append(v @ T[:3, :3].T + T[:3, 3])
                elif k == "node":
                    walk(child, T)

        for scene in (s for s in root.iter() if tag(s) == "visual_scene"):
            walk(scene, np.eye(4))
        v = np.concatenate(pts) if pts else np.concatenate(list(geoms.values()))
        up = next((el.text for el in root.iter() if tag(el) == "up_axis"), "Z_UP") or "Z_UP"
        if up.strip().upper() == "Y_UP":
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    elif path.lower().endswith(".obj"):
        v = np.array([[float(t) for t in l.split()[1:4]] for l in open(path) if l.startswith("v ")])
    else:
        raise ValueError(f"unsupported mesh {path}")
    v = v * scale
    lo, hi = v.min(0), v.max(0)
    return (lo + hi) / 2, (hi - lo) / 2


def _collect(model_el, t_model, R_model, model_dir, boxes, skipped, label):
    for link in model_el.findall("link"):
        tl, Rl = _compose(t_model, R_model, *_pose(link))
        for col in link.findall("collision"):
            tc, Rc = _compose(tl, Rl, *_pose(col))
            geom = col.find("geometry")
            box = geom.find("box/size") if geom is not None else None
            mesh = geom.find("mesh") if geom is not None else None
            if box is not None:
                boxes.append((tc, Rc, np.array(box.text.split(), dtype=float) / 2, label))
            elif mesh is not None and model_dir is not None:
                uri = mesh.find("uri").text.strip()
                rel = uri.split("/files/", 1)[1] if "/files/" in uri else uri.replace("model://", "").split("/", 1)[-1]
                scale = np.array(mesh.find("scale").text.split(), dtype=float) if mesh.find("scale") is not None else 1.0
                try:
                    c, e = _mesh_aabb(os.path.join(model_dir, rel), scale)
                    boxes.append((tc + Rc @ c, Rc, e, label))
                except (OSError, ValueError) as err:
                    skipped.append((tc, f"{label}: {err}"))
            elif geom is not None and len(geom):
                skipped.append((tc, f"{label}: {geom[0].tag} collision not rasterised"))
    for sub in model_el.findall("model"):
        _collect(sub, *_compose(t_model, R_model, *_pose(sub)), model_dir, boxes, skipped, label)


def load_world(sdf_path, exclude=("marv",)):
    """(boxes, skipped): boxes = [(centre, R, half_extents, model name)], skipped = [(xyz, why)]."""
    # comments are dropped before parsing: the arenas' comments contain "--", which Gazebo's
    # parser accepts and a strict XML parser does not
    text = re.sub(r"<!--.*?-->", "", open(sdf_path).read(), flags=re.S)
    world = ET.fromstring(text).find("world")
    boxes, skipped = [], []
    for model in world.findall("model"):
        if model.get("name") in exclude:
            continue
        _collect(model, *_pose(model), None, boxes, skipped, model.get("name"))
    for inc in world.findall("include"):
        uri = inc.find("uri").text
        name = inc.find("name").text if inc.find("name") is not None else uri
        t, R = _pose(inc)
        d = _fuel_dir(uri)
        if d is None:
            skipped.append((t, f"{name}: {uri} not in the Fuel cache"))
            continue
        sub = ET.parse(os.path.join(d, "model.sdf")).getroot().find("model")
        _collect(sub, *_compose(t, R, *_pose(sub)), d, boxes, skipped, name)
    return boxes, skipped


def height_raster(boxes, x_range, y_range, res=0.025):
    """(H, xs, ys): H[i, j] = surface height at (xs[i], ys[j]); ground plane z = 0.

    Slab test of a vertical line against every oriented box: the line leaves the box at the
    smallest exit parameter over the three slabs, provided it entered before that."""
    xs = np.arange(x_range[0], x_range[1] + res / 2, res)
    ys = np.arange(y_range[0], y_range[1] + res / 2, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    H = np.zeros(X.size)
    for c, R, e, _ in boxes:
        reach = np.linalg.norm(e)
        near = (np.abs(P[:, 0] - c[0]) <= reach) & (np.abs(P[:, 1] - c[1]) <= reach)
        if not near.any():
            continue
        o = (P[near] - c) @ R                  # ray origins in the box frame
        d = R.T @ np.array([0.0, 0.0, 1.0])    # ray direction in the box frame
        t_in = np.full(o.shape[0], -np.inf)
        t_out = np.full(o.shape[0], np.inf)
        for k in range(3):
            if abs(d[k]) < 1e-12:
                outside = np.abs(o[:, k]) > e[k]
                t_in[outside], t_out[outside] = np.inf, -np.inf
                continue
            t1, t2 = (-e[k] - o[:, k]) / d[k], (e[k] - o[:, k]) / d[k]
            t_in = np.maximum(t_in, np.minimum(t1, t2))
            t_out = np.minimum(t_out, np.maximum(t1, t2))
        hit = t_out >= t_in
        H[np.flatnonzero(near)[hit]] = np.maximum(H[np.flatnonzero(near)[hit]], t_out[hit])
    return H.reshape(X.shape), xs, ys


def raster_as_gridmap(H, xs, ys):
    """Re-express a height_raster in bagio's grid-map convention (row 0 = +x, col 0 = +y), so the
    same robot_window() samples both the true terrain and a recorded elevation map."""
    res = float(xs[1] - xs[0])
    return H[::-1, ::-1].astype(np.float32), ((xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2), res
