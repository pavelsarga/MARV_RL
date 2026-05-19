#!/usr/bin/env python3
"""
Create a simple flat-ground USD with a Cube primitive collision shape.

UsdGeom.Cube maps directly to a PhysX box shape — GPU-native, no mesh cooking,
no "failed to cook GPU-compatible mesh" warning unlike CAD-imported USDs.

The slab's TOP FACE is at z=0 in local space.  Place the prim via prim_config
at the desired world z so the surface lands at the right height:
  - ground.usd: prim_config translate z = 0.35  →  robot birth z ≈ 0.58
  - cur_mixed.usd: prim_config translate z = 0.50  →  robot birth z ≈ 0.625
  To blend alongside cur_mixed, use translate z ≈ 0.375 (midpoint).

Usage — run inside the Isaac Sim conda container:

  apptainer exec --nv <container.sif> \\
    conda run -n isaaclab python /ws/scripts/make_flat_ground_usd.py \\
      /ws/src/FTR-benchmark/ftr_envs/assets/terrain/usd/flat_patch.usd

  # Or with any Python that has 'pxr' (pip install usd-core):
  python make_flat_ground_usd.py flat_patch.usd --width 16 --depth 16

Arguments:
  output      Path to write the .usd file.
  --width W   Slab extent in the X direction (metres, default 16).
  --depth D   Slab extent in the Y direction (metres, default 16).
  --thick T   Slab thickness (metres, default 0.5).  Top face stays at z=0.
"""

import argparse
import sys
from pathlib import Path


def create_flat_ground_usd(
    output_path: str,
    width: float = 16.0,
    depth: float = 16.0,
    thickness: float = 0.5,
) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError:
        print(
            "ERROR: 'pxr' not found.\n"
            "  Option A — run inside the isaaclab conda env (already has pxr).\n"
            "  Option B — pip install usd-core"
        )
        sys.exit(1)

    output_path = str(Path(output_path).resolve())
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root xform — Isaac Sim will position this prim via prim_config.
    root = UsdGeom.Xform.Define(stage, "/ground")
    stage.SetDefaultPrim(root.GetPrim())

    # UsdGeom.Cube → PhysX box collider.  Always GPU-compatible, zero cooking.
    cube = UsdGeom.Cube.Define(stage, "/ground/slab")
    cube.CreateSizeAttr(1.0)  # unit cube shaped by XformOps

    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.AddScaleOp().Set(Gf.Vec3f(width, depth, thickness))
    # Shift down so the TOP face of the unit cube lands at z=0 in local space.
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -thickness / 2.0))

    # Static collision — no UsdPhysics.RigidBodyAPI means immovable (static) body.
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    stage.Save()
    print(f"Written : {output_path}")
    print(f"Slab    : {width:.1f} m (X) × {depth:.1f} m (Y) × {thickness:.2f} m thick")
    print(f"Top face: z = 0 in local space")
    print(f"Tip     : use prim_config xformOp:translate [X, Y, 0.375] to blend")
    print(f"         alongside cur_mixed (birth z ≈ 0.58-0.63)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Create a flat-ground USD with a GPU-friendly box collider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("output", help="Output USD file path")
    p.add_argument("--width",  type=float, default=16.0, help="X extent in metres (default 16)")
    p.add_argument("--depth",  type=float, default=16.0, help="Y extent in metres (default 16)")
    p.add_argument("--thick",  type=float, default=0.5,  help="Thickness in metres (default 0.5)")
    args = p.parse_args()
    create_flat_ground_usd(args.output, args.width, args.depth, args.thick)
