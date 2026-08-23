#!/usr/bin/env python
"""Show a scene (and optionally its raster). Opens a window unless `--out` asks for a file.

    uv run python scripts/view_scene.py data/scenes/earthquake_0.json          # window
    uv run python scripts/view_scene.py --synthetic 3 --raster --out runs/scene3.png
"""
from __future__ import annotations

import argparse
import sys

from rlplanner.scene import schema
from rlplanner.viz import display
from rlplanner.viz.raster_plot import LAYERS, plot_raster
from rlplanner.viz.scene_plot import plot_scene

DEFAULT_OUT = "runs/scene.png"


def load_scene(args) -> schema.Scene:
    if args.scene:
        return schema.Scene.from_json(args.scene)
    return schema.make_synthetic_scene(int(args.synthetic))


def try_rasterize(scene: schema.Scene, cell_m: float):
    try:
        from rlplanner.sim.raster import rasterize
    except ImportError as exc:
        print(f"[view_scene] rlplanner.sim.raster not available ({exc}); skipping --raster",
              file=sys.stderr)
        return None
    return rasterize(scene, cell_m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="scene.json")
    ap.add_argument("--synthetic", type=int, default=None, metavar="SEED",
                    help="use schema.make_synthetic_scene(SEED) instead of a file")
    ap.add_argument("--raster", action="store_true", help="add a raster panel (needs sim.raster)")
    ap.add_argument("--layer", default="cls", choices=LAYERS, help="raster layer to draw")
    ap.add_argument("--cell-m", type=float, default=2.0)
    ap.add_argument("--out", default=None,
                    help=f"write this file instead of opening a window (default {DEFAULT_OUT} "
                         f"when no window is available)")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--ids", action="store_true", help="annotate building/human ids")
    ap.add_argument("--no-damage", action="store_true")
    ap.add_argument("--no-humans", action="store_true")
    a = ap.parse_args(argv)
    if not a.scene and a.synthetic is None:
        ap.error("give a scene.json or --synthetic SEED")

    display.select_backend(a.out is None)
    import matplotlib.pyplot as plt

    scene = load_scene(a)
    errs = schema.validate(scene)
    if errs:
        print(f"[view_scene] scene has {len(errs)} validation problems, first: {errs[0]}",
              file=sys.stderr)

    raster = try_rasterize(scene, a.cell_m) if a.raster else None
    ncols = 2 if raster is not None else 1
    fig = plt.figure(figsize=(9.6 * ncols, 9), dpi=a.dpi)
    gs = fig.add_gridspec(1, ncols, left=0.06 / ncols, right=0.87 if ncols == 1 else 0.93,
                          top=0.94, bottom=0.06, wspace=0.22)   # right margin = legend room
    axs = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    plot_scene(scene, ax=axs[0], show_damage=not a.no_damage, show_humans=not a.no_humans,
               show_ids=a.ids)
    if raster is not None:
        plot_raster(raster, ax=axs[1], layer=a.layer)
    n_cas = len(scene.casualties())
    print(f"[view_scene] region={scene.region} buildings={len(scene.buildings)} "
          f"humans={len(scene.humans)} ({n_cas} casualties)"
          + (f" raster={raster.nx}x{raster.ny}@{a.cell_m}m" if raster is not None else ""))
    display.finish(fig, a.out, DEFAULT_OUT, "view_scene")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
