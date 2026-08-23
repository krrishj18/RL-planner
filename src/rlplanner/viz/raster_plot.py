"""Raster rendering. Duck-typed: any object with cell_m, origin, nx, ny, height, cls, damage."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from rlplanner.scene import schema
from rlplanner.viz import palette as P
from rlplanner.viz.layout import legend_outside

LAYERS = ("cls", "height", "damage")
_REQUIRED = ("cell_m", "origin", "nx", "ny")


def raster_extent(raster) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) in world metres for `imshow(extent=...)`."""
    for f in _REQUIRED:
        if not hasattr(raster, f):
            raise AttributeError(f"raster {type(raster).__name__} has no field {f!r}")
    cell = float(raster.cell_m)
    if cell <= 0:
        raise ValueError(f"raster.cell_m must be > 0, got {cell}")
    x0, y0 = float(raster.origin[0]), float(raster.origin[1])
    return x0, x0 + int(raster.nx) * cell, y0, y0 + int(raster.ny) * cell


def _layer(raster, name: str) -> np.ndarray:
    if not hasattr(raster, name):
        raise AttributeError(f"raster {type(raster).__name__} has no layer {name!r}")
    a = np.asarray(getattr(raster, name))
    want = (int(raster.ny), int(raster.nx))
    if a.shape != want:
        raise ValueError(f"raster.{name} shape {a.shape} != (ny, nx) {want}")
    return a


def class_rgb(raster, shade: bool = True) -> np.ndarray:
    """(ny, nx, 3) class-colour image, optionally shaded by height."""
    cls = _layer(raster, "cls").astype(np.int64)
    if cls.size and (cls.min() < 0 or cls.max() >= schema.N_CLASSES):
        raise ValueError(f"raster.cls values outside [0, {schema.N_CLASSES}): "
                         f"min={cls.min()} max={cls.max()}")
    rgb = P.class_rgb_array()[cls]
    if shade and hasattr(raster, "height"):
        rgb = P.shade_by_height(rgb, _layer(raster, "height"))
    return rgb


def plot_raster(raster, ax: Axes | None = None, layer: str = "cls", shade: bool = True,
                colorbar: bool = True, legend: bool = True, title: str | None = None) -> Axes:
    """Draw one raster layer. Returns the axes."""
    if layer not in LAYERS:
        raise ValueError(f"layer {layer!r} not in {LAYERS}")
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        if legend and layer == "cls":
            fig.subplots_adjust(left=0.08, right=0.82, top=0.94, bottom=0.07)
    extent = raster_extent(raster)

    if layer == "cls":
        im = ax.imshow(class_rgb(raster, shade=shade), origin="lower", extent=extent,
                       interpolation="nearest")
        if legend:
            add_class_legend(ax, raster)
    elif layer == "height":
        h = _layer(raster, "height").astype(np.float64)
        im = ax.imshow(h, origin="lower", extent=extent, interpolation="nearest", cmap="cividis",
                       vmin=0.0, vmax=float(np.nanmax(h)) if h.size and np.nanmax(h) > 0 else 1.0)
        if colorbar:
            ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="height [m]")
    else:
        d = _layer(raster, "damage").astype(np.float64)
        im = ax.imshow(d, origin="lower", extent=extent, interpolation="nearest",
                       cmap=P.DAMAGE_CMAP, vmin=0.0, vmax=1.0)
        if colorbar:
            ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="damage")

    if getattr(raster, "humans", None) is not None and len(raster.humans):
        hs = raster.humans
        try:
            ax.scatter(np.asarray(hs["x"]), np.asarray(hs["y"]), s=12, marker="x",
                       c=P.human_color("casualty"), linewidths=0.8, zorder=5)
        except (IndexError, KeyError, TypeError):
            pass  # humans present but not the documented struct array

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title if title is not None else
                 f"raster {layer}  {raster.nx}x{raster.ny} @ {float(raster.cell_m):g} m", fontsize=9)
    return ax


def add_class_legend(ax: Axes, raster=None, side: str = "right", ncols: int = 2) -> None:
    """Legend of the raster classes present (all classes if `raster` is None), outside the axes."""
    if raster is None:
        present = list(range(schema.N_CLASSES))
    else:
        present = sorted(int(c) for c in np.unique(_layer(raster, "cls")))
    handles = [Patch(facecolor=P.class_color(c), label=schema.CLASS_NAMES[c]) for c in present]
    legend_outside(ax, handles, side=side, ncols=ncols, fontsize=5.5)


__all__ = ["plot_raster", "add_class_legend", "class_rgb", "raster_extent", "LAYERS"]
