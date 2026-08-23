"""Legend placement. Every legend sits *outside* its axes so the map is never covered."""
from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.legend import Legend

LEGEND_STYLE = dict(framealpha=0.9, handlelength=1.2, handletextpad=0.4, columnspacing=0.7,
                    borderpad=0.3, labelspacing=0.25, borderaxespad=0.0)
RIGHT_ANCHOR = (1.01, 1.0)      # axes coordinates: just right of the axes, top-aligned
BELOW_ANCHOR = (0.0, -0.035)    # just below the axes, left-aligned
SIDES = ("right", "below")


def legend_outside(ax: Axes, handles, labels=None, side: str = "right", ncols: int = 2,
                   fontsize: float = 5.5, title: str | None = None, artist: bool = False,
                   anchor: tuple[float, float] | None = None, **kw) -> Legend | None:
    """Legend anchored outside `ax`. `artist=True` adds a second legend without replacing the first.

    Reserve room in the figure (subplots_adjust / explicit gridspec margins) rather than relying on
    a layout engine: with an equal-aspect axes, constrained layout re-expands the axes over a
    legend placed below it.
    """
    if side not in SIDES:
        raise ValueError(f"legend side {side!r} not in {SIDES}")
    if not handles:
        return None
    bbox = anchor if anchor is not None else (RIGHT_ANCHOR if side == "right" else BELOW_ANCHOR)
    kwargs = dict(LEGEND_STYLE, fontsize=fontsize, ncols=int(ncols), loc="upper left",
                  bbox_to_anchor=bbox)
    if title is not None:
        kwargs.update(title=title, title_fontsize=fontsize)
    kwargs.update(kw)
    if artist:
        lab = list(labels) if labels is not None else [h.get_label() for h in handles]
        lg = Legend(ax, handles, lab, **kwargs)
        ax.add_artist(lg)
        return lg
    if labels is not None:
        return ax.legend(handles=handles, labels=list(labels), **kwargs)
    return ax.legend(handles=handles, **kwargs)


def legend_is_outside(ax: Axes, legend: Legend | None = None) -> bool:
    """True if the legend's box does not intersect the axes box (figure coordinates, after a draw)."""
    lg = legend if legend is not None else ax.get_legend()
    if lg is None:
        return True
    fig = ax.figure
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    return not ax.get_window_extent().transformed(inv).overlaps(
        lg.get_window_extent().transformed(inv))


__all__ = ["legend_outside", "legend_is_outside", "LEGEND_STYLE", "RIGHT_ANCHOR", "BELOW_ANCHOR"]
