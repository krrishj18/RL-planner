#!/usr/bin/env python
"""Scrub through recorded episode frames with a matplotlib slider.

    uv run python scripts/episode_viewer.py runs/ep.pkl
    uv run python scripts/episode_viewer.py runs/frames/          # directory of PNGs
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from rlplanner.viz import display

IMG_EXT = (".png", ".jpg", ".jpeg")
DEFAULT_OUT = "runs/episode_viewer.png"


def load_frames(path: str | Path) -> tuple[list[np.ndarray], list[dict]]:
    """Frames + optional snapshots from a pickle (list or {"frames", "snapshots"}) or a PNG dir."""
    import imageio.v2 as iio

    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        if not files:
            raise SystemExit(f"[episode_viewer] no {'/'.join(IMG_EXT)} frames in {p}")
        return [np.asarray(iio.imread(f)) for f in files], []
    if not p.exists():
        raise SystemExit(f"[episode_viewer] {p} does not exist")
    with open(p, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        snaps = obj.get("snapshots", [])
        if obj.get("frames_png"):
            frames = [np.asarray(iio.imread(b)) for b in obj["frames_png"]]
        else:
            frames = obj.get("frames", [])
    else:
        frames, snaps = list(obj), []
    if not frames:
        raise SystemExit(f"[episode_viewer] {p} holds no frames")
    return [np.asarray(f) for f in frames], list(snaps)


def _caption(i: int, n: int, snaps: list[dict]) -> str:
    s = snaps[i] if i < len(snaps) else None
    if not s:
        return f"frame {i + 1}/{n}"
    return (f"frame {i + 1}/{n}   t={s.get('t', 0):.0f}s   "
            f"found={s.get('found')}/{s.get('n_casualties')}   "
            f"coverage={100 * s.get('coverage', 0):.1f}%   reward={s.get('reward', 0):+.3f}")


def build_viewer(frames, snaps, figsize=(16.0, 6.0)):
    """Figure + slider; returns (fig, slider, update) so tests can drive it headless."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0.02, 0.10, 0.96, 0.84))
    ax.axis("off")
    im = ax.imshow(frames[0])
    ttl = ax.set_title(_caption(0, len(frames), snaps), fontsize=9)
    ax_s = fig.add_axes((0.12, 0.03, 0.76, 0.03))
    slider = Slider(ax_s, "frame", 0, max(0, len(frames) - 1), valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        im.set_data(frames[i])
        ttl.set_text(_caption(i, len(frames), snaps))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        if event.key in ("right", "n"):
            slider.set_val(min(len(frames) - 1, int(slider.val) + 1))
        elif event.key in ("left", "p"):
            slider.set_val(max(0, int(slider.val) - 1))

    fig.canvas.mpl_connect("key_press_event", on_key)
    return fig, slider, update


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="pickled frames (or {'frames','snapshots'}) or a directory of PNGs")
    ap.add_argument("--no-show", action="store_true", help="build the figure and exit (headless)")
    ap.add_argument("--save", default=None, metavar="PNG",
                    help=f"write the current frame instead of opening a window "
                         f"(default {DEFAULT_OUT} when no window is available)")
    a = ap.parse_args(argv)

    display.select_backend(a.save is None and not a.no_show)
    frames, snaps = load_frames(a.path)
    print(f"[episode_viewer] {len(frames)} frames, {frames[0].shape} "
          f"({len(snaps)} snapshots)  left/right or the slider to scrub")
    fig, slider, update = build_viewer(frames, snaps)
    if a.no_show:
        return 0
    display.finish(fig, a.save, DEFAULT_OUT, "episode_viewer", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
