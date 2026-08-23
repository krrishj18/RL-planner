#!/usr/bin/env python
"""Single robot, BT policy: belief panels at a few decisions so the RayFronts emulation can be
eyeballed (observed area, voxel sims, rays, frontiers) plus an episode mp4.

    uv run python scripts/rayfronts_demo.py --synthetic 1                    # window
    uv run python scripts/rayfronts_demo.py --scene data/scenes_v2/downtown_tornado_17.json \
        --query rubble --robots 3 --out runs/rayfronts_demo_v2.png

`--query` takes an index or any query name the embedding table knows (a name outside
`rayfronts.queries` is derived from the stored per-cell features). v2 scenes are up to 1500 m
across, so the belief panels crop to the explored neighbourhood unless `--no-zoom` is given.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from play_episode import build_env, build_policy, load_scene
from rlplanner.sim.config import EnvConfig
from rlplanner.viz import display
from rlplanner.viz.frame import (belief_legend_handles, draw_belief_panel, n_found, render_frame)

DEFAULT_CHECKPOINTS = (1, 2, 4, 8, 16, 32)
ZOOM_PAD_M = 60.0
DEFAULT_OUT = "runs/rayfronts_demo.png"


def _zoom(ax, st) -> None:
    """Crop a belief panel to the explored neighbourhood (v2 regions dwarf a short episode)."""
    xy = np.concatenate([np.asarray(r.trajectory, float) for r in st.robots], 0)
    x0, y0, x1, y1 = st.scene.region
    lo = np.maximum(xy.min(0) - ZOOM_PAD_M, (x0, y0))
    hi = np.minimum(xy.max(0) + ZOOM_PAD_M, (x1, y1))
    side = max(hi[0] - lo[0], hi[1] - lo[1], 4 * ZOOM_PAD_M)
    cx, cy = 0.5 * (lo + hi)
    ax.set_xlim(max(x0, cx - side / 2), min(x1, cx + side / 2))
    ax.set_ylim(max(y0, cy - side / 2), min(y1, cy + side / 2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene")
    src.add_argument("--synthetic", type=int, metavar="SEED")
    ap.add_argument("--out", default=None,
                    help=f"write the panel figure here instead of opening a window "
                         f"(default {DEFAULT_OUT} when no window is available)")
    ap.add_argument("--mp4", default=None, help="mp4 path (default: --out with .mp4)")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--query", default="0", help="query index or name")
    ap.add_argument("--robots", type=int, default=1)
    ap.add_argument("--no-zoom", action="store_true",
                    help="show the whole region instead of the explored neighbourhood")
    ap.add_argument("--policy", default="nearest_frontier")
    ap.add_argument("--checkpoints", default=",".join(str(c) for c in DEFAULT_CHECKPOINTS))
    ap.add_argument("--every-n", type=int, default=2, help="mp4 frame every N decisions")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    display.select_backend(a.out is None)
    import matplotlib.pyplot as plt

    checkpoints = [int(c) for c in a.checkpoints.split(",") if c.strip()]
    if not checkpoints:
        raise SystemExit("[rayfronts_demo] --checkpoints is empty")
    cfg = EnvConfig.from_yaml(a.config) if a.config else EnvConfig()
    cfg.robot.n_robots = max(1, int(a.robots))
    query = int(a.query) if a.query.lstrip("-").isdigit() else a.query
    if isinstance(query, int) and not (0 <= query < cfg.n_queries):
        raise SystemExit(f"[rayfronts_demo] --query {query} outside [0, {cfg.n_queries})")

    scene = load_scene(a)
    env = build_env(scene, cfg, a.seed)
    policy = build_policy(a.policy, cfg, a.seed)
    obs = env.reset()

    ncols = int(math.ceil(len(checkpoints) / 2))
    fig, axs = plt.subplots(2, ncols, figsize=(6.2 * ncols, 12.4), dpi=110, squeeze=False)
    panels = [axs[i // ncols][i % ncols] for i in range(2 * ncols)]
    for ax in panels[len(checkpoints):]:
        ax.axis("off")

    want_mp4 = not a.no_mp4 and (a.mp4 is not None or a.out is not None)
    frames: list[np.ndarray] = []
    rows: list[str] = []
    limit = max(checkpoints)
    for d in range(1, limit + 1):
        actions = np.asarray(policy.act(obs, env.state))
        obs, reward, done, info = env.step(actions)
        st = env.state
        if want_mp4 and (d % a.every_n == 0 or done):
            frames.append(render_frame(st, query_idx=query, focus_robot=0))
        if d in checkpoints:
            ax = panels[checkpoints.index(d)]
            counts = draw_belief_panel(ax, st, query_idx=query, focus_robot=0, legend=False)
            if not a.no_zoom:
                _zoom(ax, st)
            live = int(st.rays.live().sum()) if st.rays is not None and st.rays.n else 0
            ax.set_title(f"decision {d}  t={st.t:.0f}s  cov={100 * st.coverage:.1f}%  "
                         f"rays={live} frontiers={len(st.frontier_clusters)} "
                         f"found={n_found(st)}/{st.n_casualties}", fontsize=8)
            rows.append(f"  d={d:<3d} t={st.t:6.0f}s cov={100 * st.coverage:5.1f}% "
                        f"live_rays={live:4d} frontiers={len(st.frontier_clusters):3d} "
                        f"tokens={counts['tokens']:3d} "
                        f"found={n_found(st)}/{st.n_casualties} R={st.cum_reward:+.2f}")
        if done:
            for ax in panels[len([c for c in checkpoints if c <= d]):]:
                ax.axis("off")
            break

    qname = cfg.rayfronts.queries[query] if isinstance(query, int) else query
    fig.suptitle(f"RayFronts emulation, {cfg.robot.n_robots} robot(s), {a.policy} policy, "
                 f"query {query}: {qname!r}   embeddings: {env.rf.emb.source} D={env.rf.D}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.985))
    fig.legend(handles=belief_legend_handles(), loc="lower center", ncols=5, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.004))   # one legend, below every panel
    print("\n".join(rows))
    display.finish(fig, a.out, DEFAULT_OUT, "rayfronts_demo")
    plt.close(fig)

    if want_mp4 and frames:
        import imageio.v2 as iio
        from rlplanner.viz.recorder import _pad_to_macro_block

        mp4 = Path(a.mp4) if a.mp4 else Path(a.out or DEFAULT_OUT).with_suffix(".mp4")
        mp4.parent.mkdir(parents=True, exist_ok=True)
        with iio.get_writer(mp4, fps=a.fps, codec="libx264", quality=8,
                            macro_block_size=None) as w:
            for f in frames:
                w.append_data(_pad_to_macro_block(f))
        print(f"[rayfronts_demo] {mp4}  {len(frames)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
