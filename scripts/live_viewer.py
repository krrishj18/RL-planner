#!/usr/bin/env python
"""Watch an episode as it runs: robots over the ground truth, the RayFronts belief being built.

    uv run python scripts/live_viewer.py --synthetic 3 --robots 3                     # window
    uv run python scripts/live_viewer.py --scene data/scenes_v2/downtown_tornado_17.json \
        --policy ray_follower --robots auto --record runs/live_demo_v2.mp4 --max-decisions 60

The env is stepped inside the animation callback (policy.act -> env.step), so nothing is
pre-recorded; `--record` writes the very frames the window shows and needs no display
(MPLBACKEND=Agg runs it headless).

Keys: space play/pause | n step | +/- speed | 0-9 query | f cycle focus robot |
      v cycle whose map the belief panel shows | s segments on/off |
      r restart (same seed) | R restart (new seed) | w write png | q quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from play_episode import ALIASES, build_env, build_policy, load_scene, policy_names
from rlplanner.sim.config import EnvConfig
from rlplanner.viz import display
from rlplanner.viz.live import LiveViewer

DEFAULT_PNG = "runs/live_frame.png"


def _mean(a) -> float:
    return sum(a) / len(a) if a else 0.0

HEADLESS_DECISIONS = 10          # decisions to run before the fallback PNG when there is no window


def auto_robots(scene) -> int:
    """train.scenes' area rule (3 per 400x400 m, clipped to [3, 8]); 3 if train is unavailable."""
    try:
        from rlplanner.train.scenes import auto_robots as rule, region_area_km2
    except ImportError:
        return 3
    x0, y0, x1, y1 = scene.region
    return int(rule(region_area_km2((x1 - x0, y1 - y0))))


def parse_robots(spec: str, scene) -> int:
    if str(spec).strip().lower() == "auto":
        return auto_robots(scene)
    n = int(spec)
    if n < 1:
        raise SystemExit(f"[live_viewer] --robots {spec!r} must be >= 1 or 'auto'")
    return n


def parse_query(spec: str) -> int | str:
    s = str(spec).strip()
    return int(s) if s.lstrip("-").isdigit() else s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="scene.json")
    src.add_argument("--synthetic", type=int, metavar="SEED")
    ap.add_argument("--policy", default="nearest_frontier",
                    help="a sim.baselines name (random|nearest_frontier|ray_follower|"
                         "segment_seeker|oracle), or ckpt:<path.pt> / ckpt:latest with --run")
    ap.add_argument("--run", default=None, help="run name under runs/ for --policy ckpt:latest")
    ap.add_argument("--device", default="auto", help="torch device for a ckpt: policy")
    ap.add_argument("--robots", default="auto", metavar="N|auto",
                    help="robot count, or 'auto' for the train.scenes area rule")
    ap.add_argument("--query", default="0", metavar="NAME|IDX",
                    help="query shown in the belief panel (index or name)")
    ap.add_argument("--focus", type=int, default=0, help="robot whose tokens are drawn")
    ap.add_argument("--robot", type=int, default=None, metavar="R",
                    help="draw robot R's own map instead of the team union (key 'v' cycles)")
    ap.add_argument("--show-local", action="store_true",
                    help="add the focused robot's local ego crop (the actor's dense input)")
    ap.add_argument("--no-segments", action="store_true",
                    help="do not overlay the segments on the belief panel")
    ap.add_argument("--speed", type=float, default=4.0, help="decisions per second")
    ap.add_argument("--record", default=None, metavar="MP4",
                    help="write the frames to an mp4 (with a window when one is available)")
    ap.add_argument("--fps", type=float, default=None, help="mp4 frame rate (default --speed)")
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--png", default=DEFAULT_PNG, help="path the 'w' key writes to")
    ap.add_argument("--gt", default="auto", choices=("auto", "scene", "raster"),
                    help="ground-truth background: baked scene plot, raster classes, or auto")
    ap.add_argument("--no-window", action="store_true", help="record without opening a window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--figsize", nargs=2, type=float, default=(19.2, 9.6), metavar=("W", "H"))
    ap.add_argument("--config", default=None, help="EnvConfig yaml")
    a = ap.parse_args(argv)

    known = policy_names()
    if not a.policy.startswith("ckpt:") and ALIASES.get(a.policy, a.policy) not in known:
        raise SystemExit(f"[live_viewer] unknown --policy {a.policy!r}: "
                         f"use one of {known} or ckpt:<path.pt>")
    display.select_backend(not a.no_window)
    scene = load_scene(a)
    cfg = EnvConfig.from_yaml(a.config) if a.config else EnvConfig()
    cfg.robot.n_robots = parse_robots(a.robots, scene)
    errs = cfg.validate()
    if errs:
        raise SystemExit(f"[live_viewer] invalid EnvConfig: {'; '.join(errs)}")
    query = parse_query(a.query)
    if isinstance(query, int) and not (0 <= query < cfg.n_queries):
        raise SystemExit(f"[live_viewer] --query {query} outside [0, {cfg.n_queries})")
    if not (0 <= a.focus < cfg.robot.n_robots):
        raise SystemExit(f"[live_viewer] --focus {a.focus} outside [0, {cfg.robot.n_robots})")
    if a.robot is not None and not (0 <= a.robot < cfg.robot.n_robots):
        raise SystemExit(f"[live_viewer] --robot {a.robot} outside [0, {cfg.robot.n_robots})")

    env = build_env(scene, cfg, a.seed)
    policy = build_policy(a.policy, cfg, a.seed, a.run, a.device)
    lv = LiveViewer(env, policy, query=query, focus=a.focus, speed=a.speed,
                    max_decisions=a.max_decisions, seed=a.seed, figsize=tuple(a.figsize),
                    dpi=a.dpi, gt=a.gt, png=a.png, robot=a.robot, show_local=a.show_local,
                    segments=not a.no_segments)
    print(f"[live_viewer] {scene.meta.preset} seed={scene.meta.seed} "
          f"robots={cfg.robot.n_robots} policy={a.policy} query={query!r} "
          f"speed={a.speed}/s cells={env.raster.ny}x{env.raster.nx}")

    if display.gui_active():
        print(f"[live_viewer] {lv.KEYS}")
        lv.start(record=a.record, fps=a.fps)
        if a.record:
            print(f"[live_viewer] wrote {a.record}")
        return 0
    if a.record:
        path, n = lv.record(a.record, fps=a.fps, max_decisions=a.max_decisions, progress=True)
        print(f"[live_viewer] wrote {path}  {n} frames  {n - 1} decisions  "
              f"{lv.record_rate:.2f} decisions/s end to end "
              f"(step {1e3 * _mean(lv.step_times):.0f} ms, artists "
              f"{1e3 * _mean(lv.draw_times):.0f} ms, draw {1e3 * _mean(lv.render_times):.0f} ms)")
        return 0
    lv.build()
    for _ in range(a.max_decisions if a.max_decisions is not None else HEADLESS_DECISIONS):
        if not lv.advance():
            break
    lv.refresh()
    print(f"[live_viewer] no window available; wrote {lv.save_png()} "
          f"after {lv.n_decisions} decisions (pass --record for an mp4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
