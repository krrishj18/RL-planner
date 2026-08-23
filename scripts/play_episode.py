#!/usr/bin/env python
"""Roll out a baseline or a trained checkpoint and show (or record) the three-panel episode.

With `--out` the frames go to an mp4/gif; without it the episode opens in a scrub window
(`--show` is implied), falling back to runs/ep.mp4 when no window is available. For a window that
plays while the episode runs, use `scripts/live_viewer.py`.

    uv run python scripts/play_episode.py --synthetic 1 --policy bt --robots 3   # window
    uv run python scripts/play_episode.py --scene s.json --policy ckpt:runs/long/latest.pt --out runs/ep.mp4
    uv run python scripts/play_episode.py --synthetic 1 --policy ckpt:latest --run long
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlplanner.scene import schema
from rlplanner.sim.config import EnvConfig
from rlplanner.viz import display
from rlplanner.viz.recorder import EpisodeRecorder

POLICIES = ("random", "nearest_frontier", "ray_follower", "segment_seeker", "oracle")
ALIASES = {"nearest": "nearest_frontier"}
DEFAULT_OUT = "runs/ep.mp4"


def policy_names() -> tuple[str, ...]:
    """What `sim.baselines` registers right now, falling back to the documented set."""
    try:
        from rlplanner.sim import baselines
    except ImportError:
        return POLICIES
    return tuple(sorted(getattr(baselines, "POLICIES", None) or POLICIES))


def load_scene(args) -> schema.Scene:
    if args.scene:
        return schema.Scene.from_json(args.scene)
    return schema.make_synthetic_scene(int(args.synthetic))


def build_env(scene, cfg: EnvConfig, seed: int):
    """Import the sim lazily so the script gives a clear message before sim/env.py lands."""
    try:
        from rlplanner.sim.env import DisasterEnv
    except ImportError as exc:
        raise SystemExit(f"[play_episode] rlplanner.sim.env is not available yet ({exc}). "
                         f"The visualizer is ready; re-run once the sim agent lands sim/env.py.")
    return DisasterEnv(scene, cfg, seed=seed)


def resolve_ckpt(ref: str, run: str | None) -> Path:
    """`ckpt:<path>` or `ckpt:latest` with --run <name> -> runs/<name>/latest.pt."""
    if ref != "latest":
        p = Path(ref)
    elif run:
        p = Path("runs") / run / "latest.pt"
    else:
        raise SystemExit("[play_episode] --policy ckpt:latest needs --run <name>")
    if not p.exists():
        raise SystemExit(f"[play_episode] checkpoint {p} not found")
    return p


def build_policy(name: str, cfg: EnvConfig, seed: int = 0, run: str | None = None,
                 device: str = "auto"):
    """Baseline name, or `ckpt:<path>` / `ckpt:latest` for a trained TokenPolicy (greedy)."""
    if name.startswith("ckpt:"):
        import torch
        from rlplanner.train.evaluate import TorchActor, load_checkpoint
        dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                           else ("cpu" if device == "auto" else device))
        path = resolve_ckpt(name[len("ckpt:"):], run)
        policy, ck = load_checkpoint(path, dev)
        actor = TorchActor(policy, dev, deterministic=True, name=path.stem)
        print(f"[play_episode] {path} (update {ck.get('update', '?')}, "
              f"{ck.get('decisions', '?')} decisions) on {dev}, greedy")
        return actor
    try:
        from rlplanner.sim import baselines
    except ImportError as exc:
        raise SystemExit(f"[play_episode] rlplanner.sim.baselines is not available yet ({exc}).")
    name = ALIASES.get(name, name)
    known = getattr(baselines, "POLICIES", {})
    if name not in known:
        raise SystemExit(f"[play_episode] unknown --policy {name!r}: use one of "
                         f"{sorted(known) or list(POLICIES)} or ckpt:<path.pt>")
    import inspect
    P = known[name]
    params = inspect.signature(P).parameters
    kw = {k: v for k, v in (("cfg", cfg), ("queries", cfg.rayfronts.queries), ("seed", seed))
          if k in params}
    try:
        return P(**kw)
    except TypeError as exc:
        raise SystemExit(f"[play_episode] cannot construct {name}{inspect.signature(P)}: {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="scene.json")
    src.add_argument("--synthetic", type=int, metavar="SEED")
    ap.add_argument("--policy", default="nearest_frontier",
                    help="a sim.baselines name (random|nearest_frontier|ray_follower|"
                         "voxel_seeker|oracle), or ckpt:<path.pt> / ckpt:latest with --run")
    ap.add_argument("--run", default=None, help="run name under runs/ for --policy ckpt:latest")
    ap.add_argument("--device", default="auto", help="torch device for a ckpt: policy")
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--query", default="0", metavar="NAME|IDX",
                    help="query shown in the belief panel: an index into the mission list, or a "
                         "name the embedding table knows")
    ap.add_argument("--out", default=None,
                    help=f"mp4 (or .gif) output path; without it the episode is shown in a "
                         f"scrub window (default {DEFAULT_OUT} when no window is available)")
    ap.add_argument("--show", action="store_true",
                    help="open the scrub window even when --out is given")
    ap.add_argument("--frames-dir", default=None, help="also write every frame as a PNG here")
    ap.add_argument("--pickle", default=None, help="also pickle frames+snapshots for episode_viewer")
    ap.add_argument("--every-n", type=int, default=1, help="record every N decisions")
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--focus", type=int, default=0, help="robot whose tokens are drawn (-1 = none)")
    ap.add_argument("--robot", type=int, default=None, metavar="R",
                    help="draw robot R's own map instead of the team union")
    ap.add_argument("--show-local", action="store_true",
                    help="add the focused robot's local ego crop panel")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="EnvConfig yaml")
    a = ap.parse_args(argv)

    display.select_backend(a.show or a.out is None)
    cfg = EnvConfig.from_yaml(a.config) if a.config else EnvConfig()
    cfg.robot.n_robots = int(a.robots)
    errs = cfg.validate()
    if errs:
        raise SystemExit(f"[play_episode] invalid EnvConfig: {'; '.join(errs)}")
    q = str(a.query).strip()
    query = int(q) if q.lstrip("-").isdigit() else q
    if isinstance(query, int) and not (0 <= query < cfg.n_queries):
        raise SystemExit(f"[play_episode] --query {query} outside [0, {cfg.n_queries}) "
                         f"({list(cfg.rayfronts.queries)})")
    if a.robot is not None and not (0 <= a.robot < cfg.robot.n_robots):
        raise SystemExit(f"[play_episode] --robot {a.robot} outside [0, {cfg.robot.n_robots})")

    known = policy_names()
    if not a.policy.startswith("ckpt:") and ALIASES.get(a.policy, a.policy) not in known:
        raise SystemExit(f"[play_episode] unknown --policy {a.policy!r}: "
                         f"use one of {known} or ckpt:<path.pt>")
    scene = load_scene(a)
    env = build_env(scene, cfg, a.seed)
    policy = build_policy(a.policy, cfg, a.seed, a.run, a.device)
    rec = EpisodeRecorder(env, every_n_decisions=a.every_n, query_idx=query,
                          focus_robot=None if a.focus < 0 else a.focus, dpi=a.dpi,
                          robot=a.robot, show_local=a.show_local)
    frames = rec.run(policy, max_decisions=a.max_decisions, progress=True)

    window = a.out is None and display.gui_active()
    out = None if window else Path(a.out or DEFAULT_OUT)
    if out is not None:
        (rec.save_gif if out.suffix.lower() == ".gif" else rec.save_mp4)(out, fps=a.fps)
    if a.frames_dir:
        rec.save_frames(a.frames_dir)
    if a.pickle:
        rec.save_pickle(a.pickle)
    last = rec.snapshots[-1] if rec.snapshots else {}
    print(f"[play_episode] {out or 'window'}  {len(frames)} frames  policy={a.policy} robots={a.robots}\n"
          f"  t={last.get('t', 0):.0f}s found={last.get('found')}/{last.get('n_casualties')}"
          f" coverage={100 * last.get('coverage', 0):.1f}% reward={last.get('reward', 0):+.3f}")
    m = (rec.info_last or {}).get("metrics")
    if m:
        print(f"  metrics: {m}")
    if window or (a.show and display.gui_active()):
        import episode_viewer
        import matplotlib.pyplot as plt
        episode_viewer.build_viewer(frames, rec.snapshots)
        print("[play_episode] left/right or the slider to scrub, q to close")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
