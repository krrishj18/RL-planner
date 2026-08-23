#!/usr/bin/env python
"""Evaluate checkpoints and/or baselines on the held-out scenes.

    uv run python scripts/eval_policy.py --policy ray_follower --scenes synthetic:0-20 --episodes 4
    uv run python scripts/eval_policy.py --policy runs/run1/latest.pt --policy oracle --episodes 20
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from rlplanner.sim.config import EnvConfig
from rlplanner.train.evaluate import (EVAL_COLS, EVAL_META, by_bucket, evaluate_policy,
                                      format_table, make_actor, meta_arrays, summarise,
                                      write_csv, write_episode_csv)
from rlplanner.train.scenes import AUTO, SceneBank, parse_robots, parse_t_max


def _load_cfg(a) -> EnvConfig:
    """EnvConfig from --config / --variant, with the comms overrides applied."""
    import yaml
    if a.variant:
        path = Path(a.variant)
        if not path.exists():
            path = Path("configs/variants") / f"{a.variant}.yaml"
        d = dict(yaml.safe_load(path.read_text()) or {})
        d.pop("train", None)
        d.pop("notes", None)
        cfg = EnvConfig.from_dict(d)
    else:
        cfg = EnvConfig.from_yaml(a.config) if a.config else EnvConfig()
    if a.comms:
        cfg.comms.mode = a.comms
    if a.comms_range is not None:
        cfg.comms.range_m = float(a.comms_range)
        cfg.comms.randomize_range = False
        if not a.comms:
            # a link range only means anything under decentralised execution: asking for one on a
            # `full` config used to be silently ignored (CONTRACTS.md 5.1)
            cfg.comms.mode = "range"
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", required=True,
                    help="baseline name (random|nearest|ray_follower|segment_seeker|oracle) "
                         "or a checkpoint path; repeatable")
    ap.add_argument("--scenes", default="synthetic:0-200", nargs="+",
                    help="synthetic:A-B or scene-json globs/paths (unquoted shell glob is fine)")
    ap.add_argument("--split", default="heldout", choices=("heldout", "train", "all"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--robots", default="3",
                    help="N or 'auto' (clip(round(3*sqrt(area_km2/0.16)), 3, 8) per scene)")
    ap.add_argument("--t-max", default=None,
                    help="seconds or 'auto' (clip(600*sqrt(area_km2/0.16), 600, 1500) per scene); "
                         "default = the EnvConfig value")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="CSV path (default runs/eval_<stamp>.csv)")
    ap.add_argument("--stochastic", action="store_true", help="sample instead of argmax")
    ap.add_argument("--config", default=None, help="EnvConfig yaml")
    ap.add_argument("--variant", default=None,
                    help="configs/variants/<name>.yaml (its EnvConfig block)")
    ap.add_argument("--comms", default=None, choices=("full", "range"),
                    help="override EnvConfig.comms.mode for the evaluation")
    ap.add_argument("--comms-range", default=None,
                    help="fixed link range in metres ('inf' allowed) for the evaluation")
    ap.add_argument("--backend", default="auto", choices=("auto", "subproc", "serial"))
    ap.add_argument("--workers", type=int, default=None,
                    help="env worker processes (default min(nproc - 2, episodes))")
    a = ap.parse_args(argv)
    a.scenes = " ".join(a.scenes) if isinstance(a.scenes, list) else a.scenes

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(a.seed)
    cfg = _load_cfg(a)
    lo, _ = parse_robots(a.robots)
    t_spec = parse_t_max(a.t_max)
    bank = SceneBank(a.scenes)
    cfg.robot.n_robots = bank.robot_bounds((lo, lo), a.split)[0]
    if t_spec > 0:
        cfg.t_max_s = float(t_spec)
    t_max = None if t_spec > 0 else float(AUTO)
    errs = cfg.validate()
    if errs:
        raise SystemExit(f"[eval_policy] invalid EnvConfig: {'; '.join(errs)}")
    keys = bank.split(a.split)
    rb = bank.robot_bounds((lo, lo), a.split)
    rob = f"auto({rb[0]}-{rb[1]})" if lo <= 0 else str(lo)
    print(f"[eval_policy] {len(keys)} {a.split} scene(s) {bank.bucket_counts(a.split)}, "
          f"{a.episodes} episodes, robots={rob}, "
          f"t_max={'auto' if t_max is not None else f'{cfg.t_max_s:.0f}s'}, device={dev}")

    out = Path(a.out) if a.out else Path("runs") / f"eval_{int(time.time())}.csv"
    ep_out = out.with_suffix(".episodes.csv")
    rows: dict[str, dict] = {}
    for spec in a.policy:
        actor = make_actor(spec, cfg, device=dev, seed=a.seed, deterministic=not a.stochastic)
        t0 = time.perf_counter()
        res, ep = evaluate_policy(actor, bank, cfg, a.episodes, lo, seed=a.seed, split=a.split,
                                  backend=a.backend, workers=a.workers, t_max=t_max,
                                  return_rows=True)
        name = getattr(actor, "name", spec)
        rows[name] = summarise({**res, **meta_arrays(ep)})
        buckets = by_bucket(ep)
        if len(buckets) > 1:                         # size-bucket breakdown
            for b, st in buckets.items():
                rows[f"{name}[{b}]"] = st
        write_episode_csv(ep_out, ep, policy=name)
        print(f"  {spec}: {time.perf_counter() - t0:.1f}s")
    print(format_table(rows, EVAL_COLS + EVAL_META) + "\n  (mean +- 95% CI of the mean; "
          "[bucket] rows are the bank's area terciles)")

    write_csv(out, rows, EVAL_COLS + EVAL_META,
              extra={"episodes": a.episodes, "robots": rob, "split": a.split,
                     "scenes": a.scenes, "t_max": "auto" if t_max is not None else cfg.t_max_s,
                     "comms": cfg.comms.mode, "comms_range_m": cfg.comms.range_m,
                     "variant": a.variant or ""})
    print(f"[eval_policy] wrote {out} and {ep_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
