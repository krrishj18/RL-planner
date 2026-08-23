#!/usr/bin/env python
"""Rollout throughput of the training loop: serial vs SubprocVecEnv, by envs and workers.

    uv run python scripts/bench_train.py
    uv run python scripts/bench_train.py --envs 8,16,32 --workers 4,8,10 --rollout 32
    uv run python scripts/bench_train.py --scenes 'data/scenes_v2/downtown_earthquake_40.json' \
        --robots auto --t-max auto --envs 4 --workers 2 --rollout 8 --no-serial

Reports decisions/s, ms/decision and (subproc) the resident memory of the env workers, which is
dominated by the scene raster + ray store of every slot the worker holds.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from rlplanner.sim.config import EnvConfig
from rlplanner.train.par_env import SerialVecEnv, SubprocVecEnv, default_workers
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig
from rlplanner.train.rollout import Collector
from rlplanner.train.scenes import AUTO, SceneBank, parse_robots, parse_scene_mix, parse_t_max


def ints(s: str) -> list[int]:
    return [int(x) for x in str(s).replace(",", " ").split()]


def rss_mb(pid: int) -> float:
    """Resident set size of one process, MB (0 if it is gone)."""
    try:
        with open(f"/proc/{int(pid)}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def worker_rss(vec) -> tuple[float, float]:
    """(total, max) worker RSS in MB; (0, 0) for the serial backend."""
    procs = getattr(vec, "procs", [])
    vals = [rss_mb(p.pid) for p in procs if p.is_alive()]
    return (sum(vals), max(vals) if vals else 0.0)


def bench(vec, dev, rollout: int, reps: int, use_bev: bool, update: bool,
          d_model: int) -> tuple[float, float]:
    col = Collector(vec, dev)
    ob = col.obs
    policy = TokenPolicy(ob.tokens.shape[3], ob.robot_feat.shape[2], ob.query_emb.shape[2],
                         d_model=d_model, use_bev=use_bev,
                         bev_channels=ob.bev.shape[1]).to(dev)
    ppo = PPO(policy, PPOConfig(epochs=2, n_minibatches=2), dev) if update else None
    col.rollout(policy, 2)                                  # warm-up (jit, cudnn, pipes)
    t_roll = t_all = 0.0
    n = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        batch, st = col.rollout(policy, rollout)
        t1 = time.perf_counter()
        if ppo is not None:
            ppo.update(batch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_all += time.perf_counter() - t0
        t_roll += t1 - t0
        n += int(st["decisions"])
    return n / max(t_roll, 1e-9), n / max(t_all, 1e-9)


def _variant_cfg(name: str | None) -> EnvConfig:
    if not name:
        return EnvConfig()
    import yaml
    path = Path(name) if Path(name).exists() else Path("configs/variants") / f"{name}.yaml"
    d = dict(yaml.safe_load(path.read_text()) or {})
    d.pop("train", None)
    d.pop("notes", None)
    return EnvConfig.from_dict(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", default="8,16,32")
    ap.add_argument("--workers", default=None, help="default: 1(serial), 4, 8, nproc-2")
    ap.add_argument("--scenes", default="synthetic:0-40")
    ap.add_argument("--variant", default=None,
                    help="configs/variants/<name>.yaml — its EnvConfig block (comms, tokens, "
                         "reward). Default: plain EnvConfig (full comms)")
    ap.add_argument("--robots", default="3", help="N, LO-HI or 'auto' (area rule)")
    ap.add_argument("--t-max", default=None, help="seconds or 'auto' (area rule)")
    ap.add_argument("--scene-mix", default=None, help="e.g. small:0.5,medium:0.3,large:0.2")
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--use-bev", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(a.seed)
    cfg = _variant_cfg(a.variant)
    lo, hi = parse_robots(a.robots)
    t_spec = parse_t_max(a.t_max)
    mix = parse_scene_mix(a.scene_mix)
    bank = SceneBank(a.scenes)
    cfg.robot.n_robots = bank.robot_bounds((lo, hi), "train")[0]
    if t_spec > 0:
        cfg.t_max_s = float(t_spec)
    vec_t_max = None if t_spec > 0 else float(AUTO)
    rb = bank.robot_bounds((lo, hi), "train")
    tb = bank.t_max_bounds(cfg.t_max_s if vec_t_max is None else float(AUTO), "train")
    envs = ints(a.envs)
    workers = ints(a.workers) if a.workers else sorted({4, 8, default_workers(max(envs))})
    print(f"[bench_train] device={dev} cores={os.cpu_count()} scenes={a.scenes} "
          f"({len(bank)} files, buckets {bank.bucket_counts('train')}) robots={a.robots}"
          f"{rb} t_max={a.t_max or cfg.t_max_s}{tuple(round(v) for v in tb)} "
          f"rollout={a.rollout} reps={a.reps} bev={a.use_bev}")
    print(f"[bench_train] variant={a.variant or '-'} comms={cfg.comms.mode}"
          + (f" range={'random' if cfg.comms.randomize_range else cfg.comms.range_m}"
             if cfg.comms.mode == "range" else "")
          + f" local={cfg.tokens.local_size} robot_bev={cfg.tokens.robot_bev_size}")
    print(f"{'backend':<10}{'envs':>6}{'workers':>9}{'rollout dec/s':>16}"
          f"{'roll+update dec/s':>20}{'ms/decision':>13}{'setup s':>9}"
          f"{'worker RSS MB':>15}{'max/worker':>12}")
    print("-" * 111)

    for e in envs:
        rows = []
        if not a.no_serial:
            rows.append(("serial", 1, SerialVecEnv))
        for w in workers:
            if w <= e:
                rows.append(("subproc", w, SubprocVecEnv))
        for name, w, klass in rows:
            t0 = time.perf_counter()
            vec = klass(a.scenes, cfg, e, (lo, hi), seed=a.seed, n_workers=w,
                        send_bev=a.use_bev, num_threads=1, t_max=vec_t_max, scene_mix=mix)
            setup = time.perf_counter() - t0
            try:
                roll, both = bench(vec, dev, a.rollout, a.reps, a.use_bev, True, a.d_model)
                tot_rss, max_rss = worker_rss(vec)
            finally:
                vec.close()
            print(f"{name:<10}{e:>6}{w:>9}{roll:>16.1f}{both:>20.1f}{1000 / roll:>13.2f}"
                  f"{setup:>9.2f}{tot_rss:>15.0f}{max_rss:>12.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
