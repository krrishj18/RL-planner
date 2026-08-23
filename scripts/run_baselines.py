"""Run the hand-coded baselines over a set of scenes/seeds and print a metrics table."""
from __future__ import annotations

import argparse
import time

import numpy as np

from rlplanner.scene.schema import Scene, make_synthetic_scene
from rlplanner.sim.baselines import POLICIES, make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv

COLS = ("reward", "frac_found", "finds_auc", "time_to_first", "time_to_half", "time_to_all",
        "coverage_end", "dist_per_find", "redundancy", "redundancy_frac",
        "intentional_revisits", "revisit_penalties")
CONTAINERS = ("open", "vehicle", "building", "rubble")


def episode(env: DisasterEnv, policy, seed: int) -> dict:
    """One episode; the metric dict plus the undiscounted episode return."""
    obs = env.reset(seed)
    policy.reset(seed)
    total = 0.0
    while True:
        obs, r, done, info = env.step(policy.act(obs, env.state))
        total += float(r)
        if done:
            return dict(info["metrics"], reward=total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--scenes", nargs="*", default=None, help="scene json files (default: synthetic)")
    ap.add_argument("--policies", nargs="*", default=list(POLICIES))
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--comms", default="full", choices=("full", "range"))
    ap.add_argument("--comms-range", type=float, default=200.0)
    ap.add_argument("--t-max", type=float, default=None,
                    help="episode horizon in seconds (default: EnvConfig's 600; scale it with the "
                         "region as train/scenes.auto_t_max does for anything bigger)")
    a = ap.parse_args()

    cfg = EnvConfig()
    cfg.robot.n_robots = a.robots
    cfg.raster.cell_m = a.cell
    cfg.comms.mode = a.comms
    cfg.comms.range_m = a.comms_range
    cfg.comms.randomize_range = False
    if a.t_max is not None:
        cfg.t_max_s = float(a.t_max)
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    scenes = [Scene.from_json(p) for p in a.scenes] if a.scenes else \
        [make_synthetic_scene(s) for s in seeds]
    envs = [DisasterEnv(sc, cfg, seed=seeds[0]) for sc in scenes]

    print(f"{len(scenes)} scene(s) x {len(seeds)} seed(s), {a.robots} robots, "
          f"cell {a.cell} m, t_max {cfg.t_max_s} s, comms {a.comms}"
          + (f" @ {a.comms_range:.0f} m" if a.comms == "range" else ""))
    head = f"{'policy':<17}" + "".join(f"{c:>15}" for c in COLS) + f"{'sec/ep':>9}"
    print(head)
    print("-" * len(head))
    per_container = {}
    for name in a.policies:
        pol = make_policy(name, queries=cfg.rayfronts.queries, seed=a.seed0)
        rows, wall = [], 0.0
        for i, seed in enumerate(seeds):
            env = envs[i % len(envs)]
            t0 = time.perf_counter()
            rows.append(episode(env, pol, seed))
            wall += time.perf_counter() - t0
        cells = []
        for c in COLS:
            v = np.array([r[c] for r in rows], float)
            cells.append(f"{v.mean():7.2f}+-{v.std():<6.2f}")
        print(f"{name:<17}" + "".join(f"{c:>15}" for c in cells) + f"{wall / len(seeds):9.2f}")
        per_container[name] = ({k: float(np.mean([r["found_by_container"].get(k, 0) for r in rows]))
                                for k in CONTAINERS},
                               {k: float(np.mean([r["n_by_container"].get(k, 0) for r in rows]))
                                for k in CONTAINERS})

    print("\nfinds per container (mean per episode, found / present)")
    head = f"{'policy':<17}" + "".join(f"{c:>18}" for c in CONTAINERS)
    print(head)
    print("-" * len(head))
    for name, (found, total) in per_container.items():
        cells = [f"{found[c]:5.2f} / {total[c]:<5.2f}" for c in CONTAINERS]
        print(f"{name:<17}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
