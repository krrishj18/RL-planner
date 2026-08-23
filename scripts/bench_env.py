"""Throughput benchmark: decisions/s and sub-steps/s for the reference bench scene.

Default = CONTRACTS.md 11: 3 robots, synthetic 240x240 m at 2 m cells, BT policy, single process,
timed after numba warm-up.
"""
from __future__ import annotations

import argparse
import time


from rlplanner.scene.schema import Scene, make_synthetic_scene
from rlplanner.sim.baselines import make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv


def run(env: DisasterEnv, policy, n_decisions: int, prof: dict | None = None) -> tuple[int, float]:
    obs = env.state.last_obs
    t0 = time.perf_counter()
    done_count = 0
    for _ in range(n_decisions):
        obs, _, done, _ = env.step(policy.act(obs, env.state))
        if done:
            obs = env.reset()
            done_count += 1
    return done_count, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="scene json (default: synthetic 240x240)")
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--policy", default="ray_follower")
    ap.add_argument("--decisions", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comms", default="full", choices=("full", "range"),
                    help="range = per-robot beliefs + gossip (CONTRACTS.md 5.1)")
    ap.add_argument("--comms-range", type=float, default=200.0,
                    help="link range in metres when --comms range")
    ap.add_argument("--robot-bev", type=int, default=0, help="per-robot actor BEV size (0 = off)")
    ap.add_argument("--local", type=int, default=None, help="local crop size (default: config)")
    a = ap.parse_args()

    scene = Scene.from_json(a.scene) if a.scene else make_synthetic_scene(a.seed)
    cfg = EnvConfig()
    cfg.robot.n_robots = a.robots
    cfg.raster.cell_m = a.cell
    cfg.record_events = True
    cfg.comms.mode = a.comms
    cfg.comms.range_m = a.comms_range
    cfg.comms.randomize_range = False
    cfg.tokens.robot_bev_size = int(a.robot_bev)
    if a.local is not None:
        cfg.tokens.local_size = int(a.local)
    env = DisasterEnv(scene, cfg, seed=a.seed)
    pol = make_policy(a.policy, queries=cfg.rayfronts.queries, seed=a.seed)

    run(env, pol, a.warmup)                       # jit + cache warm-up
    env.reset(a.seed)
    prof: dict[str, float] = {}
    env.prof = prof
    env.rf.prof = prof
    resets, el = run(env, pol, a.decisions)
    env.prof = None

    sub = a.decisions * cfg.substeps_per_decision
    print(f"scene      {scene.meta.preset}/{scene.meta.seed}  region {scene.region}  "
          f"cells {env.raster.ny}x{env.raster.nx} @ {cfg.raster.cell_m} m")
    print(f"comms      {a.comms}"
          + (f" @ {a.comms_range:.0f} m" if a.comms == "range" else "")
          + f"   local {cfg.tokens.local_size}   robot_bev {cfg.tokens.robot_bev_size}")
    print(f"policy     {a.policy}   robots {a.robots}   decisions {a.decisions} "
          f"({cfg.substeps_per_decision} sub-steps each)   episode resets {resets}")
    print(f"decisions/s {a.decisions / el:9.1f}      sub-steps/s {sub / el:9.1f}"
          f"      ms/decision {el / a.decisions * 1000:6.3f}")
    print(f"target      {'PASS' if a.decisions / el >= 200 else 'FAIL'} (>= 200 decisions/s, "
          f">= 1000 sub-steps/s)")
    print("\nbreakdown (share of wall time)")
    groups = {"sensor": ("sensor",), "rayfronts": ("humans", "voxels", "rays", "ray_resolve",
                                                   "frontiers", "ray_targets", "segments"),
              "tokens": ("tokens",), "motion": ("motion",), "comms": ("comms",)}
    acc = 0.0
    for g, keys in groups.items():
        v = sum(prof.get(k, 0.0) for k in keys)
        acc += v
        detail = "  ".join(f"{k}={prof.get(k, 0.0) / el * 100:4.1f}%" for k in keys if k in prof) \
            if len(keys) > 1 else ""
        print(f"  {g:<10} {v / el * 100:5.1f}%   {detail}")
    print(f"  {'other':<10} {(el - acc) / el * 100:5.1f}%   (env bookkeeping, metrics, python)")
    seg_bench(scene, cfg, a.seed)


def seg_bench(scene, cfg, seed: int) -> None:
    """One full segmentation pass on a well-explored belief: the cost the refresh rule amortises."""
    from rlplanner.sim.segments import segment_grid
    env = DisasterEnv(scene, cfg, seed=seed)
    pol = make_policy("nearest_frontier", queries=cfg.rayfronts.queries, seed=seed)
    obs = env.state.last_obs
    for _ in range(400):
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done or env.coverage() > 0.95:
            break
    rf = env.rf
    lab = rf.seg_labels.copy()
    segment_grid(rf.vox_feat_sum, rf.observed, float(cfg.rayfronts.segment_scale),
                 int(cfg.rayfronts.segment_min_cells), lab)          # warm
    t1 = time.perf_counter()
    for _ in range(5):
        segment_grid(rf.vox_feat_sum, rf.observed, float(cfg.rayfronts.segment_scale),
                     int(cfg.rayfronts.segment_min_cells), lab)
    ms = (time.perf_counter() - t1) / 5 * 1000.0
    print(f"\nsegmentation  {ms:6.2f} ms per full pass at coverage {env.coverage():.2f} "
          f"({int(rf.observed.sum())} of {env.raster.ny * env.raster.nx} cells) -> "
          f"{int(lab.max()) + 1} segments; re-run only every "
          f"{cfg.rayfronts.segment_refresh_frac:.0%} of new cells or "
          f"{cfg.rayfronts.segment_refresh_decisions} decisions")


if __name__ == "__main__":
    main()
