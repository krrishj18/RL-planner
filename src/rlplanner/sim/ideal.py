"""The motion-only oracle bound: perfect knowledge, straight-line flight, arrival = visit.

Not a policy — a geometric calculator over the true casualty positions and the episode's
spawns/horizon. Constraints: flight speed and the clock, nothing else (no tokens, no sensing,
no map). Its finds curve normalizes progress; its path length normalizes PPL.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .state import CASUALTY_ROLE_ID

IDEAL_NAME = "oracle_ideal"


def _tour_len(pts: np.ndarray) -> float:
    return float(np.hypot(*(np.diff(pts, axis=0).T)).sum()) if len(pts) > 1 else 0.0


def _two_opt(order: list[int], start: np.ndarray, xy: np.ndarray, iters: int = 4) -> list[int]:
    """Open-path 2-opt with a fixed start; segment reversals that shorten the tour."""
    best = list(order)
    for _ in range(iters):
        improved = False
        pts = np.vstack([start[None, :], xy[best]])
        for i in range(1, len(best)):
            for j in range(i + 1, len(best) + 1):
                a, b = pts[i - 1], pts[i]
                c = pts[j - 1]
                d = pts[j] if j < len(pts) - 0 and j < len(pts) else None
                before = math.hypot(*(b - a)) + (math.hypot(*(pts[j] - c)) if j < len(pts) else 0.0)
                after = math.hypot(*(c - a)) + (math.hypot(*(pts[j] - b)) if j < len(pts) else 0.0)
                if after + 1e-9 < before:
                    best[i - 1:j] = best[i - 1:j][::-1]
                    pts = np.vstack([start[None, :], xy[best]])
                    improved = True
        if not improved:
            break
    return best


def ideal_routes(spawns: np.ndarray, targets: np.ndarray, speed: float
                 ) -> tuple[np.ndarray, list[list[int]]]:
    """Greedy earliest-arrival assignment + per-robot 2-opt. -> (arrival_times[n_t], tours)."""
    n_r, n_t = len(spawns), len(targets)
    if n_t == 0:
        return np.zeros(0), [[] for _ in range(n_r)]
    pos = spawns.astype(np.float64).copy()
    t_r = np.zeros(n_r)
    tours: list[list[int]] = [[] for _ in range(n_r)]
    left = set(range(n_t))
    while left:
        best = None
        for r in range(n_r):
            d = np.hypot(targets[list(left), 0] - pos[r, 0], targets[list(left), 1] - pos[r, 1])
            k = int(np.argmin(d))
            c = list(left)[k]
            arr = t_r[r] + d[k] / speed
            if best is None or arr < best[0]:
                best = (arr, r, c)
        arr, r, c = best
        t_r[r] = arr
        pos[r] = targets[c]
        tours[r].append(c)
        left.discard(c)
    arrivals = np.zeros(n_t)
    for r in range(n_r):
        if tours[r]:
            tours[r] = _two_opt(tours[r], spawns[r].astype(np.float64), targets)
            t = 0.0
            p = spawns[r].astype(np.float64)
            for c in tours[r]:
                t += math.hypot(*(targets[c] - p)) / speed
                arrivals[c] = t
                p = targets[c]
    return arrivals, tours


def ideal_bound(env) -> dict[str, Any]:
    """EVAL_COLS-shaped row for one episode's scene/spawns/horizon."""
    hs = env.raster.humans
    cas = hs["role_id"] == CASUALTY_ROLE_ID
    xy = np.stack([hs["x"][cas], hs["y"][cas]], 1).astype(np.float64)
    spawns = np.stack([r.pos[:2] for r in env.state.robots], 0).astype(np.float64)
    v = float(env.cfg.robot.speed_mps)
    t_max = float(env.cfg.t_max_s)
    arrivals, tours = ideal_routes(spawns, xy, v)
    n = max(1, len(xy))
    within = np.sort(arrivals[arrivals <= t_max]) if len(arrivals) else np.zeros(0)
    frac = len(within) / n
    auc = float(((t_max - within) / t_max).sum() / n) if len(within) else 0.0
    def _t_at(k):
        return float(within[k - 1]) if len(within) >= k else t_max
    dist = 0.0
    for r, tour in enumerate(tours):
        p = spawns[r]
        t = 0.0
        for c in tour:
            leg = math.hypot(*(xy[c] - p))
            step_t = leg / v
            dist += leg if t + step_t <= t_max else max(0.0, (t_max - t)) * v
            t += step_t
            p = xy[c]
            if t >= t_max:
                break
    return {"frac_found": frac, "finds_auc": auc, "time_to_first": _t_at(1),
            "time_to_half": _t_at((n + 1) // 2), "time_to_all": _t_at(n),
            "coverage_end": float("nan"), "reward": float("nan"),
            "dist_per_find": dist / max(1, len(within)), "dist_total": dist,
            "redundancy": 0.0, "redundancy_frac": 0.0, "intentional_revisits": 0.0,
            "link_frac": 1.0, "length": 0}
