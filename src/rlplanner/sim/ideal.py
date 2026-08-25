"""The motion-only oracle bound: perfect knowledge, obstacle-aware flight, arrival = visit.

Not a policy — a geometric calculator over the true casualty positions and the episode's
spawns/horizon. Constraints: flight speed, the episode clock and the same obstacle mask the
env's own router uses (cells taller than the flight altitude); nothing else (no tokens, no
sensing, no map). Its finds curve normalizes progress; its path length normalizes PPL.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .state import CASUALTY_ROLE_ID

IDEAL_NAME = "oracle_ideal"
_SNAP_R = 12          # cells searched around a blocked point-of-interest for a free stand-in


def euclidean_matrix(spawns: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pts = np.vstack([spawns, targets]).astype(np.float64)
    d = pts[:, None, :] - pts[None, :, :]
    return np.hypot(d[..., 0], d[..., 1])


def _snap_free(obst: np.ndarray, i: int, j: int) -> tuple[int, int] | None:
    ny, nx = obst.shape
    i, j = int(np.clip(i, 0, ny - 1)), int(np.clip(j, 0, nx - 1))
    if not obst[i, j]:
        return (i, j)
    for r in range(1, _SNAP_R + 1):
        i0, i1 = max(0, i - r), min(ny, i + r + 1)
        j0, j1 = max(0, j - r), min(nx, j + r + 1)
        sub = ~obst[i0:i1, j0:j1]
        if sub.any():
            ii, jj = np.nonzero(sub)
            k = int(np.argmin((ii + i0 - i) ** 2 + (jj + j0 - j) ** 2))
            return (int(ii[k] + i0), int(jj[k] + j0))
    return None


def obstacle_matrix(obst: np.ndarray, cell_m: float, ij: np.ndarray) -> np.ndarray:
    """Pairwise shortest-path metres between the given cells over the free space
    (8-connected, octile costs) — the same connectivity the env's A* router uses."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    ny, nx = obst.shape
    free = ~obst
    idx = np.full(obst.shape, -1, np.int64)
    idx[free] = np.arange(int(free.sum()))
    rows, cols, ws = [], [], []
    for di, dj, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
        a = free[max(0, -di):ny - max(0, di), max(0, -dj):nx - max(0, dj)]
        b = free[max(0, di):ny - max(0, -di), max(0, dj):nx - max(0, -dj)]
        m = a & b
        src = idx[max(0, -di):ny - max(0, di), max(0, -dj):nx - max(0, dj)][m]
        dst = idx[max(0, di):ny - max(0, -di), max(0, dj):nx - max(0, -dj)][m]
        rows.append(src); cols.append(dst); ws.append(np.full(len(src), w * cell_m))
    g = coo_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(int(free.sum()),) * 2)
    nodes = []
    for i, j in ij:
        s = _snap_free(obst, int(i), int(j))
        nodes.append(-1 if s is None else int(idx[s]))
    valid = [k for k, n in enumerate(nodes) if n >= 0]
    D = np.full((len(ij), len(ij)), np.inf)
    if valid:
        dm = dijkstra(g.tocsr(), directed=False, indices=[nodes[k] for k in valid])
        for a, ka in enumerate(valid):
            for b, kb in enumerate(valid):
                D[ka, kb] = dm[a, nodes[kb]]
    np.fill_diagonal(D, 0.0)
    return D


def ideal_routes(D: np.ndarray, n_r: int, speed: float) -> tuple[np.ndarray, list[list[int]]]:
    """Greedy earliest-arrival + per-robot 2-opt over a distance matrix whose first `n_r`
    rows/cols are the spawns and the rest the targets. -> (arrival_times[n_t], tours)."""
    n_t = D.shape[0] - n_r
    if n_t == 0:
        return np.zeros(0), [[] for _ in range(n_r)]
    node = list(range(n_r))                      # current matrix node per robot
    t_r = np.zeros(n_r)
    tours: list[list[int]] = [[] for _ in range(n_r)]
    left = set(range(n_t))
    while left:
        best = None
        for r in range(n_r):
            for c in left:
                d = D[node[r], n_r + c]
                if not np.isfinite(d):
                    continue
                arr = t_r[r] + d / speed
                if best is None or arr < best[0]:
                    best = (arr, r, c)
        if best is None:
            break                                # the rest is unreachable
        arr, r, c = best
        t_r[r] = arr
        node[r] = n_r + c
        tours[r].append(c)
        left.discard(c)
    arrivals = np.full(n_t, np.inf)
    for r in range(n_r):
        tours[r] = _two_opt(tours[r], r, D, n_r)
        t = 0.0
        p = r
        for c in tours[r]:
            t += D[p, n_r + c] / speed
            arrivals[c] = t
            p = n_r + c
    return arrivals, tours


def _two_opt(order: list[int], start_node: int, D: np.ndarray, n_r: int,
             iters: int = 4) -> list[int]:
    best = list(order)
    if len(best) < 3:
        return best
    for _ in range(iters):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                seq = [start_node] + [n_r + c for c in best]
                a, b = seq[i], seq[i + 1]
                c1 = seq[j]
                after_j = seq[j + 1] if j + 1 < len(seq) else None
                before = D[a, b] + (D[c1, after_j] if after_j is not None else 0.0)
                after = D[a, c1] + (D[b, after_j] if after_j is not None else 0.0)
                if after + 1e-9 < before:
                    best[i:j] = best[i:j][::-1]
                    improved = True
        if not improved:
            break
    return best


def ideal_bound(env) -> dict[str, Any]:
    """EVAL_COLS-shaped row for one episode's scene/spawns/horizon, obstacle-aware."""
    hs = env.raster.humans
    cas = hs["role_id"] == CASUALTY_ROLE_ID
    xy = np.stack([hs["x"][cas], hs["y"][cas]], 1).astype(np.float64)
    spawns = np.stack([r.pos[:2] for r in env.state.robots], 0).astype(np.float64)
    n_r = len(spawns)
    v = float(env.cfg.robot.speed_mps)
    t_max = float(env.cfg.t_max_s)
    pts = np.vstack([spawns, xy])
    ij = np.stack(env.raster.xy_to_ij(pts[:, 0], pts[:, 1]), 1)
    D = obstacle_matrix(env.planner.obst, float(env.raster.cell_m), ij)
    arrivals, tours = ideal_routes(D, n_r, v)
    n = max(1, len(xy))
    within = np.sort(arrivals[np.isfinite(arrivals) & (arrivals <= t_max)])
    frac = len(within) / n
    auc = float(((t_max - within) / t_max).sum() / n) if len(within) else 0.0
    def _t_at(k):
        return float(within[k - 1]) if len(within) >= k else t_max
    dist = 0.0
    for r, tour in enumerate(tours):
        t = 0.0
        p = r
        for c in tour:
            leg = D[p, n_r + c]
            step_t = leg / v
            dist += leg if t + step_t <= t_max else max(0.0, (t_max - t)) * v
            t += step_t
            p = n_r + c
            if t >= t_max:
                break
    return {"frac_found": frac, "finds_auc": auc, "time_to_first": _t_at(1),
            "time_to_half": _t_at((n + 1) // 2), "time_to_all": _t_at(n),
            "coverage_end": float("nan"), "reward": float("nan"),
            "dist_per_find": dist / max(1, len(within)), "dist_total": dist,
            "redundancy": 0.0, "redundancy_frac": 0.0, "intentional_revisits": 0.0,
            "link_frac": 1.0, "length": 0}
