"""Numba geometry kernels: line of sight, frustum test, 8-connected A*.

All grid arguments are (ny, nx) row-major with cell (i, j) centred at
(x0 + (j+0.5)*cell_m, y0 + (i+0.5)*cell_m).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy import ndimage

SQRT2 = math.sqrt(2.0)


@njit(cache=True, inline="always")
def wrap_pi(a: float) -> float:
    return a - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi))


# ---- line of sight ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def los_one(height, cell_m, x0, y0, camx, camy, camz, tx, ty, tz, step_frac, eps):
    """True if nothing between the camera cell and the target cell rises above the ray."""
    ny, nx = height.shape
    dx = tx - camx
    dy = ty - camy
    dz = tz - camz
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d <= 1e-9:
        return True
    ci = int(math.floor((camy - y0) / cell_m))
    cj = int(math.floor((camx - x0) / cell_m))
    ti = int(math.floor((ty - y0) / cell_m))
    tj = int(math.floor((tx - x0) / cell_m))
    step = step_frac * cell_m
    if step <= 1e-9:
        step = cell_m
    n = int(d / step)
    for k in range(1, n + 1):
        s = (k * step) / d
        if s >= 1.0:
            break
        py = camy + dy * s
        px = camx + dx * s
        i = int(math.floor((py - y0) / cell_m))
        j = int(math.floor((px - x0) / cell_m))
        if i < 0 or i >= ny or j < 0 or j >= nx:
            continue
        if (i == ci and j == cj) or (i == ti and j == tj):
            continue
        if height[i, j] > camz + dz * s + eps:
            return False
    return True


@njit(cache=True, parallel=True, fastmath=True)
def los_visible(height, cell_m, origin_xy, cam_xyz, targets_xyz, step_frac, eps):
    """bool[N] line of sight from `cam_xyz` to each row of `targets_xyz` (N, 3)."""
    n = targets_xyz.shape[0]
    out = np.empty(n, np.bool_)
    x0 = origin_xy[0]
    y0 = origin_xy[1]
    cx = cam_xyz[0]
    cy = cam_xyz[1]
    cz = cam_xyz[2]
    for k in prange(n):
        out[k] = los_one(height, cell_m, x0, y0, cx, cy, cz,
                         targets_xyz[k, 0], targets_xyz[k, 1], targets_xyz[k, 2], step_frac, eps)
    return out


# ---- frustum ---------------------------------------------------------------------------------
@njit(cache=True, inline="always", fastmath=True)
def in_frustum_one(dx, dy, dz, r, cos_yaw, sin_yaw, cos_half_h, sin_lo, sin_hi):
    """Azimuth/elevation frustum test with the trig hoisted out (see `frustum_params`)."""
    if r <= 1e-9:
        return True
    s = dz / r
    if s < sin_lo or s > sin_hi:
        return False
    rxy = math.sqrt(dx * dx + dy * dy)
    if rxy <= 1e-9:
        return True          # straight below/above: azimuth is undefined, elevation already passed
    return (dx * cos_yaw + dy * sin_yaw) >= rxy * cos_half_h


def frustum_params(yaw: float, pitch: float, hfov: float, vfov: float):
    """(cos_yaw, sin_yaw, cos(hfov/2), sin(pitch - vfov/2), sin(pitch + vfov/2))."""
    lo = max(-math.pi / 2, pitch - 0.5 * vfov)
    hi = min(math.pi / 2, pitch + 0.5 * vfov)
    return (math.cos(yaw), math.sin(yaw), math.cos(0.5 * hfov), math.sin(lo), math.sin(hi))


@njit(cache=True, fastmath=True)
def _frustum_batch(cam_xyz, targets_xyz, cy, sy, chh, slo, shi):
    n = targets_xyz.shape[0]
    out = np.empty(n, np.bool_)
    for k in range(n):
        dx = targets_xyz[k, 0] - cam_xyz[0]
        dy = targets_xyz[k, 1] - cam_xyz[1]
        dz = targets_xyz[k, 2] - cam_xyz[2]
        out[k] = in_frustum_one(dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz), cy, sy, chh, slo, shi)
    return out


def in_frustum(cam_xyz, yaw, pitch, hfov, vfov, targets_xyz):
    """bool[N]: |az - yaw| <= hfov/2 and el in [pitch - vfov/2, pitch + vfov/2]."""
    p = frustum_params(float(yaw), float(pitch), float(hfov), float(vfov))
    return _frustum_batch(np.ascontiguousarray(cam_xyz, np.float64),
                          np.ascontiguousarray(targets_xyz, np.float64), *p)


# ---- ray corridor coverage --------------------------------------------------------------------
@njit(cache=True, parallel=True)
def corridor_observed_frac(observed, cell_m, x0, y0, origins, az, r_near, r_far):
    """Fraction of sampled cells (step = cell_m) along each ray's bearing in [r_near, r_far]
    that are observed. Samples outside the grid count as observed (nothing left to see)."""
    n = origins.shape[0]
    out = np.zeros(n, np.float32)
    ny, nx = observed.shape
    nstep = int((r_far - r_near) / cell_m) + 1
    if nstep < 1:
        nstep = 1
    for k in prange(n):
        c = math.cos(az[k])
        s = math.sin(az[k])
        seen = 0
        tot = 0
        li = -1
        lj = -1
        for m in range(nstep):
            r = r_near + m * cell_m
            i = int(math.floor((origins[k, 1] + s * r - y0) / cell_m))
            j = int(math.floor((origins[k, 0] + c * r - x0) / cell_m))
            if i == li and j == lj:
                continue
            li = i
            lj = j
            tot += 1
            if i < 0 or i >= ny or j < 0 or j >= nx or observed[i, j]:
                seen += 1
        out[k] = seen / tot if tot > 0 else 1.0
    return out


@njit(cache=True, parallel=True)
def disc_observed_frac(seen, cell_m, x0, y0, centres, radius):
    """Fraction of in-grid cells within `radius` of each centre that are `seen`.

    Cells outside the grid are ignored (a target near the border is judged on what exists)."""
    n = centres.shape[0]
    out = np.zeros(n, np.float32)
    ny, nx = seen.shape
    rad = int(radius / cell_m) + 1
    r2 = radius * radius
    for k in prange(n):
        ci = int(math.floor((centres[k, 1] - y0) / cell_m))
        cj = int(math.floor((centres[k, 0] - x0) / cell_m))
        tot = 0
        hit = 0
        for a in range(-rad, rad + 1):
            i = ci + a
            if i < 0 or i >= ny:
                continue
            for b in range(-rad, rad + 1):
                j = cj + b
                if j < 0 or j >= nx:
                    continue
                if (a * a + b * b) * cell_m * cell_m > r2:
                    continue
                tot += 1
                if seen[i, j]:
                    hit += 1
        out[k] = hit / tot if tot > 0 else 1.0
    return out


# ---- A* ---------------------------------------------------------------------------------------
@njit(cache=True)
def _heap_push(hf, hn, size, f, node):
    i = size
    hf[i] = f
    hn[i] = node
    while i > 0:
        p = (i - 1) >> 1
        if hf[p] <= hf[i]:
            break
        hf[p], hf[i] = hf[i], hf[p]
        hn[p], hn[i] = hn[i], hn[p]
        i = p
    return size + 1


@njit(cache=True)
def _heap_pop(hf, hn, size):
    top_f = hf[0]
    top_n = hn[0]
    size -= 1
    hf[0] = hf[size]
    hn[0] = hn[size]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        m = i
        if l < size and hf[l] < hf[m]:
            m = l
        if r < size and hf[r] < hf[m]:
            m = r
        if m == i:
            break
        hf[m], hf[i] = hf[i], hf[m]
        hn[m], hn[i] = hn[i], hn[m]
        i = m
    return top_f, top_n, size


@njit(cache=True)
def _astar(obst, si, sj, gi, gj, g, came, closed, hf, hn):
    ny, nx = obst.shape
    n_cells = ny * nx
    if si < 0 or si >= ny or sj < 0 or sj >= nx or gi < 0 or gi >= ny or gj < 0 or gj >= nx:
        return np.empty((0, 2), np.int32)
    if obst[si, sj] or obst[gi, gj]:
        return np.empty((0, 2), np.int32)
    for k in range(n_cells):
        g[k] = 1e30
        came[k] = -1
        closed[k] = False
    start = si * nx + sj
    goal = gi * nx + gj
    g[start] = 0.0
    size = _heap_push(hf, hn, 0, 0.0, start)
    found = False
    while size > 0:
        _, cur, size = _heap_pop(hf, hn, size)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == goal:
            found = True
            break
        ci = cur // nx
        cj = cur - ci * nx
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                i = ci + di
                j = cj + dj
                if i < 0 or i >= ny or j < 0 or j >= nx:
                    continue
                if obst[i, j]:
                    continue
                nb = i * nx + j
                if closed[nb]:
                    continue
                step = SQRT2 if (di != 0 and dj != 0) else 1.0
                ng = g[cur] + step
                if ng < g[nb]:
                    g[nb] = ng
                    came[nb] = cur
                    adi = abs(gi - i)
                    adj = abs(gj - j)
                    lo = adi if adi < adj else adj
                    h = (adi + adj) + (SQRT2 - 2.0) * lo
                    size = _heap_push(hf, hn, size, ng + h, nb)
    if not found:
        return np.empty((0, 2), np.int32)
    m = 0
    cur = goal
    while cur != -1:
        m += 1
        cur = came[cur]
    out = np.empty((m, 2), np.int32)
    cur = goal
    k = m - 1
    while cur != -1:
        out[k, 0] = cur // nx
        out[k, 1] = cur - (cur // nx) * nx
        cur = came[cur]
        k -= 1
    return out


class PathPlanner:
    """A* over a static obstacle mask, with a per-instance (start, goal) path cache.

    `reachable` uses 8-connected labelling of the free space, which is exactly the set of
    (start, goal) pairs for which `path` returns a path.
    """

    def __init__(self, obstacle_mask: np.ndarray):
        self.obst = np.ascontiguousarray(obstacle_mask, dtype=np.bool_)
        ny, nx = self.obst.shape
        n = ny * nx
        self._g = np.empty(n, np.float64)
        self._came = np.empty(n, np.int32)
        self._closed = np.empty(n, np.bool_)
        cap = 8 * n + 16
        self._hf = np.empty(cap, np.float64)
        self._hn = np.empty(cap, np.int32)
        self.cache: dict[tuple[int, int, int, int], np.ndarray | None] = {}
        self.labels, self.n_labels = ndimage.label(~self.obst, structure=np.ones((3, 3), bool))

    def label_at(self, i: int, j: int) -> int:
        ny, nx = self.obst.shape
        if not (0 <= i < ny and 0 <= j < nx):
            return 0
        return int(self.labels[i, j])

    def reachable(self, start_ij, goal_ij) -> bool:
        a = self.label_at(int(start_ij[0]), int(start_ij[1]))
        b = self.label_at(int(goal_ij[0]), int(goal_ij[1]))
        return a != 0 and a == b

    def path(self, start_ij, goal_ij) -> np.ndarray | None:
        key = (int(start_ij[0]), int(start_ij[1]), int(goal_ij[0]), int(goal_ij[1]))
        if key in self.cache:
            return self.cache[key]
        if not self.reachable(start_ij, goal_ij):
            self.cache[key] = None
            return None
        p = _astar(self.obst, key[0], key[1], key[2], key[3],
                   self._g, self._came, self._closed, self._hf, self._hn)
        out = None if p.shape[0] == 0 else p
        self.cache[key] = out
        return out


def astar(obstacle_mask: np.ndarray, start_ij, goal_ij) -> np.ndarray | None:
    """8-connected A* (octile heuristic). Returns int32 [m, 2] of (row, col) or None."""
    return PathPlanner(obstacle_mask).path(start_ij, goal_ij)


__all__ = ["los_one", "los_visible", "in_frustum", "in_frustum_one", "frustum_params", "corridor_observed_frac", "disc_observed_frac",
           "astar", "PathPlanner", "wrap_pi"]
