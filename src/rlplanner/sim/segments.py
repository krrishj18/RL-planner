"""Generic over-segmentation of the observed feature map (CONTRACTS.md 4).

The belief stores one feature vector per observed cell. A *segment* is a spatially connected group
of cells whose features agree — nothing more. There is no class, no query and no ranking involved:
the segmentation runs on the raw embedding image with a single scale parameter, exactly so that the
open-set content of the map survives into the observation and the policy decides what matters.

Felzenszwalb-Huttenlocher graph segmentation over the 4-neighbour grid of observed cells:
edge weight = `1 - cos(feat_a, feat_b)` in [0, 2]; two components merge while the edge is no worse
than `min_c(Int(c) + k / |c|)`; components below `min_cells` are then absorbed by the neighbour
across their cheapest edge. `k` (`rayfronts.segment_scale`) is in cells and reads as the segment
size the scale prefers: while a component is smaller than `k` its threshold exceeds the largest
possible weight, so it merges freely; past that, only genuinely similar neighbours join.

Edges are bucket-sorted (`N_BUCKETS` over the [0, 2] weight range) rather than comparison-sorted,
which makes the pass linear in the number of observed cells and keeps it deterministic (ties resolve
in row-major edge order).

Two consequences of `k = 40` worth knowing, both measured (CONTRACTS.md 12):
- a singleton tolerates any edge (`k / 1` = 40 > the maximum weight 2), so a *single* cell — a body
  at 2 m cells — is always absorbed by its neighbours. Segment tokens describe class-scale
  structure; a far casualty reaches the policy through the ray topic instead.
- how far merging gets is grid-size dependent, because `tau(C) = k / |C|` decays while the noise
  floor does not: a uniform noisy 120x120 belief collapses to one segment, the same belief at
  240x240 stops at ~170 pieces. That is honest over-segmentation, not a failure.
"""
from __future__ import annotations

import numpy as np
from numba import njit

N_BUCKETS = 2048          # weight resolution of the counting sort: 2 / 2048 ~ 1e-3
MAX_W = 2.0


@njit(cache=True, inline="always")
def _find(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:                      # path compression
        nxt = parent[x]
        parent[x] = r
        x = nxt
    return r


@njit(cache=True, fastmath=True)
def segment_grid(feat_sum, observed, k, min_cells, labels):
    """Segment the observed cells of `feat_sum [ny, nx, D]`; writes `labels [ny, nx]` (-1 = none).

    Returns the number of segments. Labels are numbered in row-major order of first appearance,
    so the same belief always yields the same labelling.
    """
    ny, nx, d_n = feat_sum.shape
    for i in range(ny):
        for j in range(nx):
            labels[i, j] = -1
    n = 0
    for i in range(ny):
        for j in range(nx):
            if observed[i, j]:
                n += 1
    if n == 0:
        return 0
    idx = np.full((ny, nx), -1, np.int32)
    norm = np.empty(n, np.float32)             # feat_sum rows are already contiguous over D
    c = 0
    for i in range(ny):
        for j in range(nx):
            if not observed[i, j]:
                continue
            idx[i, j] = c
            s = 0.0
            for t in range(d_n):
                v = feat_sum[i, j, t]
                s += v * v
            norm[c] = np.sqrt(s) if s > 1e-24 else 1e-12
            c += 1

    # -- edges (right and down neighbours of every observed cell) -------------------------------
    eu = np.empty(2 * n, np.int32)
    ev = np.empty(2 * n, np.int32)
    ew = np.empty(2 * n, np.float32)
    eb = np.empty(2 * n, np.int32)
    m = 0
    scale = (N_BUCKETS - 1) / MAX_W
    for i in range(ny):
        for j in range(nx):
            a = idx[i, j]
            if a < 0:
                continue
            for e in range(2):
                i2 = i + e
                j2 = j + (1 - e)
                if i2 >= ny or j2 >= nx:
                    continue
                b = idx[i2, j2]
                if b < 0:
                    continue
                dot = 0.0
                for t in range(d_n):
                    dot += feat_sum[i, j, t] * feat_sum[i2, j2, t]
                w = 1.0 - dot / (norm[a] * norm[b])
                if w < 0.0:
                    w = 0.0
                elif w > MAX_W:
                    w = MAX_W
                eu[m] = a
                ev[m] = b
                ew[m] = w
                eb[m] = int(w * scale)
                m += 1

    # -- counting sort by bucket ---------------------------------------------------------------
    cnt = np.zeros(N_BUCKETS + 1, np.int64)
    for e in range(m):
        cnt[eb[e] + 1] += 1
    for b in range(N_BUCKETS):
        cnt[b + 1] += cnt[b]
    order = np.empty(m, np.int32)
    pos = cnt[:N_BUCKETS].copy()
    for e in range(m):
        b = eb[e]
        order[pos[b]] = e
        pos[b] += 1

    # -- union-find with the Felzenszwalb threshold ---------------------------------------------
    parent = np.arange(n).astype(np.int32)
    size = np.ones(n, np.int32)
    intd = np.zeros(n, np.float32)
    for t in range(m):
        e = order[t]
        a = _find(parent, eu[e])
        b = _find(parent, ev[e])
        if a == b:
            continue
        w = ew[e]
        ta = intd[a] + k / size[a]
        tb = intd[b] + k / size[b]
        if w <= (ta if ta < tb else tb):
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            intd[a] = w
    # small components are absorbed by the neighbour across their cheapest remaining edge
    for t in range(m):
        e = order[t]
        a = _find(parent, eu[e])
        b = _find(parent, ev[e])
        if a == b:
            continue
        if size[a] < min_cells or size[b] < min_cells:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            if intd[b] > intd[a]:
                intd[a] = intd[b]

    # -- relabel in row-major order of first appearance ------------------------------------------
    root_label = np.full(n, -1, np.int32)
    n_seg = 0
    for i in range(ny):
        for j in range(nx):
            a = idx[i, j]
            if a < 0:
                continue
            r = _find(parent, a)
            if root_label[r] < 0:
                root_label[r] = n_seg
                n_seg += 1
            labels[i, j] = root_label[r]
    return n_seg


@njit(cache=True, fastmath=True)
def segment_stats(feat_sum, vox_cnt, last_seen, labels, n_seg, fsum, count, hits, si, sj, t_last):
    """Per segment: sum of the member cells' *unit* features, cell count, hit sum, centroid sums."""
    ny, nx, d_n = feat_sum.shape
    for i in range(ny):
        for j in range(nx):
            lb = labels[i, j]
            if lb < 0:
                continue
            s = 0.0
            for t in range(d_n):
                v = feat_sum[i, j, t]
                s += v * v
            inv = 1.0 / np.sqrt(s) if s > 1e-24 else 0.0
            for t in range(d_n):
                fsum[lb, t] += feat_sum[i, j, t] * inv
            count[lb] += 1
            hits[lb] += vox_cnt[i, j]
            si[lb] += i
            sj[lb] += j
            if last_seen[i, j] > t_last[lb]:
                t_last[lb] = last_seen[i, j]
    return


@njit(cache=True, fastmath=True)
def segment_medoids(labels, n_seg, ci, cj, mi, mj, best):
    """Member cell nearest each segment's centroid (`ci`, `cj` in cell units)."""
    ny, nx = labels.shape
    for i in range(ny):
        for j in range(nx):
            lb = labels[i, j]
            if lb < 0:
                continue
            di = i - ci[lb]
            dj = j - cj[lb]
            d = di * di + dj * dj
            if d < best[lb]:
                best[lb] = d
                mi[lb] = i
                mj[lb] = j
    return


@njit(cache=True, fastmath=True)
def segment_ray_counts(labels, org, az, r0, r1, step, x0, y0, cell_m, n_seg, stamp, out):
    """Live rays whose corridor (depth limit -> visual range) crosses each segment.

    Pure geometric bookkeeping: the corridor is marched in cell-sized steps and each segment it
    passes through is credited once per ray. It says how much a segment has been *looked at from a
    distance*, not what those looks reported.
    """
    ny, nx = labels.shape
    for k in range(org.shape[0]):
        ca = np.cos(az[k])
        sa = np.sin(az[k])
        r = r0
        while r <= r1:
            j = int((org[k, 0] + ca * r - x0) / cell_m)
            i = int((org[k, 1] + sa * r - y0) / cell_m)
            r += step
            if i < 0 or i >= ny or j < 0 or j >= nx:
                continue
            lb = labels[i, j]
            if lb < 0 or stamp[lb] == k + 1:
                continue
            stamp[lb] = k + 1
            out[lb] += 1
    return


def segment_map(feat_sum: np.ndarray, observed: np.ndarray, scale: float, min_cells: int,
                labels: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Convenience wrapper around `segment_grid` that allocates the label grid."""
    if feat_sum.ndim != 3:
        raise ValueError(f"segment_map: feat_sum must be [ny, nx, D], got {feat_sum.shape}")
    if observed.shape != feat_sum.shape[:2]:
        raise ValueError(f"segment_map: observed {observed.shape} != {feat_sum.shape[:2]}")
    if labels is None:
        labels = np.empty(feat_sum.shape[:2], np.int32)
    n = segment_grid(np.ascontiguousarray(feat_sum, np.float32),
                     np.ascontiguousarray(observed, np.bool_), float(scale), int(min_cells), labels)
    return labels, int(n)


__all__ = ["segment_grid", "segment_map", "segment_stats", "segment_medoids", "segment_ray_counts",
           "N_BUCKETS"]
