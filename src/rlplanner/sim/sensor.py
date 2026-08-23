"""Camera model: which raster cells a robot observes (voxel range) or sees far away (ray range)."""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit, prange

from .geometry import frustum_params, in_frustum_one, los_one
from .raster import Raster

FLAG_NONE, FLAG_NEAR, FLAG_FAR = 0, 1, 2


class View(NamedTuple):
    observed_ij: np.ndarray   # int32 [M, 2] cells within depth_limit with LoS
    far_ij: np.ndarray        # int32 [K, 2] cells in (depth_limit, visual_range] with LoS
    slant_r: np.ndarray       # float64 [K] slant range of the far cells


@njit(cache=True, inline="always")
def _u01(seed, k):
    """splitmix64 -> uniform [0, 1); deterministic given the seed drawn from the caller's rng."""
    z = (seed + np.uint64(0x9E3779B97F4A7C15) * (np.uint64(k) + np.uint64(1))) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, parallel=True, fastmath=True)
def _view_kernel(height, cell_m, x0, y0, i_lo, i_hi, j_lo, j_hi,
                 camx, camy, camz, cy, sy, chh, slo, shi, depth, vis_r,
                 step_frac, eps, use_frustum, far_p, seed):
    W = j_hi - j_lo
    n = (i_hi - i_lo) * W
    flags = np.zeros(n, np.uint8)
    rr = np.empty(n, np.float64)
    for idx in prange(n):
        i = i_lo + idx // W
        j = j_lo + idx % W
        tx = x0 + (j + 0.5) * cell_m
        ty = y0 + (i + 0.5) * cell_m
        tz = height[i, j]
        dx = tx - camx
        dy = ty - camy
        dz = tz - camz
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        if r > vis_r:
            continue
        if use_frustum and not in_frustum_one(dx, dy, dz, r, cy, sy, chh, slo, shi):
            continue
        if r > depth and far_p < 1.0 and _u01(seed, idx) >= far_p:
            continue
        if not los_one(height, cell_m, x0, y0, camx, camy, camz, tx, ty, tz, step_frac, eps):
            continue
        rr[idx] = r
        flags[idx] = FLAG_NEAR if r <= depth else FLAG_FAR
    n_near = 0
    n_far = 0
    for idx in range(n):
        if flags[idx] == FLAG_NEAR:
            n_near += 1
        elif flags[idx] == FLAG_FAR:
            n_far += 1
    obs_ij = np.empty((n_near, 2), np.int32)
    far_ij = np.empty((n_far, 2), np.int32)
    far_r = np.empty(n_far, np.float64)
    a = 0
    b = 0
    for idx in range(n):
        f = flags[idx]
        if f == FLAG_NONE:
            continue
        i = i_lo + idx // W
        j = j_lo + idx % W
        if f == FLAG_NEAR:
            obs_ij[a, 0] = i
            obs_ij[a, 1] = j
            a += 1
        else:
            far_ij[b, 0] = i
            far_ij[b, 1] = j
            far_r[b] = rr[idx]
            b += 1
    return obs_ij, far_ij, far_r


def visible_cells(raster: Raster, sensor, cam_xyz, yaw: float,
                  far_p: float = 1.0, rng: np.random.Generator | None = None) -> View:
    """Cells in view from `cam_xyz` with heading `yaw` (radians).

    `far_p` < 1 keeps each far candidate with that probability *before* the (expensive) LoS test.
    Bernoulli thinning commutes with the LoS filter, so the result is distributed exactly like
    thinning the full far set; it only saves work when the caller would thin anyway.
    """
    if sensor.depth_limit_m > sensor.visual_range_m:
        raise ValueError("sensor.depth_limit_m must be <= sensor.visual_range_m")
    cam = np.asarray(cam_xyz, dtype=np.float64)
    x0, y0 = raster.origin
    cell = raster.cell_m
    vis = float(sensor.visual_range_m)
    ci = int(math.floor((cam[1] - y0) / cell))
    cj = int(math.floor((cam[0] - x0) / cell))
    rad = int(math.ceil(vis / cell)) + 1
    i_lo, i_hi = max(0, ci - rad), min(raster.ny, ci + rad + 1)
    j_lo, j_hi = max(0, cj - rad), min(raster.nx, cj + rad + 1)
    if i_hi <= i_lo or j_hi <= j_lo:
        z = np.zeros((0, 2), np.int32)
        return View(z, z, np.zeros(0, np.float64))
    if far_p < 1.0 and rng is None:
        raise ValueError("visible_cells: far_p < 1 requires an rng")
    seed = np.uint64(rng.integers(1 << 62)) if far_p < 1.0 else np.uint64(0)
    fp = frustum_params(float(yaw), math.radians(sensor.pitch_deg),
                        math.radians(sensor.hfov_deg), math.radians(sensor.vfov_deg))
    obs_ij, far_ij, far_r = _view_kernel(
        raster.height, cell, x0, y0, i_lo, i_hi, j_lo, j_hi, cam[0], cam[1], cam[2],
        fp[0], fp[1], fp[2], fp[3], fp[4], float(sensor.depth_limit_m), vis,
        float(sensor.los_step_frac), float(sensor.los_eps_m), sensor.mode != "disk",
        float(far_p), seed)
    return View(obs_ij, far_ij, far_r)


@njit(cache=True, parallel=True, fastmath=True)
def _observable_kernel(height, cell_m, x0, y0, reach, ci, cj, alt, depth, slo, shi,
                       step_frac, eps, rad):
    """For each candidate cell: is its top point in view from *some* reachable free cell?"""
    n = ci.shape[0]
    out = np.zeros(n, np.bool_)
    ny, nx = height.shape
    for k in prange(n):
        i = ci[k]
        j = cj[k]
        tx = x0 + (j + 0.5) * cell_m
        ty = y0 + (i + 0.5) * cell_m
        tz = height[i, j]
        if tz - alt > 0.0:
            continue                      # above the camera: never inside a downward wedge
        hit = False
        for a in range(-rad, rad + 1):
            ii = i + a
            if ii < 0 or ii >= ny:
                continue
            for b in range(-rad, rad + 1):
                jj = j + b
                if jj < 0 or jj >= nx or not reach[ii, jj]:
                    continue
                cx = x0 + (jj + 0.5) * cell_m
                cy = y0 + (ii + 0.5) * cell_m
                dx = tx - cx
                dy = ty - cy
                dz = tz - alt
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                if r > depth:
                    continue
                if r > 1e-9:
                    sn = dz / r
                    if sn < slo or sn > shi:
                        continue
                if los_one(height, cell_m, x0, y0, cx, cy, alt, tx, ty, tz, step_frac, eps):
                    hit = True
                    break
            if hit:
                break
        out[k] = hit
    return out


def observable_mask(raster: Raster, sensor, reachable: np.ndarray, alt: float) -> np.ndarray:
    """Cells whose top point can ever be voxel-observed by a robot flying at `alt`.

    A reachable free cell is observable by hovering over it (the wedge reaches nadir). Every other
    cell — building roofs above the obstacle threshold, sealed-off free space — is tested exactly
    against the reachable set. Roofs above `alt` are excluded outright: their top point can only be
    seen at a positive elevation, which the downward wedge never contains.
    """
    reach = np.ascontiguousarray(reachable, np.bool_)
    out = reach.copy()
    ci, cj = np.nonzero(~reach)
    if ci.size == 0:
        return out
    lo = max(-math.pi / 2, math.radians(sensor.pitch_deg) - 0.5 * math.radians(sensor.vfov_deg))
    hi = min(math.pi / 2, math.radians(sensor.pitch_deg) + 0.5 * math.radians(sensor.vfov_deg))
    rad = int(math.ceil(float(sensor.depth_limit_m) / raster.cell_m))
    hits = _observable_kernel(raster.height, raster.cell_m, raster.origin[0], raster.origin[1],
                              reach, ci.astype(np.int64), cj.astype(np.int64), float(alt),
                              float(sensor.depth_limit_m), math.sin(lo), math.sin(hi),
                              float(sensor.los_step_frac), float(sensor.los_eps_m), rad)
    out[ci[hits], cj[hits]] = True
    return out


@njit(cache=True, parallel=True, fastmath=True)
def _los_batch(height, cell_m, x0, y0, cam, targets, step_frac, eps):
    n = targets.shape[0]
    out = np.empty(n, np.bool_)
    for k in prange(n):
        out[k] = los_one(height, cell_m, x0, y0, cam[0], cam[1], cam[2],
                         targets[k, 0], targets[k, 1], targets[k, 2], step_frac, eps)
    return out


def point_visibility(raster: Raster, sensor, cam_xyz, yaw: float, pts_xyz: np.ndarray):
    """(in_view[N], slant_r[N]) for arbitrary world points (used for hidden humans)."""
    cam = np.asarray(cam_xyz, dtype=np.float64)
    pts = np.ascontiguousarray(pts_xyz, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros(0, np.bool_), np.zeros(0, np.float64)
    d = pts - cam[None, :]
    r = np.sqrt((d * d).sum(axis=1))
    ok = r <= float(sensor.visual_range_m)
    if sensor.mode != "disk":
        ok &= _frustum_np(d, r, frustum_params(float(yaw), math.radians(sensor.pitch_deg),
                                               math.radians(sensor.hfov_deg),
                                               math.radians(sensor.vfov_deg)))
    idx = np.flatnonzero(ok)
    if idx.size:
        vis = _los_batch(raster.height, raster.cell_m, raster.origin[0], raster.origin[1],
                         cam, pts[idx], float(sensor.los_step_frac), float(sensor.los_eps_m))
        ok[idx] = vis
    return ok, r


def _frustum_np(d, r, fp):
    cy, sy, chh, slo, shi = fp
    rs = np.maximum(r, 1e-9)
    s = d[:, 2] / rs
    rxy = np.hypot(d[:, 0], d[:, 1])
    return (s >= slo) & (s <= shi) & ((d[:, 0] * cy + d[:, 1] * sy) >= rxy * chh)




@njit(cache=True, fastmath=True)
def _human_kernel(height, cell_m, x0, y0, camx, camy, camz, pts, cy, sy, chh, slo, shi,
                  use_frustum, vis_r, step_frac, eps):
    n = pts.shape[0]
    r_out = np.empty(n, np.float64)
    in_view = np.zeros(n, np.bool_)
    for k in range(n):
        dx = pts[k, 0] - camx
        dy = pts[k, 1] - camy
        dz = pts[k, 2] - camz
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        r_out[k] = r
        if r > vis_r:
            continue
        if use_frustum and not in_frustum_one(dx, dy, dz, r, cy, sy, chh, slo, shi):
            continue
        if los_one(height, cell_m, x0, y0, camx, camy, camz,
                   pts[k, 0], pts[k, 1], pts[k, 2], step_frac, eps):
            in_view[k] = True
    return in_view, r_out


def human_visibility(raster: Raster, sensor, cam_xyz, yaw: float, pts_xyz: np.ndarray):
    """(in_view, slant_r) for the hidden humans, one LoS test each."""
    cam = np.asarray(cam_xyz, np.float64)
    fp = frustum_params(float(yaw), math.radians(sensor.pitch_deg),
                        math.radians(sensor.hfov_deg), math.radians(sensor.vfov_deg))
    return _human_kernel(raster.height, raster.cell_m, raster.origin[0], raster.origin[1],
                         cam[0], cam[1], cam[2], pts_xyz, fp[0], fp[1], fp[2], fp[3], fp[4],
                         sensor.mode != "disk", float(sensor.visual_range_m),
                         float(sensor.los_step_frac), float(sensor.los_eps_m))


__all__ = ["View", "visible_cells", "point_visibility", "human_visibility",
           "observable_mask"]
