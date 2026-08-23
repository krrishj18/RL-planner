"""Emulation of what RayFronts publishes: a persistent semantic voxel map, semantic rays, frontiers.

Owns the whole belief state. `update()` runs one sub-step of sensing for the team;
`end_of_decision()` refreshes the derived products (ray resolution, frontiers, segments) once per
decision.

**Open-set principle.** What RayFronts stores per voxel and per ray is a *feature embedding*, not a
score against a list of words. Nothing here scans the map against the mission queries: there is no
`vox_sim` grid, no per-ray similarity column and no candidate ranking by similarity. The query views
exist (`query_sim`, `ray_query_sim`) but they are lazy, off the per-step path, and meant for
viewers, tests and the heuristic baselines. The policy receives the embeddings and the mission
queries as separate inputs and learns the relevance itself.

The three products are the raw topics, nothing more:
- **voxels**: `vox_feat_sum [ny, nx, D]` (sum of unit observation features), `vox_cnt`,
  `last_seen_t`. Persistent memory: a cell is never cleared.
- **rays**: per (origin cell x azimuth bin) the running mean feature, the peak feature, the count,
  the confidence and the direction of the bin's most salient observation. Binning repeated
  observations of the same bearing from the same origin cell is the mapper's own aggregation; rays
  are never merged, deduplicated or triangulated across origins or bearings. `conf` and `n_obs`
  count looks down the bearing (one per bin per sub-step), not the far cells behind each look.
- **frontiers**: observed cells adjacent to still-unknown observable cells, clustered.
Plus one derived structure the policy reads instead of "salient cells":
- **segments**: a generic over-segmentation of the observed feature map (`sim/segments.py`), one
  scale parameter and a minimum size, no query and no ranking.

Candidate order is *recency*, never a score: newest frontier / ray / segment first, capped by the
token counts. Anything that would choose "which items deserve a slot" on semantic grounds is the
policy's job.

Deviations from CONTRACTS.md 4, all documented at the call site:
- within one sub-step the far cells that share an (origin cell, azimuth bin) are collapsed to the
  most salient class along that bearing (highest `raster.CLASS_PRIORITY`, ties by cell count) and
  get one noise draw; the weighted running mean then merges sub-steps. A per-cell mean would bury a
  far human under the 50-odd background cells in the same bin.
- a ray's `az`/`el` are the (weighted mean) direction of the cells of that most salient class, and
  they are only rewritten when an observation at least as salient arrives — the ray points at the
  thing its `feat_peak` describes, not at the centroid of the clutter around it. The *token* a ray
  offers therefore carries `feat_peak`, not the running mean: the mean over sub-steps buries a far
  casualty under the background looks that share its 20 deg bin, and it would describe something
  other than the direction the same token reports. `RayTarget.feat_mean` keeps the mean.
- ray resolution, frontier extraction and the segment statistics run once per decision rather than
  per sub-step; the policy only reads them at decision boundaries.
- the segmentation itself is re-run only once `segment_refresh_frac` of the observed cells are new
  or `segment_refresh_decisions` decisions have passed (measured: 3.4 ms on a fully observed
  240x240 belief, 74 ms on 750x750, so per-decision segmentation would eat the frame budget); the
  per-segment statistics are refreshed every decision on the cached labels.
- a frontier connected component is split into blocks of about the info-gain radius, and the target
  is the member cell nearest the part's centroid (a medoid), so a frontier token is a reachable
  point on the frontier rather than the middle of already-explored space.
- a ray resolves when the corridor it looks down is observed, when the disc of
  `ray_resolve_radius_m` around the point it aims at is observed (RayFronts drops a ray once the
  area it points into has been mapped), or after `ray_ttl_s` without a new observation.
- cells that can never be observed (`observable`, see `sensor.observable_mask`) count as already
  seen for ray resolution, frontier adjacency and info gain, so a roof no robot can look down on
  does not generate an immortal frontier sliver.
- a human's LoS target is `min(cell height, z + 0.5)`: the body top, but never above the surface of
  the rubble or structure that contains it, so a buried casualty is only visible from near overhead.

Humans are hidden state and never rasterised. A human whose LoS point is in view is *observed* with
probability `p_observe_base[visibility] * range_factor` per sub-step:
- within `depth_limit_m` the cell it occupies takes the `human_prone`/`human_standing` row for that
  observation, so the person enters the voxel map exactly like any other semantics. `found_hits`
  such observations of that cell make the human found — the same hit-count rule RayFronts uses
  to accept a voxel, and the only find mechanism there is. Two bodies of different poses in one
  cell are two observations of it, one per row, so each is counted against its own row;
- beyond it only a human whose visibility is `open` can be seen at all, and it contributes a ray.
  A `partial` (in a vehicle, inside a damaged building) or `occluded` (under rubble) human is
  invisible from afar: what the drone sees down that bearing is the container, and the container's
  own cells already carry the ray (vehicle_toppled / building_damaged / debris).

Semantics are stored as *features* (see `sim/embeddings.py`). A voxel observation is
`normalize(class_emb[cls] + N(0, feat_noise_std))`, accumulated into `vox_feat_sum [ny, nx, D]`;
`vox_feat = normalize(vox_feat_sum)`. The noise cancels as a cell is re-observed (a single look is
~7% low because normalising a noisy unit vector shrinks its projection; the estimate converges to
the class row). Rays carry `feat` (running mean) and `feat_peak` (the most salient single look).
"""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy import ndimage

from ..scene import schema
from ..scene.schema import CLASS_ID, N_CLASSES
from .embeddings import get_embedding_table
from .geometry import corridor_observed_frac, disc_observed_frac
from .raster import CLASS_PRIORITY, Raster
from .segments import segment_grid, segment_medoids, segment_ray_counts, segment_stats
from .sensor import human_visibility, visible_cells
from .state import Event, FrontierCluster, RayStore, RayTarget, SegmentToken, query_vector

from numba import njit

_EIGHT = np.ones((3, 3), bool)
OPEN_VISIBILITY = schema.HUMAN_VISIBILITY.index("open")
CAND_POOL = 2              # candidates extracted per token slot (the builder takes the newest k)


@njit(cache=True, fastmath=True)
def _ray_accum(far_ij, far_r, cls, height, x0, y0, cell_m, rx, ry, ralt,
               vis_r, nbins, binw, sw, sn, caz, cel, cw, ccnt):
    """Bin one sub-step's far cells by azimuth, keeping weighted az/el sums per (bin, class)."""
    n = far_ij.shape[0]
    for k in range(n):
        i = far_ij[k, 0]
        j = far_ij[k, 1]
        dx = x0 + (j + 0.5) * cell_m - rx
        dy = y0 + (i + 0.5) * cell_m - ry
        r = far_r[k]
        s = (height[i, j] - ralt) / r
        if s < -1.0:
            s = -1.0
        elif s > 1.0:
            s = 1.0
        az = math.atan2(dy, dx)
        w = 1.0 - r / vis_r
        if w < 1e-3:
            w = 1e-3
        b = int((az + math.pi) / binw)
        if b < 0:
            b = 0
        elif b >= nbins:
            b = nbins - 1
        c = cls[i, j]
        sw[b] += w
        sn[b] += 1
        caz[b, c] += w * az
        cel[b, c] += w * math.asin(s)
        cw[b, c] += w
        ccnt[b, c] += 1
    return


@njit(cache=True, fastmath=True)
def _ray_accum_extra(ez, rows, nbins, binw, sw, sn, caz, cel, cw, ccnt):
    for k in range(ez.shape[0]):
        az = ez[k, 0]
        w = ez[k, 2]
        b = int((az + math.pi) / binw)
        if b < 0:
            b = 0
        elif b >= nbins:
            b = nbins - 1
        c = rows[k]
        sw[b] += w
        sn[b] += 1
        caz[b, c] += w * az
        cel[b, c] += w * ez[k, 1]
        cw[b, c] += w
        ccnt[b, c] += 1
    return


@njit(cache=True, fastmath=True)
def _ray_obs(bins, ccnt, caz, cel, cw, salience, class_emb, noise, fobs, pri, csel, oaz, oel, ow):
    """Per active bin: pick the most salient class (priority, ties by cell count), build its unit
    noisy feature, the direction of its cells and the weight of that one look (the mean
    `1 - r/visual` of the cells it describes). No query is involved."""
    d_n = class_emb.shape[1]
    for m in range(bins.shape[0]):
        b = bins[m]
        best = -1
        bs = -1.0
        for c in range(ccnt.shape[1]):
            n = ccnt[b, c]
            if n <= 0:
                continue
            sc = salience[c] * 1e6 + n
            if sc > bs:
                bs = sc
                best = c
        csel[m] = best
        pri[m] = salience[best]
        w = cw[b, best]
        oaz[m] = caz[b, best] / w
        oel[m] = cel[b, best] / w
        ow[m] = w / ccnt[b, best]
        s = 0.0
        for d in range(d_n):
            v = class_emb[best, d] + noise[m, d]
            fobs[m, d] = v
            s += v * v
        inv = 1.0 / math.sqrt(s) if s > 1e-24 else 0.0
        for d in range(d_n):
            fobs[m, d] *= inv
    return


@njit(cache=True, fastmath=True)
def _ray_merge(bins, slots, sw, ow, fobs, oaz, oel, pri, r_feat, r_peak, r_ppri, r_az, r_el, r_w,
               r_conf, r_nobs, r_tlast, conf_cap, t):
    """Weighted running-mean merge of this sub-step's ray observations into the store: the mean
    feature, and the most salient single observation — its feature and its direction.

    `conf`/`n_obs` count *looks down the bearing*, one per bin per sub-step, because that is what a
    bin aggregates; the running mean still weights a sub-step by the evidence mass `sw` it carried.
    Accumulating them per far cell instead pinned every ray at the cap after two sub-steps."""
    d_n = fobs.shape[1]
    for m in range(bins.shape[0]):
        b = bins[m]
        k = slots[m]
        W = r_w[k]
        nw = W + sw[b]
        for d in range(d_n):
            r_feat[k, d] = (r_feat[k, d] * W + fobs[m, d] * sw[b]) / nw
        if pri[m] >= r_ppri[k]:
            r_ppri[k] = pri[m]
            r_az[k] = oaz[m]
            r_el[k] = oel[m]
            for d in range(d_n):
                r_peak[k, d] = fobs[m, d]
        r_w[k] = nw
        c = r_conf[k] + ow[m]
        r_conf[k] = c if c < conf_cap else conf_cap
        r_nobs[k] += 1
        r_tlast[k] = t
    return


@njit(cache=True)
def _match_prev(xs, ys, prev_xy, tol):
    """Greedy nearest-previous-centroid matching within `tol`; each previous id used once."""
    n = xs.shape[0]
    m = prev_xy.shape[0]
    out = np.full(n, -1, np.int64)
    used = np.zeros(m, np.bool_)
    t2 = tol * tol
    for k in range(n):
        best = -1
        bd = t2
        for j in range(m):
            if used[j]:
                continue
            dx = prev_xy[j, 0] - xs[k]
            dy = prev_xy[j, 1] - ys[k]
            d = dx * dx + dy * dy
            if d <= bd:
                bd = d
                best = j
        if best >= 0:
            used[best] = True
            out[k] = best
    return out


@njit(cache=True)
def _part_medoids(fi, fj, starts, counts):
    """Per part: the member cell nearest the part's centroid."""
    n = starts.shape[0]
    mi = np.empty(n, np.int64)
    mj = np.empty(n, np.int64)
    for k in range(n):
        a = starts[k]
        c = counts[k]
        si = 0.0
        sj = 0.0
        for m in range(a, a + c):
            si += fi[m]
            sj += fj[m]
        si /= c
        sj /= c
        best = 1e30
        bi = a
        for m in range(a, a + c):
            d = (fi[m] - si) ** 2 + (fj[m] - sj) ** 2
            if d < best:
                best = d
                bi = m
        mi[k] = fi[bi]
        mj[k] = fj[bi]
    return mi, mj


@njit(cache=True)
def _track_cells(ij, dec_by, seen_by, bit):
    """Mark `ij` as observed by one robot this decision -> (cells new to it, of which redundant).

    Redundant = a cell some *other* robot had already observed before this decision (`seen_by` is
    only folded forward at `commit_decision`, so it is the decision's snapshot throughout).
    """
    fresh = 0
    red = 0
    for k in range(ij.shape[0]):
        i = ij[k, 0]
        j = ij[k, 1]
        if dec_by[i, j] & bit:
            continue
        dec_by[i, j] |= bit
        fresh += 1
        if seen_by[i, j] & ~bit:
            red += 1
    return fresh, red


@njit(cache=True, fastmath=True)
def _vox_scatter(rows_ij, row_cls, class_emb, noise, feat_sum, vox_cnt, last_seen, t):
    """One unit noisy class feature per observed cell, accumulated into the persistent map.

    Nothing per-query happens here: this is the whole voxel update."""
    m = rows_ij.shape[0]
    d_n = class_emb.shape[1]
    for k in range(m):
        i = rows_ij[k, 0]
        j = rows_ij[k, 1]
        c = row_cls[k]
        vox_cnt[i, j] += 1
        last_seen[i, j] = t
        s = 0.0
        for d in range(d_n):
            v = class_emb[c, d] + noise[k, d]
            s += v * v
        inv = 1.0 / math.sqrt(s) if s > 1e-24 else 0.0
        for d in range(d_n):
            feat_sum[i, j, d] += (class_emb[c, d] + noise[k, d]) * inv



class FrontierIndex:
    """Frontier extraction and stable cluster ids for one belief.

    One instance per belief: the team map owns one, and under `comms.mode == "range"` so does every
    robot, so a robot's frontiers are the boundary of *its* knowledge and nothing else.
    """

    def __init__(self, raster: Raster, cfg):
        self.raster = raster
        self.cfg = cfg
        rf = cfg.rayfronts
        self.ig_disc = _disc_mask(rf.frontier_ig_radius_m, raster.cell_m)
        # components are further split by blocks of about the IG radius (see extract)
        self.blk = max(2, int(round(rf.frontier_ig_radius_m / raster.cell_m)))
        self.mask = np.zeros(raster.shape, np.bool_)
        self.clusters: list[FrontierCluster] = []
        self.next_id = 0

    def extract(self, observed: np.ndarray, seen: np.ndarray
                ) -> tuple[np.ndarray, list[FrontierCluster]]:
        un = ~seen
        f = np.zeros_like(observed)
        f[:-1, :] |= un[1:, :]
        f[1:, :] |= un[:-1, :]
        f[:, :-1] |= un[:, 1:]
        f[:, 1:] |= un[:, :-1]
        f &= observed
        self.mask = f
        prev = self.clusters
        self.clusters = []
        if not f.any():
            return self.mask, self.clusters
        min_cells = self.cfg.rayfronts.frontier_min_cluster_cells
        lab, nlab = ndimage.label(f, structure=_EIGHT)
        fi, fj = np.nonzero(f)
        labs = lab[fi, fj]
        big = np.bincount(labs, minlength=nlab + 1)[labs] >= min_cells
        fi, fj, labs = fi[big], fj[big], labs[big]
        if fi.size == 0:
            return self.mask, self.clusters
        # a connected component can be one long ring; split it by coarse spatial blocks so each
        # token is a local piece of frontier rather than the whole outline of explored space
        B = self.blk
        nbj = (observed.shape[1] + B - 1) // B
        nblk = nbj * ((observed.shape[0] + B - 1) // B)
        key = labs.astype(np.int64) * nblk + (fi // B) * nbj + (fj // B)
        order = np.argsort(key, kind="stable")
        fi, fj, key = fi[order], fj[order], key[order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1]).astype(np.int64)
        counts = np.diff(np.append(starts, key.size)).astype(np.int64)
        sel = counts >= min_cells
        starts, counts = starts[sel], counts[sel]
        if starts.size == 0:
            return self.mask, self.clusters
        mi, mj = _part_medoids(fi, fj, starts, counts)
        ig = _disc_counts(un, self.ig_disc, mi, mj)
        xs, ys = self.raster.ij_to_xy(mi, mj)
        cells = np.stack([fi, fj], 1).astype(np.int32)
        pc = (np.array([p.centroid_xy for p in prev], np.float64) if prev
              else np.zeros((0, 2), np.float64))
        match = _match_prev(xs, ys, pc, 2.0 * self.raster.cell_m)
        out: list[FrontierCluster] = []
        for k in range(starts.size):
            m = int(match[k])
            if m >= 0:
                cid = prev[m].id
            else:
                cid = self.next_id
                self.next_id += 1
            a = int(starts[k])
            out.append(FrontierCluster(
                id=cid, centroid_xy=np.array([xs[k], ys[k]], np.float64), n_cells=int(counts[k]),
                info_gain=float(ig[k]), cell_ij=cells[a:a + int(counts[k])]))
        # newest first, then the bigger piece of frontier: recency and size, never info gain
        out.sort(key=lambda c: (-c.id, -c.n_cells))
        self.clusters = out
        return self.mask, self.clusters


class SegmentIndex:
    """Segment labels, stable ids and per-decision statistics for one belief.

    Same split as `FrontierIndex`: the team map owns one and every robot owns one under range
    comms. The expensive labelling runs on the refresh rule, the statistics every decision.
    """

    def __init__(self, raster: Raster, cfg, D: int):
        self.raster = raster
        self.cfg = cfg
        self.D = int(D)
        self.labels = np.full(raster.shape, -1, np.int32)
        self.n_seg = 0
        self.xy = np.zeros((0, 2), np.float64)
        self.ij = np.zeros((0, 2), np.int64)
        self.ids = np.zeros(0, np.int64)
        self.t0 = np.zeros(0, np.float64)
        self.obs_at = -1               # observed-cell count when the labels were last computed
        self.dec_at = -10 ** 9
        self.next_id = 0

    def needs_resegment(self, n_obs: int, decision: int) -> bool:
        rf = self.cfg.rayfronts
        if self.obs_at < 0:
            return n_obs > 0
        if n_obs - self.obs_at >= max(1, rf.segment_refresh_frac * max(1, self.obs_at)):
            return True
        return decision - self.dec_at >= int(rf.segment_refresh_decisions)

    def extract(self, feat_sum, observed, vox_cnt, last_seen, ray_org, ray_az, t: float,
                decision: int) -> list[SegmentToken]:
        rf = self.cfg.rayfronts
        n_obs = int(observed.sum())
        if n_obs == 0:
            self.labels[:] = -1
            self.n_seg = 0
            return []
        if self.needs_resegment(n_obs, decision):
            self.resegment(feat_sum, observed, vox_cnt, last_seen, t,
                           float(rf.segment_scale), int(rf.segment_min_cells))
            self.obs_at = n_obs
            self.dec_at = decision
        n_seg = self.n_seg
        if n_seg == 0:
            return []
        fsum = np.zeros((n_seg, self.D), np.float32)
        count = np.zeros(n_seg, np.int64)
        hits = np.zeros(n_seg, np.int64)
        si = np.zeros(n_seg, np.float64)
        sj = np.zeros(n_seg, np.float64)
        t_last = np.full(n_seg, -1.0, np.float32)
        segment_stats(feat_sum, vox_cnt, last_seen, self.labels, n_seg,
                      fsum, count, hits, si, sj, t_last)
        rays = np.zeros(n_seg, np.int64)
        if ray_org.shape[0]:
            stamp = np.zeros(n_seg, np.int64)
            segment_ray_counts(self.labels, ray_org, ray_az,
                               float(self.cfg.sensor.depth_limit_m), rf_visual(self.cfg),
                               float(self.raster.cell_m), float(self.raster.origin[0]),
                               float(self.raster.origin[1]), float(self.raster.cell_m), n_seg,
                               stamp, rays)
        keep = np.flatnonzero(count > 0)
        if keep.size == 0:
            return []
        feat = fsum[keep] / np.maximum(np.linalg.norm(fsum[keep], axis=1, keepdims=True), 1e-12)
        out: list[SegmentToken] = []
        for m, sgi in enumerate(keep):
            sgi = int(sgi)
            out.append(SegmentToken(
                id=int(self.ids[sgi]), xy=self.xy[sgi].copy(),
                ij=(int(self.ij[sgi, 0]), int(self.ij[sgi, 1])),
                feat=np.ascontiguousarray(feat[m], np.float32), n_cells=int(count[sgi]),
                mean_hits=float(hits[sgi] / max(1, count[sgi])), ray_count=int(rays[sgi]),
                t_first=float(self.t0[sgi]), t_last=float(t_last[sgi])))
        out.sort(key=lambda s: (-s.t_first, -s.id))     # newest first, no score
        return out[: int(CAND_POOL * max(1, self.cfg.tokens.k_segment))]

    def resegment(self, feat_sum, observed, vox_cnt, last_seen, t: float, scale: float,
                  min_cells: int) -> None:
        """Recompute the labels and everything that depends only on them (medoid, id, t_first)."""
        prev_xy, prev_id, prev_t0 = self.xy, self.ids, self.t0
        n_seg = segment_grid(feat_sum, observed, scale, min_cells, self.labels)
        self.n_seg = n_seg
        if n_seg == 0:
            self.xy = np.zeros((0, 2), np.float64)
            self.ids = np.zeros(0, np.int64)
            self.t0 = np.zeros(0, np.float64)
            self.ij = np.zeros((0, 2), np.int64)
            return
        ci = np.zeros(n_seg, np.float64)
        cj = np.zeros(n_seg, np.float64)
        # only the centroid is wanted here; the feature/hit statistics are refreshed per decision
        fsum = np.zeros((n_seg, self.D), np.float32)
        hits = np.zeros(n_seg, np.int64)
        tl = np.full(n_seg, -1.0, np.float32)
        count = np.zeros(n_seg, np.int64)
        segment_stats(feat_sum, vox_cnt, last_seen, self.labels, n_seg,
                      fsum, count, hits, ci, cj, tl)
        good = np.maximum(count, 1)
        ci /= good
        cj /= good
        mi = np.zeros(n_seg, np.int64)
        mj = np.zeros(n_seg, np.int64)
        best = np.full(n_seg, 1e30)
        segment_medoids(self.labels, n_seg, ci, cj, mi, mj, best)
        xs, ys = self.raster.ij_to_xy(mi, mj)
        match = _match_prev(xs, ys, prev_xy, 2.0 * self.raster.cell_m)
        ids = np.empty(n_seg, np.int64)
        t0 = np.empty(n_seg, np.float64)
        for k in range(n_seg):
            m = int(match[k])
            if m >= 0:
                ids[k], t0[k] = prev_id[m], prev_t0[m]
            else:
                ids[k], t0[k] = self.next_id, float(t)
                self.next_id += 1
        self.xy = np.stack([xs, ys], 1)
        self.ij = np.stack([mi, mj], 1)
        self.ids = ids
        self.t0 = t0


def ray_resolve_flags(seen: np.ndarray, raster: Raster, cfg, origin, az, el, t_last, t: float,
                      target_range) -> np.ndarray:
    """Which of the given rays are resolved against `seen` (CONTRACTS.md 4).

    Shared by the team belief and, under range comms, by each robot's own view of the rays it
    knows: a ray dies when *that* map has covered the corridor or the disc it points into.
    """
    rf = cfg.rayfronts
    n = origin.shape[0]
    if n == 0:
        return np.zeros(0, np.bool_)
    org = np.ascontiguousarray(origin, np.float64)
    azc = np.ascontiguousarray(az, np.float64)
    frac = corridor_observed_frac(seen, raster.cell_m, raster.origin[0], raster.origin[1], org,
                                  azc, float(cfg.sensor.depth_limit_m), rf_visual(cfg))
    r = target_range(el)
    tgt = np.ascontiguousarray(org + r[:, None] * np.stack([np.cos(azc), np.sin(azc)], 1))
    tfrac = disc_observed_frac(seen, raster.cell_m, raster.origin[0], raster.origin[1], tgt,
                               float(rf.ray_resolve_radius_m))
    stale = (t - np.asarray(t_last, np.float64)) > rf.ray_ttl_s
    return (frac >= rf.ray_resolve_frac) | (tfrac >= rf.ray_resolve_frac) | stale


class RayFrontsSim:
    def __init__(self, raster: Raster, cfg, rng: np.random.Generator):
        self.raster = raster
        self.cfg = cfg
        rf = cfg.rayfronts
        self.queries = tuple(rf.queries)
        self.nq = len(self.queries)
        self.emb = get_embedding_table(self.queries, dim=rf.embedding_dim,
                                       path=rf.embeddings_path, sim_table_path=rf.sim_table_path)
        self.class_emb = self.emb.class_emb           # [N_CLASSES, D] unit
        self.query_emb = self.emb.query_emb           # [Q, D] unit: the *mission* queries
        self.query_w = np.ones(self.nq, np.float32)   # 1.0 for a mission query; hints carry theirs
        self.D = self.emb.D
        self._salience = CLASS_PRIORITY.astype(np.float64)
        self.n_query_calls = 0     # lazy query views taken; must stay 0 across a decision
        ny, nx = raster.shape
        self.vox_feat_sum = np.zeros((ny, nx, self.D), np.float32)
        self.vox_cnt = np.zeros((ny, nx), np.int32)
        self.last_seen_t = np.full((ny, nx), -1.0, np.float32)
        self.observed = np.zeros((ny, nx), np.bool_)

        self.n_rays = 0
        self._cap = 0
        self._grow(256)
        self._ray_key: dict[tuple[int, int, int], int] = {}
        self._r_keyof: list[tuple[int, int, int]] = []
        self._next_ray_id = 0

        n_h = raster.humans.shape[0]
        self.human_hits = np.zeros(n_h, np.int32)
        self.human_found = np.zeros(n_h, np.bool_)
        self._human_pts = np.empty((n_h, 3), np.float64)
        if n_h:
            self._human_pts[:, 0] = raster.humans["x"]
            self._human_pts[:, 1] = raster.humans["y"]
            hz = raster.humans["z"] + 0.5
            zi, zj = raster.xy_to_ij(raster.humans["x"], raster.humans["y"])
            zh = raster.height[np.clip(zi, 0, ny - 1), np.clip(zj, 0, nx - 1)]
            self._human_pts[:, 2] = np.minimum(hz, zh)   # never above the surface covering the body
        pb = rf.p_observe_base
        self._p_observe = np.array([float(pb[v]) for v in schema.HUMAN_VISIBILITY], np.float64)
        self._human_row = np.where(raster.humans["pose_id"] == schema.HUMAN_POSES.index("standing"),
                                   CLASS_ID["human_standing"], CLASS_ID["human_prone"]).astype(np.int64) \
            if n_h else np.zeros(0, np.int64)
        hi, hj = raster.xy_to_ij(self._human_pts[:, 0], self._human_pts[:, 1]) if n_h else (
            np.zeros(0, np.int64), np.zeros(0, np.int64))
        self._human_ij = np.stack([np.clip(hi, 0, ny - 1), np.clip(hj, 0, nx - 1)], 1).astype(np.int32) \
            if n_h else np.zeros((0, 2), np.int32)

        self.observable = np.ones((ny, nx), np.bool_)
        self._all_observable = True
        self._seen = self.observed
        self.frontiers = FrontierIndex(raster, cfg)
        self.frontier_mask = self.frontiers.mask
        self.frontier_clusters: list[FrontierCluster] = []
        self._ray_targets: list[RayTarget] = []
        self._segments: list[SegmentToken] = []
        self.segidx = SegmentIndex(raster, cfg, self.D)
        # who observed what: bit r of `seen_by` is set once robot r has voxel-observed the cell,
        # `dec_by` the same for the decision in progress. Two grids for the whole team (uint16,
        # n_robots <= 10) rather than a mask per robot; they feed the per-robot known masks and
        # the redundancy term, and cost one indexed write per sub-step.
        self.seen_by = np.zeros((ny, nx), np.uint16)
        self.dec_by = np.zeros((ny, nx), np.uint16)
        self.redundant_cells = np.zeros(16, np.int64)   # per robot, this decision
        self.observed_cells = np.zeros(16, np.int64)    # ... of how many it observed at all
        self.redundancy_refund = np.zeros(16, np.bool_)  # ... waived by a find in a redundant cell
        self.found_this_decision: list[tuple[int, int]] = []   # (robot, human_idx) of this decision
        self.on_observe = None      # optional hook(robot_idx, ij[:, 2]) for the per-robot beliefs
        self.keep_rays = False      # True under range comms: rows stay put, so row index == ray id
        self._t_dec = 0.0
        self._last_robots: tuple = ()
        self.n_fp_rays = 0
        self.prof: dict[str, float] | None = None

        self._nbins = max(1, int(round(360.0 / rf.ray_az_bin_deg)))
        self._binw = 2 * math.pi / self._nbins
        self._b_feat = np.zeros((self._nbins, self.D), np.float32)     # per-sub-step bin scratch
        self._b_pri = np.zeros(self._nbins, np.float64)
        self._b_cls = np.zeros(self._nbins, np.int64)
        self._b_az = np.zeros(self._nbins, np.float64)
        self._b_el = np.zeros(self._nbins, np.float64)
        self._b_w = np.zeros(self._nbins, np.float64)

    # ---- ray memory -------------------------------------------------------------------------
    def _grow(self, cap: int) -> None:
        def g(old, shape, dt, fill=0):
            a = np.full((cap,) + shape, fill, dt)
            if old is not None:
                a[:self.n_rays] = old[:self.n_rays]
            return a
        first = self._cap == 0
        self._r_origin = g(None if first else self._r_origin, (2,), np.float64)
        self._r_az = g(None if first else self._r_az, (), np.float64)
        self._r_el = g(None if first else self._r_el, (), np.float64)
        self._r_feat = g(None if first else self._r_feat, (self.D,), np.float32)
        self._r_peak = g(None if first else self._r_peak, (self.D,), np.float32)
        self._r_ppri = g(None if first else self._r_ppri, (), np.float64, -1.0)
        self._r_conf = g(None if first else self._r_conf, (), np.float32)
        self._r_nobs = g(None if first else self._r_nobs, (), np.int32)
        self._r_tfirst = g(None if first else self._r_tfirst, (), np.float64)
        self._r_tlast = g(None if first else self._r_tlast, (), np.float64)
        self._r_ids = g(None if first else self._r_ids, (), np.int32, -1)
        self._r_res = g(None if first else self._r_res, (), np.bool_)
        self._r_w = g(None if first else self._r_w, (), np.float64)
        self._r_by = g(None if first else self._r_by, (), np.uint16)   # robots that fed this bin
        self._cap = cap

    # legacy attribute names of the segment index (kept: viewers and tests read them)
    @property
    def seg_labels(self) -> np.ndarray:
        return self.segidx.labels

    @property
    def _n_seg(self) -> int:
        return self.segidx.n_seg

    @property
    def _seg_xy(self) -> np.ndarray:
        return self.segidx.xy

    @property
    def _seg_ij(self) -> np.ndarray:
        return self.segidx.ij

    @property
    def _seg_id(self) -> np.ndarray:
        return self.segidx.ids

    @property
    def _seg_t0(self) -> np.ndarray:
        return self.segidx.t0

    @property
    def _next_seg_id(self) -> int:
        return self.segidx.next_id

    @property
    def _next_cluster_id(self) -> int:
        return self.frontiers.next_id

    @property
    def _seg_obs_at(self) -> int:
        return self.segidx.obs_at

    @_seg_obs_at.setter
    def _seg_obs_at(self, v: int) -> None:
        self.segidx.obs_at = int(v)

    @property
    def _seg_dec_at(self) -> int:
        return self.segidx.dec_at

    @_seg_dec_at.setter
    def _seg_dec_at(self, v: int) -> None:
        self.segidx.dec_at = int(v)

    def store(self) -> RayStore:
        n = self.n_rays
        return RayStore(origin_xy=self._r_origin[:n], az=self._r_az[:n], el=self._r_el[:n],
                        conf=self._r_conf[:n], n_obs=self._r_nobs[:n], t_first=self._r_tfirst[:n],
                        t_last=self._r_tlast[:n], ids=self._r_ids[:n], resolved=self._r_res[:n],
                        feat=self._r_feat[:n], feat_peak=self._r_peak[:n])

    def _new_ray(self, key, ox, oy, t) -> int:
        if self.n_rays >= self._cap:
            self._grow(max(256, 2 * self._cap))
        k = self.n_rays
        self.n_rays += 1
        self._r_origin[k, 0] = ox
        self._r_origin[k, 1] = oy
        self._r_feat[k] = 0.0
        self._r_peak[k] = 0.0
        self._r_ppri[k] = -1.0
        self._r_conf[k] = 0.0
        self._r_nobs[k] = 0
        self._r_w[k] = 0.0
        self._r_tfirst[k] = t
        self._r_tlast[k] = t
        self._r_ids[k] = self._next_ray_id
        self._r_res[k] = False
        self._r_by[k] = 0
        self._next_ray_id += 1
        self._ray_key[key] = k
        if k < len(self._r_keyof):
            self._r_keyof[k] = key
        else:
            self._r_keyof.append(key)
        return k

    def compact(self) -> None:
        if self.keep_rays:      # per-robot beliefs index ray rows: never move them under gossip
            return
        n = self.n_rays
        keep = np.flatnonzero(~self._r_res[:n])
        if keep.size == n:
            return
        for name in ("_r_origin", "_r_az", "_r_el", "_r_feat", "_r_peak", "_r_ppri", "_r_conf",
                     "_r_nobs", "_r_tfirst", "_r_tlast", "_r_ids", "_r_res", "_r_w", "_r_by"):
            a = getattr(self, name)
            a[:keep.size] = a[keep]
        self.n_rays = int(keep.size)
        self._r_keyof = [self._r_keyof[int(o)] for o in keep]
        self._ray_key = {k: i for i, k in enumerate(self._r_keyof)}

    def set_observable(self, mask: np.ndarray | None) -> None:
        """Cells that can never be observed count as seen everywhere the belief asks "is this
        still unknown?" (ray resolution, frontier adjacency, info gain)."""
        if mask is None:
            self.observable = np.ones(self.observed.shape, np.bool_)
        else:
            if mask.shape != self.observed.shape:
                raise ValueError(f"set_observable: mask shape {mask.shape} != {self.observed.shape}")
            self.observable = np.ascontiguousarray(mask, np.bool_)
        self._all_observable = bool(self.observable.all())

    def _refresh_seen(self) -> None:
        self._seen = self.observed if self._all_observable else (self.observed | ~self.observable)

    # ---- decision boundaries ------------------------------------------------------------------
    def begin_decision(self, n_robots: int) -> None:
        """Open a decision: nothing has been observed in it yet, so `seen_by` is the snapshot the
        redundancy term compares against for the whole decision."""
        if self.redundant_cells.shape[0] < n_robots:
            self.redundant_cells = np.zeros(n_robots, np.int64)
            self.observed_cells = np.zeros(n_robots, np.int64)
            self.redundancy_refund = np.zeros(n_robots, np.bool_)
        self.dec_by[:] = 0
        self.redundant_cells[:] = 0
        self.observed_cells[:] = 0
        self.redundancy_refund[:] = False
        self.found_this_decision = []

    def commit_decision(self) -> None:
        """Fold this decision's observations into `seen_by`."""
        np.bitwise_or(self.seen_by, self.dec_by, out=self.seen_by)

    # ---- one sub-step -----------------------------------------------------------------------
    def update(self, robots, t: float, rng: np.random.Generator) -> list[Event]:
        ev: list[Event] = []
        for rb in robots:
            self._update_robot(rb, t, rng, ev)
        return ev

    def _tick(self, key: str, t0: float) -> float:
        if self.prof is not None:
            now = perf_counter()
            self.prof[key] = self.prof.get(key, 0.0) + (now - t0)
            return now
        return 0.0

    def _update_robot(self, rb, t: float, rng: np.random.Generator, ev: list[Event]) -> None:
        cfg, rf, ras = self.cfg, self.cfg.rayfronts, self.raster
        t0 = perf_counter() if self.prof is not None else 0.0
        cam = np.array([rb.pos[0], rb.pos[1], rb.alt], np.float64)
        view = visible_cells(ras, cfg.sensor, cam, rb.heading, far_p=rf.p_ray_per_cell, rng=rng)
        t0 = self._tick("sensor", t0)

        # -- humans ---------------------------------------------------------------------------
        # keyed by (cell, class row): two bodies in one cell with different poses are two distinct
        # voxel observations, so each is credited against the row that was actually written
        near_rows: dict[tuple[int, int, int], list[int]] = {}
        far_extra: list[tuple[int, float]] = []
        depth = float(cfg.sensor.depth_limit_m)
        if self._human_pts.shape[0]:
            in_view, hr = human_visibility(ras, cfg.sensor, cam, rb.heading, self._human_pts)
            cand = np.flatnonzero(in_view)
            if cand.size:
                vis = ras.humans["visibility_id"][cand]
                p = self._p_observe[vis] * _range_factor(hr[cand], rf.far_observe_factor, depth)
                # beyond the depth limit only an open human is separable from its surroundings;
                # a human in a car or under rubble simply is not there as far as a ray is concerned
                p = np.where((hr[cand] > depth) & (vis != OPEN_VISIBILITY), 0.0, p)
                hit = cand[rng.random(cand.size) < p]
                for k in hit:
                    k = int(k)
                    if hr[k] <= depth:
                        key = (int(self._human_ij[k, 0]), int(self._human_ij[k, 1]),
                               int(self._human_row[k]))
                        near_rows.setdefault(key, []).append(k)
                    else:
                        far_extra.append((k, float(hr[k])))
        t0 = self._tick("humans", t0)

        # -- voxels ---------------------------------------------------------------------------
        obs = view.observed_ij
        if obs.shape[0] or near_rows:
            rows = ras.cls[obs[:, 0], obs[:, 1]].astype(np.int64)
            if rf.p_confuse > 0.0:
                cm = rng.random(rows.size) < rf.p_confuse
                if cm.any():
                    alt = rng.integers(0, N_CLASSES - 1, size=int(cm.sum()))
                    alt += (alt >= rows[cm])
                    rows[cm] = alt
            add_ij, add_rows = [], []
            claimed: set[tuple[int, int]] = set()
            for (i, j, c) in near_rows:
                m = (obs[:, 0] == i) & (obs[:, 1] == j) if obs.shape[0] else np.zeros(0, bool)
                if m.any() and (i, j) not in claimed:
                    rows[m] = c
                    claimed.add((i, j))
                else:
                    # either the cell's own top point failed the cell-level frustum/LoS test (nadir
                    # edges, a body beside a wall) or a second body in the cell carries another
                    # row: the camera saw that person, so the map gets its own observation
                    add_ij.append((i, j))
                    add_rows.append(c)
            if add_ij:
                obs = np.concatenate([obs, np.array(add_ij, np.int32).reshape(-1, 2)])
                rows = np.concatenate([rows, np.array(add_rows, np.int64)])
            noise = (rng.standard_normal(size=(obs.shape[0], self.D), dtype=np.float32)
                     * np.float32(rf.feat_noise_std))
            _vox_scatter(obs, rows, self.class_emb, noise, self.vox_feat_sum, self.vox_cnt,
                         self.last_seen_t, np.float32(t))
            self.observed[obs[:, 0], obs[:, 1]] = True
            self._track(rb.idx, obs)
        for ks in near_rows.values():
            for k in ks:
                self._count_hit(k, t, rb, ev)
        t0 = self._tick("voxels", t0)

        # -- rays -----------------------------------------------------------------------------
        self._emit_rays(rb, view, far_extra, t, rng, ev)
        self._tick("rays", t0)

    def _track(self, ridx: int, ij: np.ndarray) -> None:
        """Book this robot's observation of `ij`: its own knowledge, how many distinct cells it
        covered this decision, and how many of those were already somebody else's before it."""
        r = int(ridx)
        fresh, red = _track_cells(np.ascontiguousarray(ij, np.int32), self.dec_by, self.seen_by,
                                  np.uint16(1 << r))
        self.observed_cells[r] += fresh
        self.redundant_cells[r] += red
        if self.on_observe is not None:
            self.on_observe(r, ij)

    def _count_hit(self, k: int, t: float, rb, ev: list[Event]) -> None:
        """One hit per voxel observation of this human's cell carrying this human's own class row:
        found-ness is a property of the voxel map, not of a private sensor channel."""
        self.human_hits[k] += 1
        if self.human_found[k] or self.human_hits[k] < self.cfg.rayfronts.found_hits:
            return
        self.human_found[k] = True
        self.found_this_decision.append((int(rb.idx), int(k)))
        hi, hj = int(self._human_ij[k, 0]), int(self._human_ij[k, 1])
        if (int(self.raster.humans["role_id"][k]) == schema.HUMAN_ROLES.index("casualty")
                and (self.seen_by[hi, hj] & ~np.uint16(1 << int(rb.idx))) != 0):
            # a casualty in a cell a peer had already covered: the redundancy this robot paid for
            # this decision bought a find, so it is refunded
            self.redundancy_refund[int(rb.idx)] = True
        if self.cfg.record_events:
            h = self.raster.humans[k]
            ev.append(Event(t, "found", {
                "human_idx": k, "robot": rb.idx,
                "casualty": int(h["role_id"]) == schema.HUMAN_ROLES.index("casualty"),
                "container": schema.HUMAN_CONTAINERS[int(h["container_id"])],
                "visibility": schema.HUMAN_VISIBILITY[int(h["visibility_id"])]}))

    def _emit_rays(self, rb, view, far_extra, t: float, rng, ev: list[Event]) -> None:
        rf, ras = self.cfg.rayfronts, self.raster
        far = view.far_ij
        n_far = far.shape[0]
        fp = rng.random() < rf.p_fp_ray
        if n_far == 0 and not far_extra and not fp:
            return
        vis = float(self.cfg.sensor.visual_range_m)
        nb, bw = self._nbins, self._binw
        sw = np.zeros(nb, np.float64)
        sn = np.zeros(nb, np.int64)
        caz = np.zeros((nb, N_CLASSES), np.float64)
        cel = np.zeros((nb, N_CLASSES), np.float64)
        cw = np.zeros((nb, N_CLASSES), np.float64)
        ccnt = np.zeros((nb, N_CLASSES), np.int32)
        if n_far:
            _ray_accum(far, view.slant_r, ras.cls, ras.height,
                       ras.origin[0], ras.origin[1], ras.cell_m, rb.pos[0], rb.pos[1], rb.alt,
                       vis, nb, bw, sw, sn, caz, cel, cw, ccnt)
        n_extra = len(far_extra) + (1 if fp else 0)
        if n_extra:
            ez = np.empty((n_extra, 3), np.float64)      # az, el, w
            erow = np.empty(n_extra, np.int64)
            for m, (k, hrk) in enumerate(far_extra):
                dx = self._human_pts[k, 0] - rb.pos[0]
                dy = self._human_pts[k, 1] - rb.pos[1]
                ez[m, 0] = math.atan2(dy, dx)
                ez[m, 1] = math.asin(min(1.0, max(-1.0, (self._human_pts[k, 2] - rb.alt) / max(hrk, 1e-9))))
                ez[m, 2] = max(1e-3, 1.0 - hrk / vis)
                erow[m] = self._human_row[k]
            if fp:
                m = n_extra - 1
                r_fp = 0.5 * (self.cfg.sensor.depth_limit_m + vis)
                ez[m, 0] = rng.uniform(-math.pi, math.pi)
                ez[m, 1] = -math.asin(min(1.0, rb.alt / r_fp))
                ez[m, 2] = max(1e-3, 1.0 - r_fp / vis)
                erow[m] = CLASS_ID["human_prone"]
                self.n_fp_rays += 1
            _ray_accum_extra(ez, erow, nb, bw, sw, sn, caz, cel, cw, ccnt)

        bins = np.flatnonzero(sn)
        if bins.size == 0:
            return
        # one ray observation per bin: the feature of the most salient class along that bearing
        # (raster priority, ties by cell count), plus feature-space noise
        nbn = bins.size
        noise = (rng.standard_normal(size=(nbn, self.D), dtype=np.float32)
                 * np.float32(rf.ray_noise_std))
        fobs, pri, csel = self._b_feat[:nbn], self._b_pri[:nbn], self._b_cls[:nbn]
        oaz, oel, ow = self._b_az[:nbn], self._b_el[:nbn], self._b_w[:nbn]
        _ray_obs(bins, ccnt, caz, cel, cw, self._salience, self.class_emb, noise, fobs, pri, csel,
                 oaz, oel, ow)

        oc = rf.ray_origin_cell_m
        ox = int(math.floor(rb.pos[0] / oc))
        oy = int(math.floor(rb.pos[1] / oc))
        slots = np.empty(bins.size, np.int64)
        for m, bi in enumerate(bins):
            bi = int(bi)
            key = (ox, oy, bi)
            idx = self._ray_key.get(key)
            if idx is None:
                idx = self._new_ray(key, rb.pos[0], rb.pos[1], t)
            slots[m] = idx
        self._r_by[slots] |= np.uint16(1 << int(rb.idx))
        _ray_merge(bins, slots, sw, ow, fobs, oaz, oel, pri, self._r_feat, self._r_peak,
                   self._r_ppri, self._r_az, self._r_el, self._r_w, self._r_conf, self._r_nobs,
                   self._r_tlast, float(rf.ray_conf_cap), float(t))

    # ---- queries (lazy views, never on the per-step path) --------------------------------------
    @property
    def vox_feat(self) -> np.ndarray:
        """float32 [ny, nx, D] unit per-cell features (zeros where unobserved). Allocates."""
        n = np.linalg.norm(self.vox_feat_sum, axis=-1, keepdims=True)
        return self.vox_feat_sum / np.maximum(n, 1e-12)

    def query_vec(self, query) -> np.ndarray:
        """Unit [D] vector for a query name, an index into the live query list, or a raw vector."""
        return query_vector(self.emb, self.queries, query)

    def query_sim(self, query) -> np.ndarray:
        """float32 [ny, nx] cosine of the voxel map against one query, clipped to [0, 1].

        Computed on demand from the stored features. Every call bumps `n_query_calls`, which the
        tests use to prove that no per-step code path scans the map against a query.
        """
        self.n_query_calls += 1
        q = self.query_vec(query)
        n = np.linalg.norm(self.vox_feat_sum, axis=-1)
        out = (self.vox_feat_sum @ q) / np.maximum(n, 1e-12)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def ray_query_sim(self, query, peak: bool = True) -> np.ndarray:
        """float32 [n_rays] cosine of each ray against one query (max of mean and peak feature).

        `peak=False` uses only the running mean. Lazy, like `query_sim`.
        """
        self.n_query_calls += 1
        q = self.query_vec(query)
        n = self.n_rays
        if n == 0:
            return np.zeros(0, np.float32)
        s = np.clip(_cos(self._r_feat[:n], q), 0.0, 1.0)
        if peak:
            s = np.maximum(s, np.clip(_cos(self._r_peak[:n], q), 0.0, 1.0))
        return s.astype(np.float32)

    def set_queries(self, names, weights=None) -> None:
        """Swap the *mission* query list (and optionally the per-query weights).

        Nothing in the belief depends on it any more: the voxel map, the rays and the segments are
        features, so this only re-encodes the query tokens the policy is given.
        """
        names = tuple(names)
        if not names:
            raise ValueError("set_queries: empty query list")
        emb = self.emb.with_queries(names)        # unknown names raise here, before anything moves
        w = (np.ones(len(names), np.float32) if weights is None
             else np.asarray(weights, np.float32).reshape(-1))
        if w.shape[0] != len(names):
            raise ValueError(f"set_queries: {w.shape[0]} weights for {len(names)} queries")
        self.emb = emb
        self.queries = names
        self.nq = len(names)
        self.query_emb = self.emb.query_emb
        self.query_w = w

    # ---- per-decision refresh ----------------------------------------------------------------
    def end_of_decision(self, t: float, robots=()) -> list[Event]:
        ev: list[Event] = []
        self._t_dec = float(t)
        self._last_robots = tuple(robots)
        self._refresh_seen()
        t0 = perf_counter() if self.prof is not None else 0.0
        self._resolve_rays(t)
        t0 = self._tick("ray_resolve", t0)
        self._extract_frontiers()
        t0 = self._tick("frontiers", t0)
        self._extract_ray_targets(t)
        t0 = self._tick("ray_targets", t0)
        self._extract_segments(t)
        self._tick("segments", t0)
        return ev

    def target_range(self, el):
        """Ground range of what a ray at elevation `el` points at: `alt / tan(-el)` for a target on
        the ground, clipped to the ray band. `el >= 0` cannot hit the ground: fall back to
        `ray_range_m`."""
        rf = self.cfg.rayfronts
        alt = float(self.cfg.robot.flight_alt_m)
        e = np.asarray(el, np.float64)
        down = e < 0.0
        r = np.where(down, alt / np.tan(np.where(down, -e, 1.0)), float(rf.ray_range_m))
        return np.clip(r, float(self.cfg.sensor.depth_limit_m),
                       float(self.cfg.sensor.visual_range_m))

    def _ray_points(self, idx: np.ndarray) -> np.ndarray:
        r = self.target_range(self._r_el[idx])
        a = self._r_az[idx]
        return self._r_origin[idx] + r[:, None] * np.stack([np.cos(a), np.sin(a)], 1)

    def _resolve_rays(self, t: float) -> None:
        rf = self.cfg.rayfronts
        n = self.n_rays
        if n == 0:
            return
        live = np.flatnonzero(~self._r_res[:n])
        if live.size == 0:
            return
        gone = live[ray_resolve_flags(self._seen, self.raster, self.cfg, self._r_origin[live],
                                      self._r_az[live], self._r_el[live], self._r_tlast[live], t,
                                      self.target_range)]
        if gone.size == 0:
            return
        self._r_res[gone] = True
        for idx in gone:
            self._ray_key.pop(self._r_keyof[int(idx)], None)
        if self.n_rays > 512 and self._r_res[:self.n_rays].mean() > 0.5:
            self.compact()

    def _extract_frontiers(self) -> None:
        self.frontier_mask, self.frontier_clusters = self.frontiers.extract(self.observed,
                                                                           self._seen)

    # ---- candidate lists ----------------------------------------------------------------------
    def _extract_ray_targets(self, t: float) -> None:
        """Every live ray, newest first, capped by the token count. No similarity, no diversity
        pass, no merging: two rays that happen to point at one place stay two tokens."""
        self._ray_targets = []
        n = self.n_rays
        if n == 0:
            return
        live = np.flatnonzero(~self._r_res[:n])
        if live.size == 0:
            return
        k_max = int(CAND_POOL * max(1, self.cfg.tokens.k_ray))
        order = live[np.argsort(-self._r_ids[live], kind="stable")[:k_max]]
        rng_m = self.target_range(self._r_el[order])
        x0, y0, x1, y1 = self.raster.region
        eps = 1e-6
        out: list[RayTarget] = []
        for m, idx in enumerate(order):
            idx = int(idx)
            a = float(self._r_az[idx])
            tx = self._r_origin[idx, 0] + math.cos(a) * float(rng_m[m])
            ty = self._r_origin[idx, 1] + math.sin(a) * float(rng_m[m])
            out.append(RayTarget(
                id=int(self._r_ids[idx]), ray_idx=idx,
                xy=np.array([min(max(tx, x0 + eps), x1 - eps), min(max(ty, y0 + eps), y1 - eps)]),
                origin_xy=self._r_origin[idx].copy(), az=a, el=float(self._r_el[idx]),
                range_m=float(rng_m[m]), feat=self._r_peak[idx].copy(),
                feat_mean=self._r_feat[idx].copy(),
                conf=float(self._r_conf[idx]), n_obs=int(self._r_nobs[idx]),
                t_first=float(self._r_tfirst[idx]), t_last=float(self._r_tlast[idx])))
        self._ray_targets = out

    def _needs_resegment(self, n_obs: int, decision: int) -> bool:
        return self.segidx.needs_resegment(n_obs, decision)

    def _extract_segments(self, t: float, decision: int | None = None) -> None:
        """Re-segment when enough of the map is new, then refresh every segment's statistics.

        The labels are the expensive part and so are the things that depend only on them (cell
        counts, medoids, stable ids); the feature, hit and ray statistics change every decision and
        are one sweep. They run on different clocks (see `SegmentIndex`).
        """
        dec = (int(round(self._t_dec / max(self.cfg.decision_dt, 1e-9))) if decision is None
               else decision)
        live = np.flatnonzero(~self._r_res[:self.n_rays])
        org = np.ascontiguousarray(self._r_origin[live])
        az = np.ascontiguousarray(self._r_az[live])
        self._segments = self.segidx.extract(self.vox_feat_sum, self.observed, self.vox_cnt,
                                             self.last_seen_t, org, az, t, dec)

    def _resegment(self, t: float, scale: float, min_cells: int) -> None:
        self.segidx.resegment(self.vox_feat_sum, self.observed, self.vox_cnt, self.last_seen_t,
                              t, scale, min_cells)

    @property
    def ray_targets(self) -> list[RayTarget]:
        return self._ray_targets

    @property
    def segments(self) -> list[SegmentToken]:
        return self._segments


def _cos(f: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(f, axis=-1)
    return (f @ q) / np.maximum(n, 1e-12)


def rf_visual(cfg) -> float:
    return float(cfg.sensor.visual_range_m)


def _range_factor(r, far_factor: float, depth: float) -> np.ndarray:
    """1 within half the depth limit, linear to 0.5 at the depth limit, then times far_factor."""
    half = 0.5 * depth
    f = np.where(r <= half, 1.0, 1.0 - 0.5 * (r - half) / max(half, 1e-9))
    return np.where(r > depth, 0.5 * far_factor, np.clip(f, 0.5, 1.0))


def _disc_mask(radius_m: float, cell_m: float) -> np.ndarray:
    rad = int(math.ceil(radius_m / cell_m))
    a = np.arange(-rad, rad + 1)
    return (a[:, None] ** 2 + a[None, :] ** 2) * cell_m * cell_m <= radius_m * radius_m


@njit(cache=True)
def _disc_counts(grid, disc, ci, cj):
    """Count of set cells of `grid` inside a disc window around each (ci, cj)."""
    n = ci.shape[0]
    ny, nx = grid.shape
    rad = disc.shape[0] // 2
    out = np.zeros(n, np.int64)
    for k in range(n):
        c = 0
        for a in range(-rad, rad + 1):
            i = ci[k] + a
            if i < 0 or i >= ny:
                continue
            for b in range(-rad, rad + 1):
                j = cj[k] + b
                if j < 0 or j >= nx or not disc[a + rad, b + rad]:
                    continue
                if grid[i, j]:
                    c += 1
        out[k] = c
    return out


__all__ = ["RayFrontsSim"]
