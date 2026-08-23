"""Token / observation construction (CONTRACTS.md 5).

Slot order per robot: `[hold] + k_frontier + k_ray + k_segment + k_visited` — the RayFronts topics
plus "stay put" plus the team's visited-target records, newest-first inside each block and with no
priority between them.

**Open-set observation.** A token's semantics is the item's own feature vector `feat[D]`, never a
row of query scores; the mission queries arrive separately as *query tokens* (`TeamObs.query_emb` /
`query_w` / `query_mask`) for the policy to attend over. The same holds for the rasters: the BEV
and the local crop carry the features projected onto a fixed PCA basis of the embedding table's
class vectors, so every env, scene and viewer reads the same axes.

**CTDE split.** `build` takes one `RobotView` per robot — what that robot knows — and produces the
per-robot tokens, the ego-centric `local` crop, its own `robot_bev` and its `peer_tokens`. The
compressed global `bev` is the centralised critic's input. Under `comms.mode == "full"` every robot
is handed the same team view; under `"range"` every per-robot output — the peer columns of a token
included — is a function of that robot's belief and peer cache alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numba import njit

from .embeddings import FEAT_PC_DIM
from .state import (F_AGE, F_AZ_COS, F_AZ_SIN, F_BCOS, F_BSIN, F_CLAIM, F_CONF, F_COV, F_DIST,
                    F_DX, F_DY, F_EL_COS, F_EL_SIN, F_FEAT0, F_FOUND, F_HITS, F_NOBS, F_ORIGIN_DX,
                    F_ORIGIN_DY, F_OWN, F_PEER, F_RANGE, F_RAYS, F_REACH, F_SIZE, F_WHO, F_XABS,
                    F_YABS, PEER_AGE, PEER_COV, PEER_DIST, PEER_DX, PEER_DY, PEER_FEAT_DIM,
                    PEER_LINK, PEER_TDX, PEER_TDY, PEER_TYPE0, PEER_VALID, ROBOT_FEAT_DIM,
                    N_TOKEN_TYPES, TOKEN_FIXED, TOKEN_FRONTIER, TOKEN_HOLD, TOKEN_RAY, TOKEN_SEGMENT,
                    TOKEN_VISITED, TeamObs, token_feature_names)

NBHD_M = 20.0             # coverage / unobserved-fraction neighbourhood of a token target
FEAT_RADIUS_M = 10.0      # a frontier (or hold) token averages the features it can already see
CLAIM_M = 15.0
RAY_BEV_NORM = 8.0        # ray count per raster cell that saturates the "ray_count" channel
SEG_RAY_NORM = 8.0        # rays crossing a segment that saturate the token's "ray_count"
NOBS_NORM = math.log1p(500.0)   # observation counts are heavy-tailed: log1p(n) / log1p(500)
HITS_NORM = math.log1p(50.0)

BEV_CHANNELS = (("known", "hits", "frontier", "robots", "peer_targets", "ray_count")
                + tuple(f"feat_pc{i}" for i in range(FEAT_PC_DIM))
                + tuple(f"ray_feat_pc{i}" for i in range(FEAT_PC_DIM))
                + ("visited",))
LOCAL_CHANNELS = (("known", "hits", "ray_count")
                  + tuple(f"feat_pc{i}" for i in range(FEAT_PC_DIM))
                  + ("visited",))
VISIT_SIG = 1.2           # blob radius (in target cells) of a visited record in a raster stack
FOUND_NORM = 4.0          # casualties per visit that saturate the token's "visit_found"


@njit(cache=True)
def _disc_stats(known, hits, disc, ci, cj):
    """(observed fraction, mean hit count over observed cells) in a disc around each (ci, cj)."""
    n = ci.shape[0]
    ny, nx = known.shape
    rad = disc.shape[0] // 2
    cov = np.zeros(n, np.float32)
    hv = np.zeros(n, np.float32)
    for k in range(n):
        tot = 0
        seen = 0
        acc = 0.0
        for a in range(-rad, rad + 1):
            i = ci[k] + a
            if i < 0 or i >= ny:
                continue
            for b in range(-rad, rad + 1):
                if not disc[a + rad, b + rad]:
                    continue
                j = cj[k] + b
                if j < 0 or j >= nx:
                    continue
                tot += 1
                if known[i, j]:
                    seen += 1
                    acc += hits[i, j]
        if tot > 0:
            cov[k] = seen / tot
        if seen > 0:
            hv[k] = acc / seen
    return cov, hv


@njit(cache=True, fastmath=True)
def _disc_feat_mean(feat_sum, known, disc, ci, cj, out):
    """Mean of the *unit* cell features over the observed cells of a disc; zeros if none."""
    n = ci.shape[0]
    ny, nx, d_n = feat_sum.shape
    rad = disc.shape[0] // 2
    for k in range(n):
        cnt = 0
        for a in range(-rad, rad + 1):
            i = ci[k] + a
            if i < 0 or i >= ny:
                continue
            for b in range(-rad, rad + 1):
                if not disc[a + rad, b + rad]:
                    continue
                j = cj[k] + b
                if j < 0 or j >= nx or not known[i, j]:
                    continue
                s = 0.0
                for t in range(d_n):
                    v = feat_sum[i, j, t]
                    s += v * v
                if s <= 1e-24:
                    continue
                inv = 1.0 / np.sqrt(s)
                for t in range(d_n):
                    out[k, t] += feat_sum[i, j, t] * inv
                cnt += 1
        if cnt > 0:
            s = 0.0
            for t in range(d_n):
                s += out[k, t] * out[k, t]
            inv = 1.0 / np.sqrt(s) if s > 1e-24 else 0.0
            for t in range(d_n):
                out[k, t] *= inv
    return


@njit(cache=True, fastmath=True)
def _ray_raster(org, az, pcs, r0, r1, step, ox, oy, sx, sy, cnt, fsum):
    """March every live ray between the depth limit and the visual range into a grid.

    `cnt` accumulates crossings per cell and `fsum` the rays' PCA-projected features, so the caller
    can divide for the mean. It is a picture of the raw ray topic — where rays *pass*, not where
    the sim thinks they intersect.
    """
    ni, nj = cnt.shape
    p = fsum.shape[2]
    for k in range(org.shape[0]):
        ca = math.cos(az[k])
        sa = math.sin(az[k])
        r = r0
        while r <= r1:
            u = int((org[k, 0] + ca * r - ox) * sx)
            v = int((org[k, 1] + sa * r - oy) * sy)
            r += step
            if u < 0 or u >= nj or v < 0 or v >= ni:
                continue
            cnt[v, u] += 1.0
            for t in range(p):
                fsum[v, u, t] += pcs[k, t]
    return


def _disc(radius_m: float, cell_m: float) -> np.ndarray:
    rad = int(math.ceil(radius_m / cell_m))
    a = np.arange(-rad, rad + 1)
    return (a[:, None] ** 2 + a[None, :] ** 2) * cell_m * cell_m <= radius_m * radius_m


@dataclass
class RobotView:
    """What one robot knows about the world, as the token/raster builders consume it.

    Every field is per-robot by contract. `comms.mode == "full"` hands every robot the same
    instance (`team_view`); under `comms.mode == "range"` `sim/comms.py` hands each robot its own
    masks, frontier/segment/ray lists, peer cache and visited records, and the builders do not
    change. `feat_known` (default: `known`) is the part of `known` that carries a feature — a cell
    a peer only reported as *covered* is known but featureless, so it suppresses a frontier without
    contributing hits or semantics.
    """
    known: np.ndarray            # bool    [ny, nx] cells this robot has (or was told) about
    feat_sum: np.ndarray         # float32 [ny, nx, D]
    hits: np.ndarray             # int32   [ny, nx]
    last_seen: np.ndarray        # float32 [ny, nx]
    frontier_mask: np.ndarray    # bool    [ny, nx]
    frontiers: Sequence[Any]     # FrontierCluster, newest first
    rays: Sequence[Any]          # RayTarget, newest first
    segments: Sequence[Any]      # SegmentToken, newest first
    ray_store: Any               # RayStore, for the ray rasters
    feat_known: np.ndarray | None = None   # bool [ny, nx]; None = `known`
    visited: Sequence[Any] = ()            # VisitRecord, newest first
    peers: dict | None = None              # {robot: PeerRecord}; None = full comms (live peers)
    id_offset: int = 0                     # added to every token id (per-robot id namespace)
    coverage: float | None = None          # this robot's own coverage fraction (None = team's)
    robot: int = -1
    seg_labels: np.ndarray | None = None   # int32 [ny, nx] this robot's segmentation (viewers)

    @property
    def fknown(self) -> np.ndarray:
        return self.known if self.feat_known is None else self.feat_known


def team_view(rf, visited: Sequence[Any] = ()) -> RobotView:
    """The one shared belief every robot sees while `comms.mode == "full"`."""
    return RobotView(known=rf.observed, feat_sum=rf.vox_feat_sum, hits=rf.vox_cnt,
                     last_seen=rf.last_seen_t, frontier_mask=rf.frontier_mask,
                     frontiers=rf.frontier_clusters, rays=rf.ray_targets, segments=rf.segments,
                     ray_store=rf.store(), visited=visited, seg_labels=rf.seg_labels)


class TokenBuilder:
    """Holds the per-scene scratch (disc masks, BEV index maps) so building an obs allocates little."""

    def __init__(self, raster, cfg, emb):
        self.raster = raster
        self.cfg = cfg
        self.emb = emb
        self.D = int(emb.D)
        self.F = TOKEN_FIXED + self.D
        self.K = cfg.k_tokens
        self.names = token_feature_names(self.D)
        self.nbhd = _disc(NBHD_M, raster.cell_m)
        self.fdisc = _disc(FEAT_RADIUS_M, raster.cell_m)
        self.pc_mean, self.pc_comps = emb.pc_basis(FEAT_PC_DIM)
        b = cfg.tokens.bev_size
        self.bev_i, self.bev_j = self._bev_idx(b)
        self.robot_bev_size = int(cfg.tokens.robot_bev_size)
        self.rbev_i, self.rbev_j = self._bev_idx(self.robot_bev_size)
        self.local_size = int(cfg.tokens.local_size)
        self.diag = raster.diagonal_m
        self.global_view: RobotView | None = None   # set per decision under range comms (critic)

    def _bev_idx(self, b: int):
        r = self.raster
        if b <= 0:
            return None, None
        return (np.clip((np.arange(b) + 0.5) * r.ny / b, 0, r.ny - 1).astype(np.int64),
                np.clip((np.arange(b) + 0.5) * r.nx / b, 0, r.nx - 1).astype(np.int64))

    # -- main -------------------------------------------------------------------------------
    def build(self, rf, robots, t: float, planner, views: Sequence[RobotView] | None = None,
              queries=None) -> TeamObs:
        """Per-robot tokens + local crops, the shared query block and the global critic BEV.

        `views[i]` is robot `i`'s knowledge; `None` gives every robot the team view (comms full).
        Everything that does not depend on the robot is built once per distinct view (`_pack`);
        the per-robot pass only rewrites the ego columns.
        """
        cfg, ras = self.cfg, self.raster
        n_r = len(robots)
        shared = None
        if views is None:      # full comms: the team belief, visited records included
            shared = self.global_view if self.global_view is not None else team_view(rf)
            views = [shared] * n_r
        elif len(views) != n_r:
            raise ValueError(f"TokenBuilder.build: {len(views)} views for {n_r} robots")
        K, F = self.K, self.F
        tok = np.zeros((n_r, K, F), np.float32)
        mask = np.zeros((n_r, K), np.bool_)
        xy = np.full((n_r, K, 2), np.nan, np.float32)
        ttype = np.zeros((n_r, K), np.int8)
        tid = np.full((n_r, K), -1, np.int32)

        packs: dict[int, dict] = {}
        rows: dict[int, list[int]] = {}       # view -> the robots that hold it
        for ri, v in enumerate(views):
            rows.setdefault(id(v), []).append(ri)
            if id(v) not in packs:
                packs[id(v)] = self._pack(v, t, planner)

        tc = cfg.tokens
        rxy = np.array([[r.pos[0], r.pos[1]] for r in robots], np.float64)
        peer_geo = self._peer_geometry(views, robots, rxy)
        ri_i, ri_j = ras.xy_to_ij(rxy[:, 0], rxy[:, 1])
        ri_i = np.clip(ri_i, 0, ras.ny - 1).astype(np.int64)
        ri_j = np.clip(ri_j, 0, ras.nx - 1).astype(np.int64)
        r_lab = planner.labels[ri_i, ri_j]
        head = np.array([r.heading for r in robots], np.float64)
        slot0 = 1 if tc.include_hold else 0
        x0, y0, x1, y1 = ras.region
        diag = self.diag

        if tc.include_hold:
            hcov = np.zeros(n_r, np.float32)
            hhits = np.zeros(n_r, np.float32)
            hfeat = np.zeros((n_r, self.D), np.float32)
            for vid, ridx in rows.items():        # the hold token reads the robot's own belief
                sel = np.asarray(ridx, np.int64)
                v = views[ridx[0]]
                hcov[sel], hhits[sel] = _cov_hits(v, self.nbhd, ri_i[sel], ri_j[sel])
                f = np.zeros((sel.size, self.D), np.float32)
                _disc_feat_mean(v.feat_sum, v.fknown, self.fdisc, ri_i[sel], ri_j[sel], f)
                hfeat[sel] = f
            tok[:, 0, TOKEN_HOLD] = 1.0
            tok[:, 0, F_BCOS] = 1.0              # the hold target is the robot itself
            tok[:, 0, F_XABS] = 2.0 * (rxy[:, 0] - x0) / (x1 - x0) - 1.0
            tok[:, 0, F_YABS] = 2.0 * (rxy[:, 1] - y0) / (y1 - y0) - 1.0
            tok[:, 0, F_COV] = hcov
            tok[:, 0, F_HITS] = np.clip(np.log1p(hhits) / HITS_NORM, 0.0, 1.0)
            tok[:, 0, F_PEER] = [_nearest(peer_geo[ri][0], rxy[ri], diag) for ri in range(n_r)]
            tok[:, 0, F_REACH] = 1.0
            tok[:, 0, F_FEAT0:] = hfeat
            mask[:, 0] = True
            xy[:, 0] = rxy
            ttype[:, 0] = TOKEN_HOLD

        for ri in range(n_r):
            p = packs[id(views[ri])]
            n_c = p["n"]
            if n_c == 0:
                continue
            sl = p["slot"] + slot0
            cxy = p["xy"]
            tok[ri, sl] = p["base"]
            dx = cxy[:, 0] - rxy[ri, 0]
            dy = cxy[:, 1] - rxy[ri, 1]
            dist = np.hypot(dx, dy)
            br = np.arctan2(dy, dx) - head[ri]
            tok[ri, sl, F_DX] = dx / diag
            tok[ri, sl, F_DY] = dy / diag
            tok[ri, sl, F_DIST] = dist / diag
            tok[ri, sl, F_BSIN] = np.sin(br)
            tok[ri, sl, F_BCOS] = np.cos(br)
            org = p["origin"]
            tok[ri, sl, F_ORIGIN_DX] = np.where(p["is_ray"], (org[:, 0] - rxy[ri, 0]) / diag, 0.0)
            tok[ri, sl, F_ORIGIN_DY] = np.where(p["is_ray"], (org[:, 1] - rxy[ri, 1]) / diag, 0.0)
            ppos, ptgt = peer_geo[ri]
            if ppos.shape[0]:
                pd = np.hypot(cxy[None, :, 0] - ppos[None, :, 0].T,
                              cxy[None, :, 1] - ppos[None, :, 1].T).min(0)
                tok[ri, sl, F_PEER] = np.minimum(pd / diag, 1.0)
            else:
                tok[ri, sl, F_PEER] = 1.0       # nobody it knows of is anywhere near
            if ptgt.shape[0]:
                td = np.hypot(cxy[None, :, 0] - ptgt[None, :, 0].T,
                              cxy[None, :, 1] - ptgt[None, :, 1].T).min(0)
                tok[ri, sl, F_CLAIM] = (td <= CLAIM_M).astype(np.float32)
            tok[ri, sl, F_OWN] = (p["who"] == int(robots[ri].idx))
            reach = (r_lab[ri] != 0) & (p["lab"] == r_lab[ri])
            tok[ri, sl, F_REACH] = reach
            mask[ri, sl] = reach
            xy[ri, sl] = cxy
            ttype[ri, sl] = p["type"]
            tid[ri, sl] = p["id"]

        qe, qw, qm = self._query_block(rf if queries is None else queries)
        gv = shared if shared is not None else self.global_view
        if gv is None:
            gv = team_view(rf)
        return TeamObs(tokens=tok, token_mask=mask, token_xy=xy, token_type=ttype, token_id=tid,
                       robot_feat=self._robot_feat(robots, rf, t, views),
                       bev=self._bev(gv, robots, self.cfg.tokens.bev_size, self.bev_i, self.bev_j),
                       query_emb=qe, query_w=qw, query_mask=qm,
                       t=float(t), local=self._local(views, robots, ri_i, ri_j),
                       peer_tokens=self._peer_tokens(views, robots, t, self._team_cover(rf)),
                       robot_bev=self._robot_bev(views, robots))

    # -- pieces -----------------------------------------------------------------------------
    def _pack(self, v: RobotView, t: float, planner) -> dict:
        """Everything about one view's candidates that does not depend on which robot is asking:
        slot, target, type one-hot, size/age/confidence, the ray geometry and the item feature."""
        ras, tc = self.raster, self.cfg.tokens
        cfg = self.cfg
        fr = list(v.frontiers[: tc.k_frontier])
        rays = list(v.rays[: tc.k_ray])
        segs = list(v.segments[: tc.k_segment])
        vis = list(v.visited[: tc.k_visited])
        obj = fr + rays + segs + vis
        n_c = len(obj)
        n_fr, n_ray, n_seg = len(fr), len(rays), len(segs)
        base = np.zeros((n_c, self.F), np.float32)
        xy = np.zeros((n_c, 2), np.float64)
        origin = np.zeros((n_c, 2), np.float64)
        slot = np.zeros(n_c, np.int64)
        types = np.zeros(n_c, np.int8)
        ids = np.full(n_c, -1, np.int32)
        is_ray = np.zeros(n_c, np.bool_)
        who = np.full(n_c, -1, np.int64)      # which robot made a visited record (-1 = not one)
        for i, o in enumerate(obj):
            if i < n_fr:
                kind, slot[i] = TOKEN_FRONTIER, i
                xy[i] = o.centroid_xy
            elif i < n_fr + n_ray:
                kind, slot[i] = TOKEN_RAY, tc.k_frontier + (i - n_fr)
                xy[i] = o.xy
                origin[i] = o.origin_xy
                is_ray[i] = True
            elif i < n_fr + n_ray + n_seg:
                kind = TOKEN_SEGMENT
                slot[i] = tc.k_frontier + tc.k_ray + (i - n_fr - n_ray)
                xy[i] = o.xy
            else:
                kind = TOKEN_VISITED
                slot[i] = (tc.k_frontier + tc.k_ray + tc.k_segment
                           + (i - n_fr - n_ray - n_seg))
                xy[i] = o.xy
            types[i] = kind
            ids[i] = int(o.id) + int(v.id_offset)
            base[i, kind] = 1.0
        if n_c == 0:
            return {"n": 0}
        ci, cj = ras.xy_to_ij(xy[:, 0], xy[:, 1])
        ci = np.clip(ci, 0, ras.ny - 1).astype(np.int64)
        cj = np.clip(cj, 0, ras.nx - 1).astype(np.int64)
        cov, hits = _cov_hits(v, self.nbhd, ci, cj)
        x0, y0, x1, y1 = ras.region
        base[:, F_XABS] = 2.0 * (xy[:, 0] - x0) / (x1 - x0) - 1.0
        base[:, F_YABS] = 2.0 * (xy[:, 1] - y0) / (y1 - y0) - 1.0
        base[:, F_COV] = cov
        base[:, F_HITS] = np.clip(np.log1p(hits) / HITS_NORM, 0.0, 1.0)
        ttl = float(cfg.rayfronts.ray_ttl_s)
        cap = float(cfg.rayfronts.ray_conf_cap)
        if n_fr:
            f = slice(0, n_fr)
            base[f, F_SIZE] = np.clip([o.n_cells / 200.0 for o in fr], 0.0, 1.0)
            base[f, F_AGE] = np.clip((t - v.last_seen[ci[f], cj[f]]) / ttl, 0.0, 1.0)
            _disc_feat_mean(v.feat_sum, v.fknown, self.fdisc, ci[f], cj[f], base[f, F_FEAT0:])
        if n_ray:
            r = slice(n_fr, n_fr + n_ray)
            az = np.array([o.az for o in rays], np.float64)
            el = np.array([o.el for o in rays], np.float64)
            base[r, F_CONF] = np.clip([o.conf / cap for o in rays], 0.0, 1.0)
            base[r, F_AGE] = np.clip([(t - o.t_last) / ttl for o in rays], 0.0, 1.0)
            base[r, F_NOBS] = np.clip([math.log1p(o.n_obs) / NOBS_NORM for o in rays], 0.0, 1.0)
            # raw ray geometry: bearing, elevation and the range that elevation implies. Two rays
            # crossing is something the policy reads off these, not something the sim decides.
            base[r, F_AZ_SIN], base[r, F_AZ_COS] = np.sin(az), np.cos(az)
            base[r, F_EL_SIN], base[r, F_EL_COS] = np.sin(el), np.cos(el)
            base[r, F_RANGE] = [o.range_m / self.diag for o in rays]
            base[r, F_FEAT0:] = np.array([o.feat for o in rays], np.float32)
        if n_seg:
            g = slice(n_fr + n_ray, n_fr + n_ray + n_seg)
            base[g, F_SIZE] = np.clip([o.n_cells / 200.0 for o in segs], 0.0, 1.0)
            base[g, F_AGE] = np.clip([(t - o.t_last) / ttl for o in segs], 0.0, 1.0)
            base[g, F_RAYS] = np.clip([o.ray_count / SEG_RAY_NORM for o in segs], 0.0, 1.0)
            base[g, F_FEAT0:] = np.array([o.feat for o in segs], np.float32)
        if vis:
            w = slice(n_fr + n_ray + n_seg, n_c)
            # a visited record is the team's own bookkeeping: when it was made, what the token that
            # was flown to looked like, how many casualties it turned up and whose visit it was
            base[w, F_AGE] = np.clip([(t - o.t) / ttl for o in vis], 0.0, 1.0)
            base[w, F_FOUND] = np.clip([o.n_found / FOUND_NORM for o in vis], 0.0, 1.0)
            base[w, F_WHO] = [(int(o.robot) + 1) / 10.0 for o in vis]
            who[w] = [int(o.robot) for o in vis]      # F_OWN is per robot: filled by `build`
            base[w, F_FEAT0:] = np.array([o.feat for o in vis], np.float32)
        return {"n": n_c, "base": base, "xy": xy, "origin": origin, "slot": slot, "type": types,
                "id": ids, "is_ray": is_ray, "who": who, "lab": planner.labels[ci, cj],
                "ci": ci, "cj": cj}

    def _query_block(self, src) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(query_emb [Qmax, D], query_w [Qmax], query_mask [Qmax])`, zero-padded.

        The mission queries are an *input to the policy*. Their weight is 1.0 today; an LLM hint
        appended later carries its own likelihood there.
        """
        qmax = int(self.cfg.tokens.max_queries)
        qe = np.zeros((qmax, self.D), np.float32)
        qw = np.zeros(qmax, np.float32)
        qm = np.zeros(qmax, np.bool_)
        emb = np.asarray(getattr(src, "query_emb", src), np.float32).reshape(-1, self.D)
        w = np.asarray(getattr(src, "query_w", np.ones(emb.shape[0], np.float32)), np.float32)
        n = emb.shape[0]
        if n > qmax:
            raise ValueError(f"_query_block: {n} queries exceed tokens.max_queries={qmax}")
        qe[:n] = emb
        qw[:n] = w[:n]
        qm[:n] = True
        return qe, qw, qm

    def _robot_feat(self, robots, rf, t: float, views=None) -> np.ndarray:
        """The `coverage` column is the *robot's own* observed fraction under range comms (each
        robot only knows how much it has covered), and the team's under full comms."""
        cfg, ras = self.cfg, self.raster
        x0, y0, x1, y1 = ras.region
        out = np.zeros((len(robots), ROBOT_FEAT_DIM), np.float32)
        team_cover = float(rf.observed.sum()) / max(1, int(rf.observable.sum()))
        tf = t / max(cfg.t_max_s, 1e-9)
        for i, rb in enumerate(robots):
            cv = None if views is None else views[i].coverage
            cover = team_cover if cv is None else float(cv)
            out[i, 0] = 2.0 * (rb.pos[0] - x0) / (x1 - x0) - 1.0
            out[i, 1] = 2.0 * (rb.pos[1] - y0) / (y1 - y0) - 1.0
            out[i, 2] = rb.alt / 100.0
            out[i, 3] = math.sin(rb.heading)
            out[i, 4] = math.cos(rb.heading)
            out[i, 5] = tf
            out[i, 6] = 1.0 - tf
            out[i, 7] = cover
            if rb.idx < 10:
                out[i, 8 + rb.idx] = 1.0
        return out

    def _feat_pc(self, feat_sum: np.ndarray, known: np.ndarray) -> np.ndarray:
        """[..., FEAT_PC_DIM] projection of a feature grid, zeroed where there is no feature.

        `known` here is the *feature*-known mask: a cell a peer only reported as covered has no
        feature, so its channels read exactly like unknown ground while `known` still reads 1.
        """
        n = np.linalg.norm(feat_sum, axis=-1, keepdims=True)
        unit = feat_sum / np.maximum(n, 1e-12)
        pc = (unit - self.pc_mean) @ self.pc_comps.T
        return (pc * (known & (n[..., 0] > 1e-12))[..., None]).astype(np.float32)

    def _bev(self, v: RobotView, robots, b: int, bi, bj) -> np.ndarray:
        """Compressed BEV of one belief. `bev_size` over the union belief is the centralised
        critic's picture; `robot_bev_size` over a robot's own belief is the actor's."""
        ras = self.raster
        sub = np.ix_(bi, bj)
        out = np.zeros((len(BEV_CHANNELS), b, b), np.float32)
        known = v.known[sub]
        fknown = v.fknown[sub]
        out[0] = known
        out[1] = np.minimum(np.log1p(v.hits[sub].astype(np.float32)) / HITS_NORM, 1.0) * fknown
        out[2] = v.frontier_mask[sub]
        x0, y0, x1, y1 = ras.region

        def stamp(ch, x, y, amp=1.0, sig=1.5):
            u = (x - x0) / (x1 - x0) * b
            w = (y - y0) / (y1 - y0) * b
            i0, i1 = max(0, int(w - 3 * sig)), min(b, int(w + 3 * sig) + 1)
            j0, j1 = max(0, int(u - 3 * sig)), min(b, int(u + 3 * sig) + 1)
            if i1 <= i0 or j1 <= j0:
                return
            ii = np.arange(i0, i1) + 0.5 - w
            jj = np.arange(j0, j1) + 0.5 - u
            g = np.exp(-(ii[:, None] ** 2 + jj[None, :] ** 2) / (2 * sig * sig)) * amp
            np.maximum(out[ch, i0:i1, j0:j1], g, out=out[ch, i0:i1, j0:j1])

        if v.peers is None:                     # full comms: every robot, live
            for rb in robots:
                stamp(3, rb.pos[0], rb.pos[1])
                if rb.target_xy is not None:
                    stamp(4, rb.target_xy[0], rb.target_xy[1], 1.0, 1.0)
        else:                                   # what this robot knows: itself + its peer cache
            me = robots[v.robot] if 0 <= v.robot < len(robots) else None
            if me is not None:
                stamp(3, me.pos[0], me.pos[1])
                if me.target_xy is not None:
                    stamp(4, me.target_xy[0], me.target_xy[1], 1.0, 1.0)
            for pr in v.peers.values():
                stamp(3, pr.pos[0], pr.pos[1])
                if pr.target_xy is not None:
                    stamp(4, pr.target_xy[0], pr.target_xy[1], 1.0, 1.0)
        for rec in v.visited:
            stamp(len(BEV_CHANNELS) - 1, rec.xy[0], rec.xy[1], 1.0, VISIT_SIG)
        pc = self._feat_pc(v.feat_sum[sub], fknown)
        out[6: 6 + FEAT_PC_DIM] = np.moveaxis(pc, -1, 0)
        self._ray_channels(v, out[5], out[6 + FEAT_PC_DIM: 6 + 2 * FEAT_PC_DIM], x0, y0,
                           b / (x1 - x0), b / (y1 - y0), 0.5 * min(x1 - x0, y1 - y0) / b)
        return out

    def _robot_bev(self, views: Sequence[RobotView], robots) -> np.ndarray | None:
        """One BEV per robot over its own belief — the actor's global picture (CTDE)."""
        b = self.robot_bev_size
        if b <= 0:
            return None
        out = np.zeros((len(robots), len(BEV_CHANNELS), b, b), np.float32)
        for r in range(len(robots)):
            out[r] = self._bev(views[r], robots, b, self.rbev_i, self.rbev_j)
        return out

    def _team_cover(self, rf) -> float:
        return float(rf.observed.sum()) / max(1, int(rf.observable.sum()))

    def _peer_geometry(self, views: Sequence[RobotView], robots, rxy):
        """Per robot, where it *believes* the other robots and their targets are.

        `peer_dist_min` and `claimed_by_peer` are the robot's own view of the team, so under range
        comms they come from its peer cache (position and target as of the last contact) and not
        from the live robot list: a robot that has never heard from anyone must not read their
        true positions off the observation. Under full comms the cache is `None` and the live team
        *is* what every robot knows.
        """
        n_r = len(robots)
        out = []
        for ri in range(n_r):
            v = views[ri]
            if v.peers is None:
                pos = [rxy[j] for j in range(n_r) if j != ri]
                tgt = [robots[j].target_xy for j in range(n_r)
                       if j != ri and robots[j].target_xy is not None]
            else:
                pos = [pr.pos for pr in v.peers.values()]
                tgt = [pr.target_xy for pr in v.peers.values() if pr.target_xy is not None]
            out.append((np.asarray(pos, np.float64).reshape(-1, 2),
                        np.asarray(tgt, np.float64).reshape(-1, 2)))
        return out

    def _peer_tokens(self, views: Sequence[RobotView], robots, t: float,
                     team_cover: float = 0.0) -> np.ndarray:
        """One token per peer: where it was, what it was going to, how stale that is, how much it
        says it has covered, and whether the link is up right now.

        Under full comms the cache is `None` and every peer reads as in contact with age 0.
        """
        n_r = len(robots)
        out = np.zeros((n_r, max(n_r - 1, 0), PEER_FEAT_DIM), np.float32)
        if n_r < 2:
            return out
        diag, tmax = self.diag, max(float(self.cfg.t_max_s), 1e-9)
        for ri in range(n_r):
            v = views[ri]
            me = robots[ri].pos
            for c, rj in enumerate(j for j in range(n_r) if j != ri):
                if v.peers is None:
                    rb = robots[rj]
                    pos, tgt, ttype = rb.pos, rb.target_xy, int(rb.target_token_type)
                    age, cov, link = 0.0, team_cover, 1.0
                else:
                    pr = v.peers.get(rj)
                    if pr is None:
                        continue                  # never heard from: the slot stays zero/invalid
                    pos, tgt, ttype = pr.pos, pr.target_xy, int(pr.target_type)
                    age = (t - pr.t_contact) / tmax
                    cov, link = float(pr.coverage), float(pr.linked)
                dx, dy = pos[0] - me[0], pos[1] - me[1]
                row = [0.0] * PEER_FEAT_DIM
                row[PEER_DX], row[PEER_DY] = dx / diag, dy / diag
                row[PEER_DIST] = min(math.hypot(dx, dy) / diag, 1.0)
                if tgt is not None and math.isfinite(tgt[0]) and math.isfinite(tgt[1]):
                    row[PEER_TDX] = (tgt[0] - me[0]) / diag
                    row[PEER_TDY] = (tgt[1] - me[1]) / diag
                row[PEER_AGE] = min(max(age, 0.0), 1.0)
                row[PEER_COV] = min(max(cov, 0.0), 1.0)
                row[PEER_LINK] = link
                row[PEER_VALID] = 1.0
                if 0 <= ttype < N_TOKEN_TYPES:
                    row[PEER_TYPE0 + ttype] = 1.0
                out[ri, c] = row
        return out

    def _local(self, views: Sequence[RobotView], robots, ri_i, ri_j) -> np.ndarray | None:
        """Ego-centric crop of each robot's own belief at raster resolution (axis aligned).

        The dense near-field the actor plans in: what is known, how often it was looked at, the
        feature PCs, the rays crossing it and the visited records inside it. Out-of-region cells
        stay zero (= unknown).
        """
        s = self.local_size
        n_r = len(robots)
        if s <= 0:
            return None
        ras = self.raster
        out = np.zeros((n_r, len(LOCAL_CHANNELS), s, s), np.float32)
        half = s // 2
        for r, rb in enumerate(robots):
            v = views[r]
            i0, j0 = int(ri_i[r]) - half, int(ri_j[r]) - half
            si0, sj0 = max(0, -i0), max(0, -j0)
            gi0, gj0 = max(0, i0), max(0, j0)
            gi1, gj1 = min(ras.ny, i0 + s), min(ras.nx, j0 + s)
            if gi1 > gi0 and gj1 > gj0:
                h, w = gi1 - gi0, gj1 - gj0
                known = v.known[gi0:gi1, gj0:gj1]
                fknown = v.fknown[gi0:gi1, gj0:gj1]
                out[r, 0, si0:si0 + h, sj0:sj0 + w] = known
                out[r, 1, si0:si0 + h, sj0:sj0 + w] = fknown * np.minimum(
                    np.log1p(v.hits[gi0:gi1, gj0:gj1].astype(np.float32)) / HITS_NORM, 1.0)
                pc = self._feat_pc(v.feat_sum[gi0:gi1, gj0:gj1], fknown)
                out[r, 3:3 + FEAT_PC_DIM, si0:si0 + h, sj0:sj0 + w] = np.moveaxis(pc, -1, 0)
            ox = ras.origin[0] + j0 * ras.cell_m      # world -> local cell index
            oy = ras.origin[1] + i0 * ras.cell_m
            inv = 1.0 / ras.cell_m
            self._ray_channels(v, out[r, 2], None, ox, oy, inv, inv, ras.cell_m)
            _stamp_visits(out[r, len(LOCAL_CHANNELS) - 1], v.visited, ox, oy, inv, VISIT_SIG)
        return out

    def _ray_channels(self, v: RobotView, cnt_out, pc_out, ox: float, oy: float, sx: float,
                      sy: float, step: float) -> None:
        st = v.ray_store
        if st is None or st.n == 0:
            return
        live = np.flatnonzero(st.live())
        if live.size == 0:
            return
        p = FEAT_PC_DIM if pc_out is not None else 1
        feat = st.feat_peak if st.feat_peak is not None else np.zeros((st.n, self.D), np.float32)
        pcs = (np.ascontiguousarray((feat[live] - self.pc_mean) @ self.pc_comps.T, np.float64)
               if pc_out is not None else np.zeros((live.size, 1), np.float64))
        ni, nj = cnt_out.shape
        cnt = np.zeros((ni, nj), np.float64)
        fsum = np.zeros((ni, nj, p), np.float64)
        _ray_raster(np.ascontiguousarray(st.origin_xy[live]), np.ascontiguousarray(st.az[live]),
                    pcs, float(self.cfg.sensor.depth_limit_m),
                    float(self.cfg.sensor.visual_range_m), float(max(step, 1e-3)),
                    float(ox), float(oy), float(sx), float(sy), cnt, fsum)
        if pc_out is not None:                    # mean over the crossings, before cnt is scaled
            hit = (cnt > 0)[..., None]
            mean = np.zeros_like(fsum)
            np.divide(fsum, np.maximum(cnt, 1.0)[..., None], out=mean, where=hit)
            pc_out[:] = np.moveaxis(mean, -1, 0)
        cnt_out[:] = np.minimum(cnt / RAY_BEV_NORM, 1.0)


def _cov_hits(v: RobotView, disc, ci, cj) -> tuple[np.ndarray, np.ndarray]:
    """Known fraction and mean hit count around each cell. The fraction is over what the robot
    *knows* (shared coverage included, so it stops looking there); the hit count only over what
    carries a feature, so a peer's coverage report never invents looks the robot did not get."""
    cov, hits = _disc_stats(v.known, v.hits, disc, ci, cj)
    if v.feat_known is not None and v.feat_known is not v.known:
        _, hits = _disc_stats(v.feat_known, v.hits, disc, ci, cj)
    return cov, hits


def _stamp_visits(out, visited, ox: float, oy: float, inv: float, sig: float) -> None:
    """Blob per visited record into an axis-aligned raster whose (0, 0) cell starts at (ox, oy)."""
    ni, nj = out.shape
    for rec in visited:
        u = (float(rec.xy[0]) - ox) * inv
        w = (float(rec.xy[1]) - oy) * inv
        i0, i1 = max(0, int(w - 3 * sig)), min(ni, int(w + 3 * sig) + 1)
        j0, j1 = max(0, int(u - 3 * sig)), min(nj, int(u + 3 * sig) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        ii = np.arange(i0, i1) + 0.5 - w
        jj = np.arange(j0, j1) + 0.5 - u
        g = np.exp(-(ii[:, None] ** 2 + jj[None, :] ** 2) / (2 * sig * sig))
        np.maximum(out[i0:i1, j0:j1], g, out=out[i0:i1, j0:j1])


def _u(x: float) -> float:
    """Saturate a nominally-[0, 1] feature. These normalisers are nominal, not bounds: a segment
    can hold more than 200 cells and a ray more than 500 looks."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _age(t: float, t_last: float, ttl: float) -> float:
    return _u((t - t_last) / ttl)


def _nearest(pts: np.ndarray, xy, diag: float) -> float:
    """Distance to the nearest of `pts`, over the region diagonal, saturated at 1 (1 = none)."""
    if pts.shape[0] == 0:
        return 1.0
    d = float(np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1]).min())
    return min(d / diag, 1.0)


def _slot_offset(kind: int, tc, si: int, n_fr: int, n_ray: int) -> int:
    if kind == TOKEN_FRONTIER:
        return si
    if kind == TOKEN_RAY:
        return tc.k_frontier + (si - n_fr)
    return tc.k_frontier + tc.k_ray + (si - n_fr - n_ray)


def build_team_obs(raster, cfg, rf, robots, t, planner, builder: TokenBuilder | None = None) -> TeamObs:
    b = builder or TokenBuilder(raster, cfg, rf.emb)
    return b.build(rf, robots, t, planner)


__all__ = ["TokenBuilder", "RobotView", "team_view", "build_team_obs", "BEV_CHANNELS",
           "LOCAL_CHANNELS", "NBHD_M", "CLAIM_M", "FEAT_RADIUS_M"]
