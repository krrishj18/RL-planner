"""Per-robot knowledge and the gossip that fills it (CONTRACTS.md 5.1, DESIGN_VARIANTS.md B/C).

`comms.mode == "full"` keeps one shared team belief and nothing here runs. `comms.mode == "range"`
gives every robot its own `RobotBelief`: its own `known` mask, its own frontiers and segments
computed from that mask, the rays it emitted plus the ones peers handed it, a cache of what it last
heard about each peer, and the visited-target records it knows. Two robots link when they are
within `comms.range_m` of each other; a message reaches everything in the same connected component
(or within `relay_hops` edges), exactly like the range + relay model the real fleet runs
(`coordination_bringup/comms_model.py`), and every payload is small and typed.

Approximations, all deliberate and all documented here:
- **the feature map is shared**. A cell's feature is what it is whoever looked at it, so
  `vox_feat_sum` stays one array and a robot reads it only through its own `feat_known` mask. Two
  robots that observe the same cell therefore agree on it exactly instead of holding two noisy
  estimates; the noise is per observation, not per robot, so the difference is small and it saves
  D floats per cell per robot (54 MB each at 750x750).
- **`hits` are the global hit counts, masked by the robot's `feat_known`**. A robot that has looked
  at a cell twice reads the count of every look the team gave it. Same trade: one int32 grid
  instead of one per robot.
- **coverage shared as a coarse grid marks cells known without features**: `known` is set,
  `feat_known` is not, so those cells stop generating frontiers, carry no hit count and contribute
  no feature to a token, a segment or a raster channel. A coarse cell is only sent when *every*
  fine cell in it is known to the sender, so shared coverage never claims ground nobody covered.
- **rays are exchanged as snapshots**. What the receiver keeps is the state of that bin at the
  moment of contact; it does not keep updating after the link drops, and it dies when the
  *receiver's* map covers what it points at.
- **contact is evaluated once per decision**, at the positions the robots hold at the end of it.
- **a robot's token ids live in its own namespace** (`state.TOKEN_ID_STRIDE`): under range comms
  robot A's frontier 5 and robot B's frontier 5 are different things, so the ids are offset per
  robot and no cross-robot claim key can collide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .rayfronts_sim import CAND_POOL, FrontierIndex, SegmentIndex, ray_resolve_flags
from .state import (TOKEN_ID_STRIDE, PeerRecord, RayStore, RayTarget, SegmentToken, VisitRecord)
from .tokens import RobotView

RECV_ID_BASE = 1 << 18     # ids allocated to received segments, above any locally grown id
MAX_ROBOTS = 16            # width of the seen_by / dec_by bitmasks


@dataclass
class RaySnapshot:
    """One peer ray as the receiver keeps it: the bin's state at the moment of contact."""
    id: int
    origin_xy: np.ndarray
    az: float
    el: float
    conf: float
    n_obs: int
    t_first: float
    t_last: float
    feat: np.ndarray
    feat_peak: np.ndarray
    resolved: bool = False


def coarse_known(known: np.ndarray, k: int) -> np.ndarray:
    """Down-sample a known mask into blocks of `k` cells; a block is set only if all of it is."""
    if k <= 1:
        return known.copy()
    ny, nx = known.shape
    pi, pj = (-ny) % k, (-nx) % k
    pad = np.pad(known, ((0, pi), (0, pj)), constant_values=True)
    return pad.reshape(pad.shape[0] // k, k, pad.shape[1] // k, k).all(axis=(1, 3))


def expand_coarse(coarse: np.ndarray, k: int, shape: tuple[int, int]) -> np.ndarray:
    if k <= 1:
        return coarse[: shape[0], : shape[1]]
    return np.repeat(np.repeat(coarse, k, axis=0), k, axis=1)[: shape[0], : shape[1]]


class RobotBelief:
    """One robot's knowledge: what it observed, what it was told, and the products it derives."""

    def __init__(self, idx: int, raster, cfg, D: int):
        self.idx = int(idx)
        self.bit = np.uint16(1 << int(idx))
        self.raster = raster
        self.cfg = cfg
        self.D = int(D)
        ny, nx = raster.shape
        self.known = np.zeros((ny, nx), np.bool_)       # own observations + shared coverage
        self.feat_known = np.zeros((ny, nx), np.bool_)  # ... of which these carry a feature
        self.frontiers = FrontierIndex(raster, cfg)
        self.segidx = SegmentIndex(raster, cfg, D)
        self.own_res = np.zeros(0, np.bool_)            # per global ray row: resolved for *me*
        self.rays_in: dict[int, RaySnapshot] = {}
        self.segs_in: dict[tuple[int, int], SegmentToken] = {}
        self.visited: dict[tuple[int, int], VisitRecord] = {}
        self.visited_at_decision: frozenset[tuple[int, int]] = frozenset()
        self.peers: dict[int, PeerRecord] = {}
        self.coverage = 0.0
        self.n_known = 0
        self._ids: dict[tuple[str, int, int], int] = {}
        self._next_recv = RECV_ID_BASE
        self._seg_pool = int(CAND_POOL * max(1, cfg.tokens.k_segment))
        # products, refreshed once per decision
        self.frontier_list: list[Any] = []
        self.segments: list[SegmentToken] = []
        self.ray_list: list[RayTarget] = []
        self.ray_store: RayStore | None = None

    # -- own sensing -----------------------------------------------------------------------
    def ensure_rays(self, n: int) -> None:
        """Grow the per-robot resolution flags to the global ray table."""
        if self.own_res.shape[0] < n:
            g = np.zeros(n, np.bool_)
            g[: self.own_res.shape[0]] = self.own_res
            self.own_res = g

    def observe(self, ij: np.ndarray) -> None:
        i, j = ij[:, 0], ij[:, 1]
        self.known[i, j] = True
        self.feat_known[i, j] = True

    # -- receiving -------------------------------------------------------------------------
    def receive_cells(self, cells: np.ndarray, with_features: bool) -> None:
        np.logical_or(self.known, cells, out=self.known)
        if with_features:
            np.logical_or(self.feat_known, cells, out=self.feat_known)

    def receive_rays(self, snaps: Sequence[RaySnapshot], owned: np.ndarray) -> None:
        """Peer rays; a bin this robot feeds itself keeps its own (live) version."""
        for sn in snaps:
            if sn.id < owned.shape[0] and owned[sn.id]:
                continue
            self.rays_in[int(sn.id)] = sn

    def receive_segments(self, src: int, segs: Sequence[SegmentToken]) -> None:
        for sg in segs:
            key = (int(src), int(sg.id))
            self.segs_in[key] = SegmentToken(id=self._local_id("seg", key), xy=sg.xy.copy(),
                                             ij=sg.ij, feat=sg.feat.copy(), n_cells=sg.n_cells,
                                             mean_hits=sg.mean_hits, ray_count=sg.ray_count,
                                             t_first=sg.t_first, t_last=sg.t_last)
        # a peer relabels its map every resegmentation, so its old ids never come back and the
        # inbox would grow all episode. `refresh` keeps the newest `_seg_pool` of the merged list,
        # so anything older than that can never become a token again: drop it. The id memo keeps
        # its local id, so a peer that re-sends it gets the same token id back.
        if len(self.segs_in) > self._seg_pool:
            keep = sorted(self.segs_in.items(),
                          key=lambda kv: (-kv[1].t_first, -kv[1].id))[: self._seg_pool]
            self.segs_in = dict(keep)

    def receive_visited(self, recs: Sequence[VisitRecord]) -> None:
        for v in recs:
            cur = self.visited.get(v.key)
            if cur is None or v.n_found > cur.n_found:
                self.visited[v.key] = v

    def add_visit(self, rec: VisitRecord) -> None:
        self.visited[rec.key] = rec

    def _local_id(self, kind: str, key: tuple[int, int]) -> int:
        k = (kind, key[0], key[1])
        hit = self._ids.get(k)
        if hit is None:
            hit = self._next_recv
            self._next_recv += 1
            self._ids[k] = hit
        return hit

    # -- per-decision refresh --------------------------------------------------------------
    def begin_decision(self) -> None:
        """Freeze what this robot knew when it chose its target (the `known_only` revisit rule)."""
        self.visited_at_decision = frozenset(self.visited)

    def refresh(self, rf, t: float, decision: int) -> None:
        """Recompute this robot's frontiers, segments and live rays from its own knowledge."""
        seen = self.known if rf._all_observable else (self.known | ~rf.observable)
        _, self.frontier_list = self.frontiers.extract(self.known, seen)
        self._refresh_rays(rf, seen, t)
        st = self.ray_store
        org = (st.origin_xy if st is not None and st.n else np.zeros((0, 2), np.float64))
        az = (st.az if st is not None and st.n else np.zeros(0, np.float64))
        live = np.flatnonzero(st.live()) if st is not None and st.n else np.zeros(0, np.int64)
        own = self.segidx.extract(rf.vox_feat_sum, self.feat_known, rf.vox_cnt, rf.last_seen_t,
                                  np.ascontiguousarray(org[live]), np.ascontiguousarray(az[live]),
                                  t, decision)
        # a peer's segment and an own segment of the same ground stay two tokens: merging them
        # would be inference, and the policy can see both
        segs = own + list(self.segs_in.values())
        segs.sort(key=lambda s: (-s.t_first, -s.id))
        self.segments = segs[: self._seg_pool]
        self.n_known = int(self.known.sum())
        self.coverage = self.n_known / max(1, int(rf.observable.sum()))

    def _refresh_rays(self, rf, seen: np.ndarray, t: float) -> None:
        """Own rays (live rows of the global store) plus the snapshots peers sent, each resolved
        against *this* robot's map, in one pass."""
        n = rf.n_rays
        self.ensure_rays(n)
        owned = (rf._r_by[:n] & self.bit) != 0
        live = np.flatnonzero(owned & ~self.own_res[:n])
        # a snapshot leaves the inbox once this robot's own map has resolved it, or once the robot
        # feeds that bin itself (two live copies of one bin would be two tokens with one id and a
        # double count in the ray rasters). Dropping it is what the dict did anyway: a peer that
        # re-sends the bin overwrites the slot with a fresh snapshot.
        if self.rays_in:
            for i in [i for i, s in self.rays_in.items() if s.resolved or (i < n and owned[i])]:
                del self.rays_in[i]
        inbox = list(self.rays_in.values())
        k, n_in = live.size, len(inbox)
        if k or n_in:
            m0 = k + n_in
            org = np.empty((m0, 2), np.float64)
            az0 = np.empty(m0, np.float64)
            el0 = np.empty(m0, np.float64)
            tl0 = np.empty(m0, np.float64)
            if k:
                org[:k] = rf._r_origin[live]
                az0[:k] = rf._r_az[live]
                el0[:k] = rf._r_el[live]
                tl0[:k] = rf._r_tlast[live]
            for a, s in enumerate(inbox, start=k):
                org[a] = s.origin_xy
                az0[a], el0[a], tl0[a] = s.az, s.el, s.t_last
            gone = ray_resolve_flags(seen, self.raster, self.cfg, org, az0, el0, tl0, t,
                                     rf.target_range)
            if k:
                self.own_res[live[gone[:k]]] = True
                live = live[~gone[:k]]
            if n_in:
                for s, g in zip(inbox, gone[k:]):
                    s.resolved = bool(g)
                inbox = [s for s, g in zip(inbox, gone[k:]) if not g]
        m = live.size + len(inbox)
        origin = np.zeros((m, 2), np.float64)
        az = np.zeros(m, np.float64)
        el = np.zeros(m, np.float64)
        conf = np.zeros(m, np.float32)
        nobs = np.zeros(m, np.int32)
        t0 = np.zeros(m, np.float64)
        t1 = np.zeros(m, np.float64)
        ids = np.zeros(m, np.int32)
        feat = np.zeros((m, self.D), np.float32)
        peak = np.zeros((m, self.D), np.float32)
        if live.size:
            k = live.size
            origin[:k] = rf._r_origin[live]
            az[:k] = rf._r_az[live]
            el[:k] = rf._r_el[live]
            conf[:k] = rf._r_conf[live]
            nobs[:k] = rf._r_nobs[live]
            t0[:k] = rf._r_tfirst[live]
            t1[:k] = rf._r_tlast[live]
            ids[:k] = rf._r_ids[live]
            feat[:k] = rf._r_feat[live]
            peak[:k] = rf._r_peak[live]
        for a, s in enumerate(inbox, start=live.size):
            origin[a] = s.origin_xy
            az[a], el[a] = s.az, s.el
            conf[a], nobs[a] = s.conf, s.n_obs
            t0[a], t1[a] = s.t_first, s.t_last
            ids[a] = s.id
            feat[a] = s.feat
            peak[a] = s.feat_peak
        self.ray_store = RayStore(origin_xy=origin, az=az, el=el, conf=conf, n_obs=nobs,
                                  t_first=t0, t_last=t1, ids=ids,
                                  resolved=np.zeros(m, np.bool_), feat=feat, feat_peak=peak)
        # the builder takes the newest `k_ray` of this list and nothing else looks further, so
        # there is no candidate pool to extract here (the team belief keeps one for the viewers)
        self.ray_list = ray_targets(self.ray_store, rf, self.raster,
                                    max(1, self.cfg.tokens.k_ray))

    # -- what the builders read ------------------------------------------------------------
    def view(self, rf) -> RobotView:
        vis = sorted(self.visited.values(), key=lambda v: (-v.t, -v.seq))
        return RobotView(known=self.known, feat_sum=rf.vox_feat_sum, hits=rf.vox_cnt,
                         last_seen=rf.last_seen_t, frontier_mask=self.frontiers.mask,
                         frontiers=self.frontier_list, rays=self.ray_list, segments=self.segments,
                         ray_store=self.ray_store, feat_known=self.feat_known, visited=vis,
                         peers=self.peers, id_offset=self.idx * TOKEN_ID_STRIDE,
                         coverage=self.coverage, robot=self.idx, seg_labels=self.segidx.labels)


def ray_targets(store: RayStore, rf, raster, k_max: int) -> list[RayTarget]:
    """`RayTarget`s for a ray store, newest first — the same geometry the team belief uses."""
    n = store.n
    if n == 0:
        return []
    live = np.flatnonzero(store.live())
    if live.size == 0:
        return []
    order = live[np.argsort(-store.ids[live], kind="stable")[:k_max]]
    rng_m = rf.target_range(store.el[order])
    x0, y0, x1, y1 = raster.region
    eps = 1e-6
    out: list[RayTarget] = []
    for m, idx in enumerate(order):
        idx = int(idx)
        a = float(store.az[idx])
        tx = store.origin_xy[idx, 0] + math.cos(a) * float(rng_m[m])
        ty = store.origin_xy[idx, 1] + math.sin(a) * float(rng_m[m])
        out.append(RayTarget(
            id=int(store.ids[idx]), ray_idx=idx,
            xy=np.array([min(max(tx, x0 + eps), x1 - eps), min(max(ty, y0 + eps), y1 - eps)]),
            origin_xy=store.origin_xy[idx].copy(), az=a, el=float(store.el[idx]),
            range_m=float(rng_m[m]), feat=store.feat_peak[idx].copy(),
            feat_mean=store.feat[idx].copy(), conf=float(store.conf[idx]),
            n_obs=int(store.n_obs[idx]), t_first=float(store.t_first[idx]),
            t_last=float(store.t_last[idx])))
    return out


@dataclass
class GossipStats:
    """Per-decision link bookkeeping (metrics only)."""
    range_m: float = float("inf")
    linked_pairs: int = 0
    n_pairs: int = 0
    contacts: int = 0
    link_frac_sum: float = 0.0
    decisions: int = 0

    def add(self, linked: int, pairs: int, contacts: int) -> None:
        self.linked_pairs = linked
        self.n_pairs = pairs
        self.contacts = contacts
        self.link_frac_sum += linked / max(1, pairs)
        self.decisions += 1

    @property
    def link_frac(self) -> float:
        return self.link_frac_sum / max(1, self.decisions)


class CommsSim:
    """The gossip layer: who can hear whom this decision, and what crosses the link."""

    def __init__(self, raster, cfg, n_robots: int, D: int, rng: np.random.Generator):
        if n_robots > MAX_ROBOTS:
            raise ValueError(f"CommsSim: {n_robots} robots exceeds MAX_ROBOTS={MAX_ROBOTS}")
        self.raster = raster
        self.cfg = cfg
        self.n = int(n_robots)
        self.beliefs = [RobotBelief(i, raster, cfg, D) for i in range(self.n)]
        self.cov_k = max(1, int(round(cfg.comms.share.coverage_cell_m / raster.cell_m)))
        self.range_m = float(cfg.comms.range_m)
        self.stats = GossipStats(self.range_m)
        self._cov_sent: dict[tuple[int, int], int] = {}
        self._cov_cache: dict[int, tuple[int, np.ndarray]] = {}
        self.last_links = np.zeros((self.n, self.n), np.bool_)   # who reached whom last decision
        self.sample_range(rng)

    # -- episode / decision ----------------------------------------------------------------
    def sample_range(self, rng: np.random.Generator) -> float:
        c = self.cfg.comms
        if c.randomize_range and c.range_choices:
            self.range_m = float(c.range_choices[int(rng.integers(len(c.range_choices)))])
        else:
            self.range_m = float(c.range_m)
        self.stats = GossipStats(self.range_m)
        return self.range_m

    def begin_decision(self) -> None:
        for b in self.beliefs:
            b.begin_decision()

    def observe(self, ridx: int, ij: np.ndarray) -> None:
        self.beliefs[ridx].observe(ij)

    # -- links -----------------------------------------------------------------------------
    def links(self, robots) -> np.ndarray:
        """[n, n] bool: True where a message from i reaches j (relay included, i != j)."""
        n = self.n
        xy = np.array([[r.pos[0], r.pos[1]] for r in robots[:n]], np.float64)
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        # range 0 means no radio at all, not "only robots on the same spot" (decentral_blackout)
        adj = (d <= self.range_m) if self.range_m > 0.0 else np.zeros_like(d, np.bool_)
        np.fill_diagonal(adj, False)
        hops = int(self.cfg.comms.relay_hops)
        reach = adj.copy()
        limit = n - 1 if hops <= 0 else min(hops, n - 1)
        for _ in range(max(0, limit - 1)):          # closure: one more hop per round
            nxt = reach | (reach @ adj)
            np.fill_diagonal(nxt, False)
            if np.array_equal(nxt, reach):
                break
            reach = nxt
        return reach

    def exchange(self, robots, rf, t: float, force_all: bool = False) -> np.ndarray:
        """One round of contact. `force_all` reaches everyone (the spawn exchange: the team starts
        together and knows it), but only the radio's own links count towards `link_frac` — a
        blackout variant reports 0, not 1/n_decisions."""
        n = self.n
        reach = self.links(robots)
        radio = reach                              # what the radio alone would have reached
        if force_all:                              # the spawn exchange is a hand-over, not a link
            reach = ~np.eye(n, dtype=np.bool_)
        pairs = n * (n - 1) // 2
        linked = int(np.count_nonzero(np.triu(radio, 1)))
        contacts = int(reach.sum())
        share = self.cfg.comms.share
        n_obs_cells = max(1, int(rf.observable.sum()))
        for b in self.beliefs:                     # what each robot reports about itself
            b.n_known = int(b.known.sum())
            b.coverage = b.n_known / n_obs_cells
            b.ensure_rays(rf.n_rays)
            for pr in b.peers.values():            # remembered until refreshed below
                pr.linked = False
        # rows and ids coincide because compaction is off under range comms (see DisasterEnv)
        owned = [(rf._r_by[: rf.n_rays] & b.bit) != 0 for b in self.beliefs]
        # every payload is snapshotted *before* anything is delivered: the round is simultaneous,
        # so what a robot forwards is what it knew at contact, not what a lower-indexed peer just
        # handed it (that would push a message one hop further than `relay_hops` allows and make
        # the result depend on the robot indices)
        dst_of = [np.flatnonzero(reach[src]) for src in range(n)]
        payloads = [None if force_all or dst_of[src].size == 0
                    else self._payload(self.beliefs[src], rf, share) for src in range(n)]
        for src in range(n):
            dsts = dst_of[src]
            if dsts.size == 0:
                continue
            b_src = self.beliefs[src]
            payload = payloads[src]
            for dst in dsts:
                dst = int(dst)
                b = self.beliefs[dst]
                b.peers[src] = PeerRecord(
                    robot=src, pos=np.array(robots[src].pos[:2], np.float64),
                    target_xy=(None if robots[src].target_xy is None
                               else np.array(robots[src].target_xy[:2], np.float64)),
                    target_type=int(robots[src].target_token_type), t_contact=float(t),
                    coverage=float(b_src.coverage), linked=True)
                if payload is None:
                    continue
                cells, snaps, segs, vis = payload
                if cells is not None:
                    key = (src, dst)
                    cnt = int(cells[1])
                    if self._cov_sent.get(key) != cnt:
                        b.receive_cells(cells[0], share.features)
                        self._cov_sent[key] = cnt
                if snaps:
                    b.receive_rays(snaps, owned[dst])
                if segs:
                    b.receive_segments(src, segs)
                if vis:
                    b.receive_visited(vis)
        self.stats.add(linked, pairs, contacts)
        self.last_links = reach
        return reach

    def _payload(self, b: RobotBelief, rf, share) -> tuple:
        """What one robot puts on the air: coarse coverage, its own rays and segments, and every
        visited record it knows (records are the point of the epidemic, rays and segments are the
        sender's own observations and reach further hops by relay)."""
        cells = None
        if share.coverage or share.features:
            hit = self._cov_cache.get(b.idx)
            if hit is None or hit[0] != b.n_known:
                c = coarse_known(b.known, self.cov_k)
                cells = (expand_coarse(c, self.cov_k, b.known.shape), int(c.sum()))
                self._cov_cache[b.idx] = (b.n_known, cells)
            else:
                cells = hit[1]
        snaps: list[RaySnapshot] = []
        if share.rays != "none" and share.ray_cap > 0:
            n = rf.n_rays
            own = np.flatnonzero(((rf._r_by[:n] & b.bit) != 0) & ~b.own_res[:n])
            if own.size:
                order = own[np.argsort(-rf._r_ids[own], kind="stable")]
                cap = (share.ray_cap if share.rays == "all"
                       else min(share.ray_cap, share.rays_newest))
                for idx in order[:cap]:
                    idx = int(idx)
                    snaps.append(RaySnapshot(
                        id=int(rf._r_ids[idx]), origin_xy=rf._r_origin[idx].copy(),
                        az=float(rf._r_az[idx]), el=float(rf._r_el[idx]),
                        conf=float(rf._r_conf[idx]), n_obs=int(rf._r_nobs[idx]),
                        t_first=float(rf._r_tfirst[idx]), t_last=float(rf._r_tlast[idx]),
                        feat=rf._r_feat[idx].copy(), feat_peak=rf._r_peak[idx].copy()))
        segs: list[SegmentToken] = []
        if share.segments and share.segment_cap > 0:
            segs = [s for s in b.segments if s.id < RECV_ID_BASE][: share.segment_cap]
        vis: list[VisitRecord] = []
        if share.visited:
            vis = sorted(b.visited.values(), key=lambda v: -v.t)[: max(1, self.cfg.tokens.k_visited)]
        return (cells, snaps, segs, vis)

    # -- views -----------------------------------------------------------------------------
    def views(self, rf, robots, t: float, decision: int) -> list[RobotView]:
        for b in self.beliefs:
            b.refresh(rf, t, decision)
        return [b.view(rf) for b in self.beliefs]

    def known_visits(self, ridx: int) -> dict[tuple[int, int], VisitRecord]:
        return self.beliefs[ridx].visited


__all__ = ["CommsSim", "RobotBelief", "RaySnapshot", "GossipStats", "coarse_known",
           "expand_coarse", "ray_targets", "RECV_ID_BASE", "MAX_ROBOTS"]
