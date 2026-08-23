"""Mocks for the visualizer tests: a duck-typed raster, a mock EnvState and a 3-step mock env.

These follow CONTRACTS.md / sim.state shapes exactly (open-set observation: per-cell and per-ray
*features*, segments, the query block, the local crop and the peer block) so the viz can be built
and tested while the simulator is mid-edit.

`make_mock_state(..., per_robot=True)` also attaches per-robot views, so the per-drone belief panel
is testable before the gossip protocol lands: each robot knows only the cells along its own track.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlplanner.scene import schema
from rlplanner.sim import state as S
from rlplanner.sim.config import EnvConfig

HUMAN_DTYPE = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8"), ("role_id", "i4"), ("pose_id", "i4"),
                        ("container_id", "i4"), ("visibility_id", "i4"), ("scene_idx", "i4")])


def token_types() -> tuple[str, ...]:
    return tuple(S.TOKEN_TYPE_NAMES)


def local_channels() -> tuple[str, ...]:
    try:
        from rlplanner.sim.tokens import LOCAL_CHANNELS
        return tuple(LOCAL_CHANNELS)
    except Exception:                       # noqa: BLE001 - mid-edit sim
        return ("known", "hits", "ray_count") + tuple(f"feat_pc{i}" for i in range(8))


def bev_channels() -> tuple[str, ...]:
    try:
        from rlplanner.sim.tokens import BEV_CHANNELS
        return tuple(BEV_CHANNELS)
    except Exception:                       # noqa: BLE001 - mid-edit sim
        return (("known", "hits", "frontier", "robots", "peer_targets", "ray_count")
                + tuple(f"feat_pc{i}" for i in range(8))
                + tuple(f"ray_feat_pc{i}" for i in range(8)))


def embedding_table(cfg: EnvConfig):
    from rlplanner.sim.embeddings import get_embedding_table

    return get_embedding_table(cfg.rayfronts.queries, cfg.rayfronts.embedding_dim,
                               cfg.rayfronts.embeddings_path, cfg.rayfronts.sim_table_path)


# ---- fake raster --------------------------------------------------------------------------
@dataclass
class FakeRaster:
    cell_m: float
    origin: tuple[float, float]
    nx: int
    ny: int
    height: np.ndarray
    cls: np.ndarray
    damage: np.ndarray
    obj_id: np.ndarray
    humans: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    def xy_to_ij(self, x, y):
        i = np.floor((np.asarray(y) - self.origin[1]) / self.cell_m).astype(np.int64)
        j = np.floor((np.asarray(x) - self.origin[0]) / self.cell_m).astype(np.int64)
        return i, j

    def ij_to_xy(self, i, j):
        return (self.origin[0] + (np.asarray(j) + 0.5) * self.cell_m,
                self.origin[1] + (np.asarray(i) + 0.5) * self.cell_m)

    def in_bounds(self, i, j):
        i, j = np.asarray(i), np.asarray(j)
        return (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)

    def obstacle_mask(self, alt: float, clearance: float) -> np.ndarray:
        return self.height >= (alt - clearance)


def _obb_mask(X, Y, cx, cy, sx, sy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = X - cx, Y - cy
    u, v = dx * c + dy * s, -dx * s + dy * c
    return (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2)


def make_fake_raster(scene: schema.Scene, cell_m: float = 2.0) -> FakeRaster:
    """Approximate rasterisation of a Scene following the class priority in CONTRACTS.md 1."""
    x0, y0, x1, y1 = scene.region
    nx, ny = int(math.ceil((x1 - x0) / cell_m)), int(math.ceil((y1 - y0) / cell_m))
    xs = x0 + (np.arange(nx) + 0.5) * cell_m
    ys = y0 + (np.arange(ny) + 0.5) * cell_m
    X, Y = np.meshgrid(xs, ys)
    cls = np.full((ny, nx), schema.CLASS_ID["ground"], dtype=np.int8)
    height = np.zeros((ny, nx), dtype=np.float32)
    obj_id = np.full((ny, nx), -1, dtype=np.int32)
    prio = {n: i for i, n in enumerate(
        ["ground", "park", "road", "sidewalk", "tree", "street_furniture", "bus_stop",
         "building_intact", "building_damaged", "building_destroyed", "debris",
         "vehicle_intact", "vehicle_toppled"])}
    rank = np.zeros((ny, nx), dtype=np.int32)

    def paint(m, name, h, oid=-1):
        r = prio[name]
        sel = m & (rank <= r)
        cls[sel] = schema.CLASS_ID[name]
        rank[sel] = r
        height[sel] = np.maximum(height[sel], h)
        obj_id[sel] = oid

    def rect_mask(rect):
        return (X >= rect[0]) & (X <= rect[2]) & (Y >= rect[1]) & (Y <= rect[3])

    oid = 0
    for b in scene.blocks:
        if b.typology == "park":
            paint(rect_mask(b.rect), "park", 0.0)
    for r in scene.roads:
        paint(rect_mask(r.rect), "road" if r.kind == "road" else "sidewalk", 0.0)
    for p in scene.props:
        name = {"bus_stop": "bus_stop", "tree": "tree"}.get(p.category, "street_furniture")
        paint(_obb_mask(X, Y, p.center[0], p.center[1], max(p.size[0], cell_m),
                        max(p.size[1], cell_m), p.yaw_deg), name, p.resolved_height(), oid)
        oid += 1
    for b in scene.buildings:
        paint(_obb_mask(X, Y, b.center[0], b.center[1], b.size[0], b.size[1], b.yaw_deg),
              f"building_{b.fate}", b.resolved_height(), oid)
        oid += 1
    for d in scene.debris:
        m = (X - d.center[0]) ** 2 + (Y - d.center[1]) ** 2 <= d.radius_m ** 2
        paint(m, "debris", d.resolved_height(), oid)
        oid += 1
    for v in scene.vehicles:
        paint(_obb_mask(X, Y, v.center[0], v.center[1], v.size[0], v.size[1], v.yaw_deg),
              f"vehicle_{v.state}", v.resolved_height(), oid)
        oid += 1

    damage = np.array([[scene.damage_at(float(x), float(y)) for x in xs] for y in ys],
                      dtype=np.float32)
    humans = np.zeros(len(scene.humans), dtype=HUMAN_DTYPE)
    for k, h in enumerate(scene.humans):
        humans[k] = (h.pos[0], h.pos[1], h.pos[2], schema.HUMAN_ROLES.index(h.role),
                     schema.HUMAN_POSES.index(h.pose), schema.HUMAN_CONTAINERS.index(h.container),
                     schema.HUMAN_VISIBILITY.index(h.visibility), k)
    return FakeRaster(cell_m=float(cell_m), origin=(x0, y0), nx=nx, ny=ny, height=height, cls=cls,
                      damage=damage, obj_id=obj_id, humans=humans)


def empty_ray_store(dim: int = 24) -> S.RayStore:
    """A store with no rays; `dim` is the feature width, not a query count."""
    d = int(dim)
    return S.RayStore(origin_xy=np.zeros((0, 2)), az=np.zeros(0), el=np.zeros(0),
                      conf=np.zeros(0, np.float32), n_obs=np.zeros(0, np.int32),
                      t_first=np.zeros(0), t_last=np.zeros(0), ids=np.zeros(0, np.int32),
                      resolved=np.zeros(0, bool), feat=np.zeros((0, d), np.float32),
                      feat_peak=np.zeros((0, d), np.float32))


# ---- per-robot view / visited records ------------------------------------------------------
@dataclass
class MockRobotView:
    """Same field names as `sim.tokens.RobotView`, plus what gossip adds."""
    known: np.ndarray
    feat_sum: np.ndarray
    hits: np.ndarray
    last_seen: np.ndarray
    frontier_mask: np.ndarray
    frontiers: list
    rays: list
    segments: list
    ray_store: Any
    seg_labels: np.ndarray | None = None
    visited: list = field(default_factory=list)
    peers: list = field(default_factory=list)


@dataclass
class MockVisit:
    xy: np.ndarray
    token_type: int
    t: float
    robot: int
    found: int = 0


# ---- mock EnvState ------------------------------------------------------------------------
def _frontier_clusters(observed: np.ndarray, raster: FakeRaster, ig_radius_m: float,
                       min_cells: int) -> tuple[np.ndarray, list[S.FrontierCluster]]:
    from scipy import ndimage

    unobs = ~observed
    nbr = np.zeros_like(unobs)
    nbr[1:, :] |= unobs[:-1, :]
    nbr[:-1, :] |= unobs[1:, :]
    nbr[:, 1:] |= unobs[:, :-1]
    nbr[:, :-1] |= unobs[:, 1:]
    fmask = observed & nbr
    lab, n = ndimage.label(fmask, structure=np.ones((3, 3), dtype=int))
    clusters: list[S.FrontierCluster] = []
    rad_cells = ig_radius_m / raster.cell_m
    for k in range(1, n + 1):
        ij = np.argwhere(lab == k)
        if len(ij) < min_cells:
            fmask[lab == k] = False
            continue
        ci, cj = ij[:, 0].mean(), ij[:, 1].mean()
        i0, i1 = int(max(0, ci - rad_cells)), int(min(raster.ny, ci + rad_cells + 1))
        j0, j1 = int(max(0, cj - rad_cells)), int(min(raster.nx, cj + rad_cells + 1))
        ig = float((~observed[i0:i1, j0:j1]).sum())
        x, y = raster.ij_to_xy(ci, cj)
        clusters.append(S.FrontierCluster(id=len(clusters), centroid_xy=np.array([x, y]),
                                          n_cells=len(ij), info_gain=ig,
                                          cell_ij=ij.astype(np.int32)))
    clusters.sort(key=lambda c: -c.info_gain)
    return fmask, clusters


def _segments(feat_sum: np.ndarray, observed: np.ndarray, vox_cnt: np.ndarray,
              raster: FakeRaster, min_cells: int, t: float,
              cap: int) -> tuple[np.ndarray, list[S.SegmentToken]]:
    """Connected components of one class inside the observed map — what a generic over-segmentation
    of a class-driven feature map converges to, and enough to draw."""
    from scipy import ndimage

    labels = np.full(observed.shape, -1, np.int32)
    segs: list[S.SegmentToken] = []
    nxt = 0
    for c in np.unique(raster.cls[observed]) if observed.any() else ():
        lab, n = ndimage.label(observed & (raster.cls == c))
        for k in range(1, n + 1):
            m = lab == k
            cnt = int(m.sum())
            if cnt < min_cells:
                continue
            ij = np.argwhere(m)
            ci, cj = ij[:, 0].mean(), ij[:, 1].mean()
            d = (ij[:, 0] - ci) ** 2 + (ij[:, 1] - cj) ** 2
            mi, mj = (int(v) for v in ij[int(np.argmin(d))])
            labels[m] = nxt
            f = feat_sum[m].sum(0)
            f = f / max(float(np.linalg.norm(f)), 1e-12)
            x, y = raster.ij_to_xy(mi, mj)
            segs.append(S.SegmentToken(id=nxt, xy=np.array([float(x), float(y)]), ij=(mi, mj),
                                       feat=f.astype(np.float32), n_cells=cnt,
                                       mean_hits=float(vox_cnt[m].mean()), ray_count=0,
                                       t_first=0.0, t_last=float(t)))
            nxt += 1
    segs.sort(key=lambda s: -s.n_cells)
    return labels, segs[:cap]


def _ray_store(rng, robots, cfg: EnvConfig, emb, raster: FakeRaster, n_rays: int,
               t: float) -> S.RayStore:
    n = max(0, int(n_rays))
    D = int(emb.D)
    if n == 0:
        return empty_ray_store(D)
    origins = np.array([r.pos for r in robots])
    o = origins[rng.integers(0, len(robots), n)] + rng.normal(0, 5.0, (n, 2))
    cls = rng.integers(0, schema.N_CLASSES, n)
    cls[rng.random(n) < 0.25] = schema.CLASS_ID["human_prone"]      # a quarter look human
    f = np.asarray(emb.class_emb, np.float32)[cls] + rng.normal(
        0, cfg.rayfronts.ray_noise_std, (n, D)).astype(np.float32)
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    return S.RayStore(origin_xy=o.astype(np.float64), az=rng.uniform(-math.pi, math.pi, n),
                      el=rng.uniform(-0.9, -0.2, n),
                      conf=rng.uniform(0.2, cfg.rayfronts.ray_conf_cap, n).astype(np.float32),
                      n_obs=rng.integers(1, 30, n).astype(np.int32),
                      t_first=rng.uniform(0, t, n), t_last=np.full(n, t),
                      ids=np.arange(n, dtype=np.int32), resolved=rng.random(n) < 0.3,
                      feat=f.copy(), feat_peak=f.copy())


def _ray_targets(rays: S.RayStore, cfg: EnvConfig, scene, alt: float) -> list[S.RayTarget]:
    """The live rays the token builder would offer, newest first (recency, never a score)."""
    if rays.n == 0:
        return []
    live = np.flatnonzero(rays.live())
    x0, y0, x1, y1 = scene.region
    lo, hi = float(cfg.sensor.depth_limit_m), float(cfg.sensor.visual_range_m)
    out: list[S.RayTarget] = []
    for i in sorted(live, key=lambda k: -float(rays.t_first[k]))[: int(cfg.tokens.k_ray)]:
        el = float(rays.el[i])
        rng_m = float(np.clip(alt / max(1e-3, -math.tan(el)), lo, hi)) if el < 0 \
            else float(cfg.rayfronts.ray_range_m)
        o = rays.origin_xy[i]
        xy = o + rng_m * np.array([math.cos(rays.az[i]), math.sin(rays.az[i])])
        out.append(S.RayTarget(id=int(rays.ids[i]), ray_idx=int(i),
                               xy=np.array([np.clip(xy[0], x0, x1), np.clip(xy[1], y0, y1)]),
                               origin_xy=np.asarray(o, float), az=float(rays.az[i]), el=el,
                               range_m=rng_m, feat=rays.feat_peak[i].copy(),
                               feat_mean=rays.feat[i].copy(), conf=float(rays.conf[i]),
                               n_obs=int(rays.n_obs[i]), t_first=float(rays.t_first[i]),
                               t_last=float(rays.t_last[i])))
    return out


def make_mock_state(seed: int = 0, n_robots: int = 3, cfg: EnvConfig | None = None,
                    scene: schema.Scene | None = None, n_rays: int = 40, n_found: int = 5,
                    coverage: float = 0.35, t: float = 120.0, all_masked: bool = False,
                    per_robot: bool = False, n_visited: int = 4) -> S.EnvState:
    """An EnvState with the documented shapes, plausible enough to eyeball."""
    rng = np.random.default_rng(seed)
    cfg = cfg or EnvConfig()
    cfg.robot.n_robots = n_robots
    scene = scene or schema.make_synthetic_scene(seed)
    raster = make_fake_raster(scene, cfg.raster.cell_m)
    emb = embedding_table(cfg)
    D = int(emb.D)
    ny, nx = raster.ny, raster.nx

    # robots: a random walk from the spawn points, observed = union of discs along the tracks
    robots: list[S.RobotState] = []
    x0, y0, x1, y1 = scene.region
    observed = np.zeros((ny, nx), dtype=bool)
    own = [np.zeros((ny, nx), bool) for _ in range(n_robots)]
    I, J = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    CX, CY = raster.ij_to_xy(I, J)
    n_steps = max(4, int(coverage * 260))
    for r in range(n_robots):
        sp = scene.robots_spawn[r % max(1, len(scene.robots_spawn))] if scene.robots_spawn else (x0, y0, 25.0)
        pos = np.array([sp[0], sp[1]], dtype=float)
        heading = rng.uniform(0, 2 * math.pi)
        traj = [tuple(pos)]
        for _ in range(n_steps):
            heading += rng.normal(0, 0.5)
            pos = pos + cfg.robot.speed_mps * cfg.dt_sim * np.array([math.cos(heading), math.sin(heading)])
            pos[0] = float(np.clip(pos[0], x0 + 2, x1 - 2))
            pos[1] = float(np.clip(pos[1], y0 + 2, y1 - 2))
            traj.append((float(pos[0]), float(pos[1])))
            own[r] |= (CX - pos[0]) ** 2 + (CY - pos[1]) ** 2 <= cfg.sensor.depth_limit_m ** 2
        observed |= own[r]
        tgt = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        robots.append(S.RobotState(idx=r, pos=pos.copy(), alt=cfg.robot.flight_alt_m,
                                   heading=float(heading % (2 * math.pi)), target_xy=tgt,
                                   target_token_type=int(rng.integers(0, len(token_types()))),
                                   target_id=int(rng.integers(0, 20)), path=[tuple(tgt)],
                                   dist_travelled=float(n_steps * cfg.robot.speed_mps),
                                   trajectory=traj, last_action=int(rng.integers(0, cfg.k_tokens))))

    vox_cnt = np.where(observed, rng.integers(1, 8, size=(ny, nx)), 0).astype(np.int32)
    last_seen_t = np.where(observed, rng.uniform(0, t, size=(ny, nx)), -1.0).astype(np.float32)

    # the belief stores *features*: the class row, noised, summed over the looks (CONTRACTS.md 4)
    ce = np.asarray(emb.class_emb, np.float32)
    feat = ce[raster.cls.astype(np.int64)] + rng.normal(
        0, cfg.rayfronts.feat_noise_std, (ny, nx, D)).astype(np.float32)
    feat /= np.maximum(np.linalg.norm(feat, axis=-1, keepdims=True), 1e-9)
    vox_feat_sum = (feat * vox_cnt[..., None]).astype(np.float32)

    # humans whose cell has been observed enough times count as found
    n_h = len(scene.humans)
    human_hits = np.zeros(n_h, dtype=np.int32)
    human_found = np.zeros(n_h, dtype=bool)
    cand = [k for k, h in enumerate(scene.humans)
            if observed[tuple(int(v) for v in raster.xy_to_ij(h.pos[0], h.pos[1]))]]
    rng.shuffle(cand)
    hits_needed = int(getattr(cfg.rayfronts, "found_hits", 2))
    for k in cand[: max(0, n_found)]:
        h = scene.humans[k]
        human_hits[k] = int(rng.integers(hits_needed, hits_needed + 6))
        human_found[k] = True
        i, j = (int(v) for v in raster.xy_to_ij(h.pos[0], h.pos[1]))
        row = ce[schema.CLASS_ID["human_prone" if h.pose == "prone" else "human_standing"]]
        vox_feat_sum[i, j] = row * float(human_hits[k])

    rays = _ray_store(rng, robots, cfg, emb, raster, n_rays, t)
    ray_targets = _ray_targets(rays, cfg, scene, cfg.robot.flight_alt_m)
    fmask, clusters = _frontier_clusters(observed, raster, cfg.rayfronts.frontier_ig_radius_m,
                                         cfg.rayfronts.frontier_min_cluster_cells)
    seg_labels, segments = _segments(vox_feat_sum, observed, vox_cnt, raster,
                                     int(cfg.rayfronts.segment_min_cells), t,
                                     4 * int(cfg.tokens.k_segment))
    visited = [MockVisit(xy=np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]),
                         token_type=int(rng.integers(1, len(token_types()))), t=rng.uniform(0, t),
                         robot=int(rng.integers(0, n_robots))) for _ in range(max(0, n_visited))]

    obs = make_mock_obs(cfg, robots, clusters, ray_targets, segments, visited, scene, rng, emb,
                        raster, observed, vox_feat_sum, all_masked=all_masked)
    last_actions = np.array([int(np.argmax(obs.token_mask[r])) if obs.token_mask[r].any() else 0
                             for r in range(n_robots)], dtype=np.int64)
    for r, a in enumerate(last_actions):
        robots[r].last_action = int(a)

    is_cas = np.array([h.role == "casualty" for h in scene.humans], bool)
    n_cas = int(is_cas.sum())
    n_f = int((human_found & is_cas).sum())
    st = S.EnvState(
        t=float(t), decision_idx=int(t / cfg.decision_dt), scene=scene, raster=raster, cfg=cfg,
        robots=robots, observed=observed, vox_cnt=vox_cnt, last_seen_t=last_seen_t, rays=rays,
        ray_targets=ray_targets, frontier_mask=fmask, frontier_clusters=clusters,
        segments=segments, seg_labels=seg_labels, human_hits=human_hits, human_found=human_found,
        last_obs=obs, last_actions=last_actions,
        cum_reward=float(n_f - 0.01 * (t / cfg.decision_dt)),
        events=[S.Event(t=float(t), kind="decision", payload={})],
        metrics={"time_to_first": 30.0, "frac_found": n_f / max(1, n_cas),
                 "coverage_end": float(observed.mean()), "finds_auc": 0.2, "redundancy": 0},
        vox_feat_sum=vox_feat_sum, emb=emb, queries=tuple(cfg.rayfronts.queries), rf=None)
    if per_robot:
        st.robot_views = _per_robot_views(st, own, rays, visited, cfg, raster)
    return st


def _per_robot_views(st: S.EnvState, own, rays: S.RayStore, visited, cfg: EnvConfig,
                     raster: FakeRaster) -> list[MockRobotView]:
    """What each robot would know with range comms: the cells it looked at itself, the rays it
    emitted, the frontiers/segments inside its own map and the visited records it was handed."""
    out = []
    for r, known in enumerate(own):
        fs = np.where(known[..., None], st.vox_feat_sum, 0.0).astype(np.float32)
        fm = np.asarray(st.frontier_mask, bool) & known
        lab = np.where(known, st.seg_labels, -1).astype(np.int32)
        keep = set(int(v) for v in np.unique(lab) if v >= 0)
        segs = [s for s in st.segments if int(lab[s.ij]) in keep and known[s.ij]]
        fr = [c for c in st.frontier_clusters if known[tuple(int(v) for v in
                                                             raster.xy_to_ij(*c.centroid_xy))]]
        sel = np.zeros(rays.n, bool)
        if rays.n:
            d = np.hypot(rays.origin_xy[:, 0] - st.robots[r].pos[0],
                         rays.origin_xy[:, 1] - st.robots[r].pos[1])
            sel = d < 3.0 * float(cfg.sensor.visual_range_m)
        store = S.RayStore(origin_xy=rays.origin_xy[sel], az=rays.az[sel], el=rays.el[sel],
                           conf=rays.conf[sel], n_obs=rays.n_obs[sel], t_first=rays.t_first[sel],
                           t_last=rays.t_last[sel], ids=rays.ids[sel], resolved=rays.resolved[sel],
                           feat=rays.feat[sel], feat_peak=rays.feat_peak[sel])
        ids = set(int(v) for v in rays.ids[sel])
        out.append(MockRobotView(known=known, feat_sum=fs, hits=np.where(known, st.vox_cnt, 0),
                                 last_seen=np.where(known, st.last_seen_t, -1.0),
                                 frontier_mask=fm, frontiers=fr,
                                 rays=[t for t in st.ray_targets if t.id in ids], segments=segs,
                                 ray_store=store, seg_labels=lab,
                                 visited=[v for v in visited if v.robot == r or v.t < 0.6 * st.t],
                                 peers=[]))
    return out


def make_mock_obs(cfg: EnvConfig, robots, clusters, ray_targets, segments, visited, scene, rng,
                  emb, raster, observed, vox_feat_sum, all_masked: bool = False) -> S.TeamObs:
    """Tokens in the documented slot order: `[hold] + k_frontier + k_ray + k_segment (+ ...)`."""
    from rlplanner.viz.params import slot_ranges

    n = len(robots)
    D = int(emb.D)
    K = cfg.k_tokens
    F = S.TOKEN_FIXED + D
    qmax = int(cfg.tokens.max_queries)
    tokens = np.zeros((n, K, F), dtype=np.float32)
    mask = np.zeros((n, K), dtype=bool)
    xy = np.full((n, K, 2), np.nan, dtype=np.float32)
    ttype = np.zeros((n, K), dtype=np.int8)
    tid = np.full((n, K), -1, dtype=np.int32)
    names = token_types()
    ranges = slot_ranges(cfg.tokens, K)
    items = {"frontier": [(c.centroid_xy, c.id, None) for c in clusters],
             "ray": [(t.xy, t.id, t.feat) for t in ray_targets],
             "segment": [(s.xy, s.id, s.feat) for s in segments],
             "visited": [(v.xy, k, None) for k, v in enumerate(visited)]}
    diag = math.hypot(scene.region[2] - scene.region[0], scene.region[3] - scene.region[1])
    for r in range(n):
        if "hold" in ranges:
            s = ranges["hold"][0]
            ttype[r, s], mask[r, s] = names.index("hold"), not all_masked
            xy[r, s] = robots[r].pos
            tokens[r, s, names.index("hold")] = 1.0
        for name, (s0, s1) in ranges.items():
            if name == "hold":
                continue
            tt = names.index(name)
            for k in range(s1 - s0):
                s = s0 + k
                ttype[r, s] = tt
                lst = items.get(name, [])
                if k >= len(lst) or all_masked:
                    continue
                pos, ident, feat = lst[k]
                mask[r, s], xy[r, s], tid[r, s] = True, pos, ident
                tokens[r, s, tt] = 1.0
                if feat is not None:
                    tokens[r, s, S.F_FEAT0:S.F_FEAT0 + D] = feat
        for s in range(K):
            if not mask[r, s] or not np.all(np.isfinite(xy[r, s])):
                continue
            dxy = np.asarray(xy[r, s], float) - robots[r].pos
            tokens[r, s, S.F_DX:S.F_DY + 1] = dxy / diag
            tokens[r, s, S.F_DIST] = float(np.hypot(*dxy)) / diag
            b = math.atan2(dxy[1], dxy[0]) - robots[r].heading
            tokens[r, s, S.F_BSIN], tokens[r, s, S.F_BCOS] = math.sin(b), math.cos(b)
            tokens[r, s, S.F_REACH] = 1.0
    rf = np.zeros((n, S.ROBOT_FEAT_DIM), dtype=np.float32)
    for r in range(n):
        rf[r, 0:2] = robots[r].pos / 100.0
        rf[r, 2] = robots[r].alt / 100.0
        rf[r, 3], rf[r, 4] = math.sin(robots[r].heading), math.cos(robots[r].heading)
        rf[r, 8 + min(r, 9)] = 1.0
    q = tuple(cfg.rayfronts.queries)[:qmax]
    qe = np.zeros((qmax, D), np.float32)
    qe[: len(q)] = emb.embed_queries(q)
    qw = np.zeros(qmax, np.float32)
    qw[: len(q)] = 1.0
    qm = np.zeros(qmax, bool)
    qm[: len(q)] = True
    bev = rng.random((len(bev_channels()), cfg.tokens.bev_size, cfg.tokens.bev_size)).astype(np.float32)
    return S.TeamObs(tokens=tokens, token_mask=mask, token_xy=xy, token_type=ttype, token_id=tid,
                     robot_feat=rf, bev=bev, query_emb=qe, query_w=qw, query_mask=qm, t=0.0,
                     local=_mock_local(cfg, robots, emb, raster, observed, vox_feat_sum),
                     peer_tokens=np.zeros((n, max(n - 1, 0), S.PEER_FEAT_DIM), np.float32))


def _mock_local(cfg: EnvConfig, robots, emb, raster, observed, vox_feat_sum):
    """`[n_robots, Cl, S, S]` ego crop, channels per `tokens.LOCAL_CHANNELS`."""
    s = int(getattr(cfg.tokens, "local_size", 0) or 0)
    if s <= 0:
        return None
    ch = local_channels()
    out = np.zeros((len(robots), len(ch), s, s), np.float32)
    mean, comps = emb.pc_basis(len(ch) - 3)
    half = s // 2
    for r, rb in enumerate(robots):
        i, j = (int(v) for v in raster.xy_to_ij(rb.pos[0], rb.pos[1]))
        i0, j0 = i - half, j - half
        gi0, gj0 = max(0, i0), max(0, j0)
        gi1, gj1 = min(raster.ny, i0 + s), min(raster.nx, j0 + s)
        if gi1 <= gi0 or gj1 <= gj0:
            continue
        si, sj = gi0 - i0, gj0 - j0
        h, w = gi1 - gi0, gj1 - gj0
        k = observed[gi0:gi1, gj0:gj1]
        out[r, 0, si:si + h, sj:sj + w] = k
        f = vox_feat_sum[gi0:gi1, gj0:gj1]
        u = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12)
        out[r, 3:, si:si + h, sj:sj + w] = np.moveaxis(((u - mean) @ comps.T) * k[..., None], -1, 0)
    return out


# ---- mock env / policy --------------------------------------------------------------------
class MockEnv:
    """Minimal env matching the CONTRACTS.md 6 API; `done` after `n_steps` decisions."""

    def __init__(self, seed: int = 0, n_robots: int = 2, n_steps: int = 3,
                 cfg: EnvConfig | None = None, scene: schema.Scene | None = None,
                 per_robot: bool = False):
        self.cfg = cfg or EnvConfig()
        self.cfg.robot.n_robots = n_robots
        self.scene = scene or schema.make_synthetic_scene(seed)
        self.seed = seed
        self.n_steps = n_steps
        self.n_robots = n_robots
        self.per_robot = bool(per_robot)
        self.state: S.EnvState | None = None
        self._i = 0

    def reset(self, seed: int | None = None) -> S.TeamObs:
        self._i = 0
        self.state = make_mock_state(seed if seed is not None else self.seed,
                                     n_robots=self.n_robots, cfg=self.cfg, scene=self.scene,
                                     coverage=0.05, t=0.0, per_robot=self.per_robot)
        return self.state.last_obs

    def step(self, actions: np.ndarray):
        if self.state is None:
            raise RuntimeError("MockEnv.step called before reset")
        actions = np.asarray(actions)
        if actions.shape != (self.n_robots,):
            raise ValueError(f"actions shape {actions.shape} != ({self.n_robots},)")
        self._i += 1
        st = make_mock_state(self.seed + self._i, n_robots=self.n_robots, cfg=self.cfg,
                             scene=self.scene, coverage=0.05 + 0.15 * self._i,
                             t=self._i * self.cfg.decision_dt, per_robot=self.per_robot)
        st.decision_idx = self._i
        st.last_actions = actions.astype(np.int64)
        self.state = st
        done = self._i >= self.n_steps
        info = {"new_found": 0, "found_total": st.n_found, "n_casualties": st.n_casualties,
                "coverage": st.coverage, "dist_travelled": np.zeros(self.n_robots),
                "events_this_step": [], "metrics": st.metrics}
        return st.last_obs, float(-0.01), bool(done), info


class FirstValidPolicy:
    """Picks the lowest-index unmasked token per robot."""
    privileged = False

    def act(self, obs: S.TeamObs, state: S.EnvState | None = None) -> np.ndarray:
        m = obs.token_mask
        return np.array([int(np.argmax(m[r])) if m[r].any() else 0 for r in range(m.shape[0])],
                        dtype=np.int64)


@functools.cache
def sim_available() -> bool:
    """True when a tiny `DisasterEnv` can actually be built and stepped. The simulator is renamed
    in place from time to time, so importability is not enough: the viz tests fall back to the
    mocks (and skip the env-backed ones) whenever this is False."""
    try:
        import numpy as _np

        from rlplanner.sim import baselines
        from rlplanner.sim.config import EnvConfig
        from rlplanner.sim.env import DisasterEnv

        cfg = EnvConfig()
        cfg.robot.n_robots = 1
        cfg.t_max_s = 20.0
        env = DisasterEnv(schema.make_synthetic_scene(0, region_m=(80.0, 80.0)), cfg, seed=0)
        pol = baselines.make_policy(sorted(baselines.POLICIES)[0], cfg.rayfronts.queries, 0)
        env.step(_np.asarray(pol.act(env.state.last_obs, env.state)))
        return True
    except Exception:                       # noqa: BLE001 - mid-edit sim: skip, do not error
        return False


__all__ = ["FakeRaster", "make_fake_raster", "make_mock_state", "make_mock_obs", "MockEnv",
           "FirstValidPolicy", "HUMAN_DTYPE", "empty_ray_store", "sim_available",
           "MockRobotView", "MockVisit", "token_types", "local_channels", "bev_channels",
           "embedding_table"]
