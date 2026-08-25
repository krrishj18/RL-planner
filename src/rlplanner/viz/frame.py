"""Episode frame (CONTRACTS.md 9): ground truth | belief | (local crop) | status text.

**Open-set belief.** The map stores per-cell and per-ray *features*, so a query heatmap is a view
taken on demand: `EnvState.query_sim(name)` for the grid and `RayFrontsSim.ray_query_sim` for the
per-ray colour. Both are computed **once per frame** for the query on screen (`query_view`) and
handed to every overlay — never per ray, per segment or per sub-step.

**Per-robot belief.** `render_frame(..., robot=r)` draws what robot `r` knows (its `RobotView`:
known mask, its rays, its frontiers/segments, the visited records it was told about and its peer
cache) instead of the team union. Under `comms: "full"` every robot is handed the same view, so the
two coincide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

from rlplanner.sim import state as S
from rlplanner.viz import palette as P
from rlplanner.viz.layout import legend_outside
from rlplanner.viz.scene_plot import plot_scene

RAY_MIN_FRAC = 0.08      # shortest drawn ray as a fraction of visual_range
TOKEN_FONTSIZE = 6
SEG_FILL_ALPHA = 0.30    # segments are drawn as translucent patches over the heatmap ...
SEG_EDGE_ALPHA = 0.95    # ... outlined where the labelling changes
PEER_MIN_ALPHA = 0.15    # a peer arrow never fades away completely
LOCAL_CHANNEL_FALLBACK = ("known", "hits", "ray_count") + tuple(f"feat_pc{i}" for i in range(8))


# ---- state accessors ---------------------------------------------------------------------------
def query_names(state: S.EnvState) -> tuple[str, ...]:
    """The live query list: `EnvState.query_names()` when the belief re-derived it, else config."""
    f = getattr(state, "query_names", None)
    if callable(f):
        return tuple(f())
    return tuple(state.cfg.rayfronts.queries)


def token_type_names() -> tuple[str, ...]:
    """Read the sim's token vocabulary at call time — it grows (a "visited" topic is arriving)."""
    return tuple(S.TOKEN_TYPE_NAMES)


def embedding_table(state: S.EnvState):
    """The state's embedding table, or None (a colour derived from features is then skipped)."""
    return getattr(state, "emb", None)


def found_mask(state: S.EnvState) -> np.ndarray:
    """Which humans the team has found (`human_found`, filled once a cell has been voxel-observed
    with the human row `found_hits` times)."""
    n = len(state.scene.humans)
    v = getattr(state, "human_found", None)
    if v is not None:
        a = np.asarray(v, bool).ravel()
        if a.shape[0] != n:
            raise ValueError(f"human_found length {a.shape[0]} != {n} scene humans")
        return a
    return np.zeros(n, bool)


def casualty_mask(state: S.EnvState) -> np.ndarray:
    return np.array([h.role == "casualty" for h in state.scene.humans], bool)


def n_found(state: S.EnvState) -> int:
    """Casualties found so far (`EnvState.n_found` when the state offers it)."""
    v = getattr(state, "n_found", None)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return int((found_mask(state) & casualty_mask(state)).sum())


def _extent(state: S.EnvState) -> tuple[float, float, float, float]:
    r = getattr(state, "raster", None)
    if r is not None and all(hasattr(r, f) for f in ("cell_m", "origin", "nx", "ny")):
        from rlplanner.viz.raster_plot import raster_extent
        return raster_extent(r)
    x0, y0, x1, y1 = state.scene.region
    return x0, x1, y0, y1


# ---- what one robot (or the team) knows --------------------------------------------------------
@dataclass
class BeliefView:
    """The belief a panel draws: the team union, or one robot's `RobotView`.

    Field names follow `sim.tokens.RobotView`; the extras (`visited`, `peers`, `seg_labels`) are
    read with `getattr` so a view that does not carry them yet simply draws nothing for them.
    """
    robot: int | None                 # None = the team union
    known: np.ndarray
    hits: np.ndarray | None
    last_seen: np.ndarray | None
    feat_sum: np.ndarray | None
    frontier_mask: np.ndarray | None
    frontiers: Any
    rays: Any                         # RayTarget list
    segments: Any
    ray_store: Any
    seg_labels: np.ndarray | None = None
    visited: Any = ()                 # visited-target records received by this robot
    peers: Any = ()                   # peer cache (position / target / contact age)
    source: str = "team"              # where the view came from, for the panel title


def _team_view(state: S.EnvState) -> BeliefView:
    return BeliefView(robot=None, known=np.asarray(state.observed, bool),
                      hits=getattr(state, "vox_cnt", None),
                      last_seen=getattr(state, "last_seen_t", None),
                      feat_sum=getattr(state, "vox_feat_sum", None),
                      frontier_mask=getattr(state, "frontier_mask", None),
                      frontiers=getattr(state, "frontier_clusters", None) or [],
                      rays=getattr(state, "ray_targets", None) or [],
                      segments=getattr(state, "segments", None) or [],
                      ray_store=getattr(state, "rays", None),
                      seg_labels=getattr(state, "seg_labels", None), source="team")


def _adapt(v, robot: int, state: S.EnvState, source: str) -> BeliefView:
    """Wrap a sim `RobotView` (whatever it grows) in the fields the panels read."""
    lab = getattr(v, "seg_labels", None)
    if lab is None:
        lab = getattr(state, "seg_labels", None)      # team labelling, masked by what r knows
    return BeliefView(robot=int(robot), known=np.asarray(v.known, bool),
                      hits=getattr(v, "hits", None), last_seen=getattr(v, "last_seen", None),
                      feat_sum=getattr(v, "feat_sum", None),
                      frontier_mask=getattr(v, "frontier_mask", None),
                      frontiers=list(getattr(v, "frontiers", ()) or ()),
                      rays=list(getattr(v, "rays", ()) or ()),
                      segments=list(getattr(v, "segments", ()) or ()),
                      ray_store=getattr(v, "ray_store", None), seg_labels=lab,
                      visited=list(getattr(v, "visited", ()) or ()),
                      peers=list(getattr(v, "peers", ()) or ()), source=source)


def comms_sim(state: S.EnvState):
    """The gossip layer, when this episode has one (`comms.mode != "full"`), else None.

    Reached from whatever the state offers: an attribute the sim publishes, or — while it only
    wires the layer into the belief — the observer callback the layer installed on `rf`.
    """
    rf = getattr(state, "rf", None)
    for c in (getattr(state, "comms", None), getattr(rf, "comms", None),
              getattr(getattr(rf, "on_observe", None), "__self__", None)):
        if c is not None and hasattr(c, "beliefs") and hasattr(c, "links"):
            return c
    return None


def robot_views(state: S.EnvState):
    """The sim's per-robot views when it publishes them, else None (comms full: one team belief)."""
    for owner, attr in ((state, "robot_views"), (state, "views"),
                        (getattr(state, "rf", None), "robot_views"),
                        (getattr(state, "rf", None), "views")):
        v = getattr(owner, attr, None) if owner is not None else None
        if v is not None and len(v):
            return v
    c = comms_sim(state)
    if c is not None:
        rf = getattr(state, "rf", None)
        try:
            return [b.view(rf) for b in c.beliefs]     # the beliefs the last decision refreshed
        except Exception:                              # noqa: BLE001 - mid-edit comms layer
            return None
    return None


def belief_view(state: S.EnvState, robot: int | None = None) -> BeliefView:
    """The belief one panel draws. `robot=None` is the team union.

    A per-robot view comes from the simulator when it publishes one (`state.robot_views`,
    `rf.robot_view(r)`); while `comms == "full"` there is one shared belief and the robot's view
    *is* the team view, which is exactly what the sim hands the token builder.
    """
    if robot is None:
        return _team_view(state)
    r = int(robot)
    n_r = len(state.robots)
    if not (0 <= r < n_r):
        raise IndexError(f"robot={robot!r} outside [0, {n_r}) (state has {n_r} robots)")
    vs = robot_views(state)
    if vs is not None and r < len(vs):
        return _adapt(vs[r], r, state, "robot_views")
    rf = getattr(state, "rf", None)
    f = getattr(rf, "robot_view", None)
    if callable(f):
        return _adapt(f(r), r, state, "rf.robot_view")
    v = _team_view(state)                      # comms full: every robot holds the team belief
    v.robot, v.source = r, "team (comms full)"
    return v


# ---- the one query view a frame takes ----------------------------------------------------------
@dataclass
class QueryView:
    """One query's similarity, taken **once** per frame: the grid and the per-ray colours."""
    idx: int                       # index into `state.query_names()`, -1 when derived
    name: str
    vec: np.ndarray | None         # unit [D] when it had to be embedded
    grid: np.ndarray               # [ny, nx] cosine in [0, 1], 0 where unobserved
    ray_sim: np.ndarray = field(default_factory=lambda: np.zeros(0))   # [n_rays] aligned to store

    @property
    def tag(self) -> str:
        return f"{self.idx}: " if self.idx >= 0 else "derived: "


def resolve_query(state: S.EnvState, query: int | str) -> tuple[int, str]:
    """(index into `query_names()` or -1, label) for a query index or **name**.

    Keys 0-9 index the live mission list, which is 1-8 queries long; a name outside it is legal as
    long as the embedding table can encode it (that is what storing embeddings buys).
    """
    names = query_names(state)
    if isinstance(query, (bool, np.bool_)) or not isinstance(query, (int, np.integer, str)):
        raise TypeError(f"query={query!r}: pass a query index (int) or a query name (str)")
    if isinstance(query, (int, np.integer)):
        i = int(query)
        if not (0 <= i < len(names)):
            raise IndexError(f"query={query!r} outside [0, {len(names)}) "
                             f"(live queries {list(names)})")
        return i, names[i]
    if query in names:
        return names.index(query), query
    if embedding_table(state) is None or getattr(state, "vox_feat_sum", None) is None:
        raise KeyError(f"query {query!r} is not one of {list(names)} and this state carries no "
                       f"embeddings to derive it from")
    return -1, query


def query_view(state: S.EnvState, query: int | str = 0,
               view: BeliefView | None = None) -> QueryView:
    """Take the whole frame's query view in one place: the heatmap and the per-ray similarity.

    `EnvState.vox_sim` allocates one grid *per mission query* on every access, so nothing here
    touches it: `query_sim` gives the single grid being drawn (CONTRACTS.md 9).
    """
    i, label = resolve_query(state, query)
    v = view if view is not None else _team_view(state)
    spec: Any = i if i >= 0 else label
    vec = None
    grid = None
    f = getattr(state, "query_sim", None)
    if callable(f) and v.robot is None:
        grid = np.asarray(f(spec), np.float64)
    else:                                    # a per-robot view: the same cosine over its own cells
        vec = _query_vec(state, spec)
        grid = _cos_grid(v.feat_sum, vec)
    if v.known is not None and grid is not None and grid.shape == v.known.shape:
        grid = np.where(v.known, grid, 0.0)
    return QueryView(idx=i, name=label, vec=vec, grid=grid,
                     ray_sim=_ray_query_sim(state, v, spec))


def _query_vec(state: S.EnvState, spec) -> np.ndarray:
    rf = getattr(state, "rf", None)
    f = getattr(rf, "query_vec", None)
    if callable(f):
        return np.asarray(f(spec), np.float32)
    return np.asarray(S.query_vector(embedding_table(state), query_names(state), spec), np.float32)


def _cos_grid(feat_sum: np.ndarray | None, vec: np.ndarray) -> np.ndarray:
    if feat_sum is None:
        raise AttributeError("query_view: this view carries no feat_sum to derive a query from")
    f = np.asarray(feat_sum, np.float32)
    n = np.linalg.norm(f, axis=-1)
    out = np.zeros(n.shape, np.float64)
    m = n > 1e-9
    out[m] = np.clip((f[m] @ vec) / n[m], 0.0, 1.0)
    return out


def _ray_query_sim(state: S.EnvState, view: BeliefView, spec) -> np.ndarray:
    """Per-ray similarity for the query on screen, from `RayFrontsSim.ray_query_sim` when the
    view is the team belief, else the same cosine over the view's own store."""
    store = view.ray_store
    if store is None or getattr(store, "n", 0) == 0:
        return np.zeros(0)
    rf = getattr(state, "rf", None)
    f = getattr(rf, "ray_query_sim", None)
    if callable(f) and view.robot is None and getattr(rf, "store", None) is not None:
        try:
            a = np.asarray(f(spec), np.float64)
        except Exception:                    # noqa: BLE001 - a store the sim no longer owns
            a = None
        if a is not None and a.shape[0] == store.n:
            return a
    q = _query_vec(state, spec)
    out = None
    for name in ("feat", "feat_peak"):            # `ray_query_sim` is the max of mean and peak
        f = getattr(store, name, None)
        if f is None or not len(f):
            continue
        s = _cos_grid(np.asarray(f, np.float32), q)
        out = s if out is None else np.maximum(out, s)
    return out if out is not None else np.zeros(store.n)


def belief_grid(state: S.EnvState, query: int | str,
                view: BeliefView | None = None) -> tuple[np.ndarray, str, int]:
    """(similarity grid [ny, nx], label, query index or -1) for an index or *any* query name."""
    qv = query_view(state, query, view)
    return qv.grid, qv.name, qv.idx


def _check(state: S.EnvState, query: int | str, focus_robot: int | None,
           robot: int | None) -> None:
    n_r = len(state.robots)
    for name, r in (("focus_robot", focus_robot), ("robot", robot)):
        if r is None:
            continue
        if not isinstance(r, (int, np.integer)) or isinstance(r, (bool, np.bool_)) \
                or not (0 <= int(r) < n_r):
            raise IndexError(f"{name}={r!r} outside [0, {n_r}) (state has {n_r} robots)")
    resolve_query(state, query)


# ---- overlays -------------------------------------------------------------------------------
def draw_robots(ax: Axes, state: S.EnvState, fov: bool = True, traj: bool = True,
                target_line: bool = True, label: bool = True,
                only: int | None = None) -> None:
    cfg = state.cfg
    sensor = getattr(cfg, "sensor", None)
    for r in state.robots:
        if only is not None and r.idx != int(only):
            continue
        col = P.robot_color(r.idx)
        if traj and len(r.trajectory) > 1:
            a = np.asarray(r.trajectory, dtype=float)
            ax.plot(a[:, 0], a[:, 1], color=col, lw=0.9, alpha=P.TRAJ_ALPHA, zorder=10)
        if fov and sensor is not None:
            th = math.degrees(r.heading)
            half = float(sensor.hfov_deg) / 2.0
            if getattr(sensor, "mode", "cone") == "disk":
                th, half = 0.0, 180.0
            ax.add_patch(Wedge(tuple(r.pos), float(sensor.visual_range_m), th - half, th + half,
                               facecolor=col, alpha=P.FOV_FAR_ALPHA, edgecolor="none", zorder=9))
            ax.add_patch(Wedge(tuple(r.pos), float(sensor.depth_limit_m), th - half, th + half,
                               facecolor=col, alpha=P.FOV_ALPHA, edgecolor=col, lw=0.4, zorder=9))
        if target_line and r.target_xy is not None:
            t = np.asarray(r.target_xy, dtype=float)
            if np.all(np.isfinite(t)):
                ax.plot([r.pos[0], t[0]], [r.pos[1], t[1]], color=col, lw=0.7, ls="--", alpha=0.8,
                        zorder=10)
                ax.scatter([t[0]], [t[1]], s=28, marker="x", c=col, linewidths=1.0, zorder=11)
        d = 8.0
        ax.annotate("", xy=(r.pos[0] + d * math.cos(r.heading), r.pos[1] + d * math.sin(r.heading)),
                    xytext=tuple(r.pos), zorder=12,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, shrinkA=0, shrinkB=0))
        ax.scatter([r.pos[0]], [r.pos[1]], s=60, marker="o", c=col, edgecolors="#000000",
                   linewidths=0.6, zorder=12)
        if label:
            ax.annotate(str(r.idx), tuple(r.pos), fontsize=6, color="#ffffff", ha="center",
                        va="center", zorder=13)


def draw_truth_humans(ax: Axes, state: S.EnvState) -> None:
    """Casualties/bystanders coloured by role, ringed by whether the team has found them."""
    scene = state.scene
    found = found_mask(state)
    if found.shape[0] != len(scene.humans):
        raise ValueError(f"found mask length {found.shape[0]} != {len(scene.humans)} scene humans")
    for k, h in enumerate(scene.humans):
        edge = P.HUMAN_STATUS_EDGE["found" if found[k] else "unfound"]
        mk = P.human_marker(h.container)
        kw = dict(linewidths=1.6, path_effects=[pe.withStroke(linewidth=3.0, foreground=edge)]) \
            if mk == "x" else dict(edgecolors=edge, linewidths=1.1)
        ax.scatter([h.pos[0]], [h.pos[1]], s=52, marker=mk, c=P.human_color(h.role), zorder=14,
                   **kw)


def ray_geometry(state: S.EnvState, query: QueryView | int | str = 0,
                 view: BeliefView | None = None):
    """(segments [n, 2, 2], colours, n) for the live rays: bearing from the origin, length grows
    with how often the bearing was observed, colour = the shown query's similarity."""
    v = view if view is not None else _team_view(state)
    qv = query if isinstance(query, QueryView) else query_view(state, query, v)
    rays = v.ray_store
    none = (np.zeros((0, 2, 2)), np.zeros((0, 4)), 0)
    if rays is None or rays.n == 0:
        return none
    live = np.flatnonzero(rays.live())
    if live.size == 0:
        return none
    vmax = float(state.cfg.sensor.visual_range_m)
    sim = qv.ray_sim[live] if qv.ray_sim.shape[0] == rays.n else np.zeros(live.size)
    n_obs = np.asarray(rays.n_obs, float)[live]
    L = vmax * np.clip(n_obs / max(1.0, float(n_obs.max())), RAY_MIN_FRAC, 1.0)
    o = rays.origin_xy[live]
    e = o + (L[:, None] * np.stack([np.cos(rays.az[live]), np.sin(rays.az[live])], axis=1))
    return np.stack([o, e], axis=1), P.SIM_CMAP(np.clip(sim, 0, 1)), int(live.size)


def draw_rays(ax: Axes, state: S.EnvState, query: QueryView | int | str = 0,
              view: BeliefView | None = None) -> int:
    segs, cols, n = ray_geometry(state, query, view)
    if n:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8, alpha=0.75, zorder=7))
    return n


def draw_frontiers(ax: Axes, state: S.EnvState, view: BeliefView | None = None,
                   show_mask: bool = True) -> int:
    """Frontier cells as faint dots plus one disc per cluster, sized by its info gain."""
    v = view if view is not None else _team_view(state)
    if show_mask and v.frontier_mask is not None:
        ij = np.argwhere(np.asarray(v.frontier_mask, dtype=bool))
        if ij.size:
            fx, fy = state.raster.ij_to_xy(ij[:, 0], ij[:, 1])
            ax.scatter(fx, fy, s=1.5, marker="s", c=P.FRONTIER_COLOR, alpha=0.45, linewidths=0,
                       zorder=6)
    cl = list(v.frontiers or [])
    if not cl:
        return 0
    R = float(state.cfg.rayfronts.frontier_ig_radius_m)
    cell = float(getattr(state.raster, "cell_m", state.cfg.raster.cell_m))
    full = max(1.0, math.pi * (R / cell) ** 2)  # cells inside the IG radius
    for c in cl:
        rad = R * min(1.0, math.sqrt(max(0.0, float(c.info_gain)) / full))
        ax.add_patch(Circle(tuple(np.asarray(c.centroid_xy, dtype=float)), max(rad, cell),
                            facecolor=P.alpha(P.FRONTIER_COLOR, 0.10),
                            edgecolor=P.alpha(P.FRONTIER_COLOR, 0.85), lw=0.7, zorder=8))
    return len(cl)


def segment_rgba(state: S.EnvState, view: BeliefView | None = None,
                 fill: float = SEG_FILL_ALPHA, edge: float = SEG_EDGE_ALPHA) -> np.ndarray | None:
    """[ny, nx, 4] overlay: each segment's cells in its **mean-feature colour** (PCA-RGB on the
    embedding table's fixed `pc_basis`), translucent inside and opaque on the labelling's edges.

    A segment is a region of the observed map, so an image is both the honest shape and the cheap
    one: one array instead of a contour per segment.
    """
    v = view if view is not None else _team_view(state)
    emb = embedding_table(state)
    lab = v.seg_labels
    segs = list(v.segments or [])
    if emb is None or lab is None or not segs:
        return None
    lab = np.asarray(lab, np.int64)
    if v.known is not None and lab.shape == v.known.shape:
        lab = np.where(v.known, lab, -1)          # a robot never sees a segment it has not mapped
    n_lab = int(lab.max()) + 1
    if n_lab <= 0:
        return None
    cols = np.zeros((n_lab, 3), np.float64)
    have = np.zeros(n_lab, bool)
    feats = np.stack([np.asarray(s.feat, np.float32) for s in segs])
    rgb = P.feat_rgb(feats, emb)
    for s, c in zip(segs, rgb):
        i, j = int(s.ij[0]), int(s.ij[1])
        if not (0 <= i < lab.shape[0] and 0 <= j < lab.shape[1]):
            continue
        k = int(lab[i, j])                        # the label is looked up at the medoid: the
        if 0 <= k < n_lab:                        # token list is sorted by recency, not by label
            cols[k], have[k] = c, True
    idx = np.clip(lab, 0, n_lab - 1)
    shown = (lab >= 0) & have[idx]
    if not shown.any():
        return None
    out = np.zeros(lab.shape + (4,), np.float32)
    out[..., :3] = cols[idx]
    out[..., 3] = np.where(shown, fill, 0.0)
    b = np.zeros(lab.shape, bool)
    d = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    out[..., 3] = np.where(b & shown, edge, out[..., 3])
    return out


def draw_segments(ax: Axes, state: S.EnvState, view: BeliefView | None = None,
                  medoids: bool = True) -> int:
    v = view if view is not None else _team_view(state)
    rgba = segment_rgba(state, v)
    if rgba is not None:
        ax.imshow(rgba, origin="lower", extent=_extent(state), interpolation="nearest", zorder=1)
    segs = list(v.segments or [])
    if medoids and segs:
        xy = np.array([np.asarray(s.xy, float) for s in segs])
        ax.scatter(xy[:, 0], xy[:, 1], s=26, marker=P.token_marker("segment"), facecolors="none",
                   edgecolors=P.SEGMENT_COLOR, linewidths=0.7, zorder=8)
    return len(segs)


def _peer_idx(name: str, default: int) -> int:
    """Column of the peer block by the sim's own name, so a widened block still reads right."""
    names = getattr(S, "PEER_FEAT_NAMES", ())
    return names.index(name) if name in names else default


def peer_block(state: S.EnvState) -> np.ndarray | None:
    obs = getattr(state, "last_obs", None)
    pt = getattr(obs, "peer_tokens", None) if obs is not None else None
    if pt is None:
        return None
    a = np.asarray(pt, float)
    return a if (a.ndim == 3 and a.shape[2] >= 6) else None


def peer_arrows(state: S.EnvState, robot: int) -> list[tuple[np.ndarray, np.ndarray, float, int]]:
    """(peer xy, its target xy, alpha from contact age, peer index) from `TeamObs.peer_tokens`.

    The block is relative to the observing robot; the offsets are read by the sim's own
    `PEER_FEAT_NAMES` and taken as region-diagonal fractions when they are scaled like the token
    features, in metres otherwise. It is zero-filled while `comms == "full"`: nothing is drawn.
    """
    a = peer_block(state)
    r = int(robot)
    if a is None or r >= a.shape[0]:
        return []
    ix, iy = _peer_idx("dx", 0), _peer_idx("dy", 1)
    tx, ty = _peer_idx("target_dx", 2), _peer_idx("target_dy", 3)
    ia, iv = _peer_idx("contact_age", 4), _peer_idx("valid", 5)
    x0, y0, x1, y1 = state.scene.region
    diag = float(math.hypot(x1 - x0, y1 - y0))
    b = a[r]
    live = b[:, iv] > 0.0
    if not live.any():
        return []
    off = np.abs(b[live][:, [ix, iy, tx, ty]])
    scale = diag if float(np.nanmax(off)) <= 1.5 else 1.0     # normalised block, or raw metres
    pos = np.asarray(state.robots[r].pos, float)
    others = [i for i in range(len(state.robots)) if i != r]
    out = []
    for k in range(b.shape[0]):
        if b[k, iv] <= 0.0:
            continue
        d = np.array([b[k, ix], b[k, iy]], float)
        t = np.array([b[k, tx], b[k, ty]], float)
        if not np.isfinite(np.concatenate([d, t])).all():
            continue
        alpha = float(np.clip(1.0 - abs(float(b[k, ia])), PEER_MIN_ALPHA, 1.0))
        out.append((pos + d * scale, pos + t * scale, alpha,
                    others[k] if k < len(others) else k))
    return out


def draw_peer_tokens(ax: Axes, state: S.EnvState, robot: int) -> int:
    """Small arrows: a peer's last-known position -> the target it reported, faded by contact age."""
    n = 0
    for p, t, alpha, idx in peer_arrows(state, robot):
        col = P.robot_color(idx)
        ax.scatter([p[0]], [p[1]], s=26, marker="o", facecolors="none", edgecolors=col,
                   linewidths=0.9, alpha=alpha, zorder=11)
        if np.all(np.isfinite(t)) and float(np.hypot(*(t - p))) > 1e-6:
            arr = FancyArrowPatch(tuple(p), tuple(t), arrowstyle="-|>", mutation_scale=7,
                                  color=P.PEER_COLOR, lw=0.9, alpha=alpha, shrinkA=0, shrinkB=0,
                                  zorder=11)
            arr.set_clip_on(True)
            ax.add_patch(arr)
            arr.set_clip_box(ax.bbox)
        n += 1
    return n


def comms_mode(state: S.EnvState) -> str:
    """`"full"` (one shared belief) or the gossip mode this episode runs under."""
    c = getattr(state.cfg, "comms", "full")
    return str(c) if isinstance(c, str) else str(getattr(c, "mode", "full"))


def comms_links(state: S.EnvState) -> list[tuple[int, int]]:
    """Pairs of robots currently in contact, or [] when nothing exposes it.

    Read from an explicit adjacency the state may publish, else from the gossip layer, else from
    the peer block's own `link` column. Under `comms: "full"` every pair is always linked, so the
    lines would say nothing and none are drawn.
    """
    if comms_mode(state) == "full":
        return []
    for attr in ("comms_links", "links", "in_range"):
        v = getattr(state, attr, None)
        if v is None:
            continue
        a = np.asarray(v)
        if a.ndim == 2 and a.shape[0] == a.shape[1] == len(state.robots):
            ij = np.argwhere(np.asarray(a, bool))
            return [(int(i), int(j)) for i, j in ij if i < j]
        if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] != len(state.robots):
            return [(int(i), int(j)) for i, j in a]
    c = comms_sim(state)
    if c is not None:
        try:
            adj = np.asarray(c.links(state.robots), bool)
            return [(int(i), int(j)) for i, j in np.argwhere(adj) if i < j]
        except Exception:                        # noqa: BLE001 - mid-edit comms layer
            pass
    b = peer_block(state)
    if b is None or "link" not in getattr(S, "PEER_FEAT_NAMES", ()):
        return []
    il, iv = _peer_idx("link", -1), _peer_idx("valid", 5)
    out = set()
    for r in range(min(b.shape[0], len(state.robots))):
        others = [i for i in range(len(state.robots)) if i != r]
        for k in range(min(b.shape[1], len(others))):
            if b[r, k, iv] > 0.0 and b[r, k, il] > 0.5:
                out.add((min(r, others[k]), max(r, others[k])))
    return sorted(out)


def draw_comms_links(ax: Axes, state: S.EnvState) -> int:
    pairs = comms_links(state)
    if not pairs:
        return 0
    segs = [[tuple(np.asarray(state.robots[i].pos, float)),
             tuple(np.asarray(state.robots[j].pos, float))] for i, j in pairs]
    ax.add_collection(LineCollection(segs, colors=[P.LINK_COLOR] * len(segs), linewidths=0.9,
                                     linestyles=":", alpha=0.8, zorder=10))
    return len(pairs)


def draw_visited(ax: Axes, state: S.EnvState, view: BeliefView) -> int:
    """Visited-target records this robot holds (its own plus whatever peers gossiped to it)."""
    rec = list(view.visited or [])
    if not rec:
        return 0
    xy = []
    for v in rec:
        p = getattr(v, "xy", None)
        if p is None and isinstance(v, dict):
            p = v.get("xy")
        if p is not None and np.all(np.isfinite(np.asarray(p, float))):
            xy.append(np.asarray(p, float)[:2])
    if not xy:
        return 0
    a = np.asarray(xy, float)
    ax.scatter(a[:, 0], a[:, 1], s=44, marker=P.token_marker("visited"), facecolors="none",
               edgecolors=P.VISITED_COLOR, linewidths=1.0, zorder=11)
    return a.shape[0]


def draw_tokens(ax: Axes, state: S.EnvState, focus_robot: int) -> int:
    """Numbered valid tokens of one robot; the chosen action gets a red ring."""
    obs = state.last_obs
    if obs is None:
        return 0
    r = int(focus_robot)
    if r >= obs.token_mask.shape[0]:
        raise IndexError(f"focus_robot={r} outside last_obs token_mask "
                         f"[0, {obs.token_mask.shape[0]})")
    chosen = -1
    if (state.last_actions is not None and np.ndim(state.last_actions) == 1
            and r < len(state.last_actions)):
        chosen = int(state.last_actions[r])     # waypoint actions are [n, 2]: no token to ring
    n = 0
    for k in np.flatnonzero(obs.token_mask[r]):
        xy = np.asarray(obs.token_xy[r, k], dtype=float)
        if not np.all(np.isfinite(xy)):
            continue
        tt = int(obs.token_type[r, k])
        col = P.token_color(tt)
        ax.scatter([xy[0]], [xy[1]], s=70, marker=P.token_marker(tt), facecolors="none",
                   edgecolors=col, linewidths=1.1, zorder=15)
        ax.annotate(str(int(k)), xy, fontsize=TOKEN_FONTSIZE, color=col, ha="center", va="center",
                    zorder=16, bbox=dict(boxstyle="circle,pad=0.14", fc="#000000cc", ec=col,
                                         lw=0.5))
        if int(k) == chosen:
            ax.scatter([xy[0]], [xy[1]], s=210, marker="o", facecolors="none",
                       edgecolors=P.TOKEN_CHOSEN_EDGE, linewidths=1.4, zorder=15)
        n += 1
    return n


# ---- the actor's dense input -------------------------------------------------------------------
def local_channels() -> tuple[str, ...]:
    try:
        from rlplanner.sim.tokens import LOCAL_CHANNELS
        return tuple(LOCAL_CHANNELS)
    except Exception:                       # noqa: BLE001 - mid-edit sim
        return LOCAL_CHANNEL_FALLBACK


def local_crop(state: S.EnvState, robot: int, view: BeliefView | None = None):
    """(known [S, S], feat PC1-3 as RGB [S, S, 3], size_m) — the ego crop the actor plans in.

    Prefers `TeamObs.local` (the very array the network reads); falls back to cropping the robot's
    own belief when `tokens.local_size = 0` disabled it, so the panel works either way.
    """
    r = int(robot)
    v = view if view is not None else belief_view(state, r)
    emb = embedding_table(state)
    cell = float(getattr(state.raster, "cell_m", state.cfg.raster.cell_m))
    obs = getattr(state, "last_obs", None)
    loc = getattr(obs, "local", None) if obs is not None else None
    names = local_channels()
    if loc is not None and np.asarray(loc).ndim == 4 and r < np.asarray(loc).shape[0]:
        a = np.asarray(loc, np.float32)[r]
        known = a[names.index("known") if "known" in names else 0] > 0.5
        k0 = names.index("feat_pc0") if "feat_pc0" in names else 3
        pc = (np.moveaxis(a[k0:k0 + 3], 0, -1) if a.shape[0] >= k0 + 3
              else np.zeros(known.shape + (3,)))
        rgb = _pc_rgb(pc, emb)
        return known, np.where(known[..., None], rgb, 0.0), a.shape[-1] * cell
    s = int(getattr(state.cfg.tokens, "local_size", 0) or 64)
    ras = state.raster
    i, j = (int(x) for x in ras.xy_to_ij(*np.asarray(state.robots[r].pos, float)))
    half = s // 2
    known = np.zeros((s, s), bool)
    rgb = np.zeros((s, s, 3), np.float64)
    i0, j0 = i - half, j - half
    gi0, gj0 = max(0, i0), max(0, j0)
    gi1, gj1 = min(ras.ny, i0 + s), min(ras.nx, j0 + s)
    if gi1 > gi0 and gj1 > gj0 and v.known is not None:
        si, sj = gi0 - i0, gj0 - j0
        h, w = gi1 - gi0, gj1 - gj0
        k = np.asarray(v.known, bool)[gi0:gi1, gj0:gj1]
        known[si:si + h, sj:sj + w] = k
        if v.feat_sum is not None and emb is not None:
            f = np.asarray(v.feat_sum, np.float32)[gi0:gi1, gj0:gj1]
            n = np.linalg.norm(f, axis=-1, keepdims=True)
            col = P.feat_rgb(f / np.maximum(n, 1e-12), emb)
            rgb[si:si + h, sj:sj + w] = np.where(k[..., None], col, 0.0)
    return known, rgb, s * cell


def _pc_rgb(pc: np.ndarray, emb) -> np.ndarray:
    """Raw PC1-3 coordinates (as the local/BEV channels carry them) -> RGB on the fixed scale."""
    if emb is None:
        a = np.asarray(pc, np.float64)
        lo, hi = a.min(), a.max()
        return np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    lo, hi = P.pc_scale(emb)
    return np.clip((np.asarray(pc, np.float64) - lo) / (hi - lo), 0.0, 1.0)


def draw_local_panel(ax: Axes, state: S.EnvState, robot: int,
                     view: BeliefView | None = None) -> None:
    """The focused robot's local crop: feature PC1-3 as RGB, dimmed where nothing is known."""
    known, rgb, size_m = local_crop(state, robot, view)
    img = np.where(known[..., None], rgb, 0.06)
    ax.imshow(img, origin="lower", interpolation="nearest",
              extent=(-size_m / 2, size_m / 2, -size_m / 2, size_m / 2), zorder=0)
    ax.contour(np.asarray(known, float), levels=[0.5], colors=["#ffffff"], linewidths=0.6,
               extent=(-size_m / 2, size_m / 2, -size_m / 2, size_m / 2), origin="lower", zorder=2)
    ax.scatter([0.0], [0.0], s=40, marker="o", c=P.robot_color(int(robot)), edgecolors="#ffffff",
               linewidths=0.7, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("m (ego)", fontsize=7)
    ax.set_title(f"robot {int(robot)} local crop  {known.shape[0]}x{known.shape[1]}  "
                 f"feat PC1-3 + known  ({100.0 * known.mean():.0f}% known)", fontsize=8)


# ---- panels ----------------------------------------------------------------------------------
def draw_class_legend(ax: Axes, classes=None, side: str = "right", ncols: int = 2,
                      fontsize: float = 4.6, anchor: tuple[float, float] | None = None) -> None:
    """Swatch legend of the raster class colours, outside the axes and without replacing another."""
    names = tuple(classes) if classes is not None else tuple(P.CLASS_COLORS)
    handles = [Line2D([], [], ls="", marker="s", markersize=4, color=P.class_color(n),
                      markeredgecolor="#00000060", markeredgewidth=0.3) for n in names]
    legend_outside(ax, handles, labels=list(names), side=side, ncols=ncols, fontsize=fontsize,
                   title="raster class", artist=True, anchor=anchor, handlelength=0.8,
                   handletextpad=0.25, columnspacing=0.5, borderpad=0.25, labelspacing=0.18)


def truth_legend_handles() -> list[Line2D]:
    """Human role/found keys of the ground-truth panel."""
    return [
        Line2D([], [], ls="", marker="o", color=P.human_color("casualty"), markersize=6,
               markeredgecolor=P.HUMAN_STATUS_EDGE["unfound"], label="casualty"),
        Line2D([], [], ls="", marker="o", color=P.human_color("bystander"), markersize=6,
               markeredgecolor=P.HUMAN_STATUS_EDGE["unfound"], label="bystander"),
        Line2D([], [], ls="", marker="o", color="#ffffff", markersize=6,
               markeredgecolor=P.HUMAN_STATUS_EDGE["found"], markeredgewidth=1.4, label="found"),
    ]


def token_legend_handles() -> list[Line2D]:
    """One key per token type, named and marked by the sim's own `TOKEN_TYPE_NAMES`."""
    return [Line2D([], [], ls="", marker=P.token_marker(n), markerfacecolor="none",
                   markeredgecolor=P.token_color(n), color="none", markersize=6,
                   label=f"token: {n}") for n in token_type_names()]


def belief_legend_handles(tokens: bool = True, peers: bool = False) -> list[Line2D]:
    """Keys for what RayFronts publishes: cells, rays, frontiers, segments (plus the tokens)."""
    h = [
        Line2D([], [], ls="", marker="s", color=P.SIM_CMAP(0.85), markersize=6,
               label="cell (query sim)"),
        Line2D([], [], ls="", marker="s", color=P.UNOBSERVED, markeredgecolor="#666a75",
               markersize=6, label="unobserved"),
        Line2D([], [], ls="-", color=P.SIM_CMAP(0.9), lw=1.2, label="ray (query sim)"),
        Line2D([], [], ls="", marker="o", markerfacecolor=P.alpha(P.FRONTIER_COLOR, 0.3),
               markeredgecolor=P.FRONTIER_COLOR, color="none", markersize=7,
               label="frontier cluster"),
        Line2D([], [], ls="", marker="s", color=P.FRONTIER_COLOR, markersize=3,
               label="frontier cell"),
        Line2D([], [], ls="", marker="s", markerfacecolor=P.alpha(P.SEGMENT_COLOR, 0.45),
               markeredgecolor=P.SEGMENT_COLOR, color="none", markersize=7,
               label="segment (feature colour)"),
    ]
    if peers:
        h += [Line2D([], [], ls="-", color=P.PEER_COLOR, lw=1.0, label="peer -> its target"),
              Line2D([], [], ls=":", color=P.LINK_COLOR, lw=1.2, label="comms link")]
    return h + token_legend_handles() if tokens else h


def draw_truth_panel(ax: Axes, state: S.EnvState, class_legend: bool = True) -> None:
    plot_scene(state.scene, ax=ax, show_damage=True, show_humans=False, legend=False,
               title=f"ground truth   t={state.t:.0f}s")
    draw_truth_humans(ax, state)
    draw_robots(ax, state)
    draw_comms_links(ax, state)
    legend_outside(ax, truth_legend_handles(), side="right", ncols=1, fontsize=5.5,
                   handlelength=1.0)
    if class_legend:
        draw_class_legend(ax, anchor=(1.01, 0.80))


def draw_belief_panel(ax: Axes, state: S.EnvState, query_idx: int | str = 0,
                      focus_robot: int | None = None, legend: bool = True,
                      query: int | str | None = None, robot: int | None = None,
                      segments: bool = True) -> dict[str, int]:
    """`query_idx` (or the `query` alias) is a query index or **name**; `robot` picks whose map."""
    ext = _extent(state)
    view = belief_view(state, robot)
    qv = query_view(state, query_idx if query is None else query, view)
    obs = np.asarray(view.known, dtype=bool)
    if qv.grid.shape != obs.shape:
        raise ValueError(f"query_sim grid {qv.grid.shape} != known shape {obs.shape}")
    ax.imshow(P.sim_rgb(qv.grid, obs), origin="lower", extent=ext, interpolation="nearest",
              zorder=0)
    counts = {"segments": draw_segments(ax, state, view) if segments else 0,
              "frontiers": draw_frontiers(ax, state, view),
              "rays": draw_rays(ax, state, qv, view)}
    counts["visited"] = draw_visited(ax, state, view)
    counts["links"] = draw_comms_links(ax, state)
    counts["peers"] = draw_peer_tokens(ax, state, robot) if robot is not None else 0
    draw_robots(ax, state, fov=True, traj=True, target_line=True)
    counts["tokens"] = draw_tokens(ax, state, focus_robot) if focus_robot is not None else 0
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x [m]", fontsize=8)
    foc = "none" if focus_robot is None else str(int(focus_robot))
    head = "belief (team)" if robot is None else f"robot {int(robot)}'s map"
    ax.set_title(f"{head}  q[{qv.tag}{qv.name!r}]   tokens of robot {foc}", fontsize=9)
    if not legend:
        return counts
    legend_outside(ax, belief_legend_handles(peers=robot is not None), side="right", ncols=1,
                   fontsize=5.5)
    return counts


def robot_target_label(state: S.EnvState, r: S.RobotState) -> str:
    """`<token type>#<id>` for the robot's goal, falling back to the token it last chose."""
    names = token_type_names()

    def name(tt) -> str:
        tt = int(tt)
        return names[tt] if 0 <= tt < len(names) else "?"

    if r.target_xy is not None:
        return f"{name(r.target_token_type)}#{r.target_id}"
    obs, act = state.last_obs, state.last_actions
    if obs is not None and act is not None and np.ndim(act) == 1 and r.idx < len(act):
        k = int(act[r.idx])
        if 0 <= k < obs.token_type.shape[1]:
            return f"{name(obs.token_type[r.idx, k])}#{int(obs.token_id[r.idx, k])}"
    return "-"


def status_lines(state: S.EnvState, counts: dict[str, int] | None = None,
                 query_idx: int | str = 0, focus_robot: int | None = None,
                 robot: int | None = None) -> list[str]:
    cfg = state.cfg
    lines = [
        f"t          {state.t:7.1f} s / {cfg.t_max_s:.0f}",
        f"decision   {state.decision_idx}",
        f"found      {n_found(state)} / {state.n_casualties} casualties",
        f"coverage   {100.0 * state.coverage:5.1f} %",
        f"reward     {state.cum_reward:+.3f}",
        "",
    ]
    names = query_names(state)
    lines += [f"queries    {len(names)}"]
    for i, q in enumerate(names):
        star = "*" if (isinstance(query_idx, str) and q == query_idx) or \
                      (not isinstance(query_idx, str) and i == int(query_idx)) else " "
        lines.append(f" {star}{i} {q}")
    lines.append("")
    if counts:
        lines += [f"frontiers  {counts.get('frontiers', 0)}   segments {counts.get('segments', 0)}",
                  f"live rays  {counts.get('rays', 0)}   tokens {counts.get('tokens', 0)}"]
        if counts.get("peers") or counts.get("links") or counts.get("visited"):
            lines.append(f"peers      {counts.get('peers', 0)}   links "
                         f"{counts.get('links', 0)}   visited {counts.get('visited', 0)}")
        lines.append("")
    lines.append(f"map        {'team union' if robot is None else f'robot {int(robot)}'}")
    lines.append("")
    lines.append("robot   target        dist   action")
    for r in state.robots:
        star = "*" if focus_robot is not None and r.idx == int(focus_robot) else " "
        lines.append(f"{star}{r.idx:<5d} {robot_target_label(state, r):<13.13s} "
                     f"{r.dist_travelled:6.0f}m  {r.last_action}")
    m = state.metrics or {}
    if m:
        lines += ["", "metrics"]
        for k in ("time_to_first", "time_to_half", "frac_found", "finds_auc", "redundancy"):
            if k in m:
                v = m[k]
                lines.append(f"  {k:<14s} {v:.3f}" if isinstance(v, float) else
                             f"  {k:<14s} {v}")
    return lines


def draw_text_panel(ax: Axes, state: S.EnvState, counts: dict[str, int] | None = None,
                    query_idx: int | str = 0, focus_robot: int | None = None,
                    robot: int | None = None) -> None:
    ax.axis("off")
    ax.set_title("status", fontsize=9)
    txt = "\n".join(status_lines(state, counts, query_idx, focus_robot, robot))
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.0,
            family="monospace")


# ---- public entry point ------------------------------------------------------------------------
def frame_figure(state: S.EnvState, query_idx: int | str = 0, focus_robot: int | None = None,
                 figsize: tuple[float, float] = (18.0, 6.0), dpi: int = 100,
                 robot: int | None = None, show_local: bool = False) -> Figure:
    """The three-panel figure (four with `show_local`), on an Agg canvas.

    Margins leave room for the outside legends: no layout engine, because with equal-aspect axes
    constrained layout re-expands them over a legend placed below.
    """
    _check(state, query_idx, focus_robot, robot)
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ratios = [1.0, 1.0, 0.5] if not show_local else [1.0, 1.0, 0.45, 0.5]
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.34,
                          left=0.035, right=0.99, top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(ratios))]
    draw_truth_panel(axes[0], state)
    counts = draw_belief_panel(axes[1], state, query_idx=query_idx, focus_robot=focus_robot,
                               robot=robot)
    if show_local:
        r = focus_robot if focus_robot is not None else (robot if robot is not None else 0)
        draw_local_panel(axes[2], state, int(r))
    draw_text_panel(axes[-1], state, counts, query_idx=query_idx, focus_robot=focus_robot,
                    robot=robot)
    return fig


def render_frame(state: S.EnvState, query_idx: int | str = 0, focus_robot: int | None = None,
                 figsize: tuple[float, float] = (18.0, 6.0), dpi: int = 100,
                 query: int | str | None = None, robot: int | None = None,
                 show_local: bool = False) -> np.ndarray:
    """Render one episode frame to an RGB uint8 array [H, W, 3].

    `query_idx` (or `query`) is a query index or any query name the embedding table knows;
    `robot` draws that robot's own map instead of the team union.
    """
    fig = frame_figure(state, query_idx if query is None else query, focus_robot, figsize, dpi,
                       robot=robot, show_local=show_local)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    fig.clear()
    return rgb


__all__ = ["render_frame", "frame_figure", "robot_target_label", "belief_grid", "resolve_query",
           "query_view", "QueryView", "belief_view", "BeliefView", "robot_views", "comms_sim",
           "query_names",
           "token_type_names", "embedding_table", "draw_class_legend", "truth_legend_handles",
           "belief_legend_handles", "token_legend_handles", "draw_truth_panel",
           "draw_belief_panel", "draw_text_panel", "draw_robots", "draw_rays", "ray_geometry",
           "draw_frontiers", "draw_segments", "segment_rgba", "draw_tokens", "draw_truth_humans",
           "draw_peer_tokens", "peer_arrows", "peer_block", "draw_comms_links", "comms_links",
           "comms_mode",
           "draw_visited",
           "local_crop", "local_channels", "draw_local_panel", "status_lines", "found_mask",
           "casualty_mask", "n_found"]
