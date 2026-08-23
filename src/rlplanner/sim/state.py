"""Shared state/observation containers. The simulator fills these; the visualizer, metrics and
tests read them. Keep them plain (dataclasses + numpy) — no behaviour beyond light helpers.

Array layout: all (ny, nx) grids are row-major with row index = y cell, column index = x cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..scene.schema import HUMAN_ROLES

CASUALTY_ROLE_ID = HUMAN_ROLES.index("casualty")

# token types (one-hot index 0..4 in the token feature vector); the candidate set is the three
# RayFronts topics, the team's own visited-target records and "stay where you are".
TOKEN_HOLD, TOKEN_FRONTIER, TOKEN_RAY, TOKEN_SEGMENT, TOKEN_VISITED = 0, 1, 2, 3, 4
TOKEN_TYPE_NAMES = ("hold", "frontier", "ray", "segment", "visited")
N_TOKEN_TYPES = len(TOKEN_TYPE_NAMES)

# per-robot token id namespace under decentralised comms: robot r's ids are offset by
# r * TOKEN_ID_STRIDE, so two robots' private maps never collide on a `(type, id)` claim key.
TOKEN_ID_STRIDE = 1 << 20

# Token feature layout. F = TOKEN_FIXED + D, where D = rayfronts.embedding_dim: the tail of every
# token is the item's own RayFronts *feature*, never a per-query score. Relevance to the mission is
# the policy's job (it attends over the query tokens), not the simulator's.
F_DX, F_DY, F_DIST, F_BSIN, F_BCOS = 5, 6, 7, 8, 9
F_XABS, F_YABS = 10, 11
F_SIZE, F_HITS, F_CONF, F_AGE, F_NOBS = 12, 13, 14, 15, 16
F_ORIGIN_DX, F_ORIGIN_DY, F_AZ_SIN, F_AZ_COS, F_EL_SIN, F_EL_COS, F_RANGE = 17, 18, 19, 20, 21, 22, 23
F_COV, F_RAYS, F_CLAIM, F_PEER, F_REACH = 24, 25, 26, 27, 28
# the visited block, zero for every other token type: how many casualties the visit turned up,
# whether this robot made the visit itself, and which robot did
F_FOUND, F_OWN, F_WHO = 29, 30, 31
F_FEAT0 = 32
TOKEN_FIXED = F_FEAT0

# one slot per peer, filled from the robot's peer cache (what gossip last told it about that
# robot); `valid` is 0 for a peer never heard from and for the padding of a smaller team.
PEER_FEAT_NAMES = (("dx", "dy", "target_dx", "target_dy", "contact_age", "valid", "dist",
                    "coverage", "link") + tuple(f"target_{n}" for n in TOKEN_TYPE_NAMES))
PEER_DX, PEER_DY, PEER_TDX, PEER_TDY, PEER_AGE, PEER_VALID = 0, 1, 2, 3, 4, 5
PEER_DIST, PEER_COV, PEER_LINK, PEER_TYPE0 = 6, 7, 8, 9
PEER_FEAT_DIM = len(PEER_FEAT_NAMES)


def token_feature_names(dim: int) -> list[str]:
    """Column names of a token vector whose feature tail is `dim` wide."""
    names = [f"type_{n}" for n in TOKEN_TYPE_NAMES] + [
             "dx", "dy", "dist", "bearing_sin", "bearing_cos", "x_abs", "y_abs",
             "n_cells", "hits", "confidence", "age", "n_obs",
             # the ray geometry block is zero for the other token types; it is the raw ray topic
             # (origin, azimuth, elevation) so a policy can see for itself when rays point at one place
             "origin_dx", "origin_dy", "az_sin", "az_cos", "el_sin", "el_cos", "range_est",
             "coverage_nbhd", "ray_count", "claimed_by_peer", "peer_dist_min", "reachable",
             "visit_found", "visit_own", "visit_who"]
    names += [f"feat{i}" for i in range(int(dim))]
    assert len(names) == TOKEN_FIXED + int(dim)
    return names


ROBOT_FEAT_DIM = 8 + 10  # x, y, alt, heading_sin, heading_cos, t_frac, 1-t_frac, coverage + robot one-hot(10)


@dataclass
class TeamObs:
    """One decision's observation.

    CTDE split (CONTRACTS.md 5): `tokens`, `local`, `peer_tokens`, `robot_bev` and the query block
    are the *actor's* view and are shaped per robot; `bev` is the compressed global belief the
    centralised critic reads. Under `comms.mode == "full"` every robot's per-robot input is filled
    from the one team map, so there the split is a shape rather than a restriction; under
    `comms.mode == "range"` each robot's block is a function of its own belief plus gossip.
    """
    tokens: np.ndarray        # float32 [n_robots, K, F]
    token_mask: np.ndarray    # bool    [n_robots, K]   True = selectable
    token_xy: np.ndarray      # float32 [n_robots, K, 2] world xy of the token target (nan if none)
    token_type: np.ndarray    # int8    [n_robots, K]
    token_id: np.ndarray      # int32   [n_robots, K]  stable id of the frontier/ray/segment (-1 none)
    robot_feat: np.ndarray    # float32 [n_robots, ROBOT_FEAT_DIM]
    bev: np.ndarray           # float32 [C, Hb, Wb] compressed global BEV (see tokens.BEV_CHANNELS)
    query_emb: np.ndarray     # float32 [Qmax, D] unit mission-query embeddings, zero-padded
    query_w: np.ndarray       # float32 [Qmax] per-query weight (1.0 for a mission query)
    query_mask: np.ndarray    # bool    [Qmax] True = a real query
    t: float
    local: np.ndarray | None = None        # float32 [n_robots, Cl, S, S] ego-centric local crop
    peer_tokens: np.ndarray | None = None  # float32 [n_robots, n_robots - 1, PEER_FEAT_DIM]
    robot_bev: np.ndarray | None = None    # float32 [n_robots, C, Hr, Wr] BEV of each robot's own
                                           # belief (actor side); None when robot_bev_size = 0

    @property
    def n_robots(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_queries(self) -> int:
        return int(self.query_mask.sum())


@dataclass
class RobotState:
    idx: int
    pos: np.ndarray               # float64 [2] world xy
    alt: float
    heading: float                # radians, direction of last motion
    target_xy: np.ndarray | None  # current goal or None
    target_token_type: int        # TOKEN_* of the current goal (TOKEN_HOLD if none)
    target_id: int                # token_id of current goal (-1 none)
    path: list[tuple[float, float]] = field(default_factory=list)  # remaining A* waypoints (world xy)
    dist_travelled: float = 0.0
    trajectory: list[tuple[float, float]] = field(default_factory=list)  # xy per sub-step
    last_action: int = -1
    target_feat: np.ndarray | None = None  # float32 [D] feature of the token it is flying to


@dataclass
class RayStore:
    """Struct-of-arrays ray memory. Length N = number of live + resolved rays (compact periodically).

    Semantics live in `feat`/`feat_peak`; query similarities are a *view* obtained on demand from
    `RayFrontsSim.ray_query_sim`, never a stored per-query column.
    """
    origin_xy: np.ndarray     # float64 [N, 2]
    az: np.ndarray            # float64 [N] radians, direction of the bin's most salient observation
    el: np.ndarray            # float64 [N] radians (negative = below the horizon)
    conf: np.ndarray          # float32 [N]
    n_obs: np.ndarray         # int32   [N]
    t_first: np.ndarray       # float64 [N]
    t_last: np.ndarray        # float64 [N]
    ids: np.ndarray           # int32   [N] stable ids
    resolved: np.ndarray      # bool    [N]
    feat: np.ndarray | None = None      # float32 [N, D] weighted mean of the observation features
    feat_peak: np.ndarray | None = None # float32 [N, D] the most salient single observation feature

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])

    def live(self) -> np.ndarray:
        return ~self.resolved


@dataclass
class RayTarget:
    """One live ray as the policy sees it: the raw bin plus the point its bearing aims at."""
    id: int
    ray_idx: int
    xy: np.ndarray            # float64 [2] origin + bearing * range_m, clipped to the region
    origin_xy: np.ndarray     # float64 [2]
    az: float
    el: float
    range_m: float            # alt / tan(-el), clipped to [depth_limit, visual_range]
    feat: np.ndarray          # float32 [D] the bin's *most salient* observation (see feat_mean)
    feat_mean: np.ndarray     # float32 [D] the weighted running mean over the bin's looks
    conf: float
    n_obs: int
    t_first: float
    t_last: float


@dataclass
class SegmentToken:
    """A spatially connected region of the observed map whose cells share a feature.

    Produced by a generic over-segmentation (`sim/segments.py`) with one scale parameter — no
    query, no class, no ranking. What it is *worth* is for the policy to work out from `feat`.
    """
    id: int
    xy: np.ndarray            # float64 [2] medoid cell centre
    ij: tuple[int, int]
    feat: np.ndarray          # float32 [D] mean of the member cells' unit features
    n_cells: int
    mean_hits: float          # mean vox_cnt over the member cells
    ray_count: int            # live rays whose corridor crosses the segment
    t_first: float
    t_last: float


@dataclass
class FrontierCluster:
    id: int
    centroid_xy: np.ndarray   # float64 [2]
    n_cells: int
    info_gain: float          # unobserved cells within frontier_ig_radius_m
    cell_ij: np.ndarray       # int32 [m, 2] member cells (row, col)


@dataclass
class VisitRecord:
    """A target a robot chose and arrived at (CONTRACTS.md 5 / DESIGN_VARIANTS.md C).

    Bookkeeping of the robot's own actions, not an inference about the place: the feature is the
    one the chosen token carried, `n_found` counts the casualties that turned up within
    `reward.revisit_m` of it during the decision the robot arrived in.
    """
    xy: np.ndarray            # float64 [2] the target the robot flew to
    token_type: int           # TOKEN_RAY | TOKEN_SEGMENT
    token_id: int             # id of the token it chose (in the chooser's own id space)
    feat: np.ndarray          # float32 [D] the chosen token's feature
    t: float
    robot: int                # who visited
    seq: int                  # that robot's visit counter; (robot, seq) is the gossip key
    n_found: int = 0
    id: int = -1              # env-wide record id; the token id every robot uses for this record

    @property
    def key(self) -> tuple[int, int]:
        return (int(self.robot), int(self.seq))


@dataclass
class PeerRecord:
    """What a robot last heard about one peer. Always-on gossip payload: position, current target,
    contact time and the peer's reported coverage fraction."""
    robot: int
    pos: np.ndarray           # float64 [2]
    target_xy: np.ndarray | None
    target_type: int
    t_contact: float
    coverage: float = 0.0
    linked: bool = False      # in contact at this decision (vs. remembered from an earlier one)


@dataclass
class Event:
    t: float
    kind: str                 # "found" | "arrive" | "unreachable" | "decision"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvState:
    t: float
    decision_idx: int
    scene: Any                # rlplanner.scene.schema.Scene
    raster: Any               # rlplanner.sim.raster.Raster
    cfg: Any                  # EnvConfig
    robots: list[RobotState]
    observed: np.ndarray      # bool   [ny, nx] (vox_cnt > 0)
    vox_cnt: np.ndarray       # int32  [ny, nx]
    last_seen_t: np.ndarray   # float32 [ny, nx] (-1 where unobserved)
    rays: RayStore
    ray_targets: list[RayTarget]
    frontier_mask: np.ndarray # bool   [ny, nx]
    frontier_clusters: list[FrontierCluster]
    segments: list[SegmentToken]
    seg_labels: np.ndarray    # int32 [ny, nx] segment index per cell (-1 = unobserved / unassigned)
    human_hits: np.ndarray    # int32 [n_humans] voxel observations of that human's cell with its own class row
    human_found: np.ndarray   # bool  [n_humans] hits >= rayfronts.found_hits
    last_obs: TeamObs | None
    last_actions: np.ndarray | None
    cum_reward: float
    events: list[Event]
    metrics: dict[str, Any]   # running metrics (see CONTRACTS.md)
    observable: np.ndarray | None = None  # bool [ny, nx] cells any robot could ever observe (None = all)
    vox_feat_sum: np.ndarray | None = None  # float32 [ny, nx, D] sum of the unit observation features
    emb: Any = None           # sim.embeddings.EmbeddingTable behind vox_feat / rays.feat
    queries: tuple[str, ...] | None = None  # live mission-query names (cfg after set_queries)
    rf: Any = None            # the live RayFrontsSim, for the lazy query views
    robot_views: Any = None   # list[tokens.RobotView] under range comms, None under full comms
    visits: Any = None        # list[VisitRecord]: every visited target the team has recorded
    comms_links: np.ndarray | None = None   # bool [n, n] i's message reaches j this decision

    # convenience -----------------------------------------------------------------------------
    @property
    def vox_feat(self) -> np.ndarray:
        """float32 [ny, nx, D] unit per-cell features (zeros where unobserved). Allocates."""
        if self.vox_feat_sum is None:
            raise AttributeError("EnvState.vox_feat: this state carries no vox_feat_sum")
        n = np.linalg.norm(self.vox_feat_sum, axis=-1, keepdims=True)
        return self.vox_feat_sum / np.maximum(n, 1e-12)

    def query_names(self) -> tuple[str, ...]:
        return tuple(self.queries) if self.queries else tuple(self.cfg.rayfronts.queries)

    def query_sim(self, query) -> np.ndarray:
        """float32 [ny, nx] cosine of the voxel features against one query (name, index or vector).

        **Lazy and off the hot path** — nothing in `step` calls it. Viewers, tests and the
        threshold-free baselines ask for the view they want, when they want it.
        """
        if self.rf is not None:
            return self.rf.query_sim(query)
        return _query_sim(self.vox_feat_sum, self.emb, self.query_names(), query)

    @property
    def vox_sim(self) -> np.ndarray:
        """float32 [Q, ny, nx] cosines against the live mission queries. Lazily computed and
        allocated on every access: the per-step path never touches it."""
        return np.stack([self.query_sim(i) for i in range(len(self.query_names()))])

    @property
    def n_casualties(self) -> int:
        return int(sum(1 for h in self.scene.humans if h.role == "casualty"))

    @property
    def n_found(self) -> int:
        """Casualties whose cell has been voxel-observed with the human row `found_hits` times."""
        if not self.human_found.size:
            return 0
        cas = self.raster.humans["role_id"] == CASUALTY_ROLE_ID
        return int(np.count_nonzero(self.human_found & cas))

    @property
    def raw_coverage(self) -> float:
        return float(self.observed.mean())

    @property
    def coverage(self) -> float:
        """Observed fraction of the observable cells (matches info['coverage'])."""
        if self.observable is None or not self.observable.any():
            return self.raw_coverage
        return float(self.observed[self.observable].mean())


def query_vector(emb, names: tuple[str, ...], query) -> np.ndarray:
    """Resolve a query given as a name, an index into `names`, or a raw [D] vector -> unit [D]."""
    if isinstance(query, str):
        if emb is None:
            raise ValueError(f"query_vector: no embedding table to encode {query!r}")
        return np.asarray(emb.embed_queries((query,))[0], np.float32)
    if isinstance(query, (int, np.integer)):
        i = int(query)
        if not (0 <= i < len(names)):
            raise IndexError(f"query_vector: index {i} outside the {len(names)} live queries "
                             f"{list(names)!r}")
        return query_vector(emb, names, names[i])
    v = np.asarray(query, np.float32).reshape(-1)
    if emb is not None and v.shape[0] != emb.D:
        raise ValueError(f"query_vector: vector of dim {v.shape[0]} != table dim {emb.D}")
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _query_sim(feat_sum: np.ndarray, emb, names: tuple[str, ...], query) -> np.ndarray:
    """Cosine of a stored feature grid against one query, clipped to [0, 1] (0 where unobserved)."""
    if feat_sum is None:
        raise AttributeError("query_sim: this state carries no vox_feat_sum")
    q = query_vector(emb, names, query)
    f = np.asarray(feat_sum, np.float32)
    n = np.linalg.norm(f, axis=-1)
    out = (f @ q) / np.maximum(n, 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
