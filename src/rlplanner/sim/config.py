"""Environment configuration. All distances in metres, times in seconds, angles in degrees."""
from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The *mission* queries: what this episode is looking for. They are an input to the policy
# (query tokens), never a list the simulator scans the map against. LLM-proposed hints are later
# appended to this list with their own weights.
DEFAULT_QUERIES: tuple[str, ...] = ("person lying on the ground", "person")


@dataclass
class RasterConfig:
    cell_m: float = 2.0


@dataclass
class RobotConfig:
    n_robots: int = 3
    speed_mps: float = 5.0
    flight_alt_m: float = 25.0        # fixed cruise altitude (v0: no altitude actions)
    clearance_m: float = 3.0          # cells with height >= alt - clearance are obstacles for A*
    arrive_radius_m: float = 3.0
    spawn_jitter_m: float = 4.0       # used when scene.robots_spawn has fewer entries than n_robots


@dataclass
class SensorConfig:
    mode: str = "cone"                # "cone" | "disk" (disk = debugging: everything within depth_limit, LoS still applies)
    hfov_deg: float = 90.0
    vfov_deg: float = 80.0
    pitch_deg: float = -50.0          # negative = looking down; with vfov 80 the wedge spans [-90, -10] deg: nadir covered, far field kept
    depth_limit_m: float = 35.0       # RayFronts depth limit: within -> voxels; beyond -> rays (must exceed flight_alt_m)
    visual_range_m: float = 80.0      # rays are emitted for far-visible cells up to this slant range
    los_step_frac: float = 0.5        # ray-march step as fraction of cell_m
    los_eps_m: float = 0.2


@dataclass
class RayFrontsConfig:
    queries: tuple[str, ...] = DEFAULT_QUERIES   # mission queries -> query tokens, not a map scan
    sim_table_path: str | None = None # optional json overriding the hand-authored class x query table
    embedding_dim: int = 24           # D of the per-cell / per-ray semantic feature
    embeddings_path: str | None = None # cached text embeddings (see scripts/build_text_embeddings.py)
    feat_noise_std: float = 0.08      # per-dimension noise on a voxel observation feature
    p_confuse: float = 0.05           # prob. a voxel observation uses a random other class row
    ray_noise_std: float = 0.12       # per-dimension noise on a ray observation feature
    p_ray_per_cell: float = 0.3       # prob. a far-visible cell emits a ray observation per sub-step
    p_fp_ray: float = 0.005           # prob. per robot per sub-step of a spurious person-class ray
    ray_az_bin_deg: float = 20.0
    ray_origin_cell_m: float = 4.0    # rays binned by origin cell of this size x azimuth bin
    ray_conf_cap: float = 5.0
    ray_ttl_s: float = 300.0
    ray_resolve_frac: float = 0.7     # ray removed once this fraction of cells along its bearing (depth..visual) are observed
    ray_range_m: float = 50.0         # fallback ray range when the elevation is unusable (el >= 0)
    ray_resolve_radius_m: float = 15.0  # a ray also resolves once this disc around the point it aims at is observed
    p_observe_base: dict[str, float] = field(default_factory=lambda: {"open": 0.9, "partial": 0.5, "occluded": 0.15})
    far_observe_factor: float = 0.5   # multiplier on p_observe for far-visible (beyond depth limit) humans
    found_hits: int = 2               # observations of a human's cell carrying the human row before it counts as found
    frontier_ig_radius_m: float = 30.0
    frontier_min_cluster_cells: int = 3
    segment_scale: float = 40.0       # Felzenszwalb k, in cells: the segment size the scale prefers
    segment_min_cells: int = 4        # components smaller than this are merged into a neighbour
    segment_refresh_frac: float = 0.05  # re-segment once this fraction of the observed cells is new
    segment_refresh_decisions: int = 25 # ... and at least this often, so the labels track the features
    vox_noise_std: InitVar[float | None] = None   # deprecated alias of feat_noise_std

    def __post_init__(self, vox_noise_std: float | None) -> None:
        if vox_noise_std is not None:
            self.feat_noise_std = float(vox_noise_std)


def _vox_noise_get(self: RayFrontsConfig) -> float:
    return self.feat_noise_std


def _vox_noise_set(self: RayFrontsConfig, v: float) -> None:
    self.feat_noise_std = float(v)


# the noise now lives in feature space; the old name stays a live alias
RayFrontsConfig.vox_noise_std = property(_vox_noise_get, _vox_noise_set)


@dataclass
class TokenConfig:
    k_frontier: int = 32
    k_ray: int = 32
    k_segment: int = 32
    k_visited: int = 32               # visited-target records (own + gossiped), newest first
    include_hold: bool = True
    max_queries: int = 8              # Qmax: padded width of the query-token block
    bev_size: int = 64                # compressed global BEV (critic side of the CTDE split)
    local_size: int = 64              # ego-centric local crop in raster cells (0 disables it)
    robot_bev_size: int = 0           # per-robot BEV over the region (actor side); 0 disables it


@dataclass
class RewardConfig:
    casualty_reward: float = 1.0
    time_cost: float = 0.01           # per decision step (team)
    potential_shaping: bool = False
    shaping_gamma: float = 0.99
    # per-robot terms, credited to the robot that caused them and added to the team reward
    # charged as a *fraction*, averaged over the team: cost * mean_r(redundant cells of robot r /
    # cells robot r observed this decision). A decision in which everyone re-covers known ground
    # costs exactly `redundancy_cost`, whatever the footprint, the cell size or the team size
    redundancy_cost: float = 0.05
    redundancy_refund: bool = True    # ... waived for the decision if it found a casualty in one
    revisit_cost: float = 0.5         # arriving at a chosen target another robot already visited
    revisit_m: float = 15.0           # radius of "the same target" for visits and revisits
    revisit_known_only: bool = True   # only if the visited record had reached this robot in time
    revisit_refund_on_find: bool = True  # ... refunded if that arrival turns up a new casualty


@dataclass
class ShareConfig:
    """What a gossip contact hands over (DESIGN_VARIANTS.md B). Position, current target, contact
    age and the reported coverage fraction are always exchanged; these flags are the payloads."""
    rays: str = "all"                 # "none" | "newest" (rays_newest of them) | "all"
    ray_cap: int = 64                 # hard cap on either mode
    rays_newest: int = 8              # N for the "newest" mode
    coverage: bool = True             # coarse known grid: marks cells known, without features
    coverage_cell_m: float = 8.0
    segments: bool = True
    segment_cap: int = 32
    visited: bool = True              # visited-target records
    features: bool = False            # shared cells become known *with* features (bandwidth)


@dataclass
class CommsConfig:
    """`mode="full"` is one shared team belief (v0). `mode="range"` gives every robot its own
    belief and links it to the peers within `range_m`, relayed through the connected component."""
    mode: str = "full"
    range_m: float = float("inf")
    range_choices: tuple[float, ...] = (100.0, 200.0, 400.0, float("inf"))
    randomize_range: bool = True      # draw range_m from range_choices per episode (training)
    relay_hops: int = 0               # 0 = whole connected component; N = at most N hops
    spawn_exchange: bool = True       # one contact at t=0 regardless of range (positions/targets)
    share: ShareConfig = field(default_factory=ShareConfig)

    @classmethod
    def coerce(cls, v: Any) -> "CommsConfig":
        """Accept a `CommsConfig`, a dict, or the legacy `comms: "full"` string."""
        if isinstance(v, CommsConfig):
            return v
        if isinstance(v, str):
            return cls(mode=v)
        d = dict(v or {})
        sh = d.pop("share", None)
        c = cls(**d)
        if sh is not None:
            c.share = sh if isinstance(sh, ShareConfig) else ShareConfig(**dict(sh))
        c.range_choices = tuple(float(x) for x in c.range_choices)
        return c


@dataclass
class EnvConfig:
    raster: RasterConfig = field(default_factory=RasterConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    rayfronts: RayFrontsConfig = field(default_factory=RayFrontsConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dt_sim: float = 1.0
    decision_dt: float = 5.0
    t_max_s: float = 600.0
    comms: CommsConfig = field(default_factory=CommsConfig)   # "full" or per-robot + gossip
    record_events: bool = True

    def __post_init__(self) -> None:
        self.comms = CommsConfig.coerce(self.comms)

    # ---- helpers ------------------------------------------------------------------------
    @property
    def n_queries(self) -> int:
        return len(self.rayfronts.queries)

    @property
    def q_max(self) -> int:
        return int(self.tokens.max_queries)

    @property
    def k_tokens(self) -> int:
        t = self.tokens
        return ((1 if t.include_hold else 0) + t.k_frontier + t.k_ray + t.k_segment
                + t.k_visited)

    @property
    def substeps_per_decision(self) -> int:
        return max(1, int(round(self.decision_dt / self.dt_sim)))

    def validate(self) -> list[str]:
        e: list[str] = []
        if self.sensor.depth_limit_m > self.sensor.visual_range_m:
            e.append("sensor.depth_limit_m must be <= sensor.visual_range_m")
        if self.sensor.depth_limit_m <= self.robot.flight_alt_m:
            e.append("sensor.depth_limit_m must be > robot.flight_alt_m (slant range to any ground "
                     "cell is >= the altitude, so nothing would ever be voxel-observed)")
        if self.raster.cell_m <= 0:
            e.append("raster.cell_m must be > 0")
        if self.robot.n_robots < 1 or self.robot.n_robots > 10:
            e.append("robot.n_robots must be in [1, 10]")
        if self.decision_dt < self.dt_sim:
            e.append("decision_dt must be >= dt_sim")
        if self.rayfronts.embedding_dim < 2:
            e.append("rayfronts.embedding_dim must be >= 2")
        if not (0.0 <= self.rayfronts.ray_resolve_frac <= 1.0):
            e.append("rayfronts.ray_resolve_frac in [0,1]")
        if self.rayfronts.found_hits < 1:
            e.append("rayfronts.found_hits must be >= 1")
        if self.rayfronts.segment_min_cells < 1:
            e.append("rayfronts.segment_min_cells must be >= 1")
        if self.rayfronts.segment_scale <= 0:
            e.append("rayfronts.segment_scale must be > 0")
        if not self.rayfronts.queries:
            e.append("rayfronts.queries is empty: the mission needs at least one query token")
        if len(self.rayfronts.queries) > self.tokens.max_queries:
            e.append(f"rayfronts.queries has {len(self.rayfronts.queries)} entries but "
                     f"tokens.max_queries is {self.tokens.max_queries}")
        if self.tokens.local_size < 0:
            e.append("tokens.local_size must be >= 0")
        if self.sensor.mode not in ("cone", "disk"):
            e.append("sensor.mode must be 'cone' or 'disk'")
        self.comms = CommsConfig.coerce(self.comms)      # a plain string assignment is legal
        cm = self.comms
        if cm.mode not in ("full", "range"):
            e.append(f"comms.mode must be 'full' or 'range', got {cm.mode!r}")
        if cm.range_m < 0:
            e.append("comms.range_m must be >= 0")
        if any(float(x) < 0 for x in cm.range_choices):
            e.append("comms.range_choices must be >= 0")
        if cm.randomize_range and not cm.range_choices:
            e.append("comms.randomize_range needs a non-empty comms.range_choices")
        if cm.relay_hops < 0:
            e.append("comms.relay_hops must be >= 0 (0 = the whole connected component)")
        sh = cm.share
        if sh.rays not in ("none", "newest", "all"):
            e.append(f"comms.share.rays must be 'none', 'newest' or 'all', got {sh.rays!r}")
        if sh.rays_newest < 0:
            e.append("comms.share.rays_newest must be >= 0")
        if sh.ray_cap < 0 or sh.segment_cap < 0:
            e.append("comms.share caps must be >= 0")
        if sh.coverage_cell_m <= 0:
            e.append("comms.share.coverage_cell_m must be > 0")
        if self.tokens.k_visited < 0:
            e.append("tokens.k_visited must be >= 0")
        if self.tokens.robot_bev_size < 0:
            e.append("tokens.robot_bev_size must be >= 0")
        if self.reward.revisit_m <= 0:
            e.append("reward.revisit_m must be > 0")
        return e

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rayfronts"]["queries"] = list(self.rayfronts.queries)
        d["comms"]["range_choices"] = [float(x) for x in self.comms.range_choices]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvConfig":
        d = dict(d)
        sub = {"raster": RasterConfig, "robot": RobotConfig, "sensor": SensorConfig,
               "rayfronts": RayFrontsConfig, "tokens": TokenConfig, "reward": RewardConfig}
        if "comms" in d:
            d["comms"] = CommsConfig.coerce(d.pop("comms"))
        kw: dict[str, Any] = {}
        for k, klass in sub.items():
            if k in d:
                v = dict(d.pop(k))
                if k == "rayfronts" and "queries" in v:
                    v["queries"] = tuple(v["queries"])
                kw[k] = klass(**v)
        kw.update(d)
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
