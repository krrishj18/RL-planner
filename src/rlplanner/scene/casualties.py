"""Casualty and bystander placement for an exported scene.

The generator's own humans are dropped by the exporter; this module places them
instead, so the who/where/how-hidden question is answered by one policy that the
sim's detection model can be reasoned about against.

Casualties concentrate where the disaster hit: inside destroyed buildings (under
rubble, occluded), inside damaged ones (partially visible), trapped at toppled
vehicles, at bus stops inside the damage zone, and prone on sidewalks with a
damage-weighted rejection. A small tail lands on any sidewalk regardless of
damage so a searcher can never assume "no damage => nobody here".
Bystanders stand or walk on sidewalks and in parks, biased *away* from damage.
Counts are either explicit or `"auto"`, which scales them with the region area
(`HumanConfig.counts`).

Invariants every returned human satisfies (see tests/test_casualties.py):
  * inside the scene region;
  * `container_id` resolves, and the position lies inside that container's
    footprint (vehicles/buildings) or within `bus_stop_jitter_m` of the stop;
  * schema container/visibility consistency;
  * not inside a standing (non-destroyed) building unless `container="building"`;
  * `min_separation_m` from every other human, best-effort within
    `separation_tries` draws (a scene with no room degrades, never hangs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .schema import Human, Scene

# Sampling categories. The draw is restricted to the kinds that actually have
# candidates and their weights are renormalised over those, so a class the scene
# never produces (no bus stops downtown) hands its mass to the rest in
# proportion instead of to whichever kind happens to weigh most.
CASUALTY_KINDS = ("destroyed", "damaged", "vehicle", "bus_stop", "pedestrian")

# `n_casualties="auto"`: v1's 15 people per 400 x 400 m, scaled by region area
# and clipped so a 500 m scene is still worth searching and a 1500 m one is not
# a crowd.
AUTO = "auto"
AUTO_REF_KM2 = 0.16
AUTO_PER_REF = 15.0
AUTO_MIN, AUTO_MAX = 10, 80
AUTO_BYSTANDER_FRAC = 0.5

# Keep sampled points this far inside the region so `_mk`'s rounding to mm can
# never push one over the border (schema.validate rejects that).
_EDGE_M = 1e-3


@dataclass
class HumanConfig:
    # int, or "auto" for area-scaled counts (see `counts`).
    n_casualties: int | str = 15
    n_bystanders: int | str = 8
    # Relative weights over CASUALTY_KINDS (normalised; need not sum to 1).
    weights: dict[str, float] = field(default_factory=lambda: {
        "destroyed": 0.34, "damaged": 0.20, "vehicle": 0.16,
        "bus_stop": 0.10, "pedestrian": 0.20,
    })
    open_tail_frac: float = 0.10      # casualties on any sidewalk, damage ignored
    bus_stop_damage_min: float = 0.30  # a stop only counts inside the zone
    footprint_frac: float = 0.80      # inset when sampling inside a footprint
    vehicle_jitter_m: float = 1.2     # capped by the vehicle's own half-extents
    bus_stop_jitter_m: float = 2.0    # radius of the disc around the stop
    bystander_park_frac: float = 0.35
    bystander_damage_bias: float = 0.80  # accept with 1 - bias * damage
    max_tries: int = 80
    min_separation_m: float = 0.5
    separation_tries: int = 12
    z_m: float = 0.0

    def weight(self, kind: str) -> float:
        return float(self.weights.get(kind, 0.0))

    def counts(self, region: tuple[float, float, float, float]) -> tuple[int, int]:
        """`(n_casualties, n_bystanders)` for *region*.

        "auto" scales with the region area at v1's density (15 per 0.16 km2),
        clipped to [10, 80]; auto bystanders are half the *resolved* casualty
        count, so mixing an explicit count with an auto one still makes sense.
        """
        x0, y0, x1, y1 = (float(v) for v in region)
        area_km2 = max(0.0, (x1 - x0) * (y1 - y0)) / 1e6
        n_cas = (auto_casualties(area_km2) if _is_auto(self.n_casualties)
                 else _count(self.n_casualties, "n_casualties"))
        n_bys = (_iround(n_cas * AUTO_BYSTANDER_FRAC) if _is_auto(self.n_bystanders)
                 else _count(self.n_bystanders, "n_bystanders"))
        return n_cas, n_bys


def auto_casualties(area_km2: float) -> int:
    return int(min(AUTO_MAX, max(AUTO_MIN,
                                 _iround(AUTO_PER_REF * area_km2 / AUTO_REF_KM2))))


def _iround(v: float) -> int:
    return int(math.floor(float(v) + 0.5))


def _is_auto(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == AUTO


def _count(v, what: str) -> int:
    try:
        ok = not isinstance(v, (str, bool)) and int(v) == v
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{what}={v!r}: expected a non-negative integer or {AUTO!r}")
    if int(v) < 0:
        raise ValueError(f"{what}={v!r}: must be non-negative")
    return int(v)


def place_humans(scene: Scene, rng: np.random.Generator,
                 cfg: HumanConfig | None = None) -> list[Human]:
    """Casualties + bystanders for *scene* (whose `humans` are ignored).

    Deterministic in *rng*. Every returned human is inside the region and obeys
    the schema's container/visibility consistency rules.
    """
    cfg = cfg or HumanConfig()
    x0, y0, x1, y1 = scene.region
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"scene region must have positive extent, got {scene.region}")
    n_casualties, n_bystanders = cfg.counts(scene.region)
    p = _Pools(scene, cfg)
    out: list[Human] = []
    placed: list[tuple[float, float]] = []

    def add(h: Human) -> None:
        out.append(h)
        placed.append((h.pos[0], h.pos[1]))

    for i in range(n_casualties):
        add(_spaced(lambda i=i: _casualty(f"cas{i}", p, rng, cfg), placed, cfg))
    for i in range(n_bystanders):
        add(_spaced(lambda i=i: _bystander(f"bys{i}", p, rng, cfg), placed, cfg))
    return out


# ---- candidate pools -------------------------------------------------------
class _Pools:
    def __init__(self, scene: Scene, cfg: HumanConfig):
        self.region = scene.region
        self.damage_at = scene.damage_at
        self.destroyed = [b for b in scene.buildings if b.fate == "destroyed"]
        self.damaged = [b for b in scene.buildings if b.fate == "damaged"]
        self.toppled = [v for v in scene.vehicles if v.state == "toppled"]
        self.stops = [q for q in scene.props if q.category == "bus_stop"
                      and scene.damage_at(*q.center) > cfg.bus_stop_damage_min]
        walk = [r.rect for r in scene.roads if r.kind in ("sidewalk", "trail", "driveway")]
        if not walk:
            walk = [r.rect for r in scene.roads]
        self.walk = _RectPool(walk)
        self.park = _RectPool([b.rect for b in scene.blocks if b.typology == "park"])
        self.region_pool = _RectPool([self.region])

        # Standing structures: an open-air human must not end up under a roof.
        # A destroyed building is a rubble field, so lying on it is fine.
        solid = [b for b in scene.buildings if b.fate != "destroyed"]
        if solid:
            self._bc = np.array([b.center for b in solid], dtype=float)
            self._bh = np.array([[b.size[0] / 2.0, b.size[1] / 2.0] for b in solid],
                                dtype=float)
            a = np.radians(np.array([b.yaw_deg for b in solid], dtype=float))
            self._bcos, self._bsin = np.cos(a), np.sin(a)
        else:
            self._bc = None

    def has(self, kind: str) -> bool:
        return bool({"destroyed": self.destroyed, "damaged": self.damaged,
                     "vehicle": self.toppled, "bus_stop": self.stops,
                     "pedestrian": self.walk.rects}[kind])

    def in_building(self, x: float, y: float) -> bool:
        """Point-in-OBB against every standing building at once."""
        if self._bc is None:
            return False
        dx, dy = x - self._bc[:, 0], y - self._bc[:, 1]
        u = dx * self._bcos + dy * self._bsin
        v = -dx * self._bsin + dy * self._bcos
        return bool(np.any((np.abs(u) <= self._bh[:, 0]) & (np.abs(v) <= self._bh[:, 1])))

    def in_region(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.region
        return (x0 + _EDGE_M <= x <= x1 - _EDGE_M
                and y0 + _EDGE_M <= y <= y1 - _EDGE_M)


class _RectPool:
    """Area-weighted uniform sampling over a list of axis-aligned rects."""

    def __init__(self, rects: list[tuple[float, float, float, float]]):
        self.rects = [r for r in rects if r[2] > r[0] and r[3] > r[1]]
        a = np.array([(r[2] - r[0]) * (r[3] - r[1]) for r in self.rects], dtype=float)
        self.cdf = np.cumsum(a) / a.sum() if a.size and a.sum() > 0 else None

    def __bool__(self) -> bool:
        return bool(self.rects)

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        i = int(np.searchsorted(self.cdf, rng.random())) if self.cdf is not None else 0
        x0, y0, x1, y1 = self.rects[min(i, len(self.rects) - 1)]
        return float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))


# ---- casualties ------------------------------------------------------------
def _casualty(hid: str, p: _Pools, rng: np.random.Generator, cfg: HumanConfig) -> Human:
    if rng.random() < cfg.open_tail_frac and p.walk:
        xy = _clear_sample(p.walk, p, rng, cfg.max_tries)
        if xy is not None:
            return _mk(hid, p, cfg, *xy, "casualty", "prone", "open", "open", "sidewalk")

    for kind in _draw_order(rng, cfg, [k for k in CASUALTY_KINDS if p.has(k)]):
        h = _CASUALTY_FN[kind](hid, p, rng, cfg)
        if h is not None:
            return h
    pool = p.walk or p.region_pool
    ctx = "sidewalk" if p.walk else "other"
    xy = _clear_sample(pool, p, rng, cfg.max_tries) or pool.sample(rng)
    return _mk(hid, p, cfg, *xy, "casualty", "prone", "open", "open", ctx)


def _draw_order(rng: np.random.Generator, cfg: HumanConfig,
                avail: list[str]) -> list[str]:
    """A weight-proportional permutation of the kinds that have candidates.

    Only kinds with candidates take part, so a class the scene never produces
    (no bus stops downtown) hands its weight to the rest in proportion. The
    *whole* order is drawn, not just the first pick: a kind that has candidates
    but fails to place (the damage rejection on a quiet sidewalk) then falls
    through proportionally too, instead of always to the heaviest kind.

    Weighted sampling without replacement, Efraimidis-Spirakis: sort by
    `log(U) / w` descending. Zero-weight kinds sort last, in `CASUALTY_KINDS`
    order, and stay reachable as a last resort.
    """
    if not avail:
        return []
    w = np.array([max(0.0, cfg.weight(k)) for k in avail], dtype=float)
    if w.sum() <= 0:
        return sorted(avail, key=CASUALTY_KINDS.index)
    u = np.maximum(rng.random(len(avail)), 1e-300)
    key = np.where(w > 0, np.log(u) / np.where(w > 0, w, 1.0), -np.inf)
    return [avail[i] for i in np.lexsort((np.arange(len(avail)), -key))]


def _in_destroyed(hid, p, rng, cfg):
    b = p.destroyed[int(rng.integers(len(p.destroyed)))]
    xy = _sample_footprint(b, p, rng, cfg, clear=True)
    if xy is None:
        return None
    return _mk(hid, p, cfg, *xy, "casualty", "prone", "rubble", "occluded", "debris", b.id)


def _in_damaged(hid, p, rng, cfg):
    b = p.damaged[int(rng.integers(len(p.damaged)))]
    xy = _sample_footprint(b, p, rng, cfg)
    if xy is None:
        return None
    return _mk(hid, p, cfg, *xy, "casualty", "prone", "building", "partial", "building", b.id)


def _at_vehicle(hid, p, rng, cfg):
    """Trapped at a rolled car: jitter in the *vehicle's* frame, capped by its
    own half-extents, so the casualty is always on the car it belongs to."""
    v = p.toppled[int(rng.integers(len(p.toppled)))]
    hx = min(cfg.vehicle_jitter_m, abs(v.size[0]) * cfg.footprint_frac / 2.0)
    hy = min(cfg.vehicle_jitter_m, abs(v.size[1]) * cfg.footprint_frac / 2.0)
    c, s = math.cos(math.radians(v.yaw_deg)), math.sin(math.radians(v.yaw_deg))
    for _ in range(cfg.max_tries):
        u, w = float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy))
        x = v.center[0] + u * c - w * s
        y = v.center[1] + u * s + w * c
        # Downtown parks cars on strips a building sometimes overhangs; a car
        # under a roof would put a "partial" casualty inside a solid volume.
        if p.in_region(x, y) and not p.in_building(x, y):
            return _mk(hid, p, cfg, x, y, "casualty", "prone", "vehicle", "partial",
                       "vehicle", v.id)
    return None


def _at_bus_stop(hid, p, rng, cfg):
    """In a disc of radius `bus_stop_jitter_m` around the stop, so the distance to
    the stop is bounded by that radius."""
    q = p.stops[int(rng.integers(len(p.stops)))]
    for _ in range(cfg.max_tries):
        r = cfg.bus_stop_jitter_m * math.sqrt(float(rng.random()))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        x, y = q.center[0] + r * math.cos(th), q.center[1] + r * math.sin(th)
        if p.in_region(x, y) and not p.in_building(x, y):
            return _mk(hid, p, cfg, x, y, "casualty", "prone", "open", "open",
                       "bus_stop", q.id)
    return None


def _prone_pedestrian(hid, p, rng, cfg):
    """Prone on a walking surface, rejection-sampled by the damage field."""
    for _ in range(cfg.max_tries):
        x, y = p.walk.sample(rng)
        if p.in_building(x, y) or not p.in_region(x, y):
            continue
        if rng.random() < p.damage_at(x, y):
            return _mk(hid, p, cfg, x, y, "casualty", "prone", "open", "open", "sidewalk")
    return None


_CASUALTY_FN = {"destroyed": _in_destroyed, "damaged": _in_damaged,
                "vehicle": _at_vehicle, "bus_stop": _at_bus_stop,
                "pedestrian": _prone_pedestrian}


# ---- bystanders ------------------------------------------------------------
def _bystander(hid: str, p: _Pools, rng: np.random.Generator, cfg: HumanConfig) -> Human:
    # The pool is redrawn on every rejection, not once: a park that lies wholly
    # inside the zone would otherwise trap its bystanders at full damage, since
    # rejection can only refuse a point, never move it to the other pool.
    def pick():
        if p.park and rng.random() < cfg.bystander_park_frac:
            return p.park, "park"
        if p.walk:
            return p.walk, "sidewalk"
        return (p.park, "park") if p.park else (p.region_pool, "other")

    best = None
    for _ in range(cfg.max_tries):
        pool, ctx = pick()
        x, y = pool.sample(rng)
        if p.in_building(x, y) or not p.in_region(x, y):
            continue
        best = (x, y, ctx)
        if rng.random() < 1.0 - cfg.bystander_damage_bias * p.damage_at(x, y):
            break
    if best is None:
        pool, ctx = pick()
        xy = _clear_sample(pool, p, rng, cfg.max_tries) or pool.sample(rng)
        best = (xy[0], xy[1], ctx)
    pose = "standing" if rng.random() < 0.5 else "walking"
    return _mk(hid, p, cfg, best[0], best[1], "bystander", pose, "open", "open", best[2])


# ---- helpers ---------------------------------------------------------------
def _clear_sample(pool: _RectPool, p: _Pools, rng: np.random.Generator,
                  tries: int) -> tuple[float, float] | None:
    """First point from *pool* that is in-region and clear of standing buildings."""
    for _ in range(max(1, tries)):
        x, y = pool.sample(rng)
        if p.in_region(x, y) and not p.in_building(x, y):
            return x, y
    return None


def _sample_footprint(b, p: _Pools, rng: np.random.Generator, cfg: HumanConfig,
                      clear: bool = False) -> tuple[float, float] | None:
    """A point inside *b*'s footprint that is also inside the region — a building
    hanging over the border must not yield a casualty the region clamp would
    drag off its own footprint. `clear` additionally keeps the point out of any
    *standing* building (ruins can overlap a neighbour that is still up)."""
    for _ in range(cfg.max_tries):
        x, y = _in_footprint(b, rng, cfg.footprint_frac)
        if p.in_region(x, y) and not (clear and p.in_building(x, y)):
            return x, y
    return None


def _in_footprint(b, rng: np.random.Generator, frac: float) -> tuple[float, float]:
    hx, hy = b.size[0] * frac / 2.0, b.size[1] * frac / 2.0
    u, v = float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy))
    c, s = math.cos(math.radians(b.yaw_deg)), math.sin(math.radians(b.yaw_deg))
    return b.center[0] + u * c - v * s, b.center[1] + u * s + v * c


def _spaced(make, placed: list[tuple[float, float]], cfg: HumanConfig) -> Human:
    """Redraw until the candidate clears `min_separation_m`; keep the roomiest
    draw when the scene has no room left (200 casualties, one ruin)."""
    if not placed or cfg.min_separation_m <= 0.0:
        return make()
    pts = np.asarray(placed, dtype=float)
    best, best_d = None, -1.0
    for _ in range(max(1, int(cfg.separation_tries))):
        h = make()
        d = float(np.min(np.hypot(pts[:, 0] - h.pos[0], pts[:, 1] - h.pos[1])))
        if d >= cfg.min_separation_m:
            return h
        if d > best_d:
            best, best_d = h, d
    return best


def _mk(hid, p: _Pools, cfg: HumanConfig, x, y, role, pose, container, visibility,
        context, container_id=None) -> Human:
    x0, y0, x1, y1 = p.region
    # Round first, clamp second: clamping first lets round() step back over the
    # border by up to half a millimetre, which schema.validate rejects.
    return Human(id=hid, pos=(min(max(round(float(x), 3), x0), x1),
                              min(max(round(float(y), 3), y0), y1), float(cfg.z_m)),
                 role=role, pose=pose, container=container, visibility=visibility,
                 context=context, container_id=container_id)


__all__ = ["HumanConfig", "CASUALTY_KINDS", "AUTO", "auto_casualties",
           "place_humans"]
