"""Scene JSON schema (v0.1): the contract between the 2.5D exporter, the simulator and the visualizer.

Frame: generator world frame, metres, x right / y up, z up (ground = 0). Yaw in degrees CCW from +x.
Rectangles are (x0, y0, x1, y1). Sizes are (sx, sy) along the object's local axes before yaw.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "0.1"

# ---- enums ---------------------------------------------------------------------------------
BUILDING_FATES = ("intact", "damaged", "destroyed")
BUILDING_CATEGORIES = ("house", "midrise", "highrise", "other")
DEBRIS_KINDS = ("pile", "piece")
VEHICLE_STATES = ("intact", "toppled")
VEHICLE_KINDS = ("car", "bus", "truck")
PROP_CATEGORIES = ("bus_stop", "tree", "streetlight", "traffic_light", "bench", "trash_can",
                   "fence", "utility_pole", "other")
PROP_STATES = ("upright", "toppled")
ROAD_KINDS = ("road", "sidewalk", "trail", "driveway")
BLOCK_TYPOLOGIES = ("residential", "commercial", "mixed", "park", "industrial", "other")
HUMAN_ROLES = ("casualty", "bystander")
HUMAN_POSES = ("prone", "standing", "walking", "sitting")
HUMAN_CONTAINERS = ("open", "vehicle", "building", "rubble")
HUMAN_VISIBILITY = ("open", "partial", "occluded")
HUMAN_CONTEXTS = ("sidewalk", "road", "bus_stop", "park", "yard", "building", "vehicle",
                  "debris", "other")
DAMAGE_KINDS = ("uniform", "radial", "path")

# ---- raster classes (owned here so exporter, sim and viz agree) ----------------------------
CLASS_NAMES: tuple[str, ...] = (
    "ground",             # 0  grass / bare ground / unknown
    "road",               # 1
    "sidewalk",           # 2  incl. trail, driveway
    "park",               # 3
    "building_intact",    # 4
    "building_damaged",   # 5
    "building_destroyed", # 6
    "debris",             # 7
    "vehicle_intact",     # 8
    "vehicle_toppled",    # 9
    "bus_stop",           # 10
    "tree",               # 11
    "street_furniture",   # 12 lights, benches, bins, poles, fences
    "human_standing",     # 13 dynamic: only appears in the *belief* raster once detected
    "human_prone",        # 14 dynamic: idem
)
CLASS_ID: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)

# Default extruded heights (m) when the scene does not specify one.
DEFAULT_HEIGHT_M: dict[str, float] = {
    "house": 7.0, "midrise": 24.0, "highrise": 60.0, "other": 10.0,
    "debris_pile": 1.8, "debris_piece": 0.6,
    "car": 1.5, "bus": 3.2, "truck": 3.0,
    "bus_stop": 2.8, "tree": 9.0, "streetlight": 8.0, "traffic_light": 6.0,
    "bench": 0.9, "trash_can": 1.1, "fence": 1.6, "utility_pole": 9.0, "prop_other": 2.0,
}
# Height multiplier applied to a building by fate (destroyed = rubble mound).
FATE_HEIGHT_SCALE: dict[str, float] = {"intact": 1.0, "damaged": 0.8, "destroyed": 0.25}


class SchemaError(ValueError):
    pass


# ---- dataclasses ---------------------------------------------------------------------------
@dataclass
class Meta:
    preset: str = "synthetic"
    seed: int = 0
    locale: str = "suburban"
    disaster_type: str = "earthquake"
    severity: float = 1.0
    region: tuple[float, float, float, float] = (0.0, 0.0, 200.0, 200.0)
    generator_version: str = "synthetic-0.1"
    created_utc: str = ""
    notes: str = ""


@dataclass
class DamageField:
    """Analytic spec plus an optional sampled grid (row-major, ny rows of nx values, origin at
    region (x0, y0), cell_m spacing). The sim prefers `grid` when present."""
    kind: str = "uniform"
    params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] | None = None  # {"cell_m": float, "nx": int, "ny": int, "values": [[...]]}

    def value_at(self, x: float, y: float, region: tuple[float, float, float, float]) -> float:
        """Evaluate the analytic field (0..1). Mirrors scene_gen.make_damage_field semantics."""
        p = self.params
        inside = float(p.get("inside", 1.0))
        outside = float(p.get("outside", 0.0))
        if self.kind == "uniform":
            return inside

        def ease(dist: float, full: float, fall: float) -> float:
            if dist <= full:
                return inside
            t = min(1.0, max(0.0, (dist - full) / max(fall, 1e-6)))
            s = t * t * (3.0 - 2.0 * t)
            return inside + (outside - inside) * s

        if self.kind == "radial":
            cx, cy = p.get("center", [0.0, 0.0])
            return ease(math.hypot(x - cx, y - cy), float(p.get("radius_m", 80.0)),
                        float(p.get("falloff_m", 120.0)))
        if self.kind == "path":
            pts = p.get("points")
            if not pts:
                x0, y0, x1, y1 = region
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                h = math.radians(float(p.get("heading_deg", 0.0)))
                L = math.hypot(x1 - x0, y1 - y0)
                pts = [[cx - L * math.cos(h), cy - L * math.sin(h)],
                       [cx + L * math.cos(h), cy + L * math.sin(h)]]
            d = min(_point_segment_dist(x, y, *pts[i], *pts[i + 1]) for i in range(len(pts) - 1))
            return ease(d, float(p.get("width_m", 60.0)) / 2.0, float(p.get("falloff_m", 40.0)))
        raise SchemaError(f"unknown damage field kind {self.kind!r}")


@dataclass
class Road:
    id: str
    rect: tuple[float, float, float, float]
    kind: str = "road"        # ROAD_KINDS
    dir: str = "h"            # "h" | "v"
    n_lanes: int = 2


@dataclass
class Block:
    id: str
    rect: tuple[float, float, float, float]
    typology: str = "residential"
    built_frac: float = 0.0


@dataclass
class Building:
    id: str
    center: tuple[float, float]
    size: tuple[float, float]
    yaw_deg: float = 0.0
    height_m: float | None = None     # None -> DEFAULT_HEIGHT_M[category] * FATE_HEIGHT_SCALE[fate]
    fate: str = "intact"
    category: str = "house"
    block_id: str | None = None

    def resolved_height(self) -> float:
        if self.height_m is not None:
            return float(self.height_m)
        return DEFAULT_HEIGHT_M.get(self.category, DEFAULT_HEIGHT_M["other"]) * FATE_HEIGHT_SCALE.get(self.fate, 1.0)


@dataclass
class Debris:
    id: str
    center: tuple[float, float]
    radius_m: float
    height_m: float | None = None
    kind: str = "pile"
    building_id: str | None = None

    def resolved_height(self) -> float:
        return float(self.height_m) if self.height_m is not None else DEFAULT_HEIGHT_M[f"debris_{self.kind}"]


@dataclass
class Vehicle:
    id: str
    center: tuple[float, float]
    size: tuple[float, float] = (4.5, 1.9)
    yaw_deg: float = 0.0
    state: str = "intact"
    kind: str = "car"
    height_m: float | None = None

    def resolved_height(self) -> float:
        if self.height_m is not None:
            return float(self.height_m)
        h = DEFAULT_HEIGHT_M[self.kind]
        return h if self.state == "intact" else max(1.0, h * 1.1)  # on its side: roughly width


@dataclass
class Prop:
    id: str
    category: str
    center: tuple[float, float]
    size: tuple[float, float] = (1.0, 1.0)
    yaw_deg: float = 0.0
    height_m: float | None = None
    state: str = "upright"

    def resolved_height(self) -> float:
        if self.height_m is not None:
            return float(self.height_m)
        h = DEFAULT_HEIGHT_M.get(self.category, DEFAULT_HEIGHT_M["prop_other"])
        return h if self.state == "upright" else 0.5


@dataclass
class Human:
    id: str
    pos: tuple[float, float, float]
    role: str = "casualty"
    pose: str = "prone"
    container: str = "open"
    visibility: str = "open"
    context: str = "other"
    container_id: str | None = None


@dataclass
class Scene:
    meta: Meta = field(default_factory=Meta)
    damage_field: DamageField = field(default_factory=DamageField)
    roads: list[Road] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    debris: list[Debris] = field(default_factory=list)
    vehicles: list[Vehicle] = field(default_factory=list)
    props: list[Prop] = field(default_factory=list)
    humans: list[Human] = field(default_factory=list)
    robots_spawn: list[tuple[float, float, float]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    # -- (de)serialisation ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path, indent: int | None = None) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scene":
        ver = d.get("schema_version", SCHEMA_VERSION)
        if ver != SCHEMA_VERSION:
            raise SchemaError(f"schema_version {ver!r} != {SCHEMA_VERSION!r}")
        sc = cls(
            meta=Meta(**_tup(d.get("meta", {}), "region")),
            damage_field=DamageField(**d.get("damage_field", {})),
            roads=[Road(**_tup(r, "rect")) for r in d.get("roads", [])],
            blocks=[Block(**_tup(b, "rect")) for b in d.get("blocks", [])],
            buildings=[Building(**_tup(b, "center", "size")) for b in d.get("buildings", [])],
            debris=[Debris(**_tup(b, "center")) for b in d.get("debris", [])],
            vehicles=[Vehicle(**_tup(v, "center", "size")) for v in d.get("vehicles", [])],
            props=[Prop(**_tup(p, "center", "size")) for p in d.get("props", [])],
            humans=[Human(**_tup(h, "pos")) for h in d.get("humans", [])],
            robots_spawn=[tuple(float(v) for v in s) for s in d.get("robots_spawn", [])],
            schema_version=ver,
        )
        return sc

    @classmethod
    def from_json(cls, path: str | Path) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- convenience --------------------------------------------------------------------------
    @property
    def region(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.meta.region)  # type: ignore[return-value]

    def casualties(self) -> list[Human]:
        return [h for h in self.humans if h.role == "casualty"]

    def damage_at(self, x: float, y: float) -> float:
        return self.damage_field.value_at(x, y, self.region)


def _tup(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    d = dict(d)
    for k in keys:
        if k in d and d[k] is not None:
            d[k] = tuple(d[k])
    return d


def _point_segment_dist(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# ---- validation ----------------------------------------------------------------------------
def validate(scene: Scene) -> list[str]:
    """Return a list of human-readable problems (empty if valid). Raises nothing."""
    errs: list[str] = []
    x0, y0, x1, y1 = scene.region
    if not (x1 > x0 and y1 > y0):
        errs.append(f"region must have positive extent, got {scene.region}")
        return errs
    if not (0.0 <= scene.meta.severity <= 1.0):
        errs.append(f"severity {scene.meta.severity} not in [0,1]")
    if scene.damage_field.kind not in DAMAGE_KINDS:
        errs.append(f"damage_field.kind {scene.damage_field.kind!r} not in {DAMAGE_KINDS}")
    g = scene.damage_field.grid
    if g is not None:
        vals = g.get("values")
        if not (isinstance(vals, list) and len(vals) == g.get("ny") and all(len(r) == g.get("nx") for r in vals)):
            errs.append("damage_field.grid values shape != (ny, nx)")
        elif any((v < 0.0 or v > 1.0) for r in vals for v in r):
            errs.append("damage_field.grid values outside [0,1]")

    def inside(x: float, y: float, margin: float = 0.0) -> bool:
        return (x0 - margin) <= x <= (x1 + margin) and (y0 - margin) <= y <= (y1 + margin)

    seen: set[str] = set()

    def uid(prefix: str, i: str) -> None:
        key = f"{prefix}:{i}"
        if not i:
            errs.append(f"{prefix} with empty id")
        if key in seen:
            errs.append(f"duplicate id {key}")
        seen.add(key)

    for r in scene.roads:
        uid("road", r.id)
        if r.kind not in ROAD_KINDS:
            errs.append(f"road {r.id}: kind {r.kind!r}")
        if r.dir not in ("h", "v"):
            errs.append(f"road {r.id}: dir {r.dir!r}")
        if not (r.rect[2] > r.rect[0] and r.rect[3] > r.rect[1]):
            errs.append(f"road {r.id}: degenerate rect {r.rect}")
    for b in scene.blocks:
        uid("block", b.id)
        if b.typology not in BLOCK_TYPOLOGIES:
            errs.append(f"block {b.id}: typology {b.typology!r}")
    bids = set()
    for b in scene.buildings:
        uid("building", b.id)
        bids.add(b.id)
        if b.fate not in BUILDING_FATES:
            errs.append(f"building {b.id}: fate {b.fate!r}")
        if b.category not in BUILDING_CATEGORIES:
            errs.append(f"building {b.id}: category {b.category!r}")
        if b.size[0] <= 0 or b.size[1] <= 0:
            errs.append(f"building {b.id}: non-positive size {b.size}")
        if not inside(*b.center, margin=max(b.size)):
            errs.append(f"building {b.id}: center {b.center} outside region")
        if b.height_m is not None and b.height_m <= 0:
            errs.append(f"building {b.id}: non-positive height")
    for d in scene.debris:
        uid("debris", d.id)
        if d.kind not in DEBRIS_KINDS:
            errs.append(f"debris {d.id}: kind {d.kind!r}")
        if d.radius_m <= 0:
            errs.append(f"debris {d.id}: radius {d.radius_m}")
        if d.building_id is not None and d.building_id not in bids:
            errs.append(f"debris {d.id}: unknown building_id {d.building_id}")
    vids = set()
    for v in scene.vehicles:
        uid("vehicle", v.id)
        vids.add(v.id)
        if v.state not in VEHICLE_STATES:
            errs.append(f"vehicle {v.id}: state {v.state!r}")
        if v.kind not in VEHICLE_KINDS:
            errs.append(f"vehicle {v.id}: kind {v.kind!r}")
        if not inside(*v.center, margin=10.0):
            errs.append(f"vehicle {v.id}: outside region")
    pids = set()
    for p in scene.props:
        uid("prop", p.id)
        pids.add(p.id)
        if p.category not in PROP_CATEGORIES:
            errs.append(f"prop {p.id}: category {p.category!r}")
        if p.state not in PROP_STATES:
            errs.append(f"prop {p.id}: state {p.state!r}")
    for h in scene.humans:
        uid("human", h.id)
        if h.role not in HUMAN_ROLES:
            errs.append(f"human {h.id}: role {h.role!r}")
        if h.pose not in HUMAN_POSES:
            errs.append(f"human {h.id}: pose {h.pose!r}")
        if h.container not in HUMAN_CONTAINERS:
            errs.append(f"human {h.id}: container {h.container!r}")
        if h.visibility not in HUMAN_VISIBILITY:
            errs.append(f"human {h.id}: visibility {h.visibility!r}")
        if h.context not in HUMAN_CONTEXTS:
            errs.append(f"human {h.id}: context {h.context!r}")
        if not inside(h.pos[0], h.pos[1]):
            errs.append(f"human {h.id}: outside region")
        if h.pos[2] < 0:
            errs.append(f"human {h.id}: negative z")
        if h.container == "open" and h.visibility == "occluded":
            errs.append(f"human {h.id}: open container but occluded visibility")
        if h.container in ("rubble", "building") and h.visibility == "open":
            errs.append(f"human {h.id}: container {h.container} but visibility open")
        if h.container_id is not None and h.container_id not in (bids | vids | pids | {d.id for d in scene.debris}):
            errs.append(f"human {h.id}: unknown container_id {h.container_id}")
    for i, s in enumerate(scene.robots_spawn):
        if len(s) != 3 or not inside(s[0], s[1]) or s[2] < 0:
            errs.append(f"robots_spawn[{i}] invalid: {s}")
    return errs


def validate_or_raise(scene: Scene) -> Scene:
    errs = validate(scene)
    if errs:
        raise SchemaError("; ".join(errs[:20]) + (f" (+{len(errs) - 20} more)" if len(errs) > 20 else ""))
    return scene


# ---- synthetic scene (for tests and for the sim/viz before the real exporter exists) -------
def make_synthetic_scene(seed: int = 0, region_m: tuple[float, float] = (240.0, 240.0),
                         block_m: float = 60.0, road_w: float = 8.0, sidewalk_w: float = 2.0,
                         n_casualties: int = 12, n_bystanders: int = 6,
                         severity: float = 1.0, epicenter: tuple[float, float] | None = None,
                         radius_m: float = 60.0, falloff_m: float = 90.0) -> Scene:
    """A gridded suburb with a radial earthquake. Deterministic in `seed`.

    Casualty placement follows the intended generator rules in simplified form:
    destroyed buildings > damaged > intact (inside / under rubble), toppled cars, bus stops in the
    zone, prone pedestrians on sidewalks scaled by damage. Bystanders are standing people mostly
    outside the zone. This is a stand-in, not the real exporter.
    """
    import random

    rng = random.Random(seed)
    W, H = region_m
    x0, y0, x1, y1 = -W / 2, -H / 2, W / 2, H / 2
    if epicenter is None:
        epicenter = (rng.uniform(x0 + 0.25 * W, x1 - 0.25 * W), rng.uniform(y0 + 0.25 * H, y1 - 0.25 * H))
    df = DamageField(kind="radial", params={"center": list(epicenter), "radius_m": radius_m,
                                            "falloff_m": falloff_m, "inside": 1.0 * severity,
                                            "outside": 0.05 * severity})
    sc = Scene(meta=Meta(preset="synthetic", seed=seed, severity=severity, region=(x0, y0, x1, y1)),
               damage_field=df)
    dmg = lambda x, y: df.value_at(x, y, sc.region)

    # roads: grid lines every block_m (rects clipped to the region; clipped-away strips are dropped)
    nbx, nby = int(round(W / block_m)), int(round(H / block_m))

    def add_strip(rid, rect, kind, d):
        cx0, cy0 = max(rect[0], x0), max(rect[1], y0)
        cx1, cy1 = min(rect[2], x1), min(rect[3], y1)
        if cx1 - cx0 > 0.05 and cy1 - cy0 > 0.05:
            sc.roads.append(Road(id=rid, rect=(cx0, cy0, cx1, cy1), kind=kind, dir=d))

    for i in range(nbx + 1):
        xc = x0 + i * block_m
        add_strip(f"rv{i}", (xc - road_w / 2, y0, xc + road_w / 2, y1), "road", "v")
        add_strip(f"sv{i}a", (xc - road_w / 2 - sidewalk_w, y0, xc - road_w / 2, y1), "sidewalk", "v")
        add_strip(f"sv{i}b", (xc + road_w / 2, y0, xc + road_w / 2 + sidewalk_w, y1), "sidewalk", "v")
    for j in range(nby + 1):
        yc = y0 + j * block_m
        add_strip(f"rh{j}", (x0, yc - road_w / 2, x1, yc + road_w / 2), "road", "h")
        add_strip(f"sh{j}a", (x0, yc - road_w / 2 - sidewalk_w, x1, yc - road_w / 2), "sidewalk", "h")
        add_strip(f"sh{j}b", (x0, yc + road_w / 2, x1, yc + road_w / 2 + sidewalk_w), "sidewalk", "h")

    # blocks + buildings
    inset = road_w / 2 + sidewalk_w + 4.0
    bcount = 0
    for i in range(nbx):
        for j in range(nby):
            bx0, by0 = x0 + i * block_m + inset, y0 + j * block_m + inset
            bx1, by1 = min(x1, x0 + (i + 1) * block_m) - inset, min(y1, y0 + (j + 1) * block_m) - inset
            if bx1 - bx0 < 24.0 or by1 - by0 < 24.0:
                continue  # block too small for the 2x2 house pattern
            is_park = rng.random() < 0.12
            sc.blocks.append(Block(id=f"b{i}_{j}", rect=(bx0, by0, bx1, by1),
                                   typology="park" if is_park else "residential"))
            if is_park:
                continue
            # 2x2 houses per block
            for u in range(2):
                for v in range(2):
                    cx = bx0 + (u + 0.5) * (bx1 - bx0) / 2
                    cy = by0 + (v + 0.5) * (by1 - by0) / 2
                    d = dmg(cx, cy)
                    r = rng.random()
                    fate = "destroyed" if r < 0.45 * d else ("damaged" if r < 0.45 * d + 0.35 * d else "intact")
                    cat = "midrise" if rng.random() < 0.15 else "house"
                    size = (rng.uniform(10, 14), rng.uniform(9, 13)) if cat == "house" else (rng.uniform(16, 22), rng.uniform(14, 20))
                    half = (bx1 - bx0) / 4.0, (by1 - by0) / 4.0
                    size = (min(size[0], 2 * half[0] - 2.0), min(size[1], 2 * half[1] - 2.0))
                    b = Building(id=f"bld{bcount}", center=(cx, cy), size=size, yaw_deg=rng.choice([0.0, 90.0]),
                                 fate=fate, category=cat, block_id=f"b{i}_{j}")
                    sc.buildings.append(b)
                    bcount += 1
                    if fate == "destroyed":
                        for k in range(rng.randint(2, 4)):
                            ang = rng.uniform(0, 2 * math.pi)
                            rad = max(size) / 2 + rng.uniform(0.5, 3.0)
                            sc.debris.append(Debris(id=f"deb{b.id}_{k}", center=(cx + rad * math.cos(ang), cy + rad * math.sin(ang)),
                                                    radius_m=rng.uniform(1.5, 3.0), building_id=b.id))

    # vehicles on roads; toppled in proportion to damage
    vcount = 0
    for r in [r for r in sc.roads if r.kind == "road"]:
        rx0, ry0, rx1, ry1 = r.rect
        length = (rx1 - rx0) if r.dir == "h" else (ry1 - ry0)
        n = int(length / 35.0) if length >= 12.0 else 0
        for _ in range(n):
            if r.dir == "h":
                cx, cy, yaw = rng.uniform(rx0 + 5, rx1 - 5), rng.uniform(ry0 + 1.5, ry1 - 1.5), 0.0
            else:
                cx, cy, yaw = rng.uniform(rx0 + 1.5, rx1 - 1.5), rng.uniform(ry0 + 5, ry1 - 5), 90.0
            toppled = rng.random() < 0.6 * dmg(cx, cy)
            sc.vehicles.append(Vehicle(id=f"veh{vcount}", center=(cx, cy), yaw_deg=yaw, state="toppled" if toppled else "intact"))
            vcount += 1

    # bus stops at some intersections, trees in parks
    pcount = 0
    for i in range(nbx + 1):
        for j in range(nby + 1):
            if rng.random() < 0.35:
                cx = x0 + i * block_m + road_w / 2 + sidewalk_w + 1.5
                cy = y0 + j * block_m + road_w / 2 + sidewalk_w + 1.5
                if x0 < cx < x1 and y0 < cy < y1:
                    sc.props.append(Prop(id=f"bs{pcount}", category="bus_stop", center=(cx, cy), size=(3.0, 1.5),
                                         state="toppled" if rng.random() < 0.5 * dmg(cx, cy) else "upright"))
                    pcount += 1
    for b in [b for b in sc.blocks if b.typology == "park"]:
        bx0, by0, bx1, by1 = b.rect
        for _ in range(rng.randint(4, 9)):
            sc.props.append(Prop(id=f"tree{pcount}", category="tree", center=(rng.uniform(bx0, bx1), rng.uniform(by0, by1)),
                                 size=(4.0, 4.0)))
            pcount += 1

    # humans
    hcount = 0

    def add_human(x, y, z, role, pose, container, vis, ctx, cid=None):
        nonlocal hcount
        x = min(max(x, x0 + 0.5), x1 - 0.5)
        y = min(max(y, y0 + 0.5), y1 - 0.5)
        sc.humans.append(Human(id=f"h{hcount}", pos=(x, y, z), role=role, pose=pose, container=container,
                               visibility=vis, context=ctx, container_id=cid))
        hcount += 1

    # casualties: weighted sampling over contexts
    destroyed = [b for b in sc.buildings if b.fate == "destroyed"]
    damaged = [b for b in sc.buildings if b.fate == "damaged"]
    toppled = [v for v in sc.vehicles if v.state == "toppled"]
    stops = [p for p in sc.props if p.category == "bus_stop" and dmg(*p.center) > 0.3]
    sidewalks = [r for r in sc.roads if r.kind == "sidewalk"] or [r for r in sc.roads if r.kind == "road"]
    if not sidewalks:
        raise SchemaError("synthetic scene has no roads or sidewalks to place humans on")
    for _ in range(n_casualties):
        u = rng.random()
        if u < 0.35 and destroyed:
            b = rng.choice(destroyed)
            jx, jy = rng.uniform(-b.size[0] / 2, b.size[0] / 2), rng.uniform(-b.size[1] / 2, b.size[1] / 2)
            add_human(b.center[0] + jx, b.center[1] + jy, 0.0, "casualty", "prone", "rubble", "occluded", "debris", b.id)
        elif u < 0.55 and damaged:
            b = rng.choice(damaged)
            jx, jy = rng.uniform(-b.size[0] / 2, b.size[0] / 2), rng.uniform(-b.size[1] / 2, b.size[1] / 2)
            add_human(b.center[0] + jx, b.center[1] + jy, 0.0, "casualty", "prone", "building", "partial", "building", b.id)
        elif u < 0.72 and toppled:
            v = rng.choice(toppled)
            add_human(v.center[0], v.center[1], 0.0, "casualty", "prone", "vehicle", "partial", "vehicle", v.id)
        elif u < 0.82 and stops:
            p = rng.choice(stops)
            add_human(p.center[0] + rng.uniform(-2, 2), p.center[1] + rng.uniform(-2, 2), 0.0, "casualty", "prone", "open", "open", "bus_stop", p.id)
        else:
            # prone pedestrian on a sidewalk, rejection-sampled by damage
            for _try in range(50):
                r = rng.choice(sidewalks)
                rx0, ry0, rx1, ry1 = r.rect
                x, y = rng.uniform(rx0, rx1), rng.uniform(ry0, ry1)
                if rng.random() < dmg(x, y):
                    add_human(x, y, 0.0, "casualty", "prone", "open", "open", "sidewalk")
                    break
    # bystanders: standing, mostly where damage is low
    for _ in range(n_bystanders):
        for _try in range(50):
            r = rng.choice(sidewalks)
            rx0, ry0, rx1, ry1 = r.rect
            x, y = rng.uniform(rx0, rx1), rng.uniform(ry0, ry1)
            if rng.random() < (1.0 - 0.8 * dmg(x, y)):
                add_human(x, y, 0.0, "bystander", rng.choice(["standing", "walking"]), "open", "open", "sidewalk")
                break

    # spawn: a corner road intersection, at 20 m
    sc.robots_spawn = [(x0 + road_w, y0 + road_w, 20.0), (x0 + road_w + 6, y0 + road_w, 20.0),
                       (x0 + road_w, y0 + road_w + 6, 20.0), (x0 + road_w + 6, y0 + road_w + 6, 20.0)]
    validate_or_raise(sc)
    return sc


__all__ = [
    "SCHEMA_VERSION", "CLASS_NAMES", "CLASS_ID", "N_CLASSES", "DEFAULT_HEIGHT_M", "FATE_HEIGHT_SCALE",
    "SchemaError", "Meta", "DamageField", "Road", "Block", "Building", "Debris", "Vehicle", "Prop",
    "Human", "Scene", "validate", "validate_or_raise", "make_synthetic_scene",
]
