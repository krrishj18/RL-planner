"""2.5D scene exporter: vendored procedural city generator -> Scene JSON (schema v0.1).

No pxr, no Isaac, no Nucleus: `scene.gen.build_city` is pure python and every asset
footprint comes from the asset set's per-category `fallback_sizes` (`measure_usds:
false`). What the generator writes as USD references becomes 2.5D primitives here:

    usds.buildings.{intact,damaged,destroyed}   -> Building.fate
    category "debris_pile" / "debris"           -> Debris(pile|piece)
    category "car"                              -> Vehicle (toppled = rolled/pitched)
    category "bus_stop"/"tree"/"streetlight"/…  -> Prop (upright|toppled)
    layout road_corridors / block sidewalk ring -> Road(road|sidewalk|driveway|trail)
    disaster.field                              -> DamageField (spec + sampled grid)
    category "human"                            -> DROPPED, replaced by casualties.py

`pipeline="v2"` runs the generator's detailed-city pipeline instead of plain
`build_city`: anisotropic block subdivision (`layout/city_layout.py`) plus
zoning, typology rezoning, infill and park superblocks (`detail/districts.py`,
`detail/parks.py`), exactly the sub-steps `gen/generate_scene.py` performs
before it touches a USD stage. The USD-only passes (street furniture, road
surface, markings, provenance) are skipped — they add no 2.5D geometry.
"""
from __future__ import annotations

import contextlib
import io
import math
import random
import zlib
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import schema
from .casualties import HumanConfig, place_humans
from .gen import compile_disaster as _cd
from .gen import scene_generator as _sg
from .gen.detail import districts as _districts
from .gen.layout import city_layout as _city_layout
from .schema import (Block, Building, DamageField, Debris, Meta, Prop, Road, Scene,
                     Vehicle)

GENERATOR_VERSION = "scene_gen@disaster-dataset/krrishj (vendored 2026-08-20)"
PRESET_DIR = Path(_cd.DEFAULT_PRESET_DIR)

# v2 locale -> the pristine high-level preset that carries its layout/districts/
# parks blocks. The disaster axis composes on top of it (disaster-type, severity,
# region_m, seed), which is how `compile_spec` is meant to be driven.
V2_LOCALE_PRESET: dict[str, str] = {"downtown": "downtown"}

# districts typology -> schema.BLOCK_TYPOLOGIES. The generator's names are
# morphological (what stands on the block); the schema's are land use:
#   tower    -> commercial   the CBD ring: office/retail massing, one per block
#   midrise  -> mixed        apartment + light-industrial stock, packed perimeter
#   rowhouse -> residential  party-wall terraces
#   park     -> park         a reserved superblock, no building on it
# A block the rezoning pass skipped (too small to build on) has no typology.
BLOCK_TYPOLOGY: dict[str, str] = {"tower": "commercial", "midrise": "mixed",
                                  "rowhouse": "residential", "park": "park"}
# ... and -> schema.BUILDING_CATEGORIES, by the pool the block was built from.
BUILDING_CATEGORY: dict[str, str] = {"tower": "highrise", "midrise": "midrise",
                                     "rowhouse": "house"}

# Beyond this much roll/pitch a prop/vehicle is on the ground, not merely leaning
# (the generator topples at 80-100 deg or 180, and leans at most ~45).
TOPPLE_DEG = 60.0

# generator placement category -> (schema prop category, explicit height or None).
# None takes the asset set's fallback z, else the schema default for the category.
PROP_SPEC: dict[str, tuple[str, float | None]] = {
    "bus_stop": ("bus_stop", None),
    "tree": ("tree", None),
    "streetlight": ("streetlight", None),
    "traffic_light": ("traffic_light", None),
    "bench": ("bench", None),
    "trash_can": ("trash_can", None),
    "fire_hydrant": ("other", None),
    "planter": ("other", None),
    "play_structure": ("other", None),
    "plant": ("tree", 1.0),       # shrubs read as foliage, not as street furniture
    "rock": ("other", 0.8),
    # v2 only (parks.py): railings around a park, and whatever stands in the
    # middle of an attraction (fountain, bandstand).
    "fence": ("fence", None),
    "park_feature": ("other", None),
}
# Ground tiles (concrete / sidewalk / trail) become Roads; humans are dropped and
# re-placed by casualties.py.
JITTER_EXEMPT = ("concrete", "sidewalk", "brick", "trail")


def available_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


# ---- generator config ------------------------------------------------------
def load_generator_config(preset_or_config: str | Path | dict, seed: int, *,
                          region_m: tuple[float, float] | None = None,
                          severity: float | None = None,
                          disaster: str | None = None,
                          verbose: bool = False) -> tuple[dict, dict]:
    """Resolve a preset name / config path / spec dict into a validated low-level
    generator config with `measure_usds` off. Returns `(config, provenance)`.

    `disaster` swaps the high-level spec's `disaster-type`, which is how a
    pristine locale preset (v2's `downtown.yaml`) is composed with a disaster."""
    if isinstance(preset_or_config, dict):
        spec, path, name = deepcopy(preset_or_config), None, str(
            preset_or_config.get("name", "config"))
    else:
        path = _cd.resolve_config_path(str(preset_or_config))
        spec = yaml.safe_load(Path(path).read_text())
        name = Path(path).stem
    if not isinstance(spec, dict):
        raise ValueError(f"scene config {preset_or_config!r} did not parse to a mapping")

    with _quiet(verbose):
        if disaster is not None:
            if not _cd.is_high_level(spec):
                raise ValueError(f"disaster override needs a high-level preset, "
                                 f"{preset_or_config!r} is a low-level config")
            spec.pop("disaster_type", None)
            spec["disaster-type"] = str(disaster)
        if _cd.is_high_level(spec):
            spec["seed"] = int(seed)
            if region_m is not None:
                spec["region_m"] = [float(region_m[0]), float(region_m[1])]
            if severity is not None:
                spec["severity"] = float(severity)
            prov = {"disaster_type": str(spec.get("disaster-type",
                                                  spec.get("disaster_type", "none"))),
                    "severity": float(spec.get("severity", 1.0)),
                    "locale": str(spec.get("locale", "downtown"))}
            base = yaml.safe_load(Path(_cd.DEFAULT_BASE).read_text())
            cfg = _cd.compile_spec(spec, base)
        else:
            cfg = spec
            cfg["seed"] = int(seed)
            if region_m is not None:
                cfg.setdefault("layout", {})["region_m"] = [float(region_m[0]),
                                                            float(region_m[1])]
            if severity is not None:
                raise ValueError("severity override needs a high-level preset, "
                                 f"{preset_or_config!r} is a low-level config")
            dis = cfg.get("disaster", {})
            hit = max(float(dis.get("damaged_fraction", 0.0)),
                      float(dis.get("destroyed_fraction", 0.0)))
            prov = {"disaster_type": "custom", "severity": 1.0 if hit > 0 else 0.0,
                    "locale": str(cfg.get("locale", "downtown"))}
        cfg = _sg.resolve_asset_set(cfg, path)
        cfg = _sg.validate_config(cfg, path or "<dict>")
    cfg["measure_usds"] = False
    prov["preset"] = name
    prov["asset_set"] = str(cfg.get("asset_set", ""))
    return cfg, prov


def export_scene(preset_or_config: str | Path | dict, seed: int, *,
                 region_m: tuple[float, float] | None = None, cell_m: float = 2.0,
                 human_cfg: HumanConfig | None = None, severity: float | None = None,
                 spawn_corner: str = "ll", size_jitter: float = 0.0,
                 pipeline: str = "v1", disaster: str | None = None,
                 verbose: bool = False) -> Scene:
    """Build `preset_or_config` at `seed` and return a validated `Scene`.

    `region_m` overrides the config's city extent (before the disaster field is
    derived from it), `cell_m` is the damage-grid spacing, `severity` overrides a
    high-level preset's severity, `spawn_corner` is one of ll/lr/ul/ur, and
    `size_jitter` spreads the per-category fallback footprints (see `_Resolver`).

    `pipeline="v2"` builds the detailed city (see `build_v2`); for it,
    `preset_or_config` names the locale ("downtown") and `disaster` the disaster
    type composed onto it.
    """
    if cell_m <= 0:
        raise ValueError(f"cell_m must be positive, got {cell_m}")
    if size_jitter < 0.0:
        raise ValueError(f"size_jitter must be >= 0, got {size_jitter}")
    if region_m is not None:
        region_m = _check_region(region_m)
    if pipeline not in ("v1", "v2"):
        raise ValueError(f"pipeline must be 'v1' or 'v2', got {pipeline!r}")
    seed = _check_seed(seed)
    if pipeline == "v2":
        preset_or_config = _v2_preset(preset_or_config)
        if disaster is None:
            raise ValueError("pipeline='v2' needs a disaster "
                             "(earthquake / tornado / explosion / ...)")
    cfg, prov = load_generator_config(preset_or_config, seed, region_m=region_m,
                                      severity=severity, disaster=disaster,
                                      verbose=verbose)
    resolver = _Resolver(asset_scale=float(cfg.get("asset_scale", 1.0)),
                         fallback_sizes=cfg.get("fallback_sizes", {}),
                         measure=False, jitter=size_jitter)
    with _quiet(verbose):
        placements, layout = (build_v2(cfg, resolver) if pipeline == "v2"
                              else _sg.build_city(cfg, resolver))
    if pipeline == "v2":
        prov["preset"] = f"v2:{prov['locale']}:{prov['disaster_type']}"
    scene = _Assembler(cfg, prov, resolver, placements, layout, cell_m,
                       spawn_corner, pipeline).run()
    scene.humans = place_humans(scene, np.random.default_rng([int(seed), 0x1CE]),
                                human_cfg)
    return schema.validate_or_raise(scene)


def _v2_preset(locale_or_path) -> str | Path | dict:
    """Map a v2 locale name onto its pristine preset; pass anything else through
    so a caller can hand in its own spec."""
    if isinstance(locale_or_path, str) and locale_or_path in V2_LOCALE_PRESET:
        return V2_LOCALE_PRESET[locale_or_path]
    if isinstance(locale_or_path, str) and not Path(locale_or_path).exists() \
            and locale_or_path not in available_presets():
        raise ValueError(f"unknown v2 locale {locale_or_path!r}; known: "
                         f"{', '.join(sorted(V2_LOCALE_PRESET))}")
    return locale_or_path


def build_v2(cfg: dict, resolver) -> tuple[list[dict], dict]:
    """The pure-python half of `gen/generate_scene.generate_scene_on_stage`.

    Same order, same RNG, same calls, stopping where the pipeline starts writing
    USD: anisotropic subdivision around `build_city`, then district assignment
    and `remap_buildings` (rezone + infill + parks). `city_detail` and
    `road_markings` only decorate a stage and are not vendored.
    """
    aniso = ((cfg.get("layout") or {}).get("anisotropic") or {}).get("enabled")
    zoned = (cfg.get("districts") or {}).get("enabled")
    if not (aniso and zoned):
        # Without those two blocks every v2 pass no-ops and the result is a v1
        # city wearing a v2 label — worse than an error.
        raise ValueError("the v2 pipeline needs a config with "
                         "layout.anisotropic.enabled and districts.enabled "
                         f"(presets/downtown.yaml sets both); got anisotropic="
                         f"{bool(aniso)} districts={bool(zoned)}")
    rng = random.Random(int(cfg.get("seed", 0)) + 7717)
    with _city_layout.patched(cfg):
        placements, layout = _sg.build_city(cfg, resolver)
    district_at, rings = _districts.assign(cfg, layout)
    if rings:
        _districts.remap_buildings(cfg, layout, placements, resolver, rng,
                                   district_at)
    return placements, layout


def sample_region(seed: int, lo: float = 500.0, hi: float = 1500.0,
                  quantum: float = 10.0) -> tuple[float, float]:
    """Region extent for `seed`: W and H drawn independently and uniformly from
    [lo, hi] and rounded to a multiple of `quantum`, so scenes are non-square and
    the size is reproducible from the seed alone.

    Rounding rule: round to the nearest multiple of `quantum`, then clamp to the
    innermost multiples the range contains (`ceil(lo)`..`floor(hi)`), so bounds
    that are not multiples themselves — `--region-range 513 999` — still yield
    round sizes (520..990). Only a range holding no multiple at all (513..517)
    falls back to clamping at the raw bounds.
    """
    if not (0 < lo <= hi):
        raise ValueError(f"region range must satisfy 0 < lo <= hi, got {lo}, {hi}")
    if quantum <= 0:
        raise ValueError(f"region quantum must be positive, got {quantum}")
    q_lo = math.ceil(float(lo) / quantum - 1e-9) * quantum
    q_hi = math.floor(float(hi) / quantum + 1e-9) * quantum
    a, b = (q_lo, q_hi) if q_lo <= q_hi else (float(lo), float(hi))
    rng = np.random.default_rng([_check_seed(seed), 0x5217])
    w, h = rng.uniform(float(lo), float(hi), size=2)
    return tuple(float(min(max(round(v / quantum) * quantum, a), b))
                 for v in (w, h))


def sample_severity(seed: int, disaster: str, lo: float = 0.5,
                    hi: float = 1.0) -> float:
    """Severity for (seed, disaster), uniform in [lo, hi]. Keyed on the disaster
    too, so the three disasters of one seed are not all equally bad."""
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"severity range must be within [0, 1], got {lo}, {hi}")
    rng = np.random.default_rng([_check_seed(seed),
                                 int(zlib.crc32(str(disaster).encode()))])
    return round(float(rng.uniform(float(lo), float(hi))), 4)


class _Resolver(_sg.SizeResolver):
    """Fallback footprints are one constant per category, so without measuring, every
    asset in a pool packs to the same box. `jitter` (0..1) spreads them by a stable
    hash of the usd path, so a pool of 25 house models lays out as 25 sizes."""

    def __init__(self, *a, jitter: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.jitter = float(jitter)
        self._jitter_cache: dict = {}

    def get(self, usd_path: str, category: str, scale: float = None,
            axis_up: str = "Z") -> dict:
        fp = super().get(usd_path, category, scale=scale, axis_up=axis_up)
        if self.jitter <= 0.0 or category in JITTER_EXEMPT:
            return fp
        key = (usd_path, float(scale) if scale is not None else self.scale,
               str(axis_up or "Z").upper())
        out = self._jitter_cache.get(key)
        if out is None:
            h = zlib.crc32(usd_path.encode())
            fx = 1.0 + self.jitter * ((h & 0xFFFF) / 0xFFFF * 2.0 - 1.0)
            fy = 1.0 + self.jitter * (((h >> 16) & 0xFFFF) / 0xFFFF * 2.0 - 1.0)
            out = dict(fp, sx=fp["sx"] * fx, sy=fp["sy"] * fy,
                       sz=fp["sz"] * (fx + fy) / 2.0)
            self._jitter_cache[key] = out
        return out


# ---- assembly --------------------------------------------------------------
class _Assembler:
    def __init__(self, cfg: dict, prov: dict, resolver, placements: list[dict],
                 layout: dict, cell_m: float, spawn_corner: str,
                 pipeline: str = "v1"):
        self.cfg, self.prov, self.resolver, self.layout = cfg, prov, resolver, layout
        self.cell_m, self.spawn_corner = float(cell_m), spawn_corner
        self.pipeline = pipeline
        # (rect, typology) per block, as districts.rezone_blocks zoned it.
        self.typ_of: list[tuple[tuple, str]] = list(
            (layout.get("_typology_of") or {}).items())
        self.region = tuple(float(v) for v in layout["region"])
        self.asset_scale = float(cfg.get("asset_scale", 1.0))
        self.fallback = cfg.get("fallback_sizes", {}) or {}

        # Re-derive the per-asset lookups build_city made from the same config, so a
        # placement's usd tells us which pool it came from and what its base scale
        # and yaw offset were.
        root = str(cfg.get("asset_root", "") or "")
        self._scale: dict[str, float] = {}
        self._yaw_off: dict[str, float] = {}

        def pool(lst) -> set[str]:
            paths, sc, _au, yo, _tg = _sg._normalize_usd_list(lst, self.asset_scale, root)
            self._scale.update(sc)
            self._yaw_off.update(yo)
            return set(paths)

        bld = cfg["usds"].get("buildings") or cfg["usds"].get("houses") or {}
        pool(bld.get("intact") or [])
        self.damaged_set = pool(bld.get("damaged") or [])
        self.destroyed_set = pool(bld.get("destroyed") or [])
        for key in ("trees", "plants", "rocks", "streetlights", "benches", "cars",
                    "trash_cans", "traffic_lights", "fire_hydrants", "humans",
                    "planters", "bus_stops", "play_structures"):
            pool(cfg["usds"].get(key) or [])
        for key in ("piles", "pieces"):
            pool((cfg["usds"].get("debris") or {}).get(key) or [])
        for key in ("concrete", "sidewalk", "brick", "trail"):
            pool((cfg["usds"].get("tiles") or {}).get(key) or [])

        self.by_cat: dict[str, list[dict]] = {}
        for p in placements:
            self.by_cat.setdefault(p["category"], []).append(p)

    # -- per-placement helpers ------------------------------------------------
    def base_scale(self, usd: str) -> float:
        return float(self._scale.get(usd, self.asset_scale))

    def size_scale(self, p: dict) -> float:
        """Random per-instance size jitter build_city baked into `scale`. Fallback
        footprints ignore scale, so it has to be re-applied here."""
        b = self.base_scale(p["usd"])
        return float(p["scale"]) / b if abs(b) > 1e-12 else 1.0

    def footprint(self, p: dict, category: str | None = None) -> dict:
        return self.resolver.get(p["usd"], category or p["category"],
                                 scale=self.base_scale(p["usd"]),
                                 axis_up=p.get("axis_up", "Z"))

    def yaw(self, p: dict) -> float:
        """Placement yaw with the per-asset art correction removed, so it matches
        the axis the footprint was measured/declared on."""
        return float(p["yaw_deg"]) - float(self._yaw_off.get(p["usd"], 0.0))

    def height(self, category: str, fp: dict, s: float = 1.0) -> float | None:
        """The asset set's fallback z for *category*, or None when it declares none —
        the schema's per-category default is the better guess than the resolver's
        blanket 3 m."""
        fb = self.fallback.get(category)
        has_z = isinstance(fb, (list, tuple)) and len(fb) > 2
        return _r(fp["sz"] * s) if has_z else None

    # -- passes ---------------------------------------------------------------
    def run(self) -> Scene:
        sc = Scene(meta=self._meta(), damage_field=self._damage_field())
        sc.buildings = self._buildings()
        sc.blocks = self._blocks(sc.buildings)
        for b in sc.buildings:
            b.block_id = _block_id_at(b.center, sc.blocks)
        sc.debris = self._debris(sc.buildings)
        sc.vehicles = self._vehicles()
        sc.props = self._props()
        sc.roads = self._roads()
        sc.robots_spawn = self._spawn()
        return sc

    def _meta(self) -> Meta:
        x0, y0, x1, y1 = self.region
        return Meta(preset=self.prov["preset"], seed=int(self.cfg.get("seed", 0)),
                    locale=self.prov["locale"], disaster_type=self.prov["disaster_type"],
                    severity=float(self.prov["severity"]),
                    region=(_r(x0), _r(y0), _r(x1), _r(y1)),
                    generator_version=self.pipeline if self.pipeline == "v2"
                    else GENERATOR_VERSION,
                    notes=f"asset_set={self.prov['asset_set']} cell_m={self.cell_m:g} "
                          f"humans=casualties.py gen={GENERATOR_VERSION}")

    def _damage_field(self) -> DamageField:
        spec = dict((self.cfg.get("disaster") or {}).get("field") or {})
        kind = str(spec.pop("kind", "uniform")).lower()
        if kind == "path" and not spec.get("points") and "heading_deg" not in spec:
            spec["heading_deg"] = 45.0  # make_damage_field's implicit default
        df = DamageField(kind=kind, params=spec)
        f = _sg.make_damage_field({"kind": kind, **spec}, self.region)
        x0, y0, x1, y1 = self.region
        # ceil, not round: the grid must cover the whole region or the sim reads
        # no damage in the last strip. Centres are clamped into the region so an
        # over-large cell_m cannot sample the field outside the scene.
        nx = _cells(x1 - x0, self.cell_m)
        ny = _cells(y1 - y0, self.cell_m)
        xs = [min(x0 + (i + 0.5) * self.cell_m, x1) for i in range(nx)]
        ys = [min(y0 + (j + 0.5) * self.cell_m, y1) for j in range(ny)]
        vals = [[round(min(1.0, max(0.0, f(x, y))), 4) for x in xs] for y in ys]
        df.grid = {"cell_m": float(self.cell_m), "nx": nx, "ny": ny, "values": vals}
        return df

    def _buildings(self) -> list[Building]:
        cat = _building_category(self.cfg)
        out: list[Building] = []
        for i, p in enumerate(self.by_cat.get("house", [])):
            fate = self._fate(p)
            fp = self.footprint(p, "house")
            s = self.size_scale(p)
            h = self.height("house", fp, s * schema.FATE_HEIGHT_SCALE[fate])
            bcat = cat
            if self.pipeline == "v2":
                # Zoning decided what stock a block was built from, so the block
                # names the category. Height goes back to the schema default for
                # it: without Nucleus every building measures as the asset set's
                # one `house` fallback (30 x 20 x 24 m), and a blanket 24 m for
                # brownstones, mid-rise and towers alike is a worse guess than
                # house 7 / midrise 24 / highrise 60.
                bcat = BUILDING_CATEGORY.get(
                    self._typology_at(p["x_m"], p["y_m"]) or "", cat)
                h = None
            out.append(Building(
                id=f"bld{i}", center=(_r(p["x_m"]), _r(p["y_m"])),
                size=(_r(max(fp["sx"] * s, 0.5)), _r(max(fp["sy"] * s, 0.5))),
                yaw_deg=_r(self.yaw(p)), height_m=h, fate=fate, category=bcat))
        return out

    def _typology_at(self, x: float, y: float) -> str | None:
        for rect, name in self.typ_of:
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                return name
        return None

    def _fate(self, p: dict) -> str:
        usd = p["usd"]
        if usd in self.destroyed_set:
            return "destroyed"
        if usd in self.damaged_set:
            return "damaged"
        # No damaged pool in the set: build_city tilts+sinks an intact model instead.
        return "damaged" if _tilted(p) else "intact"

    def _blocks(self, buildings: list[Building]) -> list[Block]:
        """v2 reads each block's typology off the districts pass (`BLOCK_TYPOLOGY`);
        a block it left unzoned is one too small to build on, so "other".

        v1 has to infer it: a park is a block build_city skipped before it paved
        and packed, so it holds neither a building nor a hard surface. "No
        building" alone also catches blocks that merely failed to pack
        anything."""
        paved = bool((self.cfg.get("packing") or {}).get("pave_blocks", True))
        typ_of = dict(self.typ_of)
        # v2 knows each block's typology outright; only v1 has to infer it.
        conc = [] if self.pipeline == "v2" else self.by_cat.get("concrete", [])
        hx = np.array([p["x_m"] for p in conc], dtype=float)
        hy = np.array([p["y_m"] for p in conc], dtype=float)
        out: list[Block] = []
        for i, blk in enumerate(self.layout["blocks"]):
            rect = tuple(_r(v) for v in blk)
            area = max((rect[2] - rect[0]) * (rect[3] - rect[1]), 1e-6)
            built = sum(b.size[0] * b.size[1] for b in buildings if _in_rect(b.center, rect))
            if self.pipeline == "v2":
                typ = BLOCK_TYPOLOGY.get(typ_of.get(tuple(blk), ""), "other")
            else:
                empty = built <= 0 and not (hx.size and bool(np.any(
                    (hx >= rect[0]) & (hx <= rect[2]) & (hy >= rect[1]) & (hy <= rect[3]))))
                typ = ("park" if empty else ("commercial" if paved else "residential"))
            out.append(Block(id=f"blk{i}", rect=rect, typology=typ,
                             built_frac=round(min(1.0, built / area), 4)))
        return out

    def _debris(self, buildings: list[Building]) -> list[Debris]:
        out: list[Debris] = []
        ruins = [b for b in buildings if b.fate == "destroyed"]
        centers = np.array([b.center for b in ruins], dtype=float) if ruins else None
        reach = (np.array([max(b.size) for b in ruins], dtype=float) / 2.0
                 + float(((self.cfg.get("disaster") or {}).get("debris") or {})
                         .get("pieces_scatter_m", 6.0)) + 2.0) if ruins else None
        n = 0
        for gen_cat, kind in (("debris_pile", "pile"), ("debris", "piece")):
            for p in self.by_cat.get(gen_cat, []):
                fp = self.footprint(p, gen_cat)
                s = self.size_scale(p)
                bid = None
                if centers is not None:
                    d = np.hypot(centers[:, 0] - p["x_m"], centers[:, 1] - p["y_m"])
                    k = int(np.argmin(d))
                    if d[k] <= reach[k]:
                        bid = ruins[k].id
                out.append(Debris(id=f"deb{n}", center=(_r(p["x_m"]), _r(p["y_m"])),
                                  radius_m=_r(max(0.2, max(fp["sx"], fp["sy"]) * s / 2.0)),
                                  height_m=self.height(gen_cat, fp, s), kind=kind,
                                  building_id=bid))
                n += 1
        return out

    def _vehicles(self) -> list[Vehicle]:
        out: list[Vehicle] = []
        for i, p in enumerate(self.by_cat.get("car", [])):
            fp = self.footprint(p, "car")
            s = self.size_scale(p)
            out.append(Vehicle(id=f"veh{i}", center=(_r(p["x_m"]), _r(p["y_m"])),
                               size=(_r(fp["sx"] * s), _r(fp["sy"] * s)),
                               yaw_deg=_r(self.yaw(p)),
                               state="toppled" if _toppled(p) else "intact", kind="car"))
        return out

    def _props(self) -> list[Prop]:
        out: list[Prop] = []
        n = 0
        for gen_cat, (cat, h_over) in PROP_SPEC.items():
            for p in self.by_cat.get(gen_cat, []):
                fp = self.footprint(p, gen_cat)
                s = self.size_scale(p)
                state = "toppled" if _toppled(p) else "upright"
                # A toppled prop takes the schema's on-the-ground height instead.
                h = None if state == "toppled" else (
                    _r(h_over * s) if h_over is not None else self.height(gen_cat, fp, s))
                out.append(Prop(id=f"prop{n}", category=cat,
                                center=(_r(p["x_m"]), _r(p["y_m"])),
                                size=(_r(max(fp["sx"] * s, 0.1)), _r(max(fp["sy"] * s, 0.1))),
                                yaw_deg=_r(self.yaw(p)), height_m=h, state=state))
                n += 1
        return out

    def _corridor_rect(self, c: dict) -> tuple[float, float, float, float]:
        """The asphalt a corridor actually covers.

        A v2 corridor reserves a 2.4 m kerb strip either side of the carriageway
        whatever its parking policy, so the rect stays symmetric about the centre
        line; a strip the policy does not park on is handed to the abutting block
        as a kerb extension and paved over, and the road is genuinely narrower
        there. v1 corridors carry no `carriage`, so they come back unchanged.
        """
        car, pw = c.get("carriage"), float(c.get("park_w", 0.0))
        if not car or pw <= 0.0:
            return (c["x0"], c["y0"], c["x1"], c["y1"])
        pol = str(c.get("parking", "both"))
        lo = float(car[0]) - (pw if pol in ("both", "lo") else 0.0)
        hi = float(car[1]) + (pw if pol in ("both", "hi") else 0.0)
        if c["dir"] == "ns":
            return (lo, c["y0"], hi, c["y1"])
        return (c["x0"], lo, c["x1"], hi)

    def _roads(self) -> list[Road]:
        out: list[Road] = []
        for i, c in enumerate(self.layout["road_corridors"]):
            rect = _clip_rect(self._corridor_rect(c), self.region)
            if rect is None:
                continue
            out.append(Road(id=f"road{i}", rect=rect, kind="road",
                            dir="v" if c["dir"] == "ns" else "h",
                            n_lanes=int(c["n_lanes"])))
        out += self._sidewalks()
        n = len(out)
        # Driveway / trail tiles, as laid by build_city's _tile_rect / trail walk.
        for gen_cat, kind in (("concrete", "driveway"), ("trail", "trail")):
            if gen_cat == "concrete" and bool((self.cfg.get("packing") or {})
                                              .get("pave_blocks", True)):
                continue      # paved block interiors are covered by _sidewalks()
            for p in self.by_cat.get(gen_cat, []):
                rect = _clip_rect(self._tile_rect(p, gen_cat), self.region)
                if rect is None:
                    continue
                out.append(Road(id=f"{kind}{n}", rect=rect, kind=kind,
                                dir="h" if (rect[2] - rect[0]) >= (rect[3] - rect[1]) else "v"))
                n += 1
        return out

    def _sidewalks(self) -> list[Road]:
        """The per-block sidewalk ring build_city tiles between the verge and the
        block interior, plus (downtown) the paved interior itself."""
        tiles = (self.cfg["usds"].get("tiles") or {})
        sw = tiles.get("sidewalk") or tiles.get("brick")
        if not sw:
            return []
        usd = _sg._normalize_usd_list(sw, self.asset_scale,
                                      str(self.cfg.get("asset_root", "") or ""))[0][0]
        fp = self.resolver.get(usd, "sidewalk", scale=self.base_scale(usd))
        w = max(fp["sx"], fp["sy"])
        if w <= 1e-6:
            return []
        verge = float((self.cfg.get("frontage") or {}).get("verge_m", 0.0))
        paved = bool((self.cfg.get("packing") or {}).get("pave_blocks", True))
        typ_of = dict(self.typ_of)
        out: list[Road] = []
        for i, (x0, y0, x1, y1) in enumerate(self.layout["blocks"]):
            vx0, vy0, vx1, vy1 = x0 + verge, y0 + verge, x1 - verge, y1 - verge
            if not (vx1 - vx0 > 2 * w and vy1 - vy0 > 2 * w):
                continue
            ring = [((vx0, vy0, vx1, vy0 + w), "h"), ((vx0, vy1 - w, vx1, vy1), "h"),
                    ((vx0, vy0 + w, vx0 + w, vy1 - w), "v"),
                    ((vx1 - w, vy0 + w, vx1, vy1 - w), "v")]
            for k, (rect, d) in enumerate(ring):
                r = _clip_rect(rect, self.region)
                if r is not None:
                    out.append(Road(id=f"walk{i}_{k}", rect=r, kind="sidewalk", dir=d))
            # parks.py strips build_city's paving out of a park superblock; only
            # the sidewalk ring around it survives.
            if paved and typ_of.get((x0, y0, x1, y1)) != "park":
                r = _clip_rect((vx0 + w, vy0 + w, vx1 - w, vy1 - w), self.region)
                if r is not None:
                    out.append(Road(id=f"pave{i}", rect=r, kind="sidewalk",
                                    dir="h" if (r[2] - r[0]) >= (r[3] - r[1]) else "v"))
        return out

    def _tile_rect(self, p: dict, gen_cat: str) -> tuple[float, float, float, float]:
        fp = self.footprint(p, gen_cat)
        sx, sy = fp["sx"] * self.size_scale(p), fp["sy"] * self.size_scale(p)
        stx, sty = p.get("stretch", (1.0, 1.0))
        hx, hy = sx * float(stx) / 2.0, sy * float(sty) / 2.0
        if abs(p.get("yaw_deg", 0.0)) > 1e-6:          # trail tiles follow the path
            c, s = abs(math.cos(math.radians(p["yaw_deg"]))), abs(math.sin(math.radians(p["yaw_deg"])))
            hx, hy = hx * c + hy * s, hx * s + hy * c
        return (p["x_m"] - hx, p["y_m"] - hy, p["x_m"] + hx, p["y_m"] + hy)

    def _spawn(self, n: int = 8, spacing: float = 6.0, alt: float = 20.0,
               inset: float = 3.0, file_n: int = 4) -> list[tuple[float, float, float]]:
        """`n` launch points on the corridor nearest `spawn_corner`.

        The first `file_n` are what four robots always got — the step is still
        sized for four, not for `n`, so growing the team never moves them. The
        rest continue along the same corridor at `spacing`; when the corridor
        runs out of length they wrap into a second file across it, half a step
        out of phase, so a short corner corridor still yields `n` distinct
        points instead of `n` copies of its far end.
        """
        x0, y0, x1, y1 = self.region
        try:
            sx, sy = {"ll": (x0, y0), "lr": (x1, y0), "ul": (x0, y1), "ur": (x1, y1)}[
                self.spawn_corner]
        except KeyError:
            raise ValueError(f"spawn_corner {self.spawn_corner!r} not in ll/lr/ul/ur")
        rects = [_clip_rect(self._corridor_rect(c), self.region)
                 for c in self.layout["road_corridors"]]
        rects = [r for r in rects if r is not None]
        if not rects:
            return [(_r(_lerp(sx, x1 if sx == x0 else x0, 0.02 + 0.01 * i)),
                     _r(_lerp(sy, y1 if sy == y0 else y0, 0.02)), alt) for i in range(n)]
        rect = min(rects, key=lambda r: _rect_dist(r, sx, sy))
        rx0, ry0, rx1, ry1 = rect
        horiz = (rx1 - rx0) >= (ry1 - ry0)
        span = (rx1 - rx0) if horiz else (ry1 - ry0)
        cross = (ry1 - ry0) if horiz else (rx1 - rx0)
        ins = min(inset, span * 0.2)
        step = min(spacing, max(span - 2 * ins, 0.0) / max(file_n - 1, 1))
        per = (max(file_n, int((span - 2 * ins) / step) + 1) if step > 1e-9
               else file_n)
        lat = max(0.0, cross / 2.0 - 1.0)      # a metre off the kerb, still asphalt
        lanes = (0.0, lat, -lat, lat / 2.0)
        near_lo = (abs(sx - rx0) <= abs(sx - rx1)) if horiz else \
                  (abs(sy - ry0) <= abs(sy - ry1))
        out: list[tuple[float, float, float]] = []
        seen: set[tuple[float, float]] = set()
        for i in range(n):
            f, k = divmod(i, per)
            off = lanes[f % len(lanes)]
            base = ins + k * step + (step / 2.0 if f % 2 else 0.0)
            for nudge in range(24):
                d = min(base + 0.05 * nudge, max(span - ins, ins))
                if horiz:
                    x, y = (rx0 + d if near_lo else rx1 - d), (ry0 + ry1) / 2.0 + off
                else:
                    x, y = (rx0 + rx1) / 2.0 + off, (ry0 + d if near_lo else ry1 - d)
                x, y = _r(min(max(x, rx0), rx1)), _r(min(max(y, ry0), ry1))
                if (x, y) not in seen:
                    break
            seen.add((x, y))
            out.append((x, y, float(alt)))
        return out


# ---- summaries -------------------------------------------------------------
def scene_summary(scene: Scene) -> dict[str, Any]:
    fates = Counter(b.fate for b in scene.buildings)
    cont = Counter(h.container for h in scene.casualties())
    return {
        "preset": scene.meta.preset, "seed": scene.meta.seed,
        "region": f"{scene.region[2] - scene.region[0]:.0f}x{scene.region[3] - scene.region[1]:.0f}",
        "sev": round(float(scene.meta.severity), 2),
        "blocks": len(scene.blocks),
        "intact": fates["intact"], "damaged": fates["damaged"],
        "destroyed": fates["destroyed"], "debris": len(scene.debris),
        "veh_top": sum(v.state == "toppled" for v in scene.vehicles),
        "veh_int": sum(v.state == "intact" for v in scene.vehicles),
        "props": len(scene.props), "roads": len(scene.roads),
        "cas_open": cont["open"], "cas_veh": cont["vehicle"],
        "cas_bld": cont["building"], "cas_rub": cont["rubble"],
        "bystanders": sum(h.role == "bystander" for h in scene.humans),
    }


SUMMARY_COLUMNS = ("preset", "seed", "region", "sev", "blocks",
                   "intact", "damaged", "destroyed", "debris",
                   "veh_int", "veh_top", "props", "roads", "cas_open", "cas_veh",
                   "cas_bld", "cas_rub", "bystanders", "secs")


def format_summary_table(rows: list[dict[str, Any]]) -> str:
    cols = [c for c in SUMMARY_COLUMNS if any(c in r for r in rows)]
    w = {c: max(len(c), *(len(_cell(r.get(c))) for r in rows)) for c in cols}
    lines = ["  ".join(c.rjust(w[c]) for c in cols),
             "  ".join("-" * w[c] for c in cols)]
    lines += ["  ".join(_cell(r.get(c)).rjust(w[c]) for c in cols) for r in rows]
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    return f"{v:.2f}" if isinstance(v, float) else str(v)


# ---- small helpers ---------------------------------------------------------
def _r(v: float) -> float:
    return round(float(v), 3)


def _cells(extent: float, cell_m: float) -> int:
    return max(1, int(math.ceil(extent / cell_m - 1e-9)))


def _check_region(region_m) -> tuple[float, float]:
    """A degenerate extent builds a city with no roads and no blocks, and dies
    much later sampling an empty pool; name it here instead."""
    try:
        w, h = (float(v) for v in region_m)
    except (TypeError, ValueError):
        raise ValueError(f"region_m must be (W, H) in metres, got {region_m!r}") from None
    if not (w > 0 and h > 0) or not (math.isfinite(w) and math.isfinite(h)):
        raise ValueError(f"region_m must have positive extents, got {region_m!r}")
    return (w, h)


def _check_seed(seed: Any) -> int:
    """`np.random.default_rng([seed, ...])` dies deep in numpy on a negative
    seed; say what is wrong here instead."""
    try:
        s = int(seed)
    except (TypeError, ValueError):
        raise ValueError(f"seed must be an integer, got {seed!r}") from None
    if s != seed or s < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    return s


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def _axis_roll(p: dict) -> float:
    return 90.0 if str(p.get("axis_up", "Z")).upper() == "Y" else 0.0


def _toppled(p: dict, thresh: float = TOPPLE_DEG) -> bool:
    return (abs(_wrap180(p["roll_deg"] - _axis_roll(p))) > thresh
            or abs(_wrap180(p["pitch_deg"])) > thresh)


def _tilted(p: dict, eps: float = 1e-6) -> bool:
    return (abs(_wrap180(p["roll_deg"] - _axis_roll(p))) > eps
            or abs(_wrap180(p["pitch_deg"])) > eps)


def _in_rect(xy, rect) -> bool:
    return rect[0] <= xy[0] <= rect[2] and rect[1] <= xy[1] <= rect[3]


def _block_id_at(xy, blocks: list[Block]) -> str | None:
    for b in blocks:
        if _in_rect(xy, b.rect):
            return b.id
    return None


def _clip_rect(rect, region, min_m: float = 0.25):
    x0 = max(float(rect[0]), region[0])
    y0 = max(float(rect[1]), region[1])
    x1 = min(float(rect[2]), region[2])
    y1 = min(float(rect[3]), region[3])
    if x1 - x0 < min_m or y1 - y0 < min_m:
        return None
    return (_r(x0), _r(y0), _r(x1), _r(y1))


def _rect_dist(rect, x: float, y: float) -> float:
    dx = max(rect[0] - x, 0.0, x - rect[2])
    dy = max(rect[1] - y, 0.0, y - rect[3])
    return math.hypot(dx, dy)


def _building_category(cfg: dict) -> str:
    """House vs mid-rise is a locale/asset-set property: the suburban set is
    detached bungalows, the urban set is 30x20x24 m blocks."""
    name = f"{cfg.get('locale', '')} {cfg.get('asset_set', '')}".lower()
    if "suburb" in name or "rural" in name:
        return "house"
    if "urban" in name or "downtown" in name:
        return "midrise"
    return "house"


@contextlib.contextmanager
def _quiet(verbose: bool):
    """The generator narrates to stdout; the exporter is a library."""
    if verbose:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


__all__ = ["GENERATOR_VERSION", "PROP_SPEC", "TOPPLE_DEG", "BLOCK_TYPOLOGY",
           "BUILDING_CATEGORY", "V2_LOCALE_PRESET", "available_presets",
           "load_generator_config", "export_scene", "build_v2", "sample_region",
           "sample_severity", "scene_summary", "format_summary_table",
           "HumanConfig"]
