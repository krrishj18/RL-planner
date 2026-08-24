"""Scene sampling, area scaling and the train / held-out split.

    --scenes synthetic:0-200      seeds [0, 200)  of schema.make_synthetic_scene
    --scenes synthetic:200        same as synthetic:0-200
    --scenes 'data/scenes/*.json' glob(s), comma- or space-separated

Area scaling (`--robots auto`, `--t-max auto`), applied per scene at env construction:

    s          = sqrt(area_km2 / 0.16)              # 0.16 km2 = the 400 x 400 m reference
    n_robots   = clip(round(3 * s), 3, 8)           # ties round down
    t_max_s    = clip(600 * s, 600, cap)            # cap = --t-max-cap, default 1500

    400x400 m  -> 3 robots /  600 s      (s = 1.0)
    1000x1000  -> 7 robots / 1500 s      (s = 2.5, 7.5 robots -> 7)
    1500x1500  -> 8 robots / 1500 s      (s = 3.75, 11.25 robots -> clipped)

A 3-robot team at 600 s sweeps ~58k m2, so both the team and the clock grow with the linear
extent of the region, not its area: doubling the side doubles robots and time until the caps bite.
The horizon cap is the one knob of the rule a run may move (`--t-max-cap`, `SceneBank.t_max_cap`):
1500 s stops a 1500 x 1500 m scene at the same clock as a 1000 x 1000 m one, so a long-episode
study raises it without touching `--t-max`, which overrides the rule outright.

Size buckets are the area terciles of the bank (`small`/`medium`/`large`, by value, so a bank of
equal-area scenes is a single bucket); they drive `--scene-mix` sampling and the stratified
held-out split (stratum = disaster x bucket).
"""
from __future__ import annotations

import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..scene import schema
from ..sim.config import EnvConfig

DEFAULT_REGION = (240.0, 240.0)
_RANGE = re.compile(r"^(\d+)\s*[-:]\s*(\d+)$")

AUTO = -1                       # sentinel for --robots auto / --t-max auto
REF_AREA_KM2 = 0.16             # 400 x 400 m
REF_ROBOTS = 3
REF_T_MAX_S = 600.0
ROBOTS_MIN, ROBOTS_MAX = 3, 8
T_MAX_MIN_S, T_MAX_MAX_S = 600.0, 1500.0
BUCKETS = ("small", "medium", "large")
_HEADER_BYTES = 8192            # scene meta is the first object in the file


@dataclass(frozen=True)
class SceneKey:
    kind: str          # "synthetic" | "file"
    ref: str           # seed as str, or path

    def __str__(self) -> str:
        return f"{self.kind}:{self.ref}"


@dataclass(frozen=True)
class SceneMeta:
    region_m: tuple[float, float]
    area_km2: float
    disaster: str


# ---- area scaling ------------------------------------------------------------------------------
def area_scale(area_km2: float) -> float:
    """Linear size of the region relative to the 400 x 400 m reference."""
    return math.sqrt(max(float(area_km2), 1e-9) / REF_AREA_KM2)


def auto_robots(area_km2: float) -> int:
    """clip(round(3 * s), 3, 8); ties round down so 1000 x 1000 m (7.5) gives 7."""
    n = math.floor(REF_ROBOTS * area_scale(area_km2) + 0.5 - 1e-9)
    return int(min(max(n, ROBOTS_MIN), ROBOTS_MAX))


def auto_t_max(area_km2: float, cap: float = T_MAX_MAX_S) -> float:
    """clip(600 * s, 600, cap) seconds; `cap` defaults to the shipped 1500 s."""
    c = float(cap) if float(cap) > 0 else T_MAX_MAX_S
    t = REF_T_MAX_S * area_scale(area_km2)
    return float(min(max(t, T_MAX_MIN_S), max(c, T_MAX_MIN_S)))


def region_area_km2(region_m: Sequence[float]) -> float:
    return float(region_m[0]) * float(region_m[1]) / 1e6


# ---- parsing -----------------------------------------------------------------------------------
def parse_scenes(spec: str) -> list[SceneKey]:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("--scenes is empty")
    if spec == "synthetic" or spec.startswith("synthetic:"):
        rest = spec[len("synthetic:"):] if ":" in spec else ""
        seeds = _parse_range(rest) if rest else range(0, 200)
        return [SceneKey("synthetic", str(s)) for s in seeds]
    paths: list[str] = []
    for pat in re.split(r"[,\s]+", spec):
        if not pat:
            continue
        hits = sorted(glob.glob(pat))
        if not hits:
            raise ValueError(f"--scenes: pattern {pat!r} matched no files")
        paths += hits
    if not paths:
        raise ValueError(f"--scenes: nothing matched {spec!r}")
    seen: dict[str, None] = {}
    for p in paths:
        seen.setdefault(p, None)
    return [SceneKey("file", p) for p in seen]


def _parse_range(s: str) -> Iterable[int]:
    s = s.strip()
    m = _RANGE.match(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b <= a:
            raise ValueError(f"--scenes: empty seed range {s!r} (end is exclusive)")
        return range(a, b)
    if s.isdigit():
        return range(0, int(s))
    raise ValueError(f"--scenes: cannot parse seed range {s!r}; use A-B, A:B or N")


def parse_robots(s: str | int) -> tuple[int, int]:
    """'3' -> (3, 3); '2-4' -> (2, 4) (uniform per episode); 'auto' -> (AUTO, AUTO)."""
    if isinstance(s, int):
        return (s, s)
    s = str(s).strip()
    if s.lower() == "auto":
        return (AUTO, AUTO)
    m = _RANGE.match(s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi < lo:
            raise ValueError(f"--robots: bad range {s!r}")
        return (lo, hi)
    if not s.isdigit():
        raise ValueError(f"--robots: expected N, LO-HI or 'auto', got {s!r}")
    return (int(s), int(s))


def parse_t_max(s: str | float | None) -> float:
    """'auto' -> AUTO; a number -> that many seconds; None -> AUTO."""
    if s is None:
        return float(AUTO)
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s.lower() == "auto":
        return float(AUTO)
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"--t-max: expected seconds or 'auto', got {s!r}") from None
    if v <= 0:
        raise ValueError(f"--t-max must be > 0, got {v}")
    return v


def parse_scene_mix(s: str | None) -> dict[str, float] | None:
    """'small:0.5,medium:0.3,large:0.2' -> normalised weights per size bucket (None = uniform)."""
    if s is None or not str(s).strip() or str(s).strip().lower() in ("uniform", "none"):
        return None
    out: dict[str, float] = {}
    for part in re.split(r"[,\s]+", str(s).strip()):
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"--scene-mix: expected bucket:weight, got {part!r}")
        name, _, w = part.partition(":")
        name = name.strip().lower()
        if name not in BUCKETS:
            raise ValueError(f"--scene-mix: unknown bucket {name!r}; use {'/'.join(BUCKETS)}")
        try:
            val = float(w)
        except ValueError:
            raise ValueError(f"--scene-mix: weight for {name!r} is not a number: {w!r}") from None
        if val < 0:
            raise ValueError(f"--scene-mix: negative weight for {name!r}")
        out[name] = out.get(name, 0.0) + val
    tot = sum(out.values())
    if tot <= 0:
        raise ValueError(f"--scene-mix: weights sum to 0 ({s!r})")
    return {k: v / tot for k, v in out.items()}


# ---- scene metadata ----------------------------------------------------------------------------
_META_CACHE: dict[str, SceneMeta] = {}


def read_scene_meta(path: str | Path) -> SceneMeta:
    """Region and disaster type from the file header (scene JSON is multi-MB; meta is first)."""
    p = str(path)
    hit = _META_CACHE.get(p)
    if hit is not None:
        return hit
    with open(p, "rb") as fh:
        head = fh.read(_HEADER_BYTES).decode("utf-8", "ignore")
    meta = _meta_object(head)
    if meta is None:
        meta = (json.loads(Path(p).read_text()) or {}).get("meta", {})
    region = [float(v) for v in meta.get("region", (0.0, 0.0, *DEFAULT_REGION))]
    wh = (region[2] - region[0], region[3] - region[1])
    m = SceneMeta(region_m=wh, area_km2=region_area_km2(wh),
                  disaster=str(meta.get("disaster_type", "unknown")))
    _META_CACHE[p] = m
    return m


def _meta_object(text: str) -> dict[str, Any] | None:
    i = text.find('"meta"')
    j = text.find("{", i) if i >= 0 else -1
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[j: k + 1])
                except json.JSONDecodeError:
                    return None
    return None


class SceneBank:
    """Loads/caches scenes and rasters; area metadata, size buckets and the held-out split.

    The caches are bounded by raster cells, not by entry count: a v2 bank mixes 0.3 km2 scenes
    (~75k cells, ~1 MB parsed) with 2.1 km2 ones (~530k cells, ~40 MB parsed), and an unbounded
    cache walks a worker up to several GB as its slots rotate through the shard.
    """

    SCENE_CACHE_CELLS = 1_500_000     # ~3 of the largest v2 scenes, or 100 synthetic ones
    RASTER_CACHE_CELLS = 4_000_000    # rasters are ~13 bytes/cell: ~50 MB

    def __init__(self, spec: str, region_m: Sequence[float] = DEFAULT_REGION,
                 holdout_frac: float = 0.1, t_max_cap: float = T_MAX_MAX_S):
        self.spec = str(spec)
        self.holdout_frac = float(holdout_frac)
        c = float(t_max_cap)
        # the *effective* cap: `auto_t_max` clamps it to the floor and reads non-positive as the
        # default, so a run that records `bank.t_max_cap` records the horizon it actually ran
        self.t_max_cap = max(c, T_MAX_MIN_S) if c > 0 else T_MAX_MAX_S
        self.keys = parse_scenes(spec)
        self.region_m = (float(region_m[0]), float(region_m[1]))
        self.metas = {k: self._meta(k) for k in self.keys}
        areas = np.array([self.metas[k].area_km2 for k in self.keys], np.float64)
        q = np.quantile(areas, [1 / 3, 2 / 3]) if areas.size else np.zeros(2)
        self.bucket_edges = (float(q[0]), float(q[1]))
        self.buckets = {k: self._bucket(self.metas[k].area_km2) for k in self.keys}
        self.train, self.heldout = self._split()
        self._scenes: dict[SceneKey, Any] = {}
        self._rasters: dict[tuple[SceneKey, float], Any] = {}

    # -- metadata --------------------------------------------------------------------------
    def _meta(self, key: SceneKey) -> SceneMeta:
        if key.kind == "file":
            return read_scene_meta(key.ref)
        return SceneMeta(region_m=self.region_m, area_km2=region_area_km2(self.region_m),
                         disaster="synthetic")

    def _bucket(self, area_km2: float) -> str:
        lo, hi = self.bucket_edges
        return BUCKETS[0] if area_km2 <= lo else (BUCKETS[1] if area_km2 <= hi else BUCKETS[2])

    def meta(self, key: SceneKey) -> SceneMeta:
        return self.metas.get(key) or self._meta(key)

    def area(self, key: SceneKey) -> float:
        return self.meta(key).area_km2

    def bucket(self, key: SceneKey) -> str:
        return self.buckets.get(key) or self._bucket(self.area(key))

    def disaster(self, key: SceneKey) -> str:
        return self.meta(key).disaster

    def env_params(self, key: SceneKey, n_robots: int = AUTO,
                   t_max: float = AUTO) -> tuple[int, float]:
        """Resolve (n_robots, t_max_s) for one scene; non-positive arguments mean 'auto'."""
        a = self.area(key)
        nr = auto_robots(a) if int(n_robots) <= 0 else int(n_robots)
        tm = auto_t_max(a, self.t_max_cap) if float(t_max) <= 0 else float(t_max)
        return int(nr), float(tm)

    def robot_bounds(self, robots: tuple[int, int], split: str = "all") -> tuple[int, int]:
        """Actual (min, max) robot count of a run — the padded R is the max."""
        if int(robots[1]) > 0:
            return (int(robots[0]), int(robots[1]))
        n = [auto_robots(self.area(k)) for k in self.split(split)] or [ROBOTS_MIN]
        return (min(n), max(n))

    def t_max_bounds(self, t_max: float = AUTO, split: str = "all") -> tuple[float, float]:
        if float(t_max) > 0:
            return (float(t_max), float(t_max))
        v = [auto_t_max(self.area(k), self.t_max_cap) for k in self.split(split)] \
            or [T_MAX_MIN_S]
        return (min(v), max(v))

    # -- split -----------------------------------------------------------------------------
    def _split(self) -> tuple[list[SceneKey], list[SceneKey]]:
        """Held-out keys are drawn per (disaster, size bucket) stratum, so every disaster and
        every size bucket is represented in both splits at (about) the bank's holdout fraction."""
        if len(self.keys) <= 1:
            return list(self.keys), list(self.keys)
        strata: dict[tuple[str, str], list[int]] = {}
        for i, k in enumerate(self.keys):
            strata.setdefault((self.disaster(k), self.bucket(k)), []).append(i)
        hold = np.zeros(len(self.keys), np.bool_)
        for _, idx in sorted(strata.items()):
            n_hold = max(1, int(round(self.holdout_frac * len(idx)))) if len(idx) > 1 else 0
            for i in idx[len(idx) - n_hold:]:
                hold[i] = True
        train = [k for i, k in enumerate(self.keys) if not hold[i]]
        heldout = [k for i, k in enumerate(self.keys) if hold[i]]
        return (train or list(self.keys), heldout or list(self.keys))

    def __len__(self) -> int:
        return len(self.keys)

    def split(self, name: str) -> list[SceneKey]:
        if name not in ("train", "heldout", "all"):
            raise ValueError(f"unknown split {name!r}")
        return {"train": self.train, "heldout": self.heldout, "all": self.keys}[name]

    def bucket_counts(self, split: str = "all") -> dict[str, int]:
        out = {b: 0 for b in BUCKETS}
        for k in self.split(split):
            out[self.bucket(k)] += 1
        return out

    # -- sampling --------------------------------------------------------------------------
    def mix_weights(self, keys: Sequence[SceneKey],
                    mix: dict[str, float] | None) -> np.ndarray | None:
        """Per-key sampling probabilities for a bucket mix (uniform inside a bucket).

        Weights of buckets with no key in `keys` are dropped and the rest renormalised, so a
        mix stays usable on a bank (or worker shard) that holds only some sizes.
        """
        if not mix:
            return None
        buckets = [self.bucket(k) for k in keys]
        present = {b for b in buckets if mix.get(b, 0.0) > 0}
        if not present:
            return None
        counts = {b: buckets.count(b) for b in present}
        tot = sum(mix[b] for b in present)
        w = np.array([mix.get(b, 0.0) / (tot * counts[b]) if b in present else 0.0
                      for b in buckets], np.float64)
        return w / w.sum()

    def sample(self, split: str, rng: np.random.Generator,
               mix: dict[str, float] | None = None) -> SceneKey:
        keys = self.split(split)
        w = self.mix_weights(keys, mix)
        if w is None:
            return keys[int(rng.integers(len(keys)))]
        return keys[int(rng.choice(len(keys), p=w))]

    # -- scenes / envs ---------------------------------------------------------------------
    def cells(self, key: SceneKey, cell_m: float = 2.0) -> int:
        """Raster cells of a scene at `cell_m` — the size proxy the caches are budgeted in."""
        w, h = self.meta(key).region_m
        return int(max(1.0, w / cell_m) * max(1.0, h / cell_m))

    def scene(self, key: SceneKey):
        sc = self._scenes.get(key)
        if sc is None:
            sc = (schema.make_synthetic_scene(int(key.ref), region_m=self.region_m)
                  if key.kind == "synthetic" else schema.Scene.from_json(key.ref))
            self._scenes[key] = sc
            _trim(self._scenes, self.SCENE_CACHE_CELLS, lambda k: self.cells(k))
        return sc

    def raster(self, key: SceneKey, cell_m: float):
        from ..sim.raster import rasterize
        ck = (key, float(cell_m))
        r = self._rasters.get(ck)
        if r is None:
            r = rasterize(self.scene(key), float(cell_m))
            self._rasters[ck] = r
            _trim(self._rasters, self.RASTER_CACHE_CELLS, lambda k: self.cells(k[0], k[1]))
        return r

    def make_env(self, key: SceneKey, cfg: EnvConfig, seed: int, env_cls=None):
        """Build a DisasterEnv on a cached (read-only) raster."""
        if env_cls is None:
            from ..sim.env import DisasterEnv as env_cls
        return env_cls(self.scene(key), cfg, seed=int(seed),
                       raster=self.raster(key, cfg.raster.cell_m))


def _trim(cache: dict, budget: int, size, keep_last: bool = True) -> None:
    """Drop oldest entries until the cache fits `budget` size units (always keep the newest)."""
    total = sum(size(k) for k in cache)
    for k in list(cache):
        if total <= budget or (keep_last and len(cache) <= 1):
            break
        total -= size(k)
        del cache[k]


__all__ = ["SceneBank", "SceneKey", "SceneMeta", "parse_scenes", "parse_robots", "parse_t_max",
           "parse_scene_mix", "read_scene_meta", "area_scale", "auto_robots", "auto_t_max",
           "region_area_km2", "DEFAULT_REGION", "AUTO", "BUCKETS", "REF_AREA_KM2",
           "T_MAX_MIN_S", "T_MAX_MAX_S", "ROBOTS_MIN", "ROBOTS_MAX"]
