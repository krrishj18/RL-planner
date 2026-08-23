"""Adversarial regressions for the v2 (detailed city) export path.

The whole disaster x seed x size x severity matrix through one geometry audit
(humans, buildings, roads, blocks, debris, spawns, damage grid), plus the region
sampler's rounding rule and the CLI's error paths. Each test here pins a bug
that was live at some point.
"""
import contextlib
import io
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rlplanner.scene import schema as S
from rlplanner.scene.casualties import HumanConfig
from rlplanner.scene.export import (_Resolver, build_v2, export_scene,
                                    load_generator_config, sample_region)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_scenes  # noqa: E402

DISASTERS = ("earthquake", "tornado", "explosion", "flood", "hurricane")
SEEDS = (0, 17)
SIZES = ((500.0, 500.0), (1010.0, 730.0))
ASPECTS = ((500.0, 1500.0), (1500.0, 500.0))


def v2(seed=0, disaster="earthquake", region_m=(800.0, 800.0), severity=0.5, **kw):
    kw.setdefault("size_jitter", 0.25)
    return export_scene("downtown", seed, pipeline="v2", disaster=disaster,
                        region_m=region_m, severity=severity, **kw)


# ---- geometry helpers ------------------------------------------------------
def obb(cx, cy, sx, sy, yaw_deg, px, py, pad=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = px - cx, py - cy
    return (abs(dx * c + dy * s) <= sx / 2 + pad
            and abs(-dx * s + dy * c) <= sy / 2 + pad)


def in_any_obb(boxes, px, py, pad=0.0):
    """boxes = (n, 5) of cx, cy, sx, sy, yaw_deg."""
    if not len(boxes):
        return False
    a = np.radians(boxes[:, 4])
    c, s = np.cos(a), np.sin(a)
    dx, dy = px - boxes[:, 0], py - boxes[:, 1]
    u, v = dx * c + dy * s, -dx * s + dy * c
    return bool(np.any((np.abs(u) <= boxes[:, 2] / 2 + pad)
                       & (np.abs(v) <= boxes[:, 3] / 2 + pad)))


def union_area(rects):
    """Exact union of axis-aligned rects, by coordinate compression."""
    rects = [r for r in rects if r[2] > r[0] and r[3] > r[1]]
    if not rects:
        return 0.0
    xs = np.unique(np.array([v for r in rects for v in (r[0], r[2])], dtype=float))
    ys = np.unique(np.array([v for r in rects for v in (r[1], r[3])], dtype=float))
    grid = np.zeros((ys.size - 1, xs.size - 1), dtype=bool)
    for r in rects:
        grid[np.searchsorted(ys, r[1]):np.searchsorted(ys, r[3]),
             np.searchsorted(xs, r[0]):np.searchsorted(xs, r[2])] = True
    return float((grid * (np.diff(ys)[:, None] * np.diff(xs)[None, :])).sum())


def half_up(v: float) -> int:
    """The exporter rounds halves up; python's round() rounds them to even."""
    return int(math.floor(float(v) + 0.5))


def auto_counts(region) -> tuple[int, int]:
    w, h = region[2] - region[0], region[3] - region[1]
    n = min(80, max(10, half_up(15 * (w * h / 1e6) / 0.16)))
    return n, half_up(n / 2)


def solid_boxes(sc):
    return np.array([[b.center[0], b.center[1], b.size[0], b.size[1], b.yaw_deg]
                     for b in sc.buildings if b.fate != "destroyed"],
                    dtype=float).reshape(-1, 5)


def problems(sc: S.Scene) -> list[str]:
    """Every invariant the v2 export claims, as a list of failures."""
    bad: list[str] = []
    x0, y0, x1, y1 = sc.region
    bad += S.validate(sc)
    solid = solid_boxes(sc)
    by_bld = {b.id: b for b in sc.buildings}
    by_veh = {v.id: v for v in sc.vehicles}
    for h in sc.humans:
        hx, hy = h.pos[0], h.pos[1]
        if not (x0 <= hx <= x1 and y0 <= hy <= y1):
            bad.append(f"human {h.id} outside the region")
        if h.container != "building" and in_any_obb(solid, hx, hy):
            bad.append(f"human {h.id} ({h.container}) under a standing roof")
        if h.container in ("rubble", "building"):
            b = by_bld[h.container_id]
            want = "destroyed" if h.container == "rubble" else "damaged"
            if (b.fate == "destroyed") != (want == "destroyed"):
                bad.append(f"human {h.id}: {h.container} in a {b.fate} building")
            if not obb(*b.center, *b.size, b.yaw_deg, hx, hy, pad=1e-3):
                bad.append(f"human {h.id} off its own building {b.id}")
        if h.container == "vehicle":
            v = by_veh[h.container_id]
            if v.state != "toppled":
                bad.append(f"human {h.id} at an intact car")
            if not obb(*v.center, *v.size, v.yaw_deg, hx, hy, pad=1e-3):
                bad.append(f"human {h.id} beside its car {v.id}")
    if len(sc.humans) > 1:
        p = np.array([[h.pos[0], h.pos[1]] for h in sc.humans])
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        np.fill_diagonal(d, np.inf)
        if d.min() < HumanConfig.min_separation_m - 1e-9:
            bad.append(f"humans {d.min():.3f} m apart")
    for b in sc.buildings:
        if not (x0 <= b.center[0] <= x1 and y0 <= b.center[1] <= y1):
            bad.append(f"building {b.id} centred outside the region")
    for r in sc.roads:
        if not (x0 - 1e-6 <= r.rect[0] < r.rect[2] <= x1 + 1e-6
                and y0 - 1e-6 <= r.rect[1] < r.rect[3] <= y1 + 1e-6):
            bad.append(f"road {r.id} not clipped to the region")
    ruins = {b.id for b in sc.buildings if b.fate == "destroyed"}
    if any(d.building_id not in ruins for d in sc.debris if d.building_id):
        bad.append("debris linked to a building that is not destroyed")
    empty = [b.rect for b in sc.blocks if b.typology in ("park", "other")]
    if any(any(r[0] <= b.center[0] <= r[2] and r[1] <= b.center[1] <= r[3]
               for r in empty) for b in sc.buildings):
        bad.append("a building stands on a park or unzoned block")
    carriage = [r.rect for r in sc.roads if r.kind == "road"]
    every = np.array([[b.center[0], b.center[1], b.size[0], b.size[1], b.yaw_deg]
                      for b in sc.buildings], dtype=float).reshape(-1, 5)
    if len(sc.robots_spawn) != 8:
        bad.append(f"{len(sc.robots_spawn)} spawn points, want 8")
    for i, s in enumerate(sc.robots_spawn):
        if not any(r[0] <= s[0] <= r[2] and r[1] <= s[1] <= r[3] for r in carriage):
            bad.append(f"spawn[{i}] is not on a carriageway")
        if in_any_obb(every, s[0], s[1], pad=0.5):
            bad.append(f"spawn[{i}] is inside a building")
    if len(sc.robots_spawn) > 1:
        q = np.array([[s[0], s[1]] for s in sc.robots_spawn])
        sd = np.hypot(q[:, None, 0] - q[None, :, 0], q[:, None, 1] - q[None, :, 1])
        np.fill_diagonal(sd, np.inf)
        if sd.min() < 5.0:
            bad.append(f"spawns {sd.min():.2f} m apart")
    g = sc.damage_field.grid
    if g["nx"] * g["cell_m"] < (x1 - x0) - 1e-6 or g["ny"] * g["cell_m"] < (y1 - y0) - 1e-6:
        bad.append("damage grid does not cover the region")
    rng = np.random.default_rng(0)
    for _ in range(50):
        i, j = int(rng.integers(g["nx"])), int(rng.integers(g["ny"]))
        cx = min(x0 + (i + 0.5) * g["cell_m"], x1)
        cy = min(y0 + (j + 0.5) * g["cell_m"], y1)
        want = min(1.0, max(0.0, sc.damage_field.value_at(cx, cy, sc.region)))
        if abs(g["values"][j][i] - want) > 5e-4:
            bad.append(f"damage grid != field at ({cx:.1f}, {cy:.1f})")
            break
    return bad


# ---- the matrix ------------------------------------------------------------
@pytest.mark.parametrize("region_m", SIZES)
@pytest.mark.parametrize("severity", [0.0, 1.0])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("disaster", DISASTERS)
def test_v2_matrix_is_geometrically_sound(disaster, seed, severity, region_m):
    sc = v2(seed, disaster, region_m, severity)
    assert problems(sc) == []
    assert sc.buildings and sc.roads and sc.blocks and sc.humans


@pytest.mark.parametrize("region_m", ASPECTS)
@pytest.mark.parametrize("disaster", ["tornado", "flood"])
def test_extreme_aspects_stay_sound(disaster, region_m):
    sc = v2(9999, disaster, region_m, 1.0)
    assert problems(sc) == []


@pytest.mark.parametrize("disaster", DISASTERS)
def test_severity_zero_is_a_pristine_but_populated_city(disaster):
    sc = v2(1, disaster, (700.0, 500.0), 0.0)
    assert {b.fate for b in sc.buildings} == {"intact"}
    assert sc.debris == [] and all(v.state == "intact" for v in sc.vehicles)
    assert sc.casualties() and {h.container for h in sc.casualties()} == {"open"}
    assert problems(sc) == []


# ---- blocks tile the region ------------------------------------------------
@pytest.mark.parametrize("region_m", [(500.0, 500.0), (1500.0, 500.0), (1010.0, 730.0)])
def test_blocks_tile_the_region_without_overlapping(region_m):
    """Blocks and corridors partition the region: no block overlaps another or a
    carriageway, and together they leave only the border ring uncovered."""
    sc = v2(3, "earthquake", region_m, 0.5)
    x0, y0, x1, y1 = sc.region
    area = (x1 - x0) * (y1 - y0)
    blocks = [b.rect for b in sc.blocks]
    roads = [r.rect for r in sc.roads if r.kind == "road"]
    summed = sum((r[2] - r[0]) * (r[3] - r[1]) for r in blocks)
    assert summed - union_area(blocks) < 1e-3 * area
    assert union_area(blocks) + union_area(roads) - union_area(blocks + roads) \
        < 1e-3 * area
    assert (area - union_area(blocks + roads)) / area < 0.01


def test_park_blocks_hold_no_buildings_and_keep_their_ring():
    sc = v2(0, "earthquake", (1200.0, 1200.0), 0.5)
    parks = [b for b in sc.blocks if b.typology == "park"]
    assert parks
    for p in parks:
        assert not [b for b in sc.buildings
                    if p.rect[0] <= b.center[0] <= p.rect[2]
                    and p.rect[1] <= b.center[1] <= p.rect[3]]
        assert p.built_frac == 0.0


def test_unzoned_slivers_are_small_and_empty():
    """`typology="other"` is a block the rezoning pass refused as too small; it
    must stay empty rather than pick up an infill building. Seed 17 at this
    extent is the worst case in the shipped dataset (24 of 73 blocks)."""
    sc = v2(17, "earthquake", (650.0, 1490.0), 0.5)
    slivers = [b for b in sc.blocks if b.typology == "other"]
    assert slivers
    assert max(min(b.rect[2] - b.rect[0], b.rect[3] - b.rect[1])
               for b in slivers) < 25.0
    for s in slivers:
        assert s.built_frac == 0.0
        assert not [b for b in sc.buildings
                    if s.rect[0] <= b.center[0] <= s.rect[2]
                    and s.rect[1] <= b.center[1] <= s.rect[3]]


def test_corridor_class_lane_count_and_width_agree():
    cfg, _prov = load_generator_config("downtown", 0, region_m=(900.0, 700.0),
                                       severity=0.5, disaster="earthquake")
    resolver = _Resolver(1.0, cfg.get("fallback_sizes", {}), measure=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _pl, layout = build_v2(cfg, resolver)
    sc = v2(0, "earthquake", (900.0, 700.0), 0.5)
    lane_w = float(cfg["roads"]["lane_width_m"])
    roads = {r.id: r for r in sc.roads if r.kind == "road"}
    by_class: dict[str, set[int]] = {}
    for i, c in enumerate(layout["road_corridors"]):
        r = roads.get(f"road{i}")
        if r is None:
            continue
        by_class.setdefault(str(c.get("road_class")), set()).add(r.n_lanes)
        assert r.n_lanes == int(c["n_lanes"])
        w = (r.rect[2] - r.rect[0]) if c["dir"] == "ns" else (r.rect[3] - r.rect[1])
        clipped = ((r.rect[0] <= sc.region[0] or r.rect[2] >= sc.region[2])
                   if c["dir"] == "ns" else
                   (r.rect[1] <= sc.region[1] or r.rect[3] >= sc.region[3]))
        car = float(c["carriage"][1]) - float(c["carriage"][0])
        used = {"both": 2, "none": 0}.get(str(c.get("parking", "both")), 1)
        if not clipped:
            assert abs(w - (car + used * float(c.get("park_w", 0.0)))) < 1e-3
    assert by_class["arterial"] == {int(cfg["layout"]["anisotropic"]["zones"]
                                        ["core"]["lanes_main"])}
    # a wider class never carries fewer lanes than a narrower one
    assert min(by_class["arterial"]) >= max(by_class["local"])
    assert lane_w > 0


# ---- humans ----------------------------------------------------------------
def test_no_casualty_is_left_under_a_roof():
    """Downtown parks cars on strips a building overhangs; the casualty trapped
    at such a car used to be placed inside the building's footprint."""
    for disaster, seed, region_m in (("flood", 0, (500.0, 500.0)),
                                     ("flood", 1, (1010.0, 730.0)),
                                     ("earthquake", 17, (700.0, 900.0))):
        sc = v2(seed, disaster, region_m, 1.0,
                human_cfg=HumanConfig(n_casualties=120, n_bystanders=40))
        solid = solid_boxes(sc)
        for h in sc.humans:
            if h.container == "building":
                continue
            assert not in_any_obb(solid, h.pos[0], h.pos[1]), (disaster, seed, h)


def test_auto_counts_scale_with_the_region():
    cfg = HumanConfig(n_casualties="auto", n_bystanders="auto")
    for region_m, want in (((500.0, 500.0), 23), ((1010.0, 730.0), 69),
                           ((1500.0, 1500.0), 80)):
        sc = v2(0, "earthquake", region_m, 0.5, human_cfg=cfg)
        assert len(sc.casualties()) == want
        assert sum(h.role == "bystander" for h in sc.humans) == half_up(want / 2)
        assert S.validate(sc) == []


# ---- region sampling -------------------------------------------------------
def test_region_rounding_rule():
    """Sizes are multiples of 10 m *inside* the range: bounds that are not
    multiples themselves are pulled to the innermost ones (513..999 -> 520..990),
    and only a range with no multiple at all falls back to a bound."""
    got = [sample_region(s, 513.0, 999.0) for s in range(200)]
    assert got == [sample_region(s, 513.0, 999.0) for s in range(200)]
    for w, h in got:
        assert w % 10 == 0 and h % 10 == 0
        assert 520.0 <= w <= 990.0 and 520.0 <= h <= 990.0
    assert sample_region(4, 500.0, 500.0) == (500.0, 500.0)
    assert all(513.0 <= v <= 517.0 for v in sample_region(4, 513.0, 517.0))
    with pytest.raises(ValueError, match="0 < lo <= hi"):
        sample_region(0, 1500.0, 500.0)
    with pytest.raises(ValueError, match="quantum"):
        sample_region(0, 500.0, 1500.0, quantum=0.0)


@pytest.mark.parametrize("region_m", [(0.0, 500.0), (500.0, -1.0), (0.0, 0.0),
                                      (float("nan"), 500.0)])
def test_degenerate_regions_are_refused_by_name(region_m):
    """A zero extent used to build a city with no roads and die much later
    sampling an empty pool."""
    with pytest.raises(ValueError, match="region_m"):
        v2(0, "earthquake", region_m, 0.5)


def test_negative_size_jitter_is_refused():
    with pytest.raises(ValueError, match="size_jitter"):
        v2(0, "earthquake", (500.0, 500.0), 0.5, size_jitter=-0.2)


# ---- CLI -------------------------------------------------------------------
def test_count_parsing():
    assert export_scenes.parse_count("auto") == "auto"
    assert export_scenes.parse_count(" AUTO ") == "auto"
    assert export_scenes.parse_count("12") == 12
    for text in ("banana", "-3", "1.5"):
        with pytest.raises(Exception):
            export_scenes.parse_count(text)


@pytest.mark.parametrize("argv,msg", [
    (["--region-range", "1500", "500"], "0 < lo <= hi"),
    (["--severity-range", "1.0", "0.5"], "severity range"),
    (["--region", "0", "500"], "region_m"),
    (["--size-jitter", "-0.5", "--region", "500", "500"], "size_jitter"),
])
def test_cli_turns_bad_ranges_into_usage_errors(tmp_path, argv, msg, capsys):
    """A CLI reports bad input on one line; it does not unroll its own stack."""
    base = ["--pipeline", "v2", "--locale", "downtown", "--disaster", "earthquake",
            "--seeds", "0", "--out", str(tmp_path)]
    with pytest.raises(SystemExit):
        export_scenes.main(base + argv)
    assert msg in capsys.readouterr().err


def test_cli_auto_counts_end_to_end(tmp_path):
    argv = ["--pipeline", "v2", "--locale", "downtown", "--disaster", "earthquake",
            "--seeds", "0,1", "--region-range", "500", "1500", "--size-jitter",
            "0.25", "--casualties", "auto", "--bystanders", "auto",
            "--out", str(tmp_path)]
    assert export_scenes.main(argv) == 0
    for p in sorted(tmp_path.glob("*.json")):
        sc = S.Scene.from_json(p)
        want, byst = auto_counts(sc.region)
        assert len(sc.casualties()) == want
        assert sum(x.role == "bystander" for x in sc.humans) == byst
        assert S.validate(sc) == []


def test_v2_json_is_byte_identical_across_processes():
    code = ("import json,hashlib;"
            "from rlplanner.scene.export import export_scene;"
            "from rlplanner.scene.casualties import HumanConfig;"
            "print(hashlib.sha256(json.dumps(export_scene('downtown',5,"
            "pipeline='v2',disaster='explosion',region_m=(700.,510.),severity=0.6,"
            "size_jitter=0.25,human_cfg=HumanConfig(n_casualties='auto',"
            "n_bystanders='auto')).to_dict()).encode()).hexdigest())")
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout
            for h in ("0", "1", "12345")]
    assert len(set(outs)) == 1


def test_the_shipped_dataset_matches_the_documented_command():
    """data/scenes_v2 is the CLI's own output: 180 scenes, area-scaled people."""
    d = Path(__file__).resolve().parents[1] / "data" / "scenes_v2"
    if not d.is_dir():
        pytest.skip("dataset not exported")
    files = sorted(d.glob("*.json"))
    assert len(files) == 180
    for p in files[::17]:
        sc = S.Scene.from_json(p)
        assert S.validate(sc) == []
        assert sc.meta.generator_version == "v2"
        w = sc.region[2] - sc.region[0]
        h = sc.region[3] - sc.region[1]
        assert 500.0 <= w <= 1500.0 and 500.0 <= h <= 1500.0
        want, byst = auto_counts(sc.region)
        assert len(sc.casualties()) == want
        assert sum(x.role == "bystander" for x in sc.humans) == byst
        assert problems(sc) == []


# ---- spawns ----------------------------------------------------------------
SPAWN_SIZES = [(500.0, 500.0), (500.0, 1500.0), (1500.0, 500.0), (1010.0, 730.0),
               (1500.0, 1500.0)]


@pytest.mark.parametrize("region_m", SPAWN_SIZES)
@pytest.mark.parametrize("pipeline", ["v1", "v2"])
def test_eight_spawns_on_the_road_and_apart(pipeline, region_m):
    """The team grew to 8 robots; 4 spawns had the extra robots stacking on the
    last point with only `spawn_jitter_m` to separate them."""
    sc = (v2(4, "tornado", region_m, 0.7) if pipeline == "v2"
          else export_scene("earthquake", 4, region_m=region_m))
    assert len(sc.robots_spawn) == 8
    x0, y0, x1, y1 = sc.region
    carriage = [r.rect for r in sc.roads if r.kind == "road"]
    boxes = np.array([[b.center[0], b.center[1], b.size[0], b.size[1], b.yaw_deg]
                      for b in sc.buildings], dtype=float).reshape(-1, 5)
    for x, y, z in sc.robots_spawn:
        assert z == 20.0
        assert x0 <= x <= x1 and y0 <= y <= y1
        assert any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in carriage)
        assert not in_any_obb(boxes, x, y)
    p = np.array([[s[0], s[1]] for s in sc.robots_spawn])
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 5.0, d.min()
    assert S.validate(sc) == []


@pytest.mark.parametrize("corner", ["ll", "lr", "ul", "ur"])
def test_the_first_four_spawns_never_move(corner):
    """Growing the team must not re-space the four points a 3-robot run used:
    the step is still sized for four."""
    from rlplanner.scene.export import _Assembler
    for region_m in ((500.0, 1500.0), (700.0, 700.0)):
        sc = v2(2, "earthquake", region_m, 0.5, spawn_corner=corner)
        cfg, prov = load_generator_config("downtown", 2, region_m=region_m,
                                          severity=0.5, disaster="earthquake")
        resolver = _Resolver(1.0, cfg.get("fallback_sizes", {}), measure=False,
                             jitter=0.25)
        with contextlib.redirect_stdout(io.StringIO()):
            placements, layout = build_v2(cfg, resolver)
        a = _Assembler(cfg, prov, resolver, placements, layout, 2.0, corner, "v2")
        assert sc.robots_spawn[:4] == a._spawn(n=4)
        assert sc.robots_spawn == a._spawn(n=8)
