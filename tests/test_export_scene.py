"""End-to-end exporter: every scene validates, is reproducible, and maps the
generator's placements onto the schema the way the sim expects."""
import json
import time
from collections import Counter

import pytest

from rlplanner.scene import schema as S
from rlplanner.scene.casualties import HumanConfig
from rlplanner.scene.export import (export_scene, format_summary_table,
                                    load_generator_config, scene_summary)
from rlplanner.scene.gen import scene_generator as SG

PRESETS = ["earthquake", "tornado", "explosion"]
SEEDS = [0, 1, 2]
SMALL = dict(region_m=(200.0, 200.0))


@pytest.fixture(scope="module")
def scenes():
    return {(p, s): export_scene(p, s, **SMALL) for p in PRESETS for s in SEEDS}


def test_three_presets_three_seeds_validate(scenes):
    assert len(scenes) == 9
    for (preset, seed), sc in scenes.items():
        assert S.validate(sc) == [], (preset, seed)
        assert sc.schema_version == S.SCHEMA_VERSION
        assert sc.meta.preset == preset and sc.meta.seed == seed
        assert sc.buildings and sc.roads and sc.humans and sc.robots_spawn


def test_json_roundtrip(tmp_path, scenes):
    sc = scenes[("earthquake", 0)]
    p = tmp_path / "s.json"
    sc.to_json(p)
    back = S.Scene.from_json(p)
    assert back.to_dict() == sc.to_dict()
    assert S.validate(back) == []


def test_deterministic_json():
    a = json.dumps(export_scene("earthquake", 3, **SMALL).to_dict())
    b = json.dumps(export_scene("earthquake", 3, **SMALL).to_dict())
    assert a == b
    assert a != json.dumps(export_scene("earthquake", 4, **SMALL).to_dict())
    assert a != json.dumps(export_scene("tornado", 3, **SMALL).to_dict())


def test_export_is_fast():
    t0 = time.perf_counter()
    export_scene("earthquake", 0)          # full 400x400 default region
    assert time.perf_counter() - t0 < 10.0


def test_severity_zero_leaves_the_city_intact():
    sc = export_scene("earthquake", 0, severity=0.0, **SMALL)
    assert {b.fate for b in sc.buildings} == {"intact"}
    assert sc.debris == []
    assert all(v.state == "intact" for v in sc.vehicles)
    assert all(p.state == "upright" for p in sc.props)
    assert sc.damage_field.grid["values"][0][0] == 0.0


@pytest.mark.parametrize("preset", PRESETS)
def test_fates_track_severity(preset):
    def wrecked(sev):
        sc = export_scene(preset, 1, severity=sev, **SMALL)
        n = Counter(b.fate for b in sc.buildings)
        return (n["damaged"] + n["destroyed"]) / max(len(sc.buildings), 1)

    assert wrecked(0.0) == 0.0
    assert wrecked(1.0) > wrecked(0.2)


def test_region_override_and_cell_size():
    sc = export_scene("earthquake", 0, region_m=(120.0, 80.0), cell_m=4.0)
    assert sc.region == (-60.0, -40.0, 60.0, 40.0)
    g = sc.damage_field.grid
    assert (g["nx"], g["ny"], g["cell_m"]) == (30, 20, 4.0)
    assert len(g["values"]) == 20 and all(len(r) == 30 for r in g["values"])
    for obj in sc.buildings + sc.vehicles + sc.props:
        assert -70 <= obj.center[0] <= 70 and -50 <= obj.center[1] <= 50


def test_damage_grid_matches_the_analytic_field(scenes):
    sc = scenes[("tornado", 0)]
    g = sc.damage_field.grid
    x0, y0, _x1, _y1 = sc.region
    for j in (0, g["ny"] // 3, g["ny"] - 1):
        for i in (0, g["nx"] // 2, g["nx"] - 1):
            x = x0 + (i + 0.5) * g["cell_m"]
            y = y0 + (j + 0.5) * g["cell_m"]
            assert abs(g["values"][j][i] - sc.damage_at(x, y)) < 1e-3
    assert all(0.0 <= v <= 1.0 for row in g["values"] for v in row)


def test_building_fate_comes_from_the_usd_pool():
    cfg, _prov = load_generator_config("earthquake", 7, region_m=(200.0, 200.0))
    r = SG.SizeResolver(float(cfg["asset_scale"]), cfg["fallback_sizes"], measure=False)
    placements, _layout = SG.build_city(cfg, r)
    root = cfg.get("asset_root", "")
    pools = {k: set(SG._normalize_usd_list(cfg["usds"]["buildings"].get(k) or [],
                                           float(cfg["asset_scale"]), root)[0])
             for k in ("intact", "damaged", "destroyed")}
    sc = export_scene("earthquake", 7, region_m=(200.0, 200.0))
    houses = [p for p in placements if p["category"] == "house"]
    assert len(houses) == len(sc.buildings)
    for p, b in zip(houses, sc.buildings):
        assert (b.center[0], b.center[1]) == (round(p["x_m"], 3), round(p["y_m"], 3))
        if p["usd"] in pools["destroyed"]:
            assert b.fate == "destroyed"
        elif p["usd"] in pools["damaged"]:
            assert b.fate == "damaged"
        else:
            assert b.fate in ("intact", "damaged")   # damaged = tilt/sink stand-in
    assert {b.category for b in sc.buildings} == {"house"}   # suburban locale


def test_downtown_locale_exports_midrise():
    sc = export_scene("tornado", 0, **SMALL)
    assert {b.category for b in sc.buildings} == {"midrise"}
    assert sc.meta.locale == "downtown"


def test_debris_and_vehicles(scenes):
    sc = scenes[("tornado", 0)]
    assert {d.kind for d in sc.debris} <= {"pile", "piece"}
    assert all(d.radius_m > 0 for d in sc.debris)
    ruins = {b.id for b in sc.buildings if b.fate == "destroyed"}
    assert all(d.building_id in ruins for d in sc.debris if d.building_id)
    assert any(d.building_id for d in sc.debris)
    assert any(v.state == "toppled" for v in sc.vehicles)
    assert all(v.kind == "car" and v.size[0] > 0 for v in sc.vehicles)


def test_props_cover_the_street_kit(scenes):
    cats = Counter(p.category for p in scenes[("tornado", 0)].props)
    assert {"bus_stop", "tree", "streetlight", "traffic_light"} <= set(cats)
    assert all(c in S.PROP_CATEGORIES for c in cats)
    toppled = [p for p in scenes[("tornado", 0)].props if p.state == "toppled"]
    assert toppled and all(p.resolved_height() <= 1.0 for p in toppled)


def test_roads_blocks_and_spawn(scenes):
    sc = scenes[("earthquake", 0)]
    kinds = Counter(r.kind for r in sc.roads)
    assert kinds["road"] > 0 and kinds["sidewalk"] > 0
    x0, y0, x1, y1 = sc.region
    for r in sc.roads:
        assert x0 <= r.rect[0] < r.rect[2] <= x1 and y0 <= r.rect[1] < r.rect[3] <= y1
    assert {b.typology for b in sc.blocks} <= set(S.BLOCK_TYPOLOGIES)
    assert all(0.0 <= b.built_frac <= 1.0 for b in sc.blocks)
    assert {b.block_id for b in sc.buildings if b.block_id} <= {b.id for b in sc.blocks}

    roads = [r for r in sc.roads if r.kind == "road"]
    assert len(sc.robots_spawn) == 8
    for (sx, sy, sz) in sc.robots_spawn:
        assert sz == 20.0
        assert any(r.rect[0] <= sx <= r.rect[2] and r.rect[1] <= sy <= r.rect[3]
                   for r in roads)
    # lower-left corner by default
    assert all(s[0] < (x0 + x1) / 2 and s[1] < (y0 + y1) / 2 for s in sc.robots_spawn)


def test_spawn_corner_choices():
    x0, y0, x1, y1 = export_scene("earthquake", 0, **SMALL).region
    ur = export_scene("earthquake", 0, spawn_corner="ur", **SMALL).robots_spawn
    assert all(s[0] > (x0 + x1) / 2 or s[1] > (y0 + y1) / 2 for s in ur)
    with pytest.raises(ValueError):
        export_scene("earthquake", 0, spawn_corner="middle", **SMALL)


def test_human_counts_and_config(scenes):
    sc = scenes[("earthquake", 0)]
    assert len(sc.casualties()) == HumanConfig.n_casualties
    assert len(sc.humans) == HumanConfig.n_casualties + HumanConfig.n_bystanders
    small = export_scene("earthquake", 0, human_cfg=HumanConfig(n_casualties=3,
                                                                n_bystanders=1), **SMALL)
    assert len(small.humans) == 4


def test_bad_arguments():
    with pytest.raises(ValueError):
        export_scene("earthquake", 0, cell_m=0.0)
    with pytest.raises(FileNotFoundError):
        export_scene("no_such_preset", 0)


def test_summary_table(scenes):
    rows = [scene_summary(sc) for sc in list(scenes.values())[:3]]
    out = format_summary_table(rows)
    assert "preset" in out and "destroyed" in out
    assert len(out.splitlines()) == len(rows) + 2


@pytest.mark.parametrize("preset", ["earthquake", "tornado"])
def test_exported_ground_covers_the_generators_tiles(preset):
    """The sidewalk/driveway rects are recomputed from the config rather than read
    off the tile placements, so they have to land on the same ground."""
    from rlplanner.scene.export import _Resolver

    cfg, _prov = load_generator_config(preset, 0, region_m=(200.0, 200.0))
    resolver = _Resolver(float(cfg["asset_scale"]), cfg["fallback_sizes"], measure=False)
    placements, _layout = SG.build_city(cfg, resolver)
    sc = export_scene(preset, 0, region_m=(200.0, 200.0))

    def covered(tiles, rects):
        return all(any(r[0] <= t["x_m"] <= r[2] and r[1] <= t["y_m"] <= r[3]
                       for r in rects) for t in tiles)

    walks = [r.rect for r in sc.roads if r.kind == "sidewalk"]
    paved = walks + [r.rect for r in sc.roads if r.kind == "driveway"]
    sidewalk_tiles = [p for p in placements if p["category"] == "sidewalk"]
    assert sidewalk_tiles and covered(sidewalk_tiles, walks)
    assert covered([p for p in placements if p["category"] == "concrete"], paved)


def test_size_jitter_spreads_footprints():
    flat = export_scene("earthquake", 0, region_m=(200.0, 200.0))
    varied = export_scene("earthquake", 0, region_m=(200.0, 200.0), size_jitter=0.2)
    assert len({b.size for b in flat.buildings}) == 1          # one fallback per category
    assert len({b.size for b in varied.buildings}) > 3
    assert S.validate(varied) == []
