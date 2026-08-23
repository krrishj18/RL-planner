"""The v2 (detailed city) export path: fidelity against the vendored pipeline,
varied non-square regions, and the district/park structure the pipeline adds."""
import contextlib
import io
import json
import resource
import time

import pytest

from rlplanner.scene import schema as S
from rlplanner.scene.export import (BLOCK_TYPOLOGY, BUILDING_CATEGORY, _Resolver,
                                    build_v2, export_scene, load_generator_config,
                                    sample_region, sample_severity)
from rlplanner.scene.gen import generate_scene as GS
from rlplanner.scene.gen.detail import districts as D

DISASTERS = ("earthquake", "tornado", "explosion")
SEEDS = (0, 1, 17, 9999)
SIZES = ((500.0, 500.0), (500.0, 1500.0), (1500.0, 700.0), (1500.0, 1500.0))
BIG = [s for s in SIZES if max(s) >= 1000.0]


def v2(seed=0, disaster="earthquake", region_m=(800.0, 800.0), severity=0.5, **kw):
    return export_scene("downtown", seed, pipeline="v2", disaster=disaster,
                        region_m=region_m, severity=severity, **kw)


@pytest.fixture(scope="module")
def scenes():
    return {(d, s): v2(s, d, (1000.0, 1000.0)) for d in DISASTERS for s in (0, 17)}


# ---- fidelity against the vendored pipeline --------------------------------
def _upstream(cfg, monkeypatch):
    """`generate_scene_on_stage` with every stage-touching call stubbed out.

    Returns `(placements, layout)`; the layout is caught on the way out of
    `build_city`, since the pipeline function does not return it."""
    for name in ("prune_prims", "stamp_asset_provenance"):
        monkeypatch.setattr(GS, name, lambda *a, **k: 0)
    monkeypatch.setattr(GS.sg, "apply_placements", lambda *a, **k: None)
    monkeypatch.setattr(GS.sg, "apply_ground_planes", lambda *a, **k: None)
    # road_markings is not vendored (it only paints a USD stage), so the symbol
    # has to be stubbed rather than replaced.
    monkeypatch.setattr(GS, "road_markings",
                        type("_RM", (), {"apply": staticmethod(lambda *a, **k: None)}))
    seen, real = [], GS.sg.build_city

    def spy(config, resolver):
        placements, layout = real(config, resolver)
        seen.append(layout)
        return placements, layout

    monkeypatch.setattr(GS.sg, "build_city", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        placements = GS.generate_scene_on_stage(None, cfg)
    assert len(seen) == 1
    return placements, seen[0]


def _key(placements):
    return sorted((p["category"], round(p["x_m"], 6), round(p["y_m"], 6),
                   round(p["yaw_deg"], 6), p["usd"]) for p in placements)


@pytest.mark.parametrize("region_m", [(500.0, 500.0), (500.0, 1500.0)])
@pytest.mark.parametrize("disaster", ["earthquake", "tornado"])
@pytest.mark.parametrize("seed", [0, 1, 17])
def test_v2_matches_the_vendored_pipeline(monkeypatch, seed, disaster, region_m):
    """Our export path must run the same sub-steps as `generate_scene_on_stage`:
    same placements, same corridors, same blocks. We only skip the passes that
    write USD (street furniture, road surface, markings)."""
    def cfg_of():
        return load_generator_config("downtown", seed, region_m=region_m,
                                     severity=0.7, disaster=disaster)[0]

    resolver = _Resolver(1.0, cfg_of().get("fallback_sizes", {}), measure=False)
    with contextlib.redirect_stdout(io.StringIO()):
        mine, layout = build_v2(cfg_of(), resolver)
    theirs, up_layout = _upstream(cfg_of(), monkeypatch)

    assert _key(mine) == _key(theirs)
    # ... and the categories the 2.5D export actually consumes, one by one.
    for cat in ("house", "car", "debris", "debris_pile", "tree", "trail",
                "bench", "fence"):
        assert (_key([p for p in mine if p["category"] == cat])
                == _key([p for p in theirs if p["category"] == cat])), cat
    assert layout["blocks"] == up_layout["blocks"]
    assert layout["road_corridors"] == up_layout["road_corridors"]
    assert layout["_typology_of"] == up_layout["_typology_of"]

    # The exported scene keeps exactly those placements, modulo the categories
    # that are not 2.5D geometry (humans are re-placed by casualties.py).
    sc = v2(seed, disaster, region_m, severity=0.7)
    assert len(sc.buildings) == sum(p["category"] == "house" for p in theirs)
    assert len(sc.vehicles) == sum(p["category"] == "car" for p in theirs)
    assert len(sc.debris) == sum(p["category"] in ("debris", "debris_pile")
                                 for p in theirs)
    xy = {(round(p["x_m"], 3), round(p["y_m"], 3)) for p in theirs
          if p["category"] == "house"}
    assert {b.center for b in sc.buildings} <= xy


# ---- validation over disasters x seeds x sizes -----------------------------
@pytest.mark.parametrize("region_m", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("disaster", DISASTERS)
def test_v2_validates(disaster, seed, region_m):
    sc = v2(seed, disaster, region_m)
    assert S.validate(sc) == []
    assert sc.meta.preset == f"v2:downtown:{disaster}"
    assert sc.meta.generator_version == "v2"
    assert sc.meta.locale == "downtown" and sc.meta.disaster_type == disaster
    w, h = region_m
    assert sc.region == (-w / 2, -h / 2, w / 2, h / 2)
    assert sc.buildings and sc.roads and sc.blocks and sc.humans
    assert sc.robots_spawn


def test_v2_deterministic():
    a = json.dumps(v2(3, "tornado", (600.0, 900.0)).to_dict())
    assert a == json.dumps(v2(3, "tornado", (600.0, 900.0)).to_dict())
    assert a != json.dumps(v2(4, "tornado", (600.0, 900.0)).to_dict())
    assert a != json.dumps(v2(3, "explosion", (600.0, 900.0)).to_dict())
    assert a != json.dumps(v2(3, "tornado", (900.0, 600.0)).to_dict())


def test_v1_is_untouched_by_the_pipeline_switch():
    """`pipeline="v1"` is the old path, byte for byte."""
    default = json.dumps(export_scene("earthquake", 2, region_m=(200.0, 200.0)).to_dict())
    explicit = json.dumps(export_scene("earthquake", 2, region_m=(200.0, 200.0),
                                       pipeline="v1").to_dict())
    assert default == explicit
    sc = S.Scene.from_dict(json.loads(default))
    assert sc.meta.generator_version != "v2"


def test_bad_pipeline_arguments():
    with pytest.raises(ValueError, match="pipeline"):
        export_scene("earthquake", 0, pipeline="v3")
    with pytest.raises(ValueError, match="disaster"):
        export_scene("downtown", 0, pipeline="v2")
    with pytest.raises(ValueError, match="locale"):
        export_scene("no_such_locale", 0, pipeline="v2", disaster="tornado")
    with pytest.raises(ValueError, match="high-level"):
        load_generator_config("low_level/default", 0, disaster="tornado")
    # a preset without the v2 blocks would silently build a v1 city
    with pytest.raises(ValueError, match="anisotropic"):
        export_scene("tornado", 0, pipeline="v2", disaster="tornado",
                     region_m=(300.0, 300.0))


# ---- what v2 adds over v1 --------------------------------------------------
def test_districts_produce_several_building_typologies(scenes):
    for (disaster, seed), sc in scenes.items():
        cats = {b.category for b in sc.buildings}
        assert len(cats) >= 2, (disaster, seed, cats)
        assert cats <= set(S.BUILDING_CATEGORIES)
        typ = {b.typology for b in sc.blocks}
        assert len(typ) >= 3 and typ <= set(S.BLOCK_TYPOLOGIES)


@pytest.mark.parametrize("region_m", BIG)
@pytest.mark.parametrize("seed", SEEDS)
def test_park_superblocks_on_large_regions(seed, region_m):
    sc = v2(seed, "earthquake", region_m)
    parks = [b for b in sc.blocks if b.typology == "park"]
    assert parks, (seed, region_m)
    # A park superblock keeps its sidewalk ring but not build_city's paving.
    for p in parks:
        interior = [r for r in sc.roads if r.kind == "sidewalk"
                    and r.rect[0] > p.rect[0] + 1.0 and r.rect[2] < p.rect[2] - 1.0
                    and r.rect[1] > p.rect[1] + 1.0 and r.rect[3] < p.rect[3] - 1.0]
        assert not interior, p.id
    assert any(r.kind == "trail" for r in sc.roads)


def test_block_typologies_come_from_the_districts_pass():
    cfg, _prov = load_generator_config("downtown", 5, region_m=(1000.0, 1000.0),
                                       severity=0.5, disaster="earthquake")
    resolver = _Resolver(1.0, cfg.get("fallback_sizes", {}), measure=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _pl, layout = build_v2(cfg, resolver)
    sc = v2(5, "earthquake", (1000.0, 1000.0))
    assert [b.rect for b in sc.blocks] == [tuple(round(v, 3) for v in blk)
                                           for blk in layout["blocks"]]
    typ_of = layout["_typology_of"]
    for blk, b in zip(layout["blocks"], sc.blocks):
        assert b.typology == BLOCK_TYPOLOGY.get(typ_of.get(tuple(blk), ""), "other")
    assert set(typ_of.values()) <= set(BLOCK_TYPOLOGY)


def test_buildings_take_their_block_typology():
    sc = v2(5, "earthquake", (1000.0, 1000.0))
    by_id = {b.id: b for b in sc.blocks}
    want = {v: k for k, v in BLOCK_TYPOLOGY.items()}
    for b in sc.buildings:
        blk = by_id.get(b.block_id or "")
        if blk is None or blk.typology == "park":
            continue
        typ = want.get(blk.typology)
        if typ in BUILDING_CATEGORY:
            assert b.category == BUILDING_CATEGORY[typ], (b.id, blk.typology)
    # No measured footprint means no per-asset height; the schema's category
    # default is used instead.
    assert all(b.height_m is None for b in sc.buildings)
    assert {b.resolved_height() for b in sc.buildings if b.fate == "intact"} \
        <= {S.DEFAULT_HEIGHT_M[c] for c in S.BUILDING_CATEGORIES}


def test_roads_carry_per_corridor_widths_and_classes(scenes):
    sc = scenes[("earthquake", 0)]
    roads = [r for r in sc.roads if r.kind == "road"]
    widths = {round(min(r.rect[2] - r.rect[0], r.rect[3] - r.rect[1]), 1)
              for r in roads}
    assert len(widths) >= 3, widths          # v1 has 2: main and secondary
    assert {r.n_lanes for r in roads} <= {1, 2, 3, 4, 5}
    assert len({r.n_lanes for r in roads}) >= 2
    assert max(widths) >= 4 * 3.3            # an arterial, 4 x 3.3 m lanes


def test_corridor_rects_lose_only_the_ceded_kerb_strips():
    """A corridor reserves a kerb strip either side whatever its parking policy;
    the strips the policy does not use are built out as pavement, so the road is
    genuinely narrower there."""
    cfg, _prov = load_generator_config("downtown", 2, region_m=(900.0, 700.0),
                                       severity=0.5, disaster="earthquake")
    resolver = _Resolver(1.0, cfg.get("fallback_sizes", {}), measure=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _pl, layout = build_v2(cfg, resolver)
    sc = v2(2, "earthquake", (900.0, 700.0))
    roads = {r.id: r for r in sc.roads if r.kind == "road"}
    lane_w = float(cfg["roads"]["lane_width_m"])
    seen = set()
    for i, c in enumerate(layout["road_corridors"]):
        r = roads.get(f"road{i}")
        if r is None:
            continue
        w = (r.rect[2] - r.rect[0]) if c["dir"] == "ns" else (r.rect[3] - r.rect[1])
        pw = float(c.get("park_w", 0.0))
        used = {"both": 2, "none": 0}.get(str(c.get("parking", "both")), 1)
        car = float(c["carriage"][1]) - float(c["carriage"][0])
        assert abs(w - (car + used * pw)) < 1e-6, (i, c)
        assert r.n_lanes == int(c["n_lanes"])
        seen.add(str(c.get("parking")))
        # An ordinary cut reserves n_lanes x lane_w of carriageway. A cut made
        # at a forced width (the border ring, a mews between terrace rows, a
        # terrace superblock's internal street) is all carriageway and reports
        # the lane count that width rounds to, so it is exempt.
        assert abs(car - c["n_lanes"] * lane_w) < 1e-6 or pw == 0.0
    assert len(seen) >= 2                      # more than one parking policy


def test_sidewalks_follow_the_v2_frontage_width():
    """The exported ring must be the inset `districts` actually held buildings
    off the block edge by — measured with the same (warm) resolver the pipeline
    used, since SizeResolver caches per (path, scale) and not per category, and
    the sidewalk and concrete tiles are the same USD."""
    cfg, _prov = load_generator_config("downtown", 4, region_m=(800.0, 800.0),
                                       severity=0.5, disaster="earthquake")
    resolver = _Resolver(1.0, cfg.get("fallback_sizes", {}), measure=False)
    with contextlib.redirect_stdout(io.StringIO()):
        placements, _layout = build_v2(cfg, resolver)
    verge = float((cfg.get("frontage") or {}).get("verge_m", 0.0))
    assert verge == 0.0                        # downtown: kerb to building line
    want = D.block_inset(cfg, resolver) - verge
    sc = v2(4, "earthquake", (800.0, 800.0))
    rings = [r for r in sc.roads if r.id.startswith("walk")]
    assert rings
    for r in rings:
        w = min(r.rect[2] - r.rect[0], r.rect[3] - r.rect[1])
        assert abs(w - want) < 0.05 or w < want   # clipped at the region edge
    # and it has to cover the tiles build_city laid.
    walks = [r.rect for r in sc.roads if r.kind == "sidewalk"]
    tiles = [p for p in placements if p["category"] == "sidewalk"]
    assert tiles
    assert all(any(q[0] <= p["x_m"] <= q[2] and q[1] <= p["y_m"] <= q[3]
                   for q in walks) for p in tiles)


def test_casualties_concentrate_in_the_damage(scenes):
    sc = scenes[("explosion", 0)]
    cas = [sc.damage_at(h.pos[0], h.pos[1]) for h in sc.casualties()]
    byst = [sc.damage_at(h.pos[0], h.pos[1]) for h in sc.humans
            if h.role == "bystander"]
    assert cas and byst
    assert sum(cas) / len(cas) > sum(byst) / len(byst)


# ---- sizes -----------------------------------------------------------------
def test_sample_region_is_varied_and_reproducible():
    sizes = [sample_region(s, 500.0, 1500.0) for s in range(60)]
    assert sizes == [sample_region(s, 500.0, 1500.0) for s in range(60)]
    for w, h in sizes:
        assert 500.0 <= w <= 1500.0 and 500.0 <= h <= 1500.0
        assert w % 10 == 0 and h % 10 == 0
    assert sum(w != h for w, h in sizes) >= 55          # non-square by default
    assert len({w for w, _h in sizes}) > 40 and len({h for _w, h in sizes}) > 40
    assert max(max(s) / min(s) for s in sizes) > 1.5    # aspect really varies
    with pytest.raises(ValueError):
        sample_region(0, 900.0, 500.0)


def test_sample_severity_range():
    vals = [sample_severity(s, "tornado", 0.5, 1.0) for s in range(40)]
    assert all(0.5 <= v <= 1.0 for v in vals) and len(set(vals)) > 30
    assert vals == [sample_severity(s, "tornado", 0.5, 1.0) for s in range(40)]
    assert sample_severity(3, "tornado") != sample_severity(3, "earthquake")
    with pytest.raises(ValueError):
        sample_severity(0, "tornado", 0.5, 1.5)


def test_extreme_aspect_ratio_still_builds():
    for region_m in ((500.0, 1500.0), (1500.0, 500.0)):
        sc = v2(11, "tornado", region_m)
        assert S.validate(sc) == []
        assert len(sc.blocks) > 10 and len(sc.buildings) > 20


@pytest.mark.parametrize("region_m", SIZES)
def test_time_and_size_budget(region_m):
    """< 15 s and < 15 MB of JSON for the largest region, at the worst severity."""
    t0 = time.perf_counter()
    sc = v2(7, "tornado", region_m, severity=1.0)
    dt = time.perf_counter() - t0
    js = json.dumps(sc.to_dict())
    assert dt < 15.0, f"{region_m}: {dt:.1f}s"
    assert len(js) < 15e6, f"{region_m}: {len(js) / 1e6:.1f} MB"
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2e6   # KiB
    g = sc.damage_field.grid
    assert g["cell_m"] == 2.0 and g["nx"] * g["cell_m"] >= region_m[0] - 1e-6
