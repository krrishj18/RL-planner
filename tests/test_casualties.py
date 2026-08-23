"""Casualty/bystander placement: schema-consistent, in-region, damage-seeking."""
import math
from collections import Counter

import numpy as np
import pytest

from rlplanner.scene import schema as S
from rlplanner.scene.casualties import (AUTO, CASUALTY_KINDS, HumanConfig,
                                        _draw_order, auto_casualties,
                                        place_humans)
from rlplanner.scene.export import export_scene

# (preset, severity) with a field that actually varies over the region: a uniform
# field (hurricane, or an earthquake severe enough to saturate the whole city)
# cannot separate casualties from bystanders by damage, so it proves nothing.
SHAPED = [("earthquake", 0.5), ("tornado", 1.0), ("explosion", 0.8)]


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def scene():
    sc = export_scene("earthquake", 0, region_m=(200.0, 200.0))
    sc.humans = []
    return sc


def test_counts_roles_and_ids(scene):
    hs = place_humans(scene, rng(), HumanConfig(n_casualties=15, n_bystanders=8))
    assert len(hs) == 23
    assert sum(h.role == "casualty" for h in hs) == 15
    assert sum(h.role == "bystander" for h in hs) == 8
    assert len({h.id for h in hs}) == 23


def test_deterministic_in_rng(scene):
    a = place_humans(scene, rng(5))
    b = place_humans(scene, rng(5))
    c = place_humans(scene, rng(6))
    assert a == b and a != c


def test_container_visibility_consistency(scene):
    for h in place_humans(scene, rng(1), HumanConfig(n_casualties=60, n_bystanders=40)):
        assert h.role in S.HUMAN_ROLES and h.pose in S.HUMAN_POSES
        assert h.container in S.HUMAN_CONTAINERS and h.visibility in S.HUMAN_VISIBILITY
        assert h.context in S.HUMAN_CONTEXTS
        assert not (h.container == "open" and h.visibility == "occluded")
        assert not (h.container in ("rubble", "building") and h.visibility == "open")
        if h.role == "bystander":
            assert h.container == "open" and h.pose in ("standing", "walking")
        else:
            assert h.pose == "prone"


def test_all_inside_region(scene):
    x0, y0, x1, y1 = scene.region
    for h in place_humans(scene, rng(2), HumanConfig(n_casualties=80, n_bystanders=40)):
        assert x0 <= h.pos[0] <= x1 and y0 <= h.pos[1] <= y1 and h.pos[2] >= 0.0


def test_container_ids_resolve(scene):
    known = ({b.id for b in scene.buildings} | {v.id for v in scene.vehicles}
             | {p.id for p in scene.props} | {d.id for d in scene.debris})
    for h in place_humans(scene, rng(3), HumanConfig(n_casualties=60)):
        if h.container_id is not None:
            assert h.container_id in known
        if h.container == "rubble":
            assert h.container_id in {b.id for b in scene.buildings if b.fate == "destroyed"}
        if h.container == "vehicle":
            assert h.container_id in {v.id for v in scene.vehicles if v.state == "toppled"}


def test_casualties_land_inside_their_container(scene):
    by_id = {b.id: b for b in scene.buildings}
    for h in place_humans(scene, rng(4), HumanConfig(n_casualties=60)):
        if h.container in ("rubble", "building"):
            b = by_id[h.container_id]
            d = np.hypot(h.pos[0] - b.center[0], h.pos[1] - b.center[1])
            assert d <= max(b.size) * 0.75 + 1e-6


@pytest.mark.parametrize("preset,severity", SHAPED)
def test_casualties_concentrate_in_damage(preset, severity):
    cas, bys = [], []
    for seed in range(3):
        sc = export_scene(preset, seed, region_m=(300.0, 300.0), severity=severity,
                          human_cfg=HumanConfig(n_casualties=40, n_bystanders=40))
        cas += [sc.damage_at(*h.pos[:2]) for h in sc.humans if h.role == "casualty"]
        bys += [sc.damage_at(*h.pos[:2]) for h in sc.humans if h.role == "bystander"]
    assert np.mean(cas) > np.mean(bys)


def test_open_tail_places_casualties_outside_the_zone():
    """The 10% tail must reach sidewalks the damage field never touches."""
    sc = export_scene("tornado", 0, region_m=(400.0, 400.0),
                      human_cfg=HumanConfig(n_casualties=120, n_bystanders=0))
    dmg = [sc.damage_at(*h.pos[:2]) for h in sc.casualties()]
    assert min(dmg) < 0.05 and max(dmg) > 0.9


def test_weights_select_the_kind(scene):
    only = {k: 0.0 for k in CASUALTY_KINDS} | {"destroyed": 1.0}
    hs = place_humans(scene, rng(7), HumanConfig(n_casualties=25, n_bystanders=0,
                                                 weights=only, open_tail_frac=0.0))
    assert {h.container for h in hs} == {"rubble"}

    only = {k: 0.0 for k in CASUALTY_KINDS} | {"pedestrian": 1.0}
    hs = place_humans(scene, rng(7), HumanConfig(n_casualties=25, n_bystanders=0,
                                                 weights=only, open_tail_frac=0.0))
    assert {h.context for h in hs} == {"sidewalk"}


def test_zero_counts(scene):
    assert place_humans(scene, rng(), HumanConfig(n_casualties=0, n_bystanders=0)) == []
    with pytest.raises(ValueError):
        place_humans(scene, rng(), HumanConfig(n_casualties=-1))


def test_empty_scene_still_places_people():
    """No buildings, no vehicles, no roads: everyone ends up in the open."""
    sc = S.Scene(meta=S.Meta(region=(0.0, 0.0, 50.0, 50.0)),
                 damage_field=S.DamageField(kind="uniform", params={"inside": 0.5}))
    hs = place_humans(sc, rng(), HumanConfig(n_casualties=5, n_bystanders=5))
    assert len(hs) == 10
    assert {h.container for h in hs} == {"open"}
    assert {h.context for h in hs} == {"other"}
    sc.humans = hs
    assert S.validate(sc) == []


def test_no_destroyed_buildings_falls_back():
    sc = export_scene("earthquake", 0, severity=0.0, region_m=(200.0, 200.0))
    assert not [b for b in sc.buildings if b.fate != "intact"]
    assert len(sc.casualties()) == HumanConfig.n_casualties
    assert {h.container for h in sc.casualties()} == {"open"}


def test_synthetic_scene_supported():
    sc = S.make_synthetic_scene(2)
    hs = place_humans(sc, rng(), HumanConfig(n_casualties=20, n_bystanders=10))
    sc.humans = hs
    assert S.validate(sc) == []
    assert {h.container for h in hs} >= {"rubble", "open"}


# ---- placement invariants (regressions) ------------------------------------
def obb(cx, cy, sx, sy, yaw_deg, px, py, pad=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = px - cx, py - cy
    return (abs(dx * c + dy * s) <= sx / 2 + pad
            and abs(-dx * s + dy * c) <= sy / 2 + pad)


def min_pair_dist(humans):
    p = np.array([[h.pos[0], h.pos[1]] for h in humans])
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return float(d.min())


ALL_PRESETS = ["earthquake", "explosion", "flood", "hurricane",
               "suburb_earthquake", "tornado"]


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_casualty_lies_on_the_car_it_is_trapped_at(preset):
    """A square world-axis jitter of 1.2 m overhangs a 4.5 x 2.0 m car rotated
    90 deg: the casualty ended up beside the vehicle it claimed to be inside."""
    sc = export_scene(preset, 0, region_m=(300.0, 300.0),
                      human_cfg=HumanConfig(n_casualties=120, n_bystanders=0))
    by_id = {v.id: v for v in sc.vehicles}
    n = 0
    for h in sc.humans:
        if h.container != "vehicle":
            continue
        v = by_id[h.container_id]
        assert v.state == "toppled"
        assert obb(*v.center, *v.size, v.yaw_deg, h.pos[0], h.pos[1], pad=1e-3), h
        n += 1
    assert n > 0


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_bus_stop_casualty_stays_within_the_jitter_radius(preset):
    cfg = HumanConfig(n_casualties=120, n_bystanders=0)
    sc = export_scene(preset, 1, region_m=(300.0, 300.0), human_cfg=cfg)
    stops = {p.id: p for p in sc.props if p.category == "bus_stop"}
    for h in sc.humans:
        if h.context != "bus_stop":
            continue
        q = stops[h.container_id]
        d = np.hypot(h.pos[0] - q.center[0], h.pos[1] - q.center[1])
        assert d <= cfg.bus_stop_jitter_m + 1e-3 <= 3.0, (h, d)


@pytest.mark.parametrize("preset", ALL_PRESETS)
@pytest.mark.parametrize("seed", [0, 17])
def test_nobody_stands_inside_a_standing_building(preset, seed):
    """Downtown paves the whole block interior, and the exporter calls that
    pavement a sidewalk — so the walk pool overlaps every building on the block."""
    sc = export_scene(preset, seed, region_m=(300.0, 300.0),
                      human_cfg=HumanConfig(n_casualties=60, n_bystanders=60))
    solid = [b for b in sc.buildings if b.fate != "destroyed"]
    for h in sc.humans:
        if h.container_id is not None:
            continue
        assert not any(obb(*b.center, *b.size, b.yaw_deg, h.pos[0], h.pos[1])
                       for b in solid), h


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_humans_keep_half_a_metre_apart(preset):
    sc = export_scene(preset, 9999, region_m=(200.0, 200.0),
                      human_cfg=HumanConfig(n_casualties=200, n_bystanders=50))
    assert len(sc.humans) == 250
    assert min_pair_dist(sc.humans) >= HumanConfig.min_separation_m


def test_crowding_degrades_instead_of_hanging():
    """200 casualties forced into one 2 x 2 m ruin: separation is impossible, so
    it must give up on it rather than resample forever."""
    sc = S.Scene(meta=S.Meta(region=(0.0, 0.0, 50.0, 50.0)),
                 damage_field=S.DamageField(kind="uniform", params={"inside": 1.0}))
    sc.buildings = [S.Building(id="b0", center=(25.0, 25.0), size=(2.0, 2.0),
                               fate="destroyed")]
    only = {k: 0.0 for k in CASUALTY_KINDS} | {"destroyed": 1.0}
    hs = place_humans(sc, rng(), HumanConfig(n_casualties=200, n_bystanders=0,
                                             weights=only, open_tail_frac=0.0))
    assert len(hs) == 200 and {h.container for h in hs} == {"rubble"}
    sc.humans = hs
    assert S.validate(sc) == []


def test_region_bounds_with_sub_millimetre_edges():
    """`_mk` clamped and then rounded, so a human on a border whose coordinate
    has more than 3 decimals rounded back out of the region."""
    x1 = 40.58465
    sc = S.Scene(meta=S.Meta(region=(-x1, -x1, x1, x1)),
                 damage_field=S.DamageField(kind="uniform", params={"inside": 1.0}))
    sc.roads = [S.Road(id="r0", rect=(-x1, -x1, x1, x1), kind="sidewalk")]
    sc.humans = place_humans(sc, rng(4), HumanConfig(n_casualties=200, n_bystanders=200))
    assert S.validate(sc) == []
    for h in sc.humans:
        assert -x1 <= h.pos[0] <= x1 and -x1 <= h.pos[1] <= x1


def test_container_holds_when_the_building_overhangs_the_region():
    """Clamping the sampled point into the region used to drag the casualty off
    the very footprint its container_id names."""
    sc = S.Scene(meta=S.Meta(region=(0.0, 0.0, 40.0, 40.0)),
                 damage_field=S.DamageField(kind="uniform", params={"inside": 1.0}))
    sc.buildings = [S.Building(id="edge", center=(1.0, 20.0), size=(20.0, 12.0),
                               fate="destroyed"),
                    S.Building(id="mid", center=(25.0, 20.0), size=(10.0, 8.0),
                               fate="damaged")]
    hs = place_humans(sc, rng(2), HumanConfig(n_casualties=120, n_bystanders=0,
                                              open_tail_frac=0.0))
    by_id = {b.id: b for b in sc.buildings}
    hit = 0
    for h in hs:
        if h.container not in ("rubble", "building"):
            continue
        b = by_id[h.container_id]
        assert obb(*b.center, *b.size, b.yaw_deg, h.pos[0], h.pos[1], pad=1e-3), h
        hit += 1
    assert hit > 0
    sc.humans = hs
    assert S.validate(sc) == []


@pytest.mark.parametrize("preset", ["tornado", "explosion"])
@pytest.mark.parametrize("seed", [0, 1, 17, 9999])
def test_casualties_sit_in_more_damage_than_bystanders(preset, seed):
    """Only meaningful where the field varies over the region: hurricane is
    uniform and a saturated earthquake leaves nowhere cooler to stand, so the
    two means differ only by sampling noise there."""
    sc = export_scene(preset, seed, region_m=(300.0, 300.0),
                      human_cfg=HumanConfig(n_casualties=60, n_bystanders=60))
    cas = np.array([sc.damage_at(*h.pos[:2]) for h in sc.humans if h.role == "casualty"])
    bys = np.array([sc.damage_at(*h.pos[:2]) for h in sc.humans if h.role == "bystander"])
    assert cas.mean() - bys.mean() > 0.2


def test_fuzz_place_humans_on_random_small_scenes():
    """200 randomised synthetic scenes; every invariant the module documents."""
    for it in range(200):
        r = np.random.default_rng(it)
        # block_m / road_w combinations that make_synthetic_scene itself rejects
        # (it can push a building centre out of the region) are skipped.
        try:
            sc = S.make_synthetic_scene(
                seed=int(r.integers(10_000)),
                region_m=(float(r.uniform(40, 300)), float(r.uniform(40, 300))),
                block_m=float(r.uniform(30, 90)), road_w=float(r.uniform(4, 12)),
                sidewalk_w=float(r.uniform(1, 4)), n_casualties=0, n_bystanders=0,
                severity=float(r.uniform(0, 1)),
                radius_m=float(r.uniform(5, 200)), falloff_m=float(r.uniform(5, 200)))
        except S.SchemaError:
            continue
        sc.humans = []
        cfg = HumanConfig(n_casualties=int(r.integers(0, 60)),
                          n_bystanders=int(r.integers(0, 30)),
                          open_tail_frac=float(r.uniform(0, 0.5)),
                          bystander_damage_bias=float(r.uniform(0, 1)),
                          footprint_frac=float(r.uniform(0.1, 1.0)),
                          vehicle_jitter_m=float(r.uniform(0, 3)),
                          bus_stop_jitter_m=float(r.uniform(0, 4)),
                          max_tries=int(r.integers(1, 100)))
        hs = place_humans(sc, np.random.default_rng(it), cfg)
        assert len(hs) == cfg.n_casualties + cfg.n_bystanders
        sc.humans = hs
        assert S.validate(sc) == [], (it, S.validate(sc)[:3])
        bmap = {b.id: b for b in sc.buildings}
        vmap = {v.id: v for v in sc.vehicles}
        pmap = {p.id: p for p in sc.props}
        for h in hs:
            if h.container in ("rubble", "building"):
                b = bmap[h.container_id]
                assert obb(*b.center, *b.size, b.yaw_deg, h.pos[0], h.pos[1], 1e-3), (it, h)
            elif h.container == "vehicle":
                v = vmap[h.container_id]
                assert obb(*v.center, *v.size, v.yaw_deg, h.pos[0], h.pos[1], 1e-3), (it, h)
            elif h.context == "bus_stop" and h.container_id:
                q = pmap[h.container_id]
                d = np.hypot(h.pos[0] - q.center[0], h.pos[1] - q.center[1])
                assert d <= cfg.bus_stop_jitter_m + 1e-3, (it, h)
            if h.container_id is None:
                assert not any(
                    obb(*b.center, *b.size, b.yaw_deg, h.pos[0], h.pos[1])
                    for b in sc.buildings if b.fate != "destroyed"), (it, h)


# ---- area-scaled counts and weight fall-through ----------------------------
def test_auto_counts_scale_with_the_region_area():
    """15 per 400 x 400 m (v1's density), clipped to [10, 80]; auto bystanders
    are half the resolved casualty count."""
    cfg = HumanConfig(n_casualties=AUTO, n_bystanders=AUTO)

    def region(w, h):
        return (-w / 2, -h / 2, w / 2, h / 2)

    assert cfg.counts(region(400.0, 400.0)) == (15, 8)
    assert cfg.counts(region(500.0, 500.0)) == (23, 12)
    assert cfg.counts(region(1010.0, 730.0)) == (69, 35)
    assert cfg.counts(region(1500.0, 1500.0)) == (80, 40)      # clipped high
    assert cfg.counts(region(50.0, 50.0)) == (10, 5)           # clipped low
    assert auto_casualties(0.16) == 15
    # explicit counts survive, and mix with an auto one
    assert HumanConfig(n_casualties=7, n_bystanders=3).counts(region(900.0, 900.0)) == (7, 3)
    assert HumanConfig(n_casualties=30, n_bystanders=AUTO).counts(region(400.0, 400.0)) == (30, 15)
    assert HumanConfig(n_casualties=AUTO, n_bystanders=0).counts(region(400.0, 400.0)) == (15, 0)


@pytest.mark.parametrize("bad", ["fifteen", 1.5, -2])
def test_bad_counts_are_refused_by_name(scene, bad):
    with pytest.raises(ValueError, match="n_casualties"):
        place_humans(scene, rng(), HumanConfig(n_casualties=bad))


def test_auto_counts_reach_place_humans():
    sc = S.make_synthetic_scene(1, region_m=(600.0, 400.0))
    sc.humans = place_humans(sc, rng(), HumanConfig(n_casualties=AUTO, n_bystanders=AUTO))
    assert sum(h.role == "casualty" for h in sc.humans) == 22   # 0.24 km2
    assert sum(h.role == "bystander" for h in sc.humans) == 11
    assert S.validate(sc) == []


def test_a_missing_kind_spreads_its_weight_over_the_rest():
    """No bus stops (the v2 downtown case): the 0.10 that class carries must be
    shared by the survivors in proportion, not handed to the heaviest one."""
    cfg = HumanConfig()
    avail = [k for k in CASUALTY_KINDS if k != "bus_stop"]
    r = rng(11)
    drawn = Counter(_draw_order(r, cfg, avail)[0] for _ in range(20000))
    total = sum(cfg.weight(k) for k in avail)
    for k in avail:
        assert abs(drawn[k] / 20000 - cfg.weight(k) / total) < 0.02, k
    # every kind stays a fallback for the others, in descending weight
    assert set(_draw_order(r, cfg, avail)) == set(avail)
    assert _draw_order(r, cfg, []) == []


def test_the_realised_mix_follows_the_renormalised_weights():
    """End to end on the v2 downtown scenes the dataset ships, where no bus stop
    is ever placed: the missing 0.10 must spread over the other four kinds, not
    pile onto rubble, and casualties must still sit in more damage than
    bystanders."""
    cfg = HumanConfig(n_casualties=60, n_bystanders=60, open_tail_frac=0.0)
    got, cas, bys = Counter(), [], []
    for seed in range(6):
        # explosion, not earthquake: a severe earthquake saturates the whole
        # region and no placement rule can separate the two roles by damage.
        sc = export_scene("downtown", seed, pipeline="v2", disaster="explosion",
                          region_m=(900.0, 900.0), severity=0.8, size_jitter=0.25)
        assert not [q for q in sc.props if q.category == "bus_stop"]
        sc.humans = place_humans(sc, rng(seed), cfg)
        got.update(h.container for h in sc.humans if h.role == "casualty")
        cas += [sc.damage_at(*h.pos[:2]) for h in sc.humans if h.role == "casualty"]
        bys += [sc.damage_at(*h.pos[:2]) for h in sc.humans if h.role == "bystander"]
    n = sum(got.values())
    total = sum(cfg.weight(k) for k in CASUALTY_KINDS if k != "bus_stop")
    for container, kind in (("rubble", "destroyed"), ("building", "damaged"),
                            ("vehicle", "vehicle"), ("open", "pedestrian")):
        assert abs(got[container] / n - cfg.weight(kind) / total) < 0.06, container
    assert np.mean(cas) - np.mean(bys) > 0.3


def test_a_car_under_a_roof_yields_no_casualty():
    """Downtown parks cars on strips a building overhangs; the trapped casualty
    used to be placed inside that building."""
    sc = S.Scene(meta=S.Meta(region=(0.0, 0.0, 100.0, 100.0)),
                 damage_field=S.DamageField(kind="uniform", params={"inside": 1.0}))
    sc.buildings = [S.Building(id="roof", center=(25.0, 25.0), size=(30.0, 30.0),
                               fate="intact")]
    sc.vehicles = [S.Vehicle(id="hidden", center=(25.0, 25.0), state="toppled"),
                   S.Vehicle(id="street", center=(75.0, 75.0), state="toppled")]
    sc.roads = [S.Road(id="r0", rect=(60.0, 60.0, 90.0, 90.0), kind="sidewalk")]
    hs = place_humans(sc, rng(3), HumanConfig(n_casualties=60, n_bystanders=0,
                                              open_tail_frac=0.0))
    at_car = [h for h in hs if h.container == "vehicle"]
    assert at_car and {h.container_id for h in at_car} == {"street"}
    for h in hs:
        assert not obb(25.0, 25.0, 30.0, 30.0, 0.0, h.pos[0], h.pos[1]), h


def test_a_ruin_under_a_neighbours_roof_yields_no_casualty():
    sc = S.Scene(meta=S.Meta(region=(0.0, 0.0, 100.0, 100.0)),
                 damage_field=S.DamageField(kind="uniform", params={"inside": 1.0}))
    sc.buildings = [S.Building(id="roof", center=(25.0, 25.0), size=(40.0, 40.0),
                               fate="intact"),
                    S.Building(id="swallowed", center=(25.0, 25.0), size=(8.0, 8.0),
                               fate="destroyed"),
                    S.Building(id="ruin", center=(75.0, 75.0), size=(10.0, 10.0),
                               fate="destroyed")]
    only = {k: 0.0 for k in CASUALTY_KINDS} | {"destroyed": 1.0}
    hs = place_humans(sc, rng(5), HumanConfig(n_casualties=40, n_bystanders=0,
                                              weights=only, open_tail_frac=0.0))
    assert {h.container_id for h in hs if h.container == "rubble"} == {"ruin"}
    for h in hs:
        assert not obb(25.0, 25.0, 40.0, 40.0, 0.0, h.pos[0], h.pos[1]), h


def test_a_degenerate_region_is_refused(scene):
    sc = S.Scene(meta=S.Meta(region=(0.0, 0.0, 0.0, 50.0)))
    with pytest.raises(ValueError, match="positive extent"):
        place_humans(sc, rng(), HumanConfig(n_casualties=3))
