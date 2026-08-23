import json
import math

import pytest

from rlplanner.scene import schema as S


def test_synthetic_scene_valid_and_deterministic():
    a = S.make_synthetic_scene(7)
    b = S.make_synthetic_scene(7)
    assert S.validate(a) == []
    assert a.to_dict() == b.to_dict()
    assert S.make_synthetic_scene(8).to_dict() != a.to_dict()


def test_json_roundtrip(tmp_path):
    sc = S.make_synthetic_scene(1)
    p = tmp_path / "s.json"
    sc.to_json(p, indent=1)
    sc2 = S.Scene.from_json(p)
    assert sc2.to_dict() == sc.to_dict()
    # tuples survive as tuples after load
    assert isinstance(sc2.buildings[0].center, tuple)
    assert isinstance(sc2.meta.region, tuple)


def test_schema_version_mismatch_rejected():
    d = S.make_synthetic_scene(0).to_dict()
    d["schema_version"] = "9.9"
    with pytest.raises(S.SchemaError):
        S.Scene.from_dict(d)


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: s.buildings.__setitem__(0, S.Building(id=s.buildings[1].id, center=(0, 0), size=(5, 5))), "duplicate id"),
    (lambda s: setattr(s.buildings[0], "fate", "vaporised"), "fate"),
    (lambda s: setattr(s.buildings[0], "size", (0.0, 5.0)), "non-positive size"),
    (lambda s: setattr(s.humans[0], "pos", (1e6, 0.0, 0.0)), "outside region"),
    (lambda s: setattr(s.humans[0], "pos", (0.0, 0.0, -1.0)), "negative z"),
    (lambda s: (setattr(s.humans[0], "container", "open"), setattr(s.humans[0], "visibility", "occluded")), "open container but occluded"),
    (lambda s: (setattr(s.humans[0], "container", "rubble"), setattr(s.humans[0], "visibility", "open")), "visibility open"),
    (lambda s: setattr(s.humans[0], "container_id", "nope"), "unknown container_id"),
    (lambda s: setattr(s.meta, "severity", 1.5), "severity"),
    (lambda s: setattr(s.damage_field, "kind", "spiral"), "damage_field.kind"),
    (lambda s: s.robots_spawn.append((0.0, 0.0)), "robots_spawn"),
    (lambda s: setattr(s.roads[0], "rect", (0.0, 0.0, 0.0, 10.0)), "degenerate rect"),
])
def test_validate_catches(mutate, msg):
    sc = S.make_synthetic_scene(2)
    mutate(sc)
    errs = S.validate(sc)
    assert errs, "expected a validation error"
    assert any(msg in e for e in errs), errs


def test_validate_region_degenerate_short_circuits():
    sc = S.make_synthetic_scene(0)
    sc.meta.region = (0.0, 0.0, 0.0, 10.0)
    errs = S.validate(sc)
    assert len(errs) == 1 and "positive extent" in errs[0]


def test_damage_field_kinds():
    reg = (-100.0, -100.0, 100.0, 100.0)
    u = S.DamageField(kind="uniform", params={"inside": 0.3})
    assert u.value_at(0, 0, reg) == pytest.approx(0.3)
    r = S.DamageField(kind="radial", params={"center": [0, 0], "radius_m": 10, "falloff_m": 10, "inside": 1.0, "outside": 0.0})
    assert r.value_at(0, 0, reg) == 1.0
    assert r.value_at(10, 0, reg) == 1.0
    assert 0.0 < r.value_at(15, 0, reg) < 1.0
    assert r.value_at(25, 0, reg) == pytest.approx(0.0)
    p = S.DamageField(kind="path", params={"points": [[-100, 0], [100, 0]], "width_m": 20, "falloff_m": 10, "inside": 1.0, "outside": 0.0})
    assert p.value_at(0, 0, reg) == 1.0
    assert p.value_at(0, 9.9, reg) == 1.0
    assert p.value_at(0, 15, reg) == pytest.approx(0.5, abs=0.05)
    assert p.value_at(0, 30, reg) == 0.0
    pd = S.DamageField(kind="path", params={"heading_deg": 90.0, "width_m": 20, "falloff_m": 0.001})
    assert pd.value_at(0, 50, reg) == 1.0 and pd.value_at(50, 0, reg) == pytest.approx(0.0)
    with pytest.raises(S.SchemaError):
        S.DamageField(kind="spiral").value_at(0, 0, reg)


def test_damage_grid_shape_validation():
    sc = S.make_synthetic_scene(0)
    sc.damage_field.grid = {"cell_m": 2.0, "nx": 3, "ny": 2, "values": [[0.0, 0.5, 1.0], [0.0, 0.5]]}
    assert any("shape" in e for e in S.validate(sc))
    sc.damage_field.grid = {"cell_m": 2.0, "nx": 2, "ny": 1, "values": [[0.0, 1.5]]}
    assert any("outside [0,1]" in e for e in S.validate(sc))
    sc.damage_field.grid = {"cell_m": 2.0, "nx": 2, "ny": 1, "values": [[0.0, 1.0]]}
    assert S.validate(sc) == []


def test_resolved_heights():
    assert S.Building(id="a", center=(0, 0), size=(5, 5), category="house", fate="intact").resolved_height() == 7.0
    assert S.Building(id="a", center=(0, 0), size=(5, 5), category="house", fate="destroyed").resolved_height() == pytest.approx(7.0 * 0.25)
    assert S.Building(id="a", center=(0, 0), size=(5, 5), height_m=3.3).resolved_height() == 3.3
    assert S.Vehicle(id="v", center=(0, 0), state="toppled").resolved_height() >= 1.0
    assert S.Prop(id="p", category="streetlight", center=(0, 0), state="toppled").resolved_height() == 0.5
    assert S.Debris(id="d", center=(0, 0), radius_m=1, kind="piece").resolved_height() == S.DEFAULT_HEIGHT_M["debris_piece"]


def test_synthetic_casualties_follow_damage():
    # casualties should sit in higher-damage areas than bystanders, on average, across seeds
    ratios = []
    for seed in range(6):
        sc = S.make_synthetic_scene(seed)
        cas = [sc.damage_at(h.pos[0], h.pos[1]) for h in sc.humans if h.role == "casualty"]
        bys = [sc.damage_at(h.pos[0], h.pos[1]) for h in sc.humans if h.role == "bystander"]
        assert cas and bys
        ratios.append(sum(cas) / len(cas) - sum(bys) / len(bys))
    assert sum(ratios) / len(ratios) > 0.2


def test_synthetic_small_and_large_regions():
    tiny = S.make_synthetic_scene(0, region_m=(60.0, 60.0), n_casualties=2, n_bystanders=1)
    assert S.validate(tiny) == []
    big = S.make_synthetic_scene(0, region_m=(800.0, 600.0), n_casualties=40, n_bystanders=20)
    assert S.validate(big) == [] and len(big.buildings) > 200


def test_class_table_consistency():
    assert S.N_CLASSES == len(S.CLASS_NAMES) == len(set(S.CLASS_NAMES))
    assert S.CLASS_ID["ground"] == 0 and S.CLASS_ID["human_prone"] == S.N_CLASSES - 1


def test_synthetic_fuzz_parameters():
    import random
    rng = random.Random(3)
    for _ in range(150):
        W = rng.choice([60, 100, 200, 333, 500]); H = rng.choice([60, 90, 240, 410])
        bm = rng.choice([30, 60, 90, 120]); rw = rng.choice([6, 12]); sw = rng.choice([0.5, 2])
        sc = S.make_synthetic_scene(rng.randrange(1000), region_m=(W, H), block_m=bm, road_w=rw, sidewalk_w=sw,
                                    n_casualties=rng.choice([0, 1, 5, 30]), n_bystanders=rng.choice([0, 3]))
        assert S.validate(sc) == []
