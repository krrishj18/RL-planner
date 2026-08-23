"""The vendored generator must run headless: no pxr, no Nucleus, fallback sizes only."""
import math
import random

import pytest

from rlplanner.scene import schema as S
from rlplanner.scene.export import available_presets, load_generator_config
from rlplanner.scene.gen import compile_disaster as CD
from rlplanner.scene.gen import scene_generator as SG


def build(preset="earthquake", seed=0, **kw):
    cfg, prov = load_generator_config(preset, seed, **kw)
    r = SG.SizeResolver(float(cfg.get("asset_scale", 1.0)), cfg.get("fallback_sizes", {}),
                        measure=False)
    return cfg, prov, *SG.build_city(cfg, r)


def test_presets_exist():
    assert {"earthquake", "tornado", "explosion", "flood",
            "hurricane"} <= set(available_presets())
    assert CD.is_high_level({"disaster-type": "earthquake"})


def test_build_city_runs_without_pxr(capsys):
    cfg, prov, placements, layout = build()
    capsys.readouterr()
    assert placements and {"usd", "x_m", "y_m", "category"} <= set(placements[0])
    assert set(layout) == {"region", "blocks", "road_corridors", "placeholder_buildings"}
    assert layout["blocks"] and layout["road_corridors"]
    cats = {p["category"] for p in placements}
    assert {"house", "car"} <= cats


def test_build_city_deterministic():
    _c, _p, a, la = build(seed=3)
    _c, _p, b, lb = build(seed=3)
    assert a == b and la == lb
    _c, _p, c, _lc = build(seed=4)
    assert a != c


def test_resolver_falls_back_without_measuring():
    cfg, _prov = load_generator_config("earthquake", 0)
    r = SG.SizeResolver(1.0, cfg["fallback_sizes"], measure=False)
    fp = r.get("omniverse://nowhere/None.usd", "car")
    assert (fp["sx"], fp["sy"]) == tuple(cfg["fallback_sizes"]["car"][:2])
    # unknown categories take the resolver's generic 4x4 m; the cache is keyed by
    # (path, scale, axis_up) only, so a second category needs a second path.
    assert r.get("omniverse://nowhere/Other.usd", "no_such_category")["sx"] == 4.0


def test_region_and_seed_overrides_reach_the_config():
    cfg, prov = load_generator_config("earthquake", 11, region_m=(200.0, 300.0))
    assert cfg["seed"] == 11 and cfg["layout"]["region_m"] == [200.0, 300.0]
    assert cfg["measure_usds"] is False
    assert prov["locale"] == "suburban" and prov["disaster_type"] == "earthquake"


def test_severity_zero_disables_the_disaster():
    cfg, _prov = load_generator_config("earthquake", 0, severity=0.0)
    dis = cfg["disaster"]
    assert dis["damaged_fraction"] == 0.0 and dis["destroyed_fraction"] == 0.0
    assert dis["field"]["inside"] == 0.0


@pytest.mark.parametrize("preset", ["earthquake", "tornado", "explosion", "flood"])
def test_schema_damage_field_matches_the_generator(preset):
    """schema.DamageField.value_at must mirror scene_gen.make_damage_field."""
    cfg, _prov = load_generator_config(preset, 5)
    spec = dict(cfg["disaster"]["field"])
    region = (-200.0, -200.0, 200.0, 200.0)
    gen = SG.make_damage_field(spec, region)
    kind = spec.pop("kind")
    if kind == "path" and "points" not in spec:
        spec.setdefault("heading_deg", 45.0)
    mine = S.DamageField(kind=kind, params=spec)
    rng = random.Random(0)
    for _ in range(200):
        x, y = rng.uniform(-200, 200), rng.uniform(-200, 200)
        assert math.isclose(mine.value_at(x, y, region), gen(x, y), abs_tol=1e-9)
