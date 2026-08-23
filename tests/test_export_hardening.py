"""Adversarial regressions for the 2.5D exporter: every preset x seed, degenerate
regions and cell sizes, spawn sanity, park classification, and CLI argument
handling. Each test here pins a bug that was live at some point.
"""
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from rlplanner.scene import schema as S
from rlplanner.scene.casualties import HumanConfig
from rlplanner.scene.export import (PROP_SPEC, _Resolver, export_scene,
                                    load_generator_config)
from rlplanner.scene.gen import scene_generator as SG

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_scenes  # noqa: E402

PRESETS = ["earthquake", "explosion", "flood", "hurricane", "suburb_earthquake", "tornado"]
SEEDS = [0, 1, 17, 9999]


def obb(cx, cy, sx, sy, yaw_deg, px, py, pad=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = px - cx, py - cy
    return (abs(dx * c + dy * s) <= sx / 2 + pad
            and abs(-dx * s + dy * c) <= sy / 2 + pad)


# ---- the whole preset x seed matrix ----------------------------------------
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("seed", SEEDS)
def test_every_preset_and_seed_is_geometrically_sound(preset, seed):
    sc = export_scene(preset, seed, region_m=(200.0, 200.0))
    assert S.validate(sc) == []
    x0, y0, x1, y1 = sc.region
    for r in sc.roads:
        assert x0 <= r.rect[0] < r.rect[2] <= x1 and y0 <= r.rect[1] < r.rect[3] <= y1
    for b in sc.buildings:
        assert x0 <= b.center[0] <= x1 and y0 <= b.center[1] <= y1
    ruins = {b.id for b in sc.buildings if b.fate == "destroyed"}
    assert all(d.building_id in ruins for d in sc.debris if d.building_id)


@pytest.mark.parametrize("preset", PRESETS)
def test_severity_zero_leaves_an_intact_city_with_casualties(preset):
    sc = export_scene(preset, 17, region_m=(200.0, 200.0), severity=0.0)
    assert {b.fate for b in sc.buildings} <= {"intact"}
    assert sc.debris == []
    assert all(v.state == "intact" for v in sc.vehicles)
    assert all(p.state == "upright" for p in sc.props)
    # nobody to trap anywhere, so the open tail has to carry every casualty
    assert len(sc.casualties()) == HumanConfig.n_casualties
    assert {h.container for h in sc.casualties()} == {"open"}
    assert S.validate(sc) == []


# ---- damage grid -----------------------------------------------------------
@pytest.mark.parametrize("cell_m", [1.0, 1.5, 3.0, 5.0])
def test_damage_grid_covers_the_whole_region(cell_m):
    """`round` used to size the grid, so a cell that does not divide the extent
    left the last strip of the region unsampled."""
    sc = export_scene("tornado", 0, region_m=(200.0, 200.0), cell_m=cell_m)
    g = sc.damage_field.grid
    x0, y0, x1, y1 = sc.region
    assert g["nx"] * cell_m >= (x1 - x0) - 1e-9
    assert g["ny"] * cell_m >= (y1 - y0) - 1e-9
    assert len(g["values"]) == g["ny"] and all(len(r) == g["nx"] for r in g["values"])


@pytest.mark.parametrize("cell_m", [1.0, 3.0, 61.0, 200.0])
def test_damage_grid_matches_the_field_and_never_samples_outside(cell_m):
    sc = export_scene("earthquake", 3, region_m=(60.0, 60.0), cell_m=cell_m)
    g = sc.damage_field.grid
    x0, y0, x1, y1 = sc.region
    for j in range(g["ny"]):
        for i in range(g["nx"]):
            x = min(x0 + (i + 0.5) * cell_m, x1)
            y = min(y0 + (j + 0.5) * cell_m, y1)
            assert x0 <= x <= x1 and y0 <= y <= y1
            assert abs(g["values"][j][i] - sc.damage_at(x, y)) < 1e-3
    assert S.validate(sc) == []


# ---- spawns ----------------------------------------------------------------
@pytest.mark.parametrize("side", [5.0, 10.0, 20.0, 60.0])
@pytest.mark.parametrize("corner", ["ll", "ur"])
def test_spawns_are_distinct_even_on_a_tiny_region(side, corner):
    """A short corner corridor used to clamp all eight spawns onto its far end."""
    sc = export_scene("earthquake", 0, region_m=(side, side), spawn_corner=corner)
    assert len(sc.robots_spawn) == 8
    assert len({(x, y) for x, y, _ in sc.robots_spawn}) == 8
    roads = [r for r in sc.roads if r.kind == "road"]
    for x, y, _z in sc.robots_spawn:
        assert any(r.rect[0] <= x <= r.rect[2] and r.rect[1] <= y <= r.rect[3]
                   for r in roads)


@pytest.mark.parametrize("preset", ["earthquake", "tornado"])
@pytest.mark.parametrize("corner", ["ll", "lr", "ul", "ur"])
def test_spawns_are_never_inside_a_building(preset, corner):
    sc = export_scene(preset, 17, region_m=(200.0, 200.0), spawn_corner=corner)
    for x, y, _z in sc.robots_spawn:
        assert not any(obb(*b.center, *b.size, b.yaw_deg, x, y) for b in sc.buildings)


# ---- blocks ----------------------------------------------------------------
@pytest.mark.parametrize("preset,region", [("earthquake", (200.0, 200.0)),
                                           ("tornado", (800.0, 600.0))])
def test_park_blocks_match_the_generators_own_count(preset, region):
    """"No building in it" called every block that merely failed to pack a park
    (17 of 162 downtown). A park is skipped before paving, so it has no hard
    surface either."""
    cfg, _prov = load_generator_config(preset, 0, region_m=region)
    resolver = _Resolver(float(cfg["asset_scale"]), cfg["fallback_sizes"], measure=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SG.build_city(cfg, resolver)
    m = re.search(r"\((\d+) blocks, (\d+) parks", buf.getvalue())
    assert m, buf.getvalue()[-200:]
    n_blocks, n_parks = int(m.group(1)), int(m.group(2))
    sc = export_scene(preset, 0, region_m=region)
    typ = Counter(b.typology for b in sc.blocks)
    assert len(sc.blocks) == n_blocks
    assert typ["park"] == n_parks


# ---- inputs ----------------------------------------------------------------
@pytest.mark.parametrize("seed", [-1, -9999, 1.5, "0"])
def test_bad_seeds_are_rejected_with_a_readable_error(seed):
    """`np.random.default_rng([seed, ...])` died inside numpy on a negative seed."""
    with pytest.raises(ValueError, match="seed"):
        export_scene("earthquake", seed, region_m=(120.0, 120.0))


def test_seed_parsing_forms():
    assert export_scenes.parse_seeds("0:3") == [0, 1, 2]
    assert export_scenes.parse_seeds("0,2,9") == [0, 2, 9]
    assert export_scenes.parse_seeds("5") == [5]
    assert export_scenes.parse_seeds(" 5 ") == [5]


@pytest.mark.parametrize("text,msg", [("abc", "not an integer"), ("0:", "not an integer"),
                                      ("3:0", "empty range"), ("", "no seeds"),
                                      ("-1", "non-negative"), ("0,x", "not an integer")])
def test_bad_seed_strings_name_the_problem(text, msg):
    with pytest.raises(ValueError, match=msg):
        export_scenes.parse_seeds(text)


def test_out_dir_is_created_or_refused(tmp_path):
    d = export_scenes._out_dir(str(tmp_path / "a" / "b" / "c"))
    assert d.is_dir()
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        export_scenes._out_dir(str(f))


def test_cli_writes_and_overwrites(tmp_path):
    out = tmp_path / "scenes"
    argv = ["--preset", "earthquake", "--seeds", "0:2", "--region", "120", "120",
            "--out", str(out)]
    assert export_scenes.main(argv) == 0
    first = {p.name: p.read_bytes() for p in sorted(out.glob("*.json"))}
    assert set(first) == {"earthquake_0.json", "earthquake_1.json"}
    assert export_scenes.main(argv) == 0
    assert {p.name: p.read_bytes() for p in sorted(out.glob("*.json"))} == first
    # same path, different severity -> different content, no stale file left
    assert export_scenes.main(argv + ["--severity", "0.3"]) == 0
    again = {p.name: p.read_bytes() for p in sorted(out.glob("*.json"))}
    assert set(again) == set(first) and all(again[k] != first[k] for k in first)
    for k in again:
        assert S.validate(S.Scene.from_dict(json.loads(again[k]))) == []


# ---- determinism -----------------------------------------------------------
def test_json_is_byte_identical_across_processes():
    """Set/dict iteration order must not leak into the export."""
    code = ("import json,hashlib;"
            "from rlplanner.scene.export import export_scene;"
            "print(hashlib.sha256(json.dumps(export_scene('tornado',7,"
            "region_m=(160.,160.)).to_dict()).encode()).hexdigest())")
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout
            for h in ("0", "12345")]
    assert outs[0] == outs[1]


# ---- scale -----------------------------------------------------------------
def test_large_region_stays_fast_and_valid():
    t0 = time.perf_counter()
    sc = export_scene("tornado", 0, region_m=(800.0, 600.0), cell_m=1.0)
    assert time.perf_counter() - t0 < 20.0
    assert S.validate(sc) == []
    assert len(sc.buildings) > 100 and len(sc.roads) > 100


def test_no_generator_placement_category_is_dropped():
    """A category missing from PROP_SPEC would vanish from the scene silently."""
    handled = set(PROP_SPEC) | {"house", "car", "debris", "debris_pile", "human",
                                "concrete", "sidewalk", "brick", "trail"}
    for preset in ("earthquake", "tornado"):
        cfg, _prov = load_generator_config(preset, 0, region_m=(200.0, 200.0))
        resolver = _Resolver(float(cfg["asset_scale"]), cfg["fallback_sizes"],
                             measure=False)
        with contextlib.redirect_stdout(io.StringIO()):
            placements, _layout = SG.build_city(cfg, resolver)
        assert {p["category"] for p in placements} <= handled
