import math
from dataclasses import replace

import numpy as np
import pytest

from rlplanner.scene.schema import Building, DamageField, Meta, Scene, make_synthetic_scene
from rlplanner.sim.config import SensorConfig
from rlplanner.sim.raster import rasterize
from rlplanner.sim.sensor import human_visibility, point_visibility, visible_cells

SENSOR = SensorConfig()


def _flat_scene(w=120.0, buildings=()):
    return Scene(meta=Meta(region=(-w / 2, -w / 2, w / 2, w / 2)),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
                 buildings=list(buildings))


def _cam(r, x, y, alt=25.0):
    return np.array([x, y, alt], np.float64)


def test_window_stays_in_bounds_at_every_corner():
    r = rasterize(make_synthetic_scene(0, region_m=(120.0, 120.0)), 2.0)
    x0, y0, x1, y1 = r.region
    for x in (x0 + 0.01, (x0 + x1) / 2, x1 - 0.01):
        for y in (y0 + 0.01, (y0 + y1) / 2, y1 - 0.01):
            for yaw in np.linspace(-math.pi, math.pi, 9):
                v = visible_cells(r, SENSOR, _cam(r, x, y), float(yaw))
                for ij in (v.observed_ij, v.far_ij):
                    if ij.size:
                        assert ij[:, 0].min() >= 0 and ij[:, 0].max() < r.ny
                        assert ij[:, 1].min() >= 0 and ij[:, 1].max() < r.nx


def test_depth_split_is_exact():
    r = rasterize(_flat_scene(), 2.0)
    cam = _cam(r, 0.0, 0.0)
    v = visible_cells(r, SENSOR, cam, 0.0)
    for ij in v.observed_ij:
        x, y = r.ij_to_xy(int(ij[0]), int(ij[1]))
        rr = math.dist((x, y, float(r.height[ij[0], ij[1]])), tuple(cam))
        assert rr <= SENSOR.depth_limit_m + 1e-9
    assert (v.slant_r > SENSOR.depth_limit_m).all()
    assert (v.slant_r <= SENSOR.visual_range_m + 1e-9).all()
    assert v.far_ij.shape[0] == v.slant_r.shape[0]


def test_cone_is_a_subset_of_disk():
    r = rasterize(_flat_scene(), 2.0)
    cam = _cam(r, 0.0, 0.0)
    cone = visible_cells(r, SENSOR, cam, 0.0)
    disk = visible_cells(r, replace(SENSOR, mode="disk"), cam, 0.0)
    a = {tuple(x) for x in cone.observed_ij}
    b = {tuple(x) for x in disk.observed_ij}
    assert a < b
    assert disk.far_ij.shape[0] > cone.far_ij.shape[0]


def test_yaw_rotates_the_footprint():
    r = rasterize(_flat_scene(), 2.0)
    cam = _cam(r, 0.0, 0.0)
    east = visible_cells(r, SENSOR, cam, 0.0).observed_ij
    north = visible_cells(r, SENSOR, cam, math.pi / 2).observed_ij
    assert east.shape[0] == pytest.approx(north.shape[0], rel=0.25)
    ex, _ = r.ij_to_xy(east[:, 0], east[:, 1])
    _, ny = r.ij_to_xy(north[:, 0], north[:, 1])
    assert ex.mean() > 5.0 and ny.mean() > 5.0


def test_building_casts_a_shadow():
    b = Building(id="b", center=(20.0, 0.0), size=(10.0, 40.0), category="highrise")
    r = rasterize(_flat_scene(buildings=[b]), 2.0)
    cam = _cam(r, -10.0, 0.0, 25.0)
    seen = {tuple(x) for x in visible_cells(r, SENSOR, cam, 0.0).far_ij}
    r2 = rasterize(_flat_scene(), 2.0)
    seen2 = {tuple(x) for x in visible_cells(r2, SENSOR, cam, 0.0).far_ij}
    behind = [ij for ij in seen2 if r2.ij_to_xy(*ij)[0] > 30.0 and abs(r2.ij_to_xy(*ij)[1]) < 5.0]
    assert behind and not any(ij in seen for ij in behind)


def test_far_thinning_is_a_subset_and_needs_rng():
    r = rasterize(_flat_scene(), 2.0)
    cam = _cam(r, 0.0, 0.0)
    full = {tuple(x) for x in visible_cells(r, SENSOR, cam, 0.3).far_ij}
    thin = visible_cells(r, SENSOR, cam, 0.3, far_p=0.3, rng=np.random.default_rng(0)).far_ij
    assert {tuple(x) for x in thin} <= full
    assert 0 < thin.shape[0] < len(full)
    with pytest.raises(ValueError):
        visible_cells(r, SENSOR, cam, 0.0, far_p=0.3)


def test_depth_beyond_visual_range_rejected():
    r = rasterize(_flat_scene(), 2.0)
    bad = replace(SENSOR, depth_limit_m=100.0, visual_range_m=80.0)
    with pytest.raises(ValueError):
        visible_cells(r, bad, _cam(r, 0.0, 0.0), 0.0)


def test_tiny_region():
    r = rasterize(_flat_scene(20.0), 2.0)
    v = visible_cells(r, SENSOR, _cam(r, 0.0, 0.0), 0.0)
    assert v.observed_ij.shape[0] >= 1
    assert v.observed_ij[:, 0].max() < r.ny and v.observed_ij[:, 1].max() < r.nx


def test_point_and_human_visibility_agree():
    b = Building(id="b", center=(0.0, 20.0), size=(20.0, 6.0), category="highrise")
    r = rasterize(_flat_scene(buildings=[b]), 2.0)
    cam = _cam(r, 0.0, 0.0)
    pts = np.array([[20.0, 0.0, 0.5], [0.0, 40.0, 0.5], [-20.0, 0.0, 0.5]])
    iv, rr = point_visibility(r, SENSOR, cam, 0.0, pts)
    iv2, rr2 = human_visibility(r, SENSOR, cam, 0.0, pts)
    assert list(iv) == list(iv2)
    assert rr == pytest.approx(rr2)
    assert iv[0] and not iv[1] and not iv[2]      # ahead / occluded / behind
