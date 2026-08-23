import math

import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.scene.schema import (Building, DamageField, Debris, Human, Meta, Prop, Road, Scene,
                                    Vehicle, make_synthetic_scene)
from rlplanner.sim.raster import CLASS_PRIORITY, rasterize

CID = schema.CLASS_ID


def _scene(**kw) -> Scene:
    m = Meta(region=kw.pop("region", (0.0, 0.0, 20.0, 20.0)))
    return Scene(meta=m, damage_field=DamageField(kind="uniform", params={"inside": 0.0}), **kw)


def test_grid_shape_and_xy_roundtrip():
    r = rasterize(_scene(region=(-10.0, -6.0, 10.0, 6.0)), 2.0)
    assert (r.ny, r.nx) == (6, 10)
    assert r.origin == (-10.0, -6.0)
    assert r.ij_to_xy(0, 0) == (-9.0, -5.0)
    assert r.xy_to_ij(-9.0, -5.0) == (0, 0)
    assert r.xy_to_ij(9.9, 5.9) == (5, 9)
    assert r.in_bounds(0, 0) and not r.in_bounds(6, 0) and not r.in_bounds(0, -1)
    i, j = r.xy_to_ij(np.array([-9.0, 9.0]), np.array([-5.0, 5.0]))
    assert list(i) == [0, 5] and list(j) == [0, 9]


@pytest.mark.parametrize("cell", [1.0, 2.0, 5.0])
def test_cell_sizes(cell):
    r = rasterize(make_synthetic_scene(0, region_m=(60.0, 60.0)), cell)
    assert r.nx == r.ny == int(60 / cell)
    assert r.height.shape == r.cls.shape == r.damage.shape == (r.ny, r.nx)


def test_class_priority_order():
    names = ["vehicle_intact", "debris", "building_intact", "bus_stop", "street_furniture",
             "tree", "sidewalk", "road", "park", "ground"]
    p = [CLASS_PRIORITY[CID[n]] for n in names]
    assert p == sorted(p, reverse=True)
    assert CLASS_PRIORITY[CID["vehicle_toppled"]] == CLASS_PRIORITY[CID["vehicle_intact"]]


def test_overlap_priority_and_height():
    sc = _scene(roads=[Road(id="r", rect=(0.0, 0.0, 20.0, 20.0))],
                vehicles=[Vehicle(id="v", center=(10.0, 10.0), size=(6.0, 6.0))])
    r = rasterize(sc, 2.0)
    i, j = r.xy_to_ij(10.0, 10.0)
    assert r.cls[i, j] == CID["vehicle_intact"]
    assert r.cls[0, 0] == CID["road"]
    assert r.height[i, j] == pytest.approx(1.5)
    assert r.height[0, 0] == 0.0
    assert r.object_at(i, j) == ("vehicle", "v")
    assert r.object_at(0, 0) == ("road", "r")


def test_building_height_by_fate():
    sc = _scene(region=(0.0, 0.0, 60.0, 60.0), buildings=[
        Building(id="a", center=(10.0, 10.0), size=(8.0, 8.0), fate="intact", category="house"),
        Building(id="b", center=(30.0, 10.0), size=(8.0, 8.0), fate="damaged", category="house"),
        Building(id="c", center=(50.0, 10.0), size=(8.0, 8.0), fate="destroyed", category="house")])
    r = rasterize(sc, 2.0)
    h = [float(r.height[r.xy_to_ij(x, 10.0)[0], r.xy_to_ij(x, 10.0)[1]]) for x in (10.0, 30.0, 50.0)]
    assert h == pytest.approx([7.0, 5.6, 1.75])
    c = [int(r.cls[r.xy_to_ij(x, 10.0)[0], r.xy_to_ij(x, 10.0)[1]]) for x in (10.0, 30.0, 50.0)]
    assert c == [CID["building_intact"], CID["building_damaged"], CID["building_destroyed"]]


def test_obb_rotation():
    """A 12 x 2 box at yaw 0 spans x; at yaw 90 it spans y."""
    a = rasterize(_scene(region=(0.0, 0.0, 40.0, 40.0), buildings=[
        Building(id="a", center=(20.0, 20.0), size=(12.0, 2.0), yaw_deg=0.0)]), 1.0)
    b = rasterize(_scene(region=(0.0, 0.0, 40.0, 40.0), buildings=[
        Building(id="a", center=(20.0, 20.0), size=(12.0, 2.0), yaw_deg=90.0)]), 1.0)
    ma = a.cls == CID["building_intact"]
    mb = b.cls == CID["building_intact"]
    assert ma.sum() == mb.sum() == 24
    ia, ja = np.nonzero(ma)
    ib, jb = np.nonzero(mb)
    assert ja.max() - ja.min() == 11 and ia.max() - ia.min() == 1
    assert jb.max() - jb.min() == 1 and ib.max() - ib.min() == 11
    assert np.array_equal(ma, mb.T)


def test_obb_45_degrees_is_diamond():
    r = rasterize(_scene(region=(0.0, 0.0, 40.0, 40.0), buildings=[
        Building(id="a", center=(20.0, 20.0), size=(10.0, 10.0), yaw_deg=45.0)]), 1.0)
    m = r.cls == CID["building_intact"]
    i, j = np.nonzero(m)
    # rotated square: same area, corners of the bounding box empty
    assert 88 <= m.sum() <= 115
    assert not m[i.min(), j.min()] and not m[i.max(), j.max()]


def test_disc_objects():
    r = rasterize(_scene(region=(0.0, 0.0, 40.0, 40.0),
                         debris=[Debris(id="d", center=(20.0, 20.0), radius_m=5.0)]), 1.0)
    m = r.cls == CID["debris"]
    assert abs(m.sum() - math.pi * 25) < 12
    i, j = r.xy_to_ij(20.0, 20.0)
    assert r.height[i, j] == pytest.approx(1.8)


def test_small_object_keeps_its_centre_cell():
    r = rasterize(_scene(vehicles=[Vehicle(id="v", center=(9.0, 9.0), size=(0.2, 0.2))]), 2.0)
    assert (r.cls == CID["vehicle_intact"]).sum() == 1


def test_props_map_to_classes():
    sc = _scene(region=(0.0, 0.0, 40.0, 40.0), props=[
        Prop(id="p0", category="bus_stop", center=(10.0, 10.0), size=(4.0, 4.0)),
        Prop(id="p1", category="tree", center=(20.0, 20.0), size=(6.0, 6.0)),
        Prop(id="p2", category="bench", center=(30.0, 30.0), size=(4.0, 4.0))])
    r = rasterize(sc, 1.0)
    assert r.cls[r.xy_to_ij(10.0, 10.0)[0], r.xy_to_ij(10.0, 10.0)[1]] == CID["bus_stop"]
    assert r.cls[r.xy_to_ij(20.0, 20.0)[0], r.xy_to_ij(20.0, 20.0)[1]] == CID["tree"]
    assert r.cls[r.xy_to_ij(30.0, 30.0)[0], r.xy_to_ij(30.0, 30.0)[1]] == CID["street_furniture"]


def test_humans_are_not_rasterised_but_exported():
    sc = _scene(humans=[Human(id="h0", pos=(10.0, 10.0, 0.0), role="casualty", pose="prone"),
                        Human(id="h1", pos=(12.0, 12.0, 0.0), role="bystander", pose="standing",
                              container="open", visibility="open")])
    r = rasterize(sc, 2.0)
    assert not (r.cls == CID["human_prone"]).any()
    assert r.humans.shape == (2,)
    assert r.humans["role_id"][0] == schema.HUMAN_ROLES.index("casualty")
    assert r.humans["pose_id"][1] == schema.HUMAN_POSES.index("standing")
    assert list(r.humans["scene_idx"]) == [0, 1]


def test_damage_grid_preferred_over_analytic():
    vals = [[0.0] * 5 for _ in range(5)]
    vals[4][4] = 1.0
    df = DamageField(kind="uniform", params={"inside": 0.3},
                     grid={"cell_m": 4.0, "nx": 5, "ny": 5, "values": vals})
    sc = Scene(meta=Meta(region=(0.0, 0.0, 20.0, 20.0)), damage_field=df)
    r = rasterize(sc, 2.0)
    assert r.damage[0, 0] == 0.0
    assert r.damage[-1, -1] == pytest.approx(1.0)


def test_damage_analytic_matches_schema():
    sc = make_synthetic_scene(3, region_m=(60.0, 60.0))
    r = rasterize(sc, 2.0)
    for i in (0, 5, 20):
        for j in (0, 7, 29):
            x, y = r.ij_to_xy(i, j)
            assert r.damage[i, j] == pytest.approx(sc.damage_at(x, y), abs=1e-5)


def test_obstacle_mask():
    sc = _scene(region=(0.0, 0.0, 40.0, 40.0), buildings=[
        Building(id="a", center=(10.0, 10.0), size=(8.0, 8.0), category="midrise"),
        Building(id="b", center=(30.0, 30.0), size=(8.0, 8.0), category="house")])
    r = rasterize(sc, 2.0)
    m = r.obstacle_mask(25.0, 3.0)
    assert m[r.xy_to_ij(10.0, 10.0)[0], r.xy_to_ij(10.0, 10.0)[1]]
    assert not m[r.xy_to_ij(30.0, 30.0)[0], r.xy_to_ij(30.0, 30.0)[1]]
    assert not r.obstacle_mask(100.0, 3.0).any()


def test_empty_scene_and_bad_input():
    r = rasterize(_scene(), 2.0)
    assert r.cls.max() == CID["ground"] and r.height.max() == 0.0 and r.humans.shape == (0,)
    with pytest.raises(ValueError):
        rasterize(_scene(), 0.0)
    with pytest.raises(ValueError):
        rasterize(_scene(region=(0.0, 0.0, 0.0, 5.0)), 1.0)
