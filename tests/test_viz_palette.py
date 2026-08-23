import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.sim import state as S
from rlplanner.viz import palette as P


def test_every_class_has_a_colour():
    assert set(P.CLASS_COLORS) == set(schema.CLASS_NAMES)
    assert P.class_rgb_array().shape == (schema.N_CLASSES, 3)
    for i, n in enumerate(schema.CLASS_NAMES):
        assert P.class_color(i) == P.class_color(n)


def test_class_cmap_maps_ids():
    cmap, norm = P.class_cmap()
    assert cmap.N == schema.N_CLASSES
    for i in range(schema.N_CLASSES):
        assert np.allclose(cmap(norm(i))[:3], P.class_rgb_array()[i], atol=1e-6)


def test_bad_class_raises():
    with pytest.raises(IndexError):
        P.class_color(schema.N_CLASSES)
    with pytest.raises(KeyError):
        P.class_color("not_a_class")


def test_fate_vehicle_role_token_colours():
    for f in schema.BUILDING_FATES:
        assert f in P.FATE_COLORS
    for v in schema.VEHICLE_STATES:
        assert v in P.VEHICLE_COLORS
    for r in schema.HUMAN_ROLES:
        assert P.human_color(r)
    for c in schema.HUMAN_CONTAINERS:
        assert P.human_marker(c)
    for i, n in enumerate(S.TOKEN_TYPE_NAMES):
        assert P.token_color(i) == P.token_color(n)
    with pytest.raises(IndexError):
        P.token_color(len(S.TOKEN_TYPE_NAMES))
    assert len({P.robot_color(i) for i in range(10)}) == 10


def test_sim_rgb_paints_unobserved_dark():
    v = np.array([[0.0, 0.5, 1.0]])
    obs = np.array([[True, True, False]])
    rgb = P.sim_rgb(v, obs)
    assert rgb.shape == (1, 3, 3)
    from matplotlib.colors import to_rgb
    assert np.allclose(rgb[0, 2], to_rgb(P.UNOBSERVED))
    assert rgb[0, 0].max() < 0.6  # low similarity stays dark but is not the unobserved colour
    assert not np.allclose(rgb[0, 0], to_rgb(P.UNOBSERVED))
    assert rgb[0, 2].max() < rgb[0, 1].max()


def test_sim_rgb_shape_mismatch_raises():
    with pytest.raises(ValueError):
        P.sim_rgb(np.zeros((2, 2)), np.zeros((3, 3), bool))


def test_shade_by_height_keeps_flat_ground():
    rgb = np.ones((2, 2, 3))
    h = np.array([[0.0, 0.0], [5.0, 20.0]])
    out = P.shade_by_height(rgb, h)
    assert np.allclose(out[0], 1.0)
    assert out[1, 1, 0] < out[1, 0, 0] < 1.0
