import math

import numpy as np
import pytest

from rlplanner.sim import geometry as G


def _flat(n=40):
    return np.zeros((n, n), np.float32)


ORIGIN = np.array([0.0, 0.0])


def _los(h, cam, tgt, step=0.5, eps=0.2):
    return bool(G.los_visible(h, 1.0, ORIGIN, np.asarray(cam, float),
                              np.asarray([tgt], float), step, eps)[0])


def test_los_clear_ground():
    assert _los(_flat(), (5.5, 5.5, 10.0), (25.5, 5.5, 0.0))


def test_los_blocked_by_tall_building():
    h = _flat()
    h[5, 15] = 30.0                      # wall cell between camera and target
    assert not _los(h, (5.5, 5.5, 5.0), (25.5, 5.5, 0.0))


def test_los_not_blocked_by_low_object():
    h = _flat()
    h[5, 15] = 1.0                       # a car does not occlude a ray 5 m up
    assert _los(h, (5.5, 5.5, 5.0), (25.5, 5.5, 0.0))


def test_los_end_and_start_cells_are_skipped():
    h = _flat()
    h[5, 5] = 40.0                       # the camera's own cell
    h[5, 25] = 40.0                      # the target's own cell
    assert _los(h, (5.5, 5.5, 10.0), (25.5, 5.5, 0.0))


def test_los_symmetric_between_two_high_points():
    h = _flat()
    h[5, 15] = 20.0
    a, b = (5.5, 5.5, 30.0), (25.5, 5.5, 30.0)
    assert _los(h, a, b) == _los(h, b, a)
    h[5, 15] = 40.0
    assert _los(h, a, b) == _los(h, b, a) is False


def test_los_eps_tolerance():
    h = _flat()
    h[5, 15] = 5.1
    assert _los(h, (5.5, 5.5, 5.0), (25.5, 5.5, 5.0), eps=0.2)      # within eps
    assert not _los(h, (5.5, 5.5, 5.0), (25.5, 5.5, 5.0), eps=0.05)


def test_los_batch_matches_single():
    rng = np.random.default_rng(0)
    h = (rng.random((40, 40)) * 12).astype(np.float32)
    cam = np.array([2.5, 2.5, 15.0])
    tg = np.stack([rng.uniform(0, 40, 50), rng.uniform(0, 40, 50), np.zeros(50)], 1)
    out = G.los_visible(h, 1.0, ORIGIN, cam, tg, 0.5, 0.2)
    for k in range(50):
        assert out[k] == _los(h, cam, tg[k])


def test_frustum_edges():
    cam = np.array([0.0, 0.0, 0.0])
    hf, vf = math.radians(90.0), math.radians(60.0)
    inside = math.tan(math.radians(44.0))
    outside = math.tan(math.radians(46.0))
    t = np.array([[1.0, inside, 0.0], [1.0, outside, 0.0], [1.0, -inside, 0.0],
                  [1.0, -outside, 0.0], [-1.0, 0.0, 0.0]])
    assert list(G.in_frustum(cam, 0.0, 0.0, hf, vf, t)) == [True, False, True, False, False]
    el_in = math.tan(math.radians(29.0))
    el_out = math.tan(math.radians(31.0))
    t2 = np.array([[1.0, 0.0, el_in], [1.0, 0.0, el_out], [1.0, 0.0, -el_in], [1.0, 0.0, -el_out]])
    assert list(G.in_frustum(cam, 0.0, 0.0, hf, vf, t2)) == [True, False, True, False]


def test_frustum_follows_yaw_and_pitch():
    cam = np.array([0.0, 0.0, 10.0])
    t = np.array([[0.0, 10.0, 0.0]])                     # due north, 45 deg down
    assert not G.in_frustum(cam, 0.0, math.radians(-45), math.radians(90), math.radians(60), t)[0]
    assert G.in_frustum(cam, math.radians(90), math.radians(-45), math.radians(90),
                        math.radians(60), t)[0]


def test_astar_straight_line_free_space():
    p = G.astar(np.zeros((20, 20), bool), (2, 2), (2, 10))
    assert p is not None and p.shape == (9, 2)
    assert tuple(p[0]) == (2, 2) and tuple(p[-1]) == (2, 10)


def test_astar_around_a_wall():
    ob = np.zeros((20, 20), bool)
    ob[:, 10] = True
    ob[0, 10] = False                       # single gap at the top
    p = G.astar(ob, (5, 2), (5, 18))
    assert p is not None
    assert not ob[p[:, 0], p[:, 1]].any()
    assert (p[:, 1] == 10).sum() == 1 and p[np.argmax(p[:, 1] == 10), 0] == 0
    d = np.abs(np.diff(p, axis=0))
    assert d.max() <= 1 and (d.sum(axis=1) > 0).all()


def test_astar_none_when_enclosed():
    ob = np.zeros((20, 20), bool)
    ob[8:13, 8] = ob[8:13, 12] = True
    ob[8, 8:13] = ob[12, 8:13] = True
    assert G.astar(ob, (10, 10), (1, 1)) is None
    assert G.astar(ob, (1, 1), (10, 10)) is None


def test_astar_rejects_bad_endpoints():
    ob = np.zeros((10, 10), bool)
    ob[5, 5] = True
    assert G.astar(ob, (5, 5), (1, 1)) is None
    assert G.astar(ob, (1, 1), (5, 5)) is None
    assert G.astar(ob, (1, 1), (99, 99)) is None


def test_planner_cache_and_reachability():
    ob = np.zeros((20, 20), bool)
    ob[:, 10] = True
    pl = G.PathPlanner(ob)
    assert not pl.reachable((5, 2), (5, 18))
    assert pl.path((5, 2), (5, 18)) is None
    assert pl.reachable((5, 2), (5, 8))
    a = pl.path((5, 2), (5, 8))
    assert a is pl.path((5, 2), (5, 8))          # cached object identity
    assert len(pl.cache) == 2


def test_corridor_observed_frac():
    obs = np.zeros((40, 40), bool)
    origins = np.array([[5.5, 5.5]])
    az = np.array([0.0])
    assert G.corridor_observed_frac(obs, 1.0, 0.0, 0.0, origins, az, 2.0, 10.0)[0] == 0.0
    obs[5, :] = True
    assert G.corridor_observed_frac(obs, 1.0, 0.0, 0.0, origins, az, 2.0, 10.0)[0] == 1.0
    obs[5, 8:] = False
    f = G.corridor_observed_frac(obs, 1.0, 0.0, 0.0, origins, az, 2.0, 10.0)[0]
    assert 0.0 < f < 1.0
