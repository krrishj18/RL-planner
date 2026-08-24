"""The coverage and assignment baselines: `lawnmower` and `oracle_assign` (CONTRACTS.md 7).

The divert tests drive `LawnmowerPolicy.act` on a hand-built `TeamObs` rather than an episode: a
far-field human ray needs an `open`-visibility casualty to land in one 20 degree bin at the right
range, which is not something a scene seed can be asked for reproducibly.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.scene.schema import DamageField, Meta, Scene, make_synthetic_scene
from rlplanner.sim.baselines import (HUMAN_CLASSES, LawnmowerPolicy, assign_casualties,
                                     make_policy)
from rlplanner.sim.config import CommsConfig, EnvConfig
from rlplanner.sim.embeddings import get_embedding_table
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.state import (F_DIST, F_FEAT0, F_XABS, F_YABS, TOKEN_FIXED, TOKEN_FRONTIER,
                                 TOKEN_HOLD, TOKEN_RAY, TeamObs)

D = 24
REGION = (160.0, 160.0)


def _cfg(robots=3, t_max=200.0, cell=2.0):
    c = EnvConfig()
    c.robot.n_robots = robots
    c.t_max_s = t_max
    c.raster.cell_m = cell
    return c


def _scene(seed=0, region=REGION, n_casualties=6, n_bystanders=3):
    return make_synthetic_scene(seed, region_m=region, n_casualties=n_casualties,
                                n_bystanders=n_bystanders)


def _episode(env, policy, seed=0, n=None):
    obs = env.reset(seed)
    policy.reset(seed)
    k = 0
    while True:
        obs, _, done, info = env.step(policy.act(obs, env.state))
        k += 1
        if done or (n is not None and k >= n):
            return obs, info, k


# ---- a hand-built observation ------------------------------------------------------------------
def _obs(items, n_robots=1, dim=D):
    """`items` = list of (type, id, x_norm, y_norm, dist_norm, class_name|None) per robot slot.

    A `class_name` gives the token the unit class embedding of that class, which is exactly what a
    ray of that class carries; `None` leaves the feature tail zero.
    """
    emb = get_embedding_table(EnvConfig().rayfronts.queries, dim=dim)
    K = max(len(items), 1) if not isinstance(items[0], list) else max(len(x) for x in items)
    per = items if isinstance(items[0], list) else [list(items)] * n_robots
    F = TOKEN_FIXED + dim
    tok = np.zeros((n_robots, K, F), np.float32)
    mask = np.zeros((n_robots, K), np.bool_)
    xy = np.full((n_robots, K, 2), np.nan, np.float32)
    tt = np.zeros((n_robots, K), np.int8)
    tid = np.full((n_robots, K), -1, np.int32)
    for r, row in enumerate(per):
        for k, (ty, i, xn, yn, dn, cls) in enumerate(row):
            tok[r, k, F_XABS], tok[r, k, F_YABS], tok[r, k, F_DIST] = xn, yn, dn
            if cls is not None:
                tok[r, k, F_FEAT0:] = emb.class_emb[schema.CLASS_ID[cls]]
            mask[r, k] = True
            tt[r, k], tid[r, k] = ty, i
            xy[r, k] = (xn * 100.0, yn * 100.0)
    return TeamObs(tokens=tok, token_mask=mask, token_xy=xy, token_type=tt, token_id=tid,
                   robot_feat=np.zeros((n_robots, 18), np.float32),
                   bev=np.zeros((1, 4, 4), np.float32),
                   query_emb=np.zeros((8, dim), np.float32), query_w=np.zeros(8, np.float32),
                   query_mask=np.zeros(8, np.bool_), t=0.0)


HOLD = (TOKEN_HOLD, -1, 0.0, 0.0, 0.0, None)


# ---- registration -------------------------------------------------------------------------------
def test_both_policies_are_registered():
    from rlplanner.sim.baselines import POLICIES
    assert {"lawnmower", "oracle_assign"} <= set(POLICIES)
    assert not make_policy("lawnmower").privileged
    assert make_policy("oracle_assign").privileged


def test_oracle_assign_needs_the_state():
    env = DisasterEnv(_scene(), _cfg())
    with pytest.raises(ValueError):
        make_policy("oracle_assign").act(env.state.last_obs, None)


# ---- lawnmower: the sweep -----------------------------------------------------------------------
def test_lawnmower_sweeps_an_empty_scene_to_full_coverage():
    sc = Scene(meta=Meta(region=(0.0, 0.0, 160.0, 160.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}))
    env = DisasterEnv(sc, _cfg(robots=3, t_max=400.0))
    _, info, _ = _episode(env, make_policy("lawnmower"))
    assert info["coverage"] > 0.95


@pytest.mark.parametrize("robots", [1, 8, 10])      # 10 = the largest EnvConfig.validate allows
def test_lawnmower_runs_from_one_robot_to_the_largest_team(robots):
    env = DisasterEnv(_scene(), _cfg(robots=robots, t_max=120.0))
    obs, info, _ = _episode(env, make_policy("lawnmower"))
    assert obs.n_robots == robots
    assert np.isfinite(obs.tokens).all() and info["coverage"] > 0.0


def test_lawnmower_targets_stay_in_the_robots_band():
    """Whenever an in-band frontier is on offer, the chosen token is in that band."""
    env = DisasterEnv(_scene(region=(200.0, 200.0)), _cfg(robots=3, t_max=200.0))
    pol = make_policy("lawnmower")
    obs = env.reset(0)
    pol.reset(0)
    inband, total = 0, 0
    while True:
        act = pol.act(obs, env.state)
        band = 2.0 / obs.n_robots
        for r in range(obs.n_robots):
            lo = -1.0 + band * r
            hi = 1.0 + 1e-6 if r == obs.n_robots - 1 else lo + band
            yn = obs.tokens[r, :, F_YABS]
            frontier = obs.token_mask[r] & (obs.token_type[r] == TOKEN_FRONTIER)
            k = int(act[r])
            if not (frontier & (yn >= lo) & (yn < hi)).any():
                continue
            if int(obs.token_type[r, k]) != TOKEN_FRONTIER:
                continue
            total += 1
            inband += int(lo <= float(obs.tokens[r, k, F_YABS]) < hi)
        obs, _, done, _ = env.step(act)
        if done:
            break
    # the only way out of the band is a lower-index robot's position claim on the frontier the
    # sweep wanted (bands touch, and the claim radius is 15 m)
    assert total > 20 and inband / total > 0.9


def test_lawnmower_holds_when_nothing_is_on_offer():
    obs = _obs([HOLD])
    assert int(make_policy("lawnmower").act(obs, None)[0]) == 0


def test_lawnmower_with_no_tokens_at_all_does_not_crash():
    obs = _obs([HOLD])
    obs.token_mask[:] = False
    assert int(make_policy("lawnmower").act(obs, None)[0]) == 0


def _stub_state(region=(0.0, 0.0, 200.0, 200.0), pos=(0.0, 0.0)):
    """The only three things the sweep reads off the state: the region, the sensor and the pose."""
    return SimpleNamespace(cfg=EnvConfig(), emb=None,
                           raster=SimpleNamespace(region=region),
                           robots=[SimpleNamespace(pos=np.asarray(pos, np.float64))])


def test_lawnmower_sweep_is_a_serpentine_over_the_lanes():
    """A 200 m band is ~4 sensor swaths: the low lane runs +x, so the leftmost frontier of the
    *lowest* lane is the next waypoint and a lane-3 frontier further left does not jump the queue."""
    items = [HOLD,
             (TOKEN_FRONTIER, 1, 0.9, -0.9, 0.5, None),    # lane 0, far right
             (TOKEN_FRONTIER, 2, -0.8, -0.9, 0.4, None),   # lane 0, left: next in the order
             (TOKEN_FRONTIER, 3, -0.9, 0.9, 0.6, None)]    # lane 3, further left still
    pol = make_policy("lawnmower")
    pol.min_travel_m = 0.0
    obs = _obs(items)
    a = int(pol.act(obs, _stub_state())[0])
    assert int(obs.token_id[0, a]) == 2


def test_lawnmower_resumes_the_sweep_where_it_stopped():
    """No sweep cursor: the pick is the min-key frontier still on offer, so a divert cannot
    restart the band — coming back lands on the same token a never-diverted policy picks."""
    swept = (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None)
    rest = [HOLD, (TOKEN_FRONTIER, 2, -0.2, -0.9, 0.4, None),
            (TOKEN_FRONTIER, 3, 0.7, -0.9, 0.6, None)]
    ray = (TOKEN_RAY, 9, 0.1, 0.1, 0.3, "human_prone")

    pol = make_policy("lawnmower")
    assert int(pol.act(_obs([HOLD, swept] + rest[1:]), None)[0]) == 1     # first sweep waypoint
    assert int(pol.act(_obs(rest + [ray]), None)[0]) == 3                 # diverts to the ray
    fresh = make_policy("lawnmower")
    assert int(pol.act(_obs(rest), None)[0]) == int(fresh.act(_obs(rest), None)[0]) == 1


# ---- lawnmower: investigate ---------------------------------------------------------------------
def test_a_human_ray_diverts_the_sweep():
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
             (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_prone")]
    assert int(make_policy("lawnmower").act(_obs(items), None)[0]) == 2


def test_a_container_ray_does_not_divert_the_sweep():
    for cls in ("vehicle_toppled", "building_damaged", "debris"):
        items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
                 (TOKEN_RAY, 7, 0.5, 0.5, 0.6, cls)]
        assert int(make_policy("lawnmower").act(_obs(items), None)[0]) == 1, cls


def test_the_nearest_human_ray_wins():
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
             (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_standing"),
             (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone"),
             (TOKEN_RAY, 9, 0.2, -0.3, 0.9, "human_prone")]
    assert int(make_policy("lawnmower").act(_obs(items), None)[0]) == 3


def test_the_investigation_is_sticky_until_the_ray_goes_away():
    near = (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")
    chased = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None), near]
    pol = make_policy("lawnmower")
    assert int(pol.act(_obs(chased), None)[0]) == 2
    # a nearer human ray appears: the robot stays on the one it is already investigating
    with_nearer = chased + [(TOKEN_RAY, 11, 0.1, 0.1, 0.05, "human_prone")]
    ob = _obs(with_nearer)
    assert int(ob.token_id[0, int(pol.act(ob, None)[0])]) == 8
    # the ray resolves (leaves the token set): back to the sweep
    assert int(pol.act(_obs(chased[:2]), None)[0]) == 1


def test_reset_forgets_the_investigation():
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
             (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")]
    pol = make_policy("lawnmower")
    pol.act(_obs(items), None)
    assert pol._chasing
    pol.reset(0)
    assert not pol._chasing


def test_human_classification_is_the_argmax_over_the_whole_class_set():
    """Threshold-free: every class row is classified as itself, and only the two human rows
    are treated as a person."""
    emb = get_embedding_table(EnvConfig().rayfronts.queries, dim=D)
    pol = make_policy("lawnmower")
    for name, cid in schema.CLASS_ID.items():
        items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None), (TOKEN_RAY, 7, 0.5, 0.5, 0.6,
                                                                    name)]
        obs = _obs(items)
        hit = pol._human_rays(obs, 0, np.asarray(emb.class_emb, np.float32))
        assert bool(hit[2]) == (cid in HUMAN_CLASSES), name


def test_two_robots_do_not_chase_the_same_ray():
    row = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.5, 0.2, None),
           (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_prone")]
    obs = _obs([row, list(row)], n_robots=2)
    a = make_policy("lawnmower").act(obs, None)
    assert int(a[0]) == 2 and int(a[1]) != 2


def test_the_divert_fires_in_a_real_episode():
    """The synthetic-obs tests pin the rule; this one says the rule actually meets a human ray
    on a live belief (27 of 90 robot-decisions on this seed) and the sweep still covers."""
    env = DisasterEnv(_scene(seed=0, region=(240.0, 240.0)), _cfg(robots=3, t_max=200.0))
    pol = make_policy("lawnmower", queries=env.cfg.rayfronts.queries)
    obs = env.reset(0)
    pol.reset(0)
    diverts = 0
    while True:
        a = pol.act(obs, env.state)
        diverts += int((obs.token_type[np.arange(obs.n_robots), a] == TOKEN_RAY).sum())
        obs, _, done, info = env.step(a)
        if done:
            break
    assert diverts > 0
    assert info["coverage"] > 0.5


# ---- oracle_assign ------------------------------------------------------------------------------
def test_assignment_is_the_min_cost_matching():
    pos = np.array([[0.0, 0.0], [10.0, 0.0]])
    tx, ty = np.array([9.0, 1.0]), np.array([0.0, 0.0])
    # greedy on robot 0 would take the casualty at x=1 and leave robot 1 the one at x=9;
    # the matching agrees here, and swapping the robots swaps the answer
    assert assign_casualties(pos, tx, ty).tolist() == [1, 0]
    assert assign_casualties(pos[::-1], tx, ty).tolist() == [0, 1]


def test_assignment_beats_greedy_where_greedy_is_wrong():
    """Robot 0 is marginally nearer both casualties; greedy takes the far one and strands robot 1."""
    pos = np.array([[0.0, 0.0], [0.0, 100.0]])
    tx, ty = np.array([0.0, 0.0]), np.array([1.0, 101.0])
    plan = assign_casualties(pos, tx, ty)
    assert plan.tolist() == [0, 1]
    cost = np.hypot(pos[:, 0] - tx[plan], pos[:, 1] - ty[plan]).sum()
    assert cost < np.hypot(pos[:, 0] - tx[[1, 0]], pos[:, 1] - ty[[1, 0]]).sum()


def test_more_robots_than_casualties_leaves_the_extras_unassigned():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    plan = assign_casualties(pos, np.array([1.0]), np.array([0.0]))
    assert plan.tolist() == [0, -1, -1]
    assert assign_casualties(pos, np.zeros(0), np.zeros(0)).tolist() == [-1, -1, -1]


def test_oracle_assign_finds_all_casualties_no_slower_than_oracle():
    cfg = _cfg(robots=3, t_max=400.0)
    scene = _scene(seed=0, region=(200.0, 200.0), n_casualties=6, n_bystanders=2)
    out = {}
    for name in ("oracle", "oracle_assign"):
        env = DisasterEnv(scene, cfg, seed=0)
        _, info, k = _episode(env, make_policy(name, queries=cfg.rayfronts.queries), seed=0)
        out[name] = (info["metrics"]["frac_found"], k)
    assert out["oracle_assign"][0] >= out["oracle"][0]
    if out["oracle_assign"][0] >= 1.0 and out["oracle"][0] >= 1.0:
        assert out["oracle_assign"][1] <= out["oracle"][1]


def test_oracle_assign_with_no_casualties_runs_to_t_max():
    sc = make_synthetic_scene(0, region_m=(100.0, 100.0), n_casualties=0, n_bystanders=3)
    env = DisasterEnv(sc, _cfg(robots=2, t_max=60.0))
    _, info, _ = _episode(env, make_policy("oracle_assign"))
    assert info["n_casualties"] == 0 and info["found_total"] == 0


def test_oracle_assign_with_more_robots_than_casualties():
    sc = make_synthetic_scene(1, region_m=(140.0, 140.0), n_casualties=2, n_bystanders=2)
    env = DisasterEnv(sc, _cfg(robots=6, t_max=200.0))
    obs, info, _ = _episode(env, make_policy("oracle_assign"))
    assert obs.n_robots == 6 and np.isfinite(obs.tokens).all()
    assert info["metrics"]["frac_found"] > 0.0


@pytest.mark.parametrize("name", ["lawnmower", "oracle_assign"])
def test_same_seed_gives_identical_trajectories(name):
    cfg = _cfg(robots=3, t_max=120.0)
    scene = _scene(seed=2)
    runs = []
    for _ in range(2):
        env = DisasterEnv(scene, cfg, seed=3)
        pol = make_policy(name, queries=cfg.rayfronts.queries, seed=3)
        obs = env.reset(3)
        pol.reset(3)
        acts, pos = [], []
        while True:
            a = pol.act(obs, env.state)
            acts.append(a.copy())
            obs, _, done, _ = env.step(a)
            pos.append(np.array([r.pos.copy() for r in env.state.robots]))
            if done:
                break
        runs.append((np.array(acts), np.array(pos)))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])


# ---- lawnmower: hostile observations -------------------------------------------------------------
@pytest.mark.parametrize("qdim", [0, 8, 64])
def test_the_human_argmax_reads_the_token_feature_width_not_the_query_block(qdim):
    """The class table falls back at the width of the token feature tail, which is the only D the
    argmax can use. Taking it from `query_emb` instead raises a matmul shape error on any
    observation whose query block is a different width (or empty)."""
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
             (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_prone")]
    obs = _obs(items)
    obs.query_emb = np.zeros((4, qdim), np.float32)
    obs.query_w, obs.query_mask = np.zeros(4, np.float32), np.zeros(4, np.bool_)
    assert int(make_policy("lawnmower").act(obs, None)[0]) == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_ray_feature_never_diverts_and_never_warns(bad):
    obs = _obs([HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
                (TOKEN_RAY, 7, 0.5, 0.5, 0.6, None)])
    obs.tokens[0, 2, F_FEAT0:] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert int(make_policy("lawnmower").act(obs, None)[0]) == 1


# ---- lawnmower: degenerate geometry --------------------------------------------------------------
@pytest.mark.parametrize("alt,depth", [(25.0, 35.0), (34.999, 35.0), (35.0, 35.0), (60.0, 35.0)])
def test_the_lane_height_stays_finite_when_the_sensor_cannot_reach_the_ground(alt, depth):
    """`depth_limit <= flight_alt` is a config `validate` rejects, but the swath must not become a
    nan (sqrt of a negative) or a zero lane height if the sweep is ever handed one."""
    st = _stub_state()
    st.cfg.robot.flight_alt_m, st.cfg.sensor.depth_limit_m = alt, depth
    h = LawnmowerPolicy._lane_height(st)
    assert np.isfinite(h) and h > 0.0
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None),
             (TOKEN_FRONTIER, 2, 0.9, 0.9, 0.5, None)]
    assert int(make_policy("lawnmower").act(_obs(items), st)[0]) in (1, 2)


@pytest.mark.parametrize("region", [(0.0, 0.0, 4.0, 4.0),          # a handful of cells
                                    (0.0, 0.0, 200.0, 0.0),        # zero height
                                    (0.0, 0.0, 500.0, 1500.0),     # tall
                                    (0.0, 0.0, 1500.0, 500.0)])    # wide
def test_the_sweep_survives_a_degenerate_or_lopsided_region(region):
    st = _stub_state(region=region)
    h = LawnmowerPolicy._lane_height(st)
    assert np.isfinite(h) and h > 0.0
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None),
             (TOKEN_FRONTIER, 2, 0.9, 0.9, 0.5, None)]
    assert int(make_policy("lawnmower").act(_obs(items), st)[0]) in (1, 2)


@pytest.mark.parametrize("robots", [1, 10])
def test_every_robot_of_a_crowded_team_picks_a_valid_slot(robots):
    """Bands thinner than a lane and more robots than the region has lanes: still one valid slot
    each (the extras hold once every frontier is claimed)."""
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None),
             (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_prone")]
    obs = _obs(items, n_robots=robots)
    a = make_policy("lawnmower").act(obs, None)
    assert a.shape == (robots,) and obs.token_mask[np.arange(robots), a].all()


# ---- lawnmower: the chase ends -------------------------------------------------------------------
def test_a_chase_that_resolves_mid_flight_moves_to_the_next_human_ray():
    base = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None)]
    near = (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")
    far = (TOKEN_RAY, 9, 0.6, 0.6, 0.8, "human_standing")
    pol = make_policy("lawnmower")
    assert int(pol.act(_obs(base + [near, far]), None)[0]) == 2      # the nearer one first
    ob = _obs(base + [far])                                          # it resolves mid-chase
    assert int(ob.token_id[0, int(pol.act(ob, None)[0])]) == 9       # on to the other one


def test_the_chase_ends_when_the_ray_falls_inside_the_min_travel_guard():
    base = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None)]
    pol = make_policy("lawnmower")
    assert int(pol.act(_obs(base + [(TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")]), None)[0]) == 2
    arrived = base + [(TOKEN_RAY, 8, -0.4, 0.2, 0.0, "human_prone")]
    assert int(pol.act(_obs(arrived), None)[0]) == 1                 # arrived: back to the sweep
    assert not pol._chasing


def test_a_ray_id_that_comes_back_as_something_else_drops_the_chase():
    base = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None)]
    pol = make_policy("lawnmower")
    pol.act(_obs(base + [(TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")]), None)
    assert pol._chasing[0] == 8
    ob = _obs(base + [(TOKEN_RAY, 8, 0.5, 0.5, 0.6, "vehicle_toppled"),
                      (TOKEN_RAY, 12, -0.3, 0.1, 0.3, "human_standing")])
    assert int(ob.token_id[0, int(pol.act(ob, None)[0])]) == 12


def test_a_sweep_that_diverts_to_every_ray_still_covers():
    """The resolution rule is what keeps the divert from livelocking: classify *every* ray as a
    person and the band is still swept, because a chase ends when the robot arrives."""
    class AllHuman(LawnmowerPolicy):
        def _human_rays(self, obs, r, cls):
            return obs.token_mask[r] & (obs.token_type[r] == TOKEN_RAY)
    env = DisasterEnv(_scene(region=(200.0, 200.0)), _cfg(robots=3, t_max=300.0))
    _, info, _ = _episode(env, AllHuman(queries=env.cfg.rayfronts.queries))
    assert info["coverage"] > 0.9


def test_the_divert_runs_on_the_robots_own_view_under_range_comms():
    """Nothing in the rule reads the team map, so a blackout changes neither the classification nor
    the sweep: the robot investigates the human rays its own belief carries."""
    cfg = _cfg(robots=3, t_max=200.0)
    cfg.comms = CommsConfig(mode="range", range_m=0.0)
    env = DisasterEnv(_scene(seed=0, region=(240.0, 240.0)), cfg)
    pol = make_policy("lawnmower", queries=cfg.rayfronts.queries)
    obs = env.reset(0)
    pol.reset(0)
    diverts = 0
    while True:
        a = pol.act(obs, env.state)
        assert obs.token_mask[np.arange(obs.n_robots), a].all()
        diverts += int((obs.token_type[np.arange(obs.n_robots), a] == TOKEN_RAY).sum())
        obs, _, done, info = env.step(a)
        if done:
            break
    assert diverts > 0 and info["coverage"] > 0.3


# ---- oracle_assign: more edges ------------------------------------------------------------------
def test_the_matching_is_deterministic_when_the_costs_tie():
    pos = np.zeros((2, 2))
    tx, ty = np.array([1.0, -1.0]), np.zeros(2)
    assert {tuple(assign_casualties(pos, tx, ty).tolist()) for _ in range(8)} == {(0, 1)}
    sq = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    assert assign_casualties(sq, sq[:, 0].copy(), sq[:, 1].copy()).tolist() == [0, 1, 2, 3]


def test_oracle_assign_with_every_casualty_in_one_cell():
    """A degenerate cost matrix (identical columns): the matching still hands each robot one."""
    sc = _scene(seed=5, region=(160.0, 160.0), n_casualties=4, n_bystanders=1)
    for h in sc.humans:
        if h.role == "casualty":
            h.x, h.y = 80.0, 80.0
    env = DisasterEnv(sc, _cfg(robots=3, t_max=200.0))
    _, info, _ = _episode(env, make_policy("oracle_assign", queries=env.cfg.rayfronts.queries))
    assert info["metrics"]["frac_found"] > 0.5


def test_oracle_assign_does_not_thrash_with_one_robot_and_many_casualties():
    """Re-solving from scratch every decision is only safe because the answer is stable: the goal
    moves when a casualty is found, not every step."""
    env = DisasterEnv(_scene(seed=7, region=(200.0, 200.0), n_casualties=8, n_bystanders=1),
                      _cfg(robots=1, t_max=400.0))
    pol = make_policy("oracle_assign", queries=env.cfg.rayfronts.queries)
    obs = env.reset(0)
    pol.reset(0)
    goals = []
    while True:
        a = pol.act(obs, env.state)
        goals.append(-1 if pol._plan is None else int(pol._plan[0]))
        obs, _, done, _ = env.step(a)
        if done:
            break
    switches = sum(1 for i in range(1, len(goals)) if goals[i] != goals[i - 1])
    assert len(goals) > 40 and switches < len(goals) / 4


# ---- instant confirm for privileged rows -------------------------------------------------------
def test_instant_confirm_overrides_perception():
    from rlplanner.sim.config import EnvConfig, instant_confirm
    cfg = EnvConfig()
    c = instant_confirm(cfg)
    assert c.rayfronts.found_hits == 1
    assert all(v == 1.0 for v in c.rayfronts.p_observe_base.values())
    assert c.rayfronts.far_observe_factor == 1.0
    assert cfg.rayfronts.found_hits == 2, "original untouched"


def test_privileged_episodes_confirm_on_arrival_where_stochastic_never_could():
    """With p_observe forced to 0 nothing is ever found -- except by a privileged row, whose
    episode runs under `instant_confirm` (the oracle bounds planning, not perception)."""
    from rlplanner.sim.config import EnvConfig
    from rlplanner.train.par_env import _run_episodes
    from rlplanner.train.scenes import SceneBank
    cfg = EnvConfig()
    cfg.rayfronts.p_observe_base = {k: 0.0 for k in cfg.rayfronts.p_observe_base}
    bank = SceneBank("synthetic:0-2", region_m=(200.0, 200.0))
    key = bank.split("train")[0]
    blind = _run_episodes(bank, cfg, "ray_follower", [(key, 0)], 2, max_decisions=40)
    assert blind[0]["frac_found"] == 0.0
    seen = _run_episodes(bank, cfg, "oracle", [(key, 0)], 2, max_decisions=40)
    assert seen[0]["frac_found"] > 0.0
