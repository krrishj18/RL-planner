import math

import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.scene.schema import (Building, DamageField, Human, Meta, Road, Scene,
                                    make_synthetic_scene)
from rlplanner.sim.baselines import POLICIES, make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.state import TOKEN_FRONTIER, TOKEN_HOLD
from rlplanner.sim.vec_env import VecEnv


def _cfg(robots=3, t_max=60.0, team_terms=False, **kw):
    c = EnvConfig(**kw)
    c.robot.n_robots = robots
    c.t_max_s = t_max
    if not team_terms:      # the reward tests below check casualty + time only
        c.reward.redundancy_cost = 0.0
        c.reward.revisit_cost = 0.0
    return c


def _small_scene(seed=0):
    return make_synthetic_scene(seed, region_m=(120.0, 120.0), n_casualties=6, n_bystanders=3)


def _run(env, policy, n=None):
    obs = env.state.last_obs
    k = 0
    while True:
        obs, r, done, info = env.step(policy.act(obs, env.state))
        k += 1
        if done or (n is not None and k >= n):
            return obs, info, k


# ---- basic contract ---------------------------------------------------------------------------
def test_reset_shapes_and_types():
    env = DisasterEnv(_small_scene(), _cfg())
    o = env.state.last_obs
    from rlplanner.sim.state import TOKEN_FIXED
    from rlplanner.sim.tokens import BEV_CHANNELS, LOCAL_CHANNELS
    K, F = env.cfg.k_tokens, TOKEN_FIXED + env.cfg.rayfronts.embedding_dim
    assert o.tokens.shape == (3, K, F) and o.tokens.dtype == np.float32
    assert o.token_mask.shape == (3, K) and o.token_mask.dtype == np.bool_
    assert o.token_xy.shape == (3, K, 2)
    assert o.token_type.shape == (3, K) and o.token_id.shape == (3, K)
    assert o.robot_feat.shape == (3, 18)
    assert o.bev.shape == (len(BEV_CHANNELS), env.cfg.tokens.bev_size, env.cfg.tokens.bev_size)
    assert o.query_emb.shape == (env.cfg.tokens.max_queries, env.cfg.rayfronts.embedding_dim)
    assert int(o.query_mask.sum()) == env.cfg.n_queries
    assert o.local.shape == (3, len(LOCAL_CHANNELS), env.cfg.tokens.local_size,
                            env.cfg.tokens.local_size)
    assert o.t == 0.0
    assert o.token_mask[:, 0].all()                 # the hold slot is always selectable
    assert (o.token_type[:, 0] == TOKEN_HOLD).all()


def test_obs_has_no_nan_or_inf():
    env = DisasterEnv(_small_scene(), _cfg())
    pol = make_policy("ray_follower")
    obs, _, _ = _run(env, pol, n=8)
    for a in (obs.tokens, obs.robot_feat, obs.bev, obs.local, obs.query_emb):
        assert np.isfinite(a).all()
    nan = ~np.isfinite(obs.token_xy).all(axis=2)
    assert not obs.token_mask[nan].any()            # nan xy only on unselectable slots
    assert (obs.token_id[nan] == -1).all()
    assert (obs.tokens[nan] == 0).all()
    assert np.abs(obs.tokens).max() <= 5.0


def test_empty_slots_are_zeroed():
    env = DisasterEnv(_small_scene(), _cfg())
    o = env.state.last_obs
    empty = o.token_id[0] == -1
    empty[0] = False                                 # hold also carries id -1
    assert empty.any()
    assert not o.token_mask[0][empty].any()
    assert np.isnan(o.token_xy[0][empty]).all()


def test_masked_action_raises_naming_robot_and_token():
    env = DisasterEnv(_small_scene(), _cfg())
    o = env.state.last_obs
    bad = int(np.flatnonzero(~o.token_mask[1])[0])
    with pytest.raises(ValueError) as e:
        env.step(np.array([0, bad, 0]))
    assert "robot 1" in str(e.value) and f"token {bad}" in str(e.value)
    with pytest.raises(ValueError, match="out of range"):
        env.step(np.array([0, 999, 0]))
    with pytest.raises(ValueError, match="expected 3 actions"):
        env.step(np.array([0, 0]))


def test_hold_keeps_the_robot_still_but_sensing():
    env = DisasterEnv(_small_scene(), _cfg())
    p0 = env.state.robots[0].pos.copy()
    cov0 = env.state.observed.sum()
    env.step(np.zeros(3, int))
    assert np.allclose(env.state.robots[0].pos, p0)
    assert env.state.robots[0].dist_travelled == 0.0
    assert env.state.observed.sum() >= cov0


def test_robot_moves_toward_a_chosen_token():
    env = DisasterEnv(_small_scene(), _cfg())
    o = env.state.last_obs
    k = int(np.flatnonzero(o.token_mask[0] & (o.token_type[0] != TOKEN_HOLD))[0])
    tgt = o.token_xy[0, k].astype(float)
    d0 = math.hypot(*(env.state.robots[0].pos - tgt))
    env.step(np.array([k, 0, 0]))
    d1 = math.hypot(*(env.state.robots[0].pos - tgt))
    assert d1 < d0
    assert env.state.robots[0].dist_travelled > 0


# ---- determinism -------------------------------------------------------------------------------
def _fingerprint(env):
    st = env.state
    return (st.vox_feat_sum.copy(), st.vox_cnt.copy(), st.observed.copy(),
            np.array([r.pos for r in st.robots]), np.array([r.dist_travelled for r in st.robots]),
            st.rays.feat.copy(), st.rays.ids.copy(), st.rays.conf.copy(),
            st.human_hits.copy(), st.human_found.copy(), st.seg_labels.copy(),
            [c.id for c in st.segments], float(st.cum_reward), float(st.t))


def test_same_seed_gives_identical_state_after_20_decisions():
    sc = _small_scene()
    outs = []
    for _ in range(2):
        env = DisasterEnv(sc, _cfg(t_max=200.0), seed=7)
        _run(env, make_policy("ray_follower", seed=7), n=20)
        outs.append(_fingerprint(env))
    for a, b in zip(*outs):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b


def test_different_seed_gives_a_different_run():
    sc = _small_scene()
    fps = []
    for s in (1, 2):
        env = DisasterEnv(sc, _cfg(t_max=200.0), seed=s)
        _run(env, make_policy("ray_follower", seed=0), n=20)
        fps.append(_fingerprint(env))
    assert not np.array_equal(fps[0][0], fps[1][0])


def test_reset_restores_a_clean_belief():
    env = DisasterEnv(_small_scene(), _cfg())
    _run(env, make_policy("ray_follower"), n=5)
    o = env.reset(0)
    assert env.state.t == 0.0 and env.state.decision_idx == 0
    assert env.state.cum_reward == 0.0 and not env.state.events
    assert env.state.observed.sum() < 2000
    assert (env.state.human_hits <= 1).all()      # at most the one free look of reset()
    assert env.state.rays.n < 200
    o2 = DisasterEnv(_small_scene(), _cfg(), seed=0).state.last_obs
    assert np.array_equal(o.tokens, o2.tokens)


# ---- reward accounting --------------------------------------------------------------------------
def test_reward_counts_each_casualty_once_and_bystanders_zero():
    cfg = _cfg(t_max=300.0)
    cfg.reward.time_cost = 0.0
    env = DisasterEnv(_small_scene(1), cfg, seed=1)
    obs = env.state.last_obs
    pol = make_policy("oracle")
    total = 0.0
    found = 0
    while True:
        obs, r, done, info = env.step(pol.act(obs, env.state))
        total += r
        found += sum(1 for e in info["events_this_step"]
                     if e.kind == "found" and e.payload["casualty"])
        assert r == pytest.approx(info["new_found"] * cfg.reward.casualty_reward)
        if done:
            break
    assert found > 0
    assert total == pytest.approx(found * cfg.reward.casualty_reward)
    assert info["found_total"] == found
    ids = [e.payload["human_idx"] for e in env.state.events if e.kind == "found"]
    assert len(ids) == len(set(ids))                       # no double counting
    cas = [k for k in ids
           if env.raster.humans["role_id"][k] == schema.HUMAN_ROLES.index("casualty")]
    assert len(cas) == found


def test_a_found_bystander_pays_nothing():
    """Its voxels look like a standing person; the map cannot tell, and the reward does not care."""
    cfg = _cfg(t_max=200.0)
    cfg.reward.time_cost = 0.0
    sc = _small_scene()
    sc.humans = [h for h in sc.humans if h.role == "bystander"]
    env = DisasterEnv(sc, cfg, seed=0)
    obs = env.state.last_obs
    pol = make_policy("nearest_frontier")
    total = 0.0
    while True:
        obs, r, done, info = env.step(pol.act(obs, env.state))
        total += r
        if done:
            break
    assert env.state.human_found.any(), "no bystander was ever mapped"
    assert total == pytest.approx(0.0)
    assert info["found_total"] == 0


def test_time_cost_only():
    """A step that finds nothing costs exactly `time_cost`. Holding at the spawn can now see a
    casualty (the wedge reaches nadir), so the premise is asserted rather than assumed."""
    sc = _small_scene()
    sc.humans = [h for h in sc.humans if h.role != "casualty"]
    env = DisasterEnv(sc, _cfg())
    _, r, _, info = env.step(np.zeros(3, int))
    assert info["new_found"] == 0
    assert r == pytest.approx(-env.cfg.reward.time_cost)


# ---- termination ---------------------------------------------------------------------------------
def test_terminates_at_t_max():
    env = DisasterEnv(_small_scene(), _cfg(t_max=50.0))
    _, info, k = _run(env, make_policy("random"))
    assert k == 10 and env.state.t == pytest.approx(50.0)
    assert info["metrics"]["n_decisions"] == 10


def test_terminates_when_frontiers_are_gone():
    env = DisasterEnv(_small_scene(), _cfg(t_max=50.0))
    env.rf.observed[:] = True
    env.rf.vox_cnt[:] = 1
    env.rf.end_of_decision(0.0)
    assert not env.rf.frontier_clusters
    obs, info, k = _run(env, make_policy("nearest_frontier"))
    assert k == 10


def test_scene_without_casualties_runs_the_full_clock():
    """Nothing to find is not "everything found": the episode still fills t_max, paying only the
    time cost, or an RL run would see zero-length episodes."""
    sc = _small_scene()
    sc.humans = [h for h in sc.humans if h.role == "bystander"]
    env = DisasterEnv(sc, _cfg(t_max=30.0))
    tot = 0.0
    for k in range(6):
        _, r, done, info = env.step(np.zeros(3, int))
        assert r == pytest.approx(-env.cfg.reward.time_cost)
        tot += r
        assert done == (k == 5)
    assert info["n_casualties"] == 0 and info["found_total"] == 0
    assert info["metrics"]["frac_found"] == 1.0
    assert tot == pytest.approx(-6 * env.cfg.reward.time_cost)


# ---- team sizes / geometry -------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 10])
def test_team_sizes(n):
    env = DisasterEnv(_small_scene(), _cfg(robots=n, t_max=30.0))
    assert len(env.state.robots) == n
    obs, info, _ = _run(env, make_policy("ray_follower"))
    assert obs.tokens.shape[0] == n
    assert info["dist_travelled"].shape == (n,)
    assert np.isfinite(obs.tokens).all()
    pos = np.array([r.pos for r in env.state.robots])
    assert len(np.unique(pos.round(3), axis=0)) >= min(n, 4)      # jittered spawns are distinct


def test_tiny_region():
    sc = Scene(meta=Meta(region=(0.0, 0.0, 20.0, 20.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.5}),
               roads=[Road(id="r", rect=(0.0, 0.0, 20.0, 6.0))],
               humans=[Human(id="h", pos=(10.0, 12.0, 0.0), role="casualty", pose="prone")],
               robots_spawn=[(3.0, 3.0, 20.0)])
    env = DisasterEnv(sc, _cfg(robots=2, t_max=40.0))
    assert env.raster.shape == (10, 10)
    obs, info, _ = _run(env, make_policy("ray_follower"))
    assert np.isfinite(obs.tokens).all()
    assert 0.0 <= info["coverage"] <= 1.0


def test_spawn_inside_a_building_is_pushed_out():
    sc = Scene(meta=Meta(region=(0.0, 0.0, 80.0, 80.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
               buildings=[Building(id="b", center=(20.0, 20.0), size=(20.0, 20.0),
                                   category="highrise")],
               robots_spawn=[(20.0, 20.0, 20.0)])
    env = DisasterEnv(sc, _cfg(robots=1, t_max=20.0))
    p = env.state.robots[0].pos
    i, j = env.raster.xy_to_ij(p[0], p[1])
    assert not env.planner.obst[i, j]


def test_scene_without_spawns_uses_a_road_cell():
    sc = Scene(meta=Meta(region=(0.0, 0.0, 60.0, 60.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
               roads=[Road(id="r", rect=(10.0, 10.0, 50.0, 16.0))])
    env = DisasterEnv(sc, _cfg(robots=1, t_max=20.0))
    p = env.state.robots[0].pos
    assert 10.0 <= p[0] <= 50.0 and 10.0 <= p[1] <= 16.0


def test_walled_in_target_is_masked_and_unreachable():
    """A robot inside a sealed courtyard can only reach tokens inside it."""
    walls = []
    for k, (cx, cy, sx, sy) in enumerate([(40.0, 25.0, 30.0, 2.0), (40.0, 55.0, 30.0, 2.0),
                                          (25.0, 40.0, 2.0, 30.0), (55.0, 40.0, 2.0, 30.0)]):
        walls.append(Building(id=f"w{k}", center=(cx, cy), size=(sx, sy), category="highrise"))
    sc = Scene(meta=Meta(region=(0.0, 0.0, 80.0, 80.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
               buildings=walls, robots_spawn=[(40.0, 40.0, 20.0)])
    env = DisasterEnv(sc, _cfg(robots=1, t_max=30.0))
    o = env.state.last_obs
    ras = env.raster
    inside = env.planner.label_at(*ras.xy_to_ij(40.0, 40.0))
    for k in range(o.tokens.shape[1]):
        if not np.isfinite(o.token_xy[0, k]).all():
            continue
        lbl = env.planner.label_at(*ras.xy_to_ij(*o.token_xy[0, k]))
        if bool(o.token_mask[0, k]):
            assert lbl == inside
    _run(env, make_policy("ray_follower"))                       # must not crash


# ---- waypoint actions (CONTRACTS.md 6) ----------------------------------------------------------
def _flat_scene(region=(160.0, 160.0), buildings=()):
    return Scene(meta=Meta(region=(0.0, 0.0, region[0], region[1])),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
                 buildings=list(buildings))


def test_the_observation_carries_the_region():
    """The deployment extent as team metadata, so a waypoint policy need not guess it from the
    tokens it happens to hold."""
    env = DisasterEnv(_small_scene(), _cfg())
    o = env.state.last_obs
    assert o.region.shape == (4,) and np.allclose(o.region, env.raster.region)


def test_a_waypoint_action_routes_the_robot_to_the_point():
    env = DisasterEnv(_flat_scene(), _cfg(robots=2))
    rb0, rb1 = env.state.robots
    goal = rb0.pos + np.array([40.0, 20.0])
    d0, held = float(np.hypot(*(rb0.pos - goal))), rb1.pos.copy()
    a = np.full((2, 2), np.nan)
    a[0] = goal
    env.step(a)
    assert rb0.target_waypoint and rb0.target_token_type == TOKEN_FRONTIER
    assert rb0.target_id == -1 and rb0.last_action == -1 and rb0.target_feat is None
    assert float(np.hypot(*(rb0.pos - goal))) < d0            # A* routed it towards the point
    assert not rb1.target_waypoint and rb1.target_xy is None  # the nan row is hold
    assert np.allclose(rb1.pos, held)
    assert env.state.last_actions.shape == (2, 2)


def test_a_waypoint_outside_the_region_is_clipped():
    env = DisasterEnv(_flat_scene(), _cfg(robots=1))
    x0, y0, x1, y1 = env.raster.region
    env.step(np.array([[x1 + 500.0, y1 + 500.0]], np.float64))
    tx, ty = env.state.robots[0].target_xy
    assert x0 <= tx <= x1 and y0 <= ty <= y1


def test_an_unreachable_waypoint_holds_and_is_reported():
    """The same treatment a frontier token target gets: no path, no goal, an `unreachable` event."""
    tower = Building(id="tower", center=(80.0, 80.0), size=(40.0, 40.0), height_m=60.0)
    env = DisasterEnv(_flat_scene(buildings=[tower]), _cfg(robots=1))
    _, _, _, info = env.step(np.array([[80.0, 80.0]], np.float64))
    rb = env.state.robots[0]
    assert rb.target_xy is None and rb.path == [] and rb.target_token_type == TOKEN_HOLD
    assert [e.payload["robot"] for e in info["events_this_step"] if e.kind == "unreachable"] == [0]


def test_the_waypoint_is_what_the_metrics_call_the_assigned_point():
    """`assigned` drives the redundancy metric, so it has to be the waypoint itself."""
    env = DisasterEnv(_flat_scene(), _cfg(robots=2, team_terms=True))
    p = env.state.robots[0].pos.copy()
    _, _, _, i1 = env.step(np.stack([p + np.array([30.0, 0.0]), p + np.array([31.0, 0.0])]))
    assert i1["metrics"]["redundancy"] == 1.0        # 1 of 1 decision has two targets within 20 m
    _, _, _, i2 = env.step(np.stack([p + np.array([90.0, 0.0]), p + np.array([0.0, 90.0])]))
    assert i2["metrics"]["redundancy"] == 0.5        # ... and 1 of 2 after they split up


def test_a_waypoint_episode_records_no_visit_and_owes_no_revisit():
    """A waypoint names no map item: nothing to record and nothing to charge (CONTRACTS.md 6)."""
    env = DisasterEnv(_small_scene(), _cfg(robots=3, t_max=120.0, team_terms=True))
    pol = make_policy("lawnmower")
    obs = env.reset(0)
    pol.reset(0)
    while True:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        if done:
            break
    assert env.visits == [] and info["metrics"]["visits"] == 0
    assert info["metrics"]["intentional_revisits"] == 0
    assert not any(rb.target_id != -1 for rb in env.state.robots)


def test_the_two_action_shapes_do_not_collide():
    env = DisasterEnv(_flat_scene(), _cfg(robots=2))
    with pytest.raises(ValueError):
        env.step(np.zeros((3, 2), np.float64))            # wrong robot count
    with pytest.raises(ValueError):
        env.step(np.zeros((2, 2), np.int64))              # int [n, 2] is 4 token actions, not 2
    env.step(np.zeros(2, np.int64))                       # ... and the int path still works


def test_the_int_action_path_is_unchanged_by_the_waypoint_branch():
    """Regression for the additive branch in `step`: the reference numbers were recorded from
    these same episodes before waypoint actions existed, on the same scene and seed."""
    ref = {"nearest_frontier": (-0.7858005358275456, 0, 0.8382825040128411, 1788.63541438418, 0),
           "ray_follower": (0.9157588113704965, 2, 0.8007624398073836, 2229.853798908589, 5)}
    for name, (reward, found, cov, dist, visits) in ref.items():
        cfg = EnvConfig()
        cfg.robot.n_robots, cfg.t_max_s = 3, 150.0
        env = DisasterEnv(make_synthetic_scene(1, region_m=(200.0, 200.0), n_casualties=6,
                                               n_bystanders=3), cfg, seed=7)
        pol = make_policy(name, queries=cfg.rayfronts.queries, seed=7)
        obs = env.reset(7)
        pol.reset(7)
        total = 0.0
        while True:
            a = pol.act(obs, env.state)
            assert a.dtype.kind == "i"
            obs, r, done, info = env.step(a)
            total += float(r)
            if done:
                break
        assert total == pytest.approx(reward, abs=1e-9), name
        assert info["found_total"] == found and len(env.visits) == visits, name
        assert info["coverage"] == pytest.approx(cov, abs=1e-12), name
        assert sum(rb.dist_travelled for rb in env.state.robots) == pytest.approx(dist, abs=1e-6)

# ---- policies -----------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(n for n in POLICIES
                                        if not getattr(POLICIES[n], "waypoint_policy", False)))
def test_baselines_never_select_a_masked_token(name):
    env = DisasterEnv(_small_scene(), _cfg(t_max=60.0))
    pol = make_policy(name)
    obs = env.state.last_obs
    for _ in range(12):
        a = pol.act(obs, env.state)
        assert a.shape == (3,) and a.dtype.kind == "i"
        for r, k in enumerate(a):
            assert obs.token_mask[r, k], (name, r, k)
        obs, _, done, _ = env.step(a)
        if done:
            break


@pytest.mark.parametrize("name", sorted(n for n in POLICIES
                                        if getattr(POLICIES[n], "waypoint_policy", False)))
def test_waypoint_baselines_return_waypoints(name):
    """The other action shape: float [n_robots, 2] world points, nan = hold (CONTRACTS.md 6)."""
    env = DisasterEnv(_small_scene(), _cfg(t_max=60.0))
    pol = make_policy(name)
    obs = env.state.last_obs
    x0, y0, x1, y1 = env.raster.region
    for _ in range(12):
        a = pol.act(obs, env.state)
        assert a.shape == (3, 2) and a.dtype.kind == "f"
        fin = np.isfinite(a).all(axis=1)
        assert (np.isfinite(a).all(axis=1) == np.isfinite(a).any(axis=1)).all()  # no half rows
        assert ((a[fin, 0] >= x0) & (a[fin, 0] <= x1)).all()
        assert ((a[fin, 1] >= y0) & (a[fin, 1] <= y1)).all()
        obs, _, done, _ = env.step(a)
        if done:
            break


def test_privileged_flags():
    assert make_policy("oracle").privileged
    assert not any(make_policy(n).privileged
                   for n in ("random", "ray_follower", "segment_seeker", "nearest_frontier"))
    with pytest.raises(ValueError):
        make_policy("oracle").act(DisasterEnv(_small_scene(), _cfg()).state.last_obs, None)
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("nope")


# ---- metrics -------------------------------------------------------------------------------------
def test_metrics_are_monotonic_and_match_the_event_log():
    env = DisasterEnv(_small_scene(1), _cfg(t_max=300.0), seed=1)
    obs = env.state.last_obs
    pol = make_policy("ray_follower")
    prev = None
    while True:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        m = info["metrics"]
        if prev is not None:
            assert m["n_found"] >= prev["n_found"]
            assert m["frac_found"] >= prev["frac_found"]
            assert m["finds_auc"] >= prev["finds_auc"]
            assert m["dist_total"] >= prev["dist_total"]
        prev = m
        if done:
            break
    ev = env.state.events
    assert m["n_found"] == sum(1 for e in ev if e.kind == "found" and e.payload["casualty"])
    assert m["coverage_end"] == pytest.approx(env.coverage())
    assert env.coverage() >= env.state.observed.mean() - 1e-9
    assert m["dist_per_find"] == pytest.approx(m["dist_total"] / max(1, m["n_found"]))
    assert 0.0 <= m["finds_auc"] <= 1.0 and 0.0 <= m["redundancy"] <= 1.0
    assert m["time_to_first"] <= m["time_to_half"] <= m["time_to_all"]
    assert m["dist_total"] <= 3 * env.cfg.robot.speed_mps * env.state.t + 1e-6
    assert sum(m["found_by_container"].values()) == m["n_found"]
    assert sum(m["n_by_container"].values()) == info["n_casualties"]
    for k, v in m["found_by_container"].items():
        assert 0 <= v <= m["n_by_container"][k]


def test_found_by_container_matches_the_scene():
    env = DisasterEnv(_small_scene(1), _cfg(t_max=200.0), seed=1)
    obs = env.state.last_obs
    pol = make_policy("oracle")
    while True:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        if done:
            break
    want = {}
    for h in env.scene.humans:
        if h.role == "casualty":
            want[h.container] = want.get(h.container, 0) + 1
    got = info["metrics"]["n_by_container"]
    assert {k: v for k, v in got.items() if v} == want


def test_metrics_defaults_when_nothing_found():
    env = DisasterEnv(_small_scene(), _cfg(t_max=20.0))
    _, info, _ = _run(env, make_policy("random"))
    m = info["metrics"]
    if m["n_found"] == 0:
        assert m["time_to_first"] == m["time_to_all"] == env.cfg.t_max_s


# ---- vec env --------------------------------------------------------------------------------------
def test_vec_env_stacks_pads_and_auto_resets():
    c3, c1 = _cfg(3, 60.0), _cfg(1, 20.0)
    envs = [DisasterEnv(_small_scene(0), c3, seed=0), DisasterEnv(_small_scene(1), c1, seed=1)]
    ve = VecEnv(envs)
    o = ve.reset([0, 1])
    assert o.tokens.shape[:2] == (2, 3)
    assert list(o.robot_mask.sum(axis=1)) == [3, 1]
    assert not o.tokens[1, 1:].any() and not o.token_mask[1, 1:].any()
    assert o.env_obs(1).tokens.shape[0] == 1
    pols = [make_policy("ray_follower"), make_policy("ray_follower")]
    finals = 0
    for _ in range(6):
        a = np.zeros((2, ve.R), int)
        for e in range(2):
            act = pols[e].act(o.env_obs(e), envs[e].state)
            a[e, :act.size] = act
        o, r, d, infos = ve.step(a)
        for i, inf in enumerate(infos):
            if d[i]:
                assert "final_info" in inf
                assert inf["final_info"]["metrics"]["n_decisions"] == 4
                finals += 1
                assert envs[i].state.t == 0.0             # auto-reset happened
    assert finals >= 1
    assert r.shape == (2,) and d.shape == (2,)


def test_vec_env_without_auto_reset():
    envs = [DisasterEnv(_small_scene(0), _cfg(2, 20.0), seed=0)]
    ve = VecEnv(envs, auto_reset=False)
    ve.reset([0])
    for _ in range(4):
        o, r, d, infos = ve.step(np.zeros((1, 2), int))
    assert d[0] and "final_info" not in infos[0]
    assert envs[0].state.t == pytest.approx(20.0)
    with pytest.raises(ValueError):
        ve.step(np.zeros(2, int))


# ---- config guards ----------------------------------------------------------------------------------
def test_bad_config_is_rejected():
    c = _cfg()
    c.sensor.depth_limit_m = 200.0
    with pytest.raises(ValueError, match="depth_limit_m"):
        DisasterEnv(_small_scene(), c)
    c = _cfg()
    c.robot.flight_alt_m = 60.0
    with pytest.raises(ValueError, match="flight_alt_m"):
        DisasterEnv(_small_scene(), c)


def test_humans_on_the_region_border_and_coarse_cells():
    sc = Scene(meta=Meta(region=(0.0, 0.0, 60.0, 60.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.5}),
               roads=[Road(id="r", rect=(0.0, 0.0, 60.0, 8.0))],
               humans=[Human(id="h0", pos=(0.0, 0.0, 0.0), role="casualty", pose="prone"),
                       Human(id="h1", pos=(60.0, 60.0, 0.0), role="casualty", pose="prone"),
                       Human(id="h2", pos=(0.0, 60.0, 0.0), role="bystander", pose="standing")],
               robots_spawn=[(30.0, 4.0, 20.0)])
    cfg = _cfg(robots=2, t_max=60.0)
    cfg.raster.cell_m = 5.0
    env = DisasterEnv(sc, cfg, seed=0)
    assert env.raster.shape == (12, 12)
    obs, info, _ = _run(env, make_policy("oracle"))
    assert np.isfinite(obs.tokens).all()
    assert info["n_casualties"] == 2


def test_all_casualties_occluded_still_runs():
    sc = _small_scene()
    for h in sc.humans:
        if h.role == "casualty":
            h.container, h.visibility = "rubble", "occluded"
    env = DisasterEnv(sc, _cfg(t_max=60.0))
    obs, info, _ = _run(env, make_policy("ray_follower"))
    assert 0 <= info["found_total"] <= info["n_casualties"]
    assert np.isfinite(obs.tokens).all()


def test_disk_sensor_mode_runs():
    cfg = _cfg(t_max=30.0)
    cfg.sensor.mode = "disk"
    env = DisasterEnv(_small_scene(), cfg)
    obs, info, _ = _run(env, make_policy("ray_follower"))
    assert info["coverage"] > 0.0
    assert np.isfinite(obs.tokens).all()


def test_potential_shaping_is_policy_relevant_but_optional():
    cfg = _cfg(t_max=40.0)
    cfg.reward.potential_shaping = True
    env = DisasterEnv(_small_scene(), cfg, seed=1)
    _, r, _, _ = env.step(np.zeros(3, int))
    assert np.isfinite(r)
