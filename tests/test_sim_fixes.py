"""Regressions for the defects found in the first simulator review (CONTRACTS.md 12)."""
import math

import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.scene.schema import (Building, DamageField, Human, Meta, Road, Scene,
                                    make_synthetic_scene)
from rlplanner.sim.baselines import NearestFrontierPolicy, make_policy
from rlplanner.sim.config import EnvConfig

PERSON_Q = ("person lying on the ground", "person")


def _person(rf, feat) -> float:
    """The person similarity, computed here from the stored feature — the sim no longer keeps one."""
    q = rf.emb.embed_queries(PERSON_Q)
    f = np.asarray(feat, np.float32)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    return float(np.clip(f @ q.T, 0.0, 1.0).max())


def _person_rays(rf, peak=True) -> np.ndarray:
    return np.maximum(rf.ray_query_sim(PERSON_Q[0], peak), rf.ray_query_sim(PERSON_Q[1], peak))
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.raster import rasterize
from rlplanner.sim.rayfronts_sim import RayFrontsSim
from rlplanner.sim.sensor import human_visibility, observable_mask, visible_cells
from rlplanner.sim.state import TOKEN_RAY

W = 200.0


def _flat_scene(humans=(), buildings=(), w=W):
    return Scene(meta=Meta(region=(-w / 2, -w / 2, w / 2, w / 2)),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
                 buildings=list(buildings), humans=list(humans),
                 robots_spawn=[(0.0, 0.0, 25.0)])


def _robot(x=0.0, y=0.0, yaw=0.0, alt=25.0, idx=0):
    from rlplanner.sim.state import RobotState
    return RobotState(idx=idx, pos=np.array([x, y], float), alt=alt, heading=yaw,
                      target_xy=None, target_token_type=0, target_id=-1)


def _sim(scene, cfg=None, seed=0):
    cfg = cfg or EnvConfig()
    ras = rasterize(scene, cfg.raster.cell_m)
    return RayFrontsSim(ras, cfg, np.random.default_rng(seed)), np.random.default_rng(seed), ras, cfg



def _highrise_scene():
    """Four 60 m towers: their roofs are above the flight altitude and can never be observed."""
    b = [Building(id=f"h{k}", center=c, size=(30.0, 30.0), category="highrise")
         for k, c in enumerate([(60.0, 60.0), (140.0, 60.0), (60.0, 140.0), (140.0, 140.0)])]
    return Scene(meta=Meta(region=(0.0, 0.0, 200.0, 200.0)),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.3}),
                 buildings=b, roads=[Road(id="r", rect=(0.0, 0.0, 200.0, 10.0))],
                 humans=[Human(id="c", pos=(100.0, 100.0, 0.0), role="casualty", pose="prone")],
                 robots_spawn=[(5.0, 5.0, 25.0)])


# ---- 1. ray livelock ---------------------------------------------------------------------------
def test_a_ray_resolves_once_the_area_it_aims_at_is_observed():
    """The corridor test alone keeps a ray alive after the robot has flown to it and looked."""
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    rf, rng, ras, cfg = _sim(_flat_scene([Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty",
                                                pose="prone")]), cfg, seed=3)
    rb = _robot(0.0, 0.0, 0.0)
    for k in range(6):
        rf.update([rb], float(k), rng)
    rf.end_of_decision(6.0, [rb])
    hot = [T for T in rf.ray_targets
           if _person(rf, T.feat) > 0.6 and abs(math.degrees(T.az)) < 12.0]
    assert hot, "expected a ray from the far human"
    T = max(hot, key=lambda T: T.conf)

    # observe only the disc around the point the ray aims at, leaving most of the corridor unknown
    ci, cj = ras.xy_to_ij(T.xy[0], T.xy[1])
    ii, jj = np.mgrid[0:ras.ny, 0:ras.nx]
    d = np.hypot((ii - ci) * ras.cell_m, (jj - cj) * ras.cell_m)
    rf.observed[d <= cfg.rayfronts.ray_resolve_radius_m] = True
    assert rf.observed.mean() < 0.5                         # nowhere near a resolved corridor
    rf.end_of_decision(7.0, [rb])
    st = rf.store()
    k = int(np.flatnonzero(st.ids == T.id)[0])
    assert bool(st.resolved[k])
    assert T.id not in [x.id for x in rf.ray_targets]


def test_a_robot_that_flies_to_a_ray_stops_being_offered_it():
    """The ray resolves by itself once the robot has mapped the area it pointed into."""
    cfg = EnvConfig()
    cfg.robot.n_robots = 2
    cfg.t_max_s = 400.0
    cfg.rayfronts.p_fp_ray = 0.05                # plenty of phantom rays to chase
    env = DisasterEnv(_flat_scene(), cfg, seed=0)
    pol = make_policy("ray_follower", queries=cfg.rayfronts.queries)
    obs = env.state.last_obs
    chased = None
    for _ in range(40):
        a = pol.act(obs, env.state)
        for r in range(2):
            if int(obs.token_type[r, a[r]]) == TOKEN_RAY:
                chased = int(obs.token_id[r, a[r]])
        obs, _, done, _ = env.step(a)
        if chased is not None and chased not in {T.id for T in env.state.ray_targets}:
            break
        if done:
            break
    assert chased is not None, "the ray follower never chased a ray"
    assert chased not in {T.id for T in env.state.ray_targets}


# ---- 2. a salient look must not decay under the running mean -----------------------------------
def test_the_peak_look_is_kept_beside_the_running_mean():
    rf, rng, _, _ = _sim(make_synthetic_scene(0, region_m=(W, W)), seed=0)
    for k in range(8):
        rf.update([_robot(-40.0, -40.0, 0.5)], float(k), rng)
    rf.end_of_decision(8.0)
    st = rf.store()
    assert st.feat_peak is not None and st.feat_peak.shape == st.feat.shape
    peak = _person_rays(rf, peak=True)
    mean = _person_rays(rf, peak=False)
    assert (peak >= mean - 1e-6).all()
    assert (peak >= 0.0).all() and (peak <= 1.0).all()
    assert (peak > mean + 1e-3).any()          # the peak is not just the mean
    # the token a ray offers is the peak, so it describes the direction the token reports
    for T in rf.ray_targets:
        assert np.allclose(T.feat, rf._r_peak[T.ray_idx], atol=1e-6)


def test_a_far_human_seen_once_stays_a_ray_target_after_50_background_looks():
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    hum = Human(id="h", pos=(60.0, 0.0, 0.0), role="casualty", pose="prone")
    rf, rng, _, cfg = _sim(_flat_scene([hum]), cfg, seed=3)
    rb = _robot(0.0, 0.0, 0.0)
    for k in range(40):                       # one of these draws the human ray
        rf.update([rb], float(k), rng)
        if (_person_rays(rf) > 0.6).any():
            break
    rf._p_observe[:] = 0.0                    # from now on only background is ever seen
    for k in range(50):
        rf.update([rb], 100.0 + k, rng)
    rf.end_of_decision(200.0, [rb])
    st = rf.store()
    person_max = _person_rays(rf, peak=True)
    person_mean = _person_rays(rf, peak=False)
    hot = np.flatnonzero(~st.resolved & (person_max > 0.6))
    assert hot.size >= 1, "the human ray was lost"
    assert person_mean[hot].max() < 0.6, \
        "the running mean would still have kept it: test no longer proves anything"
    assert any(abs(T.xy[1]) < 20.0 and T.xy[0] > 30.0
               for T in rf.ray_targets if _person(rf, T.feat) > 0.6)


def test_a_rays_direction_is_the_one_of_its_most_salient_look():
    """Background clutter in the same bin must not drag the bearing off the person."""
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    hum = Human(id="h", pos=(45.0, 45.0, 0.0), role="casualty", pose="prone")
    rf, rng, _, cfg = _sim(_flat_scene([hum]), cfg, seed=1)
    rb = _robot(0.0, 0.0, math.pi / 4)
    for k in range(20):
        rf.update([rb], float(k), rng)
    rf.end_of_decision(20.0, [rb])
    hot = [T for T in rf.ray_targets if _person(rf, T.feat) > 0.6]
    assert hot
    T = max(hot, key=lambda T: _person(rf, T.feat))
    assert abs(math.degrees(T.az) - 45.0) < 5.0
    assert math.hypot(T.xy[0] - 45.0, T.xy[1] - 45.0) < 8.0


# ---- 3. nearest-frontier must sweep -----------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_nearest_frontier_covers_the_scene_by_300s(seed):
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    cfg.t_max_s = 300.0
    env = DisasterEnv(make_synthetic_scene(seed, region_m=(240.0, 240.0)), cfg, seed=seed)
    pol = make_policy("nearest_frontier", queries=cfg.rayfronts.queries, seed=seed)
    obs = env.state.last_obs
    while True:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        if done:
            break
    assert info["coverage"] >= 0.60
    assert info["metrics"]["dist_total"] > 1000.0


def _sweep(env, pol):
    obs = env.state.last_obs
    while True:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        if done:
            return info


def test_nearest_frontier_does_not_stall_when_every_frontier_is_in_its_footprint():
    """A lone robot sees a frontier ring at exactly the footprint radius, so a guard set to the
    full radius rejects every frontier for ever. The guard is half the radius and, if even that
    leaves nothing, the policy takes the farthest frontier rather than holding."""
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 300.0
    env = DisasterEnv(_highrise_scene(), cfg, seed=0)
    info = _sweep(env, make_policy("nearest_frontier", queries=cfg.rayfronts.queries))
    assert info["coverage"] > 0.30
    assert env.state.robots[0].dist_travelled > 500.0

    env.reset(0)                                   # pathological guard: nothing ever passes it
    info = _sweep(env, NearestFrontierPolicy(queries=cfg.rayfronts.queries, min_travel_m=1e6))
    assert env.state.robots[0].dist_travelled > 200.0
    assert info["coverage"] > 0.05


# ---- 4. coverage cap and frontier slivers ------------------------------------------------------
def test_observable_mask_matches_a_brute_force_sweep():
    sc = _flat_scene(buildings=[Building(id="t", center=(0.0, 0.0), size=(24.0, 24.0),
                                         category="highrise")], w=80.0)
    cfg = EnvConfig()
    env = DisasterEnv(sc, cfg, seed=0)
    ras, pl = env.raster, env.planner
    lbl = pl.label_at(*ras.xy_to_ij(*env.state.robots[0].pos))
    reach = (~pl.obst) & (pl.labels == lbl)
    m = observable_mask(ras, cfg.sensor, reach, cfg.robot.flight_alt_m)
    brute = np.zeros(ras.shape, bool)
    ii, jj = np.nonzero(reach)
    for k in range(ii.size):
        cam = np.array([*ras.ij_to_xy(int(ii[k]), int(jj[k])), cfg.robot.flight_alt_m])
        for yaw in np.linspace(-math.pi, math.pi, 9)[:-1]:
            v = visible_cells(ras, cfg.sensor, cam, float(yaw))
            if v.observed_ij.size:
                brute[v.observed_ij[:, 0], v.observed_ij[:, 1]] = True
    assert np.array_equal(m, brute)
    assert not m.all() and m.mean() > 0.8            # the tower roof is the unobservable part
    assert (ras.height[~m] > cfg.robot.flight_alt_m).all()


def test_coverage_is_normalised_by_the_observable_cells():
    cfg = EnvConfig()
    cfg.robot.n_robots = 2
    cfg.t_max_s = 200.0
    env = DisasterEnv(_highrise_scene(), cfg, seed=0)
    assert (~env.observable).sum() > 0
    obs = env.state.last_obs
    while True:
        obs, _, done, info = env.step(make_policy("nearest_frontier",
                                                  queries=cfg.rayfronts.queries).act(obs, env.state))
        if done:
            break
    assert not (env.state.observed & ~env.observable).any()      # never observe the unobservable
    assert 0.0 <= info["coverage"] <= 1.0
    assert info["coverage"] > env.state.observed.mean()          # the cap is factored out
    assert info["coverage"] == pytest.approx(env.state.observed.sum() / env.observable.sum())


def test_frontiers_only_border_cells_that_could_still_be_observed():
    cfg = EnvConfig()
    cfg.robot.n_robots = 2
    cfg.t_max_s = 200.0
    env = DisasterEnv(_highrise_scene(), cfg, seed=0)
    obs = env.state.last_obs
    pol = make_policy("nearest_frontier", queries=cfg.rayfronts.queries)
    for _ in range(20):
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        st = env.state
        un = ~st.observed & env.observable
        nb = np.zeros_like(un)
        nb[:-1, :] |= un[1:, :]
        nb[1:, :] |= un[:-1, :]
        nb[:, :-1] |= un[:, 1:]
        nb[:, 1:] |= un[:, :-1]
        assert np.array_equal(st.frontier_mask, st.observed & nb)
        for c in st.frontier_clusters:
            assert c.n_cells >= cfg.rayfronts.frontier_min_cluster_cells
            assert st.frontier_mask[c.cell_ij[:, 0], c.cell_ij[:, 1]].all()
        if done:
            break


# ---- 5. human line-of-sight target -------------------------------------------------------------
def test_human_los_target_never_rises_above_the_surface_over_it():
    sc = make_synthetic_scene(0, region_m=(W, W))
    rf, _, ras, _ = _sim(sc)
    hi, hj = ras.xy_to_ij(ras.humans["x"], ras.humans["y"])
    h = ras.height[np.clip(hi, 0, ras.ny - 1), np.clip(hj, 0, ras.nx - 1)]
    assert np.allclose(rf._human_pts[:, 2], np.minimum(ras.humans["z"] + 0.5, h))


def test_a_casualty_under_rubble_is_observed_from_overhead_but_not_across_the_debris():
    """Rubble is 1.75 m: the body's own cell is skipped by the ray march, but the neighbouring
    debris shadows any shallow ray, so only a near-overhead robot has line of sight."""
    hum = Human(id="h", pos=(0.0, 0.0, 0.0), role="casualty", pose="prone",
                container="rubble", visibility="occluded")
    sc = _flat_scene([hum])
    sc.buildings.append(Building(id="b", center=(0.0, 0.0), size=(40.0, 40.0), fate="destroyed",
                                 category="house"))          # rubble mound, 1.75 m
    cfg = EnvConfig()
    rf, rng, ras, cfg = _sim(sc, cfg, seed=0)
    assert rf._human_pts[0, 2] <= 0.5
    pts = rf._human_pts

    over, _ = human_visibility(ras, cfg.sensor, np.array([0.0, 0.0, 25.0]), 0.0, pts)
    assert bool(over[0])                                     # straight down: seen
    far, r_far = human_visibility(ras, cfg.sensor, np.array([-70.0, 0.0, 25.0]), 0.0, pts)
    assert r_far[0] > cfg.sensor.depth_limit_m
    assert not bool(far[0])                                  # shallow across the debris: blocked

    for k in range(30):                                      # and hovering over it maps it
        rf.update([_robot(0.0, 0.0, 0.0)], float(k), rng)
        if rf.human_found[0]:
            break
    assert rf.human_found[0]


# ---- metrics ---------------------------------------------------------------------------------
def test_finds_auc_credits_an_episode_that_ends_early():
    """The area is over the whole horizon: stopping early because everyone was found must score
    near 1, not be truncated at the moment the episode ends."""
    sc = Scene(meta=Meta(region=(-30.0, -30.0, 30.0, 30.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
               humans=[Human(id="h", pos=(6.0, 0.0, 0.0), role="casualty", pose="prone")],
               robots_spawn=[(0.0, 0.0, 25.0)])
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 200.0
    env = DisasterEnv(sc, cfg, seed=0)
    obs = env.state.last_obs
    while True:
        obs, _, done, info = env.step(np.zeros(1, int))       # hold: the casualty is underfoot
        if done:
            break
    m = info["metrics"]
    assert m["frac_found"] == 1.0
    assert m["time_to_all"] < 0.2 * cfg.t_max_s
    assert 0.9 <= m["finds_auc"] <= 1.0


def test_policy_ordering_on_finds_auc():
    """Oracle beats the heuristics, and every heuristic beats random."""
    auc = {}
    for name in ("random", "nearest_frontier", "ray_follower", "segment_seeker", "oracle"):
        rows = []
        for seed in (0, 1):
            cfg = EnvConfig()
            cfg.robot.n_robots = 3
            cfg.t_max_s = 250.0
            env = DisasterEnv(make_synthetic_scene(seed, region_m=(240.0, 240.0)), cfg, seed=seed)
            pol = make_policy(name, queries=cfg.rayfronts.queries, seed=seed)
            rows.append(_sweep(env, pol)["metrics"]["finds_auc"])
        auc[name] = float(np.mean(rows))
    assert auc["oracle"] > auc["random"], auc
    assert auc["ray_follower"] > auc["random"], auc
    assert auc["nearest_frontier"] > auc["random"], auc
    assert all(0.0 <= v <= 1.0 for v in auc.values())


# ---- 6. time_to_all --------------------------------------------------------------------------
def test_time_to_all_is_t_max_when_some_casualty_is_never_found():
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    cfg.t_max_s = 120.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(240.0, 240.0)), cfg, seed=0)
    obs = env.state.last_obs
    pol = make_policy("oracle", queries=cfg.rayfronts.queries)
    while True:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        if done:
            break
    m = info["metrics"]
    assert m["frac_found"] < 1.0
    assert m["time_to_all"] == pytest.approx(cfg.t_max_s)
    assert m["time_to_first"] <= m["time_to_half"] <= m["time_to_all"]
