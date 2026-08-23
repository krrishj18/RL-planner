"""Edge cases from CONTRACTS.md 11: degenerate scenes, extreme sizes, determinism, guards."""
import math
import time
from pathlib import Path

import numpy as np
import pytest

from rlplanner.scene.schema import (Building, DamageField, Human, Meta, Prop, Road, Scene,
                                    make_synthetic_scene)
from rlplanner.sim import geometry as G
from rlplanner.sim.baselines import make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.state import TOKEN_HOLD
from rlplanner.sim.vec_env import VecEnv

SCENES = Path(__file__).resolve().parents[1] / "data" / "scenes"


def _cfg(robots=2, t_max=40.0, cell=2.0):
    c = EnvConfig()
    c.robot.n_robots = robots
    c.t_max_s = t_max
    c.raster.cell_m = cell
    return c


def _run(env, policy, n=None):
    obs = env.state.last_obs
    k = 0
    while True:
        obs, r, done, info = env.step(policy.act(obs, env.state))
        k += 1
        if done or (n is not None and k >= n):
            return obs, info, k


# ---- degenerate scenes -------------------------------------------------------------------------
def test_empty_scene_runs_and_terminates():
    sc = Scene(meta=Meta(region=(0.0, 0.0, 60.0, 60.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}))
    env = DisasterEnv(sc, _cfg(robots=1, t_max=20.0))
    assert env.state.robots[0].pos is not None
    obs, info, k = _run(env, make_policy("ray_follower"))
    assert k == 4 and info["n_casualties"] == 0
    assert np.isfinite(obs.tokens).all()
    assert env.observable.all()


def test_severity_zero_scene_has_no_damage_and_still_runs():
    sc = make_synthetic_scene(3, region_m=(120.0, 120.0), severity=0.0, n_casualties=4,
                              n_bystanders=2)
    env = DisasterEnv(sc, _cfg(t_max=30.0))
    assert float(env.raster.damage.max()) == pytest.approx(0.0)
    obs, info, _ = _run(env, make_policy("ray_follower"))
    assert np.isfinite(obs.tokens).all() and 0.0 <= info["coverage"] <= 1.0


def test_scene_with_only_bystanders_pays_only_time():
    sc = make_synthetic_scene(0, region_m=(80.0, 80.0), n_casualties=0, n_bystanders=4)
    env = DisasterEnv(sc, _cfg(robots=1, t_max=20.0))
    _, info, k = _run(env, make_policy("ray_follower"))
    assert k == 4 and info["n_casualties"] == 0 and info["found_total"] == 0
    assert env.state.cum_reward == pytest.approx(-4 * env.cfg.reward.time_cost)


# ---- geometry guards ----------------------------------------------------------------------------
def test_target_outside_the_region_is_clipped_to_it():
    env = DisasterEnv(make_synthetic_scene(0, region_m=(80.0, 80.0)), _cfg(robots=1, t_max=20.0))
    rb = env.state.robots[0]
    rb.target_xy = np.array([1e4, -1e4])
    ev = []
    env._plan(rb, ev, 0.0)
    x0, y0, x1, y1 = env.raster.region
    assert rb.path, "a clipped in-region goal must still be plannable"
    for x, y in rb.path:
        assert x0 <= x <= x1 and y0 <= y <= y1


def test_unreachable_target_makes_the_robot_hold_and_emits_the_event():
    walls = [Building(id=f"w{k}", center=(cx, cy), size=(sx, sy), category="highrise")
             for k, (cx, cy, sx, sy) in enumerate([(40.0, 25.0, 30.0, 2.0), (40.0, 55.0, 30.0, 2.0),
                                                   (25.0, 40.0, 2.0, 30.0), (55.0, 40.0, 2.0, 30.0)])]
    sc = Scene(meta=Meta(region=(0.0, 0.0, 80.0, 80.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
               buildings=walls, robots_spawn=[(40.0, 40.0, 25.0)])
    env = DisasterEnv(sc, _cfg(robots=1, t_max=20.0))
    rb = env.state.robots[0]
    rb.target_xy = np.array([5.0, 5.0])              # outside the sealed courtyard
    rb.target_token_type = 1
    ev = []
    env._plan(rb, ev, 0.0)
    assert rb.target_xy is None and rb.path == []
    assert [e.kind for e in ev] == ["unreachable"]
    assert rb.target_token_type == TOKEN_HOLD
    o = env.state.last_obs
    inside = env.planner.label_at(*env.raster.xy_to_ij(40.0, 40.0))
    for k in np.flatnonzero(o.token_mask[0]):
        if np.isfinite(o.token_xy[0, k]).all():
            assert env.planner.label_at(*env.raster.xy_to_ij(*o.token_xy[0, k])) == inside


def test_frustum_edges_of_the_default_wedge():
    """pitch -50 with vfov 80 spans [-90, -10] deg: nadir in, anything shallower than -10 out."""
    cfg = EnvConfig()
    pitch, vfov = math.radians(cfg.sensor.pitch_deg), math.radians(cfg.sensor.vfov_deg)
    hfov = math.radians(cfg.sensor.hfov_deg)
    cam = np.array([0.0, 0.0, 0.0])

    def at(el_deg, az_deg=0.0):
        el, az = math.radians(el_deg), math.radians(az_deg)
        return [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]

    t = np.array([at(-90.0), at(-89.0), at(-11.0), at(-9.0), at(-50.0), at(0.0), at(10.0)])
    got = list(G.in_frustum(cam, 0.0, pitch, hfov, vfov, t))
    assert got == [True, True, True, False, True, False, False]
    az = np.array([at(-50.0, 44.0), at(-50.0, 46.0), at(-50.0, -44.0), at(-50.0, -46.0)])
    assert list(G.in_frustum(cam, 0.0, pitch, hfov, vfov, az)) == [True, False, True, False]
    assert bool(G.in_frustum(cam, 0.0, pitch, hfov, vfov, np.array([[0.0, 0.0, -1.0]]))[0])


def test_los_is_blocked_by_a_building_but_not_by_a_car():
    sc = Scene(meta=Meta(region=(-60.0, -60.0, 60.0, 60.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
               buildings=[Building(id="b", center=(0.0, 20.0), size=(20.0, 6.0),
                                   category="highrise")],
               props=[Prop(id="p", category="bench", center=(0.0, -20.0), size=(4.0, 2.0))])
    from rlplanner.sim.raster import rasterize
    r = rasterize(sc, 2.0)
    cam = np.array([0.0, 0.0, 25.0])
    tall = G.los_visible(r.height, r.cell_m, np.array(r.origin), cam,
                         np.array([[0.0, 40.0, 0.5]]), 0.5, 0.2)[0]
    low = G.los_visible(r.height, r.cell_m, np.array(r.origin), cam,
                        np.array([[0.0, -40.0, 0.5]]), 0.5, 0.2)[0]
    assert not tall and low


def test_astar_path_is_cached_until_the_target_changes():
    env = DisasterEnv(make_synthetic_scene(0, region_m=(120.0, 120.0)), _cfg(robots=1, t_max=60.0))
    o = env.state.last_obs
    ks = [int(k) for k in np.flatnonzero(o.token_mask[0] & (o.token_type[0] != TOKEN_HOLD))]
    assert len(ks) >= 2
    ras, pl = env.raster, env.planner
    start = ras.xy_to_ij(*env.state.robots[0].pos)
    goals = [ras.xy_to_ij(*o.token_xy[0, k]) for k in ks[:2]]
    p0 = pl.path(start, goals[0])
    assert p0 is pl.path(start, goals[0])                     # same key -> cached object
    n1 = len(pl.cache)
    assert pl.path(start, goals[1]) is not p0                 # new target -> new plan
    assert len(pl.cache) == n1 + 1
    rb = env.state.robots[0]
    ev = []
    for k in ks[:2]:                                          # the env re-plans on every new target
        rb.target_xy = np.asarray(o.token_xy[0, k], np.float64)
        env._plan(rb, ev, 0.0)
        assert rb.path and np.allclose(rb.path[-1], env.raster.clip_xy(*rb.target_xy))
    assert not ev


# ---- sizes ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cell", [1.0, 2.0, 5.0])
def test_cell_sizes(cell):
    env = DisasterEnv(make_synthetic_scene(0, region_m=(80.0, 80.0)), _cfg(robots=2, t_max=20.0,
                                                                          cell=cell))
    assert env.raster.cell_m == cell
    obs, info, _ = _run(env, make_policy("ray_follower"))
    assert np.isfinite(obs.tokens).all()
    assert 0.0 <= info["coverage"] <= 1.0
    assert not (env.state.observed & ~env.observable).any()


def test_huge_region_stays_within_budget():
    sc = make_synthetic_scene(0, region_m=(1600.0, 1200.0))
    t0 = time.perf_counter()
    env = DisasterEnv(sc, _cfg(robots=3, t_max=50.0))
    build = time.perf_counter() - t0
    assert env.raster.shape == (600, 800)
    assert build < 10.0
    nbytes = (env.state.vox_feat_sum.nbytes + env.state.vox_cnt.nbytes
              + env.rf.vox_feat_sum.nbytes + env.rf.last_seen_t.nbytes)
    assert nbytes < 100e6
    t0 = time.perf_counter()
    obs, info, k = _run(env, make_policy("ray_follower"))
    assert (time.perf_counter() - t0) / k < 0.5
    assert np.isfinite(obs.tokens).all() and np.isfinite(obs.bev).all()


# ---- determinism / obs hygiene --------------------------------------------------------------------
def _fingerprint(env):
    st = env.state
    return (st.vox_feat_sum.copy(), st.vox_cnt.copy(), st.observed.copy(),
            np.array([r.pos for r in st.robots]), st.rays.feat.copy(),
            st.rays.feat_peak.copy(), st.rays.ids.copy(), st.seg_labels.copy(),
            st.human_found.copy(), float(st.cum_reward), float(st.t))


def test_same_seed_identical_after_40_decisions():
    sc = make_synthetic_scene(4, region_m=(120.0, 120.0))
    outs = []
    for _ in range(2):
        env = DisasterEnv(sc, _cfg(robots=3, t_max=400.0), seed=5)
        _run(env, make_policy("ray_follower", seed=5), n=40)
        outs.append(_fingerprint(env))
    for a, b in zip(*outs):
        assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def test_vec_env_is_deterministic_and_reports_final_info():
    def build():
        envs = [DisasterEnv(make_synthetic_scene(s, region_m=(100.0, 100.0)),
                            _cfg(robots=1 + s, t_max=60.0), seed=s) for s in (0, 1)]
        return VecEnv(envs), envs

    outs = []
    for _ in range(2):
        ve, envs = build()
        o = ve.reset([0, 1])
        pols = [make_policy("ray_follower", seed=0), make_policy("ray_follower", seed=1)]
        finals = 0
        for _ in range(40):
            a = np.zeros((2, ve.R), int)
            for e in range(2):
                act = pols[e].act(o.env_obs(e), envs[e].state)
                a[e, :act.size] = act
            o, r, d, infos = ve.step(a)
            finals += sum(1 for i, inf in enumerate(infos) if d[i] and "final_info" in inf)
        outs.append((o.tokens.copy(), o.token_mask.copy(), np.array([e.state.t for e in envs]),
                     finals))
    assert outs[0][3] >= 2
    for a, b in zip(outs[0], outs[1]):
        assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def test_obs_stays_finite_and_masked_slots_stay_empty_over_40_decisions():
    env = DisasterEnv(make_synthetic_scene(2, region_m=(160.0, 160.0)), _cfg(robots=3, t_max=400.0))
    pol = make_policy("ray_follower")
    obs = env.state.last_obs
    for _ in range(40):
        obs, r, done, info = env.step(pol.act(obs, env.state))
        assert np.isfinite(r)
        for a in (obs.tokens, obs.robot_feat, obs.bev):
            assert np.isfinite(a).all()
        nan = ~np.isfinite(obs.token_xy).all(axis=2)
        assert not obs.token_mask[nan].any()
        assert (obs.token_id[nan] == -1).all() and (obs.tokens[nan] == 0).all()
        if done:
            break


# ---- exported scenes --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["earthquake_0", "tornado_0"])
def test_exported_scene_runs(name):
    path = SCENES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not exported")
    sc = Scene.from_json(path)
    env = DisasterEnv(sc, _cfg(robots=3, t_max=50.0))
    obs, info, _ = _run(env, make_policy("ray_follower"))
    assert info["n_casualties"] > 0
    assert np.isfinite(obs.tokens).all()
    assert 0.0 <= info["coverage"] <= 1.0
    assert not (env.state.observed & ~env.observable).any()
