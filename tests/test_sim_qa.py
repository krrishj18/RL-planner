"""Second simulator QA pass: found semantics, ray visibility rules, token feature bounds."""
import math

import numpy as np
import pytest

from rlplanner.scene.schema import (Building, DamageField, Human, Meta, Road, Scene,
                                    Vehicle, make_synthetic_scene)
from rlplanner.sim.baselines import POLICIES, make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.raster import rasterize
from rlplanner.sim.rayfronts_sim import RayFrontsSim
from rlplanner.sim.state import TOKEN_FIXED, RobotState, token_feature_names

W = 200.0
PERSON_Q = ("person lying on the ground", "person")
ALL_Q = ("person lying on the ground", "person", "collapsed building", "damaged building",
         "rubble", "overturned car", "car", "bus stop", "road", "tree", "house")


def _scene(humans=(), w=W, **kw):
    return Scene(meta=Meta(region=(-w / 2, -w / 2, w / 2, w / 2)),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
                 humans=list(humans), robots_spawn=[(0.0, 0.0, 25.0)], **kw)


def _robot(x=0.0, y=0.0, yaw=0.0, alt=25.0, idx=0):
    return RobotState(idx=idx, pos=np.array([x, y], float), alt=alt, heading=yaw,
                      target_xy=None, target_token_type=0, target_id=-1)


def _sim(scene, cfg=None, seed=0):
    # spurious person rays have their own test; everything here asserts on what a *real* bearing
    # reports, so one random false ray inside the tolerance must not decide the outcome
    if cfg is None:
        cfg = EnvConfig()
        cfg.rayfronts.p_fp_ray = 0.0
    ras = rasterize(scene, cfg.raster.cell_m)
    return RayFrontsSim(ras, cfg, np.random.default_rng(seed)), np.random.default_rng(seed), ras, cfg


def _look(rf, rng, robots, n, t0=0.0):
    for k in range(n):
        rf.update(robots, t0 + k, rng)
    rf.end_of_decision(t0 + n, robots)


def _cos(rf, feat, names):
    """Query similarity taken *by the test*, from the stored feature — the belief keeps none."""
    q = rf.emb.embed_queries(tuple(names))
    f = np.asarray(feat, np.float32)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    return np.clip(f @ q.T, 0.0, 1.0)


def _person(rf, feat):
    return float(_cos(rf, feat, PERSON_Q).max())


def _argmax_query(rf, feat, names=ALL_Q):
    return names[int(np.argmax(_cos(rf, feat, names)))]


def _ray_person(rf):
    return np.maximum(rf.ray_query_sim(PERSON_Q[0]), rf.ray_query_sim(PERSON_Q[1]))


def _forward(rf, tol_deg=10.0):
    """The ray aimed most nearly along +x — picked by geometry, so no query biases the choice."""
    c = [T for T in rf.ray_targets if abs(math.degrees(T.az)) < tol_deg]
    return min(c, key=lambda T: (abs(T.az), -T.conf)) if c else None


# ---- 1. found semantics -------------------------------------------------------------------------
def test_two_bodies_in_one_cell_are_counted_against_their_own_rows():
    """One cell, two poses: each body is its own voxel observation, so neither is credited with
    the other's row and the cell's hit count covers both."""
    sc = _scene([Human(id="a", pos=(0.3, 0.3, 0.0), role="casualty", pose="prone"),
                 Human(id="b", pos=(0.7, 0.7, 0.0), role="casualty", pose="standing")])
    rf, rng, ras, cfg = _sim(sc)
    assert tuple(rf._human_ij[0]) == tuple(rf._human_ij[1])          # really the same cell
    assert rf._human_row[0] != rf._human_row[1]
    _look(rf, rng, [_robot(0.0, -8.0, math.pi / 2)], 30)
    i, j = rf._human_ij[0]
    assert rf.human_hits[0] > 0 and rf.human_hits[1] > 0
    assert rf.human_found.all()
    # every hit is one voxel observation of the cell carrying that body's own row
    assert int(rf.human_hits.sum()) <= int(rf.vox_cnt[i, j])


def test_a_casualty_seen_once_is_not_found_and_the_reward_lands_once():
    sc = _scene([Human(id="a", pos=(3.0, 0.0, 0.0), role="casualty", pose="prone")])
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 300.0
    env = DisasterEnv(sc, cfg, seed=0)
    total, events = 0.0, 0
    while True:
        _, r, done, info = env.step(np.zeros(1, int))
        total += r + cfg.reward.time_cost
        events += sum(1 for e in info["events_this_step"] if e.kind == "found")
        if done:
            break
    assert events == 1
    assert total == pytest.approx(cfg.reward.casualty_reward)
    assert env.state.human_hits[0] > cfg.rayfronts.found_hits      # kept looking, no extra reward


@pytest.mark.parametrize("found_hits", [1, 5])
def test_found_lands_exactly_on_the_configured_hit_count(found_hits):
    cfg = EnvConfig()
    cfg.rayfronts.found_hits = found_hits
    rf, rng, ras, cfg = _sim(_scene([Human(id="a", pos=(3.0, 0.0, 0.0), role="casualty",
                                           pose="prone")]), cfg)
    rb = _robot(0.0, -8.0, math.pi / 2)
    hits_when_found = None
    for k in range(60):
        rf.update([rb], float(k), rng)
        if rf.human_found[0] and hits_when_found is None:
            hits_when_found = int(rf.human_hits[0])
    assert hits_when_found == found_hits


def test_a_bystander_never_rewards_and_never_counts():
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 100.0
    env = DisasterEnv(_scene([Human(id="b", pos=(3.0, 0.0, 0.0), role="bystander",
                                    pose="standing")]), cfg, seed=0)
    total = 0.0
    while True:
        _, r, done, info = env.step(np.zeros(1, int))
        total += r + cfg.reward.time_cost
        if done:
            break
    assert env.state.human_found[0]                 # it crosses the same threshold
    assert env.state.n_found == 0 and total == 0.0
    assert sum(info["metrics"]["found_by_container"].values()) == 0


@pytest.mark.parametrize("pos,rpos,yaw", [((-W / 2 + 0.5, 0.0), (-W / 2 + 8.5, 0.0), math.pi),
                                          ((W / 2 - 0.5, 0.0), (W / 2 - 8.5, 0.0), 0.0),
                                          ((0.0, -W / 2 + 0.5), (0.0, -W / 2 + 8.5), -math.pi / 2),
                                          ((0.0, W / 2 - 0.5), (0.0, W / 2 - 8.5), math.pi / 2)])
def test_a_casualty_on_the_region_border_is_found(pos, rpos, yaw):
    rf, rng, ras, cfg = _sim(_scene([Human(id="a", pos=(pos[0], pos[1], 0.0), role="casualty",
                                           pose="prone")]))
    i, j = rf._human_ij[0]
    assert i in (0, ras.ny - 1) or j in (0, ras.nx - 1)
    _look(rf, rng, [_robot(rpos[0], rpos[1], yaw)], 20)
    assert rf.human_found[0]


def test_found_events_carry_the_container_and_the_totals_agree():
    sc = _scene([Human(id="a", pos=(3.0, 0.0, 0.0), role="casualty", pose="prone",
                       container="vehicle", visibility="partial")],
                vehicles=[Vehicle(id="v", center=(3.0, 0.0), size=(4.5, 1.9), state="toppled")])
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 200.0
    env = DisasterEnv(sc, cfg, seed=0)
    payloads = []
    while True:
        _, _, done, info = env.step(np.zeros(1, int))
        payloads += [e.payload for e in info["events_this_step"] if e.kind == "found"]
        if done:
            break
    assert payloads and payloads[0]["container"] == "vehicle"
    assert payloads[0]["visibility"] == "partial" and payloads[0]["casualty"] is True
    m = info["metrics"]
    assert m["found_by_container"]["vehicle"] == 1
    assert sum(m["found_by_container"].values()) == m["n_found"] == info["found_total"]


def test_found_by_container_sums_to_n_found_on_a_populated_scene():
    cfg = EnvConfig()
    cfg.robot.n_robots = 4
    cfg.t_max_s = 200.0
    env = DisasterEnv(make_synthetic_scene(2, region_m=(240.0, 240.0)), cfg, seed=0)
    pol = make_policy("segment_seeker", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    while True:
        obs, _, done, info = env.step(pol.act(obs, env.state))
        if done:
            break
    m = info["metrics"]
    assert sum(m["found_by_container"].values()) == m["n_found"] == info["found_total"]
    assert sum(m["n_by_container"].values()) == info["n_casualties"]
    ij = env.rf._human_ij
    assert (env.state.human_hits <= env.rf.vox_cnt[ij[:, 0], ij[:, 1]]).all()


# ---- 2. ray visibility rules --------------------------------------------------------------------
def _car_person_sim(with_human, seed, n=250):
    hs = [Human(id="a", pos=(60.0, 0.0, 0.0), role="casualty", pose="prone",
                container="vehicle", visibility="partial")] if with_human else []
    sc = _scene([*hs], w=300.0,
                vehicles=[Vehicle(id="v", center=(60.0, 0.0), size=(4.5, 1.9), state="toppled")])
    rf, rng, ras, cfg = _sim(sc, seed=seed)
    _look(rf, rng, [_robot(0.0, 0.0, 0.0)], n)
    T = _forward(rf)
    return (None if T is None else
            (_argmax_query(rf, T.feat), _person(rf, T.feat)))


def test_a_far_human_in_a_toppled_car_adds_no_person_similarity():
    """The container owns the bearing; a `partial` body raises nothing from afar."""
    with_h = [_car_person_sim(True, s) for s in range(6)]
    without = [_car_person_sim(False, s) for s in range(6)]
    assert all(x is not None for x in with_h + without)
    assert all(x[0] in ("overturned car", "car", "road") for x in with_h)
    lift = np.mean([x[1] for x in with_h]) - np.mean([x[1] for x in without])
    assert lift < 0.05, f"person-similarity lift from a hidden body: {lift:.3f}"


def test_a_far_human_in_a_damaged_building_reads_as_the_building():
    sc = _scene([Human(id="a", pos=(60.0, 0.0, 0.0), role="casualty", pose="prone",
                       container="building", visibility="partial")], w=300.0,
                buildings=[Building(id="b", center=(60.0, 0.0), size=(20.0, 20.0), fate="damaged",
                                    category="house")])
    rf, rng, ras, cfg = _sim(sc)
    _look(rf, rng, [_robot(0.0, 0.0, 0.0)], 120)
    T = _forward(rf)
    assert T is not None
    assert _argmax_query(rf, T.feat) in ("damaged building", "collapsed building", "house")
    assert _person(rf, T.feat) < 0.6
    assert rf.human_hits[0] == 0


def test_an_open_human_beyond_the_visual_range_contributes_nothing():
    rf, rng, ras, cfg = _sim(_scene([Human(id="a", pos=(120.0, 0.0, 0.0), role="casualty",
                                           pose="prone")], w=300.0))
    _look(rf, rng, [_robot(0.0, 0.0, 0.0)], 120)
    assert rf.human_hits[0] == 0
    am = [_argmax_query(rf, f) for f in rf._r_peak[:rf.n_rays]]
    assert not set(am) & set(PERSON_Q)


def test_the_depth_limit_splits_voxel_from_ray_consistently():
    """Just inside the limit the body is a voxel hit and raises no ray; just outside, the reverse."""
    cfg = EnvConfig()
    h = math.sqrt(cfg.sensor.depth_limit_m ** 2 - cfg.robot.flight_alt_m ** 2)
    inside, outside = [], []
    for off, bucket in ((-1.0, inside), (1.0, outside)):
        rf, rng, ras, c = _sim(_scene([Human(id="a", pos=(h + off, 0.0, 0.0), role="casualty",
                                             pose="prone")], w=300.0))
        _look(rf, rng, [_robot(0.0, 0.0, 0.0)], 40)
        T = _forward(rf, 6.0)
        bucket.append((int(rf.human_hits[0]), None if T is None else _person(rf, T.feat)))
    assert inside[0][0] > 0 and (inside[0][1] is None or inside[0][1] < 0.6)
    assert outside[0][0] == 0 and outside[0][1] is not None and outside[0][1] > 0.6


def test_without_false_positives_no_ray_argmax_is_a_person_query():
    cfg = EnvConfig()
    cfg.rayfronts.p_fp_ray = 0.0
    sc = make_synthetic_scene(3, region_m=(240.0, 240.0))
    sc.humans = [h for h in sc.humans if h.visibility != "open"]
    rf, rng, ras, cfg = _sim(sc, cfg)
    rbs = [_robot(-60.0, -60.0, 0.6), _robot(40.0, 20.0, 2.5)]
    _look(rf, rng, rbs, 150)
    am = [_argmax_query(rf, f) for f in rf._r_peak[:rf.n_rays]]
    assert not set(am) & set(PERSON_Q)
    assert rf.n_fp_rays == 0


# ---- 3. ray persistence -------------------------------------------------------------------------
def test_a_ray_stays_live_while_the_area_it_aims_at_is_unknown():
    rf, rng, ras, cfg = _sim(_scene([Human(id="a", pos=(60.0, 0.0, 0.0), role="casualty",
                                           pose="prone")], w=300.0))
    rb = _robot(0.0, 0.0, 0.0)
    _look(rf, rng, [rb], 10)
    ids = {T.id for T in rf.ray_targets}
    assert ids
    for d in range(20):                       # keep looking, never fly there
        _look(rf, rng, [rb], 5, t0=10.0 + 5 * d)
        assert ids <= {T.id for T in rf.ray_targets}


def test_compaction_keeps_ids_and_rebuilds_the_bin_key_map():
    rf, rng, ras, cfg = _sim(make_synthetic_scene(0, region_m=(240.0, 240.0)))
    rbs = [_robot(-80.0 + 10 * i, -80.0, 0.4 * i, idx=i) for i in range(6)]
    for k in range(30):
        for rb in rbs:
            rb.pos[0] += 1.5
            rb.heading += 0.3
        rf.update(rbs, float(k), rng)
    rf.end_of_decision(30.0, rbs)
    n = rf.n_rays
    live = ~rf._r_res[:n]
    live_ids = set(rf._r_ids[:n][live].tolist())
    keys = {int(i): rf._r_keyof[k] for k, i in enumerate(rf._r_ids[:n]) if live[k]}
    assert not live.all(), "nothing resolved: the test would not exercise compaction"
    rf.compact()
    n2 = rf.n_rays
    assert set(rf._r_ids[:n2].tolist()) == live_ids
    assert all(rf._ray_key[keys[int(i)]] == k for k, i in enumerate(rf._r_ids[:n2]))


# ---- 4. segments --------------------------------------------------------------------------------
def test_segments_partition_the_observed_map():
    rf, rng, ras, cfg = _sim(make_synthetic_scene(0, region_m=(240.0, 240.0)))
    rbs = [_robot(-60.0, -60.0, 0.5), _robot(30.0, 20.0, 2.0)]
    _look(rf, rng, rbs, 40)
    assert rf.segments
    lab = rf.seg_labels
    assert (lab[rf.observed] >= 0).all() and (lab[~rf.observed] == -1).all()
    # every offered segment is a real, non-empty, spatially connected piece of the observed map
    for S in rf.segments:
        assert rf.observed[S.ij] and lab[S.ij] >= 0
        assert S.n_cells == int((lab == lab[S.ij]).sum())
        assert S.n_cells >= 1 and abs(float(np.linalg.norm(S.feat)) - 1.0) < 1e-5
    ids = [S.id for S in rf.segments]
    assert len(ids) == len(set(ids))


def test_a_fully_observed_map_offers_no_frontiers_and_one_segment_family():
    sc = Scene(meta=Meta(region=(-12.0, -12.0, 12.0, 12.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.0}),
               roads=[Road(id="r", rect=(-12.0, -12.0, 24.0, 24.0))],
               robots_spawn=[(0.0, 0.0, 25.0)])
    cfg = EnvConfig()
    cfg.sensor.mode = "disk"
    rf, rng, ras, cfg = _sim(sc, cfg)
    _look(rf, rng, [_robot(0.0, 0.0, 0.0)], 30)
    assert rf.observed.all()
    assert not rf.frontier_mask.any() and rf.frontier_clusters == []
    assert rf.segments                        # one uniform road surface still segments into pieces
    assert (rf.seg_labels >= 0).all()


def test_the_segment_cap_and_the_belief_footprint_on_a_750x750_map():
    cfg = EnvConfig()
    cfg.raster.cell_m = 2.0
    cfg.robot.n_robots = 2
    env = DisasterEnv(make_synthetic_scene(0, region_m=(1500.0, 1500.0)), cfg, seed=0)
    assert env.raster.shape == (750, 750)
    rf = env.rf
    rng = np.random.default_rng(0)
    rf.observed[:] = True                       # worst case: every cell is in play
    rf.vox_cnt[:] = 3
    rf.vox_feat_sum[:] = rng.standard_normal(rf.vox_feat_sum.shape, dtype=np.float32)
    rf._seg_obs_at = -1
    rf._extract_segments(0.0, decision=0)
    assert len(rf.segments) <= 2 * cfg.tokens.k_segment
    belief = sum(a.nbytes for a in (rf.vox_feat_sum, rf.vox_cnt, rf.last_seen_t, rf.observed,
                                    rf.observable, rf.frontier_mask, rf.seg_labels))
    assert belief <= 90e6, f"belief is {belief / 1e6:.1f} MB"


def test_segmentation_sanity_two_blobs_one_uniform():
    """Two classes side by side must give two segments; one class must give one."""
    from rlplanner.sim.segments import segment_map
    from rlplanner.scene.schema import CLASS_ID
    rf, _, _, cfg = _sim(_scene())
    ny = nx = 120
    obs = np.ones((ny, nx), np.bool_)
    rng = np.random.default_rng(0)

    def belief(cls_grid, looks=3):
        f = np.zeros((ny, nx, rf.D), np.float32)
        for _ in range(looks):
            v = rf.class_emb[cls_grid] + rng.standard_normal((ny, nx, rf.D)).astype(np.float32) * 0.08
            f += v / np.linalg.norm(v, axis=-1, keepdims=True)
        return f

    road = np.full((ny, nx), CLASS_ID["road"])
    two = road.copy()
    two[:, nx // 2:] = CLASS_ID["building_intact"]
    k, mn = cfg.rayfronts.segment_scale, cfg.rayfronts.segment_min_cells
    lab_u, n_u = segment_map(belief(road), obs, k, mn)
    lab_t, n_t = segment_map(belief(two), obs, k, mn)
    assert n_u == 1, f"a uniform map segmented into {n_u} pieces"
    assert n_t == 2, f"two classes segmented into {n_t} pieces"
    assert set(np.unique(lab_t[:, : nx // 2 - 1])) != set(np.unique(lab_t[:, nx // 2 + 1:]))


def test_segments_never_contain_unobserved_cells():
    from rlplanner.sim.segments import segment_map
    rf, rng, ras, cfg = _sim(make_synthetic_scene(0, region_m=(240.0, 240.0)))
    _look(rf, rng, [_robot(-60.0, -60.0, 0.5)], 20)
    lab, n = segment_map(rf.vox_feat_sum, rf.observed, cfg.rayfronts.segment_scale,
                         cfg.rayfronts.segment_min_cells)
    assert (lab[~rf.observed] == -1).all()
    assert n > 0 and (lab[rf.observed] >= 0).all()


# ---- 5. tokens ----------------------------------------------------------------------------------
@pytest.mark.parametrize("cell,robots,policy", [(2.0, 3, "ray_follower"), (1.0, 2, "nearest_frontier")])
def test_every_token_feature_stays_in_range_over_200_decisions(cell, robots, policy):
    """The normalisers are nominal, so the builder saturates the unbounded ones."""
    cfg = EnvConfig()
    cfg.raster.cell_m = cell
    cfg.robot.n_robots = robots
    cfg.t_max_s = 2000.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(240.0, 240.0)), cfg, seed=0)
    pol = make_policy(policy, queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    lo = np.full(env.builder.F, np.inf)
    hi = np.full(env.builder.F, -np.inf)
    for _ in range(200):
        t = obs.tokens[obs.token_mask]
        assert np.isfinite(obs.tokens).all()
        if t.size:
            lo = np.minimum(lo, t.min(0))
            hi = np.maximum(hi, t.max(0))
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            obs = env.reset()
    names = token_feature_names(env.rf.D)
    bad = [(n, float(a), float(b)) for n, a, b in zip(names, lo, hi)
           if a < -1.5 - 1e-6 or b > 1.5 + 1e-6]
    assert not bad, bad


def test_token_xy_is_nan_exactly_on_the_empty_slots():
    """A slot with a candidate keeps its xy even when the candidate is unreachable (mask False):
    NaN marks *empty*, the mask marks *selectable*."""
    # a 23 m block: below the flight altitude (so its roof is observed and offers candidates) but
    # above the A* clearance, so those candidates are unreachable
    sc = Scene(meta=Meta(region=(0.0, 0.0, 120.0, 120.0)),
               damage_field=DamageField(kind="uniform", params={"inside": 0.3}),
               buildings=[Building(id="b", center=(60.0, 60.0), size=(40.0, 40.0), height_m=23.0,
                                   fate="damaged", category="midrise")],
               roads=[Road(id="r", rect=(0.0, 0.0, 120.0, 12.0))],
               robots_spawn=[(10.0, 6.0, 25.0)])
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 120.0
    env = DisasterEnv(sc, cfg, seed=0)
    pol = make_policy("nearest_frontier", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    seen_unreachable = False
    while True:
        for k in range(obs.tokens.shape[1]):
            empty = int(obs.token_id[0, k]) == -1 and k != 0
            finite = bool(np.isfinite(obs.token_xy[0, k]).all())
            assert finite != empty
            if finite and not obs.token_mask[0, k]:
                seen_unreachable = True
                si = env.raster.xy_to_ij(*env.state.robots[0].pos)
                gi = env.raster.xy_to_ij(*obs.token_xy[0, k])
                gi = (int(np.clip(gi[0], 0, env.raster.ny - 1)),
                      int(np.clip(gi[1], 0, env.raster.nx - 1)))
                assert env.planner.path(si, gi) is None
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            break
    assert seen_unreachable, "no unreachable candidate was ever offered"


def test_a_ray_token_aims_at_origin_plus_bearing_times_range():
    cfg = EnvConfig()
    env = DisasterEnv(make_synthetic_scene(0, region_m=(240.0, 240.0)), cfg, seed=0)
    pol = make_policy("ray_follower", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    x0, y0, x1, y1 = env.raster.region
    checked = 0
    for _ in range(30):
        for T in env.state.ray_targets:
            p = T.origin_xy + T.range_m * np.array([math.cos(T.az), math.sin(T.az)])
            if x0 + 1e-3 < p[0] < x1 - 1e-3 and y0 + 1e-3 < p[1] < y1 - 1e-3:
                assert np.allclose(p, T.xy, atol=1e-9)
                checked += 1
            assert (cfg.sensor.depth_limit_m - 1e-6 <= T.range_m
                    <= cfg.sensor.visual_range_m + 1e-6)
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            break
    assert checked > 20


@pytest.mark.parametrize("n_extra", [0, 6])
def test_set_queries_leaves_the_token_width_alone(n_extra):
    """The observation width is `TOKEN_FIXED + embedding_dim` whatever the mission asks for, so a
    trained network survives a query change."""
    base = list(EnvConfig().rayfronts.queries)
    names = tuple(base + list(ALL_Q[2: 2 + n_extra])) if n_extra else ("person", "road")
    env = DisasterEnv(make_synthetic_scene(0, region_m=(200.0, 200.0)), EnvConfig(), seed=0)
    for _ in range(5):
        env.step(np.zeros(env.n_robots, int))
    width = env.state.last_obs.tokens.shape[2]
    feat = env.rf.vox_feat_sum.copy()
    obs = env.set_queries(names)
    assert obs.tokens.shape[2] == width == TOKEN_FIXED + env.rf.D
    assert len(token_feature_names(env.rf.D)) == width
    assert int(obs.query_mask.sum()) == len(names)
    assert np.array_equal(feat, env.rf.vox_feat_sum)     # the belief did not move
    env.step(np.zeros(env.n_robots, int))       # and it keeps stepping


# ---- 6. env -------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(POLICIES))
def test_baselines_run_without_the_hold_token(name):
    """`include_hold=False` moves slot 0 to a frontier that can be masked: nothing may fall back
    to a literal index 0."""
    cfg = EnvConfig()
    cfg.tokens.include_hold = False
    cfg.robot.n_robots = 2
    cfg.t_max_s = 60.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(200.0, 200.0)), cfg, seed=0)
    assert env.k_tokens == (cfg.tokens.k_frontier + cfg.tokens.k_ray + cfg.tokens.k_segment
                            + cfg.tokens.k_visited)
    pol = make_policy(name, queries=cfg.rayfronts.queries, seed=0)
    waypoints = bool(getattr(pol, "waypoint_policy", False))
    obs = env.state.last_obs
    while True:
        a = pol.act(obs, env.state)
        if waypoints:                     # a waypoint names no token, so no mask to respect
            assert a.shape == (obs.n_robots, 2) and a.dtype.kind == "f"
        else:
            assert obs.token_mask[np.arange(obs.n_robots), a].all()
        obs, _, done, _ = env.step(a)
        if done:
            break


def test_a_policy_with_no_valid_token_at_all_still_returns_a_slot():
    cfg = EnvConfig()
    cfg.tokens.include_hold = False
    cfg.robot.n_robots = 1
    env = DisasterEnv(make_synthetic_scene(0, region_m=(200.0, 200.0)), cfg, seed=0)
    obs = env.state.last_obs
    obs.token_mask[:] = False
    for name in sorted(POLICIES):
        pol = make_policy(name, queries=cfg.rayfronts.queries, seed=0)
        a = pol.act(obs, env.state)
        if getattr(pol, "waypoint_policy", False):
            assert a.shape == (1, 2) and a.dtype.kind == "f"     # a waypoint needs no token
            continue
        assert a.shape == (1,) and 0 <= int(a[0]) < obs.tokens.shape[1]


def test_the_belief_is_bit_identical_across_two_envs_after_40_decisions():
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    sc = make_synthetic_scene(0, region_m=(240.0, 240.0))
    out = []
    for _ in range(2):
        env = DisasterEnv(sc, cfg, seed=7)
        pol = make_policy("ray_follower", queries=cfg.rayfronts.queries, seed=7)
        obs = env.state.last_obs
        for _ in range(40):
            obs, _, _, _ = env.step(pol.act(obs, env.state))
        st, r = env.state, env.state.rays
        out.append([st.vox_feat_sum.copy(), st.seg_labels.copy(), st.vox_cnt.copy(), r.az.copy(),
                    r.el.copy(), r.feat.copy(), r.feat_peak.copy(), r.ids.copy(), r.conf.copy(),
                    st.human_hits.copy(), obs.tokens.copy(), obs.bev.copy(), obs.local.copy(),
                    obs.token_xy.copy()])
    for k, (a, b) in enumerate(zip(*out)):
        assert np.array_equal(a, b, equal_nan=True), f"array {k} differs between identical seeds"


def test_determinism_survives_the_training_subprocess_pool():
    par = pytest.importorskip("rlplanner.train.par_env")
    cfg = EnvConfig()
    cfg.robot.n_robots = 2
    cfg.t_max_s = 120.0
    out = []
    for _ in range(2):
        with par.SubprocVecEnv("synthetic:0-3", cfg, 2, robots=(2, 2), seed=5, n_workers=2) as p:
            o = p.reset_all()
            rng = np.random.default_rng(1)
            for _ in range(40):
                a = np.zeros(o.token_mask.shape[:2], np.int64)
                for e in range(a.shape[0]):
                    for r in range(a.shape[1]):
                        v = np.flatnonzero(o.token_mask[e, r])
                        a[e, r] = int(rng.choice(v)) if v.size else 0
                o, rw, _, _ = p.step(a)
            out.append((o.tokens.copy(), o.token_xy.copy(), rw.copy()))
    for a, b in zip(*out):
        assert np.array_equal(a, b, equal_nan=True)


def test_ray_confidence_and_look_count_are_not_pinned_at_their_ceilings():
    """`conf`/`n_obs` count looks down the bearing. Accumulating them per far cell instead put
    every ray at `ray_conf_cap` after two sub-steps and saturated the `n_obs` feature by decision
    five, so both observation columns were constant 1.0 for the whole episode."""
    cfg = EnvConfig()
    rf, rng, ras, cfg = _sim(make_synthetic_scene(0, region_m=(240.0, 240.0)), cfg)
    rb = _robot(-60.0, -60.0, 0.5)
    for k in range(1, 11):
        rf.update([rb], float(k), rng)
        n = rf.n_rays
        assert n and (rf._r_nobs[:n] <= k).all()
        assert (rf._r_conf[:n] < cfg.rayfronts.ray_conf_cap).all(), f"conf capped by sub-step {k}"
    rf.end_of_decision(10.0, [rb])
    nobs = np.array([T.n_obs for T in rf.ray_targets])
    assert nobs.max() <= 10 and (nobs >= 1).all()
    conf = np.array([T.conf for T in rf.ray_targets]) / cfg.rayfronts.ray_conf_cap
    assert 0.0 < conf.max() < 1.0


def test_a_ray_points_at_the_thing_it_describes_within_one_bin():
    """az/el are the direction of the bin's most salient look: a lone toppled car 60 m out owns
    its bearing, and the elevation recovers its ground range."""
    for az_deg in (0.0, 35.0, -140.0, 170.0):
        a = math.radians(az_deg)
        cx, cy = 60.0 * math.cos(a), 60.0 * math.sin(a)
        sc = _scene(w=300.0, vehicles=[Vehicle(id="v", center=(cx, cy), size=(4.5, 1.9),
                                               state="toppled")])
        cfg = EnvConfig()
        cfg.sensor.mode = "disk"                       # look all round from one pose
        rf, rng, ras, cfg = _sim(sc, cfg)
        _look(rf, rng, [_robot(0.0, 0.0, a)], 40)
        cand = [T for T in rf.ray_targets
                if _cos(rf, T.feat, ("overturned car",))[0] > 0.6]
        assert cand, f"no toppled-car ray at {az_deg} deg"
        T = max(cand, key=lambda t: _cos(rf, t.feat, ("overturned car",))[0])
        bin_rad = math.radians(cfg.rayfronts.ray_az_bin_deg)
        d_az = abs((T.az - a + math.pi) % (2 * math.pi) - math.pi)
        assert d_az <= bin_rad, f"az off by {math.degrees(d_az):.1f} deg at {az_deg}"
        assert T.el < 0.0
        assert abs(T.range_m - 60.0) / 60.0 < 0.10
        assert math.hypot(T.xy[0] - cx, T.xy[1] - cy) < 0.10 * 60.0


def test_the_ray_store_stays_well_formed_over_a_whole_episode():
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    cfg.t_max_s = 200.0
    env = DisasterEnv(make_synthetic_scene(1, region_m=(240.0, 240.0)), cfg, seed=0)
    pol = make_policy("ray_follower", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    while True:
        r = env.state.rays
        assert r.n == 0 or np.isfinite(r.feat).all()
        assert r.n == 0 or (np.linalg.norm(r.feat, axis=1) <= 1.0 + 1e-5).all()
        assert r.n == 0 or (r.conf <= cfg.rayfronts.ray_conf_cap + 1e-6).all()
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            break


def test_no_query_is_evaluated_during_a_decision():
    """A counter on the belief's lazy query view: the per-step path must never bump it."""
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    cfg.t_max_s = 120.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(240.0, 240.0)), cfg, seed=0)
    pol = make_policy("ray_follower", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    n0 = env.rf.n_query_calls
    for _ in range(20):
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            break
    assert env.rf.n_query_calls == n0
    env.state.query_sim("person")               # a viewer asking for one still works
    assert env.rf.n_query_calls == n0 + 1
