import math

import numpy as np
import pytest

from rlplanner.scene.schema import make_synthetic_scene
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.state import (F_AGE, F_AZ_COS, F_AZ_SIN, F_BCOS, F_BSIN, F_CLAIM, F_CONF,
                                 F_COV, F_DIST, F_DX, F_DY, F_EL_COS, F_EL_SIN, F_FEAT0, F_HITS,
                                 F_NOBS, F_ORIGIN_DX, F_ORIGIN_DY, F_PEER, F_RANGE, F_RAYS,
                                 F_REACH, F_SIZE, N_TOKEN_TYPES, PEER_FEAT_DIM, TOKEN_FIXED,
                                 TOKEN_FRONTIER, TOKEN_HOLD, TOKEN_RAY, TOKEN_SEGMENT,
                                 TOKEN_VISITED, token_feature_names)
from rlplanner.sim.tokens import BEV_CHANNELS, CLAIM_M, LOCAL_CHANNELS, NBHD_M


def _env(**kw):
    cfg = EnvConfig()
    cfg.robot.n_robots = kw.pop("robots", 3)
    cfg.t_max_s = kw.pop("t_max", 120.0)
    for k, v in kw.items():
        setattr(cfg.tokens, k, v)
    return DisasterEnv(make_synthetic_scene(0, region_m=(120.0, 120.0)), cfg, seed=0)


def _warm(env, n=10):
    from rlplanner.sim.baselines import make_policy
    pol = make_policy("ray_follower")
    obs = env.state.last_obs
    for _ in range(n):
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            break
    return obs


def test_feature_layout_matches_state_helper():
    env = _env()
    names = token_feature_names(env.rf.D)
    assert len(names) == env.state.last_obs.tokens.shape[2]
    assert names[:4] == ["type_hold", "type_frontier", "type_ray", "type_segment"]
    assert names[TOKEN_FIXED] == "feat0"
    # the observation width no longer depends on the number of mission queries
    assert env.state.last_obs.tokens.shape[2] == TOKEN_FIXED + env.cfg.rayfronts.embedding_dim


def test_no_query_column_survives_in_a_token():
    env = _env()
    assert not any(n.startswith("sim:") for n in token_feature_names(env.rf.D))


def test_slot_order_is_fixed():
    env = _env()
    obs = _warm(env)
    tc = env.cfg.tokens
    tt = obs.token_type[0]
    assert tt[0] == TOKEN_HOLD
    s0 = 1 + tc.k_frontier
    s1 = s0 + tc.k_ray
    s2 = s1 + tc.k_segment
    fr = slice(1, s0)
    ry = slice(s0, s1)
    sg = slice(s1, s2)
    vi = slice(s2, s2 + tc.k_visited)
    filled = obs.token_id[0] >= 0
    assert set(tt[fr][filled[fr]].tolist()) <= {TOKEN_FRONTIER}
    assert set(tt[ry][filled[ry]].tolist()) <= {TOKEN_RAY}
    assert set(tt[sg][filled[sg]].tolist()) <= {TOKEN_SEGMENT}
    assert set(tt[vi][filled[vi]].tolist()) <= {TOKEN_VISITED}
    assert s2 + tc.k_visited == obs.tokens.shape[1]
    onehot = obs.tokens[0, :, :N_TOKEN_TYPES]
    for k in range(obs.tokens.shape[1]):
        if filled[k] or k == 0:
            assert onehot[k].sum() == 1 and onehot[k, tt[k]] == 1


def test_candidates_are_newest_first():
    """Recency is the only ordering rule: no info gain, no similarity, no diversity pass."""
    env = _env()
    _warm(env, 12)
    st = env.state
    fr_ids = [c.id for c in st.frontier_clusters]
    assert fr_ids == sorted(fr_ids, reverse=True)
    ray_ids = [r.id for r in st.ray_targets]
    assert ray_ids == sorted(ray_ids, reverse=True)
    seg_t = [(s.t_first, s.id) for s in st.segments]
    assert seg_t == sorted(seg_t, reverse=True)


def test_candidate_sets_are_shared_across_robots():
    env = _env()
    obs = _warm(env)
    assert np.array_equal(obs.token_id[0], obs.token_id[1])
    assert np.array_equal(obs.token_type[0], obs.token_type[1])
    xy = obs.token_xy[:, 1:]                    # slot 0 (hold) is the robot's own position
    keep = np.isfinite(xy[0]).all(1)
    assert np.array_equal(xy[0][keep], xy[1][keep])


def test_ego_features_are_per_robot():
    env = _env()
    obs = _warm(env)
    k = int(np.flatnonzero(obs.token_id[0] >= 0)[0])
    r0, r1 = env.state.robots[0], env.state.robots[1]
    tgt = obs.token_xy[0, k]
    diag = env.raster.diagonal_m
    assert obs.tokens[0, k, F_DX] == pytest.approx((tgt[0] - r0.pos[0]) / diag, abs=1e-5)
    assert obs.tokens[1, k, F_DY] == pytest.approx((tgt[1] - r1.pos[1]) / diag, abs=1e-5)
    d = math.hypot(*(tgt - r0.pos))
    assert obs.tokens[0, k, F_DIST] == pytest.approx(d / diag, abs=1e-5)
    br = math.atan2(tgt[1] - r0.pos[1], tgt[0] - r0.pos[0]) - r0.heading
    assert obs.tokens[0, k, F_BSIN] == pytest.approx(math.sin(br), abs=1e-5)
    assert obs.tokens[0, k, F_BCOS] == pytest.approx(math.cos(br), abs=1e-5)


def test_features_are_finite_and_bounded():
    env = _env()
    obs = _warm(env, 20)
    filled = obs.token_id[0] >= 0
    v = obs.tokens[:, filled]
    assert np.isfinite(v).all()
    assert v.min() >= -1.01 and v.max() <= 1.01
    for f in (F_SIZE, F_HITS, F_CONF, F_AGE, F_NOBS, F_COV, F_RAYS, F_CLAIM, F_PEER, F_REACH):
        col = obs.tokens[:, filled, f]
        assert col.min() >= 0.0 and col.max() <= 1.0, f


def test_token_feature_tail_is_a_unit_embedding():
    env = _env()
    obs = _warm(env, 15)
    filled = obs.token_id[0] >= 0
    feat = obs.tokens[0, filled, F_FEAT0:]
    n = np.linalg.norm(feat, axis=1)
    assert feat.shape[1] == env.rf.D
    assert ((n > 0.99) & (n < 1.01) | (n < 1e-6)).all()   # unit, or zero where nothing was seen
    assert n.max() > 0.99


def test_ray_and_segment_tokens_carry_their_own_feature():
    env = _env(robots=2, t_max=200.0)
    obs = _warm(env, 12)
    rays = {r.id: r for r in env.state.ray_targets}
    segs = {s.id: s for s in env.state.segments}
    n_r = n_s = 0
    for k in range(obs.tokens.shape[1]):
        tt, tid = int(obs.token_type[0, k]), int(obs.token_id[0, k])
        if tid < 0:
            continue
        if tt == TOKEN_RAY:
            assert obs.tokens[0, k, F_FEAT0:] == pytest.approx(rays[tid].feat, abs=1e-5)
            n_r += 1
        elif tt == TOKEN_SEGMENT:
            assert obs.tokens[0, k, F_FEAT0:] == pytest.approx(segs[tid].feat, abs=1e-5)
            n_s += 1
    assert n_r > 0 and n_s > 0


def test_reachable_feature_matches_the_planner_and_the_mask():
    env = _env()
    obs = _warm(env)
    ras = env.raster
    for r, rb in enumerate(env.state.robots):
        si, sj = ras.xy_to_ij(rb.pos[0], rb.pos[1])
        for k in range(obs.tokens.shape[1]):
            if not np.isfinite(obs.token_xy[r, k]).all():
                continue
            gi, gj = ras.xy_to_ij(*np.clip(obs.token_xy[r, k], [ras.region[0], ras.region[1]],
                                           [ras.region[2] - 1e-6, ras.region[3] - 1e-6]))
            want = env.planner.reachable((si, sj), (gi, gj)) if k > 0 else True
            assert bool(obs.tokens[r, k, F_REACH]) == want
            assert bool(obs.token_mask[r, k]) == want


def test_claimed_by_peer_and_peer_dist():
    env = _env()
    obs = _warm(env)
    tgt = [r.target_xy for r in env.state.robots]
    for r in range(3):
        for k in range(obs.tokens.shape[1]):
            xy = obs.token_xy[r, k]
            if not np.isfinite(xy).all() or k == 0:
                continue
            want = any(t is not None and math.hypot(t[0] - xy[0], t[1] - xy[1]) <= CLAIM_M
                       for j, t in enumerate(tgt) if j != r)
            assert bool(obs.tokens[r, k, F_CLAIM]) == want
            pd = min(math.hypot(o.pos[0] - xy[0], o.pos[1] - xy[1])
                     for j, o in enumerate(env.state.robots) if j != r)
            assert obs.tokens[r, k, F_PEER] == pytest.approx(
                min(pd / env.raster.diagonal_m, 1.0), abs=1e-5)


def test_coverage_neighbourhood():
    env = _env()
    obs = _warm(env, 15)
    ras, rf = env.raster, env.rf
    k = int(np.flatnonzero(obs.token_id[0] >= 0)[0])
    xy = obs.token_xy[0, k]
    ci, cj = ras.xy_to_ij(xy[0], xy[1])
    ii, jj = np.meshgrid(np.arange(ras.ny), np.arange(ras.nx), indexing="ij")
    x, y = ras.ij_to_xy(ii, jj)
    cx, cy = ras.ij_to_xy(ci, cj)
    disc = (x - cx) ** 2 + (y - cy) ** 2 <= NBHD_M ** 2
    assert obs.tokens[0, k, F_COV] == pytest.approx(rf.observed[disc].mean(), abs=0.02)


def test_empty_slots_when_there_are_no_candidates():
    env = _env(k_frontier=40, k_ray=20, k_segment=20, k_visited=8)
    obs = env.state.last_obs
    assert obs.tokens.shape[1] == 1 + 40 + 20 + 20 + 8
    assert (obs.token_id[0] == -1).sum() > 40
    assert obs.token_mask[0].sum() < 41
    empty = obs.token_id[0] < 0
    empty[0] = False
    assert np.isnan(obs.token_xy[0][empty]).all()
    assert not np.isnan(obs.token_xy[0][~empty]).any()


def test_robot_features():
    env = _env(robots=2)
    obs = _warm(env, 4)
    x0, y0, x1, y1 = env.raster.region
    for i, rb in enumerate(env.state.robots):
        f = obs.robot_feat[i]
        assert f[0] == pytest.approx(2 * (rb.pos[0] - x0) / (x1 - x0) - 1, abs=1e-5)
        assert f[2] == pytest.approx(rb.alt / 100.0)
        assert f[3] == pytest.approx(math.sin(rb.heading), abs=1e-5)
        assert f[5] + f[6] == pytest.approx(1.0)
        assert f[5] == pytest.approx(env.state.t / env.cfg.t_max_s, abs=1e-5)
        assert f[8 + i] == 1.0 and f[8:].sum() == 1.0
    assert np.abs(obs.robot_feat).max() <= 1.0


# ---- query block -------------------------------------------------------------------------------
def test_query_tokens_are_padded_and_masked():
    env = _env()
    obs = env.state.last_obs
    q = env.cfg.tokens.max_queries
    assert obs.query_emb.shape == (q, env.rf.D)
    assert obs.query_mask.tolist() == [True, True] + [False] * (q - 2)
    assert obs.query_w[:2].tolist() == [1.0, 1.0] and obs.query_w[2:].max() == 0.0
    n = np.linalg.norm(obs.query_emb[:2], axis=1)
    assert np.allclose(n, 1.0, atol=1e-5)
    assert np.abs(obs.query_emb[2:]).max() == 0.0


@pytest.mark.parametrize("names", [("person",),
                                   ("person lying on the ground", "person", "collapsed building",
                                    "damaged building", "rubble", "overturned car", "car",
                                    "bus stop")])
def test_one_and_eight_queries(names):
    env = _env()
    obs = env.set_queries(names)
    assert int(obs.query_mask.sum()) == len(names)
    assert np.allclose(np.linalg.norm(obs.query_emb[: len(names)], axis=1), 1.0, atol=1e-5)
    pad = obs.query_emb[len(names):]
    assert pad.size == 0 or np.abs(pad).max() == 0.0


def test_set_queries_changes_only_the_query_tokens():
    env = _env()
    obs0 = _warm(env, 8)
    before = {k: np.array(getattr(obs0, k), copy=True)
              for k in ("tokens", "token_mask", "token_xy", "token_type", "token_id",
                        "robot_feat", "bev", "local")}
    obs1 = env.set_queries(("car", "tree"))
    for k, v in before.items():
        got = getattr(obs1, k)
        assert np.array_equal(np.nan_to_num(v, nan=-7.0), np.nan_to_num(got, nan=-7.0)), k
    assert not np.allclose(obs0.query_emb, obs1.query_emb)


def test_too_many_queries_is_refused():
    env = _env()
    with pytest.raises(ValueError, match="max_queries"):
        env.set_queries(tuple(f"q{i}" for i in range(env.cfg.tokens.max_queries + 1)))


# ---- rasters -----------------------------------------------------------------------------------
def test_bev_channels():
    env = _env()
    obs = _warm(env, 10)
    assert obs.bev.shape == (len(BEV_CHANNELS), 64, 64)
    assert np.isfinite(obs.bev).all()
    assert obs.bev.min() >= -1.5 and obs.bev.max() <= 1.5
    ch = {n: i for i, n in enumerate(BEV_CHANNELS)}
    assert obs.bev[ch["known"]].mean() == pytest.approx(env.state.observed.mean(), abs=0.06)
    assert obs.bev[ch["frontier"]].any()
    assert obs.bev[ch["robots"]].max() == pytest.approx(1.0, abs=0.2)
    assert np.abs(obs.bev[ch["feat_pc0"]]).max() > 0.0
    # the feature channels are zero exactly where nothing is known
    unknown = obs.bev[ch["known"]] == 0
    assert np.abs(obs.bev[ch["feat_pc0"]][unknown]).max() == 0.0
    assert not any(c.startswith("person") or c.startswith("collapsed") for c in BEV_CHANNELS)


def test_bev_ray_channels_show_the_raw_rays():
    env = _env(robots=3, t_max=200.0)
    obs = _warm(env, 12)
    ch = {n: i for i, n in enumerate(BEV_CHANNELS)}
    assert 0.0 < obs.bev[ch["ray_count"]].max() <= 1.0
    assert np.abs(obs.bev[ch["ray_feat_pc0"]]).max() > 0.0
    env.rf._r_res[:env.rf.n_rays] = True                 # no live rays -> empty channels
    o2 = env.builder.build(env.rf, env.state.robots, env.state.t, env.planner)
    assert o2.bev[ch["ray_count"]].max() == 0.0
    assert np.abs(o2.bev[ch["ray_feat_pc0"]]).max() == 0.0


def test_local_crop_is_ego_centred():
    env = _env()
    obs = _warm(env, 10)
    s = env.cfg.tokens.local_size
    assert obs.local.shape == (3, len(LOCAL_CHANNELS), s, s)
    assert np.isfinite(obs.local).all()
    ch = {n: i for i, n in enumerate(LOCAL_CHANNELS)}
    ras = env.raster
    for r, rb in enumerate(env.state.robots):
        i, j = ras.xy_to_ij(rb.pos[0], rb.pos[1])
        assert obs.local[r, ch["known"], s // 2, s // 2] == float(env.state.observed[i, j])
    # the crop follows the robot: two robots at different places see different neighbourhoods
    assert not np.allclose(obs.local[0], obs.local[1])


def test_local_crop_can_be_switched_off():
    env = _env(local_size=0)
    obs = env.state.last_obs
    assert obs.local is None


def test_peer_tokens_under_full_comms():
    """Full comms = everyone in contact: every peer slot is valid, linked and fresh."""
    from rlplanner.sim.state import PEER_AGE, PEER_LINK, PEER_VALID
    env = _env(robots=3)
    obs = env.state.last_obs
    assert obs.peer_tokens.shape == (3, 2, PEER_FEAT_DIM)
    assert np.isfinite(obs.peer_tokens).all()
    assert (obs.peer_tokens[..., PEER_VALID] == 1.0).all()
    assert (obs.peer_tokens[..., PEER_LINK] == 1.0).all()
    assert (obs.peer_tokens[..., PEER_AGE] == 0.0).all()


def test_ray_tokens_carry_the_raw_ray_geometry():
    """origin, azimuth, elevation and the elevation-derived range, so a policy can work out for
    itself that two rays cross. Nothing in the sim merges them."""
    env = _env(robots=2, t_max=200.0)
    obs = _warm(env, 12)
    by_id = {T.id: T for T in env.state.ray_targets}
    seen = 0
    for r, rb in enumerate(env.state.robots):
        for k in range(obs.tokens.shape[1]):
            if int(obs.token_type[r, k]) != TOKEN_RAY or obs.token_id[r, k] < 0:
                continue
            T = by_id[int(obs.token_id[r, k])]
            v = obs.tokens[r, k]
            diag = env.raster.diagonal_m
            assert v[F_ORIGIN_DX] == pytest.approx((T.origin_xy[0] - rb.pos[0]) / diag, abs=1e-5)
            assert v[F_ORIGIN_DY] == pytest.approx((T.origin_xy[1] - rb.pos[1]) / diag, abs=1e-5)
            assert v[F_AZ_SIN] == pytest.approx(math.sin(T.az), abs=1e-5)
            assert v[F_AZ_COS] == pytest.approx(math.cos(T.az), abs=1e-5)
            assert v[F_EL_SIN] == pytest.approx(math.sin(T.el), abs=1e-5)
            assert v[F_EL_COS] == pytest.approx(math.cos(T.el), abs=1e-5)
            assert v[F_RANGE] == pytest.approx(T.range_m / diag, abs=1e-5)
            tgt = T.origin_xy + T.range_m * np.array([math.cos(T.az), math.sin(T.az)])
            assert np.allclose(obs.token_xy[r, k],
                               np.clip(tgt, [env.raster.region[0], env.raster.region[1]],
                                       [env.raster.region[2], env.raster.region[3]]), atol=0.5)
            seen += 1
    assert seen > 0


def test_ray_tokens_are_one_per_bin_and_never_merged():
    env = _env(robots=3, t_max=200.0)
    obs = _warm(env, 12)
    ray_ids = [int(obs.token_id[0, k]) for k in range(obs.tokens.shape[1])
               if int(obs.token_type[0, k]) == TOKEN_RAY and obs.token_id[0, k] >= 0]
    assert len(ray_ids) == len(set(ray_ids))
    assert len(ray_ids) == len(env.state.ray_targets[:env.cfg.tokens.k_ray])


def test_segment_tokens_are_map_regions():
    env = _env(robots=2, t_max=200.0)
    obs = _warm(env, 12)
    by_id = {s.id: s for s in env.state.segments}
    lab = env.state.seg_labels
    n = 0
    for k in range(obs.tokens.shape[1]):
        if int(obs.token_type[0, k]) != TOKEN_SEGMENT or obs.token_id[0, k] < 0:
            continue
        s = by_id[int(obs.token_id[0, k])]
        assert np.allclose(obs.token_xy[0, k], s.xy, atol=1e-3)
        assert lab[s.ij] >= 0 and env.state.observed[s.ij]
        assert obs.tokens[0, k, F_SIZE] == pytest.approx(min(s.n_cells / 200.0, 1.0), abs=1e-5)
        assert obs.tokens[0, k, F_RAYS] == pytest.approx(min(s.ray_count / 8.0, 1.0), abs=1e-5)
        n += 1
    assert n > 0


# ---- the CTDE / per-robot-view seam -------------------------------------------------------------
def test_per_robot_views_are_honoured():
    """`build` takes one RobotView per robot. Under comms=full they are the same object (today's
    behaviour); hand one robot a *smaller* known mask and only that robot's observation changes."""
    from rlplanner.sim.tokens import RobotView, team_view
    env = _env(robots=2)
    _warm(env, 12)
    shared = team_view(env.rf, env.visits)      # the team view the env itself builds
    blind = RobotView(known=np.zeros_like(shared.known), feat_sum=shared.feat_sum,
                      hits=shared.hits, last_seen=shared.last_seen,
                      frontier_mask=np.zeros_like(shared.frontier_mask), frontiers=[], rays=[],
                      segments=[], ray_store=shared.ray_store)
    both = env.builder.build(env.rf, env.state.robots, env.state.t, env.planner,
                             views=[shared, shared])
    split = env.builder.build(env.rf, env.state.robots, env.state.t, env.planner,
                              views=[shared, blind])
    assert np.array_equal(both.tokens, env.state.last_obs.tokens)   # explicit team view == default
    assert np.array_equal(split.tokens[0], both.tokens[0])          # robot 0 unchanged
    assert (split.token_id[1] < 0).all()                            # robot 1 has no candidates
    assert split.token_mask[1].sum() == 1                           # ... only its hold slot
    assert np.abs(split.local[1, :2]).max() == 0.0                  # nothing known in its crop
    assert np.abs(split.local[1, 3:]).max() == 0.0                  # ... and no features either
    assert np.abs(both.local[0] - split.local[0]).max() == 0.0
    # the critic's BEV is global either way: that is the CTDE split
    assert np.array_equal(both.bev, split.bev)


def test_build_rejects_a_view_list_of_the_wrong_length():
    from rlplanner.sim.tokens import team_view
    env = _env(robots=3)
    with pytest.raises(ValueError, match="views for"):
        env.builder.build(env.rf, env.state.robots, env.state.t, env.planner,
                          views=[team_view(env.rf)])
