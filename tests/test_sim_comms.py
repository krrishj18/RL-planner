"""Per-robot knowledge and the gossip layer (CONTRACTS.md 5.1, DESIGN_VARIANTS.md B/C)."""
from types import SimpleNamespace

import numpy as np
import pytest

from rlplanner.scene.schema import make_synthetic_scene
from rlplanner.sim.baselines import make_policy
from rlplanner.sim.comms import RobotBelief, coarse_known, expand_coarse
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.state import (PEER_AGE, PEER_LINK, PEER_VALID, TOKEN_ID_STRIDE, TOKEN_SEGMENT,
                                 VisitRecord)

SCENE = dict(region_m=(200.0, 200.0), n_casualties=6, n_bystanders=3)


def _cfg(mode="range", robots=3, t_max=300.0, **share):
    c = EnvConfig()
    c.robot.n_robots = robots
    c.t_max_s = t_max
    c.comms.mode = mode
    c.comms.randomize_range = False
    c.comms.range_m = float(share.pop("range_m", np.inf))
    c.comms.relay_hops = int(share.pop("relay_hops", 0))
    c.comms.spawn_exchange = bool(share.pop("spawn_exchange", True))
    for k, v in share.items():
        setattr(c.comms.share, k, v)
    return c


def _env(cfg, seed=0, scene_seed=0):
    return DisasterEnv(make_synthetic_scene(scene_seed, **SCENE), cfg, seed=seed)


def _run(env, n, policy="ray_follower"):
    pol = make_policy(policy, queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    info = None
    for _ in range(n):
        obs, _, done, info = env.step(pol.act(obs, env.state))
        if done:
            break
    return obs, info


def _line(env, spacing):
    """Put the robots on a straight line `spacing` apart (a chain for the relay tests)."""
    for i, rb in enumerate(env.state.robots):
        rb.pos[:] = (env.raster.region[0] + 10.0 + i * spacing, env.raster.region[1] + 10.0)


# ---- per-robot beliefs -------------------------------------------------------------------------
def test_full_comms_gives_every_robot_the_same_candidates():
    env = _env(_cfg("full"))
    obs, _ = _run(env, 12)
    ids = [set(zip(obs.token_type[r][obs.token_mask[r]].tolist(),
                   obs.token_id[r][obs.token_mask[r]].tolist())) for r in range(3)]
    assert ids[0] == ids[1] == ids[2]
    assert env.comms is None


def test_blackout_gives_every_robot_a_different_belief():
    env = _env(_cfg(range_m=0.0, rays="none", coverage=False, segments=False, visited=False))
    obs, _ = _run(env, 12)
    known = [b.known for b in env.comms.beliefs]
    assert not np.array_equal(known[0], known[1])
    assert known[0].sum() < env.rf.observed.sum()          # nobody knows the whole team map
    # ids are namespaced per robot, so no two robots can ever claim the same (type, id)
    for r, b in enumerate(env.comms.beliefs):
        real = obs.token_id[r][obs.token_mask[r] & (obs.token_id[r] >= 0)]
        assert (real // TOKEN_ID_STRIDE == r).all()


def test_out_of_range_robots_receive_nothing():
    env = _env(_cfg(range_m=5.0, spawn_exchange=False))
    _run(env, 6)
    for i, rb in enumerate(env.state.robots):       # push them far apart, then contact
        rb.pos[:] = (env.raster.region[0] + 20.0 + 60.0 * i, env.raster.region[1] + 20.0)
    for b in env.comms.beliefs:
        b.peers.clear()
        b.rays_in.clear()
        b.segs_in.clear()
    own = env.comms.beliefs[0].known.copy()
    env.comms.exchange(env.state.robots, env.rf, env.state.t)
    assert all(not b.peers for b in env.comms.beliefs)
    assert all(not b.rays_in and not b.segs_in for b in env.comms.beliefs)
    assert np.array_equal(env.comms.beliefs[0].known, own)


def test_three_hop_chain_relays_and_hop_limit_bites():
    cfg = _cfg(robots=4, range_m=100.0, spawn_exchange=False)
    env = _env(cfg)
    _run(env, 4)
    _line(env, 90.0)                                  # 0 - 1 - 2 - 3, only neighbours in range
    for b in env.comms.beliefs:
        b.peers.clear()
    env.comms.exchange(env.state.robots, env.rf, env.state.t)
    assert set(env.comms.beliefs[3].peers) == {0, 1, 2}, "relay must cross the whole component"
    env2 = _env(_cfg(robots=4, range_m=100.0, relay_hops=1, spawn_exchange=False))
    _run(env2, 4)
    _line(env2, 90.0)
    for b in env2.comms.beliefs:
        b.peers.clear()
    env2.comms.exchange(env2.state.robots, env2.rf, env2.state.t)
    assert set(env2.comms.beliefs[3].peers) == {2}, "relay_hops=1 is direct neighbours only"


def test_received_items_persist_after_the_link_drops():
    env = _env(_cfg(range_m=1e6))
    _run(env, 10)
    b = env.comms.beliefs[0]
    rays, segs, known = dict(b.rays_in), dict(b.segs_in), b.known.copy()
    assert rays and segs, "the sender had rays and segments to give"
    live_before = {i for i, s in rays.items() if not s.resolved}
    env.cfg.comms.range_m = env.comms.range_m = 0.0    # blackout from here on
    _run(env, 6)
    assert set(segs) <= set(b.segs_in)
    assert (b.known | known == b.known).all()
    assert not any(p.linked for p in b.peers.values())
    # a snapshot only ever leaves the inbox by dying against *this* robot's own map (or by the
    # robot taking the bin over itself), never because the link went down
    assert set(b.rays_in) <= live_before
    owned = (env.rf._r_by[: env.rf.n_rays] & b.bit) != 0
    for i in live_before - set(b.rays_in):
        assert rays[i].resolved or owned[i], f"snapshot {i} was dropped while still useful"


def test_coverage_sharing_marks_cells_known_but_featureless():
    env = _env(_cfg(range_m=1e6, rays="none", segments=False, visited=False, features=False))
    _run(env, 10)
    b = env.comms.beliefs[0]
    shared = b.known & ~b.feat_known
    assert shared.any(), "a peer's coverage report should have reached robot 0"
    assert not (b.feat_known & ~b.known).any()
    # such a cell is known (no frontier, no exploration pull) but carries no hits and no feature
    assert (env.rf.observed[shared]).all(), "coverage is only claimed for cells someone observed"


def test_coarse_grid_is_conservative():
    k = 4
    known = np.zeros((9, 9), bool)
    known[:4, :4] = True
    known[4, 4] = True
    c = coarse_known(known, k)
    assert c[0, 0] and not c[1, 1]
    back = expand_coarse(c, k, known.shape)
    assert back.shape == known.shape and not (back & ~known).any()


def test_share_features_hands_over_the_features_too():
    env = _env(_cfg(range_m=1e6, features=True))
    _run(env, 10)
    b = env.comms.beliefs[0]
    assert np.array_equal(b.known, b.feat_known)


# ---- visited records ---------------------------------------------------------------------------
def test_visited_records_propagate_and_become_tokens():
    from rlplanner.sim.state import TOKEN_VISITED
    env = _env(_cfg(range_m=1e6), seed=1)
    obs, _ = _run(env, 30)
    assert env.visits, "the ray_follower baseline arrives at ray targets"
    owner = env.visits[0].robot
    other = (owner + 1) % 3
    assert env.visits[0].key in env.comms.beliefs[other].visited, "gossip carries the record"
    seen = [(obs.token_type[r] == TOKEN_VISITED).sum() for r in range(3)]
    assert min(seen) > 0, "every robot offers visited tokens once it knows some records"


def test_visited_records_stay_local_without_the_flag():
    env = _env(_cfg(range_m=1e6, visited=False), seed=1)
    _run(env, 30)
    assert env.visits
    for rec in env.visits:
        for r, b in enumerate(env.comms.beliefs):
            assert (rec.key in b.visited) == (r == rec.robot)


# ---- peer tokens ---------------------------------------------------------------------------
def test_peer_tokens_track_contact_and_stay_finite():
    env = _env(_cfg(range_m=1e6))
    obs, _ = _run(env, 8)
    pt = obs.peer_tokens
    assert pt.shape == (3, 2, pt.shape[-1]) and np.isfinite(pt).all()
    assert (pt[..., PEER_VALID] == 1.0).all() and (pt[..., PEER_LINK] == 1.0).all()
    env.cfg.comms.range_m = env.comms.range_m = 0.0
    obs, _ = _run(env, 6)
    pt = obs.peer_tokens
    assert (pt[..., PEER_VALID] == 1.0).all(), "a peer once heard from stays in the cache"
    assert (pt[..., PEER_LINK] == 0.0).all(), "... but the link is down"
    assert (pt[..., PEER_AGE] > 0.0).all() and (pt[..., PEER_AGE] <= 1.0).all()


def test_peer_tokens_are_masked_for_padded_robots():
    from rlplanner.sim.vec_env import VecEnv
    envs = [_env(_cfg(robots=n, range_m=1e6), scene_seed=n) for n in (2, 3)]
    vec = VecEnv(envs)
    v = vec.step(np.zeros((2, 3), np.int64))[0]
    assert v.peer_tokens.shape == (2, 3, 2, v.peer_tokens.shape[-1])
    assert np.isfinite(v.peer_tokens).all()
    assert (v.peer_tokens[0, 2] == 0.0).all(), "the padded robot slot stays zero"
    assert (v.peer_tokens[0, 0, 1] == 0.0).all(), "... and so does its peer slot"


# ---- determinism / memory ------------------------------------------------------------------
def test_range_comms_is_deterministic():
    a, b = _env(_cfg(range_m=150.0), seed=7), _env(_cfg(range_m=150.0), seed=7)
    pol_a = make_policy("ray_follower", seed=0)
    pol_b = make_policy("ray_follower", seed=0)
    oa, ob = a.state.last_obs, b.state.last_obs
    for _ in range(40):
        oa, ra, da, ia = a.step(pol_a.act(oa, a.state))
        ob, rb, db, ib = b.step(pol_b.act(ob, b.state))
        assert ra == rb and da == db
        assert np.array_equal(oa.tokens, ob.tokens) and np.array_equal(oa.token_id, ob.token_id)
        assert np.array_equal(oa.peer_tokens, ob.peer_tokens)
        assert np.array_equal(ia["redundant_cells"], ib["redundant_cells"])
    for x, y in zip(a.comms.beliefs, b.comms.beliefs):
        assert np.array_equal(x.known, y.known) and np.array_equal(x.feat_known, y.feat_known)
        assert set(x.rays_in) == set(y.rays_in) and set(x.visited) == set(y.visited)


def test_randomised_range_follows_the_seed():
    cfg = _cfg()
    cfg.comms.randomize_range = True
    cfg.comms.range_choices = (100.0, 200.0, 400.0, float("inf"))
    got = {DisasterEnv(make_synthetic_scene(0, **SCENE), cfg, seed=s).comms.range_m
           for s in range(12)}
    assert len(got) > 1 and got <= set(cfg.comms.range_choices)
    r1 = DisasterEnv(make_synthetic_scene(0, **SCENE), cfg, seed=3).comms.range_m
    r2 = DisasterEnv(make_synthetic_scene(0, **SCENE), cfg, seed=3).comms.range_m
    assert r1 == r2


def test_per_robot_memory_at_750x750():
    """<= 10 MB of private state per robot on the largest shipped raster (1500 m at 2 m cells)."""
    cfg = _cfg(robots=8)
    raster = SimpleNamespace(shape=(750, 750), cell_m=2.0)
    beliefs = [RobotBelief(i, raster, cfg, 24) for i in range(8)]
    per = []
    for b in beliefs:
        n = (b.known.nbytes + b.feat_known.nbytes + b.frontiers.mask.nbytes
             + b.segidx.labels.nbytes + b.own_res.nbytes)
        per.append(n / 1e6)
    assert max(per) <= 10.0, f"{max(per):.1f} MB per robot"
    assert sum(per) <= 8 * 10.0


def test_eight_robots_run_with_range_comms():
    cfg = _cfg(robots=8, range_m=120.0, t_max=120.0)
    env = _env(cfg)
    obs, info = _run(env, 20, policy="nearest_frontier")
    assert obs.peer_tokens.shape == (8, 7, obs.peer_tokens.shape[-1])
    assert np.isfinite(obs.tokens).all()
    assert 0.0 <= info["link_frac"] <= 1.0


@pytest.mark.parametrize("mode", ["full", "range"])
def test_observation_stays_finite_and_bounded(mode):
    env = _env(_cfg(mode, range_m=200.0))
    env.cfg.tokens.robot_bev_size = 32
    env.builder = None
    env.reset(0)
    obs, _ = _run(env, 15)
    assert np.isfinite(obs.tokens).all() and np.abs(obs.tokens).max() <= 5.0
    assert obs.robot_bev is not None and np.isfinite(obs.robot_bev).all()
    assert np.isfinite(obs.local).all() and np.isfinite(obs.bev).all()


def test_set_queries_moves_only_the_query_block_under_range_comms():
    env = _env(_cfg(range_m=150.0))
    obs, _ = _run(env, 8)
    before = (obs.tokens.copy(), obs.token_mask.copy(), obs.token_xy.copy(),
              obs.token_id.copy(), obs.robot_feat.copy(), obs.bev.copy(),
              obs.local.copy(), obs.peer_tokens.copy())
    out = env.set_queries(("car", "person"))
    after = (out.tokens, out.token_mask, out.token_xy, out.token_id, out.robot_feat, out.bev,
             out.local, out.peer_tokens)
    for k, (x, y) in enumerate(zip(before, after)):
        assert np.array_equal(x, y, equal_nan=True), f"field {k} moved"
    assert not np.array_equal(obs.query_emb, out.query_emb)


# ---- gossip semantics (QA pass 2026-08-21) ------------------------------------------------------
def test_link_range_is_inclusive_symmetric_and_switches_off_at_zero():
    env = _env(_cfg(robots=2, range_m=100.0, spawn_exchange=False))
    _run(env, 3)
    a, b = env.state.robots
    a.pos[:] = (20.0, 20.0)
    b.pos[:] = (120.0, 20.0)                       # exactly `range_m` apart
    lk = env.comms.links(env.state.robots)
    assert lk[0, 1] and lk[1, 0] and np.array_equal(lk, lk.T)
    assert not lk.diagonal().any()
    b.pos[:] = (120.0 + 1e-3, 20.0)
    assert not env.comms.links(env.state.robots)[0, 1]
    env.comms.range_m = 0.0                        # a blackout, not "same spot only"
    a.pos[:] = b.pos[:] = (20.0, 20.0)
    assert not env.comms.links(env.state.robots).any()
    env.comms.range_m = float("inf")
    b.pos[:] = (env.raster.region[2] - 1.0, env.raster.region[3] - 1.0)
    assert env.comms.links(env.state.robots)[0, 1]


def test_relay_hop_limits_zero_one_and_two():
    got = {}
    for hops in (0, 1, 2):
        env = _env(_cfg(robots=4, range_m=100.0, relay_hops=hops, spawn_exchange=False))
        _run(env, 3)
        _line(env, 90.0)                            # 0 - 1 - 2 - 3
        got[hops] = set(np.flatnonzero(env.comms.links(env.state.robots)[0]).tolist())
    assert got[1] == {1}, "one hop is the direct neighbour"
    assert got[2] == {1, 2}
    assert got[0] == {1, 2, 3}, "0 = the whole connected component"


def test_a_payload_never_crosses_more_hops_than_allowed_in_one_round():
    """The round is simultaneous: what a robot forwards is what it knew at contact, not what a
    lower-indexed peer handed it in the same exchange."""
    env = _env(_cfg(robots=3, range_m=100.0, relay_hops=1, spawn_exchange=False))
    _run(env, 5)
    _line(env, 90.0)
    for b in env.comms.beliefs:
        b.visited.clear()
    rec = VisitRecord(xy=np.array([25.0, 25.0]), token_type=TOKEN_SEGMENT, token_id=1,
                      feat=np.zeros(env.rf.D, np.float32), t=1.0, robot=0, seq=0, id=99)
    env.comms.beliefs[0].add_visit(rec)
    env.comms.exchange(env.state.robots, env.rf, env.state.t)
    assert [i for i in range(3) if rec.key in env.comms.beliefs[i].visited] == [0, 1]


def test_a_robot_that_meets_nobody_holds_exactly_its_own_observations():
    env = _env(_cfg(range_m=0.0, spawn_exchange=False))     # every payload flag still on
    _run(env, 20)
    for r, b in enumerate(env.comms.beliefs):
        assert not b.peers and not b.rays_in and not b.segs_in
        assert np.array_equal(b.known, b.feat_known), "no coverage-only cells without a peer"
        assert all(v.robot == r for v in b.visited.values())
        assert b.known.sum() < env.rf.observed.sum()
        assert not (b.known & ~env.rf.observed).any(), "it knows nothing nobody observed"


def test_a_coverage_only_cell_reads_as_covered_with_no_hits_and_no_feature():
    """What the receiver's crop and BEV show where a peer only reported coverage."""
    from rlplanner.sim.tokens import BEV_CHANNELS, LOCAL_CHANNELS
    env = _env(_cfg(range_m=1e6, rays="none", segments=False, visited=False, features=False))
    env.cfg.tokens.robot_bev_size = 32
    env.builder = None
    env.reset(0)
    obs, _ = _run(env, 12)
    b = env.comms.beliefs[0]
    shared = b.known & ~b.feat_known
    assert shared.any()
    assert (env.rf.vox_cnt[shared] > 0).all(), "somebody did observe those cells"
    v = env.state.robot_views[0]
    # the belief itself: known, but no feature and (through fknown) no hit count
    assert v.known[shared].all() and not v.fknown[shared].any()
    # ... and the rasters agree: the hit and feature channels are exactly zero there
    ki = BEV_CHANNELS.index("known")
    hi = BEV_CHANNELS.index("hits")
    bev = obs.robot_bev[0]
    lit = (bev[ki] > 0) & (bev[hi] == 0)
    assert lit.any(), "coverage-only ground shows as known with a zero hit channel"
    assert LOCAL_CHANNELS[0] == "known" and LOCAL_CHANNELS[1] == "hits"
    loc = obs.local[0]
    assert np.isfinite(loc).all() and (loc[1][loc[0] == 0] == 0).all()


def test_ray_and_segment_caps_keep_the_newest():
    env = _env(_cfg(range_m=1e6, ray_cap=4, segment_cap=3), seed=2)
    _run(env, 15)
    rf = env.rf
    for src, b in enumerate(env.comms.beliefs):
        payload = env.comms._payload(b, rf, env.cfg.comms.share)
        snaps, segs = payload[1], payload[3 - 1]
        assert len(snaps) <= 4 and len(segs) <= 3
        own = np.flatnonzero(((rf._r_by[: rf.n_rays] & b.bit) != 0) & ~b.own_res[: rf.n_rays])
        if own.size > 4:
            newest = set(np.sort(rf._r_ids[own])[-4:].tolist())
            assert {s.id for s in snaps} == newest, "the cap keeps the newest rays"
        if len(b.segments) > 3:
            assert [s.id for s in segs] == [s.id for s in b.segments
                                            if s.id < 1 << 18][:3]


def test_receiving_the_same_ray_twice_changes_nothing():
    from rlplanner.sim.comms import RaySnapshot
    env = _env(_cfg(range_m=1e6))
    _run(env, 8)
    b = env.comms.beliefs[0]
    sn = RaySnapshot(id=10 ** 6, origin_xy=np.array([30.0, 30.0]), az=0.3, el=-0.6, conf=0.4,
                     n_obs=3, t_first=1.0, t_last=2.0, feat=np.zeros(env.rf.D, np.float32),
                     feat_peak=np.zeros(env.rf.D, np.float32))
    owned = np.zeros(env.rf.n_rays, np.bool_)
    b.receive_rays([sn], owned)
    n1, obs1 = len(b.rays_in), b.rays_in[sn.id].n_obs
    b.receive_rays([sn], owned)
    b.receive_rays([sn], owned)
    assert len(b.rays_in) == n1 and b.rays_in[sn.id].n_obs == obs1


def test_a_bin_the_robot_feeds_itself_is_never_also_a_peer_snapshot():
    """One ray bin, one token: a received snapshot of a bin the robot later feeds itself would
    otherwise give two tokens the same (type, id) and double-count it in the ray rasters."""
    env = _env(_cfg(range_m=1e6), seed=3)
    for _ in range(6):
        obs, _ = _run(env, 10)
        for r, b in enumerate(env.comms.beliefs):
            ids = [t.id for t in b.ray_list]
            assert len(ids) == len(set(ids)), f"robot {r} offers one ray bin twice"
            st = b.ray_store
            assert st.n == len(set(st.ids.tolist()))
            owned = (env.rf._r_by[: env.rf.n_rays] & b.bit) != 0
            assert not any(i < owned.shape[0] and owned[i] for i in b.rays_in)
        for r in range(3):
            m = obs.token_mask[r] & (obs.token_id[r] >= 0)
            keys = list(zip(obs.token_type[r][m].tolist(), obs.token_id[r][m].tolist()))
            assert len(keys) == len(set(keys))


def test_the_inboxes_stay_bounded_over_a_long_episode():
    """Received rays and segments are dropped once they can never be a token again, so neither
    inbox grows with the episode length."""
    env = _env(_cfg(range_m=1e6, t_max=1200.0), seed=1)
    pool = 2 * env.cfg.tokens.k_segment
    sizes = []
    for _ in range(6):
        _run(env, 30)
        sizes.append(max(len(b.rays_in) for b in env.comms.beliefs))
        n = env.rf.n_rays
        for b in env.comms.beliefs:
            assert len(b.segs_in) <= pool
            owned = (env.rf._r_by[:n] & b.bit) != 0
            assert not any(i < n and owned[i] for i in b.rays_in)
    assert env.state.decision_idx >= 150
    assert sizes[-1] <= sizes[0], f"the ray inbox grows with the episode: {sizes}"


def test_gossip_runs_once_per_decision():
    env = _env(_cfg(range_m=150.0))
    calls = []
    real = env.comms.exchange
    env.comms.exchange = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    obs = env.state.last_obs
    pol = make_policy("ray_follower", queries=env.cfg.rayfronts.queries, seed=0)
    for _ in range(7):
        obs, _, done, _ = env.step(pol.act(obs, env.state))
        if done:
            break
    assert len(calls) == 7


def test_token_ids_stay_in_each_robots_namespace_while_sharing():
    env = _env(_cfg(range_m=1e6), seed=1)
    obs, _ = _run(env, 25)
    for r in range(3):
        m = obs.token_mask[r] & (obs.token_id[r] >= 0)
        assert (obs.token_id[r][m] // TOKEN_ID_STRIDE == r).all()
        keys = list(zip(obs.token_type[r][m].tolist(), obs.token_id[r][m].tolist()))
        assert len(keys) == len(set(keys)), "(type, id) is the claim key and must be unique"


def test_reported_coverage_is_the_senders_own_fraction():
    from rlplanner.sim.state import PEER_COV
    env = _env(_cfg(range_m=1e6))
    obs, info = _run(env, 15)
    n_obs = max(1, int(env.rf.observable.sum()))
    for r, b in enumerate(env.comms.beliefs):
        assert b.coverage == pytest.approx(b.known.sum() / n_obs)
        assert obs.robot_feat[r, 7] == pytest.approx(b.coverage, abs=1e-6)
    for c, rj in enumerate(j for j in range(3) if j != 0):
        pr = env.comms.beliefs[0].peers[rj]
        assert obs.peer_tokens[0, c, PEER_COV] == pytest.approx(pr.coverage, abs=1e-6)
        # what the sender put on the air is its own fraction as of the contact, so it lags its
        # post-exchange belief by this round's receipts and is never the team's coverage
        assert pr.coverage <= env.comms.beliefs[rj].coverage + 1e-9
        assert pr.coverage <= info["coverage"] + 1e-9
    assert 0.0 <= info["coverage"] <= 1.0


def test_ten_robots_all_in_range():
    env = _env(_cfg(robots=10, range_m=1e6, t_max=100.0))
    obs, info = _run(env, 12, policy="nearest_frontier")
    assert obs.peer_tokens.shape == (10, 9, obs.peer_tokens.shape[-1])
    assert np.isfinite(obs.tokens).all()
    assert info["link_frac"] == pytest.approx(1.0)
    assert env.state.comms_links.sum() == 90


def test_a_tokens_peer_columns_come_from_the_peer_cache_not_the_true_team():
    """`peer_dist_min` / `claimed_by_peer` are part of the robot's own view: a robot that has
    never heard from anybody must not read the others' true positions off its observation."""
    from rlplanner.sim.state import F_CLAIM, F_PEER
    env = _env(_cfg(range_m=0.0, spawn_exchange=False))
    obs, _ = _run(env, 12)
    for r in range(3):
        m = obs.token_mask[r]
        assert (obs.tokens[r, m, F_PEER] == 1.0).all(), "no peer it knows of is anywhere near"
        assert (obs.tokens[r, m, F_CLAIM] == 0.0).all()
    # with contact the columns come alive again, from what the cache says
    env2 = _env(_cfg(range_m=1e6))
    obs2, _ = _run(env2, 12)
    m = obs2.token_mask[0]
    assert (obs2.tokens[0, m, F_PEER] < 1.0).any()


def test_metrics_link_frac_covers_the_decision_it_is_reported_for():
    env = _env(_cfg(robots=2, range_m=5.0, spawn_exchange=False, t_max=200.0))
    obs, info = _run(env, 8)
    assert info["metrics"]["link_frac"] == pytest.approx(info["link_frac"])
    assert info["metrics"]["link_frac"] < 1.0, "two robots 5 m apart are rarely in contact"


def test_a_robot_that_only_passes_through_range_mid_decision_gets_nothing():
    """Contact is the end-of-decision geometry, once per decision (CONTRACTS.md 5.1)."""
    env = _env(_cfg(robots=2, range_m=50.0, spawn_exchange=False))
    _run(env, 4)
    home = np.array([env.raster.region[0] + 20.0, env.raster.region[1] + 20.0])
    for b in env.comms.beliefs:
        b.peers.clear()
        b.rays_in.clear()
        b.segs_in.clear()
    real = env._move
    n_sub = env.cfg.substeps_per_decision
    calls = []

    def move(robots, ev, t):
        real(robots, ev, t)
        calls.append(1)
        robots[0].pos[:] = home                      # in range for the middle of the decision,
        robots[1].pos[:] = home + np.array(          # ... and 300 m away when it ends
            [10.0, 0.0] if len(calls) < n_sub else [300.0, 0.0])

    env._move = move
    env.step(np.zeros(2, np.int64))
    env._move = real
    assert len(calls) == n_sub
    assert all(not b.peers for b in env.comms.beliefs)
    assert all(not b.rays_in and not b.segs_in for b in env.comms.beliefs)
