"""The two decentralised reward terms (DESIGN_VARIANTS.md D): redundant coverage and the
intentional revisit, their refund, their `known_only` semantics and their magnitudes."""
import math

import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.scene.schema import make_synthetic_scene
from rlplanner.sim.baselines import make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.state import (CASUALTY_ROLE_ID, TOKEN_FRONTIER, TOKEN_SEGMENT, Event,
                                 RobotState, VisitRecord)

SCENE = dict(region_m=(200.0, 200.0), n_casualties=6, n_bystanders=3)


def _cfg(mode="full", robots=3, t_max=300.0, **reward):
    c = EnvConfig()
    c.robot.n_robots = robots
    c.t_max_s = t_max
    c.comms.mode = mode
    c.comms.randomize_range = False
    c.comms.range_m = float("inf")
    for k, v in reward.items():
        setattr(c.reward, k, v)
    return c


def _env(cfg, seed=0):
    return DisasterEnv(make_synthetic_scene(0, **SCENE), cfg, seed=seed)


def _record(env, xy, robot, t=None, share_with=(), n_found=1):
    """A visited record from `robot`, by default made one second before now and *confirmed*
    (`n_found=1`): the revisit charge only applies to confirmed records (`revisit_confirmed_only`),
    so tests of the other rules stage a chargeable one. Pass `n_found=0` for an unconfirmed visit."""
    t = env.state.t - 1.0 if t is None else t
    rec = VisitRecord(xy=np.asarray(xy, np.float64), token_type=TOKEN_SEGMENT, token_id=7,
                      feat=np.zeros(env.rf.D, np.float32), t=float(t), robot=int(robot),
                      seq=len(env.visits), id=len(env.visits), n_found=int(n_found))
    env.visits.append(rec)
    if env.comms is not None:
        env.comms.beliefs[robot].add_visit(rec)
        for r in share_with:
            env.comms.beliefs[r].add_visit(rec)
    return rec


def _arrive(env, robot: int, xy, ttype=TOKEN_SEGMENT, waypoint=False):
    """Drive one arrival at a chosen target without waiting for the robot to fly there.

    The env only counts an arrival the robot *travelled* to, so the decision-start position is
    pushed away from the target first. `waypoint` marks the goal as a direct waypoint action.
    """
    rb = env.state.robots[robot]
    env._dec_pos0[robot] = np.asarray(xy, np.float64) + np.array([50.0, 50.0])
    rb.target_xy = np.asarray(xy, np.float64)
    rb.target_token_type = TOKEN_FRONTIER if waypoint else ttype
    rb.target_waypoint = bool(waypoint)
    rb.target_id = -1 if waypoint else 1
    rb.target_feat = None if waypoint else np.zeros(env.rf.D, np.float32)
    env._on_arrive(rb, env.state.t)


def test_a_waypoint_arrival_records_nothing_and_owes_nothing():
    """A waypoint names no map item, so it writes no visited record and pays no revisit -- the
    same arrival reached through a token target does both (CONTRACTS.md 6)."""
    env = _env(_cfg())
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=1)
    env._begin_decision()
    _arrive(env, 0, here + np.array([2.0, 0.0]), waypoint=True)
    assert env._revisits[0] == 0 and len(env.visits) == 1
    _arrive(env, 0, here + np.array([2.0, 0.0]))
    assert env._revisits[0] == 1 and len(env.visits) == 2


# ---- revisit ------------------------------------------------------------------------------
def test_intentional_revisit_costs_and_a_fly_by_does_not():
    env = _env(_cfg())
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=1)
    env._begin_decision()          # the record is on the books when the robot chooses
    # a fly-by: the robot is inside revisit_m of the record but its target is elsewhere
    far = here + np.array([60.0, 0.0])
    _arrive(env, 0, far)
    assert env._revisits[0] == 0
    # intentional: it arrives at a target that *is* the recorded one
    _arrive(env, 0, here + np.array([2.0, 0.0]))
    assert env._revisits[0] == 1


def test_a_robot_does_not_pay_for_its_own_visit():
    env = _env(_cfg())
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=0)
    env._begin_decision()
    _arrive(env, 0, here)
    assert env._revisits[0] == 0


def test_revisit_radius_is_respected():
    env = _env(_cfg(revisit_m=15.0))
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=1)
    env._begin_decision()
    _arrive(env, 0, here + np.array([14.0, 0.0]))
    _arrive(env, 0, here + np.array([16.0, 0.0]))
    assert env._revisits[0] == 1


def test_known_only_spares_a_robot_that_was_never_told():
    """A record another robot made but never gossiped: `known_only=True` (default) does not
    charge for ignorance, `known_only=False` does."""
    for known_only, expect in ((True, 0), (False, 1)):
        env = _env(_cfg("range", revisit_known_only=known_only))
        env.cfg.comms.share.visited = False
        here = env.state.robots[0].pos.copy()
        _record(env, here, robot=1)          # lands in robot 1's belief only
        env._begin_decision()
        _arrive(env, 0, here)
        assert env._revisits[0] == expect, (known_only, env._revisits)


def test_known_only_charges_once_the_record_has_arrived():
    env = _env(_cfg("range", revisit_known_only=True))
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=1, share_with=(0,))
    env._begin_decision()                            # ... and the snapshot is taken after it did
    _arrive(env, 0, here)
    assert env._revisits[0] == 1


def test_a_record_made_later_in_the_same_decision_is_not_a_revisit():
    env = _env(_cfg())
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=1, t=env.state.t + 5.0)      # in the future
    env._begin_decision()
    _arrive(env, 0, here)
    assert env._revisits[0] == 0


# ---- visited records ---------------------------------------------------------------------------
def test_only_ray_and_segment_arrivals_make_a_record():
    from rlplanner.sim.state import TOKEN_FRONTIER, TOKEN_RAY
    env = _env(_cfg())
    env._begin_decision()
    here = env.state.robots[0].pos.copy()
    _arrive(env, 0, here + np.array([5.0, 0.0]), ttype=TOKEN_FRONTIER)
    assert not env.visits
    _arrive(env, 0, here + np.array([5.0, 0.0]), ttype=TOKEN_RAY)
    _arrive(env, 1, here + np.array([9.0, 0.0]), ttype=TOKEN_SEGMENT)
    assert [v.robot for v in env.visits] == [0, 1]
    assert env.visits[0].id != env.visits[1].id


def test_a_visit_records_the_casualties_it_turned_up():
    env = _env(_cfg())
    env._begin_decision()
    hs = env.raster.humans
    cas = np.flatnonzero(hs["role_id"] == CASUALTY_ROLE_ID)
    k = int(cas[0])
    xy = np.array([hs["x"][k], hs["y"][k]], np.float64)
    _arrive(env, 0, xy)
    env.rf.found_this_decision.append((0, k))
    env._close_visits()
    assert env.visits[-1].n_found == 1
    env._dec_visits[-1].n_found = 0
    env.rf.found_this_decision[:] = [(1, k)]         # a different robot's find does not count
    env._close_visits()
    assert env.visits[-1].n_found == 0


# ---- redundancy ----------------------------------------------------------------------------
def test_redundant_cells_count_only_what_a_peer_had_already_covered():
    env = _env(_cfg(robots=2))
    rf = env.rf
    rf.seen_by[:] = 0                                # the spawn look already covered the spawn
    rf.begin_decision(2)
    cells = np.array([[10, 10], [10, 11], [11, 10]], np.int32)
    rf._track(0, cells)
    assert rf.redundant_cells[0] == 0                # nobody else had seen them
    rf._track(1, cells)
    assert rf.redundant_cells[1] == 0                # ... and within a decision they are equals
    rf.commit_decision()
    rf.begin_decision(2)
    rf._track(1, cells)
    assert rf.redundant_cells[1] == 3                # now robot 0 had them first
    rf._track(1, cells)
    assert rf.redundant_cells[1] == 3                # each cell counts once per decision


def test_redundancy_refund_needs_a_casualty_in_a_redundant_cell():
    env = _env(_cfg(robots=2))
    rf = env.rf
    hs = env.raster.humans
    cas = np.flatnonzero(hs["role_id"] == CASUALTY_ROLE_ID)
    byst = np.flatnonzero(hs["role_id"] != CASUALTY_ROLE_ID)
    k, kb = int(cas[0]), int(byst[0])
    rb = env.state.robots[0]
    for idx, expect in ((k, True), (kb, False)):
        rf.begin_decision(2)
        rf.human_hits[:] = 0
        rf.human_found[:] = False
        i, j = rf._human_ij[idx]
        rf.seen_by[i, j] = np.uint16(1 << 1)         # robot 1 had covered that cell
        for _ in range(env.cfg.rayfronts.found_hits):
            rf._count_hit(idx, 0.0, rb, [])
        assert bool(rf.redundancy_refund[0]) is expect
    rf.begin_decision(2)                             # no peer had the cell: no refund
    rf.human_hits[:] = 0
    rf.human_found[:] = False
    i, j = rf._human_ij[k]
    rf.seen_by[i, j] = np.uint16(1)
    for _ in range(env.cfg.rayfronts.found_hits):
        rf._count_hit(k, 0.0, rb, [])
    assert not rf.redundancy_refund[0]


def test_reward_is_exactly_its_documented_terms():
    env = _env(_cfg(t_max=200.0))
    rw = env.cfg.reward
    pol = make_policy("ray_follower", queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    seen_red, seen_rev = 0, 0
    for _ in range(40):
        obs, r, done, info = env.step(pol.act(obs, env.state))
        n_r = len(env.state.robots)
        red = info["redundant_cells"] / np.maximum(info["observed_cells"], 1)
        red[info["redundancy_refunds"]] = 0.0
        assert np.allclose(info["redundancy_cost"], rw.redundancy_cost * red / n_r)
        rev = info["intentional_revisits"] - info["revisit_refunds"]
        expect = (rw.casualty_reward * info["new_found"] - rw.time_cost
                  - rw.redundancy_cost * red.mean()
                  - rw.revisit_cost * rev.sum())
        assert r == pytest.approx(expect, abs=1e-9)
        assert np.allclose(info["revisit_penalties"], rw.revisit_cost * rev)
        seen_red += int(info["redundant_cells"].sum())
        seen_rev += int(info["intentional_revisits"].sum())
        if done:
            break
    assert seen_red > 0, "3 robots from one spawn must re-cover each other's ground"
    m = info["metrics"]
    assert m["redundant_cells"] >= seen_red and m["intentional_revisits"] >= seen_rev


def test_redundancy_is_a_fraction_and_is_bounded_by_its_cost():
    """The term is `cost * redundant / observed`, so a fully redundant decision costs exactly
    `redundancy_cost` (0.05) however many cells the footprint holds, and never more."""
    env = _env(_cfg(t_max=200.0))
    rw = env.cfg.reward
    pol = make_policy("ray_follower", queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    fracs = []
    for _ in range(40):
        obs, r, done, info = env.step(pol.act(obs, env.state))
        red, seen = info["redundant_cells"], info["observed_cells"]
        assert (red <= seen).all()
        assert info["redundancy_cost"].sum() <= rw.redundancy_cost + 1e-12
        fracs.append(float((red / np.maximum(seen, 1)).mean()))
        if done:
            break
    assert 0.0 < max(fracs) <= 1.0
    assert env.cfg.reward.revisit_cost == 0.5


def test_a_fully_redundant_decision_costs_exactly_the_coefficient():
    env = _env(_cfg(robots=2, t_max=100.0))
    rf = env.rf
    rf.seen_by[:] = 0
    rf.begin_decision(2)
    cells = np.stack(np.nonzero(np.ones((6, 6), bool)), 1).astype(np.int32) + 20
    rf._track(0, cells)
    rf.commit_decision()
    rf.begin_decision(2)
    rf._track(1, cells)
    assert rf.redundant_cells[1] == rf.observed_cells[1] == cells.shape[0]
    frac = rf.redundant_cells[1] / rf.observed_cells[1]
    assert env.cfg.reward.redundancy_cost * frac == pytest.approx(0.05)


def test_terms_can_be_switched_off():
    env = _env(_cfg(redundancy_cost=0.0, revisit_cost=0.0, t_max=100.0))
    pol = make_policy("nearest_frontier", queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    for _ in range(10):
        obs, r, done, info = env.step(pol.act(obs, env.state))
        assert r == pytest.approx(info["new_found"] - env.cfg.reward.time_cost)


def test_refund_zeroes_only_that_robots_term():
    env = _env(_cfg(robots=2, t_max=100.0))
    pol = make_policy("nearest_frontier", queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    obs, r, done, info = env.step(pol.act(obs, env.state))
    rf = env.rf
    # force a refund for robot 0 only and re-derive the reward the env would have paid
    red = info["redundant_cells"].astype(np.float64).copy()
    refunds = np.array([True, False])
    red[refunds] = 0.0
    assert red[1] == info["redundant_cells"][1]
    assert red[0] == 0.0


def test_a_target_the_robot_was_already_sitting_on_is_not_a_visit():
    """Re-selecting the token you are parked on is not a journey: no record, no revisit charge
    (otherwise a policy that holds pays 0.5 per decision and the record list fills with copies)."""
    env = _env(_cfg())
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=1)
    env._begin_decision()
    rb = env.state.robots[0]
    rb.target_xy = here.copy()
    rb.target_token_type = TOKEN_SEGMENT
    rb.target_id = 1
    rb.target_feat = np.zeros(env.rf.D, np.float32)
    env._on_arrive(rb, env.state.t)
    assert env._revisits[0] == 0 and not env.visits[1:]


# ---- exact arithmetic (QA pass 2026-08-21) ------------------------------------------------------
def test_a_scripted_three_robot_episode_adds_up_term_by_term():
    """Every decision re-derived from the raw counters, and the episode total from the sums."""
    env = _env(_cfg(robots=3, t_max=100.0))
    rw = env.cfg.reward
    pol = make_policy("ray_follower", queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    finds, red_sum, revisits, n_dec, gross = 0, 0.0, 0, 0, 0
    fracs = []
    while True:
        obs, r, done, info = env.step(pol.act(obs, env.state))
        n_dec += 1
        raw = info["redundant_cells"] / np.maximum(info["observed_cells"], 1)
        assert ((raw >= 0.0) & (raw <= 1.0)).all()
        fracs.append(float(raw.mean()))
        charged = raw.copy()
        charged[info["redundancy_refunds"]] = 0.0
        rev = int(info["intentional_revisits"].sum() - info["revisit_refunds"].sum())
        expect = (rw.casualty_reward * info["new_found"] - rw.time_cost
                  - rw.redundancy_cost * charged.mean()
                  - rw.revisit_cost * rev)
        assert r == pytest.approx(expect, abs=1e-12)
        finds += int(info["new_found"])
        red_sum += float(charged.mean())
        revisits += rev
        gross += int(info["intentional_revisits"].sum())
        if done:
            break
    total = (rw.casualty_reward * finds - rw.time_cost * n_dec
             - rw.redundancy_cost * red_sum - rw.revisit_cost * revisits)
    assert env.state.cum_reward == pytest.approx(total, abs=1e-9)
    m = info["metrics"]
    assert m["n_decisions"] == n_dec
    # the metric column is the same team mean the reward charges (before the refund)
    assert m["redundancy_frac"] == pytest.approx(float(np.mean(fracs)), abs=1e-9)
    assert m["intentional_revisits"] == gross
    assert m["intentional_revisits"] - m["revisit_refunds"] == revisits
    assert m["visits"] == len(env.visits)


def test_a_robot_that_parks_for_ever_pays_only_the_time_cost():
    cfg = _cfg(robots=1, t_max=60.0)
    env = DisasterEnv(make_synthetic_scene(0, region_m=(200.0, 200.0), n_casualties=0,
                                           n_bystanders=3), cfg, seed=0)
    n = 0
    while True:
        obs, r, done, info = env.step(np.zeros(1, np.int64))     # slot 0 is `hold`
        n += 1
        assert info["redundant_cells"].sum() == 0, "one robot can never be redundant"
        assert info["intentional_revisits"].sum() == 0
        assert r == pytest.approx(-cfg.reward.time_cost, abs=1e-12)
        if done:
            break
    assert n == 12 and env.state.cum_reward == pytest.approx(-0.12, abs=1e-12)
    assert not env.visits, "holding is not a journey, so it writes no record"


def test_the_refund_zeroes_only_the_decision_that_earned_it():
    env = _env(_cfg(robots=2, t_max=100.0))
    pol = make_policy("nearest_frontier", queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    obs, _, _, i1 = env.step(pol.act(obs, env.state))
    assert not i1["redundancy_refunds"].any()
    real = env.rf.commit_decision

    def refund_robot_0():
        env.rf.redundancy_refund[0] = True
        real()

    env.rf.commit_decision = refund_robot_0
    obs, _, _, i2 = env.step(pol.act(obs, env.state))
    env.rf.commit_decision = real
    assert i2["redundancy_refunds"][0] and not i2["redundancy_refunds"][1]
    assert i2["redundancy_cost"][0] == 0.0
    assert i2["redundant_cells"][0] > 0, "it was redundant; it is simply not charged"
    if i2["redundant_cells"][1] > 0:
        assert i2["redundancy_cost"][1] > 0.0, "the other robot still pays"
    obs, _, _, i3 = env.step(pol.act(obs, env.state))
    assert not i3["redundancy_refunds"].any()
    if i3["redundant_cells"][0] > 0:
        assert i3["redundancy_cost"][0] > 0.0, "the refund does not carry over"


def test_an_arrival_writes_exactly_one_record():
    env = _env(_cfg())
    here = env.state.robots[0].pos.copy()
    env._begin_decision()
    _arrive(env, 0, here + np.array([20.0, 0.0]))
    assert len(env.visits) == 1 and len(env._dec_visits) == 1
    env._begin_decision()                                   # a later journey to the same place
    _arrive(env, 0, here + np.array([20.0, 0.0]))
    assert len(env.visits) == 2
    assert [v.seq for v in env.visits] == [0, 1] and env.visits[0].id != env.visits[1].id
    assert env.visits[1].robot == 0 and env._revisits[0] == 0, "its own record costs nothing"


def test_revisit_and_redundancy_switch_off_independently():
    for red, rev in ((0.0, 0.5), (0.05, 0.0)):
        env = _env(_cfg(robots=2, t_max=60.0, redundancy_cost=red, revisit_cost=rev))
        pol = make_policy("ray_follower", queries=env.cfg.rayfronts.queries, seed=0)
        obs = env.state.last_obs
        while True:
            obs, r, done, info = env.step(pol.act(obs, env.state))
            if red == 0.0:
                assert info["redundancy_cost"].sum() == 0.0
            if rev == 0.0:
                assert info["revisit_penalties"].sum() == 0.0
            if done:
                break


# ---- revisit refund on a find -------------------------------------------------------------------
def _cas_xy(env, n: int = 1):
    hs = env.raster.humans
    cas = np.flatnonzero(hs["role_id"] == CASUALTY_ROLE_ID)[:n]
    return [(int(k), np.array([hs["x"][k], hs["y"][k]], np.float64)) for k in cas]


def test_a_revisit_that_finds_a_casualty_is_refunded():
    """The same principle as the redundancy refund: an arrival that turns up somebody was not a
    wasted journey, so the 0.5 is handed back."""
    env = _env(_cfg(robots=2))
    k, xy = _cas_xy(env)[0]
    _record(env, xy, robot=1)
    env._begin_decision()
    _arrive(env, 0, xy)
    assert env._revisits[0] == 1
    env.rf.found_this_decision.append((0, k))
    env._settle_revisits()
    assert env._revisit_refunds[0] == 1
    assert not env._pending_revisits, "a refunded arrival is closed"


def test_a_fruitless_revisit_still_pays():
    env = _env(_cfg(robots=2))
    xy = env.state.robots[0].pos.copy() + np.array([80.0, 0.0])
    _record(env, xy, robot=1)
    env._begin_decision()
    _arrive(env, 0, xy)
    env._settle_revisits()               # nothing found this decision
    assert env._revisits[0] == 1 and env._revisit_refunds[0] == 0


def test_the_refund_needs_the_find_to_be_this_robots_and_near_the_target():
    env = _env(_cfg(robots=2))
    k, xy = _cas_xy(env)[0]
    for who, off, expect in ((1, 0.0, 0), (0, 400.0, 0), (0, 0.0, 1)):
        env._begin_decision()
        env._pending_revisits = []
        _record(env, xy + np.array([off, 0.0]), robot=1)
        _arrive(env, 0, xy + np.array([off, 0.0]))
        env.rf.found_this_decision[:] = [(who, k)]
        env._settle_revisits()
        assert env._revisit_refunds[0] == expect, (who, off)


def test_a_bystander_does_not_refund_a_revisit():
    env = _env(_cfg(robots=2))
    hs = env.raster.humans
    kb = int(np.flatnonzero(hs["role_id"] != CASUALTY_ROLE_ID)[0])
    xy = np.array([hs["x"][kb], hs["y"][kb]], np.float64)
    _record(env, xy, robot=1)
    env._begin_decision()
    _arrive(env, 0, xy)
    env.rf.found_this_decision.append((0, kb))
    env._settle_revisits()
    assert env._revisit_refunds[0] == 0, "bystanders are worth exactly zero"


def test_the_refund_carries_while_the_robot_is_still_on_the_target():
    """`found_hits` looks can take another decision: the charge stays open until it leaves."""
    env = _env(_cfg(robots=2))
    k, xy = _cas_xy(env)[0]
    rb = env.state.robots[0]
    _record(env, xy, robot=1)
    env._begin_decision()
    _arrive(env, 0, xy)
    rb.pos[:2] = xy                                  # parked on the target
    env._settle_revisits()
    assert env._revisit_refunds[0] == 0 and len(env._pending_revisits) == 1
    env._begin_decision()                            # next decision, still there, now it finds
    env.rf.found_this_decision.append((0, k))
    env._settle_revisits()
    assert env._revisit_refunds[0] == 1 and not env._pending_revisits


def test_a_pending_revisit_is_dropped_once_the_robot_leaves():
    env = _env(_cfg(robots=2))
    k, xy = _cas_xy(env)[0]
    rb = env.state.robots[0]
    _record(env, xy, robot=1)
    env._begin_decision()
    _arrive(env, 0, xy)
    rb.pos[:2] = xy + np.array([100.0, 0.0])         # flew off
    env._settle_revisits()
    assert not env._pending_revisits
    env._begin_decision()
    env.rf.found_this_decision.append((0, k))
    env._settle_revisits()
    assert env._revisit_refunds[0] == 0, "a find after it left refunds nothing"


def test_the_refund_can_be_switched_off():
    env = _env(_cfg(robots=2, revisit_refund_on_find=False))
    k, xy = _cas_xy(env)[0]
    _record(env, xy, robot=1)
    env._begin_decision()
    _arrive(env, 0, xy)
    env.rf.found_this_decision.append((0, k))
    env._settle_revisits()
    assert env._revisits[0] == 1 and env._revisit_refunds[0] == 0
    assert not env._pending_revisits


def test_the_refunded_revisit_is_not_charged_in_the_reward():
    """The team pays `revisit_cost * (revisits - refunds)`, and the metric keeps both."""
    env = _env(_cfg(robots=3, t_max=200.0))
    rw = env.cfg.reward
    pol = make_policy("ray_follower", queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    gross = refunds = 0
    while True:
        obs, r, done, info = env.step(pol.act(obs, env.state))
        net = info["intentional_revisits"] - info["revisit_refunds"]
        assert np.allclose(info["revisit_penalties"], rw.revisit_cost * net)
        assert (info["revisit_refunds"] >= 0).all()
        gross += int(info["intentional_revisits"].sum())
        refunds += int(info["revisit_refunds"].sum())
        if done:
            break
    m = info["metrics"]
    assert m["intentional_revisits"] == gross and m["revisit_refunds"] == refunds
    assert m["revisit_penalties"] == pytest.approx(rw.revisit_cost * (gross - refunds), abs=1e-9)


# ---- confirmed-only revisit --------------------------------------------------------------------
def test_an_unconfirmed_record_does_not_charge_by_default():
    """`revisit_confirmed_only` (default): a record whose visit confirmed nothing is free to
    revisit — occluded detection is stochastic, so the target may still hold a casualty."""
    env = _env(_cfg())
    here = env.state.robots[0].pos.copy()
    rec = _record(env, here, robot=1, n_found=0)
    env._begin_decision()
    _arrive(env, 0, here)
    assert env._revisits[0] == 0
    rec.n_found = 1                                   # now it is a confirmed target
    env._begin_decision()
    _arrive(env, 0, here + np.array([2.0, 0.0]))
    assert env._revisits[0] == 1


def test_confirmed_only_off_restores_the_old_rule():
    env = _env(_cfg(revisit_confirmed_only=False))
    here = env.state.robots[0].pos.copy()
    _record(env, here, robot=1)
    env._begin_decision()
    _arrive(env, 0, here)
    assert env._revisits[0] == 1


def test_the_confirmed_copy_wins_the_gossip_merge():
    """`receive_visited` keeps the higher `n_found`, so a robot that first heard the unconfirmed
    version is charged once the confirmed copy reaches it."""
    env = _env(_cfg("range"))
    here = env.state.robots[0].pos.copy()
    rec = _record(env, here, robot=1, n_found=0)
    env.comms.beliefs[0].receive_visited([rec])       # the unconfirmed copy arrives first
    env._begin_decision()
    _arrive(env, 0, here)
    assert env._revisits[0] == 0
    confirmed = VisitRecord(xy=rec.xy.copy(), token_type=rec.token_type, token_id=rec.token_id,
                            feat=rec.feat.copy(), t=rec.t, robot=rec.robot, seq=rec.seq,
                            n_found=2, id=rec.id)
    env.comms.beliefs[0].receive_visited([confirmed])
    env._begin_decision()
    _arrive(env, 0, here + np.array([2.0, 0.0]))
    assert env._revisits[0] == 1
