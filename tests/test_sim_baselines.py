"""The coverage and assignment baselines: `lawnmower` and `oracle_assign` (CONTRACTS.md 7).

`lawnmower` acts in **waypoints** — float `[n_robots, 2]` world points, nan = hold — so its tests
read the points it flies to, not a token index. The divert tests drive `LawnmowerPolicy.act` on a
hand-built `TeamObs` rather than an episode: a far-field human ray needs an `open`-visibility
casualty to land in one 20 degree bin at the right range, which is not something a scene seed can
be asked for reproducibly. `_obs` puts a token whose normalised position is `(xn, yn)` at the world
point `(xn, yn) * 100`, so an expected waypoint reads straight off the item list.
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


def _flat(region=REGION):
    """An empty, undamaged region: the scene a textbook sweep should cover completely."""
    return Scene(meta=Meta(region=(0.0, 0.0, region[0], region[1])),
                 damage_field=DamageField(kind="uniform", params={"inside": 0.0}))


def _episode(env, policy, seed=0, n=None):
    obs = env.reset(seed)
    policy.reset(seed)
    k = 0
    while True:
        obs, _, done, info = env.step(policy.act(obs, env.state))
        k += 1
        if done or (n is not None and k >= n):
            return obs, info, k


def _waypoint_episode(env, policy, seed=0, count_diverts=False):
    """One episode of a waypoint policy -> the [decisions, n_robots, 2] points it flew to."""
    obs = env.reset(seed)
    policy.reset(seed)
    out, diverts = [], 0
    while True:
        a = policy.act(obs, env.state)
        out.append(np.asarray(a, np.float64).copy())
        diverts += len(policy._chasing)
        obs, _, done, info = env.step(a)
        if done:
            w = np.array(out)
            return (w, info, diverts) if count_diverts else (w, info)


# ---- a hand-built observation ------------------------------------------------------------------
def _obs(items, n_robots=1, dim=D, region=None, pos=None):
    """`items` = list of (type, id, x_norm, y_norm, dist_norm, class_name|None) per robot slot.

    A `class_name` gives the token the unit class embedding of that class, which is exactly what a
    ray of that class carries; `None` leaves the feature tail zero. `region` fills the observation's
    own `(x0, y0, x1, y1)` metadata and `pos` the robot poses `robot_feat` carries in [-1, 1] — the
    two things the waypoint sweep reads. Without a region there is nothing to cut into strips.
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
    rfeat = np.zeros((n_robots, 18), np.float32)
    reg = None
    if region is not None:
        reg = np.asarray(region, np.float32)
        x0, y0, x1, y1 = (float(v) for v in reg)
        if x1 > x0 and y1 > y0:
            p = (np.zeros((n_robots, 2)) if pos is None
                 else np.asarray(pos, np.float64).reshape(n_robots, 2))
            rfeat[:, 0] = 2.0 * (p[:, 0] - x0) / (x1 - x0) - 1.0
            rfeat[:, 1] = 2.0 * (p[:, 1] - y0) / (y1 - y0) - 1.0
    return TeamObs(tokens=tok, token_mask=mask, token_xy=xy, token_type=tt, token_id=tid,
                   robot_feat=rfeat, bev=np.zeros((1, 4, 4), np.float32),
                   query_emb=np.zeros((8, dim), np.float32), query_w=np.zeros(8, np.float32),
                   query_mask=np.zeros(8, np.bool_), t=0.0, region=reg)


HOLD = (TOKEN_HOLD, -1, 0.0, 0.0, 0.0, None)


def _stub_state(region=(0.0, 0.0, 200.0, 200.0), pos=(0.0, 0.0), n=1):
    """The only three things the sweep reads off the state: the region, the sensor and the pose."""
    return SimpleNamespace(cfg=EnvConfig(), emb=None,
                           raster=SimpleNamespace(region=region),
                           robots=[SimpleNamespace(pos=np.asarray(pos, np.float64))
                                   for _ in range(n)])


# ---- registration -------------------------------------------------------------------------------
def test_both_policies_are_registered():
    from rlplanner.sim.baselines import POLICIES
    assert {"lawnmower", "oracle_assign"} <= set(POLICIES)
    assert not make_policy("lawnmower").privileged
    assert make_policy("oracle_assign").privileged
    # the action shape is a class fact, so a caller can branch on it without running an episode
    assert getattr(POLICIES["lawnmower"], "waypoint_policy", False)
    assert not getattr(POLICIES["oracle_assign"], "waypoint_policy", False)


def test_oracle_assign_needs_the_state():
    env = DisasterEnv(_scene(), _cfg())
    with pytest.raises(ValueError):
        make_policy("oracle_assign").act(env.state.last_obs, None)


# ---- lawnmower: the waypoint sweep ---------------------------------------------------------------
def test_lawnmower_acts_in_waypoints_not_tokens():
    """A stripe pattern flies over ground that is already mapped, where no token is ever offered:
    the action is a world point (CONTRACTS.md 6), one row per robot."""
    env = DisasterEnv(_scene(), _cfg(robots=3))
    assert LawnmowerPolicy.waypoint_policy
    a = make_policy("lawnmower").act(env.state.last_obs, env.state)
    assert a.shape == (3, 2) and a.dtype.kind == "f"


def test_the_strips_partition_the_region():
    """`strips` is the partition itself, exposed so a viewer draws the same one the sweep flies."""
    reg = (10.0, -5.0, 210.0, 95.0)
    for n in (1, 2, 3, 7, 10):
        st = LawnmowerPolicy.strips(reg, n)
        assert len(st) == n
        assert st[0][0] == pytest.approx(reg[0]) and st[-1][1] == pytest.approx(reg[2])
        w = [hi - lo for lo, hi in st]
        assert max(w) - min(w) < 1e-9                        # equal width
        for i in range(1, n):
            assert st[i][0] == pytest.approx(st[i - 1][1])   # contiguous and disjoint
    assert LawnmowerPolicy.strips(None, 3) is None
    assert LawnmowerPolicy.strips((0.0, 0.0, 0.0, 10.0), 3) is None      # degenerate: no strips


def test_the_waypoint_list_is_a_vertical_serpentine():
    """Four 50 m lanes over a 200 m region: +y on an even lane, -y on an odd one, the ends inset
    half a swath so the footprint still reaches the boundary."""
    way = LawnmowerPolicy(swath_m=50.0)._strip_lanes((0.0, 0.0, 200.0, 200.0), 1, 0, 50.0)
    assert way.tolist() == [[25.0, 25.0], [25.0, 175.0], [75.0, 175.0], [75.0, 25.0],
                            [125.0, 25.0], [125.0, 175.0], [175.0, 175.0], [175.0, 25.0]]


def test_every_robot_gets_its_own_lanes():
    pol = LawnmowerPolicy(swath_m=50.0)
    reg = (0.0, 0.0, 200.0, 200.0)
    xs = [np.unique(pol._strip_lanes(reg, 4, r, 50.0)[:, 0]).tolist() for r in range(4)]
    assert xs == [[25.0], [75.0], [125.0], [175.0]]


def test_lawnmower_sweeps_an_empty_scene_to_full_coverage():
    env = DisasterEnv(_flat((160.0, 160.0)), _cfg(robots=3, t_max=400.0))
    _, info = _waypoint_episode(env, make_policy("lawnmower"))
    assert info["coverage"] > 0.95


@pytest.mark.parametrize("robots", [1, 8, 10])      # 10 = the largest EnvConfig.validate allows
def test_lawnmower_runs_from_one_robot_to_the_largest_team(robots):
    env = DisasterEnv(_scene(), _cfg(robots=robots, t_max=120.0))
    w, info = _waypoint_episode(env, make_policy("lawnmower"))
    assert w.shape[1:] == (robots, 2) and info["coverage"] > 0.0


@pytest.mark.parametrize("robots", [1, 8])
def test_the_sweep_is_piecewise_constant_x_lanes_inside_a_disjoint_strip(robots):
    """The lawnmower property itself: on an empty scene each robot's waypoints are a handful of
    lane x values no more than a swath apart, alternating between the two ends of the region in y,
    all inside its own strip — and the ride between them is the lane, not a zig-zag."""
    cfg = _cfg(robots=robots, t_max=900.0)
    cfg.rayfronts.p_fp_ray = 0.0        # a spurious person ray would (rightly) divert off the lane
    env = DisasterEnv(_flat((240.0, 240.0)), cfg)
    pol = make_policy("lawnmower")
    w, _ = _waypoint_episode(env, pol)
    swath = pol._swath(env.state)
    strips = LawnmowerPolicy.strips(env.raster.region, robots)
    seen = []
    for r in range(robots):
        pts = w[:, r][np.isfinite(w[:, r]).all(axis=1)]
        assert pts.size, r
        lanes = np.unique(np.round(pts[:, 0], 6))
        assert (lanes >= strips[r][0]).all() and (lanes <= strips[r][1]).all()
        d = np.diff(lanes)
        assert d.size == 0 or ((d <= swath + 1e-6).all() and (d >= 0.5 * swath).all())
        assert np.unique(np.round(pts[:, 1], 6)).size == 2       # the two ends of the strip in y
        seen.append(lanes)
        traj = np.array(env.state.robots[r].trajectory)
        inside = (traj[:, 0] >= strips[r][0] - 1e-9) & (traj[:, 0] <= strips[r][1] + 1e-9)
        first = int(np.argmax(inside))
        assert inside.any() and inside[first:].all(), r    # once in its strip it never leaves it
        near = np.abs(traj[first:, 0][:, None] - lanes[None, :]).min(axis=1)
        assert (near <= 1.0).mean() > 0.7, r
    for r in range(1, robots):
        assert seen[r - 1].max() < seen[r].min()             # and no two robots share ground


def test_a_robot_with_no_region_to_sweep_holds():
    """No region on the observation and no state to read one off: a waypoint policy that guessed
    the extent from the tokens it happens to hold would be inventing the map. The row is nan."""
    a = make_policy("lawnmower").act(_obs([HOLD]), None)
    assert a.shape == (1, 2) and not np.isfinite(a).any()


def test_lawnmower_with_no_tokens_at_all_does_not_crash():
    obs = _obs([HOLD], region=(0.0, 0.0, 200.0, 200.0))
    obs.token_mask[:] = False
    a = make_policy("lawnmower").act(obs, None)
    assert a.shape == (1, 2) and np.isfinite(a).all()      # the sweep needs no token


def test_the_cursor_advances_only_once_the_robot_arrives():
    reg = (0.0, 0.0, 200.0, 200.0)
    pol = LawnmowerPolicy(swath_m=100.0, arrive_radius_m=3.0)

    def at(x, y):
        return pol.act(_obs([HOLD], region=reg, pos=(x, y)),
                       _stub_state(region=reg, pos=(x, y)))[0]

    assert np.allclose(at(0.0, 0.0), [50.0, 50.0])
    assert np.allclose(at(10.0, 10.0), [50.0, 50.0])       # still on its way: same waypoint
    assert np.allclose(at(50.0, 51.0), [50.0, 150.0])      # arrived: the other end of lane 0
    assert np.allclose(at(50.0, 149.0), [150.0, 150.0])    # ... and on to lane 1


def test_an_unreachable_waypoint_does_not_park_the_robot_forever():
    """A lane end inside a building leaves A* nothing to plan and the robot does not move; after
    `_STALL_DECISIONS` motionless sweep decisions the cursor steps past it."""
    reg = (0.0, 0.0, 200.0, 200.0)
    pol = LawnmowerPolicy(swath_m=100.0)
    ob, st = _obs([HOLD], region=reg, pos=(0.0, 0.0)), _stub_state(region=reg, pos=(0.0, 0.0))
    seen = [pol.act(ob, st)[0].copy() for _ in range(4)]
    assert np.allclose(seen[0], [50.0, 50.0]) and np.allclose(seen[1], [50.0, 50.0])
    assert np.allclose(seen[2], [50.0, 150.0]) and np.allclose(seen[3], [50.0, 150.0])


# ---- lawnmower: investigate ---------------------------------------------------------------------
def test_a_human_ray_diverts_the_sweep():
    """The waypoint becomes the ray's own target point (`_obs` puts a token at (xn, yn) * 100)."""
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
             (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_prone")]
    assert np.allclose(make_policy("lawnmower").act(_obs(items), None)[0], [50.0, 50.0])


def test_a_container_ray_does_not_divert_the_sweep():
    for cls in ("vehicle_toppled", "building_damaged", "debris"):
        items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
                 (TOKEN_RAY, 7, 0.5, 0.5, 0.6, cls)]
        assert not np.isfinite(make_policy("lawnmower").act(_obs(items), None)).any(), cls


def test_a_human_ray_outside_the_robots_own_strip_is_ignored():
    """The partition is strict: a ray across the boundary is a teammate's to investigate, and
    chasing it would both break the strip and put two robots on one body."""
    reg = (-100.0, -100.0, 100.0, 100.0)
    row = [HOLD, (TOKEN_RAY, 7, 0.6, 0.2, 0.5, "human_prone")]      # world (60, 20)
    obs = _obs([row, list(row)], n_robots=2, region=reg, pos=[(-50.0, 0.0), (50.0, 0.0)])
    a = make_policy("lawnmower").act(obs, None)
    assert np.allclose(a[1], [60.0, 20.0])         # robot 1 owns that ground and investigates
    assert a[0, 0] <= 0.0                          # robot 0 keeps mowing its own strip


def test_the_nearest_human_ray_wins():
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
             (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_standing"),
             (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone"),
             (TOKEN_RAY, 9, 0.2, -0.3, 0.9, "human_prone")]
    assert np.allclose(make_policy("lawnmower").act(_obs(items), None)[0], [-40.0, 20.0])


def test_the_investigation_is_sticky_until_the_ray_goes_away():
    near = (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")
    chased = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None), near]
    pol = make_policy("lawnmower")
    assert np.allclose(pol.act(_obs(chased), None)[0], [-40.0, 20.0])
    # a nearer human ray appears: the robot stays on the one it is already investigating
    with_nearer = chased + [(TOKEN_RAY, 11, 0.1, 0.1, 0.05, "human_prone")]
    assert np.allclose(pol.act(_obs(with_nearer), None)[0], [-40.0, 20.0])
    # the ray resolves (leaves the token set): back to the sweep, which has no region here
    assert not np.isfinite(pol.act(_obs(chased[:2]), None)).any()


def test_the_sweep_resumes_at_the_waypoint_the_divert_interrupted():
    reg = (-100.0, -100.0, 100.0, 100.0)
    ray = (TOKEN_RAY, 9, 0.1, 0.1, 0.3, "human_prone")
    pol = make_policy("lawnmower")
    st = _stub_state(region=reg, pos=(0.0, 0.0))
    before = pol.act(_obs([HOLD], region=reg, pos=(0.0, 0.0)), st)[0].copy()
    assert np.allclose(pol.act(_obs([HOLD, ray], region=reg, pos=(0.0, 0.0)), st)[0], [10.0, 10.0])
    after = pol.act(_obs([HOLD], region=reg, pos=(0.0, 0.0)), st)[0]
    assert np.allclose(after, before)              # the cursor did not move while it was away


def test_reset_forgets_the_investigation():
    items = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
             (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")]
    pol = make_policy("lawnmower")
    pol.act(_obs(items), None)
    assert pol._chasing
    pol.reset(0)
    assert not pol._chasing and not pol._cursor


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


def test_only_the_robot_whose_strip_holds_the_ray_investigates_it():
    reg = (-100.0, -100.0, 100.0, 100.0)
    row = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.5, 0.2, None),
           (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_prone")]              # world (50, 50)
    obs = _obs([row, list(row)], n_robots=2, region=reg, pos=[(-50.0, 0.0), (50.0, 0.0)])
    a = make_policy("lawnmower").act(obs, None)
    assert np.allclose(a[1], [50.0, 50.0]) and not np.allclose(a[0], [50.0, 50.0])


def test_the_divert_fires_in_a_real_episode():
    """The synthetic-obs tests pin the rule; this one says the rule actually meets a human ray
    on a live belief and the sweep still covers."""
    env = DisasterEnv(_scene(seed=0, region=(240.0, 240.0)), _cfg(robots=3, t_max=200.0))
    pol = make_policy("lawnmower", queries=env.cfg.rayfronts.queries)
    _, info, diverts = _waypoint_episode(env, pol, count_diverts=True)
    assert diverts > 0
    assert info["coverage"] > 0.8


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
    assert np.array_equal(runs[0][0], runs[1][0], equal_nan=runs[0][0].dtype.kind == "f")
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
    assert np.allclose(make_policy("lawnmower").act(obs, None)[0], [50.0, 50.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_ray_feature_never_diverts_and_never_warns(bad):
    obs = _obs([HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None),
                (TOKEN_RAY, 7, 0.5, 0.5, 0.6, None)])
    obs.tokens[0, 2, F_FEAT0:] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not np.isfinite(make_policy("lawnmower").act(obs, None)).any()


def test_a_ray_with_no_target_point_never_diverts():
    """`token_xy` is nan for a token that aims at nothing: there is no waypoint to hand the env."""
    obs = _obs([HOLD, (TOKEN_RAY, 7, 0.5, 0.5, 0.6, "human_prone")])
    obs.token_xy[0, 1] = np.nan
    assert not np.isfinite(make_policy("lawnmower").act(obs, None)).any()


# ---- lawnmower: degenerate geometry --------------------------------------------------------------
@pytest.mark.parametrize("alt,depth", [(25.0, 35.0), (34.999, 35.0), (35.0, 35.0), (60.0, 35.0)])
def test_the_lane_spacing_stays_finite_when_the_sensor_cannot_reach_the_ground(alt, depth):
    """`depth_limit <= flight_alt` is a config `validate` rejects, but the swath must not become a
    nan (sqrt of a negative) or a zero lane spacing if the sweep is ever handed one."""
    st = _stub_state()
    st.cfg.robot.flight_alt_m, st.cfg.sensor.depth_limit_m = alt, depth
    pol = make_policy("lawnmower")
    w = pol._swath(st)
    assert np.isfinite(w) and w > 0.0
    a = pol.act(_obs([HOLD], region=(0.0, 0.0, 200.0, 200.0)), st)
    assert np.isfinite(a).all()


@pytest.mark.parametrize("region", [(0.0, 0.0, 4.0, 4.0),          # a handful of cells
                                    (0.0, 0.0, 200.0, 0.0),        # zero height
                                    (0.0, 0.0, 500.0, 1500.0),     # tall
                                    (0.0, 0.0, 1500.0, 500.0)])    # wide
def test_the_sweep_survives_a_degenerate_or_lopsided_region(region):
    x0, y0, x1, y1 = region
    pol = make_policy("lawnmower")
    a = pol.act(_obs([HOLD], region=region), _stub_state(region=region))
    if x1 <= x0 or y1 <= y0:
        assert not np.isfinite(a).any()          # nothing to cut into strips: hold
    else:
        assert np.isfinite(a).all()
        assert x0 <= a[0, 0] <= x1 and y0 <= a[0, 1] <= y1


@pytest.mark.parametrize("robots", [1, 10])
def test_every_robot_of_a_crowded_team_gets_a_waypoint_in_its_own_strip(robots):
    """Strips thinner than one lane: still exactly one lane each, and no two robots share ground."""
    reg = (0.0, 0.0, 200.0, 200.0)
    obs = _obs([HOLD], n_robots=robots, region=reg, pos=[(100.0, 100.0)] * robots)
    a = make_policy("lawnmower").act(obs, _stub_state(region=reg, pos=(100.0, 100.0), n=robots))
    assert a.shape == (robots, 2) and np.isfinite(a).all()
    for r, (lo, hi) in enumerate(LawnmowerPolicy.strips(reg, robots)):
        assert lo <= a[r, 0] <= hi


# ---- lawnmower: the chase ends -------------------------------------------------------------------
def test_a_chase_that_resolves_mid_flight_moves_to_the_next_human_ray():
    base = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.2, None)]
    near = (TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")
    far = (TOKEN_RAY, 9, 0.6, 0.6, 0.8, "human_standing")
    pol = make_policy("lawnmower")
    assert np.allclose(pol.act(_obs(base + [near, far]), None)[0], [-40.0, 20.0])   # the nearer one
    assert np.allclose(pol.act(_obs(base + [far]), None)[0], [60.0, 60.0])          # on to the other


def test_the_chase_ends_when_the_ray_falls_inside_the_min_travel_guard():
    base = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None)]
    pol = make_policy("lawnmower")
    chased = base + [(TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")]
    assert np.allclose(pol.act(_obs(chased), None)[0], [-40.0, 20.0])
    arrived = base + [(TOKEN_RAY, 8, -0.4, 0.2, 0.0, "human_prone")]
    assert not np.isfinite(pol.act(_obs(arrived), None)).any()      # arrived: back to the sweep
    assert not pol._chasing


def test_a_ray_id_that_comes_back_as_something_else_drops_the_chase():
    base = [HOLD, (TOKEN_FRONTIER, 1, -0.9, -0.9, 0.5, None)]
    pol = make_policy("lawnmower")
    pol.act(_obs(base + [(TOKEN_RAY, 8, -0.4, 0.2, 0.25, "human_prone")]), None)
    assert pol._chasing[0] == 8
    ob = _obs(base + [(TOKEN_RAY, 8, 0.5, 0.5, 0.6, "vehicle_toppled"),
                      (TOKEN_RAY, 12, -0.3, 0.1, 0.3, "human_standing")])
    assert np.allclose(pol.act(ob, None)[0], [-30.0, 10.0])


def test_a_sweep_that_diverts_to_every_ray_still_covers():
    """The resolution rule is what keeps the divert from livelocking: classify *every* ray as a
    person and the strip is still swept, because a chase ends when the robot arrives."""
    class AllHuman(LawnmowerPolicy):
        def _human_rays(self, obs, r, cls):
            return obs.token_mask[r] & (obs.token_type[r] == TOKEN_RAY)
    env = DisasterEnv(_scene(region=(200.0, 200.0)), _cfg(robots=3, t_max=300.0))
    _, info = _waypoint_episode(env, AllHuman(queries=env.cfg.rayfronts.queries))
    assert info["coverage"] > 0.9


def test_the_divert_runs_on_the_robots_own_view_under_range_comms():
    """Nothing in the rule reads the team map, so a blackout changes neither the classification nor
    the sweep: the robot investigates the human rays its own belief carries, inside its own strip."""
    cfg = _cfg(robots=3, t_max=200.0)
    cfg.comms = CommsConfig(mode="range", range_m=0.0)
    env = DisasterEnv(_scene(seed=0, region=(240.0, 240.0)), cfg)
    pol = make_policy("lawnmower", queries=cfg.rayfronts.queries)
    w, info, diverts = _waypoint_episode(env, pol, count_diverts=True)
    assert w.shape[1:] == (3, 2)
    assert diverts > 0 and info["coverage"] > 0.8


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


# ---- motion-only ideal bound -------------------------------------------------------------------
def test_ideal_routes_single_robot_line():
    from rlplanner.sim.ideal import euclidean_matrix, ideal_routes
    D = euclidean_matrix(np.array([[0.0, 0.0]]), np.array([[100.0, 0.0], [200.0, 0.0]]))
    arr, tours = ideal_routes(D, 1, speed=5.0)
    assert np.allclose(sorted(arr), [20.0, 40.0])


def test_ideal_bound_counts_only_within_horizon():
    from rlplanner.sim.ideal import euclidean_matrix, ideal_routes
    D = euclidean_matrix(np.array([[0.0, 0.0]]), np.array([[100.0, 0.0], [200.0, 0.0]]))
    arr, _ = ideal_routes(D, 1, speed=5.0)
    assert (arr <= 25.0).sum() == 1


def test_ideal_two_robots_split_clusters():
    from rlplanner.sim.ideal import euclidean_matrix, ideal_routes
    D = euclidean_matrix(np.array([[0.0, 0.0], [1000.0, 0.0]]),
                         np.array([[10.0, 0.0], [20.0, 0.0], [990.0, 0.0], [980.0, 0.0]]))
    arr, tours = ideal_routes(D, 2, speed=5.0)
    assert sorted(tours[0]) == [0, 1] and sorted(tours[1]) == [2, 3]


def test_obstacle_matrix_routes_around_a_wall():
    from rlplanner.sim.ideal import obstacle_matrix
    obst = np.zeros((21, 21), bool)
    obst[:20, 10] = True                        # wall with a gap at the bottom
    ij = np.array([[0, 0], [0, 20]])
    D = obstacle_matrix(obst, 1.0, ij)
    assert D[0, 1] > 20.0 + 5                   # detour well beyond the straight line
    assert np.isfinite(D[0, 1])


def test_blocked_poi_snaps_to_adjacent_free_cell():
    from rlplanner.sim.ideal import obstacle_matrix
    obst = np.zeros((11, 11), bool)
    obst[4:7, 4:7] = True                       # a target inside a tower footprint
    ij = np.array([[0, 0], [5, 5]])
    D = obstacle_matrix(obst, 1.0, ij)
    assert np.isfinite(D[0, 1]) and D[0, 1] > 0


def test_ideal_bound_on_real_env():
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.config import EnvConfig
    from rlplanner.sim.env import DisasterEnv
    from rlplanner.sim.ideal import ideal_bound
    cfg = EnvConfig(); cfg.robot.n_robots = 3; cfg.t_max_s = 300.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(200.0, 200.0), n_casualties=6,
                                           n_bystanders=2), cfg, seed=0)
    row = ideal_bound(env)
    assert 0.0 < row["frac_found"] <= 1.0 and 0.0 < row["finds_auc"] <= 1.0
    assert row["time_to_first"] <= row["time_to_half"] <= row["time_to_all"]


def test_ideal_via_evaluate_policy():
    from rlplanner.sim.config import EnvConfig
    from rlplanner.train.evaluate import evaluate_policy
    from rlplanner.train.scenes import SceneBank
    bank = SceneBank("synthetic:0-3", region_m=(200.0, 200.0))
    res = evaluate_policy("oracle_ideal", bank, EnvConfig(), episodes=2, robots=3, split="train")
    assert res["frac_found"].shape == (2,) and np.all(res["frac_found"] > 0)
