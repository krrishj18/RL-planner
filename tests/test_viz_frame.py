import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from rlplanner.scene import schema
from rlplanner.sim.config import EnvConfig
from rlplanner.viz.frame import render_frame, status_lines
from viz_mocks import empty_ray_store, make_mock_state, sim_available

needs_sim = pytest.mark.skipif(not sim_available(), reason="rlplanner.sim env mid-edit")


def _rgb_ok(img, figsize=(18.0, 6.0), dpi=100):
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[:2] == (int(figsize[1] * dpi), int(figsize[0] * dpi))
    assert img.max() > 0


@pytest.mark.parametrize("n_robots", [1, 3, 10])
def test_render_frame_robot_counts(n_robots):
    st = make_mock_state(2, n_robots=n_robots)
    _rgb_ok(render_frame(st, query_idx=0, focus_robot=0))


def test_render_frame_small_figure():
    st = make_mock_state(0, n_robots=2)
    img = render_frame(st, figsize=(9.0, 3.0), dpi=60, focus_robot=1)
    _rgb_ok(img, figsize=(9.0, 3.0), dpi=60)


def test_render_frame_zero_rays():
    st = make_mock_state(1, n_robots=2, n_rays=0)
    assert st.rays.n == 0
    _rgb_ok(render_frame(st, focus_robot=0))
    st.rays = empty_ray_store(st.emb.D)
    _rgb_ok(render_frame(st, focus_robot=0))


def test_render_frame_all_rays_resolved():
    st = make_mock_state(1, n_robots=2)
    st.rays.resolved[:] = True
    _rgb_ok(render_frame(st, focus_robot=0))


def test_render_frame_zero_frontiers():
    st = make_mock_state(1, n_robots=2)
    st.frontier_clusters = []
    st.frontier_mask[:] = False
    _rgb_ok(render_frame(st, focus_robot=0))


def test_render_frame_nothing_found_yet():
    st = make_mock_state(3, n_robots=2, n_found=0)
    assert not st.human_found.any()
    _rgb_ok(render_frame(st, focus_robot=0))


def test_render_frame_all_tokens_masked():
    st = make_mock_state(1, n_robots=3, all_masked=True)
    assert not st.last_obs.token_mask.any()
    _rgb_ok(render_frame(st, focus_robot=0))


def test_render_frame_no_last_obs_or_actions():
    st = make_mock_state(1, n_robots=2)
    st.last_obs = None
    st.last_actions = None
    _rgb_ok(render_frame(st, focus_robot=0))


def test_render_frame_focus_none():
    st = make_mock_state(1, n_robots=2)
    _rgb_ok(render_frame(st, focus_robot=None))


@pytest.mark.parametrize("bad", [-1, 2, 99, 1.5])
def test_focus_robot_out_of_range_raises(bad):
    st = make_mock_state(1, n_robots=2)
    with pytest.raises(IndexError):
        render_frame(st, focus_robot=bad)


@pytest.mark.parametrize("bad", [-1, 11, 999])
def test_query_idx_out_of_range_raises(bad):
    st = make_mock_state(1, n_robots=2)
    assert len(st.query_names()) == st.cfg.n_queries
    with pytest.raises(IndexError):
        render_frame(st, query_idx=bad)


@pytest.mark.parametrize("n_q", [1, 3, 8])
def test_every_key_indexes_a_query_for_any_list_length(n_q):
    """Keys 0-9 index the live mission list, which is 1 to `tokens.max_queries` long."""
    from rlplanner.viz.frame import resolve_query
    cfg = EnvConfig()
    names = ("person lying on the ground", "person", "rubble", "car", "collapsed building",
             "damaged building", "road", "tree")[:n_q]
    cfg.rayfronts.queries = names
    st = make_mock_state(1, n_robots=2, cfg=cfg)
    assert len(st.query_names()) == n_q
    for k in range(n_q):
        assert resolve_query(st, k) == (k, names[k])
        _rgb_ok(render_frame(st, query_idx=k, focus_robot=0))
    with pytest.raises(IndexError):
        resolve_query(st, n_q)


def test_every_query_renders():
    st = make_mock_state(4, n_robots=1)
    for q in range(st.cfg.n_queries):
        _rgb_ok(render_frame(st, query_idx=q, focus_robot=0))


def test_scene_without_humans():
    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(5, n_casualties=0, n_bystanders=0)
    st = make_mock_state(5, n_robots=2, cfg=cfg, scene=scene, n_found=0)
    assert st.n_casualties == 0
    _rgb_ok(render_frame(st, focus_robot=0))


def test_query_grid_shape_mismatch_raises():
    st = make_mock_state(1, n_robots=1)
    st.observed = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        render_frame(st, focus_robot=0)


def test_human_flag_length_mismatch_raises():
    st = make_mock_state(1, n_robots=1)
    st.human_found = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError):
        render_frame(st, focus_robot=0)


def test_status_lines_report_the_contract_fields():
    st = make_mock_state(1, n_robots=3)
    txt = "\n".join(status_lines(st, {"frontiers": 2, "rays": 5, "tokens": 9}, focus_robot=1))
    assert "found" in txt and "coverage" in txt and "reward" in txt
    assert f"{st.t:7.1f}" in txt
    for r in st.robots:
        assert f"{r.idx:<5d}" in txt


def test_frames_are_deterministic():
    a = render_frame(make_mock_state(7, n_robots=2), focus_robot=0)
    b = render_frame(make_mock_state(7, n_robots=2), focus_robot=0)
    assert np.array_equal(a, b)


def test_no_figures_leak():
    import matplotlib.pyplot as plt
    before = len(plt.get_fignums())
    for _ in range(3):
        render_frame(make_mock_state(1, n_robots=2), focus_robot=0)
    assert len(plt.get_fignums()) == before


def test_tiny_region_scene():
    scene = schema.make_synthetic_scene(0, region_m=(40.0, 40.0), block_m=20.0, n_casualties=1,
                                        n_bystanders=1)
    st = make_mock_state(0, n_robots=2, scene=scene, n_found=1)
    _rgb_ok(render_frame(st, focus_robot=0))


def test_bare_scene_without_roads_or_spawns():
    scene = schema.Scene(meta=schema.Meta(region=(0.0, 0.0, 60.0, 60.0)))
    st = make_mock_state(0, n_robots=1, scene=scene, n_found=0)
    assert st.scene.robots_spawn == []
    _rgb_ok(render_frame(st, focus_robot=0))


def test_robot_target_label_falls_back_to_the_chosen_token():
    from rlplanner.viz.frame import robot_target_label
    st = make_mock_state(1, n_robots=2)
    r = st.robots[0]
    from rlplanner.sim import state as S
    assert robot_target_label(st, r).startswith(tuple(S.TOKEN_TYPE_NAMES))
    r.target_xy = None
    st.last_actions = np.array([1, 1])
    lbl = robot_target_label(st, r)
    assert lbl != "-" and "#" in lbl
    st.last_obs = None
    assert robot_target_label(st, r) == "-"


# ---- queries by name / derived from the stored features -------------------------------------
def test_render_frame_accepts_a_query_name():
    st = make_mock_state(3, n_robots=2)
    name = st.cfg.rayfronts.queries[-1]
    k = len(st.cfg.rayfronts.queries) - 1
    _rgb_ok(render_frame(st, query_idx=name, focus_robot=0))
    _rgb_ok(render_frame(st, query=name, focus_robot=0))
    from rlplanner.viz.frame import belief_grid
    g, label, i = belief_grid(st, name)
    assert label == name and i == k
    assert np.allclose(g, np.asarray(st.query_sim(k), float))


def test_belief_grid_rejects_an_unknown_query_on_a_state_without_embeddings():
    from rlplanner.viz.frame import belief_grid
    st = make_mock_state(3, n_robots=1)
    st.emb = None
    with pytest.raises(KeyError, match="no embeddings"):
        belief_grid(st, "a fire hydrant")
    with pytest.raises(TypeError):
        belief_grid(st, 1.5)


def test_the_frame_takes_one_query_view_and_never_the_stored_grid_stack():
    """`EnvState.vox_sim` allocates one grid per mission query on every access. A frame must take
    exactly one view of the query it draws, not one per ray, panel or sub-step."""
    from rlplanner.viz.frame import query_view
    st = make_mock_state(3, n_robots=2)
    real = st.query_sim
    calls = []

    def counted(q):
        calls.append(q)
        return real(q)

    st.query_sim = counted
    qv = query_view(st, 0)
    assert len(calls) == 1
    assert qv.grid.shape == st.observed.shape
    assert qv.ray_sim.shape[0] == st.rays.n
    calls.clear()
    render_frame(st, query_idx=0, focus_robot=0)
    assert len(calls) == 1, f"the frame took {len(calls)} query views, expected 1"


def test_ray_colours_come_from_the_ray_features_not_a_stored_column():
    """`RayStore` has no `sims`: a ray's colour is the cosine of its peak feature (CONTRACTS 9)."""
    from rlplanner.viz.frame import query_view
    st = make_mock_state(3, n_robots=2)
    assert not hasattr(st.rays, "sims") and not hasattr(st.rays, "sims_max")
    qv = query_view(st, 0)
    q = st.emb.embed_queries((st.query_names()[0],))[0]
    f = st.rays.feat_peak
    want = np.clip((f @ q) / np.maximum(np.linalg.norm(f, axis=1), 1e-12), 0, 1)
    assert np.allclose(qv.ray_sim, want, atol=1e-5)


def test_a_different_query_repaints_the_belief_and_the_rays():
    st = make_mock_state(3, n_robots=2)
    cfg = st.cfg
    cfg.rayfronts.queries = ("person lying on the ground", "rubble", "car")
    st.queries = cfg.rayfronts.queries
    a = render_frame(st, query_idx=0, focus_robot=0)
    b = render_frame(st, query_idx="car", focus_robot=0)
    assert not np.array_equal(a, b)


# ---- segments -------------------------------------------------------------------------------
def test_segments_are_drawn_as_translucent_patches_in_their_feature_colour():
    from rlplanner.viz import palette as P
    from rlplanner.viz.frame import segment_rgba
    st = make_mock_state(3, n_robots=2)
    assert st.segments and st.seg_labels.max() >= 1
    rgba = segment_rgba(st)
    assert rgba.shape == st.observed.shape + (4,)
    assert (rgba[..., 3] > 0).any() and rgba[..., 3].max() <= 1.0
    assert (rgba[~st.observed, 3] == 0).all()          # a segment never covers unobserved space
    # each segment carries the PCA-RGB of its own mean feature, on the table's fixed basis
    s = st.segments[0]
    want = P.feat_rgb(np.asarray(s.feat)[None], st.emb)[0]
    assert np.allclose(rgba[s.ij[0], s.ij[1], :3], want, atol=1e-5)
    # two different classes get two different colours
    cols = {tuple(np.round(P.feat_rgb(np.asarray(x.feat)[None], st.emb)[0], 3))
            for x in st.segments[:8]}
    assert len(cols) > 1


def test_segments_can_be_turned_off():
    from matplotlib.figure import Figure
    from rlplanner.viz.frame import draw_belief_panel
    st = make_mock_state(3, n_robots=2)
    fig = Figure(figsize=(4, 4), dpi=50)
    on = draw_belief_panel(fig.add_subplot(211), st, legend=False)
    off = draw_belief_panel(fig.add_subplot(212), st, legend=False, segments=False)
    assert on["segments"] > 0 and off["segments"] == 0


def test_segment_overlay_is_empty_without_labels_or_an_embedding_table():
    from rlplanner.viz.frame import segment_rgba
    st = make_mock_state(3, n_robots=1)
    st.seg_labels = None
    assert segment_rgba(st) is None
    st = make_mock_state(3, n_robots=1)
    st.emb = None
    assert segment_rgba(st) is None
    st = make_mock_state(3, n_robots=1)
    st.segments = []
    assert segment_rgba(st) is None


# ---- per-robot belief -----------------------------------------------------------------------
def test_comms_full_makes_a_robot_view_the_team_view():
    """While `comms == "full"` there is one shared belief, so `robot=r` draws the team map."""
    from rlplanner.viz.frame import belief_view
    st = make_mock_state(3, n_robots=3)          # no per-robot views published
    team = belief_view(st, None)
    for r in range(3):
        v = belief_view(st, r)
        assert v.robot == r and "comms full" in v.source
        assert np.array_equal(v.known, team.known)
        assert list(v.segments) == list(team.segments)
        assert list(v.rays) == list(team.rays)
    a = render_frame(st, query_idx=0, focus_robot=0)
    b = render_frame(st, query_idx=0, focus_robot=0, robot=0)
    assert np.array_equal(_belief_panel(a), _belief_panel(b))


def _belief_panel(img):
    """The middle third of a three-panel frame (the belief map), minus its title strip."""
    w = img.shape[1] // 3
    return img[60:, w: 2 * w]


def test_a_per_robot_view_draws_less_than_the_team_union():
    from rlplanner.viz.frame import belief_view
    st = make_mock_state(3, n_robots=3, per_robot=True)
    team = belief_view(st, None)
    v = belief_view(st, 1)
    assert v.source == "robot_views"
    assert v.known.sum() < team.known.sum()
    assert (team.known | v.known).sum() == int(team.known.sum())   # a subset of the union
    assert len(v.segments) <= len(team.segments)
    assert v.ray_store.n <= team.ray_store.n
    img_team = render_frame(st, query_idx=0, focus_robot=1)
    img_r1 = render_frame(st, query_idx=0, focus_robot=1, robot=1)
    assert not np.array_equal(img_team, img_r1)


def test_the_per_robot_panel_says_whose_map_it_is():
    from matplotlib.figure import Figure
    from rlplanner.viz.frame import draw_belief_panel
    st = make_mock_state(3, n_robots=3, per_robot=True)
    fig = Figure(figsize=(4, 4), dpi=50)
    ax = fig.add_subplot(111)
    draw_belief_panel(ax, st, robot=2, legend=False)
    assert "robot 2's map" in ax.get_title()
    ax2 = Figure(figsize=(4, 4), dpi=50).add_subplot(111)
    draw_belief_panel(ax2, st, legend=False)
    assert "team" in ax2.get_title()


@pytest.mark.parametrize("bad", [-1, 3, 99, 1.5])
def test_robot_out_of_range_raises(bad):
    st = make_mock_state(1, n_robots=3)
    with pytest.raises(IndexError):
        render_frame(st, robot=bad)


def test_visited_records_of_a_robot_are_drawn():
    from matplotlib.figure import Figure
    from rlplanner.viz.frame import belief_view, draw_visited
    st = make_mock_state(3, n_robots=3, per_robot=True, n_visited=6)
    v = belief_view(st, 0)
    assert v.visited                                   # gossiped records reached this robot
    ax = Figure(figsize=(3, 3), dpi=50).add_subplot(111)
    assert draw_visited(ax, st, v) == len(v.visited)
    assert draw_visited(ax, st, belief_view(st, None)) == 0   # the union carries none


# ---- peers and comms links ------------------------------------------------------------------
def test_peer_tokens_are_zero_under_comms_full_and_arrows_appear_once_they_are_not():
    from rlplanner.sim import state as S
    from rlplanner.viz.frame import comms_links, peer_arrows
    st = make_mock_state(3, n_robots=3)
    assert not np.asarray(st.last_obs.peer_tokens).any()
    assert peer_arrows(st, 0) == [] and comms_links(st) == []
    pt = np.asarray(st.last_obs.peer_tokens).copy()
    x0, y0, x1, y1 = st.scene.region
    diag = float(np.hypot(x1 - x0, y1 - y0))
    names = S.PEER_FEAT_NAMES
    pt[0, 0, names.index("dx")] = 30.0 / diag
    pt[0, 0, names.index("dy")] = -20.0 / diag
    pt[0, 0, names.index("target_dx")] = 60.0 / diag
    pt[0, 0, names.index("target_dy")] = 10.0 / diag
    pt[0, 0, names.index("contact_age")] = 0.5
    pt[0, 0, names.index("valid")] = 1.0
    if "link" in names:
        pt[0, 0, names.index("link")] = 1.0
    st.last_obs.peer_tokens = pt
    arrows = peer_arrows(st, 0)
    assert len(arrows) == 1
    p, t, alpha, idx = arrows[0]
    assert np.allclose(p, st.robots[0].pos + np.array([30.0, -20.0]), atol=1e-3)
    assert np.allclose(t, st.robots[0].pos + np.array([60.0, 10.0]), atol=1e-3)
    assert 0.0 < alpha < 1.0 and idx == 1
    assert comms_links(st) == []                 # comms full: every pair is linked, always
    if "link" in names:
        st.cfg.comms.mode = "range"
        assert comms_links(st) == [(0, 1)]
    _rgb_ok(render_frame(st, focus_robot=0, robot=0))


def test_an_explicit_link_matrix_is_used_when_the_state_publishes_one():
    from rlplanner.viz.frame import comms_links
    st = make_mock_state(1, n_robots=3)
    st.cfg.comms.mode = "range"
    st.comms_links = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)
    assert comms_links(st) == [(0, 1), (1, 2)]
    _rgb_ok(render_frame(st, focus_robot=0))


# ---- the actor's local crop -----------------------------------------------------------------
def test_local_crop_panel_reads_the_observation_array():
    from rlplanner.viz.frame import local_crop
    st = make_mock_state(3, n_robots=2)
    assert st.last_obs.local is not None
    known, rgb, size_m = local_crop(st, 0)
    s = int(st.cfg.tokens.local_size)
    assert known.shape == (s, s) and rgb.shape == (s, s, 3)
    assert size_m == pytest.approx(s * st.raster.cell_m)
    assert known.any() and rgb[known].std() > 0.0
    assert np.allclose(rgb[~known], 0.0)


def test_local_crop_falls_back_to_the_belief_when_the_crop_is_disabled():
    from rlplanner.viz.frame import local_crop
    cfg = EnvConfig()
    cfg.tokens.local_size = 0
    st = make_mock_state(3, n_robots=2, cfg=cfg)
    assert st.last_obs.local is None
    known, rgb, size_m = local_crop(st, 1)
    assert known.shape == rgb.shape[:2] and known.any()


def test_show_local_adds_a_fourth_panel():
    from rlplanner.viz.frame import frame_figure
    st = make_mock_state(3, n_robots=2)
    assert len(frame_figure(st, focus_robot=0).axes) == 3
    fig = frame_figure(st, focus_robot=1, show_local=True)
    assert len(fig.axes) == 4
    assert "local crop" in fig.axes[2].get_title() and "robot 1" in fig.axes[2].get_title()
    _rgb_ok(render_frame(st, focus_robot=1, show_local=True))


# ---- token markers --------------------------------------------------------------------------
def test_every_token_type_has_its_own_marker_and_colour():
    from rlplanner.sim import state as S
    from rlplanner.viz import palette as P
    from rlplanner.viz.frame import token_legend_handles
    want = {"hold": ".", "frontier": "o", "ray": "*", "segment": "D", "visited": "s"}
    for n in S.TOKEN_TYPE_NAMES:
        assert P.token_marker(n) == want.get(n, P.token_marker(n))
        assert P.token_color(n)
    marks = [P.token_marker(n) for n in S.TOKEN_TYPE_NAMES]
    assert len(set(marks)) == len(marks)
    assert len(token_legend_handles()) == len(S.TOKEN_TYPE_NAMES)


def test_token_markers_survive_a_token_type_the_palette_does_not_know():
    from rlplanner.viz import palette as P
    assert P.token_marker(0) == P.token_marker("hold")
    with pytest.raises(IndexError):
        P.token_color(99)


@needs_sim
def test_belief_grid_derives_a_query_outside_the_belief_query_list():
    from rlplanner.sim.env import DisasterEnv
    from rlplanner.viz.frame import belief_grid
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 20.0
    cfg.rayfronts.queries = ("person", "road")
    env = DisasterEnv(schema.make_synthetic_scene(0, region_m=(120.0, 120.0)), cfg, seed=0)
    env.step(np.zeros(1, np.int64))
    st = env.state
    g, label, i = belief_grid(st, "rubble")            # not in cfg.rayfronts.queries
    assert i == -1 and label == "rubble"
    assert g.shape == st.observed.shape
    assert (g[~st.observed] == 0).all() and g[st.observed].max() > 0.0
    feat = st.vox_feat[st.observed]
    qe = st.emb.embed_queries(("rubble",))[0]
    assert np.allclose(g[st.observed], np.clip(feat @ qe, 0, 1), atol=1e-5)
    _rgb_ok(render_frame(st, query="rubble", focus_robot=0))
    with pytest.raises(ValueError, match="a fire hydrant"):
        belief_grid(st, "a fire hydrant")


def test_class_legend_is_drawn_on_the_truth_panel():
    from matplotlib.figure import Figure
    from rlplanner.viz.frame import draw_class_legend, draw_truth_panel
    from rlplanner.scene.schema import CLASS_NAMES
    st = make_mock_state(1, n_robots=1)
    fig = Figure(figsize=(6, 6), dpi=60)
    ax = fig.add_subplot(111)
    draw_truth_panel(ax, st)
    legs = [a for a in ax.get_children() if a.__class__.__name__ == "Legend"]
    labels = {t.get_text() for l in legs for t in l.get_texts()}
    assert set(CLASS_NAMES) <= labels
    ax2 = fig.add_subplot(212)
    draw_class_legend(ax2, classes=("road", "tree"))
    assert len([a for a in ax2.get_children() if a.__class__.__name__ == "Legend"]) == 1


@needs_sim
def test_set_queries_moves_only_the_query_view():
    """There is no privileged query column any more: the panel follows the *name* asked for, and
    swapping the mission list moves nothing in the belief."""
    from rlplanner.sim.env import DisasterEnv
    from rlplanner.viz.frame import query_view
    cfg = EnvConfig()
    cfg.robot.n_robots = 2
    cfg.t_max_s = 120.0
    env = DisasterEnv(schema.make_synthetic_scene(0, region_m=(160.0, 160.0)), cfg, seed=0)
    for _ in range(4):
        env.step(np.zeros(2, np.int64))
    st = env.state
    before = query_view(st, "person lying on the ground")
    env.set_queries(("rubble", "person lying on the ground", "car"))
    st = env.state
    assert st.query_names() == ("rubble", "person lying on the ground", "car")
    after = query_view(st, "person lying on the ground")
    assert after.idx == 1 and np.allclose(before.grid, after.grid)
    assert np.allclose(before.ray_sim, after.ray_sim)
    _rgb_ok(render_frame(st, query_idx="person lying on the ground", focus_robot=0))
    _rgb_ok(render_frame(st, query_idx=2, focus_robot=0))


@needs_sim
def test_a_decision_takes_no_query_view_but_a_frame_takes_one():
    """`RayFrontsSim.n_query_calls` must not move across `step`; a frame moves it by one grid
    plus one per-ray view."""
    from rlplanner.sim.env import DisasterEnv
    cfg = EnvConfig()
    cfg.robot.n_robots = 1
    cfg.t_max_s = 60.0
    env = DisasterEnv(schema.make_synthetic_scene(0, region_m=(120.0, 120.0)), cfg, seed=0)
    n0 = env.rf.n_query_calls
    env.step(np.zeros(1, np.int64))
    assert env.rf.n_query_calls == n0
    render_frame(env.state, query_idx=0, focus_robot=0)
    assert env.rf.n_query_calls == n0 + 2       # one grid + one per-ray view, once each


@needs_sim
def test_render_frame_on_the_real_env_per_robot_and_with_the_local_crop():
    from rlplanner.sim.env import DisasterEnv
    cfg = EnvConfig()
    cfg.robot.n_robots = 2
    cfg.t_max_s = 60.0
    env = DisasterEnv(schema.make_synthetic_scene(0, region_m=(160.0, 160.0)), cfg, seed=0)
    for _ in range(3):
        env.step(np.zeros(2, np.int64))
    st = env.state
    _rgb_ok(render_frame(st, focus_robot=0, robot=1))
    _rgb_ok(render_frame(st, focus_robot=0, show_local=True))


@needs_sim
def test_comms_full_on_the_real_env_gives_every_robot_the_team_map():
    from rlplanner.sim.env import DisasterEnv
    from rlplanner.viz.frame import belief_view, comms_links, peer_arrows, robot_views
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    cfg.t_max_s = 60.0
    env = DisasterEnv(schema.make_synthetic_scene(0, region_m=(200.0, 200.0)), cfg, seed=0)
    for _ in range(4):
        env.step(np.zeros(3, np.int64))
    st = env.state
    assert robot_views(st) is None                 # one shared belief: nothing per robot to publish
    team = belief_view(st, None)
    for r in range(3):
        v = belief_view(st, r)
        assert np.array_equal(v.known, team.known)
        assert list(v.segments) == list(team.segments) and list(v.rays) == list(team.rays)
    # peer tokens are still filled (everyone is always in contact), so the arrows are drawn
    arrows = peer_arrows(st, 0)
    assert len(arrows) == 2
    for p, t, alpha, idx in arrows:
        assert np.allclose(p, st.robots[idx].pos, atol=1.0) and alpha == pytest.approx(1.0)
    # the map itself is the same picture; only the peer overlay and the title differ
    from rlplanner.viz.frame import query_view
    qa, qb = query_view(st, 0, team), query_view(st, 0, belief_view(st, 1))
    assert np.allclose(qa.grid, qb.grid) and np.allclose(qa.ray_sim, qb.ray_sim)
    _rgb_ok(render_frame(st, query_idx=0, focus_robot=1, robot=1))


@needs_sim
def test_range_comms_gives_each_robot_less_than_the_union():
    from rlplanner.sim.env import DisasterEnv
    from rlplanner.viz.frame import belief_view, comms_links, peer_arrows, robot_views
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    cfg.t_max_s = 120.0
    cfg.comms = {"mode": "range", "range_m": 60.0, "randomize_range": False}
    env = DisasterEnv(schema.make_synthetic_scene(0, region_m=(300.0, 300.0)), cfg, seed=0)
    for _ in range(8):
        env.step(np.zeros(3, np.int64))
    st = env.state
    vs = robot_views(st)
    assert vs is not None and len(vs) == 3
    team = belief_view(st, None)
    v = belief_view(st, 0)
    assert v.source == "robot_views"
    assert v.known.sum() <= team.known.sum()
    assert not (v.known & ~team.known).any()       # a robot never knows what the union does not
    arrows = peer_arrows(st, 0)
    assert len(arrows) == 2                        # one per peer it has heard from
    for p, t, alpha, idx in arrows:
        assert np.all(np.isfinite(p)) and 0.0 < alpha <= 1.0 and idx in (1, 2)
    assert comms_links(st)                         # everyone is in range at spawn
    _rgb_ok(render_frame(st, query_idx=0, focus_robot=0, robot=0, show_local=True))
