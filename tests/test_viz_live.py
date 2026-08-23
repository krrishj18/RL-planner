"""Live episode window: the update callback steps the env and mutates the artists in place."""
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rlplanner.scene import schema  # noqa: E402
from rlplanner.viz import live as L  # noqa: E402


def _have(mod: str) -> bool:
    """Importable, not merely present: the sim is renamed in place from time to time."""
    try:
        importlib.import_module(mod)
        return True
    except Exception:                       # noqa: BLE001 - mid-edit sim: skip, do not error
        return False


from viz_mocks import sim_available  # noqa: E402

HAVE_SIM = _have("rlplanner.sim.env") and _have("rlplanner.sim.baselines") and sim_available()
needs_sim = pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines mid-edit")
V2 = sorted((Path(__file__).resolve().parents[1] / "data" / "scenes_v2")
            .glob("downtown_tornado_17.json"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def make_viewer(n_robots=2, seed=0, region=(160.0, 160.0), scene=None, dpi=50, sim=None, **kw):
    """A viewer on the real env when the sim imports, else on the mock env (same API)."""
    from rlplanner.sim.config import EnvConfig

    cfg = EnvConfig()
    cfg.robot.n_robots = int(n_robots)
    scene = scene if scene is not None else schema.make_synthetic_scene(seed, region_m=region)
    if HAVE_SIM if sim is None else sim:
        from rlplanner.sim import baselines
        from rlplanner.sim.env import DisasterEnv
        env = DisasterEnv(scene, cfg, seed=seed)
        policy = baselines.make_policy("nearest_frontier", cfg.rayfronts.queries, seed)
    else:
        from viz_mocks import FirstValidPolicy, MockEnv
        env = MockEnv(seed=seed, n_robots=int(n_robots), n_steps=10 ** 6, cfg=cfg, scene=scene,
                      per_robot=bool(kw.pop("per_robot", False)))
        policy = FirstValidPolicy()
    kw.pop("per_robot", None)
    return L.LiveViewer(env, policy, dpi=dpi, figsize=(9.6, 4.8), seed=seed, **kw)


# ---- pure helpers ---------------------------------------------------------------------------
def test_block_max_reduces_and_keeps_hot_cells():
    g = np.zeros((8, 8))
    g[3, 5] = 1.0
    assert L.block_max(g, 1).shape == (8, 8)
    r = L.block_max(g, 2)
    assert r.shape == (4, 4) and r[1, 2] == 1.0 and r.sum() == 1.0
    b = np.zeros((7, 5), bool)
    b[0, 0] = True
    assert L.block_max(b, 2, "any").shape == (3, 2)
    assert L.block_max(g, 99).shape == (8, 8)      # a factor larger than the grid is a no-op


def test_geometry_helpers_on_an_empty_and_a_full_state():
    from viz_mocks import empty_ray_store, make_mock_state
    from rlplanner.viz.frame import ray_geometry
    st = make_mock_state(0, n_robots=2)
    segs, cols, n = ray_geometry(st, 0)
    assert segs.shape[1:] == (2, 2) and segs.shape[0] == n == len(cols)
    xy, sizes = L.frontier_points(st)
    assert xy.shape[0] == sizes.shape[0]
    cells = L.frontier_cells(st)
    assert cells.ndim == 2 and cells.shape[1] == 2
    hp = L.human_points(st)
    assert sum(v.shape[0] for v in hp.values()) == len(st.scene.humans)
    cas = np.array([h.role == "casualty" for h in st.scene.humans], bool)
    assert hp["found"].shape[0] == int((st.human_found & cas).sum())
    assert hp["unfound"].shape[0] == int((~st.human_found & cas).sum())

    st.rays = empty_ray_store(st.emb.D)
    st.frontier_clusters = []
    st.frontier_mask[:] = False
    assert ray_geometry(st, 0)[2] == 0
    assert L.frontier_points(st)[0].shape == (0, 2)
    assert L.frontier_cells(st).shape == (0, 2)


def test_frontier_cells_are_capped():
    from viz_mocks import make_mock_state
    st = make_mock_state(0, n_robots=2)
    st.frontier_mask[:] = True
    assert L.frontier_cells(st, max_points=50).shape[0] <= 50


def test_gt_background_modes():
    scene = schema.make_synthetic_scene(0, region_m=(120.0, 120.0))
    from rlplanner.sim.raster import rasterize
    ras = rasterize(scene, 2.0)
    img, origin = L.gt_background(scene, ras, "raster")
    assert img.shape == (ras.ny, ras.nx, 3) and origin == "lower"
    img, origin = L.gt_background(scene, ras, "scene")
    assert img.ndim == 3 and origin == "upper" and img.max() <= 1.0
    assert L.gt_background(scene, ras, "auto")[1] == "upper"   # a small scene is baked
    with pytest.raises(ValueError, match="background mode"):
        L.gt_background(scene, ras, "nope")


# ---- the update callback --------------------------------------------------------------------
@needs_sim
def test_update_steps_the_env_and_mutates_the_artists():
    lv = make_viewer(n_robots=2)
    fig = lv.build()
    assert lv.env.state.decision_idx == 0
    traj0 = lv.traj[0].get_data()[0].size
    rays0 = len(lv.ray_lc.get_segments())
    img0 = lv.im_bel.get_array().copy()
    pos0 = np.asarray(lv.robot_dots.get_offsets()).copy()
    for _ in range(6):
        lv.update()
    assert lv.env.state.decision_idx == 6 and lv.n_decisions == 6
    assert lv.traj[0].get_data()[0].size > traj0            # trajectory grew
    assert len(lv.ray_lc.get_segments()) != rays0           # rays appeared/resolved
    assert not np.array_equal(lv.im_bel.get_array(), img0)  # the belief image changed
    assert not np.allclose(np.asarray(lv.robot_dots.get_offsets()), pos0)
    assert np.asarray(lv.bel_robots.get_offsets()).shape == (2, 2)
    assert lv.build() is fig                                # artists are built once


def test_update_on_the_mock_env_mutates_the_artists():
    """Same contract as above, without the sim: the callback advances and the artists change."""
    lv = make_viewer(n_robots=2, sim=False)
    lv.build()
    img0 = lv.im_bel.get_array().copy()
    pos0 = np.asarray(lv.robot_dots.get_offsets()).copy()
    for _ in range(3):
        lv.update()
    assert lv.n_decisions == 3
    assert not np.array_equal(lv.im_bel.get_array(), img0)
    assert not np.allclose(np.asarray(lv.robot_dots.get_offsets()), pos0)
    assert np.asarray(lv.front_cells.get_offsets()).ndim == 2


def test_finds_are_flashed_and_logged():
    lv = make_viewer(n_robots=2, sim=False)
    lv.build()
    st = lv.env.state
    k = next(i for i, h in enumerate(st.scene.humans) if h.role == "casualty")
    lv._prev_found = np.zeros(len(st.scene.humans), bool)
    st.human_found[k] = True
    ev = [type("E", (), {"kind": "found", "payload": {"human_idx": k, "robot": 1}})()]
    lv._log_finds({"events_this_step": ev})
    assert lv.log and "robot 1 found casualty" in lv.log[-1]
    assert st.scene.humans[k].container in lv.log[-1]
    assert len(lv.flashes) == 1
    lv.refresh()
    assert np.asarray(lv.flash_gt.get_offsets()).shape == (1, 2)
    assert np.asarray(lv.flash_bel.get_offsets()).shape == (1, 2)
    for _ in range(L.FLASH_FRAMES + 1):
        lv.refresh()
    assert np.asarray(lv.flash_gt.get_offsets()).shape[0] == 0   # the flash fades out
    assert "finds" in "\n".join(lv.status_lines(lv.env.state, {"rays": 0, "frontiers": 0,
                                                               "tokens": 0}))


def test_a_find_without_an_event_still_logs_the_team():
    lv = make_viewer(n_robots=1, sim=False)
    lv.build()
    st = lv.env.state
    k = next(i for i, h in enumerate(st.scene.humans) if h.role == "casualty")
    lv._prev_found = np.zeros(len(st.scene.humans), bool)
    st.human_found[k] = True
    lv._log_finds(None)
    assert lv.log and lv.log[-1].startswith("t=") and "team found casualty" in lv.log[-1]


def test_live_viewer_legends_sit_outside_every_map():
    lv = make_viewer(n_robots=2, sim=False)
    fig = lv.build()
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax in (lv.ax_gt, lv.ax_bel, lv.ax_spark):
        lg = ax.get_legend()
        assert lg is not None, f"{ax} lost its legend"
        ab = ax.get_window_extent().transformed(inv)
        lb = lg.get_window_extent().transformed(inv)
        assert not ab.overlaps(lb)
        assert lb.x0 >= 0.0 and lb.y0 >= 0.0 and lb.x1 <= 1.0 and lb.y1 <= 1.0


def test_paused_update_does_not_step_but_a_single_step_does():
    lv = make_viewer(n_robots=1, autoplay=False, sim=False)
    lv.build()
    lv.update()
    assert lv.n_decisions == 0
    lv._on_step()
    lv.update()
    assert lv.n_decisions == 1
    lv.update()
    assert lv.n_decisions == 1


def test_max_decisions_ends_the_episode():
    lv = make_viewer(n_robots=1, max_decisions=3, sim=False)
    lv.build()
    for _ in range(6):
        lv.update()
    assert lv.n_decisions == 3 and lv.done
    assert lv.advance() is False


def test_token_artists_have_one_marker_per_type_and_follow_the_focus():
    from rlplanner.sim import state as S
    from rlplanner.viz import palette as P
    lv = make_viewer(n_robots=3, sim=False)
    lv.build()
    for _ in range(3):
        lv.update()
    assert set(lv.tok_sc) == set(S.TOKEN_TYPE_NAMES)
    markers = [P.token_marker(n) for n in S.TOKEN_TYPE_NAMES]
    assert len(set(markers)) == len(markers)          # one marker per token type, no reuse
    drawn = sum(np.asarray(sc.get_offsets()).shape[0] for sc in lv.tok_sc.values())
    assert drawn >= 1
    assert np.asarray(lv.tok_chosen.get_offsets()).shape[0] == 1   # exactly one chosen token
    before = {n: np.asarray(sc.get_offsets()).copy() for n, sc in lv.tok_sc.items()}
    lv.focus = 1
    lv.refresh()
    after = {n: np.asarray(sc.get_offsets()) for n, sc in lv.tok_sc.items()}
    assert any(not np.array_equal(before[n], after[n]) for n in before) or drawn == 0


# ---- per-robot map, segments, local crop ----------------------------------------------------
def test_the_belief_panel_follows_the_robot_selection():
    lv = make_viewer(n_robots=3, sim=False, per_robot=True)
    lv.build()
    lv.update()
    assert "belief (team)" in lv.ax_bel.get_title()
    img_team = lv.im_bel.get_array().copy()
    lv.robot = 1
    lv.refresh()
    assert "robot 1's map" in lv.ax_bel.get_title()
    assert not np.array_equal(lv.im_bel.get_array(), img_team)
    known_team = lv.env.state.observed.sum()
    from rlplanner.viz.frame import belief_view
    assert belief_view(lv.env.state, 1).known.sum() < known_team


def test_the_segment_overlay_is_drawn_and_can_be_turned_off():
    lv = make_viewer(n_robots=2, sim=False)
    lv.build()
    lv.update()
    rgba = np.asarray(lv.im_seg.get_array())
    assert rgba.shape[:2] == np.asarray(lv.im_bel.get_array()).shape[:2]
    assert rgba.shape[2] == 4 and rgba[..., 3].max() > 0.0
    assert np.asarray(lv.seg_sc.get_offsets()).shape[0] > 0
    lv.segments = False
    lv.refresh()
    assert np.asarray(lv.im_seg.get_array())[..., 3].max() == 0.0


def test_the_local_crop_panel_is_optional():
    lv = make_viewer(n_robots=2, sim=False)
    assert lv.build() is not None and lv.ax_local is None
    lv2 = make_viewer(n_robots=2, sim=False, show_local=True)
    lv2.build()
    lv2.update()
    assert lv2.ax_local is not None and lv2.ax_local.images
    assert "local crop" in lv2.ax_local.get_title()


def test_peer_arrows_and_links_appear_when_the_observation_carries_them():
    from rlplanner.sim import state as S
    lv = make_viewer(n_robots=3, sim=False, robot=0)
    lv.build()
    lv.update()
    assert np.asarray(lv.peer_lc.get_segments()).size == 0      # comms full: nothing to draw
    st = lv.env.state
    pt = np.asarray(st.last_obs.peer_tokens).copy()
    names = S.PEER_FEAT_NAMES
    x0, y0, x1, y1 = st.scene.region
    diag = float(np.hypot(x1 - x0, y1 - y0))
    pt[0, 0, names.index("dx")] = 25.0 / diag
    pt[0, 0, names.index("target_dy")] = 40.0 / diag
    pt[0, 0, names.index("valid")] = 1.0
    if "link" in names:
        pt[0, 0, names.index("link")] = 1.0
    st.last_obs.peer_tokens = pt
    lv.refresh()
    assert len(lv.peer_lc.get_segments()) == 1
    assert np.asarray(lv.peer_sc.get_offsets()).shape == (1, 2)
    assert not lv.link_lc.get_segments()               # comms full: every pair is always linked
    if "link" in names:
        st.cfg.comms.mode = "range"                    # ... now the links carry information
        lv.refresh()
        assert len(lv.link_lc.get_segments()) == 1


def test_the_status_panel_lists_the_live_queries():
    lv = make_viewer(n_robots=2, sim=False)
    lv.build()
    txt = "\n".join(lv.status_lines(lv.env.state, {"rays": 0, "frontiers": 0, "tokens": 0}))
    for q in lv.env.state.query_names():
        assert q in txt
    assert "team union" in txt
    lv.robot = 1
    assert "robot 1" in "\n".join(
        lv.status_lines(lv.env.state, {"rays": 0, "frontiers": 0, "tokens": 0}))


# ---- keys and buttons -----------------------------------------------------------------------
def _key(lv, k):
    from matplotlib.backend_bases import KeyEvent
    canvas = lv.fig.canvas
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, k))


def test_key_handlers():
    lv = make_viewer(n_robots=3, sim=False)
    lv.build()
    for _ in range(2):
        lv.update()

    assert lv.playing
    _key(lv, " ")
    assert not lv.playing
    n0 = lv.n_decisions
    _key(lv, "n")
    lv.update()
    assert lv.n_decisions == n0 + 1

    s0 = lv.speed
    _key(lv, "+")
    assert lv.speed > s0
    _key(lv, "-")
    assert lv.speed == pytest.approx(s0)

    n_q = len(lv.env.state.query_names())
    _key(lv, "3")
    assert lv.query == min(3, n_q - 1)              # keys 0-9 index the live mission list
    _key(lv, "9")
    assert lv.query == n_q - 1
    _key(lv, "0")
    assert lv.query == 0

    f0 = lv.focus
    _key(lv, "f")
    assert lv.focus == (f0 + 1) % 3

    assert lv.robot is None                          # v cycles: team union -> 0 .. n-1 -> union
    for r in range(3):
        _key(lv, "v")
        assert lv.robot == r
    _key(lv, "v")
    assert lv.robot is None

    assert lv.segments
    _key(lv, "s")
    assert not lv.segments
    _key(lv, "s")
    assert lv.segments

    _key(lv, "z")                                   # unhandled: no crash
    assert lv.n_decisions == n0 + 1


def test_speed_keys_retime_a_running_animation():
    from matplotlib.animation import FuncAnimation
    lv = make_viewer(n_robots=1, sim=False)
    fig = lv.build()
    lv.anim = FuncAnimation(fig, lv.update, interval=1000.0 / lv.speed, blit=False,
                            cache_frame_data=False)
    _key(lv, "+")
    assert lv.speed == pytest.approx(6.0)
    assert lv.anim._interval == pytest.approx(1000.0 / 6.0)   # survives TimedAnimation._step
    assert lv.anim.event_source.interval == int(1000.0 / 6.0)
    for _ in range(9):
        _key(lv, "-")
    assert lv.speed == pytest.approx(L.SPEED_MIN)             # clamped, never 0
    for _ in range(20):
        _key(lv, "+")
    assert lv.speed == pytest.approx(L.SPEED_MAX)
    lv.anim = None


def test_restart_keys_reset_the_episode():
    lv = make_viewer(n_robots=2, sim=False)
    lv.build()
    for _ in range(4):
        lv.update()
    assert lv.env.state.decision_idx == 4
    seed = lv.seed
    _key(lv, "r")
    assert lv.env.state.decision_idx == 0 and lv.n_decisions == 0 and lv.seed == seed
    assert lv.hist_t == [0.0] and not lv.done
    lv.update()
    _key(lv, "R")
    assert lv.seed == seed + 1 and lv.env.state.decision_idx == 0


def test_w_writes_a_png_and_q_closes(tmp_path):
    lv = make_viewer(n_robots=1, png=tmp_path / "shot.png", sim=False)
    fig = lv.build()
    lv.update()
    _key(lv, "w")
    assert (tmp_path / "shot.png").exists()
    _key(lv, "q")
    assert not plt.fignum_exists(fig.number)


def test_toolbar_buttons_drive_the_same_actions():
    lv = make_viewer(n_robots=1, sim=False)
    lv.build()
    assert set(lv.buttons) == {"play", "step", "restart"}
    lv.buttons["play"].on_clicked(lambda _e: None)
    lv._on_play()
    assert not lv.playing
    lv._on_step()
    lv.update()
    assert lv.n_decisions == 1
    lv._on_restart()
    assert lv.n_decisions == 0


# ---- recording ------------------------------------------------------------------------------
def _frame_count(path):
    import imageio.v2 as iio
    with iio.get_reader(path) as r:
        return sum(1 for _ in r)


@needs_sim
def test_record_headless_writes_an_mp4_of_the_episode(tmp_path):
    lv = make_viewer(n_robots=2)
    out, n = lv.record(tmp_path / "syn.mp4", fps=4, max_decisions=10)
    assert out.exists() and out.stat().st_size > 0
    assert n == 11                                   # the initial state plus one per decision
    assert _frame_count(out) == 11
    assert lv.record_rate > 0.0
    assert lv.env.state.decision_idx == 10


@needs_sim
@pytest.mark.skipif(not V2, reason="data/scenes_v2/downtown_tornado_17.json not exported")
def test_record_headless_on_a_v2_scene(tmp_path):
    scene = schema.Scene.from_json(V2[0])
    lv = make_viewer(n_robots=3, scene=scene, dpi=45)
    out, n = lv.record(tmp_path / "v2.mp4", fps=4, max_decisions=10)
    assert n == 11 and _frame_count(out) == 11
    assert lv.stride >= 1                            # the heatmap is downsampled when it is huge
    assert lv.im_bel.get_array().shape[0] <= L.HEATMAP_MAX_PX


def test_record_refuses_a_path_it_cannot_write(tmp_path):
    lv = make_viewer(n_robots=1, sim=False)
    with pytest.raises(RuntimeError, match="could not open"):
        lv.record(tmp_path / "nope.unknownext", max_decisions=1)


# ---- the script -----------------------------------------------------------------------------
@needs_sim
def test_live_viewer_script_records_headless(tmp_path):
    import live_viewer
    out = tmp_path / "live.mp4"
    assert live_viewer.main(["--synthetic", "0", "--policy", "nearest_frontier", "--robots", "2",
                             "--record", str(out), "--max-decisions", "4", "--dpi", "45",
                             "--figsize", "9.6", "4.8"]) == 0
    assert out.exists() and _frame_count(out) == 5


@needs_sim
def test_live_viewer_script_writes_a_png_without_a_window(tmp_path):
    import live_viewer
    png = tmp_path / "frame.png"
    assert live_viewer.main(["--synthetic", "0", "--policy", "random", "--robots", "2",
                             "--max-decisions", "3", "--png", str(png), "--dpi", "45",
                             "--figsize", "9.6", "4.8"]) == 0
    assert png.exists()


@needs_sim
def test_live_viewer_script_argument_validation(tmp_path):
    import live_viewer
    scene = schema.make_synthetic_scene(0)
    assert live_viewer.parse_robots("auto", scene) >= 3
    assert live_viewer.parse_robots("5", scene) == 5
    assert live_viewer.parse_query("2") == 2 and live_viewer.parse_query("rubble") == "rubble"
    with pytest.raises(SystemExit):
        live_viewer.parse_robots("0", scene)
    for bad in (["--synthetic", "0", "--query", "99"],
                ["--synthetic", "0", "--focus", "7", "--robots", "2"],
                ["--synthetic", "0", "--policy", "nope"]):
        with pytest.raises(SystemExit):
            live_viewer.main(bad)


def test_live_viewer_auto_robots_matches_the_train_rule():
    import live_viewer
    from rlplanner.train.scenes import auto_robots, region_area_km2
    scene = schema.make_synthetic_scene(0, region_m=(400.0, 400.0))
    assert live_viewer.auto_robots(scene) == auto_robots(region_area_km2((400.0, 400.0)))
