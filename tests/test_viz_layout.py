"""Window-vs-file policy and legend placement (legends must never cover a map)."""
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
from rlplanner.viz import display  # noqa: E402
from rlplanner.viz.layout import legend_is_outside, legend_outside  # noqa: E402
from rlplanner.viz.raster_plot import plot_raster  # noqa: E402
from rlplanner.viz.scene_plot import plot_scene  # noqa: E402


def _have(mod: str) -> bool:
    """Importable, not merely present: the sim is renamed in place from time to time."""
    try:
        importlib.import_module(mod)
        return True
    except Exception:                       # noqa: BLE001 - mid-edit sim: skip, do not error
        return False


from viz_mocks import sim_available  # noqa: E402

HAVE_SIM = _have("rlplanner.sim.env") and _have("rlplanner.sim.baselines") and sim_available()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- backend selection ---------------------------------------------------------------------
def test_explicit_agg_never_opens_a_window(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("DISPLAY", ":1")
    assert display.gui_possible() is False
    assert display.select_backend(True).lower() == "agg"
    assert display.gui_active() is False


def test_no_display_means_no_window(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert display.has_display() is False
    assert display.gui_possible() is False
    assert display.select_backend(True).lower() == "agg"


def test_a_display_without_an_explicit_backend_would_use_a_window(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(sys, "platform", "linux")
    assert display.gui_possible() is True          # select_backend would try QtAgg/TkAgg


def test_finish_writes_the_default_when_there_is_no_window(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    fig = plt.figure(figsize=(2, 2))
    default = tmp_path / "default.png"
    assert display.finish(fig, None, default, "test") == default
    assert default.exists()
    out = tmp_path / "explicit.png"
    assert display.finish(fig, out, default, "test") == out
    assert out.exists()


# ---- every script: no --out and no window => a file, and it says where -----------------------
def test_view_scene_without_out_writes_the_default(tmp_path, monkeypatch, capsys):
    import view_scene
    default = tmp_path / "scene.png"
    monkeypatch.setattr(view_scene, "DEFAULT_OUT", str(default))
    assert view_scene.main(["--synthetic", "0", "--dpi", "50"]) == 0
    assert default.exists()
    assert str(default) in capsys.readouterr().out


def test_episode_viewer_without_save_writes_the_default(tmp_path, monkeypatch):
    import episode_viewer
    import imageio.v2 as iio
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(2):
        iio.imwrite(d / f"f_{i}.png", np.full((8, 12, 3), 10 * i + 5, np.uint8))
    default = tmp_path / "viewer.png"
    monkeypatch.setattr(episode_viewer, "DEFAULT_OUT", str(default))
    assert episode_viewer.main([str(d)]) == 0
    assert default.exists()


@pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines not available yet")
def test_play_episode_without_out_writes_the_default_mp4(tmp_path, monkeypatch):
    import play_episode
    default = tmp_path / "ep.mp4"
    monkeypatch.setattr(play_episode, "DEFAULT_OUT", str(default))
    assert play_episode.main(["--synthetic", "0", "--policy", "random", "--robots", "2",
                              "--max-decisions", "1", "--dpi", "50"]) == 0
    assert default.exists() and default.stat().st_size > 0


@pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines not available yet")
def test_rayfronts_demo_without_out_writes_the_default(tmp_path, monkeypatch):
    import rayfronts_demo
    default = tmp_path / "demo.png"
    monkeypatch.setattr(rayfronts_demo, "DEFAULT_OUT", str(default))
    assert rayfronts_demo.main(["--synthetic", "1", "--checkpoints", "1"]) == 0
    assert default.exists()
    assert not (tmp_path / "demo.mp4").exists()    # no --out/--mp4 => no video either


def test_sensor_inspector_without_out_writes_the_default(tmp_path, monkeypatch):
    import sensor_inspector
    default = tmp_path / "insp.png"
    monkeypatch.setattr(sensor_inspector, "DEFAULT_OUT", str(default))
    assert sensor_inspector.main(["--synthetic", "0", "--dpi", "40", "--pov", "40", "30"]) == 0
    assert default.exists()


# ---- legends outside the axes --------------------------------------------------------------
def test_plot_scene_legend_is_outside_the_map():
    ax = plot_scene(schema.make_synthetic_scene(1))
    assert ax.get_legend() is not None
    assert legend_is_outside(ax)


def test_plot_raster_legend_is_outside_the_map():
    from rlplanner.sim.raster import rasterize
    ras = rasterize(schema.make_synthetic_scene(0, region_m=(120.0, 120.0)), 2.0)
    ax = plot_raster(ras)
    assert legend_is_outside(ax)


@pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines not available yet")
def test_render_frame_panel_legends_are_outside_their_axes():
    from viz_mocks import make_mock_state
    from rlplanner.viz.frame import frame_figure
    st = make_mock_state(0, n_robots=3)
    fig = frame_figure(st, query_idx=0, focus_robot=0, figsize=(18.0, 6.0), dpi=60)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ax_gt, ax_bel = fig.axes[0], fig.axes[1]
    legends = [lg for ax in (ax_gt, ax_bel) for lg in
               ([ax.get_legend()] if ax.get_legend() else []) +
               [a for a in ax.artists if hasattr(a, "get_window_extent") and hasattr(a, "texts")]]
    assert len(legends) >= 3               # gt: humans + raster classes, belief: overlays
    for ax in (ax_gt, ax_bel):
        ab = ax.get_window_extent().transformed(inv)
        for lg in legends:
            if lg.axes is ax:
                assert not ab.overlaps(lg.get_window_extent().transformed(inv))


def _legends_of(ax):
    return ([ax.get_legend()] if ax.get_legend() else []) + \
           [a for a in ax.artists if hasattr(a, "texts") and hasattr(a, "get_window_extent")]


def _assert_legends_outside(fig, axes):
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    n = 0
    for ax in axes:
        ab = ax.get_window_extent().transformed(inv)
        for lg in _legends_of(ax):
            n += 1
            assert not ab.overlaps(lg.get_window_extent().transformed(inv)), \
                f"a legend covers {ax.get_title()!r}"
    assert n >= len(axes), "a panel lost its legend"


@pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines not available yet")
def test_recorder_output_frames_keep_their_legends_outside():
    """The frames play_episode writes come from this figure at the recorder's own size."""
    import inspect

    from viz_mocks import make_mock_state
    from rlplanner.viz.frame import frame_figure
    from rlplanner.viz.recorder import EpisodeRecorder
    sig = inspect.signature(EpisodeRecorder.__init__).parameters
    figsize = sig["figsize"].default
    dpi = sig["dpi"].default
    st = make_mock_state(1, n_robots=3)
    fig = frame_figure(st, query_idx=0, focus_robot=0, figsize=figsize, dpi=dpi)
    _assert_legends_outside(fig, fig.axes[:2])


def test_sensor_inspector_pov_legend_is_outside_the_frame():
    import sensor_inspector as si
    from rlplanner.sim.config import EnvConfig
    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(0, region_m=(160.0, 160.0))
    insp = si.Inspector(scene, cfg, si.default_pose(scene, cfg), pov=(48, 36), zoom=160.0, dpi=45)
    insp.sense(1)
    fig = insp.build()
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ab = insp.ax_pov.get_window_extent().transformed(inv)
    legs = [a for a in insp.ax_pov.artists if hasattr(a, "texts")] + \
           ([insp.ax_pov.get_legend()] if insp.ax_pov.get_legend() else [])
    assert legs, "the POV panel lost its class legend"
    for lg in legs:
        assert not ab.overlaps(lg.get_window_extent().transformed(inv))


def test_legend_outside_rejects_a_bad_side():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="legend side"):
        legend_outside(ax, [plt.Line2D([], [], label="x")], side="middle")
    assert legend_outside(ax, [], side="right") is None


# ---- vocabulary guard -------------------------------------------------------------------------
def test_the_visualizer_uses_the_simulator_vocabulary():
    """The sim publishes cells, rays, frontiers, segments and visited records; the retired words
    (including the hand-rule ones the open-set change removed) must not come back."""
    import re
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.glob("src/rlplanner/viz/*.py")) + [
        SCRIPTS / f for f in ("view_scene.py", "play_episode.py", "rayfronts_demo.py",
                              "episode_viewer.py", "sensor_inspector.py", "live_viewer.py")
    ] + sorted(root.glob("tests/test_viz_*.py")) + [root / "tests" / "viz_mocks.py"]
    retired = ("l" + "ead", "detec" + "tion", "con" + "firm", "RA" + "VEN", "black" + "list",
               "dis" + "miss")
    bad = []
    for f in files:
        if f.name == Path(__file__).name:
            continue                                   # this file spells them to test for them
        text = f.read_text()
        for w in retired:
            for m in re.finditer(w, text, re.IGNORECASE):
                line = text[:m.start()].count("\n") + 1
                bad.append(f"{f.relative_to(root)}:{line}: {w}")
    assert not bad, "retired vocabulary: " + "; ".join(bad)


def test_the_visualizer_does_not_use_the_retired_open_set_accessors():
    """`sims`/`sims_max`, `PERSON_QUERY_IDX`, `voxel_candidates` and `k_voxel` are gone; a viewer
    that still names them would break the moment the sim is imported."""
    import re
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.glob("src/rlplanner/viz/*.py")) + [
        SCRIPTS / f for f in ("view_scene.py", "play_episode.py", "rayfronts_demo.py",
                              "episode_viewer.py", "sensor_inspector.py", "live_viewer.py")
    ]
    gone = ("PERSON_QUERY_IDX", "person_query_indices", "PERSON_QUERY_NAMES", "sims_max",
            "voxel_candidates", "VoxelCandidate", "k_voxel", "TOKEN_VOXEL", "F_SIM0")
    bad = []
    for f in files:
        text = f.read_text()
        for w in gone:
            for m in re.finditer(re.escape(w), text):
                bad.append(f"{f.relative_to(root)}:{text[:m.start()].count(chr(10)) + 1}: {w}")
    assert not bad, "retired accessors: " + "; ".join(bad)


def test_no_viz_module_reads_the_lazy_vox_sim_stack():
    """`EnvState.vox_sim` allocates [Q, ny, nx] on every access, so no panel may read it: the
    frame asks `query_sim` for the single grid it draws (the count is asserted in test_viz_frame)."""
    import ast
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted(root.glob("src/rlplanner/viz/*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == "vox_sim":
                bad.append(f"{f.relative_to(root)}:{node.lineno}")
    assert not bad, "vox_sim read at " + "; ".join(bad)
