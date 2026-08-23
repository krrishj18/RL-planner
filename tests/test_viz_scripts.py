import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import episode_viewer  # noqa: E402
import play_episode  # noqa: E402
import view_scene  # noqa: E402

def _have(mod: str) -> bool:
    """Importable, not merely present: the sim is renamed in place from time to time."""
    try:
        importlib.import_module(mod)
        return True
    except Exception:                       # noqa: BLE001 - mid-edit sim: skip, do not error
        return False


from viz_mocks import sim_available  # noqa: E402

HAVE_ENV = _have("rlplanner.sim.env")
HAVE_SIM = HAVE_ENV and _have("rlplanner.sim.baselines") and sim_available()


def test_view_scene_synthetic(tmp_path):
    out = tmp_path / "scene.png"
    assert view_scene.main(["--synthetic", "2", "--out", str(out), "--dpi", "60"]) == 0
    assert out.exists() and out.stat().st_size > 0


def test_view_scene_from_json(tmp_path):
    from rlplanner.scene import schema
    p = tmp_path / "scene.json"
    schema.make_synthetic_scene(1).to_json(p)
    out = tmp_path / "s.png"
    assert view_scene.main([str(p), "--out", str(out), "--dpi", "60", "--ids"]) == 0
    assert out.exists()


def test_view_scene_raster_flag(tmp_path):
    out = tmp_path / "r.png"
    assert view_scene.main(["--synthetic", "0", "--raster", "--out", str(out), "--dpi", "60"]) == 0
    assert out.exists()


def test_view_scene_requires_a_source():
    with pytest.raises(SystemExit):
        view_scene.main([])


def test_play_episode_query_validation():
    with pytest.raises(SystemExit):
        play_episode.main(["--synthetic", "0", "--query", "999"])


@pytest.mark.skipif(HAVE_ENV, reason="the sim module imports; the missing-sim path is moot")
def test_play_episode_reports_missing_sim():
    with pytest.raises(SystemExit) as e:
        play_episode.main(["--synthetic", "0", "--out", "/dev/null"])
    assert "not available yet" in str(e.value)


@pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines not available yet")
def test_play_episode_short_rollout(tmp_path):
    out = tmp_path / "ep.gif"
    assert play_episode.main(["--synthetic", "1", "--policy", "random", "--robots", "2",
                              "--max-decisions", "2", "--dpi", "50", "--out", str(out)]) == 0
    assert out.exists()


def test_episode_viewer_reads_pickle_and_dir(tmp_path):
    import imageio.v2 as iio
    import pickle

    frames = [np.full((8, 12, 3), v, np.uint8) for v in (10, 20, 30)]
    pkl = tmp_path / "ep.pkl"
    snaps = [{"t": float(i), "found": i, "n_casualties": 3, "coverage": 0.1 * i,
              "reward": float(i)} for i in range(3)]
    pickle.dump({"frames": frames, "snapshots": snaps}, open(pkl, "wb"))
    f, s = episode_viewer.load_frames(pkl)
    assert len(f) == 3 and len(s) == 3

    plain = tmp_path / "plain.pkl"
    pickle.dump(frames, open(plain, "wb"))
    f2, s2 = episode_viewer.load_frames(plain)
    assert len(f2) == 3 and s2 == []

    d = tmp_path / "frames"
    d.mkdir()
    for i, fr in enumerate(frames):
        iio.imwrite(d / f"frame_{i:04d}.png", fr)
    f3, s3 = episode_viewer.load_frames(d)
    assert len(f3) == 3 and s3 == []


def test_episode_viewer_slider_updates(tmp_path):
    frames = [np.full((8, 12, 3), v, np.uint8) for v in (10, 200)]
    fig, slider, update = episode_viewer.build_viewer(frames, [], figsize=(3, 2))
    slider.set_val(1)
    assert np.array_equal(fig.axes[0].images[0].get_array(), frames[1])
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_episode_viewer_missing_path(tmp_path):
    with pytest.raises(SystemExit):
        episode_viewer.load_frames(tmp_path / "nope.pkl")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        episode_viewer.load_frames(empty)


@pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines not available yet")
def test_rayfronts_demo_runs(tmp_path):
    import rayfronts_demo
    out = tmp_path / "demo.png"
    assert rayfronts_demo.main(["--synthetic", "1", "--out", str(out), "--checkpoints", "1,2",
                                "--no-mp4"]) == 0
    assert out.exists()


@pytest.mark.skipif(not HAVE_SIM, reason="rlplanner.sim env/baselines not available yet")
def test_rayfronts_demo_accepts_a_query_name(tmp_path):
    import rayfronts_demo
    out = tmp_path / "demo_named.png"
    assert rayfronts_demo.main(["--synthetic", "1", "--out", str(out), "--checkpoints", "1",
                                "--query", "rubble", "--robots", "2", "--no-mp4"]) == 0
    assert out.exists()
    with pytest.raises(SystemExit):
        rayfronts_demo.main(["--synthetic", "1", "--query", "99", "--no-mp4"])


# ---- sensor_inspector ---------------------------------------------------------------------------
def _inspector():
    import sensor_inspector
    return sensor_inspector


def test_sensor_inspector_headless_png(tmp_path):
    si = _inspector()
    out = tmp_path / "insp.png"
    assert si.main(["--synthetic", "0", "--out", str(out), "--dpi", "50", "--sense", "2",
                    "--pov", "60", "45"]) == 0
    assert out.exists() and out.stat().st_size > 0


def test_sensor_inspector_sweep_gif(tmp_path):
    si = _inspector()
    gif = tmp_path / "sweep.gif"
    assert si.main(["--synthetic", "0", "--sweep", "4", "--gif", str(gif), "--gif-every", "2",
                    "--dpi", "45", "--pov", "48", "36"]) == 0
    import imageio.v2 as iio
    assert len(iio.mimread(gif, memtest=False)) >= 2


def test_sensor_inspector_wedge_masks_partition_the_view():
    si = _inspector()
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    from rlplanner.sim.raster import rasterize
    cfg = EnvConfig()
    ras = rasterize(schema.make_synthetic_scene(0, region_m=(160.0, 160.0)), cfg.raster.cell_m)
    cam = np.array([0.0, 0.0, cfg.robot.flight_alt_m])
    obs, far, occ = si.wedge_masks(ras, cfg.sensor, cam, 0.3)
    assert obs.any() and far.any()
    assert not (obs & far).any() and not (obs & occ).any() and not (far & occ).any()
    # everything observed is inside the depth limit, everything far beyond it
    xs, ys = ras.ij_to_xy(*np.nonzero(obs))
    r = np.sqrt(xs ** 2 + ys ** 2 + cam[2] ** 2)
    assert r.max() <= cfg.sensor.depth_limit_m + 2 * cfg.raster.cell_m
    xs, ys = ras.ij_to_xy(*np.nonzero(far))
    assert np.sqrt(xs ** 2 + ys ** 2).max() <= cfg.sensor.visual_range_m + 1e-6


def test_sensor_inspector_pov_projects_the_footprint():
    si = _inspector()
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    from rlplanner.sim.raster import rasterize
    cfg = EnvConfig()
    ras = rasterize(schema.make_synthetic_scene(0, region_m=(160.0, 160.0)), cfg.raster.cell_m)
    cam = np.array([0.0, 0.0, 25.0])
    obs, far, _ = si.wedge_masks(ras, cfg.sensor, cam, 0.0)
    img = si.pov_image(ras, cam, 0.0, cfg.sensor, np.argwhere(obs).astype(np.int32),
                       np.argwhere(far).astype(np.int32), 64, 48)
    assert img.shape == (48, 64, 3)
    assert (img > 0.1).any()                       # something was painted
    empty = si.pov_image(ras, cam, 0.0, cfg.sensor, np.zeros((0, 2), np.int32),
                         np.zeros((0, 2), np.int32), 8, 8)
    assert np.allclose(empty, 0.06)


def test_sensor_inspector_embedding_rgb_separates_classes():
    si = _inspector()
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    from rlplanner.sim.raster import rasterize
    from rlplanner.sim.rayfronts_sim import RayFrontsSim
    from rlplanner.sim.state import RobotState
    cfg = EnvConfig()
    ras = rasterize(schema.make_synthetic_scene(0, region_m=(160.0, 160.0)), cfg.raster.cell_m)
    rng = np.random.default_rng(0)
    rf = RayFrontsSim(ras, cfg, rng)
    rb = RobotState(idx=0, pos=np.array([0.0, 0.0]), alt=25.0, heading=0.3, target_xy=None,
                    target_token_type=0, target_id=-1)
    for k in range(4):
        rf.update([rb], float(k), rng)
    rgb, evr = si.embedding_rgb(rf.vox_feat_sum, rf.observed)
    assert rgb.shape == ras.shape + (3,) and 0.0 < evr <= 1.0
    assert (rgb[~rf.observed] == 0.08).all()
    assert rgb[rf.observed].std() > 0.05          # not a flat colour: classes separate


def test_sensor_inspector_pose_and_query_validation(tmp_path):
    si = _inspector()
    from rlplanner.sim.config import EnvConfig
    n_q = EnvConfig().n_queries
    with pytest.raises(SystemExit):
        si.main(["--synthetic", "0", "--query", "99"])
    with pytest.raises(SystemExit):
        si.main([])
    out = tmp_path / "p.png"
    assert si.main(["--synthetic", "0", "--pose", "0", "0", "30", "45", "-60",
                    "--query", str(n_q - 1),
                    "--out", str(out), "--dpi", "45", "--pov", "48", "36"]) == 0
    assert out.exists()


def test_sensor_inspector_takes_a_query_by_name(tmp_path):
    si = _inspector()
    from rlplanner.sim.config import EnvConfig
    name = EnvConfig().rayfronts.queries[0]
    out = tmp_path / "byname.png"
    assert si.main(["--synthetic", "0", "--query", name, "--out", str(out), "--dpi", "40",
                    "--pov", "40", "30"]) == 0
    assert out.exists()


def test_sensor_inspector_extra_query_needs_a_text_tower(tmp_path):
    """The factorized hand table has no encoder behind it, so a free-text query must say so."""
    si = _inspector()
    with pytest.raises(SystemExit, match="cannot be embedded"):
        si.main(["--synthetic", "0", "--extra-query", "a fire hydrant", "--out",
                 str(tmp_path / "x.png"), "--dpi", "40", "--pov", "40", "30"])


def test_sensor_inspector_extra_query_outside_the_mission_list(tmp_path):
    """A phrase the table can embed becomes an extra similarity bar without touching the belief."""
    si = _inspector()
    out = tmp_path / "extra.png"
    assert si.main(["--synthetic", "0", "--extra-query", "rubble", "--out", str(out),
                    "--dpi", "45", "--sense", "2", "--pov", "48", "36"]) == 0
    assert out.exists()


def test_sensor_inspector_extra_query_with_the_cached_text_embeddings(tmp_path):
    """The SigLIP cache path: a query outside the mission list, resolved from the cached bank.

    A phrase in neither the bank nor the hand table needs the text tower itself, which is not
    downloaded here — `test_..._needs_a_text_tower` covers that message.
    """
    si = _inspector()
    from pathlib import Path as _P
    cache = (_P(si.__file__).resolve().parents[1] / "src" / "rlplanner" / "sim" / "data"
             / "text_embeddings_siglip_vitb16.json")
    if not cache.exists():
        pytest.skip("no cached text embeddings")
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(f"rayfronts:\n  embeddings_path: {cache}\n  embedding_dim: 24\n")
    out = tmp_path / "extra_siglip.png"
    assert si.main(["--synthetic", "0", "--config", str(cfg_yaml), "--extra-query", "bus stop",
                    "--out", str(out), "--dpi", "45", "--sense", "2", "--pov", "48", "36"]) == 0
    assert out.exists()


def test_sensor_inspector_bars_come_from_query_sim(tmp_path):
    si = _inspector()
    import numpy as _np
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(0, region_m=(160.0, 160.0))
    insp = si.Inspector(scene, cfg, si.default_pose(scene, cfg), pov=(40, 30), zoom=120.0, dpi=40)
    insp.sense(2)
    grids = insp.query_grids()
    assert list(grids) == list(cfg.rayfronts.queries)
    for q, g in grids.items():
        assert g.shape == insp.raster.shape
        assert _np.allclose(g, insp.rf.query_sim(q), atol=1e-6)
    n0 = insp.rf.n_query_calls
    insp.query_grids()                       # cached per belief version, not re-scanned
    assert insp.rf.n_query_calls == n0
    insp.sense(1)
    insp.query_grids()
    assert insp.rf.n_query_calls > n0


def test_sensor_inspector_key_and_click_handlers():
    si = _inspector()
    from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    import matplotlib.pyplot as plt

    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(0, region_m=(160.0, 160.0))
    insp = si.Inspector(scene, cfg, si.default_pose(scene, cfg), pov=(48, 36), zoom=160.0, dpi=45)
    insp.sense(1)
    fig = insp.build()
    insp.connect("/dev/null")
    canvas = fig.canvas

    def key(k):
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, k))

    yaw0, alt0, pitch0, n0 = insp.yaw, insp.alt, insp.cfg.sensor.pitch_deg, insp.n_sensed
    key("left")
    assert insp.yaw > yaw0
    key("right")
    assert insp.yaw == pytest.approx(yaw0)
    key("+")
    assert insp.alt == alt0 + si.ALT_STEP
    key("-")
    assert insp.alt == pytest.approx(alt0)
    key("[")
    assert insp.cfg.sensor.pitch_deg == pitch0 - si.PITCH_STEP
    key("]")
    assert insp.cfg.sensor.pitch_deg == pytest.approx(pitch0)
    key("s")
    assert insp.n_sensed == n0 + 1
    key("3")
    assert insp.query == min(3, len(insp.queries()) - 1)
    xy0 = (insp.x, insp.y)
    key("up")
    assert (insp.x, insp.y) != xy0
    key("a")
    assert insp.n_sensed > n0 + 1
    key("r")
    assert insp.n_sensed == 0 and not insp.rf.observed.any()

    px, py = insp.ax_gt.transData.transform((insp.x + 5.0, insp.y + 5.0))
    canvas.callbacks.process("button_press_event",
                             MouseEvent("button_press_event", canvas, px, py, MouseButton.LEFT))
    assert abs(insp.x - (xy0[0])) >= 0.0                      # moved to the clicked point
    canvas.callbacks.process("motion_notify_event",
                             MouseEvent("motion_notify_event", canvas, px, py))
    assert insp.probe_ij is not None
    plt.close(fig)


# ---- sensor_inspector: geometry and pose validation (QA pass) -------------------------------------
def _flat_scene(w: float = 200.0):
    from rlplanner.scene import schema
    return schema.Scene(meta=schema.Meta(region=(-w / 2, -w / 2, w / 2, w / 2)),
                        damage_field=schema.DamageField(kind="uniform", params={"inside": 0.0}))


def _flat_raster(cell: float = 2.0, w: float = 200.0):
    from rlplanner.sim.raster import rasterize
    return rasterize(_flat_scene(w), cell)


@pytest.mark.parametrize("yaw_deg", [0.0, 40.9, 177.0, -93.0])
def test_sensor_inspector_pov_centres_the_boresight_cell(yaw_deg):
    """A cell exactly on the boresight projects to the middle column of the POV frame."""
    si = _inspector()
    import math
    from rlplanner.sim.config import EnvConfig
    cfg = EnvConfig()
    ras = _flat_raster()
    cam = np.array([0.0, 0.0, 25.0])
    yaw = math.radians(yaw_deg)
    i, j = ras.xy_to_ij(cam[0] + 20.0 * math.cos(yaw), cam[1] + 20.0 * math.sin(yaw))
    cx, cy = ras.ij_to_xy(i, j)
    yaw = math.atan2(cy - cam[1], cx - cam[0])              # aim exactly at the cell centre
    w, h = 200, 150
    img = si.pov_image(ras, cam, yaw, cfg.sensor, np.array([[i, j]], np.int32),
                       np.zeros((0, 2), np.int32), w, h)
    painted = np.argwhere((img != 0.06).any(-1))
    assert painted.size, "the boresight cell was not painted"
    assert painted[:, 1].mean() == pytest.approx(0.5 * w, abs=1.0)


def test_sensor_inspector_occluded_cells_are_absent_from_both_panels():
    si = _inspector()
    import math
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    from rlplanner.sim.raster import rasterize
    cfg = EnvConfig()
    scene = _flat_scene(200.0)
    scene.buildings = [schema.Building(id="b0", center=(20.0, 0.0), size=(12.0, 40.0),
                                       height_m=45.0, category="highrise")]
    ras = rasterize(scene, cfg.raster.cell_m)
    cam = np.array([0.0, 0.0, 25.0])
    obs, far, occ = si.wedge_masks(ras, cfg.sensor, cam, 0.0)
    shadow = np.argwhere(occ)
    assert shadow.size, "the tall building casts no shadow"
    xs, ys = ras.ij_to_xy(shadow[:, 0], shadow[:, 1])
    assert (xs > 20.0).all()                                 # every shadow cell is behind it
    assert not (occ & obs).any() and not (occ & far).any()   # panel 1 keeps them separate
    for grid in (obs, far):                                  # nothing outside the azimuth wedge
        gx, gy = ras.ij_to_xy(*np.nonzero(grid))
        if gx.size:
            az = np.abs(np.arctan2(gy - cam[1], gx - cam[0]))
            assert az.max() <= math.radians(0.5 * cfg.sensor.hfov_deg) + 1e-6
    # the same cells never reach the POV: it is fed obs/far only
    k = int(np.argmax(xs))
    i, j = int(shadow[k, 0]), int(shadow[k, 1])
    img = si.pov_image(ras, cam, 0.0, cfg.sensor, np.argwhere(obs).astype(np.int32),
                       np.argwhere(far).astype(np.int32), 160, 120)
    flat = rasterize(_flat_scene(200.0), cfg.raster.cell_m)  # same cell with nothing in the way
    obs2, far2, _ = si.wedge_masks(flat, cfg.sensor, cam, 0.0)
    assert obs2[i, j] or far2[i, j]
    img2 = si.pov_image(flat, cam, 0.0, cfg.sensor, np.argwhere(obs2).astype(np.int32),
                        np.argwhere(far2).astype(np.int32), 160, 120)
    px, py = _project(ras, cfg.sensor, cam, 0.0, i, j, 160, 120)
    assert not np.allclose(img[py, px], img2[py, px])


def _project(ras, sensor, cam, yaw, i, j, w, h):
    import math
    x, y = ras.ij_to_xy(i, j)
    d = np.array([x - cam[0], y - cam[1], float(ras.height[i, j]) - cam[2]])
    pitch = math.radians(sensor.pitch_deg)
    fwd = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch),
                    math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.cross(right, fwd)
    z = d @ fwd
    u = (d @ right) / z / math.tan(0.5 * math.radians(sensor.hfov_deg))
    v = (d @ up) / z / math.tan(0.5 * math.radians(sensor.vfov_deg))
    return (int(np.clip((u + 1.0) * 0.5 * w, 0, w - 1)), int(np.clip((1.0 - v) * 0.5 * h, 0, h - 1)))


def test_sensor_inspector_footprint_follows_pitch_and_altitude():
    """Pitching down pulls the footprint in; climbing walks the aim point out."""
    si = _inspector()
    import math
    from rlplanner.sim.config import EnvConfig
    ras = _flat_raster()

    def mean_range(alt, pitch):
        cfg = EnvConfig()
        cfg.sensor.pitch_deg = pitch
        obs, _, _ = si.wedge_masks(ras, cfg.sensor, np.array([0.0, 0.0, alt]), 0.0)
        xs, ys = ras.ij_to_xy(*np.nonzero(obs))
        return float(np.hypot(xs, ys).mean()), int(obs.sum())

    shallow, _ = mean_range(25.0, -20.0)
    steep, _ = mean_range(25.0, -70.0)
    assert steep < shallow                             # pitching down pulls the footprint in
    # altitude: the boresight aim point walks out, the far edge (a slant depth limit) walks in
    from rlplanner.scene import schema
    cfg = EnvConfig()
    aims, edges = [], []
    for alt in (12.0, 25.0):
        insp = si.Inspector(_flat_scene(), cfg, [0.0, 0.0, alt, 0.0, -50.0], pov=(24, 18), dpi=40)
        aims.append(math.hypot(*insp.aim_xy()))
        obs, _, _ = si.wedge_masks(ras, cfg.sensor, np.array([0.0, 0.0, alt]), 0.0)
        xs, ys = ras.ij_to_xy(*np.nonzero(obs))
        edges.append(float(np.hypot(xs, ys).max()))
    assert aims[0] < aims[1] and edges[0] > edges[1]
    assert edges[1] == pytest.approx(math.sqrt(cfg.sensor.depth_limit_m ** 2 - 25.0 ** 2), abs=3.0)


def test_sensor_inspector_rejects_an_unusable_pose(tmp_path):
    si = _inspector()
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    from rlplanner.sim.raster import rasterize
    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(0)
    args = ["--synthetic", "0", "--dpi", "40", "--pov", "40", "30", "--out", str(tmp_path / "x.png")]
    with pytest.raises(SystemExit, match="outside the region"):
        si.main(["--pose", "9999", "9999", "25", "0", "-50"] + args)
    ras = rasterize(scene, cfg.raster.cell_m)
    i, j = np.argwhere(ras.obstacle_mask(cfg.robot.flight_alt_m, cfg.robot.clearance_m))[0]
    x, y = ras.ij_to_xy(int(i), int(j))
    with pytest.raises(SystemExit, match="inside an obstacle"):
        si.main(["--pose", str(x), str(y), "25", "0", "-50"] + args)
    with pytest.raises(SystemExit, match="depth_limit_m"):
        si.main(["--pose", "0", "0", "60", "0", "-50"] + args)


def test_sensor_inspector_says_so_when_the_drone_is_blocked_instead_of_crashing():
    si = _inspector()
    import matplotlib.pyplot as plt
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    from rlplanner.sim.raster import rasterize
    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(0)
    ras = rasterize(scene, cfg.raster.cell_m)
    i, j = np.argwhere(ras.obstacle_mask(cfg.robot.flight_alt_m, cfg.robot.clearance_m))[0]
    x, y = ras.ij_to_xy(int(i), int(j))
    insp = si.Inspector(scene, cfg, [x, y, 25.0, 0.0, -50.0], pov=(40, 30), zoom=120.0, dpi=40)
    insp.sense(1)
    assert any("obstacle" in w for w in insp.check_pose())
    fig = insp.build()
    assert "!" in fig._suptitle.get_text()
    assert insp.frame().ndim == 3                      # renders, does not raise
    plt.close(fig)


def test_sensor_inspector_border_and_yaw_wraparound_keys():
    si = _inspector()
    import math
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(0)
    x0, y0, x1, y1 = scene.region
    insp = si.Inspector(scene, cfg, [x1 - 1.0, y1 - 1.0, 25.0, 0.0, -50.0], pov=(40, 30),
                        zoom=120.0, dpi=40)
    insp.sense(1)
    fig = insp.build()
    insp.connect("/dev/null")

    def key(k):
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, k))

    for _ in range(30):                                # 30 x 15 deg = 450 deg: past +-pi
        key("left")
    assert math.isfinite(insp.yaw)
    for _ in range(6):                                 # push into the corner
        key("up")
    assert x0 <= insp.x <= x1 and y0 <= insp.y <= y1
    for _ in range(6):                                 # descend below the tallest building
        key("-")
    assert insp.alt == 2.0 or insp.alt < 25.0
    assert insp.check_pose()                           # the inspector says so
    assert insp.frame().ndim == 3
    plt.close(fig)


def test_sensor_inspector_handles_every_key(tmp_path):
    si = _inspector()
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent
    from rlplanner.scene import schema
    from rlplanner.sim.config import EnvConfig
    cfg = EnvConfig()
    scene = schema.make_synthetic_scene(0)
    out = tmp_path / "w.png"
    insp = si.Inspector(scene, cfg, si.default_pose(scene, cfg), pov=(40, 30), zoom=120.0, dpi=40)
    insp.sense(1)
    fig = insp.build()
    insp.connect(str(out))

    def key(k):
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, k))

    xy0, alt0 = (insp.x, insp.y), insp.alt
    key("down")
    assert (insp.x, insp.y) != xy0
    key("=")
    assert insp.alt == alt0 + si.ALT_STEP
    key("9")                                           # clamped to the last query
    assert insp.query == min(9, len(insp.queries()) - 1)
    key("z")                                           # unhandled: no redraw, no crash
    key("w")
    assert out.exists()
    key("q")
    assert not plt.fignum_exists(fig.number)


def test_sensor_inspector_headless_on_a_v2_scene(tmp_path):
    si = _inspector()
    scenes = sorted((Path(__file__).resolve().parents[1] / "data" / "scenes_v2").glob("*_36.json"))
    if not scenes:
        pytest.skip("no v2 scenes exported")
    out = tmp_path / "v2.png"
    assert si.main(["--scene", str(scenes[0]), "--out", str(out), "--dpi", "45", "--sense", "1",
                    "--pov", "48", "36", "--cell", "4"]) == 0
    assert out.exists() and out.stat().st_size > 0


def test_sensor_inspector_sweep_one_step(tmp_path):
    si = _inspector()
    gif = tmp_path / "one.gif"
    assert si.main(["--synthetic", "0", "--sweep", "1", "--gif", str(gif), "--gif-every", "1",
                    "--dpi", "40", "--pov", "40", "30"]) == 0
    assert gif.exists()


def test_sensor_inspector_headless_without_an_output_writes_the_default(tmp_path, monkeypatch,
                                                                        capsys):
    si = _inspector()
    monkeypatch.setenv("MPLBACKEND", "Agg")
    default = tmp_path / "inspector.png"
    monkeypatch.setattr(si, "DEFAULT_OUT", str(default))
    assert si.main(["--synthetic", "0", "--sweep", "0", "--dpi", "40", "--pov", "40", "30"]) == 0
    assert default.exists()
    assert "no window available" in capsys.readouterr().out
