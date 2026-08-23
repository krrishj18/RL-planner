import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import numpy as np
import pytest

from rlplanner.viz.recorder import EpisodeRecorder, _pad_to_macro_block
from viz_mocks import FirstValidPolicy, MockEnv

SMALL = dict(figsize=(6.0, 2.0), dpi=50)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    env = MockEnv(seed=0, n_robots=2, n_steps=3)
    rec = EpisodeRecorder(env, every_n_decisions=1, query_idx=0, **SMALL)
    rec.run(FirstValidPolicy())
    return rec


def test_run_stops_at_episode_end(recorded):
    assert len(recorded.frames) == 4          # reset + 3 decisions
    assert len(recorded.snapshots) == 4
    for f in recorded.frames:
        assert f.dtype == np.uint8 and f.shape == (100, 300, 3)


def test_snapshots_carry_the_documented_fields(recorded):
    s = recorded.snapshots[-1]
    assert set(s) >= {"t", "decision_idx", "robot_xy", "found", "coverage", "reward", "actions"}
    assert s["robot_xy"].shape == (2, 2)
    assert s["actions"] is not None and len(s["actions"]) == 2
    assert recorded.snapshots[0]["actions"] is None
    assert [s["t"] for s in recorded.snapshots] == sorted(s["t"] for s in recorded.snapshots)


def test_save_gif(recorded, tmp_path):
    p = recorded.save_gif(tmp_path / "ep.gif", fps=4)
    assert p.exists() and p.stat().st_size > 0
    import imageio.v2 as iio
    assert len(iio.mimread(p)) == len(recorded.frames)


def test_save_mp4(recorded, tmp_path):
    p = recorded.save_mp4(tmp_path / "sub" / "ep.mp4", fps=4)
    assert p.exists() and p.stat().st_size > 0


def test_save_frames_and_pickle(recorded, tmp_path):
    paths = recorded.save_frames(tmp_path / "frames")
    assert len(paths) == len(recorded.frames) and all(p.exists() for p in paths)
    import pickle
    pkl = recorded.save_pickle(tmp_path / "ep.pkl")
    obj = pickle.load(open(pkl, "rb"))
    assert len(obj["frames_png"]) == len(recorded.frames)
    assert len(obj["snapshots"]) == len(recorded.snapshots)

    raw = recorded.save_pickle(tmp_path / "raw.pkl", raw=True)
    assert len(pickle.load(open(raw, "rb"))["frames"]) == len(recorded.frames)
    assert raw.stat().st_size > pkl.stat().st_size

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import episode_viewer
    for p in (pkl, raw):
        frames, snaps = episode_viewer.load_frames(p)
        assert len(frames) == len(recorded.frames) and len(snaps) == len(recorded.snapshots)
        assert np.array_equal(frames[0], recorded.frames[0])


def test_every_n_decisions_subsamples():
    env = MockEnv(seed=1, n_robots=1, n_steps=3)
    rec = EpisodeRecorder(env, every_n_decisions=2, **SMALL)
    frames = rec.run(FirstValidPolicy())
    assert len(frames) == 3                    # reset, decision 2, decision 3 (done)


def test_max_decisions_caps_the_rollout():
    env = MockEnv(seed=1, n_robots=1, n_steps=50)
    rec = EpisodeRecorder(env, **SMALL)
    assert len(rec.run(FirstValidPolicy(), max_decisions=2)) == 3


def test_bad_every_n_raises():
    with pytest.raises(ValueError):
        EpisodeRecorder(MockEnv(), every_n_decisions=0)


def test_save_before_run_raises(tmp_path):
    rec = EpisodeRecorder(MockEnv(), **SMALL)
    with pytest.raises(RuntimeError):
        rec.save_gif(tmp_path / "x.gif")


def test_focus_robot_out_of_range_is_dropped():
    env = MockEnv(seed=1, n_robots=1, n_steps=1)
    rec = EpisodeRecorder(env, focus_robot=5, **SMALL)
    assert len(rec.run(FirstValidPolicy())) == 2


def test_pad_to_macro_block():
    out = _pad_to_macro_block(np.zeros((100, 300, 3), np.uint8))
    assert out.shape[0] % 16 == 0 and out.shape[1] % 16 == 0
    assert _pad_to_macro_block(np.zeros((32, 48, 3), np.uint8)).shape == (32, 48, 3)
