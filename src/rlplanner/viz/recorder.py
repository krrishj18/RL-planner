"""Roll out a policy in an env and record three-panel frames (CONTRACTS.md 9)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from rlplanner.viz.frame import n_found, render_frame

MACRO_BLOCK = 16


def _pad_to_macro_block(frame: np.ndarray, block: int = MACRO_BLOCK) -> np.ndarray:
    """Pad with white so h/w are multiples of `block` (ffmpeg would otherwise rescale)."""
    h, w = frame.shape[:2]
    ph, pw = (-h) % block, (-w) % block
    if not (ph or pw):
        return frame
    return np.pad(frame, ((0, ph), (0, pw), (0, 0)), constant_values=255)


class EpisodeRecorder:
    """Drives `env` with `policy`, rendering a frame every `every_n_decisions` decisions."""

    def __init__(self, env, every_n_decisions: int = 1, query_idx: int | str = 0,
                 focus_robot: int | None = 0, figsize: tuple[float, float] = (18.0, 6.0),
                 dpi: int = 100, robot: int | None = None, show_local: bool = False):
        if int(every_n_decisions) < 1:
            raise ValueError(f"every_n_decisions must be >= 1, got {every_n_decisions}")
        self.env = env
        self.every_n_decisions = int(every_n_decisions)
        self.query_idx = query_idx if isinstance(query_idx, str) else int(query_idx)
        self.focus_robot = focus_robot
        self.robot = robot
        self.show_local = bool(show_local)
        self.figsize = figsize
        self.dpi = int(dpi)
        self.frames: list[np.ndarray] = []
        self.snapshots: list[dict[str, Any]] = []
        self.info_last: dict[str, Any] = {}

    # -- rollout ---------------------------------------------------------------------------
    def _default_max(self) -> int:
        cfg = getattr(self.env, "cfg", None)
        if cfg is None:
            return 500
        return int(np.ceil(float(cfg.t_max_s) / float(cfg.decision_dt))) + 1

    def _focus(self, state) -> int | None:
        if self.focus_robot is None:
            return None
        f = int(self.focus_robot)
        return f if 0 <= f < len(state.robots) else None

    def _robot(self, state) -> int | None:
        """Whose map the belief panel draws; out of range falls back to the team union."""
        if self.robot is None:
            return None
        r = int(self.robot)
        return r if 0 <= r < len(state.robots) else None

    def _capture(self, actions: np.ndarray | None) -> None:
        st = self.env.state
        self.frames.append(render_frame(st, query_idx=self.query_idx,
                                        focus_robot=self._focus(st), figsize=self.figsize,
                                        dpi=self.dpi, robot=self._robot(st),
                                        show_local=self.show_local))
        self.snapshots.append(self.snapshot(actions))

    def snapshot(self, actions: np.ndarray | None = None) -> dict[str, Any]:
        """Lightweight per-decision record for the interactive viewer."""
        st = self.env.state
        return {
            "t": float(st.t),
            "decision_idx": int(st.decision_idx),
            "robot_xy": np.array([r.pos for r in st.robots], dtype=np.float32),
            "found": n_found(st),
            "n_casualties": int(st.n_casualties),
            "coverage": float(st.coverage),
            "reward": float(st.cum_reward),
            "actions": (None if actions is None or np.ndim(actions) != 1
                        else np.asarray(actions).astype(np.int32).tolist()),
        }

    def run(self, policy, max_decisions: int | None = None, seed: int | None = None,
            progress: bool = False) -> list[np.ndarray]:
        """Reset, roll out until done (or `max_decisions`), return the recorded frames."""
        self.frames, self.snapshots = [], []
        obs = self.env.reset(seed) if seed is not None else self.env.reset()
        if self.env.state is None:
            raise RuntimeError(f"{type(self.env).__name__}.reset() left env.state = None")
        limit = self._default_max() if max_decisions is None else int(max_decisions)
        self._capture(None)
        it = range(limit)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="decisions")
        for i in it:
            actions = np.asarray(policy.act(obs, self.env.state))
            obs, reward, done, info = self.env.step(actions)
            self.info_last = info
            if ((i + 1) % self.every_n_decisions == 0) or done:
                self._capture(actions)
            if done:
                break
        return self.frames

    # CONTRACTS.md spells this `record`
    def record(self, policy, every_n_decisions: int | None = None,
               max_decisions: int | None = None, **kw) -> list[np.ndarray]:
        if every_n_decisions is not None:
            self.every_n_decisions = int(every_n_decisions)
        return self.run(policy, max_decisions=max_decisions, **kw)

    # -- output ----------------------------------------------------------------------------
    def _check_frames(self) -> list[np.ndarray]:
        if not self.frames:
            raise RuntimeError("no frames recorded: call run(policy) first")
        return self.frames

    def save_mp4(self, path: str | Path, fps: int = 4) -> Path:
        import imageio.v2 as iio

        frames = [_pad_to_macro_block(f) for f in self._check_frames()]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with iio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None) as w:
                for f in frames:
                    w.append_data(f)
        except Exception as exc:  # missing ffmpeg is the usual cause
            raise RuntimeError(f"could not write mp4 {path}: {exc}. Is imageio-ffmpeg installed?") from exc
        return path

    def save_gif(self, path: str | Path, fps: int = 4, downscale: int = 2) -> Path:
        import imageio.v2 as iio

        frames = self._check_frames()
        if downscale > 1:
            frames = [f[::downscale, ::downscale] for f in frames]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.mimsave(path, frames, format="GIF", duration=1.0 / max(1e-6, fps), loop=0)
        return path

    def save_frames(self, directory: str | Path, prefix: str = "frame") -> list[Path]:
        import imageio.v2 as iio

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for i, f in enumerate(self._check_frames()):
            p = d / f"{prefix}_{i:04d}.png"
            iio.imwrite(p, f)
            out.append(p)
        return out

    def save_pickle(self, path: str | Path, raw: bool = False) -> Path:
        """Frames + snapshots for `scripts/episode_viewer.py` (PNG-encoded unless `raw`)."""
        import imageio.v2 as iio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = self._check_frames()
        if raw:
            blob = {"frames": frames, "snapshots": self.snapshots}
        else:
            blob = {"frames_png": [iio.imwrite("<bytes>", f, format="PNG") for f in frames],
                    "snapshots": self.snapshots}
        with open(path, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path


__all__ = ["EpisodeRecorder"]
