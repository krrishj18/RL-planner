"""Live episode window (CONTRACTS.md 9): ground truth | belief | status, stepped as it is watched.

The env is stepped inside the animation callback (`policy.act` -> `env.step`), so nothing is
pre-recorded. Every artist is created once in `build()` and updated in place in `refresh()`
(images `set_data`, collections `set_segments`/`set_offsets`, texts `set_text`); the ground truth
is baked to a single image because redrawing a v2 scene plot costs ~0.5 s a frame.

The belief panel shows what RayFronts publishes and nothing derived from it: the observed cells
coloured by the selected query's similarity, segments as translucent patches in their mean-feature
colour, rays (origin + bearing, length growing with the observation count, coloured by the same
query) and frontiers (cells + cluster medoids), plus the focused robot's tokens with a marker per
type taken from the sim's own `TOKEN_TYPE_NAMES`.

`--robot r` (key `v`) swaps the team union for what that robot knows — its `RobotView`, the visited
records it holds and its peer cache — and `--show-local` adds the ego crop the actor plans in (a
panel, so it is a launch flag, not a key). The query's similarity is taken **once** per refresh (`query_view`), never per artist.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge

from rlplanner.sim import state as S
from rlplanner.viz import palette as P
from rlplanner.viz.frame import (belief_legend_handles, belief_view, casualty_mask,
                                 comms_links, draw_local_panel, found_mask, local_crop, n_found,
                                 peer_arrows, query_names, query_view, ray_geometry,
                                 robot_target_label, segment_rgba, token_type_names,
                                 truth_legend_handles)
from rlplanner.viz.layout import legend_outside
from rlplanner.viz.scene_plot import region_extent

HEATMAP_MAX_PX = 520          # longest side of the belief image; larger rasters are block-maxed
GT_DIM = 0.55                 # the ground truth is dimmed so the live overlays read on top of it
BAKE_MAX_OBJECTS = 4000       # above this a scene plot is too slow to bake: use the raster classes
BAKE_PX = 800
FRONTIER_MAX_POINTS = 4000    # frontier cells drawn per frame (strided beyond this)
FLASH_FRAMES = 5              # how many refreshes a find stays flashed
LOG_LINES = 6
SPEED_MIN, SPEED_MAX = 0.25, 32.0


# ---- static background ---------------------------------------------------------------------
def raster_rgb(raster, dim: float = GT_DIM) -> np.ndarray:
    """Class-coloured, height-shaded raster image (what the sim actually sees), dimmed."""
    from rlplanner.viz.raster_plot import class_rgb

    return np.clip(class_rgb(raster, shade=True) * float(dim), 0.0, 1.0)


def bake_scene_rgb(scene, px: int = BAKE_PX, dim: float = GT_DIM, dpi: int = 100) -> np.ndarray:
    """`plot_scene` rendered once to an RGB image covering exactly the region, dimmed."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from rlplanner.viz.scene_plot import plot_scene

    x0, y0, x1, y1 = region_extent(scene)
    w = int(px)
    h = max(1, int(round(px * (y1 - y0) / (x1 - x0))))
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    plot_scene(scene, ax=ax, show_damage=True, show_humans=False, legend=False, title="")
    ax.set_axis_off()
    ax.set_aspect("auto")                       # the figure already has the region's aspect
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba(), np.uint8)[..., :3].astype(np.float64) / 255.0
    fig.clear()
    return np.clip(rgb * float(dim), 0.0, 1.0)


def scene_objects(scene) -> int:
    return (len(scene.buildings) + len(scene.roads) + len(scene.debris) + len(scene.vehicles)
            + len(scene.props) + len(scene.blocks))


def gt_background(scene, raster, mode: str = "auto", dim: float = GT_DIM):
    """(image, imshow origin) for the ground-truth panel. `auto` bakes the scene plot when it is
    cheap enough, else falls back to the raster class image."""
    if mode not in ("auto", "scene", "raster"):
        raise ValueError(f"gt background mode {mode!r} not in ('auto', 'scene', 'raster')")
    if mode == "scene" or (mode == "auto" and scene_objects(scene) <= BAKE_MAX_OBJECTS):
        return bake_scene_rgb(scene, dim=dim), "upper"
    return raster_rgb(raster, dim=dim), "lower"


# ---- per-frame geometry ---------------------------------------------------------------------
def frontier_points(state: S.EnvState, view=None) -> tuple[np.ndarray, np.ndarray]:
    """(cluster medoids [n, 2], marker sizes) — area grows with the cluster's info gain."""
    cl = list((view.frontiers if view is not None else state.frontier_clusters) or [])
    if not cl:
        return np.zeros((0, 2)), np.zeros(0)
    R = float(state.cfg.rayfronts.frontier_ig_radius_m)
    cell = float(getattr(state.raster, "cell_m", state.cfg.raster.cell_m))
    full = max(1.0, math.pi * (R / cell) ** 2)
    xy = np.array([np.asarray(c.centroid_xy, float) for c in cl])
    ig = np.array([max(0.0, float(c.info_gain)) for c in cl])
    return xy, 25.0 + 400.0 * np.clip(ig / full, 0.0, 1.0)


def frontier_cells(state: S.EnvState, max_points: int = FRONTIER_MAX_POINTS,
                   view=None) -> np.ndarray:
    """The frontier cells themselves, strided down to `max_points`."""
    m = view.frontier_mask if view is not None else getattr(state, "frontier_mask", None)
    if m is None:
        return np.zeros((0, 2))
    ij = np.argwhere(np.asarray(m, bool))
    if ij.shape[0] == 0:
        return np.zeros((0, 2))
    if ij.shape[0] > max_points:
        ij = ij[:: int(math.ceil(ij.shape[0] / max_points))]
    x, y = state.raster.ij_to_xy(ij[:, 0], ij[:, 1])
    return np.stack([np.asarray(x, float), np.asarray(y, float)], axis=1)


def human_points(state: S.EnvState) -> dict[str, np.ndarray]:
    """Casualty positions split by whether they have been found, plus the bystanders."""
    found = found_mask(state)
    cas = casualty_mask(state)
    xy = np.array([[float(h.pos[0]), float(h.pos[1])] for h in state.scene.humans]) \
        if state.scene.humans else np.zeros((0, 2))
    if xy.shape[0] == 0:
        return {k: np.zeros((0, 2)) for k in ("unfound", "found", "bystander")}
    return {"unfound": xy[cas & ~found], "found": xy[cas & found], "bystander": xy[~cas]}


def block_max(grid: np.ndarray, s: int, how: str = "max") -> np.ndarray:
    """Reduce a grid by an integer factor, keeping hot cells (max) or coverage (any)."""
    if s <= 1:
        return grid
    ny, nx = grid.shape
    h, w = ny // s, nx // s
    if h == 0 or w == 0:
        return grid
    b = grid[:h * s, :w * s].reshape(h, s, w, s)
    return b.any((1, 3)) if how == "any" else b.max((1, 3))


def find_events(state: S.EnvState, info: dict | None, prev_found: np.ndarray) -> list[dict]:
    """Casualties found since the last decision: from the env's own events when it emits them,
    else by diffing the found mask (which robot did it is then unknown)."""
    found = found_mask(state)
    cas = casualty_mask(state)
    fresh = np.flatnonzero(found & cas & ~prev_found)
    by_human = {}
    for e in (info or {}).get("events_this_step", []) or []:
        if getattr(e, "kind", None) == "found":
            p = getattr(e, "payload", {}) or {}
            k = p.get("human_idx", p.get("human", -1))
            by_human[int(k)] = int(p.get("robot", -1))
    out = []
    for k in fresh:
        h = state.scene.humans[int(k)]
        out.append({"human_idx": int(k), "robot": by_human.get(int(k), -1),
                    "xy": (float(h.pos[0]), float(h.pos[1])), "container": h.container,
                    "id": getattr(h, "id", str(int(k))), "t": float(state.t)})
    return out


class LiveViewer:
    """Steps `env` with `policy` inside the animation callback and updates the artists in place."""

    KEYS = ("space play/pause | n step | +/- speed | 0-9 query | f focus | v robot map | "
            "s segments | r restart | R new seed | w png | q quit")

    def __init__(self, env, policy, query: int | str = 0, focus: int = 0, speed: float = 4.0,
                 max_decisions: int | None = None, seed: int = 0,
                 figsize: tuple[float, float] = (19.2, 9.6), dpi: int = 100, gt: str = "auto",
                 png: str | Path = "runs/live_frame.png", heatmap_max_px: int = HEATMAP_MAX_PX,
                 autoplay: bool = True, robot: int | None = None, show_local: bool = False,
                 segments: bool = True):
        self.env = env
        self.policy = policy
        self.query: int | str = query
        self.robot = None if robot is None else int(robot)
        self.show_local = bool(show_local)
        self.segments = bool(segments)
        self.speed = float(np.clip(speed, SPEED_MIN, SPEED_MAX))
        self.max_decisions = None if max_decisions is None else int(max_decisions)
        self.seed = int(seed)
        self.figsize = figsize
        self.dpi = int(dpi)
        self.gt_mode = gt
        self.png = Path(png)
        self.heatmap_max_px = int(heatmap_max_px)
        self.playing = bool(autoplay)
        self.pending = 0
        self.done = False
        self.n_decisions = 0
        self.writer = None
        self.fig = None
        self.ax_local = None
        self.anim = None
        self.draw_times: list[float] = []      # artist updates
        self.render_times: list[float] = []    # canvas draws
        self.step_times: list[float] = []      # policy + env
        self.frame_times: list[float] = []     # wall time between canvas draws
        self.record_rate = 0.0
        self._last_draw = 0.0
        self.log: list[str] = []
        self.flashes: list[list] = []          # [x, y, frames left]
        self.obs = self.env.reset(self.seed)
        st = self.env.state
        n_r = len(st.robots)
        self.focus = int(focus) % n_r if n_r else 0
        if self.robot is not None and n_r:
            self.robot %= n_r
        self.decision_obs = self.obs
        self.decision_actions: np.ndarray | None = None
        self._prev_found = found_mask(st).copy()
        self.hist_t: list[float] = []
        self.hist_found: list[float] = []
        self.hist_cov: list[float] = []
        self._record_hist()

    # -- episode ------------------------------------------------------------------------------
    def _record_hist(self) -> None:
        st = self.env.state
        n_cas = max(1, st.n_casualties)
        self.hist_t.append(float(st.t))
        self.hist_found.append(self.n_found / n_cas)
        self.hist_cov.append(float(st.coverage))

    @property
    def n_found(self) -> int:
        return n_found(self.env.state)

    def reset(self, seed: int | None = None) -> None:
        """Restart the episode (same seed unless one is given) and clear the history."""
        if seed is not None:
            self.seed = int(seed)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(self.seed)
        self.obs = self.env.reset(self.seed)
        self.decision_obs, self.decision_actions = self.obs, None
        self.done = False
        self.n_decisions = 0
        self.hist_t, self.hist_found, self.hist_cov = [], [], []
        self.log, self.flashes = [], []
        self._prev_found = found_mask(self.env.state).copy()
        self._record_hist()
        n_r = max(0, len(self.env.state.robots) - 1)
        self.focus = min(self.focus, n_r)
        if self.robot is not None:
            self.robot = min(self.robot, n_r)

    def advance(self) -> bool:
        """One decision: `policy.act` then `env.step`. False when the episode is already over."""
        if self.done:
            return False
        if self.max_decisions is not None and self.n_decisions >= self.max_decisions:
            self.done = True
            return False
        t0 = time.perf_counter()
        actions = np.asarray(self.policy.act(self.obs, self.env.state))
        self.decision_obs, self.decision_actions = self.obs, actions
        self.obs, _reward, done, info = self.env.step(actions)
        self.step_times.append(time.perf_counter() - t0)
        self.n_decisions += 1
        self.done = bool(done)
        self._log_finds(info)
        self._record_hist()
        return True

    def _log_finds(self, info: dict | None) -> None:
        st = self.env.state
        for f in find_events(st, info, self._prev_found):
            who = f"robot {f['robot']}" if f["robot"] >= 0 else "team"
            self.log.append(f"t={f['t']:5.0f}s  {who} found casualty {f['id']} ({f['container']})")
            self.flashes.append([f["xy"][0], f["xy"][1], FLASH_FRAMES])
        del self.log[:-LOG_LINES]
        self._prev_found = found_mask(st).copy()

    # -- figure -------------------------------------------------------------------------------
    def build(self):
        """Create the figure and every artist once. Returns the figure."""
        if self.fig is not None:
            return self.fig
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        st = self.env.state
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi, facecolor="white")
        cols = [1.0, 1.0, 0.62] if not self.show_local else [1.0, 1.0, 0.45, 0.75]
        gs = self.fig.add_gridspec(1, len(cols), width_ratios=cols, left=0.035, right=0.995,
                                   top=0.93, bottom=0.10, wspace=0.30)  # wspace: legend room
        self.ax_gt = self.fig.add_subplot(gs[0, 0])
        self.ax_bel = self.fig.add_subplot(gs[0, 1])
        self.ax_local = self.fig.add_subplot(gs[0, 2]) if self.show_local else None
        sub = gs[0, len(cols) - 1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.22)
        self.ax_txt = self.fig.add_subplot(sub[0, 0])
        self.ax_spark = self.fig.add_subplot(sub[1, 0])
        self._build_gt(st)
        self._build_belief(st)
        self._build_status(st)
        self._build_toolbar(Button)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.suptitle(self._suptitle(), fontsize=10)
        self.refresh()
        return self.fig

    def _suptitle(self) -> str:
        m = self.env.state.scene.meta
        return (f"{m.preset} seed={m.seed}  {len(self.env.state.robots)} robots  "
                f"policy={getattr(self.policy, 'name', type(self.policy).__name__)}   {self.KEYS}")

    def _extent(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = region_extent(self.env.state.scene)
        return x0, x1, y0, y1

    def _build_gt(self, st: S.EnvState) -> None:
        ax = self.ax_gt
        ext = self._extent()
        img, origin = gt_background(st.scene, st.raster, self.gt_mode)
        ax.imshow(img, extent=ext, origin=origin, interpolation="nearest", zorder=0)
        hp = human_points(st)
        b = hp["bystander"]
        ax.scatter(b[:, 0], b[:, 1], s=26, marker="o", c=P.human_color("bystander"),
                   edgecolors="#ffffff", linewidths=0.5, zorder=7)
        self.cas = {
            "unfound": ax.scatter([], [], s=48, marker="o", c=P.human_color("casualty"),
                                  edgecolors=P.HUMAN_STATUS_EDGE["unfound"], linewidths=1.2,
                                  zorder=14),
            "found": ax.scatter([], [], s=72, marker="o", c=P.human_color("casualty"),
                                edgecolors=P.HUMAN_STATUS_EDGE["found"], linewidths=1.8,
                                zorder=15),
        }
        self.flash_gt = ax.scatter([], [], s=[], marker="o", facecolors="none",
                                   edgecolors=P.FLASH_COLOR, linewidths=2.0, zorder=16)
        cols = [P.robot_color(r.idx) for r in st.robots]
        self.traj = [ax.plot([], [], color=c, lw=1.0, alpha=P.TRAJ_ALPHA, zorder=10)[0]
                     for c in cols]
        sensor = st.cfg.sensor
        self.fov_far, self.fov_near = [], []
        for c in cols:
            w_far = Wedge((0.0, 0.0), float(sensor.visual_range_m), 0.0, 0.0, facecolor=c,
                          alpha=P.FOV_FAR_ALPHA, edgecolor="none", zorder=8)
            w_near = Wedge((0.0, 0.0), float(sensor.depth_limit_m), 0.0, 0.0, facecolor=c,
                           alpha=P.FOV_ALPHA, edgecolor=c, lw=0.5, zorder=9)
            ax.add_patch(w_far)
            ax.add_patch(w_near)
            self.fov_far.append(w_far)
            self.fov_near.append(w_near)
        self.tgt_lc = LineCollection([], colors=cols, linewidths=0.8, linestyles="--", alpha=0.85,
                                     zorder=10)
        ax.add_collection(self.tgt_lc)
        nan = np.full((len(st.robots), 2), np.nan)
        self.tgt_marks = ax.scatter(nan[:, 0], nan[:, 1], s=34, marker="x", c=cols, linewidths=1.2,
                                    zorder=11)
        self.robot_dots = ax.scatter(nan[:, 0], nan[:, 1], s=62, marker="o", c=cols,
                                     edgecolors="#000000", linewidths=0.7, zorder=12)
        self.arrow_len = 0.03 * max(ext[1] - ext[0], ext[3] - ext[2])
        self.heading = ax.quiver(nan[:, 0], nan[:, 1], np.zeros(len(cols)), np.zeros(len(cols)),
                                 color=cols, angles="xy", scale_units="xy", scale=1.0, width=0.005,
                                 zorder=12)
        self.robot_tags = [ax.text(0.0, 0.0, str(r.idx), fontsize=6, color="#ffffff", ha="center",
                                   va="center", zorder=13) for r in st.robots]
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        legend_outside(ax, truth_legend_handles(), side="right", ncols=1, fontsize=6,
                       handlelength=1.0)

    def _build_belief(self, st: S.EnvState) -> None:
        ax = self.ax_bel
        ny, nx = np.asarray(st.observed).shape
        self.stride = max(1, int(math.ceil(max(ny, nx) / max(64, self.heatmap_max_px))))
        cell = float(getattr(st.raster, "cell_m", st.cfg.raster.cell_m))
        x0, x1, y0, y1 = self._extent()
        s = self.stride
        self.bel_extent = (x0, x0 + (nx // s) * s * cell, y0, y0 + (ny // s) * s * cell) \
            if s > 1 else (x0, x1, y0, y1)
        view = self.view()
        qv = query_view(st, self.query, view)
        self.im_bel = ax.imshow(self.belief_rgb(qv, view), extent=self.bel_extent, origin="lower",
                                interpolation="nearest", zorder=0)
        self.im_seg = ax.imshow(np.zeros(self.im_bel.get_array().shape[:2] + (4,), np.float32),
                                extent=self.bel_extent, origin="lower", interpolation="nearest",
                                zorder=1)
        self.seg_sc = ax.scatter([], [], s=26, marker=P.token_marker("segment"), facecolors="none",
                                 edgecolors=P.SEGMENT_COLOR, linewidths=0.7, zorder=8)
        self.visited_sc = ax.scatter([], [], s=44, marker=P.token_marker("visited"),
                                     facecolors="none", edgecolors=P.VISITED_COLOR, linewidths=1.0,
                                     zorder=11)
        self.front_cells = ax.scatter([], [], s=2.0, marker="s", c=P.FRONTIER_COLOR, alpha=0.5,
                                      linewidths=0, zorder=6)
        self.front_sc = ax.scatter([], [], s=[], marker="o", facecolors="none",
                                   edgecolors=P.FRONTIER_COLOR, linewidths=0.8, zorder=8)
        self.ray_lc = LineCollection([], linewidths=0.9, alpha=0.8, zorder=7)
        ax.add_collection(self.ray_lc)
        self.peer_lc = LineCollection([], linewidths=1.0, zorder=11)
        ax.add_collection(self.peer_lc)
        self.peer_sc = ax.scatter([], [], s=26, marker="o", facecolors="none",
                                  edgecolors=P.PEER_COLOR, linewidths=0.9, zorder=11)
        self.link_lc = LineCollection([], linewidths=0.9, linestyles=":", colors=P.LINK_COLOR,
                                      alpha=0.8, zorder=10)
        ax.add_collection(self.link_lc)
        cols = [P.robot_color(r.idx) for r in st.robots]
        nan = np.full((len(st.robots), 2), np.nan)
        self.bel_robots = ax.scatter(nan[:, 0], nan[:, 1], s=48, marker="o", c=cols,
                                     edgecolors="#000000", linewidths=0.6, zorder=13)
        self.flash_bel = ax.scatter([], [], s=[], marker="o", facecolors="none",
                                    edgecolors=P.FLASH_COLOR, linewidths=2.0, zorder=16)
        self.tok_sc = {n: ax.scatter([], [], s=70, marker=P.token_marker(n), facecolors="none",
                                     edgecolors=P.token_color(n), linewidths=1.1, zorder=15)
                       for n in token_type_names()}
        self.tok_chosen = ax.scatter([], [], s=210, marker="o", facecolors="none",
                                     edgecolors=P.TOKEN_CHOSEN_EDGE, linewidths=1.6, zorder=15)
        k = int(st.last_obs.token_mask.shape[1]) if st.last_obs is not None else 0
        self.tok_texts = [ax.text(0.0, 0.0, "", fontsize=6, ha="center", va="center", zorder=17,
                                  visible=False) for _ in range(k)]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        legend_outside(ax, belief_legend_handles(peers=True), side="right", ncols=1, fontsize=6)

    def _build_status(self, st: S.EnvState) -> None:
        self.ax_txt.axis("off")
        self.ax_txt.set_title("status", fontsize=9)
        self.txt = self.ax_txt.text(0.0, 1.0, "", transform=self.ax_txt.transAxes, va="top",
                                    ha="left", fontsize=7.0 if self.show_local else 8,
                                    family="monospace")
        ax = self.ax_spark
        self.spark_found, = ax.plot([], [], color=P.human_color("casualty"), lw=1.4, label="found")
        self.spark_cov, = ax.plot([], [], color=P.FRONTIER_COLOR, lw=1.0, label="coverage")
        ax.set_xlim(0.0, float(st.cfg.t_max_s))
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("t [s]", fontsize=7)
        ax.set_title("fraction found / coverage", fontsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        legend_outside(ax, [self.spark_found, self.spark_cov], side="below", ncols=2, fontsize=6)

    def _build_toolbar(self, Button) -> None:
        self.buttons = {}
        specs = (("play", "play/pause", self._on_play), ("step", "step", self._on_step),
                 ("restart", "restart", self._on_restart))
        for i, (key, label, cb) in enumerate(specs):
            ax = self.fig.add_axes((0.035 + 0.085 * i, 0.018, 0.075, 0.042))
            b = Button(ax, label, color="#e8e8ee", hovercolor="#cfd6ff")
            b.label.set_fontsize(8)
            b.on_clicked(cb)
            self.buttons[key] = b

    # -- artist updates -----------------------------------------------------------------------
    def view(self):
        """The belief on screen: the team union, or `--robot r`'s own `RobotView`."""
        return belief_view(self.env.state, self.robot)

    def belief_rgb(self, qv, view) -> np.ndarray:
        sim = qv.grid
        obs = np.asarray(view.known, bool)
        if self.stride > 1:
            sim = block_max(sim, self.stride, "max")
            obs = block_max(obs, self.stride, "any")
        return P.sim_rgb(sim, obs)

    def segment_rgba(self, view) -> np.ndarray:
        shape = self.im_bel.get_array().shape[:2]
        rgba = segment_rgba(self.env.state, view) if self.segments else None
        if rgba is None:
            return np.zeros(shape + (4,), np.float32)
        s = self.stride
        if s > 1:
            rgba = rgba[::s, ::s]
        return rgba[: shape[0], : shape[1]]

    def belief_image(self) -> tuple[np.ndarray, str, int]:
        """Kept for callers that only want the picture: takes the query view itself."""
        v = self.view()
        qv = query_view(self.env.state, self.query, v)
        return self.belief_rgb(qv, v), qv.name, qv.idx

    def refresh(self) -> None:
        """Pull every artist's data from the current `env.state`."""
        if self.fig is None:
            self.build()                       # build() ends with a refresh
            return
        t0 = time.perf_counter()
        st = self.env.state
        self._refresh_flashes()
        self._refresh_gt(st)
        counts = self._refresh_belief(st)
        self._refresh_status(st, counts)
        self.draw_times.append(time.perf_counter() - t0)

    def _refresh_flashes(self) -> None:
        for f in self.flashes:
            f[2] -= 1
        self.flashes = [f for f in self.flashes if f[2] > 0]
        xy = np.array([[f[0], f[1]] for f in self.flashes], float) if self.flashes \
            else np.zeros((0, 2))
        sizes = np.array([120.0 + 240.0 * (1.0 - f[2] / FLASH_FRAMES) for f in self.flashes]) \
            if self.flashes else np.zeros(0)
        for sc in (self.flash_gt, self.flash_bel):
            sc.set_offsets(xy)
            sc.set_sizes(sizes)

    def _refresh_gt(self, st: S.EnvState) -> None:
        hp = human_points(st)
        for status, sc in self.cas.items():
            sc.set_offsets(hp[status])
        pos = np.array([r.pos for r in st.robots], float)
        self.robot_dots.set_offsets(pos)
        self.heading.set_offsets(pos)
        self.heading.set_UVC(self.arrow_len * np.cos([r.heading for r in st.robots]),
                             self.arrow_len * np.sin([r.heading for r in st.robots]))
        segs, tgts = [], []
        sensor = st.cfg.sensor
        half = 0.5 * float(sensor.hfov_deg)
        for i, r in enumerate(st.robots):
            traj = np.asarray(r.trajectory, float) if len(r.trajectory) > 1 else pos[i:i + 1]
            self.traj[i].set_data(traj[:, 0], traj[:, 1])
            th = math.degrees(r.heading)
            if getattr(sensor, "mode", "cone") == "disk":
                th, half = 0.0, 180.0
            for w in (self.fov_far[i], self.fov_near[i]):
                w.set_center(tuple(r.pos))
                w.set_theta1(th - half)
                w.set_theta2(th + half)
            t = np.asarray(r.target_xy, float) if r.target_xy is not None else None
            if t is not None and np.all(np.isfinite(t)):
                segs.append([(r.pos[0], r.pos[1]), (t[0], t[1])])
                tgts.append(t)
            else:
                tgts.append((np.nan, np.nan))
            self.robot_tags[i].set_position((r.pos[0], r.pos[1]))
        self.tgt_lc.set_segments(segs)
        self.tgt_marks.set_offsets(np.asarray(tgts, float))
        self.ax_gt.set_title(f"ground truth   t={st.t:.0f}s   decision {st.decision_idx}",
                             fontsize=9)

    def _refresh_belief(self, st: S.EnvState) -> dict[str, int]:
        view = self.view()
        qv = query_view(st, self.query, view)            # one query view per refresh, not per artist
        self.im_bel.set_data(self.belief_rgb(qv, view))
        self.im_seg.set_data(self.segment_rgba(view))
        segs_list = list(view.segments or [])
        self.seg_sc.set_offsets(np.array([np.asarray(s.xy, float) for s in segs_list], float)
                                if segs_list else np.zeros((0, 2)))
        self.front_cells.set_offsets(frontier_cells(st, view=view))
        xy, sizes = frontier_points(st, view)
        self.front_sc.set_offsets(xy)
        self.front_sc.set_sizes(sizes)
        rsegs, cols, n_rays = ray_geometry(st, qv, view)
        self.ray_lc.set_segments(list(rsegs))
        self.ray_lc.set_color(cols)
        n_peers = self._refresh_peers(st, view)
        n_links = self._refresh_links(st)
        n_vis = self._refresh_visited(view)
        self.bel_robots.set_offsets(np.array([r.pos for r in st.robots], float))
        n_tok = self._refresh_tokens(st)
        head = "belief (team)" if self.robot is None else f"robot {self.robot}'s map"
        self.ax_bel.set_title(f"{head}  q[{qv.tag}{qv.name!r}]   robot {self.focus} tokens",
                              fontsize=9)
        if self.ax_local is not None:
            self.ax_local.clear()
            draw_local_panel(self.ax_local, st, self.focus)
        return {"rays": n_rays, "frontiers": int(xy.shape[0]), "tokens": n_tok,
                "segments": len(segs_list), "peers": n_peers, "links": n_links, "visited": n_vis}

    def _refresh_peers(self, st: S.EnvState, view) -> int:
        """Peer tokens as arrows: last-known position -> reported target, alpha from contact age."""
        arrows = peer_arrows(st, self.robot) if self.robot is not None else []
        if not arrows:
            self.peer_lc.set_segments([])
            self.peer_sc.set_offsets(np.zeros((0, 2)))
            return 0
        from matplotlib.colors import to_rgba
        segs = [[tuple(p), tuple(t)] for p, t, _a, _i in arrows]
        cols = [to_rgba(P.PEER_COLOR, a) for _p, _t, a, _i in arrows]
        self.peer_lc.set_segments(segs)
        self.peer_lc.set_color(cols)
        self.peer_sc.set_offsets(np.array([p for p, _t, _a, _i in arrows], float))
        return len(arrows)

    def _refresh_links(self, st: S.EnvState) -> int:
        pairs = comms_links(st)
        self.link_lc.set_segments([[tuple(np.asarray(st.robots[i].pos, float)),
                                    tuple(np.asarray(st.robots[j].pos, float))]
                                   for i, j in pairs])
        return len(pairs)

    def _refresh_visited(self, view) -> int:
        xy = []
        for v in list(view.visited or []):
            p = getattr(v, "xy", None)
            if p is None and isinstance(v, dict):
                p = v.get("xy")
            if p is not None and np.all(np.isfinite(np.asarray(p, float))):
                xy.append(np.asarray(p, float)[:2])
        self.visited_sc.set_offsets(np.asarray(xy, float) if xy else np.zeros((0, 2)))
        return len(xy)

    def _refresh_tokens(self, st: S.EnvState) -> int:
        obs = self.decision_obs
        r = int(self.focus)
        by_type: dict[str, list] = {n: [] for n in S.TOKEN_TYPE_NAMES}
        chosen = np.zeros((0, 2))
        shown: set[int] = set()
        if obs is not None and r < obs.token_mask.shape[0]:
            a_ = self.decision_actions
            act = (int(a_[r]) if a_ is not None and np.ndim(a_) == 1 and r < len(a_) else None)
            for k in np.flatnonzero(obs.token_mask[r]):
                xy = np.asarray(obs.token_xy[r, k], float)
                if not np.all(np.isfinite(xy)):
                    continue
                name = P.token_name(int(obs.token_type[r, k]))
                by_type.setdefault(name, []).append(xy)
                shown.add(int(k))
                if act is not None and int(k) == act:
                    chosen = xy[None, :]
                if int(k) < len(self.tok_texts):
                    t = self.tok_texts[int(k)]
                    t.set_position((xy[0], xy[1]))
                    t.set_text(str(int(k)))
                    t.set_color(P.token_color(name))
                    t.set_visible(True)
        for k, t in enumerate(self.tok_texts):
            if k not in shown:
                t.set_visible(False)
        n = 0
        for name, sc in self.tok_sc.items():
            pts = by_type.get(name) or []
            sc.set_offsets(np.asarray(pts, float) if pts else np.zeros((0, 2)))
            n += len(pts)
        self.tok_chosen.set_offsets(chosen)
        return n

    def _refresh_status(self, st: S.EnvState, counts: dict[str, int]) -> None:
        self.txt.set_text("\n".join(self.status_lines(st, counts)))
        self.spark_found.set_data(self.hist_t, self.hist_found)
        self.spark_cov.set_data(self.hist_t, self.hist_cov)
        self.ax_spark.set_xlim(0.0, max(float(st.cfg.t_max_s), max(self.hist_t or [1.0])))

    def _on_draw(self, _event=None) -> None:
        now = time.perf_counter()
        if self._last_draw:
            self.frame_times.append(now - self._last_draw)
            del self.frame_times[:-40]
        self._last_draw = now

    def work_fps(self) -> float:
        """Frames per second the sim + artist update + canvas draw allow (no idle waiting)."""
        per = sum(sum(a[-20:]) / max(1, len(a[-20:]))
                  for a in (self.step_times, self.draw_times, self.render_times) if a)
        return 1.0 / per if per > 1e-9 else 0.0

    def display_fps(self) -> float:
        """Achieved rate between canvas draws (capped by `speed`), else the work rate."""
        f = self.frame_times[-20:]
        if len(f) >= 3:
            return 1.0 / max(1e-9, float(np.median(f)))
        return self.work_fps()

    def query_label(self, st: S.EnvState) -> tuple[int, str]:
        """(index or -1, name) of the query on screen; keys 0-9 index `state.query_names()`."""
        names = query_names(st)
        if isinstance(self.query, str):
            return (names.index(self.query) if self.query in names else -1), self.query
        i = int(np.clip(int(self.query), 0, max(0, len(names) - 1)))
        return i, (names[i] if names else "-")

    def status_lines(self, st: S.EnvState, counts: dict[str, int]) -> list[str]:
        names = query_names(st)
        qi, qname = self.query_label(st)
        state = "done" if self.done else ("playing" if self.playing else "paused")
        lines = [
            f"t          {st.t:7.1f} s / {st.cfg.t_max_s:.0f}",
            f"decision   {st.decision_idx}" + (f" / {self.max_decisions}"
                                               if self.max_decisions else ""),
            f"found      {self.n_found} / {st.n_casualties} casualties",
            f"coverage   {100.0 * st.coverage:5.1f} %",
            f"reward     {st.cum_reward:+.3f}",
            "",
            f"live rays  {counts['rays']}   frontiers {counts['frontiers']}",
            f"segments   {counts.get('segments', 0)}   visited {counts.get('visited', 0)}",
            f"peers      {counts.get('peers', 0)}   links {counts.get('links', 0)}",
            f"tokens     {counts['tokens']} of robot {self.focus}",
            f"map        {'team union' if self.robot is None else f'robot {self.robot}'}",
            "",
            f"[{state}]  speed {self.speed:.2g}/s   display {self.display_fps():.1f}/s "
            f"(max {self.work_fps():.1f}/s)",
            f"query {qi if qi >= 0 else 'derived'} {qname!r}   (0-9 of {len(names)})",
        ]
        lines += [f"  {'*' if i == qi else ' '}{i} {q}" for i, q in enumerate(names)]
        lines += [
            "",
            "robot  target          dist",
        ]
        for r in st.robots:
            star = "*" if r.idx == self.focus else " "
            if r.target_xy is not None:
                d = float(np.hypot(r.target_xy[0] - r.pos[0], r.target_xy[1] - r.pos[1]))
                dist = f"{d:6.0f} m"
            else:
                dist = "     - "
            lines.append(f"{star}{r.idx:<5d} {robot_target_label(st, r):<15.15s} {dist}")
        lines += ["", "finds"] + ([f"  {s}" for s in self.log] or ["  (none yet)"])
        return lines

    # -- interaction --------------------------------------------------------------------------
    def _on_play(self, _event=None) -> None:
        self.playing = not self.playing

    def _on_step(self, _event=None) -> None:
        self.playing = False
        self.pending += 1

    def _on_restart(self, _event=None) -> None:
        self.reset()
        self.refresh()

    def set_speed(self, speed: float) -> float:
        self.speed = float(np.clip(speed, SPEED_MIN, SPEED_MAX))
        if self.anim is not None:
            ms = 1000.0 / self.speed
            self.anim._interval = ms          # TimedAnimation._step resets the timer to this
            es = getattr(self.anim, "event_source", None)
            if es is not None:
                es.interval = ms
        return self.speed

    def on_key(self, event) -> None:
        k = getattr(event, "key", None)
        if k is None:
            return
        st = self.env.state
        if k == " ":
            self._on_play()
        elif k == "n":
            self._on_step()
        elif k in ("+", "="):
            self.set_speed(self.speed * 1.5)
        elif k == "-":
            self.set_speed(self.speed / 1.5)
        elif k == "f":
            self.focus = (self.focus + 1) % max(1, len(st.robots))
        elif k == "v":
            n = max(1, len(st.robots))          # team union -> robot 0 .. n-1 -> team union
            self.robot = 0 if self.robot is None else (None if self.robot + 1 >= n
                                                       else self.robot + 1)
        elif k == "s":
            self.segments = not self.segments
        elif k == "r":
            self.reset()
        elif k == "R":
            self.reset(self.seed + 1)
        elif k == "w":
            print(f"[live_viewer] {self.save_png()}")
            return
        elif k == "q":
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            return
        elif k.isdigit():
            self.query = min(int(k), max(0, len(query_names(st)) - 1))   # 1-8 live queries
        else:
            return
        self.refresh()
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    # -- output -------------------------------------------------------------------------------
    def frame_rgb(self) -> np.ndarray:
        self.build()
        t0 = time.perf_counter()
        self.fig.canvas.draw()
        self.render_times.append(time.perf_counter() - t0)
        return np.asarray(self.fig.canvas.buffer_rgba(), np.uint8)[..., :3].copy()

    def save_png(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.png)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.build().savefig(p, dpi=self.dpi)
        return p

    def update(self, _frame=None):
        """Animation callback: step the env when playing, then update every artist in place."""
        if self.playing and not self.done:
            self.advance()
        elif self.pending > 0:
            self.pending -= 1
            self.advance()
        self.refresh()
        if self.writer is not None:
            self.writer.append_data(self._mp4_frame())
        return ()

    def _mp4_frame(self) -> np.ndarray:
        from rlplanner.viz.recorder import _pad_to_macro_block

        return _pad_to_macro_block(self.frame_rgb())

    def record(self, path: str | Path, fps: float | None = None,
               max_decisions: int | None = None, progress: bool = False) -> tuple[Path, int]:
        """Headless: step the episode and write one frame per decision through the same artists."""
        import imageio.v2 as iio

        self.build()
        limit = max_decisions if max_decisions is not None else self.max_decisions
        if limit is None:
            limit = int(math.ceil(float(self.env.cfg.t_max_s) / float(self.env.cfg.decision_dt)))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rng = range(int(limit))
        if progress:
            from tqdm import tqdm
            rng = tqdm(rng, desc="decisions")
        n = 0
        t_start = time.perf_counter()
        try:
            w = iio.get_writer(p, fps=float(fps or self.speed), codec="libx264", quality=8,
                               macro_block_size=None)
        except Exception as exc:                     # noqa: BLE001 - usually a missing ffmpeg
            raise RuntimeError(f"could not open {p} for writing: {exc}. "
                               f"Is imageio-ffmpeg installed?") from exc
        with w:
            self.refresh()
            w.append_data(self._mp4_frame())
            n += 1
            for _ in rng:
                if not self.advance():
                    break
                self.refresh()
                w.append_data(self._mp4_frame())
                n += 1
        self.record_rate = (n - 1) / max(1e-9, time.perf_counter() - t_start)
        return p, n

    def start(self, record: str | Path | None = None, fps: float | None = None):
        """Open the window (and optionally record it). Returns the FuncAnimation."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self.build()
        if record is not None:
            import imageio.v2 as iio
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            self.writer = iio.get_writer(record, fps=float(fps or self.speed), codec="libx264",
                                         quality=8, macro_block_size=None)
        self.anim = FuncAnimation(self.fig, self.update, interval=1000.0 / self.speed,
                                  blit=False, cache_frame_data=False)
        try:
            plt.show()
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        return self.anim


__all__ = ["LiveViewer", "gt_background", "bake_scene_rgb", "raster_rgb", "frontier_points",
           "frontier_cells", "human_points", "find_events", "block_max", "local_crop"]
