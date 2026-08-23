#!/usr/bin/env python
"""Interactive four-panel inspector for the sensor cone, the per-cell embeddings and a city layout.

    # headless still frame (the scene may be positional or --scene)
    uv run python scripts/sensor_inspector.py data/scenes_v2/downtown_tornado_17.json \
        --out runs/inspector_v2_tornado17.png
    # belief building up along a straight flight
    uv run python scripts/sensor_inspector.py --scene ... --sweep 40 --gif runs/sweep.gif
    # interactive (a window whenever a display is available)
    uv run python scripts/sensor_inspector.py --synthetic 0

Panels: (1) ground-truth classes with the actual sensor footprint (observed bright and outlined,
far-visible hatched, in-wedge-but-occluded grey) and the semantic rays; (2) the POV frame, a
pinhole projection of every visible cell's top point, each a disc coloured by its class with a
radius equal to its angular size -- the "pixels" that carry embeddings; (3) the belief's per-cell
features projected on their top 3 principal components as RGB, plus the similarity bars of the
probed cell; (4) the belief similarity map for the selected query.

Panel 3's bars are `RayFrontsSim.query_sim` at the probed cell for the live mission list plus the
free-text `--extra-query`; panel 1's rays are coloured by `ray_query_sim` for the query on screen.

Keys: left-click move | arrows yaw +-15 deg / move 10 m | +/- altitude | [ ] pitch |
      s sense one sub-step | a auto-fly | r reset belief | 0-9 query | w write png | q quit
"""
from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from matplotlib.patches import FancyArrowPatch

from rlplanner.scene import schema
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.geometry import frustum_params
from rlplanner.sim.raster import rasterize
from rlplanner.sim.rayfronts_sim import RayFrontsSim
from rlplanner.sim.sensor import visible_cells
from rlplanner.sim.state import RobotState
from rlplanner.viz import display
from rlplanner.viz import palette as P

DEFAULT_OUT = "runs/inspector.png"
YAW_STEP = math.radians(15.0)
MOVE_STEP = 10.0
ALT_STEP = 5.0
PITCH_STEP = 5.0
OCCLUDED_RGB = np.array([0.32, 0.32, 0.35])


# ---- geometry helpers ---------------------------------------------------------------------------
def wedge_masks(raster, sensor, cam, yaw: float):
    """(observed, far, occluded) bool grids for the camera at `cam` looking along `yaw`.

    `occluded` = inside the azimuth/elevation wedge and within visual range, but no line of sight.
    """
    ny, nx = raster.shape
    obs = np.zeros((ny, nx), bool)
    far = np.zeros((ny, nx), bool)
    view = visible_cells(raster, sensor, cam, yaw, far_p=1.0)
    if view.observed_ij.shape[0]:
        obs[view.observed_ij[:, 0], view.observed_ij[:, 1]] = True
    if view.far_ij.shape[0]:
        far[view.far_ij[:, 0], view.far_ij[:, 1]] = True

    x0, y0 = raster.origin
    cell = raster.cell_m
    vis = float(sensor.visual_range_m)
    rad = int(math.ceil(vis / cell)) + 1
    ci = int(math.floor((cam[1] - y0) / cell))
    cj = int(math.floor((cam[0] - x0) / cell))
    i0, i1 = max(0, ci - rad), min(ny, ci + rad + 1)
    j0, j1 = max(0, cj - rad), min(nx, cj + rad + 1)
    wedge = np.zeros((ny, nx), bool)
    if i1 > i0 and j1 > j0:
        yy = y0 + (np.arange(i0, i1) + 0.5) * cell - cam[1]
        xx = x0 + (np.arange(j0, j1) + 0.5) * cell - cam[0]
        dz = raster.height[i0:i1, j0:j1].astype(np.float64) - cam[2]
        dx = np.broadcast_to(xx[None, :], dz.shape)
        dy = np.broadcast_to(yy[:, None], dz.shape)
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        rs = np.maximum(r, 1e-9)
        cy, sy, chh, slo, shi = frustum_params(yaw, math.radians(sensor.pitch_deg),
                                               math.radians(sensor.hfov_deg),
                                               math.radians(sensor.vfov_deg))
        s = dz / rs
        rxy = np.hypot(dx, dy)
        ok = (r <= vis) & (s >= slo) & (s <= shi)
        if sensor.mode != "disk":
            ok &= (dx * cy + dy * sy) >= rxy * chh
        wedge[i0:i1, j0:j1] = ok
    return obs, far, wedge & ~obs & ~far


def pov_image(raster, cam, yaw: float, sensor, obs_ij, far_ij, w: int, h: int) -> np.ndarray:
    """Pinhole projection of the visible cells' top points: one class-coloured disc per cell,
    radius = the cell's angular size, near cells painted over far ones."""
    img = np.full((h, w, 3), 0.06, np.float64)
    if obs_ij.shape[0] == 0 and far_ij.shape[0] == 0:
        return img
    ij = np.concatenate([obs_ij, far_ij], 0).astype(np.int64)
    dim = np.concatenate([np.zeros(obs_ij.shape[0], bool), np.ones(far_ij.shape[0], bool)])
    xs, ys = raster.ij_to_xy(ij[:, 0], ij[:, 1])
    zs = raster.height[ij[:, 0], ij[:, 1]].astype(np.float64)
    d = np.stack([xs - cam[0], ys - cam[1], zs - cam[2]], 1)

    pitch = math.radians(sensor.pitch_deg)
    fwd = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch),
                    math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.cross(right, fwd)
    z = d @ fwd
    keep = z > 1e-6
    if not keep.any():
        return img
    d, z, ij, dim = d[keep], z[keep], ij[keep], dim[keep]
    th = math.tan(0.5 * math.radians(sensor.hfov_deg))
    tv = math.tan(0.5 * math.radians(sensor.vfov_deg))
    u = (d @ right) / z / th
    v = (d @ up) / z / tv
    px = (u + 1.0) * 0.5 * w
    py = (1.0 - v) * 0.5 * h
    r = np.linalg.norm(d, axis=1)
    rad = np.maximum(0.55, 0.5 * raster.cell_m / np.maximum(r, 1e-6) * (0.5 * w) / th)
    cols = P.class_rgb_array()[raster.cls[ij[:, 0], ij[:, 1]].astype(np.int64)]
    cols = np.where(dim[:, None], cols * 0.45, cols)

    order = np.argsort(-r)                       # painter's algorithm: far first
    for k in order:
        cx, cy, rr = px[k], py[k], rad[k]
        j0, j1 = int(max(0, math.floor(cx - rr))), int(min(w, math.ceil(cx + rr) + 1))
        i0, i1 = int(max(0, math.floor(cy - rr))), int(min(h, math.ceil(cy + rr) + 1))
        if j1 <= j0 or i1 <= i0:
            continue
        jj = np.arange(j0, j1) + 0.5 - cx
        ii = np.arange(i0, i1) + 0.5 - cy
        m = (ii[:, None] ** 2 + jj[None, :] ** 2) <= rr * rr
        if m.any():
            img[i0:i1, j0:j1][m] = cols[k]
    return img


def embedding_rgb(feat_sum: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, float]:
    """Observed cells' unit features projected on their top 3 PCs, scaled to RGB."""
    rgb = np.full(observed.shape + (3,), 0.08)
    m = observed & (np.linalg.norm(feat_sum, axis=-1) > 1e-9)
    n = int(m.sum())
    if n < 4:
        return rgb, 0.0
    f = feat_sum[m]
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    f = f - f.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(f, full_matrices=False)
    p = f @ vt[:3].T
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    rgb[m] = np.clip((p - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    evr = float((s[:3] ** 2).sum() / max((s ** 2).sum(), 1e-12))
    return rgb, evr


def _arrow(ax, a, b, color, lw: float, zorder: int) -> None:
    """A clipped arrow in data coordinates (annotate() would spill outside a zoomed axes)."""
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                        shrinkA=0, shrinkB=0, zorder=zorder)
    p.set_clip_on(True)
    ax.add_patch(p)
    p.set_clip_box(ax.bbox)


def pose_problems(raster, cfg: EnvConfig, x: float, y: float, alt: float) -> list[str]:
    """Why this pose is not a place a robot could sense from (empty = fine)."""
    out: list[str] = []
    x0, y0, x1, y1 = raster.region
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        out.append(f"pose ({x:.1f}, {y:.1f}) is outside the region "
                   f"({x0:.1f}, {y0:.1f}) .. ({x1:.1f}, {y1:.1f})")
    else:
        i, j = raster.xy_to_ij(x, y)
        h = float(raster.height[i, j])
        thr = alt - cfg.robot.clearance_m
        if h >= thr:
            out.append(f"pose ({x:.1f}, {y:.1f}) is inside an obstacle: cell ({i}, {j}) is "
                       f"{h:.1f} m high and the obstacle mask at alt {alt:.1f} m blocks everything "
                       f"above {thr:.1f} m (clearance {cfg.robot.clearance_m:.1f} m), so no robot "
                       f"could ever be here")
    if alt >= cfg.sensor.depth_limit_m:
        out.append(f"alt {alt:.1f} m >= sensor.depth_limit_m {cfg.sensor.depth_limit_m:.1f} m: no "
                   f"cell is within the voxel range, so nothing is ever observed")
    return out


# ---- the inspector ------------------------------------------------------------------------------
class Inspector:
    def __init__(self, scene, cfg: EnvConfig, pose, seed: int = 0, query: int | str = 0,
                 pov: tuple[int, int] = (240, 180), zoom: float = 0.0, dpi: int = 100,
                 extra_query: str | None = None):
        self.scene = scene
        self.cfg = deepcopy(cfg)
        self.raster = rasterize(scene, self.cfg.raster.cell_m)
        self.seed = int(seed)
        self.pov_wh = pov
        self.x, self.y, self.alt = float(pose[0]), float(pose[1]), float(pose[2])
        self.yaw = math.radians(float(pose[3]))          # --pose takes degrees, the sim radians
        self.cfg.sensor.pitch_deg = float(pose[4])
        self.zoom_m = float(zoom) if zoom else 0.0
        self.dpi = int(dpi)
        self.query: int | str = query
        self.extra_query = str(extra_query) if extra_query else None
        self._sims: tuple[Any, dict[str, np.ndarray]] = (None, {})
        self.probe_ij: tuple[int, int] | None = None
        self.n_sensed = 0
        self.pose_warn: list[str] = []
        self.reset_belief()
        self.fig = None

    # -- belief ---------------------------------------------------------------------------------
    def reset_belief(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.rf = RayFrontsSim(self.raster, self.cfg, self.rng)
        self.n_sensed = 0
        self._sims = (None, {})

    @property
    def cam(self) -> np.ndarray:
        return np.array([self.x, self.y, self.alt], np.float64)

    def _robot(self) -> RobotState:
        return RobotState(idx=0, pos=np.array([self.x, self.y], np.float64), alt=self.alt,
                          heading=self.yaw, target_xy=None, target_token_type=0, target_id=-1)

    def sense(self, n: int = 1) -> None:
        for _ in range(n):
            self.rf.update([self._robot()], float(self.n_sensed) * self.cfg.dt_sim, self.rng)
            self.n_sensed += 1
        self._sims = (None, {})

    # -- query views ----------------------------------------------------------------------------
    def queries(self) -> list[str]:
        """The live mission list plus the free-text `--extra-query`, in that order."""
        q = list(self.rf.queries)
        if self.extra_query and self.extra_query not in q:
            q.append(self.extra_query)
        return q

    def query_grids(self) -> dict[str, np.ndarray]:
        """`{query name: [ny, nx] similarity}` from the lazy `RayFrontsSim.query_sim`.

        Taken once per belief version and cached: a mouse move re-reads the same grids instead of
        rescanning the map, and nothing on the per-decision path ever asks for one.
        """
        key = (self.n_sensed, tuple(self.queries()))
        if self._sims[0] == key:
            return self._sims[1]
        out = {q: np.asarray(self.rf.query_sim(q), np.float64) for q in self.queries()}
        self._sims = (key, out)
        return out

    def query_name(self) -> str:
        q = self.queries()
        if isinstance(self.query, str):
            return self.query if self.query in q else (q[0] if q else "")
        return q[int(np.clip(int(self.query), 0, max(0, len(q) - 1)))] if q else ""

    def ray_sims(self) -> np.ndarray:
        """Per-ray cosine against the shown query — `ray_query_sim`, never a stored column."""
        st = self.rf.store()
        if st.n == 0:
            return np.zeros(0)
        return np.asarray(self.rf.ray_query_sim(self.query_name()), np.float64)

    def aim_xy(self) -> tuple[float, float]:
        """Where the boresight meets the ground: the middle of the footprint."""
        pitch = math.radians(self.cfg.sensor.pitch_deg)
        d = self.alt / max(1e-3, -math.tan(pitch)) if pitch < -1e-3 else self.cfg.sensor.depth_limit_m
        d = min(d, float(self.cfg.sensor.depth_limit_m))
        return self.raster.clip_xy(self.x + d * math.cos(self.yaw), self.y + d * math.sin(self.yaw))

    def move(self, dx: float, dy: float) -> None:
        x0, y0, x1, y1 = self.raster.region
        self.x = float(np.clip(self.x + dx, x0 + 0.5, x1 - 0.5))
        self.y = float(np.clip(self.y + dy, y0 + 0.5, y1 - 0.5))

    def check_pose(self) -> list[str]:
        """Flag a pose the sim would refuse (inside a building, above the depth limit) instead of
        drawing an empty cone: the interactive drone can be flown anywhere."""
        self.pose_warn = pose_problems(self.raster, self.cfg, self.x, self.y, self.alt)
        return self.pose_warn

    def fly_to(self, tx: float, ty: float, steps: int, sense_each: bool = True):
        """Straight line to (tx, ty), yaw locked on the target; yields after every step."""
        dx, dy = tx - self.x, ty - self.y
        if math.hypot(dx, dy) > 1e-6:
            self.yaw = math.atan2(dy, dx)
        for k in range(steps):
            self.move(dx / steps, dy / steps)
            if sense_each:
                self.sense(1)
            yield k

    # -- panels ---------------------------------------------------------------------------------
    def _extent(self):
        x0, y0, x1, y1 = self.raster.region
        return x0, x1, y0, y1

    def draw_truth(self, ax) -> dict:
        ras = self.raster
        obs, far, occ = wedge_masks(ras, self.cfg.sensor, self.cam, self.yaw)
        full = P.shade_by_height(P.class_rgb_array()[ras.cls.astype(np.int64)], ras.height)
        rgb = 0.42 * full                                  # context outside the wedge, dimmed
        rgb[occ] = 0.35 * full[occ] + 0.65 * OCCLUDED_RGB  # in the wedge, no line of sight
        rgb[far] = 0.72 * full[far]                        # ray candidates
        ii, jj = np.mgrid[0:ras.ny, 0:ras.nx]
        stripe = ((ii + jj) % 6) < 2                       # a hatch the imshow grid can carry
        rgb[far & stripe] = np.clip(1.05 * full[far & stripe] + 0.10, 0.0, 1.0)
        rgb[obs] = full[obs]                               # voxel-observed: the only full-colour cells
        ax.imshow(rgb, origin="lower", extent=self._extent(), interpolation="nearest", zorder=0)
        ax.contour(np.asarray(obs, float), levels=[0.5], colors=["#ff2020"], linewidths=0.8,
                   extent=self._extent(), origin="lower", zorder=3)
        self._draw_rays(ax)
        self._draw_drone(ax)
        ax.set_title(f"ground truth + sensor cone   observed {int(obs.sum())} cells, "
                     f"far {int(far.sum())}, occluded {int(occ.sum())}", fontsize=8)
        self._square(ax, zoom=True)
        return {"obs": obs, "far": far, "occ": occ}

    def _draw_drone(self, ax, col: str = "#0057ff") -> None:
        d = max(8.0, 0.03 * (self._extent()[1] - self._extent()[0]))
        _arrow(ax, (self.x, self.y),
               (self.x + d * math.cos(self.yaw), self.y + d * math.sin(self.yaw)), col, 1.4, 12)
        ax.scatter([self.x], [self.y], s=55, marker="o", c=col, edgecolors="#000000",
                   linewidths=0.6, zorder=12)
        if self.probe_ij is not None:
            px, py = self.raster.ij_to_xy(*self.probe_ij)
            ax.scatter([px], [py], s=70, marker="s", facecolors="none", edgecolors="#00ff9d",
                       linewidths=1.2, zorder=13)

    def _draw_rays(self, ax) -> None:
        from matplotlib.collections import LineCollection
        st = self.rf.store()
        if st.n == 0:
            return
        live = np.flatnonzero(st.live())
        if live.size == 0:
            return
        sim = self.ray_sims()[live]
        cap = max(1e-6, float(self.cfg.rayfronts.ray_conf_cap))
        L = float(self.cfg.sensor.visual_range_m) * np.clip(st.conf[live] / cap, 0.15, 1.0)
        o = st.origin_xy[live]
        e = o + L[:, None] * np.stack([np.cos(st.az[live]), np.sin(st.az[live])], 1)
        ax.add_collection(LineCollection(np.stack([o, e], 1), colors=P.SIM_CMAP(np.clip(sim, 0, 1)),
                                         linewidths=0.9, alpha=0.85, zorder=6))
        # arrowheads on the bearings most like the query on screen: a reading aid for this panel,
        # not a decision — the belief ranks nothing.
        for m in np.argsort(-sim)[: min(20, live.size)]:
            if sim[m] > 0.0:
                _arrow(ax, tuple(o[m]), tuple(e[m]), P.SIM_CMAP(float(min(sim[m], 1.0))), 1.0, 7)

    def draw_pov(self, ax, masks: dict) -> None:
        from rlplanner.viz.frame import draw_class_legend
        ras = self.raster
        obs_ij = np.argwhere(masks["obs"]).astype(np.int32)
        far_ij = np.argwhere(masks["far"]).astype(np.int32)
        w, h = self.pov_wh
        img = pov_image(ras, self.cam, self.yaw, self.cfg.sensor, obs_ij, far_ij, w, h)
        ax.imshow(img, origin="upper", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        draw_class_legend(ax, side="below", ncols=5, fontsize=4.6)
        ax.set_title(f"POV  {w}x{h}  hfov {self.cfg.sensor.hfov_deg:.0f} x "
                     f"vfov {self.cfg.sensor.vfov_deg:.0f}  pitch {self.cfg.sensor.pitch_deg:+.0f}"
                     f"  yaw {math.degrees(self.yaw):+.0f}  alt {self.alt:.0f} m", fontsize=8)

    def draw_embedding(self, ax_img, ax_bar) -> None:
        rgb, evr = embedding_rgb(self.rf.vox_feat_sum, self.rf.observed)
        ax_img.imshow(rgb, origin="lower", extent=self._extent(), interpolation="nearest")
        self._draw_drone(ax_img, col="#ffffff")
        ax_img.set_title(f"vox_feat PC1-3 as RGB  D={self.rf.D}  {100 * evr:.0f}% var", fontsize=8)
        self._square(ax_img, zoom=True)

        ij = self.probe_ij or self.raster.xy_to_ij(*self.aim_xy())
        i = int(np.clip(ij[0], 0, self.raster.ny - 1))
        j = int(np.clip(ij[1], 0, self.raster.nx - 1))
        grids = self.query_grids()
        q = list(grids)
        vals = np.array([grids[k][i, j] for k in q], float)
        seen = bool(self.rf.observed[i, j])
        ax_bar.barh(np.arange(len(q)), vals, color=P.SIM_CMAP(np.clip(vals, 0, 1)),
                    edgecolor="#00000030", linewidth=0.3)
        ax_bar.set_yticks(np.arange(len(q)))
        ax_bar.set_yticklabels([f"{k}*" if k == self.extra_query else k for k in q], fontsize=5.5)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, 1)
        ax_bar.tick_params(axis="x", labelsize=6)
        cls = schema.CLASS_NAMES[int(self.raster.cls[i, j])]
        ax_bar.set_title(f"query_sim of cell ({i},{j})  gt={cls}\n"
                         f"n_obs={int(self.rf.vox_cnt[i, j])}"
                         f"{'' if seen else '  [unobserved]'}"
                         f"{'   (* free text)' if self.extra_query else ''}", fontsize=6.5)

    def draw_belief(self, ax) -> None:
        name = self.query_name()
        sim = self.query_grids()[name]
        ax.imshow(P.sim_rgb(sim, self.rf.observed), origin="lower", extent=self._extent(),
                  interpolation="nearest")
        self._draw_drone(ax, col="#ffffff")
        cov = float(self.rf.observed.mean())
        ax.set_title(f"belief  query_sim[{name!r}]   sub-steps {self.n_sensed}  "
                     f"coverage {100 * cov:.1f}%", fontsize=8)
        self._square(ax)

    def _square(self, ax, zoom: bool = False) -> None:
        e = self._extent()
        if zoom and self.zoom_m > 0.0:
            h = 0.5 * self.zoom_m
            ax.set_xlim(max(e[0], self.x - h), min(e[1], self.x + h))
            ax.set_ylim(max(e[2], self.y - h), min(e[3], self.y + h))
        else:
            ax.set_xlim(e[0], e[1])
            ax.set_ylim(e[2], e[3])
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)

    # -- figure ----------------------------------------------------------------------------------
    def build(self, figsize=(15.0, 10.0), dpi: int | None = None):
        import matplotlib.pyplot as plt

        if self.fig is None:
            self.fig = plt.figure(figsize=figsize, dpi=dpi or self.dpi, facecolor="white")
            gs = self.fig.add_gridspec(2, 2, hspace=0.26, wspace=0.10, left=0.04, right=0.99,
                                       top=0.93, bottom=0.05)   # hspace: POV legend room
            sub = gs[1, 0].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.45)
            self.ax_gt = self.fig.add_subplot(gs[0, 0])
            self.ax_pov = self.fig.add_subplot(gs[0, 1])
            self.ax_emb = self.fig.add_subplot(sub[0, 0])
            self.ax_bar = self.fig.add_subplot(sub[0, 1])
            self.ax_bel = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax_gt, self.ax_pov, self.ax_emb, self.ax_bar, self.ax_bel):
            ax.clear()
        masks = self.draw_truth(self.ax_gt)
        self.draw_pov(self.ax_pov, masks)
        self.draw_embedding(self.ax_emb, self.ax_bar)
        self.draw_belief(self.ax_bel)
        warn = self.check_pose()
        self.fig.suptitle(
            f"{self.scene.meta.preset} seed {self.scene.meta.seed}   "
            f"pose x={self.x:.0f} y={self.y:.0f} alt={self.alt:.0f} "
            f"yaw={math.degrees(self.yaw):+.0f} pitch={self.cfg.sensor.pitch_deg:+.0f}   "
            f"embeddings: {self.rf.emb.source} D={self.rf.D}"
            + ("\n! " + " | ".join(warn) if warn else ""),
            fontsize=10, color="#b00020" if warn else "black")
        return self.fig

    def frame(self) -> np.ndarray:
        self.build()
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba(), np.uint8)[..., :3].copy()

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.build().savefig(p)
        return p

    # -- interaction -------------------------------------------------------------------------------
    def connect(self, out: str) -> None:
        import matplotlib.pyplot as plt

        def redraw():
            self.build()
            self.fig.canvas.draw_idle()

        def on_click(ev):
            if ev.inaxes in (self.ax_gt, self.ax_emb, self.ax_bel) and ev.button == 1 \
                    and ev.xdata is not None:
                self.x, self.y = float(ev.xdata), float(ev.ydata)
                redraw()

        def on_move(ev):
            if ev.inaxes in (self.ax_gt, self.ax_emb, self.ax_bel) and ev.xdata is not None:
                i, j = self.raster.xy_to_ij(float(ev.xdata), float(ev.ydata))
                ij = (int(np.clip(i, 0, self.raster.ny - 1)), int(np.clip(j, 0, self.raster.nx - 1)))
                if ij != self.probe_ij:
                    self.probe_ij = ij
                    self.ax_bar.clear()
                    self.draw_embedding(self.ax_emb, self.ax_bar)
                    self.fig.canvas.draw_idle()

        def on_key(ev):
            k = ev.key
            if k == "q":
                plt.close(self.fig)
                return
            if k == "left":
                self.yaw += YAW_STEP
            elif k == "right":
                self.yaw -= YAW_STEP
            elif k == "up":
                self.move(MOVE_STEP * math.cos(self.yaw), MOVE_STEP * math.sin(self.yaw))
            elif k == "down":
                self.move(-MOVE_STEP * math.cos(self.yaw), -MOVE_STEP * math.sin(self.yaw))
            elif k in ("+", "="):
                self.alt += ALT_STEP
            elif k == "-":
                self.alt = max(2.0, self.alt - ALT_STEP)
            elif k == "[":
                self.cfg.sensor.pitch_deg = max(-90.0, self.cfg.sensor.pitch_deg - PITCH_STEP)
            elif k == "]":
                self.cfg.sensor.pitch_deg = min(0.0, self.cfg.sensor.pitch_deg + PITCH_STEP)
            elif k == "s":
                self.sense(1)
            elif k == "a":
                for _ in self.fly_to(self.x + 120.0 * math.cos(self.yaw),
                                     self.y + 120.0 * math.sin(self.yaw), 12):
                    pass
            elif k == "r":
                self.reset_belief()
            elif k == "w":
                print(f"[sensor_inspector] {self.save(out)}")
            elif k is not None and k.isdigit():
                self.query = min(int(k), max(0, len(self.queries()) - 1))
            else:
                return
            redraw()

        self.fig.canvas.mpl_connect("button_press_event", on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", on_move)
        self.fig.canvas.mpl_connect("key_press_event", on_key)


# ---- cli ------------------------------------------------------------------------------------------
def load_scene(a) -> schema.Scene:
    if a.scene:
        return schema.Scene.from_json(a.scene)
    return schema.make_synthetic_scene(int(a.synthetic))


def default_pose(scene, cfg: EnvConfig):
    x0, y0, x1, y1 = scene.region
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if scene.robots_spawn:
        x, y = float(scene.robots_spawn[0][0]), float(scene.robots_spawn[0][1])
    else:
        x, y = x0 + 0.25 * (x1 - x0), y0 + 0.25 * (y1 - y0)
    return [x, y, cfg.robot.flight_alt_m, math.degrees(math.atan2(cy - y, cx - x)),
            cfg.sensor.pitch_deg]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_pos", nargs="?", metavar="SCENE",
                    help="scene.json (the same as --scene)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--scene")
    src.add_argument("--synthetic", type=int, metavar="SEED")
    ap.add_argument("--pose", nargs=5, type=float, metavar=("X", "Y", "ALT", "YAW", "PITCH"),
                    default=None)
    ap.add_argument("--query", default="0", metavar="NAME|IDX",
                    help="query drawn in the belief panel: an index into the mission list "
                         "(0-9 in the window) or a name the embedding table knows")
    ap.add_argument("--extra-query", default=None, metavar="TEXT",
                    help="free text added to the similarity bars; needs an embedding table with a "
                         "text tower behind it (rayfronts.embeddings_path), the factorized hand "
                         "table cannot encode a new phrase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cell", type=float, default=None, help="override raster.cell_m")
    ap.add_argument("--sense", type=int, default=1, help="sensing sub-steps before the first draw")
    ap.add_argument("--out", default=None,
                    help=f"write a PNG and exit instead of opening a window "
                         f"(default {DEFAULT_OUT} when no window is available)")
    ap.add_argument("--sweep", type=int, default=0, metavar="N",
                    help="fly N steps toward the region centre, accumulating the belief")
    ap.add_argument("--gif", default=None, help="GIF path for --sweep (default: --out with .gif)")
    ap.add_argument("--gif-every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--pov", nargs=2, type=int, default=(240, 180), metavar=("W", "H"))
    ap.add_argument("--zoom", type=float, default=None, metavar="M",
                    help="side of the map window around the drone in panels 1 and 3 "
                         "(default 2.4x visual_range; 0 = whole region)")
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args(argv)
    if a.scene_pos:
        if a.scene or a.synthetic is not None:
            ap.error("give a scene once: either the positional SCENE or --scene/--synthetic")
        a.scene = a.scene_pos
    if not a.scene and a.synthetic is None:
        ap.error("give a scene.json (positional or --scene) or --synthetic SEED")

    display.select_backend(a.out is None and a.sweep <= 0)
    cfg = EnvConfig.from_yaml(a.config) if a.config else EnvConfig()
    if a.cell:
        cfg.raster.cell_m = float(a.cell)
    errs = cfg.validate()
    if errs:
        raise SystemExit("[sensor_inspector] EnvConfig invalid: " + "; ".join(errs))
    scene = load_scene(a)
    pose = a.pose if a.pose is not None else default_pose(scene, cfg)
    q = str(a.query).strip()
    query: int | str = int(q) if q.lstrip("-").isdigit() else q
    n_q = cfg.n_queries + (1 if a.extra_query else 0)
    if isinstance(query, int) and not (0 <= query < n_q):
        raise SystemExit(f"[sensor_inspector] --query {query} outside [0, {n_q}) "
                         f"({list(cfg.rayfronts.queries)})")

    zoom = 2.4 * cfg.sensor.visual_range_m if a.zoom is None else a.zoom
    insp = Inspector(scene, cfg, pose, seed=a.seed, query=query, pov=tuple(a.pov), zoom=zoom,
                     dpi=a.dpi, extra_query=a.extra_query)
    if a.extra_query:
        try:
            insp.rf.query_vec(a.extra_query)
        except Exception as exc:                     # noqa: BLE001 - the table says why itself
            raise SystemExit(f"[sensor_inspector] --extra-query {a.extra_query!r} cannot be "
                             f"embedded: {exc}")
    if isinstance(query, str) and query not in insp.queries():
        try:
            insp.rf.query_vec(query)
        except Exception as exc:                     # noqa: BLE001
            raise SystemExit(f"[sensor_inspector] --query {query!r} cannot be embedded: {exc}")
    bad = insp.check_pose()
    if bad and a.pose is not None:
        raise SystemExit("[sensor_inspector] --pose is not usable: " + "; ".join(bad))
    for w in bad:
        print(f"[sensor_inspector] warning: {w}")
    insp.sense(max(0, a.sense))
    print(f"[sensor_inspector] {scene.meta.preset}/{scene.meta.seed} region {scene.region} "
          f"cells {insp.raster.ny}x{insp.raster.nx} @ {cfg.raster.cell_m} m   "
          f"embeddings {insp.rf.emb.source} D={insp.rf.D}")

    if a.sweep > 0:
        import imageio.v2 as iio
        x0, y0, x1, y1 = scene.region
        gif = Path(a.gif) if a.gif else (Path(a.out).with_suffix(".gif") if a.out
                                         else Path("runs/inspector_sweep.gif"))
        gif.parent.mkdir(parents=True, exist_ok=True)
        frames = [insp.frame()]
        for k in insp.fly_to(0.5 * (x0 + x1), 0.5 * (y0 + y1), a.sweep):
            if (k + 1) % a.gif_every == 0 or k == a.sweep - 1:
                frames.append(insp.frame())
        iio.mimsave(gif, frames, duration=1000.0 / max(1, a.fps), loop=0)
        print(f"[sensor_inspector] {gif}  {len(frames)} frames, {insp.n_sensed} sub-steps, "
              f"coverage {100 * insp.rf.observed.mean():.1f}%")

    if a.out:
        print(f"[sensor_inspector] {insp.save(a.out)}")
        return 0
    if a.sweep > 0:
        return 0
    if not display.gui_active():                     # headless and no --out: write the default PNG
        print(f"[sensor_inspector] no window available; {insp.save(DEFAULT_OUT)}")
        return 0

    import matplotlib.pyplot as plt
    insp.build()
    insp.connect(DEFAULT_OUT)
    print("Keys:" + __doc__.split("Keys:")[1])
    try:
        plt.show()
    except Exception as exc:                         # noqa: BLE001 - a GUI backend that cannot open
        print(f"[sensor_inspector] {type(exc).__name__}: {exc}; writing {DEFAULT_OUT}")
        insp.save(DEFAULT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
