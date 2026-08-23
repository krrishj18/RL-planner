"""Ground-truth scene rendering (CONTRACTS.md 9)."""
from __future__ import annotations

import numpy as np
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from rlplanner.scene import schema
from rlplanner.viz import palette as P
from rlplanner.viz.layout import legend_outside

_DAMAGE_SAMPLES = 160  # per axis when the scene has no sampled damage grid


def region_extent(scene: schema.Scene) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = scene.region
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"scene region has non-positive extent: {scene.region}")
    return x0, y0, x1, y1


def _rect_patch(rect, color, **kw) -> Rectangle:
    x0, y0, x1, y1 = rect
    return Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, **kw)


def _obb_patch(center, size, yaw_deg, color, **kw) -> Rectangle:
    """Rectangle centred on `center`, extent `size` along local axes, rotated by yaw."""
    sx, sy = float(size[0]), float(size[1])
    return Rectangle((center[0] - sx / 2, center[1] - sy / 2), sx, sy, angle=float(yaw_deg),
                     rotation_point="center", facecolor=color, **kw)


def damage_grid(scene: schema.Scene, n: int = _DAMAGE_SAMPLES):
    """(X, Y, D) sampled damage field; uses `damage_field.grid` when the scene ships one."""
    x0, y0, x1, y1 = region_extent(scene)
    g = scene.damage_field.grid
    if g is not None:
        d = np.asarray(g["values"], dtype=np.float64)
        cell = float(g["cell_m"])
        xs = x0 + (np.arange(d.shape[1]) + 0.5) * cell
        ys = y0 + (np.arange(d.shape[0]) + 0.5) * cell
        X, Y = np.meshgrid(xs, ys)
        return X, Y, d
    xs = np.linspace(x0, x1, int(n))
    ys = np.linspace(y0, y1, int(n))
    X, Y = np.meshgrid(xs, ys)
    D = np.array([[scene.damage_at(float(x), float(y)) for x in xs] for y in ys], dtype=np.float64)
    return X, Y, D


def _draw_damage(ax: Axes, scene: schema.Scene) -> bool:
    X, Y, D = damage_grid(scene)
    lo, hi = float(np.nanmin(D)), float(np.nanmax(D))
    levels = [lv for lv in P.DAMAGE_CONTOUR_LEVELS if lo < lv < hi]
    if not levels:
        return False
    cs = ax.contour(X, Y, D, levels=levels, colors=P.DAMAGE_CONTOUR_COLOR,
                    linewidths=[0.8, 1.1, 1.4][: len(levels)], alpha=0.75, zorder=6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    return True


def plot_scene(scene: schema.Scene, ax: Axes | None = None, show_damage: bool = True,
               show_humans: bool = True, show_ids: bool = False, legend: bool = True,
               title: str | None = None) -> Axes:
    """Draw the ground-truth scene. Returns the axes it drew on."""
    if not isinstance(scene, schema.Scene):
        raise TypeError(f"plot_scene expects a schema.Scene, got {type(scene).__name__}")
    own_fig = ax is None
    if own_fig:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        if legend:
            fig.subplots_adjust(left=0.08, right=0.76, top=0.94, bottom=0.07)  # room for the legend
    x0, y0, x1, y1 = region_extent(scene)

    ax.add_patch(_rect_patch((x0, y0, x1, y1), P.GROUND_COLOR, edgecolor="none", zorder=0))

    for b in scene.blocks:
        if b.typology == "park":
            ax.add_patch(_rect_patch(b.rect, P.PARK_COLOR, edgecolor="none", zorder=1))
        else:
            ax.add_patch(_rect_patch(b.rect, "none", edgecolor=P.BLOCK_EDGE, linewidth=0.3,
                                     linestyle=":", zorder=1))

    for r in scene.roads:
        col = P.ROAD_COLOR if r.kind == "road" else P.SIDEWALK_COLOR
        ax.add_patch(_rect_patch(r.rect, col, edgecolor="none", zorder=2))

    for d in scene.debris:
        ax.add_patch(Circle(d.center, float(d.radius_m), facecolor=P.DEBRIS_COLOR, edgecolor="none",
                            alpha=0.85, zorder=3))

    for b in scene.buildings:
        col = P.FATE_COLORS.get(b.fate, P.FATE_COLORS["intact"])
        ax.add_patch(_obb_patch(b.center, b.size, b.yaw_deg, col, edgecolor="#4a4a4a",
                                linewidth=0.4, zorder=4))
        if show_ids:
            ax.annotate(b.id, b.center, fontsize=4, ha="center", va="center", color="#202020",
                        zorder=9)

    for v in scene.vehicles:
        col = P.VEHICLE_COLORS.get(v.state, P.VEHICLE_COLORS["intact"])
        ax.add_patch(_obb_patch(v.center, v.size, v.yaw_deg, col, edgecolor="#101010",
                                linewidth=0.3, zorder=5))

    props: dict[str, list[tuple[float, float]]] = {}
    for p in scene.props:
        key = p.category if p.category in ("bus_stop", "tree") else "other"
        props.setdefault(key, []).append(tuple(p.center))
    marker = {"bus_stop": "H", "tree": "*", "other": "."}
    msize = {"bus_stop": 22, "tree": 34, "other": 10}
    for key, pts in props.items():
        a = np.asarray(pts, dtype=float)
        ax.scatter(a[:, 0], a[:, 1], s=msize[key], marker=marker[key], c=P.prop_color(key),
                   edgecolors="none", zorder=5)

    if show_damage:
        _draw_damage(ax, scene)

    n_cas = sum(1 for h in scene.humans if h.role == "casualty")
    if show_humans:
        by_key: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for h in scene.humans:
            by_key.setdefault((h.role, h.container), []).append((h.pos[0], h.pos[1]))
            if show_ids:
                ax.annotate(h.id, (h.pos[0], h.pos[1]), fontsize=4, ha="left", va="bottom",
                            color=P.human_color(h.role), zorder=9)
        halo = [pe.withStroke(linewidth=2.0, foreground="#ffffff")]
        for (role, container), pts in by_key.items():
            a = np.asarray(pts, dtype=float)
            mk = P.human_marker(container)
            kw = dict(linewidths=1.4, path_effects=halo) if mk == "x" else \
                dict(edgecolors="#ffffff", linewidths=0.8)
            ax.scatter(a[:, 0], a[:, 1], s=44, marker=mk, c=P.human_color(role), zorder=8, **kw)

    for i, s in enumerate(scene.robots_spawn):
        ax.scatter([s[0]], [s[1]], s=45, marker="P", c=P.robot_color(i), edgecolors="#000000",
                   linewidths=0.5, zorder=8)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    if title is None:
        m = scene.meta
        title = (f"{m.preset} seed={m.seed} {m.disaster_type} sev={m.severity:.2f} | "
                 f"{len(scene.buildings)} bld  {len(scene.humans)} humans ({n_cas} casualties)")
    ax.set_title(title, fontsize=9)
    if legend:
        add_scene_legend(ax, scene, show_humans=show_humans)
    return ax


def add_scene_legend(ax: Axes, scene: schema.Scene | None = None, show_humans: bool = True,
                     side: str = "right", ncols: int = 2) -> None:
    """Compact legend of the entity classes present (all of them if scene is None), outside the axes."""
    have = lambda pred: scene is None or pred()
    handles: list = []
    if have(lambda: bool(scene.roads)):
        handles += [Patch(facecolor=P.ROAD_COLOR, label="road"),
                    Patch(facecolor=P.SIDEWALK_COLOR, label="sidewalk")]
    if have(lambda: any(b.typology == "park" for b in scene.blocks)):
        handles.append(Patch(facecolor=P.PARK_COLOR, label="park"))
    for fate in schema.BUILDING_FATES:
        if have(lambda f=fate: any(b.fate == f for b in scene.buildings)):
            handles.append(Patch(facecolor=P.FATE_COLORS[fate], label=f"bld {fate}"))
    if have(lambda: bool(scene.debris)):
        handles.append(Patch(facecolor=P.DEBRIS_COLOR, label="debris"))
    for st in schema.VEHICLE_STATES:
        if have(lambda s=st: any(v.state == s for v in scene.vehicles)):
            handles.append(Patch(facecolor=P.VEHICLE_COLORS[st], label=f"veh {st}"))
    if have(lambda: any(p.category == "bus_stop" for p in scene.props)):
        handles.append(Line2D([], [], ls="", marker="H", color=P.prop_color("bus_stop"),
                              markersize=5, label="bus stop"))
    if have(lambda: any(p.category == "tree" for p in scene.props)):
        handles.append(Line2D([], [], ls="", marker="*", color=P.prop_color("tree"), markersize=7,
                              label="tree"))
    if show_humans:
        for container, mk in P.HUMAN_MARKERS.items():
            if have(lambda c=container: any(h.container == c and h.role == "casualty"
                                            for h in scene.humans)):
                handles.append(Line2D([], [], ls="", marker=mk, color=P.human_color("casualty"),
                                      markersize=5, label=f"casualty/{container}"))
        if have(lambda: any(h.role == "bystander" for h in scene.humans)):
            handles.append(Line2D([], [], ls="", marker="o", color=P.human_color("bystander"),
                                  markersize=5, label="bystander"))
    legend_outside(ax, handles, side=side, ncols=ncols, fontsize=5.5)


__all__ = ["plot_scene", "add_scene_legend", "damage_grid", "region_extent"]
