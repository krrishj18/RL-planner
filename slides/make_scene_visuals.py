"""Slide visuals: top-down scene map, 2.5D heightfield render, architecture diagram."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from rlplanner.scene import schema
from rlplanner.sim.raster import rasterize
from rlplanner.viz.palette import CLASS_COLORS

CAS = schema.HUMAN_ROLES.index("casualty")


def pick_scene():
    best = None
    for f in sorted(glob.glob("data/scenes_slides/*.json")):
        sc = schema.Scene.from_json(f)
        w, h = float(sc.region[2] - sc.region[0]), float(sc.region[3] - sc.region[1])
        area, aspect = w * h / 1e6, max(w, h) / min(w, h)
        fates = [b.fate for b in sc.buildings]
        dmg = sum(1 for x in fates if x != "intact") / max(1, len(fates))
        score = 2 * (0.12 < dmg < 0.45) + (0.55 < area < 1.3) + (1.15 < aspect < 1.8)
        if best is None or score > best[0]:
            best = (score, f, sc, w, h, dmg)
        if score == 4:
            break
    _, f, sc, w, h, dmg = best
    print(f"scene: {f}  {w:.0f}x{h:.0f} m  damaged_frac={dmg:.2f}")
    return f, sc, w, h


def topdown(sc, w, h, out):
    r = rasterize(sc, 2.0)
    cmap = ListedColormap([CLASS_COLORS[n] for n in schema.CLASS_NAMES])
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(r.cls, cmap=cmap, vmin=0, vmax=len(schema.CLASS_NAMES) - 1,
              origin="lower", extent=r.region[::2] + r.region[1::2], interpolation="nearest")
    hs = r.humans
    cas = hs["role_id"] == CAS
    ax.scatter(hs["x"][cas], hs["y"][cas], s=26, c="#d40000", edgecolors="white",
               linewidths=0.7, zorder=5, label="casualty")
    ax.scatter(hs["x"][~cas], hs["y"][~cas], s=16, c="#ff2d95", edgecolors="white",
               linewidths=0.5, zorder=5, label="bystander")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Urban – Earthquake – {w:.0f} × {h:.0f} m", fontsize=14)
    handles = [Patch(facecolor=CLASS_COLORS[n], edgecolor="#00000030", label=n)
               for n in schema.CLASS_NAMES if n not in ("human_standing", "human_prone")]
    handles += [plt.Line2D([], [], ls="", marker="o", color="#d40000",
                           markeredgecolor="white", label="casualty"),
                plt.Line2D([], [], ls="", marker="o", color="#ff2d95",
                           markeredgecolor="white", label="bystander")]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc="upper left",
              fontsize=9, frameon=False, title="semantic class", title_fontsize=10)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _surface(ax, r, i0, i1, j0, j1, stride, w0, h0):
    sub_h = r.height[i0:i1, j0:j1]
    sub_c = r.cls[i0:i1, j0:j1]
    s = stride
    ny, nx = (sub_h.shape[0] // s) * s, (sub_h.shape[1] // s) * s
    hb = sub_h[:ny, :nx].reshape(ny // s, s, nx // s, s)
    cb = sub_c[:ny, :nx].reshape(ny // s, s, nx // s, s)
    flat = hb.reshape(ny // s, nx // s, -1)
    k = flat.argmax(2)
    H = np.take_along_axis(flat, k[..., None], 2)[..., 0]
    C = np.take_along_axis(cb.reshape(ny // s, nx // s, -1), k[..., None], 2)[..., 0]
    H = np.kron(H, np.ones((2, 2)))
    C = np.kron(C, np.ones((2, 2), C.dtype))
    rgba = np.array([to_rgba(CLASS_COLORS[n]) for n in schema.CLASS_NAMES])[C]
    shade = 1.0 - 0.32 * (np.roll(H, 1, 1) > H) - 0.12 * (np.roll(H, 1, 0) > H)
    rgba[..., :3] *= np.clip(shade, 0.4, 1.0)[..., None]
    X, Y = np.meshgrid(np.linspace(0, w0, H.shape[1]), np.linspace(0, h0, H.shape[0]))
    ax.plot_surface(X, Y, H, facecolors=rgba, linewidth=0, antialiased=False,
                    rstride=1, cstride=1, shade=False)
    return H


def _style3d(ax, w0, h0, zmax, exag):
    ax.set_box_aspect((w0, h0, exag * max(w0, h0)))
    ax.view_init(elev=38, azim=-58)
    ax.set_zlim(0, max(1.0, zmax))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis._axinfo["grid"]["linewidth"] = 0.0
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zticks([])


def heightfield(sc, w, h, out):
    r = rasterize(sc, 2.0)
    hs = r.humans; cas = hs["role_id"] == CAS

    # full scene
    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    stride = max(1, int(np.ceil(max(r.ny, r.nx) / 190)))
    H = _surface(ax, r, 0, r.ny, 0, r.nx, stride, w, h)
    zi, zj = r.xy_to_ij(hs["x"][cas], hs["y"][cas])
    zc = r.height[np.clip(zi, 0, r.ny - 1), np.clip(zj, 0, r.nx - 1)] + 3
    ax.scatter(hs["x"][cas] - r.region[0], hs["y"][cas] - r.region[1], zc,
               s=13, c="#d40000", edgecolors="white", linewidths=0.4, depthshade=False)
    _style3d(ax, w, h, float(H.max()), 0.16)
    ax.set_title("2.5D heightfield — full scene", fontsize=13)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # zoom: densest 320x320 m casualty neighbourhood, full resolution
    win = int(320 / r.cell_m)
    ii = np.clip(zi, 0, r.ny - 1); jj = np.clip(zj, 0, r.nx - 1)
    dens = np.zeros((r.ny, r.nx)); dens[ii, jj] = 1.0
    cs = dens.cumsum(0).cumsum(1)
    best, bi, bj = -1.0, 0, 0
    for i in range(0, r.ny - win, 8):
        for j in range(0, r.nx - win, 8):
            n = cs[i + win, j + win] - cs[i, j + win] - cs[i + win, j] + cs[i, j]
            if n > best:
                best, bi, bj = n, i, j
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    Hz = _surface(ax, r, bi, bi + win, bj, bj + win, 1, 320, 320)
    sel = (ii >= bi) & (ii < bi + win) & (jj >= bj) & (jj < bj + win)
    zc = r.height[ii, jj][sel] + 2.5
    ax.scatter(hs["x"][cas][sel] - r.region[0] - bj * r.cell_m,
               hs["y"][cas][sel] - r.region[1] - bi * r.cell_m, zc,
               s=42, c="#d40000", edgecolors="white", linewidths=0.8, depthshade=False)
    _style3d(ax, 320, 320, float(Hz.max()), 0.22)
    ax.set_title("2.5D heightfield — 320 m detail", fontsize=13)
    fig.savefig(out.replace(".png", "_zoom.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.replace(".png", "_zoom.png"))


def _box(ax, x, y, wd, ht, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=1.6))
    ax.text(x + wd / 2, y + ht - 0.045, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#1a1a1a")
    ax.text(x + wd / 2, y + ht / 2 - 0.045, body, ha="center", va="center",
            fontsize=8.4, color="#333333", linespacing=1.35)
    return (x, y, wd, ht)


def _arrow(ax, p, q, color="#444444", lw=1.8, curve=0.0):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2,
                                mutation_scale=16, connectionstyle=f"arc3,rad={curve}"))


def architecture(out):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    Y, HT, WD = 0.40, 0.30, 0.138
    xs = [0.012 + i * 0.1665 for i in range(6)]
    _box(ax, xs[0], Y, WD, HT, "Scene generator",
         "procedural cities\nv2 layouts\ndisaster fates\ncasualty placement",
         "#eef4e4", "#6ea44c")
    _box(ax, xs[1], Y, WD, HT, "2.5D disaster sim",
         "heightfield, 2 m cells\ndrones at real altitude\n3D ray-marched LoS\nfrustum sensing",
         "#f4ecdf", "#b08a55")
    _box(ax, xs[2], Y, WD, HT, "Simulated RayFronts",
         "voxels · rays · frontiers\nopen-set embeddings\npersistent semantic map",
         "#fdeeda", "#e8912a")
    _box(ax, xs[3], Y, WD, HT, "Per-drone belief",
         "own sensing + gossip\nrange / relay comms\nrays, coverage,\nsegments, visited",
         "#ececf0", "#7a7a88")
    _box(ax, xs[4], Y, WD, HT, "Token builder",
         "frontier · ray · segment\nvisited · peer tokens\n+ query tokens",
         "#e8eef8", "#5b7fb4")
    _box(ax, xs[5], Y, WD, HT, "TokenPolicy",
         "transformer ≈ 0.4 M\nruns on every drone\n→ next target",
         "#e4f0f0", "#00a0a0")
    for i in range(5):
        _arrow(ax, (xs[i] + WD + 0.004, Y + HT / 2), (xs[i + 1] - 0.004, Y + HT / 2))
    _box(ax, 0.40, 0.80, 0.37, 0.155, "LLM query controller (Claude)",
         "disaster context + live map digest → add / remove / reweight hint queries",
         "#f6e9fb", "#a020c0")
    _arrow(ax, (xs[3] + WD * 0.35, Y + HT + 0.01), (0.455, 0.795), color="#a020c0", curve=0.22)
    ax.text(0.492, 0.685, "live semantics\ndigest", fontsize=8.2, color="#a020c0",
            ha="center", linespacing=1.2)
    _arrow(ax, (0.71, 0.795), (xs[4] + WD * 0.65, Y + HT + 0.01), color="#a020c0", curve=0.22)
    ax.text(0.775, 0.685, "query edits", fontsize=8.2, color="#a020c0", ha="center")
    _box(ax, 0.40, 0.045, 0.42, 0.155, "CTDE training",
         "DAgger from privileged oracle → MAPPO fine-tune · central critic (training only)\n"
         "reward: finds − time − redundancy − confirmed-target revisits",
         "#e8eef8", "#2f6fdb")
    _arrow(ax, (0.76, 0.205), (xs[5] + WD / 2, Y - 0.012), color="#2f6fdb", curve=0.18)
    ax.text(0.905, 0.27, "gradients", fontsize=8.2, color="#2f6fdb", ha="center")
    ax.set_title("Casualty-search planner — system architecture", fontsize=14, pad=14)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    f, sc, w, h = pick_scene()
    topdown(sc, w, h, "slides/topdown.png")
    heightfield(sc, w, h, "slides/heightfield.png")
    architecture("slides/architecture.png")


def sim_pipeline(out):
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    Y, HT, WD = 0.47, 0.34, 0.21
    xs = [0.015, 0.268, 0.521, 0.774]
    _box(ax, xs[0], Y, WD, HT, "Procedural generator",
         "v2 city layouts · damage field\nbuilding fates · debris · wrecked cars\n"
         "casualties placed where a disaster\nputs them (rubble, cars, buildings)",
         "#eef4e4", "#6ea44c")
    _box(ax, xs[1], Y, WD, HT, "2.5D heightfield world",
         "2 m cells, extruded heights\ndrones fly at real altitude\n"
         "3D ray-marched line of sight\n(occlusion is exact for extrusions)",
         "#f4ecdf", "#b08a55")
    _box(ax, xs[2], Y, WD, HT, "Frustum sensing",
         "pitch −50°, vfov 80°\nvoxels to 35 m depth\nsemantic rays to 80 m\nper-pixel GT embeddings",
         "#ececf0", "#7a7a88")
    _box(ax, xs[3], Y, WD, HT, "RayFronts emulation",
         "persistent open-set map:\nvoxels + features (D=24)\nsemantic rays (origin+bearing)\nfrontiers",
         "#fdeeda", "#e8912a")
    for i in range(3):
        _arrow(ax, (xs[i] + WD + 0.004, Y + HT / 2), (xs[i + 1] - 0.004, Y + HT / 2))
    ax.text(0.5, 0.30, "stochastic elements", fontsize=10.5, fontweight="bold",
            ha="center", color="#8b1a1a")
    sxs = [0.015, 0.268, 0.521, 0.774]
    labels = [("Bernoulli detection", "p(open)=0.9 · p(in car/bldg)=0.5\np(under rubble)=0.15 per look\n×0.5 beyond 35 m · 2 hits to confirm"),
              ("Feature noise", "every voxel look =\nnormalize(class emb + N(0,σ))\nconverges with re-observation"),
              ("Container masking", "hidden humans invisible far-field:\nthe ray shows the container\n(toppled car · damaged bldg · debris)"),
              ("Per-episode draws", "new city, damage, casualties\ncomms range from {100,200,400,∞} m\nquery set subsampled + edited live")]
    for x, (t, b) in zip(sxs, labels):
        _box(ax, x, 0.03, WD, 0.22, t, b, "#fbeaea", "#8b1a1a")
        _arrow(ax, (x + WD / 2, 0.255), (x + WD / 2, Y - 0.012), color="#8b1a1a", lw=1.2)
    ax.set_title("Emulation: procedural scenes → 2.5D sim → simulated RayFronts", fontsize=14, pad=12)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def model_arch(out):
    fig, ax = plt.subplots(figsize=(15, 6.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ins = [("Candidate tokens", "hold · frontier×32 · ray×32\nsegment×32 · visited\ngeometry + raw feature (D=24)", "#fdeeda", "#e8912a"),
           ("Query tokens", "mission queries (LLM-editable)\ntext embedding + weight", "#f6e9fb", "#a020c0"),
           ("Peer tokens", "teammate position · waypoint\nlast-contact age", "#ececf0", "#7a7a88"),
           ("Robot state", "pose · altitude · time left\nown coverage summary", "#e8eef8", "#5b7fb4")]
    y0 = 0.76
    for i, (t, b, fc, ec) in enumerate(ins):
        _box(ax, 0.015, y0 - i * 0.235, 0.20, 0.20, t, b, fc, ec)
        _arrow(ax, (0.219, y0 - i * 0.235 + 0.10), (0.265, 0.42 + (1.5 - i) * 0.05), lw=1.4)
    _box(ax, 0.27, 0.33, 0.155, 0.38, "Embed  d=128",
         "token MLP (geometry)\n+ feature projection\n+ type embedding\nqueries: shared feat proj\n+ weight proj + kind marker",
         "#f4f4f8", "#5b7fb4")
    _arrow(ax, (0.429, 0.52), (0.465, 0.52))
    _box(ax, 0.47, 0.33, 0.165, 0.38, "Transformer encoder",
         "2 layers · 4 heads · d=128\nself-attention over\n[robot ∥ tokens ∥ queries ∥ peers]\nshared weights across drones",
         "#e4f0f0", "#00a0a0")
    _arrow(ax, (0.639, 0.52), (0.675, 0.52))
    _box(ax, 0.68, 0.52, 0.30, 0.21, "Actor head (per drone)",
         "q·k attention score robot vs tokens\n→ distribution over candidates → next target\n(sequential decode optional, full comms only)",
         "#e4f0f0", "#00a0a0")
    _box(ax, 0.68, 0.10, 0.30, 0.21, "Critic — training only (CTDE)",
         "attention-pool all robots + global BEV CNN\n→ V(s) for MAPPO · discarded at deployment",
         "#e8eef8", "#2f6fdb")
    _arrow(ax, (0.655, 0.42), (0.70, 0.28), color="#2f6fdb", curve=0.15)
    ax.text(0.83, 0.045, "TokenPolicy ≈ 0.43 M parameters", fontsize=9.5, ha="center",
            color="#555555")
    ax.set_title("Model: per-drone TokenPolicy (decentralized execution, centralized training)",
                 fontsize=14, pad=12)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
