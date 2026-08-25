"""Lawnmower sections from the real LawnmowerPolicy.strips() partition."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from rlplanner.scene import schema
from rlplanner.sim.baselines import LawnmowerPolicy, make_policy
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.viz.palette import CLASS_COLORS

scene = schema.Scene.from_json("data/scenes_v2/downtown_earthquake_59.json")
cfg = EnvConfig(); cfg.robot.n_robots = 8
env = DisasterEnv(scene, cfg, seed=1)
r = env.raster
x0, y0, x1, y1 = r.region
pol = make_policy("lawnmower", queries=cfg.rayfronts.queries, seed=0)
strips = LawnmowerPolicy.strips((x0, y0, x1, y1), 8)
cols = plt.cm.tab10(np.linspace(0, 1, 10))
fig, ax = plt.subplots(figsize=(10, 8.4))
cmap = ListedColormap([CLASS_COLORS[n] for n in schema.CLASS_NAMES])
ax.imshow(r.cls, cmap=cmap, vmin=0, vmax=len(schema.CLASS_NAMES) - 1, origin="lower",
          extent=(x0, x1, y0, y1), interpolation="nearest", alpha=0.35)
for i, (sx0, sx1) in enumerate(strips):
    ax.add_patch(plt.Rectangle((sx0, y0), sx1 - sx0, y1 - y0, facecolor=cols[i], alpha=0.16,
                               edgecolor=cols[i], linewidth=1.8))
    ax.text((sx0 + sx1) / 2, y1 + 12, f"drone {i}", ha="center", fontsize=10, color=cols[i],
            fontweight="bold")
swath = pol._swath(env.state)
W = pol._strip_lanes((x0, y0, x1, y1), 8, 2, swath)
if W is not None and len(W):
    W = np.asarray(W, float)
    ax.plot(W[:, 0], W[:, 1], color=cols[2], lw=2.0)
    ax.plot(W[0, 0], W[0, 1], "s", color=cols[2], ms=7)
for rb in env.state.robots:
    ax.plot(*rb.pos[:2], "o", color=cols[rb.idx], ms=9, mec="black", mew=1.0)
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1 + 40)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Lawnmower sections — LawnmowerPolicy.strips(), lanes at 34.6 m FoV-corrected swath",
             fontsize=12)
fig.savefig("slides/lawnmower_sections.png", dpi=200, bbox_inches="tight")
print("wrote slides/lawnmower_sections.png")
