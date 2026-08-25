"""Slide graphs: training curve + baseline comparison for the 8-robot v2 run."""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "runs_8robot"
LEARNED = "#00a0a0"; HEUR = "#8a8a94"; PRIV = "#5b7fb4"; ACC = "#e8912a"

def rows(f):
    return list(csv.DictReader(open(f)))

ev = rows(f"{SRC}/ft/eval.csv")
bl = {r["policy"]: r for r in rows(f"{SRC}/ft/baselines.csv")}
import os
if os.path.exists("runs/oracle_instant_8robot.csv"):      # corrected privileged rows
    for r in rows("runs/oracle_instant_8robot.csv"):
        bl[r["policy"]] = r
if os.path.exists("runs/lawnmower_waypoint_8robot.csv"):  # true waypoint lawnmower row
    for r in rows("runs/lawnmower_waypoint_8robot.csv"):
        if r["policy"] == "lawnmower":
            bl["lawnmower"] = r
fin = rows(f"{SRC}/ft/latest_eval.csv")[0]
log = rows(f"{SRC}/ft/log.csv")

# ---- training curve --------------------------------------------------------------------------
up = [int(r["update"]) for r in ev]
ff = [float(r["frac_found"]) for r in ev]
rw = [float(r["reward"]) for r in ev]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
a1.plot(up, ff, "-o", color=LEARNED, lw=2.2, ms=5, label="learned policy (in-train eval)")
for name, ls in (("nearest", "--"), ("lawnmower", ":"), ("ray_follower", "-.")):
    v = float(bl[name]["frac_found"])
    a1.axhline(v, ls=ls, color=HEUR, lw=1.3)
    a1.annotate(name.replace("_", " "), (202, v), fontsize=8.5, color=HEUR,
                va="bottom", ha="right")
a1.annotate("ideal oracle (motion-only bound): 1.00, off scale", (0.03, 0.97),
            xycoords="axes fraction", fontsize=8.5, color=PRIV, va="top")
a1.set_xlabel("PPO update"); a1.set_ylabel("casualties found (fraction)")
a1.set_title("Held-out v2 cities — found vs training", fontsize=12)
a1.set_xlim(20, 205); a1.legend(fontsize=9, loc="lower right", frameon=False)
tr_u = [int(r["update"]) for r in log if r["ep_reward"] not in ("nan", "")]
tr_r = [float(r["ep_reward"]) for r in log if r["ep_reward"] not in ("nan", "")]
if len(tr_r) > 10:
    k = np.ones(9) / 9
    sm = np.convolve(tr_r, k, "valid")
    a2.plot(tr_u[4:-4], sm, color="#c9c9d1", lw=1.2, label="train episodes (smoothed)")
a2.plot(up, rw, "-o", color=LEARNED, lw=2.2, ms=5, label="held-out eval")
a2.axhline(0, color="#00000030", lw=0.8)
a2.set_xlabel("PPO update"); a2.set_ylabel("episode reward")
a2.set_title("Reward vs training", fontsize=12)
a2.legend(fontsize=9, loc="lower right", frameon=False)
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("8 robots · v2 cities (0.3–2 km²) · DAgger warm-start → MAPPO", fontsize=13, y=1.02)
fig.savefig("slides/training_curve.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote slides/training_curve.png")

# ---- baseline comparison ---------------------------------------------------------------------
names = {"nearest": "nearest frontier", "ray_follower": "ray follower",
         "segment_seeker": "segment seeker", "random": "random", "lawnmower": "lawnmower"}
vals = [(names[p], float(bl[p]["frac_found"]), float(bl[p]["frac_found_ci"]), HEUR, None)
        for p in names]
vals.append(("learned policy", float(fin["frac_found"]), float(fin["frac_found_ci"]),
             LEARNED, None))
ideal = [float(r["frac_found"]) for r in rows("runs/oracle_ideal_8robot.csv")]
im = float(np.mean(ideal)); ic = 1.96 * float(np.std(ideal)) / max(1, len(ideal)) ** 0.5
vals.append(("ideal oracle*", im, ic, PRIV, "//"))
vals.sort(key=lambda v: v[1])
fig, ax = plt.subplots(figsize=(9.5, 4.8))
y = np.arange(len(vals))
for i, (n, v, ci, c, h) in enumerate(vals):
    ax.barh(i, v, xerr=ci, color=c, hatch=h, edgecolor="white", height=0.68,
            error_kw=dict(ecolor="#333333", lw=1.2, capsize=3))
    ax.text(v + ci + 0.006, i, f"{v:.2f}", va="center", fontsize=10,
            fontweight="bold" if c == LEARNED else "normal")
ax.set_yticks(y, [v[0] for v in vals], fontsize=11)
ax.set_xlabel("casualties found (fraction of scene total, 24 held-out episodes)")
ax.set_title("v2 cities · 8 robots · horizon ≤ 3000 s — learned planner vs baselines",
             fontsize=12.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1.06)
ax.annotate("* motion-only bound: all casualty locations known, straight-line flight,\n"
            "arrival = visit — every casualty reachable in ~11 km of the ~62 km budget.\n"
            "The gap from the ideal to everything else is search + confirmation.",
            (0.995, 0.06), xycoords="axes fraction", ha="right", fontsize=8.8, color="#555555")
fig.savefig("slides/baselines_8robot.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote slides/baselines_8robot.png")
