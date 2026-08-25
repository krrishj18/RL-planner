"""2x2 race GIF: four planners on one episode, synchronized clock."""
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from rlplanner.scene import schema
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.ideal import ideal_routes, obstacle_matrix
from rlplanner.sim.state import CASUALTY_ROLE_ID
from rlplanner.train.evaluate import TorchActor, load_checkpoint, make_actor
from rlplanner.train.scenes import auto_t_max
from rlplanner.viz.palette import CLASS_COLORS

SCENE = "data/scenes_v2/downtown_earthquake_59.json"
CKPT = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_FRAMES, FPS = 90, 10
ROBOT_COLS = plt.cm.tab10(np.linspace(0, 1, 10))


def make_cfg(scene):
    cfg = EnvConfig()
    cfg.robot.n_robots = 8
    x0, y0, x1, y1 = scene.region
    cfg.t_max_s = auto_t_max((x1 - x0) * (y1 - y0) / 1e6, 3000.0)
    return cfg


def run_env_policy(scene, name):
    cfg = make_cfg(scene)
    env = DisasterEnv(scene, cfg, seed=SEED)
    if name == "learned":
        pol, _ = load_checkpoint(CKPT)
        actor = TorchActor(pol, "cpu", deterministic=True)
    else:
        actor = make_actor(name, cfg, seed=0)
    obs = env.state.last_obs
    cas = env.raster.humans["role_id"] == CASUALTY_ROLE_ID
    prev = np.zeros(len(env.raster.humans), bool)
    found_t = {}
    done = False
    while not done:
        obs, _, done, info = env.step(actor.act(obs, env.state))
        new = cas & env.rf.human_found & ~prev
        for k in np.flatnonzero(new):
            found_t[int(k)] = float(env.state.t)
        prev = env.rf.human_found.copy()
    trajs = [np.asarray(r.trajectory, float) for r in env.state.robots]
    return {"trajs": trajs, "dt": float(env.cfg.dt_sim), "found_t": found_t,
            "t_max": float(cfg.t_max_s)}


def run_ideal(scene):
    cfg = make_cfg(scene)
    env = DisasterEnv(scene, cfg, seed=SEED)
    hs = env.raster.humans
    cas = np.flatnonzero(hs["role_id"] == CASUALTY_ROLE_ID)
    xy = np.stack([hs["x"][cas], hs["y"][cas]], 1)
    spawns = np.stack([r.pos[:2] for r in env.state.robots], 0)
    pts = np.vstack([spawns, xy])
    ij = np.stack(env.raster.xy_to_ij(pts[:, 0], pts[:, 1]), 1)
    D = obstacle_matrix(env.planner.obst, float(env.raster.cell_m), ij)
    arr, tours = ideal_routes(D, len(spawns), cfg.robot.speed_mps)
    dt = float(env.cfg.dt_sim)
    trajs, found_t = [], {}
    for r, tour in enumerate(tours):
        way = [spawns[r]]
        for c in tour:
            si, sj = env.raster.xy_to_ij(*way[-1])
            gi, gj = env.raster.xy_to_ij(*xy[c])
            p = env.planner.path((int(si), int(sj)), (int(gi), int(gj)))
            if p is not None:
                xs, ys = env.raster.ij_to_xy(p[:, 0], p[:, 1])
                way += list(zip(xs, ys))
            way.append(tuple(xy[c]))
            if np.isfinite(arr[c]) and arr[c] <= cfg.t_max_s:
                found_t[int(cas[c])] = float(arr[c])
        w = np.asarray(way, float)
        seg = np.hypot(*np.diff(w, axis=0).T)
        ts = np.concatenate([[0.0], np.cumsum(seg)]) / cfg.robot.speed_mps
        grid = np.arange(0.0, cfg.t_max_s + dt, dt)
        trajs.append(np.stack([np.interp(grid, ts, w[:, 0]), np.interp(grid, ts, w[:, 1])], 1))
    return {"trajs": trajs, "dt": dt, "found_t": found_t, "t_max": float(cfg.t_max_s)}


def main():
    scene = schema.Scene.from_json(SCENE)
    runs = {"ideal oracle (knows all)": run_ideal(scene)}
    for label, name in (("learned policy", "learned"), ("lawnmower", "lawnmower"),
                        ("nearest frontier", "nearest")):
        print("running", label, flush=True)
        runs[label] = run_env_policy(scene, name)
    env = DisasterEnv(scene, make_cfg(scene), seed=0)
    r = env.raster
    hs = r.humans
    cas_idx = np.flatnonzero(hs["role_id"] == CASUALTY_ROLE_ID)
    cx, cy = hs["x"][cas_idx], hs["y"][cas_idx]
    cmap = ListedColormap([CLASS_COLORS[n] for n in schema.CLASS_NAMES])
    ext = r.region[::2] + r.region[1::2]
    t_max = runs["learned policy"]["t_max"]
    frames = []
    order = ["ideal oracle (knows all)", "learned policy", "lawnmower", "nearest frontier"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6))
    for t in np.linspace(0, t_max, N_FRAMES):
        for ax, label in zip(axes.flat, order):
            ax.clear()
            ax.imshow(r.cls, cmap=cmap, vmin=0, vmax=len(schema.CLASS_NAMES) - 1,
                      origin="lower", extent=ext, interpolation="nearest", alpha=0.45)
            d = runs[label]
            k = int(t / d["dt"])
            for ri, tr in enumerate(d["trajs"]):
                if len(tr) > 1:
                    kk = min(k, len(tr) - 1)
                    ax.plot(tr[:kk + 1, 0], tr[:kk + 1, 1], color=ROBOT_COLS[ri], lw=1.1,
                            alpha=0.85)
                    ax.plot(tr[kk, 0], tr[kk, 1], "o", color=ROBOT_COLS[ri], ms=5,
                            mec="black", mew=0.5)
            ft = d["found_t"]
            done_m = np.array([ft.get(int(kc), np.inf) <= t for kc in cas_idx])
            ax.scatter(cx[~done_m], cy[~done_m], s=14, facecolor="white",
                       edgecolor="#d40000", linewidth=0.8, zorder=4)
            ax.scatter(cx[done_m], cy[done_m], s=30, facecolor="#d40000",
                       edgecolor="white", linewidth=0.8, zorder=5)
            ax.set_title(f"{label} — {int(done_m.sum())}/{len(cas_idx)} found", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"t = {t:5.0f} s / {t_max:.0f} s   ·   8 robots   ·   held-out v2 scene",
                     fontsize=12)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave("slides/race.gif", frames, fps=FPS, loop=0)
    imageio.imwrite("slides/race_final.png", frames[-1])
    imageio.imwrite("slides/race_mid.png", frames[N_FRAMES // 3])
    print("wrote slides/race.gif", len(frames), "frames")


if __name__ == "__main__":
    main()
