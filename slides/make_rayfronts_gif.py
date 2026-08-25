"""RayFronts emulation as a GIF: the live composite frame over a long episode."""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from play_episode import build_env, build_policy, load_scene  # noqa: E402

from rlplanner.sim.config import EnvConfig  # noqa: E402
from rlplanner.viz import display  # noqa: E402

display.select_backend(False)
import matplotlib.pyplot as plt  # noqa: E402

from rlplanner.viz.frame import belief_legend_handles, draw_belief_panel, n_found  # noqa: E402

ZOOM_PAD = 60.0


class _A:
    scene = "data/scenes_slides/downtown_earthquake_101.json"
    synthetic = None


def _zoom(ax, st):
    xy = np.concatenate([np.asarray(r.trajectory, float) for r in st.robots], 0)
    x0, y0, x1, y1 = st.scene.region
    lo = np.maximum(xy.min(0) - ZOOM_PAD, (x0, y0))
    hi = np.minimum(xy.max(0) + ZOOM_PAD, (x1, y1))
    side = max(hi[0] - lo[0], hi[1] - lo[1], 4 * ZOOM_PAD)
    cx, cy = 0.5 * (lo + hi)
    ax.set_xlim(max(x0, cx - side / 2), min(x1, cx + side / 2))
    ax.set_ylim(max(y0, cy - side / 2), min(y1, cy + side / 2))


def main():
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    scene = load_scene(_A)
    env = build_env(scene, cfg, seed=0)
    policy = build_policy("ray_follower", cfg, seed=0)
    obs = env.reset()
    frames = []
    fig, ax = plt.subplots(figsize=(7.8, 8.4), dpi=100)
    for d in range(1, 161):
        obs, _, done, _ = env.step(np.asarray(policy.act(obs, env.state)))
        st = env.state
        if d % 2 == 0 or done:
            ax.clear()
            draw_belief_panel(ax, st, query_idx="rubble", focus_robot=0, legend=(d <= 2))
            _zoom(ax, st)
            live = int(st.rays.live().sum()) if st.rays is not None and st.rays.n else 0
            ax.set_title(f"RayFronts emulation — decision {d}   t={st.t:.0f}s   "
                         f"cov={100 * st.coverage:.1f}%   rays={live}   "
                         f"frontiers={len(st.frontier_clusters)}   "
                         f"found={n_found(st)}/{st.n_casualties}", fontsize=10)
            if len(frames) == 0:
                ax.legend(handles=belief_legend_handles(), loc="upper left", fontsize=7,
                          frameon=True)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            print(f"frame {len(frames)} d={d} t={st.t:.0f}s", flush=True)
        if done:
            break
    plt.close(fig)
    imageio.mimsave("slides/rayfronts.gif", frames, fps=6, loop=0)
    imageio.imwrite("slides/rayfronts_gif_mid.png", frames[len(frames) // 2])
    print("wrote slides/rayfronts.gif", len(frames), "frames")


if __name__ == "__main__":
    main()
