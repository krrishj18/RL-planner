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


def _final_extent(scene, cfg, decisions):
    """Pass 1: run the episode once to get the fixed viewport (final explored bbox)."""
    env = build_env(scene, cfg, seed=0)
    policy = build_policy("ray_follower", cfg, seed=0)
    obs = env.reset()
    for _ in range(decisions):
        obs, _, done, _ = env.step(np.asarray(policy.act(obs, env.state)))
        if done:
            break
    st = env.state
    xy = np.concatenate([np.asarray(r.trajectory, float) for r in st.robots], 0)
    x0, y0, x1, y1 = st.scene.region
    lo = np.maximum(xy.min(0) - ZOOM_PAD, (x0, y0))
    hi = np.minimum(xy.max(0) + ZOOM_PAD, (x1, y1))
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


DECISIONS, EVERY, FPS = 240, 2, 4


def main():
    cfg = EnvConfig()
    cfg.robot.n_robots = 3
    scene = load_scene(_A)
    ext = _final_extent(scene, cfg, DECISIONS)
    env = build_env(scene, cfg, seed=0)
    policy = build_policy("ray_follower", cfg, seed=0)
    obs = env.reset()
    frames = []
    h = 8.4
    w = max(5.5, 1.2 + (h - 1.4) * (ext[1] - ext[0]) / (ext[3] - ext[2]))
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.subplots_adjust(bottom=0.16)
    for d in range(1, DECISIONS + 1):
        obs, _, done, _ = env.step(np.asarray(policy.act(obs, env.state)))
        st = env.state
        if d % EVERY == 0 or done:
            ax.clear()
            for lg in fig.legends:
                lg.remove()
            draw_belief_panel(ax, st, query_idx="rubble", focus_robot=0, legend=False)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            live = int(st.rays.live().sum()) if st.rays is not None and st.rays.n else 0
            ax.set_title(f"t={st.t:.0f}s  cov {100 * st.coverage:.1f}%  rays {live}  "
                         f"frontiers {len(st.frontier_clusters)}  "
                         f"found {n_found(st)}/{st.n_casualties}", fontsize=9)
            fig.legend(handles=belief_legend_handles(), loc="lower center", ncol=4,
                       fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 0.005))
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if len(frames) % 20 == 0:
                print(f"frame {len(frames)} d={d} t={st.t:.0f}s", flush=True)
        if done:
            break
    plt.close(fig)
    imageio.mimsave("slides/rayfronts.gif", frames, fps=FPS, loop=0)
    imageio.imwrite("slides/rayfronts_gif_mid.png", frames[len(frames) // 2])
    imageio.imwrite("slides/rayfronts_gif_late.png", frames[-1])
    print("wrote slides/rayfronts.gif", len(frames), "frames")


if __name__ == "__main__":
    main()
