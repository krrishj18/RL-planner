# Sources: training in 2D / 2.5D before moving to 3D

## 1. The field trains in 2D and evaluates 3D zero-shot — nobody fine-tunes 2D→3D
- **MAANS** (Yu et al., ECCV 2022) — multi-agent active SLAM on stacked 2D occupancy maps, MAPPO; −8% steps on unseen Habitat scenes.
- **NeuralCoMapping** (Ye et al., CVPR 2022) — robots×frontiers bipartite matching learned in 2D; evaluated in iGibson physics.
- **ARiADNE** (Cao et al., ICRA 2023) and **HEADER** (2025) — attention over a 2D viewpoint graph; HEADER beats TARE by 6–24% and deployed on a 300×230 m outdoor site, but "does not consider multilayer environments".
- **IR2** (2024), **MARVEL** (2024), **Explore-Bench** (Xu et al., ICRA 2022) — 2D-trained exploration; Explore-Bench's DRL planner finished behind three frontier heuristics; IR2 trained holonomic failed non-holonomic. Failures land exactly on the axis the 2D abstraction dropped.
- **Labiosa & Hanna (2025)** — the controlled abstraction result: a 30× cheaper sim transfers zero-shot *only* after injecting the one stochastic effect that mattered (contact noise); without it 9/10 real trials failed. → keep altitude + occlusion in the training sim, don't plan to bolt them on later.

## 2. Native 3D / 2.5D learned search exists and is cheap
- **Vashisth et al. (2024/2025)** — multi-UAV target discovery, (x, y, z, heading) actions, PPO CTDE, ~120k interactions on one A30; 66.0% vs 52.3% targets found; trained with 3 robots, generalizes to 64; validated on 44 real Tellos.
- **GenNBV** (Chen et al., CVPR 2024) — 5-DoF next-best-view policy, Isaac Gym, 256 envs ≈ 1000 FPS, 24 h on one V100.
- **Westheider & Popović** (IROS 2023) — multi-UAV **2.5D** informative path planning with altitude-dependent sensor accuracy, COMA credit assignment.
- **APEX** (2026) — aerial ObjectNav with height-banded BEV (z on the channel axis), PPO; removing the 3D map collapses success to ~1.7%.
- **Aerial Gym / Isaac Lab** — ray-cast depth+segmentation at ~3.9k FPS across 1024 envs on one RTX 3090: abstract 3D sensing is not the bottleneck.

## 3. Does 3D input help? (controlled comparisons)
- **Zhang et al.** (CVPR 2023) — adding a 3D point branch to a 2D map: +1.8 SPL / +2.7 SR on MP3D (additive, modest).
- **Ravichandran et al.** (ICRA 2022) — 3D scene graph as RL input: 44.2% vs 39.7% hidden targets found over RGB-D+semantics, at 5.5× compute.
- **VLFM** (Yokoyama et al., ICRA 2024) — 2D value map works for a ground robot on one floor; fails 10–15% of episodes that need stairs. The 2D simplification is unavailable to aerial robots over rubble and multi-storey ruins.

## Our position (one line for the slide)
2D-trained planners break on exactly the axis 2D drops, so we train in a **2.5D heightfield with real drone altitude and 3D ray-marched line of sight** — exact for extruded outdoor geometry (minus overhangs/interiors), runs at numpy speed, validated later in Isaac Sim.

## Key works in detail (for speaker notes)

**ARiADNE → HEADER (Cao et al., ICRA 2023; HEADER 2025).** Training: pure 2D occupancy grid; viewpoint
graph over known free space; attention/pointer policy trained with SAC (~24 h, one consumer GPU) for
exploration progress. Deployment: real ground robot via ROS — HEADER beats TARE by 6–24% path length,
300×230 m outdoor site. The 2D→3D step is not transfer: the SLAM stack collapses the 3D world to the
same 2D occupancy map the policy trained on, valid only because a ground robot lives on a plane
("does not consider multilayer environments"). Unavailable to drones over rubble.

**MAANS (Yu et al., ECCV 2022).** Multi-agent exploration in 3D Habitat, but the learned team planner
never sees 3D: per-robot observations are projected to stacked 2D maps, a CNN picks each robot's
spatial global goal (MAPPO, ~10⁴ episodes), a frozen local policy moves. The 2D map is the interface
between 3D world and 2D-trained planner — the pattern we generalize with a height-keeping interface.
Gains shrink −20.6% → −8.0% from train to unseen scenes; never deployed outside simulation.

**Labiosa & Hanna (2025).** The abstraction-fidelity result: a ~30× cheaper training sim transfers
zero-shot to the real robot only after injecting the one stochastic effect that matters (contact
noise); without it 9/10 real trials fail. Principle: abstractions must keep the axis the task depends
on. For aerial search that axis is altitude + occlusion → our 2.5D heightfield with real altitudes
and 3D ray-marched line of sight.
