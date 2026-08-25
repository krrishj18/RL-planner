# RL-planner

Proof-of-concept for RL-based planning/allocation of a multi-robot casualty-search team on
procedurally generated 2.5D post-disaster scenes, with a simulated RayFronts semantic mapper.
Background and design rationale: `../rl_scoping.md`. Component contracts: `CONTRACTS.md`.

```
src/rlplanner/scene   Scene JSON schema, 2.5D exporter (vendored generator code), casualty placement
src/rlplanner/sim     raster, sensor/LoS, RayFronts emulation, tokens, env, baselines, metrics
src/rlplanner/viz     scene/raster/episode rendering, recorder, interactive viewer
src/rlplanner/train   policy, PPO/MAPPO, evaluation
scripts/              CLIs (export_scenes, view_scene, play_episode, bench_env, run_baselines, train)
tests/                pytest
data/scenes/          exported scenes (git-ignored)
runs/                 training/eval outputs (git-ignored)
```

```bash
uv sync --extra dev
uv run pytest -n auto                                   # ~500 tests, < 2 min

# scenes: v1 (build_city, 400x400) or v2 (urban v2 layout, 500-1500 m non-square)
uv run python scripts/export_scenes.py --preset earthquake tornado --seeds 0:20 --out data/scenes --size-jitter 0.25
uv run python scripts/export_scenes.py --pipeline v2 --locale downtown --disaster earthquake tornado explosion \
    --severity-range 0.5 1.0 --seeds 0:60 --region-range 500 1500 --size-jitter 0.25 --casualties auto --out data/scenes_v2
uv run python scripts/view_scene.py data/scenes_v2/downtown_tornado_17.json --out runs/scene.png

# simulator: baselines, bench, a rendered episode
uv run python scripts/run_baselines.py --episodes 10
uv run python scripts/bench_env.py
uv run python scripts/play_episode.py --synthetic 3 --policy bt --robots 3 --out runs/ep_bt.mp4
uv run python scripts/rayfronts_demo.py --synthetic 1 --out runs/rayfronts_demo.png

# training / evaluation
uv run python scripts/train.py --name run1 --scenes synthetic:0-200 --robots 3 --envs 32 --workers 10 --rollout 64 --updates 500
uv run python scripts/eval_policy.py --policy runs/run1/latest.pt --scenes synthetic:0-200 --split heldout --episodes 24
uv run python scripts/play_episode.py --synthetic 190 --policy ckpt:runs/run1/latest.pt --robots 3 --out runs/ep_trained.mp4
```

## Status (2026-08-24)

**Warm-start results (OSMO GPU, held-out synthetic 240×240 m, 24 episodes, sampled decode):** DAgger from the
privileged `oracle_sweep` teacher (144k labels) then PPO fine-tune (`scripts/warmstart.sh`) beats the best heuristic
in-distribution — central 0.68 ± 0.06 found / 0.47 ± 0.04 AUC and decentral 0.62 ± 0.05 / 0.44 ± 0.04 vs
`ray_follower` 0.56 / 0.38 (central) and 0.63 / 0.41 (decentral); PPO also repairs pure imitation's behaviour
(reward −24 → +3.5). **Zero-shot transfer to the v2 downtown set fails** (0.06 found vs nearest-frontier 0.15), but trained *on* the
v2 distribution (`ws_v2_central`: DAgger 345k labels → PPO 200 updates) the policy becomes the **best method at
city scale**: 0.19 ± 0.03 found / 0.11 AUC vs nearest-frontier 0.15 ± 0.03 / 0.08, ray_follower 0.11 / 0.06 — and
it beats the greedy *privileged* oracle (0.14, reward −256), which collapses by beelining at this scale.
With the revisit penalty scoped to confirmed targets (`ws_v2_central2`, DESIGN_VARIANTS §H4) the same
recipe reaches **0.21 ± 0.03 / 0.12, reward +8.3** — and the greedy oracle's reward flips −256 → +0.9,
confirming the penalty now charges only true waste.
At 8 robots / horizon ≤ 3000 s (`ws_v2_8robot`, dynamic queries on): **0.203 ± 0.017 found**, top of
the table vs nearest 0.17, lawnmower 0.16, ray_follower 0.16; the motion-only **`oracle_ideal`**
bound (all locations known, obstacle-aware flight, arrival = visit) reaches 1.00 in ~11 km of the
62 km budget — the gap is search + confirmation, not flight time. LLM hint ablation: no lift on the
closed-set sim (DESIGN_VARIANTS §H5). PPO-from-scratch plateaus at the heuristic level
(0.52–0.55 / 0.38); the 7-variant sweep (OSMO) found: reward penalties cost ~0.06–0.10 found (noreward 0.66–0.67 leads every
comms setting), decentral ≈ central, blackout ≈ −0.05, rays are the sharing payload that matters, and
noreward PPO-from-scratch matches the warm-start on synthetic (see DESIGN_VARIANTS.md §H2).

**OSMO:** `osmo/submit.sh --job {train,sweep,imitate,warmstart,eval_v2,...} --tag T --args "…" [--scenes v1|v2]
[--fetch prior_tag] [--env K=V]` — pods clone this repo, `uv sync`, regenerate scenes, run on a pool GPU and rsync
`runs/` to `/volume4/dsta/rl-planner/<tag>/runs/` every 10 min (credential: `airlab-storage`).

## Status (2026-08-21)

The simulator emits exactly RayFronts' three raw outputs — persistent voxels (feature sum + hit count, no query grid),
rays (per origin-cell × azimuth bin; never merged or triangulated; az/el + ground range from elevation), frontiers — and
nothing else. It is **open-set**: nothing is ever scanned against a query list. Every token ends in the item's own
feature embedding, the mission queries arrive as their own tokens with a weight each, and the policy learns relevance by
attention. A casualty is *found* when its cell is voxel-observed with the human embedding ≥ 2 times. Hidden casualties
(in cars, inside damaged buildings, under rubble) never appear in rays; only their container does.
Tokens = raw items (32 frontiers, 32 rays, 32 feature segments, hold), newest first, plus up to 8 query tokens.
~205 decisions/s on the bench scene (240×240 m, 3 robots) with the 64×64 ego crop on, 237 without it.

Heuristic baselines on the new simulator (synthetic 240×240 m, 3 robots, 600 s, 20 episodes; finds by container =
open · vehicle · building · rubble, of 2.9 · 1.75 · 2.95 · 4.4 present):

| policy | fraction found | finds-AUC | time to first (s) | finds by container |
|---|---|---|---|---|
| random | 0.47 ± 0.18 | 0.27 | 108 | 2.15 · 0.90 · 1.10 · 1.45 |
| nearest_frontier | 0.41 ± 0.15 | 0.28 | 75 | 2.70 · 0.95 · 0.80 · 0.50 |
| segment_seeker | 0.03 ± 0.06 | 0.02 | 473 | 0.10 · 0.05 · 0.00 · 0.20 |
| ray_follower | 0.49 ± 0.18 | 0.35 | 50 | 2.75 · 1.35 · 0.95 · 0.85 |
| oracle (knows casualties) | 0.93 ± 0.12 | 0.77 | 29 | 2.85 · 1.75 · 2.35 · 4.15 |

`segment_seeker` (argmax of the query cosine over segment tokens) livelocks: a segment is a persistent region of the
map with no revisit suppression, so a threshold-free greedy pick keeps returning to the same neighbourhood and coverage
stops at 0.09 against 0.98 for a plain sweep. Reported, not tuned away — moving on is what a learned policy is for.

Two later baselines, measured on the same configuration but over **10** episodes (so not directly comparable with the
20-episode table above; `random` 0.41 / 0.23, `nearest_frontier` 0.42 / 0.28, `ray_follower` 0.52 / 0.33 on the same
10):

| policy | fraction found | finds-AUC | time to half (s) | redundancy fraction | finds by container |
|---|---|---|---|---|---|
| lawnmower | 0.34 ± 0.14 | 0.22 | 569 | **0.39** | 2.00 · 0.70 · 0.70 · 0.70 |
| oracle_assign (knows casualties) | 0.93 ± 0.10 | 0.77 | **98** | 0.50 | 2.40 · 1.90 · 2.30 · 4.50 |

`lawnmower` is the coverage baseline: disjoint horizontal bands, a serpentine sweep inside each, and a divert onto any
live ray whose feature argmaxes to `human_standing`/`human_prone` (9% of its robot-decisions on synthetic seeds 0-5).
It re-covers the least ground of any baseline (redundancy fraction 0.39 against `nearest_frontier` 0.46 and
`ray_follower` 0.72) and it finds **every** open casualty, but finding needs `found_hits` looks at one cell, so a
steady sweep dwells less than a policy that ping-pongs and it trails on the container casualties. Only an
`open`-visibility human raises a far-field human ray at all, so a casualty in a car, a building or under rubble is a
container ray it sweeps past — the gap a learned policy has to close.

`oracle_assign` is `oracle` with the greedy nearest-casualty claim replaced by the optimal matching
(`scipy.optimize.linear_sum_assignment`, re-solved every decision). It reaches half the casualties in 98 s against the
greedy oracle's 139 s on the same 10 episodes, and finds slightly more of them (0.93 / 0.77 against 0.91 / 0.75) — but
it pays 5.5 intentional revisits per episode against the greedy oracle's 0, because concentrating the team on the
casualties that are left *is* revisiting (DESIGN_VARIANTS.md H), so its raw reward is lower.

No policy has been trained on this simulator yet. The earlier result (`runs/syn_fixed/`, trained policy 0.80 found /
0.61 AUC vs BT 0.67 / 0.55) was obtained on the previous simulator design and is kept only as evidence that the training
pipeline learns.

Viewing: `scripts/live_viewer.py` (live episode window), `scripts/sensor_inspector.py` (one drone: cone footprint, POV
frame, per-cell embeddings), `scripts/view_scene.py`, `scripts/play_episode.py` + `scripts/episode_viewer.py`.
