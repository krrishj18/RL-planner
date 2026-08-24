# RL-planner — contracts between components

Read this before touching any module. Exact field definitions live in code and win over prose:
`src/rlplanner/scene/schema.py` (Scene JSON, raster classes, default heights),
`src/rlplanner/sim/config.py` (EnvConfig), `src/rlplanner/sim/state.py` (TeamObs, EnvState, RayStore …).
Changing any of those three files is a contract change: do it only if unavoidable, keep backward
compatibility, and say so in your report.

## 0. Ground rules
- Python 3.12, `uv` only (`uv run …`, `uv run pytest`). Don't add dependencies without a reason; never pip.
- Pure numpy/numba in the simulator (no torch). Torch only under `rlplanner/train`.
- Units: metres, seconds, radians inside code (degrees only in config/JSON). World frame = scene frame
  (x right, y up, z up, ground z=0). Grids are `(ny, nx)` row-major: `row = y index`, `col = x index`.
- Determinism: every stochastic component takes an `np.random.Generator`; same seed ⇒ bit-identical run.
- Never silently clip/ignore bad input — raise with a message that names the offending object.
- Comments: terse, only where the code isn't self-explanatory. No change narration.
- Every feature ships with pytest tests in `tests/`. A separate tester agent will attack each module
  for edge cases; make that easy by keeping functions small and pure.
- No git commits/pushes from agents.

## 1. Coordinates, raster (owner: sim)
`Raster` (in `sim/raster.py`) = `cell_m`, `origin=(x0,y0)`, `nx, ny`, and per-cell layers
`height[f32]` (max extruded height), `cls[i8]` (`schema.CLASS_ID`), `damage[f32]` (field sampled at
cell centres; prefer `damage_field.grid` when present), `obj_id[i32]` (index into a flat object table,
-1 = none). Helpers: `xy_to_ij`, `ij_to_xy` (cell centre), `in_bounds`, `obstacle_mask(alt, clearance)`.
Humans are *not* rasterised (they are hidden state): `Raster.humans` = struct array
`(x, y, z, role_id, pose_id, container_id, visibility_id, scene_idx)`.
Class priority when footprints overlap (highest wins): vehicle_* > debris > building_* > bus_stop >
street_furniture > tree > sidewalk > road > park > ground. Rotated rectangles are rasterised by a
vectorised point-in-OBB test on cell centres; debris/trees are discs.

## 2. World dynamics (owner: sim)
- Robots are kinematic points at fixed altitude `robot.flight_alt_m`, speed `speed_mps`.
  Motion toward the current target follows an A* path (8-connected) over
  `obstacle_mask(alt, clearance)`; the path is cached until the target changes. No path ⇒ the
  token is `reachable=0` and masked; a robot whose target became unreachable holds and emits an
  `unreachable` event.
- Time: sub-step `dt_sim` (sensing + RayFronts update + motion every sub-step); one decision per
  `decision_dt` (= `substeps_per_decision` sub-steps). Arrival = within `arrive_radius_m`; an arrived
  robot holds (keeps sensing) until the next decision.
- Episode ends when every casualty is found or `t >= t_max_s`.
- Spawn from `scene.robots_spawn` (first `n_robots`; if fewer, jitter the last by `spawn_jitter_m`;
  if none, use the region's lowest-left road cell).

## 3. Sensor (owner: sim)
Camera at the robot position, yaw = heading, pitch `sensor.pitch_deg`, `hfov/vfov`. A cell is
*in view* if the vector from the camera to the cell's top point `(xc, yc, height)` (for humans:
`(x, y, 0.5)`) is inside the frustum and has line of sight. LoS = ray-march from camera to target in
steps of `los_step_frac * cell_m`; blocked if any intermediate sample has `height[cell] > z_ray +
los_eps_m` (skip the start and end cells). Slant range `r`: `r <= depth_limit_m` ⇒ **observed**
(voxel update); `depth_limit_m < r <= visual_range_m` ⇒ **far-visible** (ray candidate).
`sensor.mode = "disk"` keeps LoS but ignores the frustum (debugging). Implement LoS in numba; it is the
hot loop.

## 4. RayFronts emulation (owner: sim, `sim/rayfronts_sim.py` + `sim/embeddings.py` + `sim/segments.py` + `sim/similarity_table.py`)
Emulates what the real RayFronts publishes and **nothing else**: a persistent semantic voxel map,
semantic rays, and frontiers. What it publishes per voxel and per ray is a *feature embedding*, not
a score against a list of words. There is no separate tracker, no per-target record, no association
or triangulation step, **and no query scan**: `vox_sim` as a stored grid is gone, rays carry no
per-query column, and nothing in the per-decision path evaluates a query. `RayFrontsSim.n_query_calls`
counts every lazy query view taken and must not move across a decision.
- `SIM_TABLE[class, query] ∈ [0,1]`, hand-authored in `similarity_table.py` (document each row), with
  `load_sim_table(path)` to override from JSON. It is the *source of the embeddings*, not something
  the belief is indexed by. Background classes score low (≤0.2) on person queries; plausible
  confusions are encoded (debris↔collapsed building 0.5, human_standing→"person lying" 0.4,
  vehicle_toppled→"car" 0.7 …).
- `sim/embeddings.py` turns that into unit vectors: `EmbeddingTable(class_emb [N_CLASSES, D],
  query_emb [Q, D])` with `cos(class_i, query_j) = SIM_TABLE[i, j]`. Source order: the cached JSON
  from `scripts/build_text_embeddings.py` (a real CLIP/SigLIP text tower, PCA-reduced to
  `rayfronts.embedding_dim`, keeping the basis so new queries can be encoded at runtime), else a
  factorization of the hand table (eigendecomposition of the joint Gram + projected-gradient
  refinement, exact to <1e-6; no model download, so tests never hit the network).
  `EmbeddingTable.pc_basis(k)` is the **fixed feature basis**: the PCA of the *class* vectors, so it
  depends only on the table and is identical in every env, scene and viewer. `project(feat, k)` maps
  a feature onto it; that is what every raster channel carries.
- **Voxels** (persistent memory: never cleared, never decayed): each observed cell per sub-step
  contributes `normalize(class_emb[cls] + N(0, feat_noise_std))` into `vox_feat_sum [ny, nx, D]`
  (with prob `p_confuse` a random other class); `vox_cnt` and `last_seen_t` as before.
  `vox_feat = normalize(vox_feat_sum)`. That is the whole voxel update — no derived cache.
  `feat_noise_std` replaces `vox_noise_std`, which stays a live alias.
- **Humans**: hidden state, never rasterised. A human whose LoS point is in view is *observed* with
  probability `p = p_observe_base[visibility] * range_factor` per sub-step, `range_factor = 1` for
  `r ≤ 0.5·depth`, linear to 0.5 at `depth_limit`, times `far_observe_factor` beyond it.
  - within `depth_limit_m`: that sub-step's voxel observation of the human's own cell uses the
    `human_prone`/`human_standing` row, so the person enters the map like any other semantics (if
    the cell itself failed the cell-level frustum/LoS test — nadir edges, a body beside a wall — it
    is added to the scatter anyway: the camera saw the person).
  - beyond it: **only `visibility == "open"`** can be seen at all, and it contributes a ray.
    `partial` (in a vehicle, inside a damaged building) and `occluded` (under rubble) humans are
    invisible from afar; what the drone sees down that bearing is the container, whose own cells
    already emit the ray (vehicle_toppled / building_damaged / debris).
  - **Found** = the human's cell has been voxel-observed with its own human row at least
    `found_hits` times (default 2) — RayFronts' own hit-count acceptance of a voxel, and the only
    find mechanism there is. Casualty ⇒ reward once + a `found` event; a bystander crosses the same
    threshold, looks like a standing person in the map, and earns nothing. There is no approach,
    approach-and-verify step, and no radius around the robot that reveals roles.
- **Rays**: each far-visible cell emits with prob `p_ray_per_cell`: origin = robot xy, `az/el` to the
  cell, an observation feature `normalize(class_emb[cls] + N(0, ray_noise_std))`, weight
  `w = 1 - r/visual_range`. `RayStore.feat` is the weighted mean feature and `RayStore.feat_peak`
  the most salient single observation; there is no `sims` column. Rays are binned by (origin cell of
  `ray_origin_cell_m`, azimuth bin `ray_az_bin_deg`) — the mapper's own aggregation of repeated
  looks down one bearing, and the *only* aggregation there is: weighted running mean of `feat`,
  `conf = min(conf + w, ray_conf_cap)`, `n_obs`, `t_first/t_last`, stable `id`. `conf` and `n_obs`
  count *looks*, one per bin per sub-step (the sub-step's far cells are already collapsed into one
  observation), with `w` the mean `1 - r/visual_range` of the cells that look describes. Rays are never
  merged, deduplicated or intersected across origins or bearings; two rays that happen to point at
  one place stay two rays, and working that out is the policy's job.
  A bin's `az/el` is the direction of its most salient observation (the one `feat_peak` records).
  False positives: per robot per sub-step with prob `p_fp_ray` a spurious ray at a random azimuth
  carries the `human_prone` row; it is not labelled anywhere — the belief cannot tell.
  **Ray target point**: `range = alt / tan(-el)` (the ground point that elevation implies), clipped
  to `[depth_limit_m, visual_range_m]`, falling back to `ray_range_m` when `el >= 0`; the point is
  `origin + bearing · range`, clipped to the region. It is geometry for placing a waypoint, not an
  inference about what the ray sees.
  A ray is **resolved** (removed from the live set) when ≥ `ray_resolve_frac` of the cells along its
  bearing between `depth_limit` and `visual_range` are observed, when the disc of
  `ray_resolve_radius_m` around its target point is observed (the area it points into has been
  mapped), or when `t - t_last > ray_ttl_s`.
- **Frontiers**: observed cells 4-adjacent to an unobserved in-region cell. Cluster by connected
  components (8-conn) after dropping clusters `< frontier_min_cluster_cells`; per cluster: centroid,
  `n_cells`, `info_gain` = unobserved cells within `frontier_ig_radius_m` of the centroid (kept for
  the visualizer; it ranks nothing). Stable ids: match a new cluster to the previous one whose
  centroid is nearest within 2 cells, else new id. Frontiers are recomputed from the persistent
  voxel map every decision.
- **Segments** (`sim/segments.py`): a generic over-segmentation of the *observed feature map* —
  Felzenszwalb-Huttenlocher graph segmentation over the 4-neighbour grid with edge weight
  `1 - cos(feat_a, feat_b)`, one scale parameter `segment_scale` (in cells) and a minimum size
  `segment_min_cells`. No class, no query, no ranking: it is how the open-set content of the map
  reaches the policy in place of "salient voxels". Per segment: mean of the member cells' unit
  features, medoid, `n_cells`, mean hit count, ray count (live rays whose corridor crosses it —
  geometric bookkeeping, one credit per ray per segment), `t_first`/`t_last`. Ids are matched to the
  previous set by medoid within 2 cells. The labelling is re-run only once `segment_refresh_frac`
  of the observed cells are new or `segment_refresh_decisions` decisions have passed; the statistics
  are refreshed every decision on the cached labels. Segments never contain unobserved cells.
- **Candidate order**: recency, never a score. Frontiers newest-first then by size, rays newest-first,
  segments newest-first — each capped at `CAND_POOL ×` its token count. Nothing in the simulator
  chooses *which* items deserve a slot on semantic grounds; that is what the policy is for.
- **Query views** (lazy, off the per-step path, for viewers/tests/heuristic baselines):
  `RayFrontsSim.query_sim(query) -> [ny, nx]`, `RayFrontsSim.ray_query_sim(query, peak=True) -> [n_rays]`,
  `RayFrontsSim.query_vec(query) -> [D]`. `query` is a name, an index into the live mission list, or
  a raw `[D]` vector. Each call bumps `n_query_calls`.
- `set_queries(names, weights=None)` swaps the *mission* query list and its weights. Nothing in the
  belief depends on it, so nothing moves but the query tokens.

## 5. Tokens / observation (owner: sim, `sim/tokens.py`)

**Open-set principle.** RayFronts is an open-set mapper: what it stores is a per-voxel and per-ray
*feature embedding*, and a query similarity is one optional view of it. Scanning every cell against
a fixed list of eleven words throws that away — it decides in advance what the map is allowed to
contain, and it makes the observation's width, meaning and training a function of that list. So the
observation carries the **embeddings**: every token ends in the item's own `feat[D]`, and both
raster stacks carry features projected onto a fixed PCA basis. The mission queries are **inputs to
the policy** — their own tokens, with a weight each — not a fixed block baked into every item. The
policy learns relevance by attending from the items to the queries; adding a query, dropping one, or
handing it a hint proposed by an LLM changes the input, not the network, and the simulator ranks
nothing. Nothing anywhere selects which items the policy may consider beyond recency and the slot
caps.

**CTDE split — actor = local + gossip, critic = global.** Execution is decentralised: each drone
runs the policy on its own RayFronts map plus whatever peers gossip to it. The *actor* reads
per-robot things: `tokens`, the ego-centric `local` crop, `peer_tokens`, its own `robot_bev`, and
the query block. The *critic* reads the compressed global `bev` (the union belief) and nothing
else that the actor does not have. `TokenBuilder.build(rf, robots, t, planner, views=None)` takes
one `RobotView` per robot — `known` mask, `feat_known` mask, `feat_sum`, `hits`, `last_seen`,
`frontier_mask`, its frontier/ray/segment lists, its `RayStore`, its visited records and its peer
cache — so a robot's observation is a function of that robot's knowledge and nothing else.
Under `comms.mode == "full"` every robot is handed the same `team_view(rf)` (`views=None`), which
is the old behaviour; under `comms.mode == "range"` `sim/comms.py` supplies one view per robot and
the builders do not change.

Per robot, `K = k_tokens` tokens in fixed slot order:
`[hold] + k_frontier + k_ray + k_segment + k_visited` (1 + 32 + 32 + 32 + 32 = 129). The candidate
set *is* the RayFronts topics plus "stay put" plus the team's own visited records — no priority
between them, no token type that stands for a tracked target.
- frontier slots: every cluster up to the cap, newest first then by size; target = the cluster medoid.
- ray slots: every live ray up to the cap, newest first; target = the elevation-derived point above.
- segment slots: every segment up to the cap, newest first; target = the segment medoid.
- visited slots: the newest visited records this robot knows (own + gossiped), target = the
  recorded xy.
Empty slots: zeros, `mask=False`, `xy=nan`, `id=-1`. Ids are per-type counters, so the claim key is
`(type, id)`. Under `comms.mode == "range"` every robot has its *own* frontier/segment numbering,
so the ids it emits are offset by `robot_idx * state.TOKEN_ID_STRIDE` (2^20): two robots can never
collide on a claim key, and the sequential-decode claim mask (a centralised-execution device,
switched off for every decentral variant — see `policy.sequential_decode` in 6) therefore never
fires across robots. Ray ids are the global ray-bin ids, offset the same
way; a visited record keeps one env-wide id, so the same record is the same token for everyone.

Feature vector layout = `state.token_feature_names(D)`: `TOKEN_FIXED = 32` columns plus the item's
`feat[D]` tail (`D = rayfronts.embedding_dim`, 24 ⇒ `F = 56`). **The width does not depend on the
mission queries**, so a trained network survives `set_queries`. All columns scaled to roughly
[-1, 1]: `dx, dy, dist` ÷ region diagonal; `bearing` relative to heading; `x_abs, y_abs` = the target
in region coordinates ∈ [-1, 1]; `n_cells` ÷ 200; `hits` = `log1p(mean hit count)/log1p(50)` over the
observed cells within 20 m; `confidence` = ray conf ÷ `ray_conf_cap`; `age` = time since last
observed ÷ `ray_ttl_s`; `n_obs` = `log1p(looks)/log1p(500)`; the **ray geometry block**
`origin_dx, origin_dy, az_sin, az_cos, el_sin, el_cos, range_est` — the raw ray topic, so a policy
can see for itself that several rays converge, zero for the other token types; `coverage_nbhd` =
observed fraction within 20 m (its complement is the unobserved fraction a frontier or segment
borders); `ray_count` = live rays crossing a segment ÷ 8; `claimed_by_peer` = 1 if another robot's
current target is within 15 m; `peer_dist_min` ÷ diagonal — both are read from the robot's **peer
cache** under `comms.mode == "range"` (a robot that has heard from nobody reads `peer_dist_min = 1`
and `claimed_by_peer = 0`, never the peers' true positions) and from the live team under `"full"`;
`reachable` = A* path exists; and the **visited block**
`visit_found` (casualties that visit turned up ÷ 4), `visit_own` (1 if this robot made the visit)
and `visit_who` ((robot index + 1)/10), zero for the other token types. The
nominal normalisers (`n_cells`, `hits`, `age`, `n_obs`, `ray_count`, `visit_found`) are saturated
at 1.
The `feat[D]` tail is unit-norm: a ray token carries the bin's **most salient** observation
(`feat_peak`, the one its `az/el` describe), a segment token the mean of its cells' unit features,
a visited token the feature the chosen token carried at the time of the visit, and a frontier or
hold token the mean unit feature of the observed cells within 10 m of its target (zeros when
nothing there is known).

**Query block** (shared by every robot of an env): `query_emb [Qmax, D]` unit mission-query
embeddings zero-padded to `Qmax = tokens.max_queries` (8), `query_w [Qmax]` (1.0 for a mission
query; an LLM-proposed hint carries its likelihood there) and `query_mask [Qmax]`.

Robot features: `x, y` ÷ region, `alt/100`, heading sin/cos, `t/t_max`, `1 - t/t_max`,
observed fraction of the observable map, one-hot robot index (10).

**Local crop** `local [n_robots, Cl, S, S]` (`S = tokens.local_size`, 64; 0 disables it): an
axis-aligned ego-centric window of *that robot's* belief at raster resolution — `known`, `hits`,
`ray_count`, `feat_pc[8]` and `visited` (a blob per record it knows). Out-of-region cells stay zero
(= unknown). This is the dense near-field the actor plans in.

**BEV** `[C, size, size]` over the whole region, 23 channels: `known`, `hits`, `frontier`,
`robots` (gaussian blobs), `peer_targets`, `ray_count`, `feat_pc[8]` (features on the fixed
class-PCA basis, zeroed where there is no feature) `ray_feat_pc[8]` (mean feature of the live rays
crossing each cell, same basis) and `visited`. No ground truth (the old `height` channel was the
raster's, not the belief's) and no per-query channel. It is built twice:
- `bev [C, bev_size, bev_size]` (64) over the **union** belief and every visited record — the
  centralised critic's view, one per env;
- `robot_bev [n_robots, C, robot_bev_size, robot_bev_size]` (32; 0 disables it) over **that
  robot's** belief, with the robots/targets channels drawn from its peer cache — the actor's
  global picture. It is per robot, so 64x64x23xR does not fit through the worker pipes at 32 envs
  (36 MB per decision); 32x32 costs 94 kB per robot per decision and is the shipped default.

**Peer tokens** `peer_tokens [n_robots, n_robots - 1, 14]`, from the robot's peer cache: relative
position, relative target, contact age (÷ `t_max_s`), `valid`, distance, the peer's reported
coverage fraction, `link` (in contact this decision) and the peer's target type one-hot. A peer
never heard from is all zeros (`valid = 0`), and so is the padding of a smaller team. Under
`comms.mode == "full"` every peer reads as linked with age 0 and the team coverage.

Action = token index per robot (int). Invalid (masked) action ⇒ raise in `step` (the policy must mask).

## 5.1 Comms and gossip (owner: sim, `sim/comms.py`)
`cfg.comms` is a `CommsConfig`: `mode` (`"full" | "range"`), `range_m`, `range_choices`,
`randomize_range`, `relay_hops`, `spawn_exchange` and a `share` block. Assigning the string
`"full"`/`"range"` to `cfg.comms` still works — `validate()` coerces it.

- **full**: one shared team belief, `views=None`, `env.comms is None`. The reference variant.
- **range**: every robot owns a `RobotBelief` — `known` (own observations ∪ received coverage),
  `feat_known` (the part that carries a feature), its own `FrontierIndex` and `SegmentIndex`
  computed from those masks, the rays it emitted plus snapshots of the rays peers sent, a peer
  cache and its visited records. `range_m` is drawn per episode from `range_choices` when
  `randomize_range` (training); `range_m <= 0` is a blackout (no link at any distance).

**Links.** Two robots are linked when they are within `range_m` (inclusive); a payload reaches
everything in the same connected component, or within `relay_hops` edges when that is > 0 — the
range + relay model of `AirStack/.../coordination_bringup/comms_model.py`. Contact is evaluated
**once per decision**, at the positions the robots hold at the end of it, after the belief has been
updated and before the observation is built. The round is **simultaneous**: every payload is
snapshotted before anything is delivered, so what a robot forwards is what it knew at contact and
not what a lower-indexed peer handed it in the same round (otherwise a message crossed one hop more
than `relay_hops` allows, in an order decided by the robot indices). `spawn_exchange` gives one
contact at t = 0 regardless of range (the team starts together and knows it); it carries positions
and targets only, and it does **not** count towards `link_frac`, so a blackout variant reports 0.

**Payloads** (small and typed, per DESIGN_VARIANTS.md B). Always: position, current target
(type + xy), contact time and the sender's reported coverage fraction. Then, per `share`:
`rays` (`none` | `newest` (`rays_newest` = 8 of them) | `all`, hard-capped at `ray_cap` = 64,
newest first, the sender's *own* live rays),
`coverage` (its `known` mask coarsened to `coverage_cell_m` = 8 m blocks), `segments` (its own
segment tokens, capped at `segment_cap` = 32), `visited` (the newest `tokens.k_visited` records it
knows, its own and gossiped alike — a record is the point of the epidemic, while rays and segments
are only the sender's own), and `features`
(off by default: makes the shared coverage cells known *with* features). Received items persist:
nothing is dropped when the link goes down (see **Inboxes** below for the two ways an item does
die, neither of them a link event).

**Approximations** (documented in `sim/comms.py`): the feature map `vox_feat_sum` and the hit
counts `vox_cnt` stay global and a robot reads them through its own `feat_known` mask (a per-robot
copy would be 54 MB per robot at 750x750); a cell known only through shared coverage carries
`known = 1`, no hits and no feature, so on a receiver's BEV *and* local crop it reads as covered
ground with exactly zero in the hit and feature channels (and it suppresses a frontier); a received
ray is a **snapshot** of that bin at contact time, does not keep updating, and resolves against the
*receiver's* map; a peer's segment and the robot's own segment of the same ground stay two tokens.
Ray rows are never compacted under range comms, so a row index is a ray id.

**Inboxes.** A ray snapshot leaves the inbox when the receiver's own map resolves it, and when the
receiver starts feeding that bin itself — two live copies of one bin would be two tokens with the
same `(type, id)` and a double count in the ray rasters and in a segment's `ray_count`. A peer
re-sending the bin puts a fresh snapshot back, which is what the dict did before. Received segments
are capped at the same candidate pool `refresh` keeps (`CAND_POOL * k_segment`, newest by
`t_first`), because a peer relabels its map on every resegmentation and its old ids never come
back; the id memo keeps a re-received segment's token id stable. Neither rule drops anything that
could still become a token, and without them both inboxes grew all episode (5300 ray snapshots per
robot by decision 180 of a synthetic episode, of which 0 were still live).

**Memory**: per robot `known`, `feat_known`, the frontier mask (bool) and the segment labels
(int32) = 3.9 MB at 750x750 cells (1500 m at 2 m), plus the ray/segment/visited inboxes; the
budget is 10 MB per robot and the test asserts it.

## 6. Env API (owner: sim, `sim/env.py`, `sim/vec_env.py`)
```python
env = DisasterEnv(scene, cfg: EnvConfig, seed=0)
obs: TeamObs = env.reset(seed=None)
obs, reward: float, done: bool, info: dict = env.step(actions: np.ndarray[int, (n_robots,)])
obs = env.set_queries(names, weights=None)   # mission queries only; the belief does not move
env.state -> EnvState     # live view (do not mutate from outside)
```
`info` keys: `new_found, found_total, n_casualties, coverage, dist_travelled[n], events_this_step,
metrics`, plus the per-robot arrays `redundant_cells`, `redundancy_refunds`,
`intentional_revisits`, `revisit_refunds`, `revisit_penalties` and the scalars `comms_range_m`,
`link_frac`.
Reward (team, shared):

    casualty_reward * new_found - time_cost
      - redundancy_cost * mean_r (redundant cells of robot r / cells robot r observed this
                                  decision; 0 for a robot refunded this decision)
      - revisit_cost    * sum_r  (intentional revisits by robot r this decision
                                  - the ones refunded by a find)

(+ optional potential shaping, off by default); bystanders are worth exactly zero. The two new
terms are **per robot but paid by the team**: MAPPO gives every robot the same return, so the team
carries the cost and `info` records who caused it. Redundancy is charged as a *fraction of what
the robot looked at*, averaged over the team, so a decision in which everyone re-covers known
ground costs exactly `redundancy_cost` (0.05) whatever the footprint, the cell size or the team
size — counting cells made the term scale with both (2000+ cells per robot per decision on a v2
scene). "Already observed" is snapshotted at the start of the decision, so two robots that cover
the same new ground in the same decision pay nothing.
The refund (`reward.redundancy_refund`) zeroes that robot's redundancy term for the decision if it
found a casualty in one of its redundant cells. A **visited record** is written when a robot
arrives at a ray or segment target it chose (xy, type, the chosen token's feature, time, the
casualties *it* found within `revisit_m` during that decision, and who); an **intentional revisit**
is charged when a robot arrives at the target it chose and that target lies within `revisit_m` of a
record another robot made earlier **whose visit confirmed a casualty** (`n_found > 0`;
`reward.revisit_confirmed_only`, default on: occluded detection is stochastic, so an unconfirmed
target may still hold a casualty and going back to it is search, not waste) — with
`reward.revisit_known_only` (default) only if the record
had reached this robot before the decision started. **Only another robot's record counts**
(DESIGN_VARIANTS.md D says "another drone"): a robot that flies back to a place it visited itself
pays nothing, and neither does a record made at or after the arrival instant. A fly-by never triggers it: the check runs on
arrival at the chosen target, not on proximity. An arrival only counts if the robot was **outside
`arrive_radius_m` of that target when the decision started**: re-selecting the token you are parked
on is not a journey, so it writes no record and pays no revisit (without that rule a policy that
holds position pays 0.5 every decision and the record list fills with copies of one visit).
`reward.revisit_refund_on_find` (default on) applies the redundancy term's principle to the
revisit: an arrival that turns up **a new casualty within `revisit_m` of the target** was not a
wasted journey, so its `revisit_cost` is handed back. The charge stays open while the robot is
still within `arrive_radius_m` of that target — a find needs `found_hits` looks and can land a
decision or two after the arrival — so a refund can arrive after the decision that was charged;
the episode total nets out, and `info` / the metrics carry the gross `intentional_revisits` and
the `revisit_refunds` separately. Without it the term dominates every other: the oracle finds 85%
of the casualties and pays 84 revisits (-42) per synthetic episode, because concentrating on where
the casualties are *is* revisiting (DESIGN_VARIANTS.md H).
`VecEnv(envs)` steps a list of envs (different scenes/seeds), stacks obs with padding, auto-resets
and returns `final_info` for finished episodes. The CTDE split of §5 holds through the whole train
stack: `train/policy.py` feeds `local` (+ the token and query sets) to the actor and `bev` only to
the value head, so `use_bev` is a **critic** switch and `use_local` an **actor** one.
**CTDE execution constraint: `policy.sequential_decode`.** The sequential decode of
`train/policy.py` lets robot `r` see the tokens robots `< r` claimed **in the same decision**.
That information does not exist at decentralised execution — a peer's choice reaches this robot
only through gossip, one decision later, as `claimed_by_peer` and the peer tokens. So the switch
follows the variant's execution model, not the training convenience: `sequential_decode = True`
only for `central_full` (centralised execution), `False` for every `decentral_*` variant, where
the robots decode independently and **two robots may pick the same token in one decision**. It is
a `TokenPolicy` argument, so `policy_config` carries it in the checkpoint and it holds in
rollouts, in PPO's re-evaluation of the same actions and in evaluation alike; the variant yaml
sets it in its `train:` block (`sequential_decode`), `scripts/train.py` takes
`--sequential-decode/--no-sequential-decode` and records it in `args.json` (as
`policy.sequential_decode`), and the sweep prints it in the summary header. Nothing else
de-conflicts a decentralised team: what the policy learns from the peer block is the whole of it.
`VecObs` carries the query block, `local`, `peer_tokens` and `robot_bev` alongside the token
arrays; `train/par_env.py` ships the same fields (`_OBS_FIELDS`). `ObsBatch` mirrors them, and
`TokenPolicy(use_peers=..., use_robot_bev=...)` are actor switches like `use_local`.
`set_queries` never changes an array shape, so envs sharing a config need not switch together and a
checkpoint keeps loading.

## 6.1 Query hints and query churn (owner: llm, `rlplanner/llm/`)
The mission query list is an *input*, so something may edit it while an episode runs. Two users of
that, both going through `DisasterEnv.set_queries` and nothing else:

- **`EnvConfig.queries_dynamic`** (`QueryScheduleConfig`, **default `enabled = False`**) — training
  robustness, no LLM. `llm/schedule.py: QueryScheduleSampler` draws an initial subset of a query
  pool at `reset` and applies one add/remove/reweight edit every `every` decisions with probability
  `p_edit`; `noise_std > 0` registers a jittered copy of a pool query in the embedding table's bank
  under a name hashed from its vector (so two envs sharing the cached table can never give one name
  two meanings). It is a function of `np.random.default_rng([env.seed, …])`, so a seed fixes the
  whole schedule. `reset` restores the *mission* list before the belief is rebuilt, so an episode's
  draw is never the next episode's embedding table — a noised name exists only in the bank of the
  table it was drawn against. Off, `DisasterEnv` never calls it and every run reproduces bit-for-bit.
- **`llm/hint_agent.py: HintAgent` + `HintController`** — the closed loop. `digest.py` turns the
  live belief into a length-capped text digest (new-this-interval rays and segments summarised by
  **nearest-class decode**, coverage, casualties found by container, the current list with its
  weights); the agent returns `{add, remove, reweight}` over query *strings*; `embed.py` turns a
  string into a unit `[D]` vector (a least-squares SigLIP→sim projection when a text tower is
  installed, otherwise a lexicon of the class prompts and the query bank with the prompt
  constrained to it) and registers it in `EmbeddingTable.bank`, which is what makes
  `set_queries(("crushed vehicle", …))` legal. The decode is **for the LLM's eyes only** — the
  policy still receives raw `feat[D]`. The agent never picks a target, a waypoint or a token, and
  any backend failure (missing CLI, timeout, non-zero exit, unparseable output) returns no-op edits
  and a warning; it never raises into an episode. The active list is capped at `tokens.max_queries`,
  weights are clamped to [0, 1], and a removal that would empty the list is refused (the sim rejects
  an empty query list, and the agent's list must not drift from the env's).
- `scripts/llm_hints_eval.py` evaluates one policy under `none` (the query block zeroed on the
  observation handed to the policy) / `static` / `llm` / `scripted`, prints the `train/evaluate.py`
  table and writes the query-edit log to CSV + JSONL.

## 7. Baselines (owner: sim, `sim/baselines.py`)
`Policy.act(obs: TeamObs, state: EnvState | None) -> np.ndarray[int]`; `privileged: bool`. Simple
heuristics, one line of arbitration each and **no threshold** — deliberately not a behaviour tree, so
that what a learned policy adds stays legible. Each skips a candidate already claimed by a
lower-index robot this step — claimed by target **position** (within 15 m) as well as by
`(type, id)`, because under range comms every robot numbers its own frontiers, rays and segments,
so an id key can never match across robots and the arbitration would silently do nothing (measured:
`ray_follower`, 3 robots from one spawn, identical actions in 35 of 40 decisions and 0.55 coverage
against 0.92 with the position key). Like the sequential-decode claim mask of §5 this is a
**centralised-execution device** — it reads the choices a lower-index robot has just made — and it
is why a baseline row and a learned decentralised row are not the same kind of team: a
decentralised policy runs with `sequential_decode = False` (6), and under `comms.mode == "range"`
its claim key could not match across robots anyway (per-robot id namespaces), so nothing
de-conflicts a learned team except what it reads from `claimed_by_peer` and the peer tokens. A baseline that wants a query view computes it *itself* from the
observation (`Policy.query_scores(obs, r)` = best cosine of each token's `feat` against the active
query tokens); the environment scores nothing.
- `RandomPolicy`, `NearestFrontierPolicy` (nearest frontier at least `min_travel_m` away, else the
  farthest one).
- `RayFollowerPolicy` ("ray_follower"): argmax of `query_scores` over the ray tokens; if no ray token
  is on offer, nearest frontier.
- `SegmentSeekerPolicy` ("segment_seeker"): the same over the segment tokens.
- `LawnmowerPolicy` ("lawnmower"): boustrophedon coverage that breaks off for a person it sees.
  The region is cut into `n_robots` contiguous horizontal bands, robot `r` owning band `r`; inside
  its band the robot follows a fixed serpentine order (lanes one sensor swath tall, +x on an even
  lane, -x on an odd one) and the action is **the in-band frontier with the smallest sweep key**.
  There is no sweep cursor: frontiers vanish as the ground behind them is mapped, so the smallest
  remaining key *is* the next unswept point and an interrupted sweep resumes instead of restarting.
  A frontier the robot has effectively reached sorts last (the `min_travel` guard, folded into the
  key); an empty band falls back to the nearest frontier anywhere; only a robot with no frontier
  holds. Bands are disjoint, so the position claims only bite on that fallback, and the whole thing
  runs on the robot's own view — unchanged under range comms and for any `n_robots >= 1`.
  **Investigate**: a live ray whose feature *is* a person diverts the robot, classified
  threshold-free as the argmax cosine of the ray token's `feat` over the **whole class-embedding
  set** (a human ray iff the argmax is `human_standing`/`human_prone`; no score meets a cutoff and
  no mission query is read, which would make the divert a function of the word list). The nearest
  such ray wins and the robot stays on it until the ray resolves or its target falls inside the
  `min_travel` guard, then the sweep resumes at its next waypoint. Only rays divert: only an
  `open`-visibility human raises a far-field human ray (4), so a casualty in a car, a building or
  under rubble is a `vehicle_toppled`/`building_damaged`/`debris` ray this baseline sweeps past —
  the gap a learned policy has to close. Segments do not divert either: at `segment_scale = 40` a
  body is absorbed by its neighbours and a persistent region with no resolution rule livelocks (12).
- `OraclePolicy(privileged)`: holds while an unfound casualty is inside the camera within the depth
  limit (finding is by hit count, so dwelling is the productive action), else the token closest to
  the nearest unfound casualty — claimed greedily, nearest-first, in robot-index order.
- `OracleAssignPolicy(privileged)` ("oracle_assign"): `OraclePolicy` with the greedy claim replaced
  by the **optimal** assignment — `scipy.optimize.linear_sum_assignment` over the robots x unfound
  casualties straight-line distance matrix, re-solved from scratch every decision as casualties are
  found (`baselines.assign_casualties`). Same dwell rule, same mechanical channel to a casualty, so
  a row against `oracle` isolates the assignment and nothing else. More casualties than robots: the
  matching picks which ones. More robots than casualties: the matching leaves robots over and each
  takes the oracle's own no-claim line, its nearest unfound casualty. Deterministic.

**Privileged rows evaluate under `instant_confirm`** (`sim/config.py`): `found_hits=1`, `p_observe=1` for every visibility. The stochastic hit model emulates RayFronts confirmation and binds policies and heuristics; an oracle row bounds *planning* with perfect knowledge, so arrival confirms.


## 8. Metrics (owner: sim, `sim/metrics.py`)
Per episode: `time_to_first`, `time_to_half`, `time_to_all` (t_max if not reached), `finds_auc`
(normalised area under fraction-found vs t/t_max), `frac_found`, `n_found`, `dist_per_find`,
`dist_total`, `coverage_end`, `redundancy` (decision steps where ≥2 robots target within 20 m),
`n_decisions`, the per-container breakdown `found_by_container` / `n_by_container`
(`open`/`vehicle`/`building`/`rubble`, from `Human.container`) — the interesting split, because the
container decides whether a casualty can ever raise a ray — and the decentralised-execution
counters `redundant_cells` (episode total, summed over robots), `redundant_per_decision`,
`redundancy_frac` (**the same team mean of per-robot fractions the reward charges**, before the
refund, averaged over decisions — not the pooled ratio of the two totals), `redundancy_refunds`,
`intentional_revisits` (gross, before the refund), `revisit_refunds`, `revisit_penalties`
(`revisit_cost` x the *charged* revisits, net of the refunds), `visits`, `link_frac` (mean fraction of robot pairs in
radio contact per decision; the spawn hand-over is not a link) and `comms_range_m`. Cheap to compute
incrementally; exposed in `EnvState.metrics` and `info["metrics"]`.

## 9. Visualizer (owner: viz, `viz/`)
- `plot_scene(scene, ax=None, show_damage=True)` — GT map: roads/sidewalks/parks, buildings by
  fate (intact tan / damaged orange / destroyed dark red), debris brown, vehicles (intact blue /
  toppled magenta), bus stops, trees; casualties red with marker by container (o open, s vehicle,
  ^ building, x rubble), bystanders green; damage-field contours; legend.
- `plot_raster(raster, ax=None)` — class colormap with height shading.
- `render_frame(state: EnvState, query_idx=0, focus_robot=None, robot=None, show_local=False)
  -> np.ndarray[H,W,3]` — three panels (four with `show_local`): (1) GT + trajectories + robots
  (heading arrow, FoV cone, target line) + casualties found/unfound; (2) belief: unobserved dark,
  the chosen query's heatmap, segments as translucent patches outlined at the label edges in their
  **mean-feature colour** (PCA-RGB of `feat` on `EmbeddingTable.pc_basis`) plus their medoids,
  frontier clusters (cyan, size∝IG), live rays (segments from origin along bearing, colour = the
  same query's `ray_query_sim`), the focused robot's tokens numbered with the chosen one
  highlighted; (3) optional local crop — the focused robot's ego window as feature PC1-3 RGB
  masked by `known`, the dense input the actor plans in; (4) text: t, found/total, coverage,
  reward, the live query list, per-robot target.
  `query_idx` is an index into `state.query_names()` **or a name**; a name outside the mission list
  is embedded from `state.emb` and derived from the stored features.
  `robot=r` draws **robot r's own map** — its `RobotView` (known mask, its rays, frontiers and
  segments, the visited records it holds, its peer cache) — titled "robot r's map"; peer tokens
  become arrows from a peer's last-known position to its reported target, faded by contact age, and
  robots in contact are joined by a link line. Under `comms.mode == "full"` a robot's view *is* the
  team view and every pair is always linked, so neither differs from the union and no link is drawn.
  The query view is taken **once per frame** (`viz.frame.query_view`): `EnvState.vox_sim` allocates
  one grid per mission query and is never read by the visualizer.
- `EpisodeRecorder(env).record(policy, every_n_decisions=1)` ⇒ `save_mp4/save_gif/save_frames`.
- Scripts: `view_scene.py`, `play_episode.py --scene --policy --robots --query NAME|IDX --robot R
  --show-local --out`, `live_viewer.py` (same flags plus `--record`; keys `0-9` query, `f` focus,
  `v` cycle whose map the belief panel shows, `s` segments on/off), `rayfronts_demo.py`
  (single robot or `--robots N`, belief growing, `--query` by index or name),
  `episode_viewer.py` (matplotlib slider over a saved state sequence),
  `sensor_inspector.py` (four panels for one drone at a pose: GT classes with the actual cone
  footprint, the POV pinhole frame of the visible cells, the per-cell features as PC1-3 RGB plus
  the probed cell's `query_sim` bars for the live mission list and a free-text `--extra-query`, and
  the belief map; scene positional or `--scene`, interactive, `--out` for a headless PNG and
  `--sweep N` for a GIF of the belief building up).

**Accessors (changed with the open-set observation).** The belief no longer stores per-query grids,
so every query heatmap is taken *on demand*:
- `EnvState.vox_sim` is now a **lazily computed property**, not an array: it allocates
  `[Q, ny, nx]` cosines against the live mission queries on every access. Cheap for a frame, wrong
  in a loop — prefer `EnvState.query_sim(query)` for the one grid you are drawing. `query` may be a
  query *name*, an index into `EnvState.query_names()`, or a raw `[D]` vector.
- `RayFrontsSim.query_sim(query) -> [ny, nx]` and `RayFrontsSim.ray_query_sim(query, peak=True)
  -> [n_rays]` are the same views on the live belief; `RayFrontsSim.query_vec(query) -> [D]`
  resolves a query to its unit embedding. All three bump `n_query_calls` (a test asserts the
  per-step path never does).
- `RayStore` no longer has `sims` / `sims_max`; it has `feat` (weighted running mean) and
  `feat_peak` (most salient look). Use `ray_query_sim` for a per-ray colour.
- `EnvState.voxel_candidates` is gone. `EnvState.segments` (`SegmentToken`: `xy`, `ij`, `feat`,
  `n_cells`, `mean_hits`, `ray_count`, `t_first`, `t_last`) and `EnvState.seg_labels [ny, nx]`
  (segment index per cell, -1 = unobserved) replace it. `RayTarget.sims` is gone; `RayTarget.feat`
  is the bin's peak feature and `RayTarget.feat_mean` its running mean.
- `TOKEN_TYPE_NAMES` is read at call time, never hard-coded: `TOKEN_VOXEL` became
  `TOKEN_SEGMENT` and a `visited` type has since been appended. Markers: `.` hold, `o` frontier,
  `*` ray, `D` segment, `s` visited; a type the palette does not know falls back to its slot.
- The token feature block is `state.token_feature_names(D)` (a dim, not a query tuple); there is no
  `F_SIM0` and no `sim:<query>` column. `EmbeddingTable.project(feat, k)` gives the same PCA
  coordinates the BEV and the local crop use, if a viewer wants to draw features as colour.
- `TeamObs` gained `query_emb`, `query_w`, `query_mask`, `local`, `peer_tokens` and `robot_bev`
  (None unless `tokens.robot_bev_size > 0`); `sim.tokens.BEV_CHANNELS` (23) and
  `sim.tokens.LOCAL_CHANNELS` (12) name the raster channels — both gained a trailing `visited`
  channel, so no existing channel index moved.
- `RobotView` gained `feat_known`, `visited`, `peers`, `id_offset`, `coverage`, `robot` and
  `seg_labels`, all with defaults; `RobotView.fknown` is `feat_known or known`, and
  `team_view(rf, visited=())` takes the record list. `EnvState.robot_views` is the per-robot list
  under range comms and None under full comms; `EnvState.visits` is the env-wide list of
  `state.VisitRecord` in **both** modes and `EnvState.comms_links [n, n]` is who reached whom this
  decision (all-True off-diagonal under full comms). `DisasterEnv.comms` (a `CommsSim` or None)
  owns the beliefs, `CommsSim.last_links` keeps the same matrix and `DisasterEnv.visits` is the
  same record list.
- `sim.config.PERSON_QUERY_IDX` and `similarity_table.person_query_indices` /
  `PERSON_QUERY_NAMES` are removed: there is no privileged query column any more. A viewer that
  wants "the person heatmap" should ask for `state.query_sim("person lying on the ground")`.
- Per-robot views come from `state.robot_views` / `rf.robot_view(r)` when the simulator publishes
  them, else from the gossip layer's own beliefs; failing both, the team view stands in (comms
  full). Every extra a `RobotView` grows (`visited`, `peers`, `seg_labels`) is read with `getattr`,
  so the visualizer degrades instead of breaking while the sim moves.
All plotting must work headless (`MPLBACKEND=Agg`) and on the synthetic scene.

## 10. Scene export (owner: scene, `scene/gen/`, `scene/export.py`, `scene/casualties.py`)
Vendored, pxr-free copy of the generator's layout/packing/disaster code producing `Scene` JSON
(schema v0.1) for `preset × seed` without Isaac or Nucleus (fallback sizes, `measure=False`).
Casualty/bystander placement per §4 of `rl_scoping.md` (destroyed > damaged > intact buildings,
toppled cars, bus stops in the zone, prone pedestrians by damage; bystanders mostly outside).
CLI: `scripts/export_scenes.py --preset earthquake --seeds 0:100 --out data/scenes`.
`export_scene(..., pipeline="v2", disaster=…)` runs the generator's detailed-city pipeline
(`gen/generate_scene.py`: anisotropic subdivision with per-corridor widths/parking/classes,
district zoning by typology, infill, park superblocks) instead of plain `build_city`; the passes
that only write USD (street furniture, road surface, markings) are skipped. Block typology and
building category come from the districts pass — tower → commercial/highrise, midrise →
mixed/midrise, rowhouse → residential/house, park → park, unzoned → other — and v2 buildings
carry no explicit height (nothing is measured without Nucleus), so the sim resolves
`DEFAULT_HEIGHT_M[category] × FATE_HEIGHT_SCALE`. `meta.generator_version = "v2"`,
`meta.preset = "v2:<locale>:<disaster>"`.
CLI: `--pipeline v2 --locale downtown --disaster earthquake tornado explosion
--severity-range 0.5 1.0 --seeds 0:60 --region-range 500 1500 --size-jitter 0.25
--casualties auto --bystanders auto --out data/scenes_v2`; `--region-range` samples W and H
independently per seed (multiples of 10 m inside the range, non-square), `--severity-range`
samples per (seed, disaster). `--casualties auto` scales the count with the region area —
15 per 400x400 m (v1's density), clipped to [10, 80] — and `--bystanders auto` is half of it;
integers still override. Every scene exports **8** `robots_spawn` points on the corridor
nearest `--spawn-corner`, 6 m apart along it (second file across the corridor if it is too
short); the first four are exactly what four robots always got.

## 11. Testing policy
`uv run pytest -n auto`. Edge cases every tester should try: empty scene; no casualties; all
casualties occluded; spawn inside a building; target outside region; unreachable target (walled
in); fewer candidates than slots; 1 and 10 robots; tiny (20×20 m) and huge (1600×1200 m) regions;
`cell_m` ∈ {1, 2, 5}; `depth_limit > visual_range` rejected; severity 0; humans on the region
border; duplicate ids rejected; determinism (same seed ⇒ identical EnvState); no NaN/inf in obs;
reward accounting (no double count, bystanders give 0); rays resolve; frontiers empty ⇒ episode still
terminates; LoS blocked by a building in between, not by lower objects; frustum edges; a hidden
human (in a car, under rubble) never reaching a ray; the ray range against the true distance.
Open-set specifics: **no query is evaluated during a decision** (`RayFrontsSim.n_query_calls` must
not move across `step`); token features finite and bounded and the `feat` tail unit-norm or zero;
segmentation sanity (two class blobs ⇒ two segments, a uniform map ⇒ one, a segment never contains
an unobserved cell); the query block padded and masked for 1 and for `Qmax` queries; `set_queries`
changes the query block and nothing else (tokens, masks, xy, types, ids, robot features, BEV and
local crop must come back byte-identical).
Speed target (`scripts/bench_env.py`): ≥ 200 decisions/s (≥ 1000 sub-steps/s) for 3 robots on a
240×240 m scene at 2 m cells, single process, after numba warm-up.

## 12. As-built notes (deviations accepted after review)
Defaults changed from the first draft, all for physical consistency: `sensor.depth_limit_m` 20→35
(a *slant* range, must exceed `flight_alt_m`=25, validated); `sensor.pitch_deg=-50, vfov_deg=80` so
the elevation wedge is [-90°, -10°] — nadir is covered (no blind disk under the robot, no permanent
shadow strip at building feet) while the far field is kept; `p_fp_ray` 0.02→0.005 (~3–5 spurious
person rays per 3-robot episode). `RayStore.sims_max` (per-bin max, so one salient look at a real
far human is not averaged away by the background behind it) was removed with the query columns; the
same job is now done by `feat_peak`, which is what a ray *token* carries.

Simulator implementation choices that differ from §3–§5 prose, kept because they are either
observationally equivalent or fix a degenerate behaviour:
- Ray bins: within a sub-step, far cells sharing (origin cell, azimuth bin) are combined by max over
  class rows with one noise draw; the weighted running mean merges sub-steps (a per-cell mean buried
  a far person under the ~50 background cells in the same 20° bin).
- Frontier clusters: 8-connected components are split into blocks of ~the IG radius and the target
  is the medoid (nearest member cell to the centroid), never the centroid (which usually lies in
  already-observed space with zero gain).
- Ray resolution and frontier extraction run once per decision (policy only reads them then).
- A ray bin's `az`/`el` are the direction of its most salient observation, rewritten only when an
  observation at least as salient arrives; the elevation is a weighted mean over the cells *of that
  class*, so a far person's ray points at the person and `alt / tan(-el)` recovers its range to
  within a few percent instead of the average of everything in the 20° bin.
- Rasterisation always paints at least the centre cell of an object (a 4.5 m car at 2 m cells cannot
  vanish). `reachable` uses 8-connected free-space labelling (identical to 8-connected A* success).
- `visible_cells` Bernoulli-thins far candidates before the LoS test (same distribution, ~3× cheaper).
- Frustum is an azimuth/elevation wedge, not a pinhole rectangle.
- Baselines may skip frontier tokens closer than a footprint radius (`min_travel_m`) to avoid
  crawling on the ring of tiny frontiers around the robot's own footprint.
- Anti-livelock is structural, not bookkeeping: a ray resolves once the disc around the point it
  aims at is observed and a frontier disappears when its unknown neighbour is mapped. Nothing keeps
  a per-robot ban list. **Segments have no such escape**: a segment is a persistent region of the
  observed map, so a greedy threshold-free heuristic over segment tokens can re-select the same
  neighbourhood for ever (measured below). Giving them a revisit rule would be the kind of hand rule
  the open-set change removed; a learned policy has to move on by itself.

Embedding pass (`sim/embeddings.py`), all documented at the call site:
- a ray bin's per-sub-step semantics is the embedding of the **most salient** class along that
  bearing (highest `raster.CLASS_PRIORITY`, ties by cell count), not an elementwise max over class
  rows — a max has no meaning in feature space. Humans outrank everything, so a far *open* casualty
  owns its ray; one inside a car or under rubble never enters the far-field set at all, and the
  container's own class owns the bearing.
- one look is ~7% low (normalising a noisy unit vector shrinks its projection) and the estimate
  converges to the class row as a cell is re-observed, where the old per-query mean stayed noisy.
- memory per env at 750x750 cells (1500 m at 2 m): `vox_feat_sum` 54.0 MB at D=24 float32, whole
  belief 62.4 MB with the per-query grid gone and `seg_labels` added; `embedding_dim=16` brings it
  to 44 MB.
- the shipped `sim/data/text_embeddings_siglip_vitb16.json` is opt-in via
  `rayfronts.embeddings_path`: real SigLIP cosines compress into ~[-0.5, 0.8] after mean-centring
  and top out near 0.56 on the person queries, so a threshold calibrated on the hand table does not
  transfer and the default belief stays the factorized hand table.

Further as-built notes (sim QA pass):
- Human LoS target is `(x, y, min(cell height, z + 0.5))` — the body top, never above the surface
  covering it; a roofed casualty is visible only from (nearly) overhead, a rubble casualty from
  any angle that clears the debris but with the occluded observation probability.
- `finds_auc` integrates over the full horizon with the final fraction held after termination
  (finishing early can only raise it). There is no pursuit metric any more: with no target records
  there is nothing to call a pursuit, and `found_by_container` measures the same thing honestly
  (a casualty in a car or under rubble can only be found by going there).
- A scene with no casualties runs to `t_max`. Coverage everywhere (`info`, metrics,
  `EnvState.coverage`) is observed / observable; `EnvState.raw_coverage` is the plain mean. The
  only unobservable cells in shipped scenes are intact midrise roof interiors (tornado 4–6%).
- Frontier guard: baselines skip frontier tokens within half the sensor footprint radius and fall
  back to the farthest frontier if none remain (the 90° hfov always leaves an unobserved wedge
  beside the robot, so the nearest frontier is otherwise 1–2 m away).
- Find ceiling: with `p_observe_base["occluded"] = 0.15`, `found_hits = 2` and the overhead-only
  window for roofed casualties, even the (dwelling) oracle saturates near 0.85–0.9 found at t_max
  600 s; use `finds_auc` / `time_to_half` as primary metrics, `time_to_all` as a diagnostic.
  Finding now requires the robot to *look* at the person for long enough, so `hold` is a real
  action rather than a no-op.
- numba `cache=True` does not invalidate on edits to `inline="always"` callees — delete
  `src/rlplanner/sim/__pycache__` after touching `geometry.py` kernels.

Second sim QA pass (2026-08-21):
- Token features are saturated where the normaliser is nominal (`n_cells`, `age`, `n_obs`);
  measured ranges before saturation were `age` up to 3.2 (a frontier cell last seen ten minutes ago
  against a 300 s TTL) and `n_obs` up to 1.7.
- A ray's `conf`/`n_obs` count looks, not far cells: per-cell accumulation put every ray at
  `ray_conf_cap` after two sub-steps and at the `n_obs` ceiling by decision five, so two columns of
  the observation were constant 1.0 for a whole episode.
- Two bodies in one cell are two voxel observations of it, one per class row, so each is counted
  against its own `human_prone`/`human_standing` row (a single row per cell credited a prone
  casualty with a standing body's observation).
- A `found` event payload carries `container` and `visibility` alongside `casualty`.
- `token_xy` is NaN exactly on the *empty* slots. A slot holding a candidate the robot cannot reach
  keeps its coordinates and is masked instead: the mask says selectable, NaN says empty.
- Baselines fall back to the first *selectable* slot rather than index 0, which is the hold token
  only while `tokens.include_hold`; `OraclePolicy` uses the belief's own human LoS point
  (`min(z + 0.5, cell height)`), so it does not dwell on a body the sensor cannot see.

Semantic embeddings (as built, 2026-08-21):
- Every observed cell accumulates a unit-norm embedding `vox_feat_sum[ny, nx, D]` (D =
  `rayfronts.embedding_dim`, default 24); each observation is `normalize(class_emb[cls] +
  N(0, feat_noise_std))` with `p_confuse` mixing in another class. Rays carry `feat` (running mean)
  and `feat_peak`. Nothing derives a per-query cache any more (see the open-set notes below).
- Embedding sources: default = factorization of the hand-authored class×query table (exact for
  D ≥ 12; zero-padded to `embedding_dim`); opt-in SigLIP cache via `rayfronts.embeddings_path`
  (`scripts/build_text_embeddings.py`, `--extra embed`); a cache of another D is refused.
  The SigLIP cache runs but is uncalibrated: cosines compress to [−0.56, 0.80], so the baselines'
  person-similarity thresholds do not transfer and it must not be used for training until they are
  recalibrated.
- The mission query set is runtime-mutable through `DisasterEnv.set_queries(names, weights=None)`.
  It rebuilds only the query tokens: the observation width is `TOKEN_FIXED + embedding_dim`, so a
  trained network survives the switch and `RayFrontsSim.set_queries` alone is no longer a problem.
  Nothing privileges the person queries, so a query list without one is legal.
- `scripts/sensor_inspector.py`: cone footprint / POV projection / PCA-RGB embeddings / belief
  heatmap; `--pose` is validated (inside region, not in a building, altitude below the depth limit).

Open-set observation (as built, 2026-08-21) — the change that removed the query scan:
- **Why**: the sim scanned every observed cell against all eleven configured queries every sub-step
  and ranked candidates by those scores. That is a closed set: it fixes what the map may contain,
  makes the observation width and every trained network a function of the query list, and throws
  away exactly what RayFronts is for. Now the belief stores only features, the policy receives the
  embeddings plus the mission queries as separate inputs, and relevance is learned.
- `vox_sim` as a stored grid, `RayStore.sims`/`sims_max`, `RayTarget.sims`, `VoxelCandidate`, the
  `sim:<query>` token columns, the per-query BEV channels, `PERSON_QUERY_IDX`,
  `person_query_indices` and the `voxel_seeker` baseline are all gone. `EnvState.vox_sim` survives
  only as a lazily computed property for viewers (§9).
- A ray **token** carries `feat_peak`, not the running mean. The mean over sub-steps buries a far
  casualty under the background looks that share its 20° bin (this is why `sims_max` existed), and
  the peak is the observation the token's own `az/el` describe. `RayTarget.feat_mean` keeps the mean.
- Candidate order is recency. Frontiers are newest-first then by size, rays and segments
  newest-first; `info_gain` is still computed for the visualizer but ranks nothing, and the
  farthest-point diversity pass over frontiers is gone.
- **Segments** replace "salient voxels". Felzenszwalb over the 4-neighbour grid of observed cells,
  weight `1 - cos`, bucket-sorted (2048 buckets over [0, 2]) so the pass is linear and
  deterministic. `segment_scale = 40` cells, `segment_min_cells = 4`. Measured on the bench scene
  (240×240 m at 2 m = 120×120 cells) at 96% coverage: **0.75 ms** per full pass, 69 segments; on a
  fully observed 750×750 belief (1500 m at 2 m) with random features: **75 ms**. Per-decision
  segmentation would therefore cost more than the whole 5 ms decision budget on a big map, so the
  labelling is re-run only every `segment_refresh_frac` (5%) of new observed cells or every
  `segment_refresh_decisions` (25) decisions, while the per-segment statistics run every decision
  on the cached labels (one bincount-style sweep). Amortised, segmentation is 9% of a decision on
  the bench scene.
- Scale-parameter behaviour, measured on a 120×120 synthetic belief: with `k = 40` a uniform
  (single-class) map collapses to 1 segment and two classes side by side give exactly 2; the cost is
  that a *single* cell (a body at 2 m cells) is absorbed by its neighbours, because
  `tau(C) = k/|C|` is 40 for a singleton and no edge weight can exceed 2. Values in the 1–4 range
  keep a lone person cell as its own 4–6 cell segment but over-segment a uniform surface into
  ~100 pieces at that grid size. 40 is the shipped default (per the change brief); the consequence
  is that segment tokens describe class-scale structure (rubble fields, car clusters, roofs) and a
  far casualty reaches the policy through the ray topic, not through a segment.
- `bev` lost the `height` channel: it was `raster.height`, i.e. ground truth, not belief.
- Token features are saturated where the normaliser is nominal (`n_cells`, `hits`, `age`, `n_obs`,
  `ray_count`); the `feat` tail is unit-norm or exactly zero (a frontier with nothing observed
  within 10 m).
- Throughput on the bench scene (3 robots, 240×240 m at 2 m, `ray_follower`): **202–210 dec/s** with
  the 64×64 local crop on, **237 dec/s** with `tokens.local_size = 0`. Both clear the 200 dec/s
  target; `scripts/train.py` sets `local_size = 0` unless `--use-local`, so nothing is built or
  shipped through the worker pipes that the network does not read. Breakdown at `local_size=64`:
  sensor 18%, rayfronts 44% (voxels 13, segments 9, rays 8, frontiers 5, ray targets 2.5, resolve
  2.5, humans 5), tokens 30%, motion 1%, other 7%.
- The token builder is per-robot by signature (`RobotView`), and everything robot-independent is
  built once per distinct view (`_pack`), so K rising from 33 to 97 slots costs ~1 ms, not ~4.
- **Baseline ordering on 20 synthetic episodes** (3 robots, 600 s, `finds_auc`):
  oracle 0.77 > ray_follower 0.35 > nearest_frontier 0.28 ≈ random 0.27 >> segment_seeker 0.02.
  `segment_seeker` is **not** ≥ `nearest_frontier`, and this is reported rather than tuned away.
  It livelocks: a segment is a persistent region of the map with no revisit suppression (the voxel
  candidates it replaced vanished within `revisit_m` of a robot), so a threshold-free argmax keeps
  re-selecting the same neighbourhood and coverage stops at 0.09 against 0.98 for a sweep. Adding a
  revisit rule would be exactly the kind of hand rule this change removed; a learned policy is free
  to move on, and that is the point of the comparison.

Decentralised execution (as built, 2026-08-21) — `sim/comms.py`, DESIGN_VARIANTS.md G:
- The gossip layer is a *simulation* of the fleet's radio, not a message bus: one contact per
  decision, range + relay, small typed payloads, everything persists on the receiver.
- What stays global and why: `vox_feat_sum` (a cell's feature is the same whoever looked at it;
  per-robot copies would be 54 MB each at 750x750), `vox_cnt` (same trade) and the ray table (rows
  are ids). A robot only ever reads them through its own `feat_known` mask, and its rays through
  its own ownership bit and its own resolution flags.
- Who observed what is one pair of `uint16` bitmask grids on `RayFrontsSim` (`seen_by` = before
  this decision, `dec_by` = during it), not a mask per robot: it costs one indexed write per robot
  per sub-step and it is what both the per-robot `known` masks and the redundancy term read.
- `_extract_frontiers` / `_extract_segments` moved into `FrontierIndex` / `SegmentIndex` so a robot
  can run them on its own masks; `RayFrontsSim` keeps the old attribute names (`seg_labels`,
  `_seg_obs_at`, `frontier_clusters`, …) as delegating properties.
- Under range comms `RayFrontsSim.compact()` is disabled (`keep_rays`), so resolved rays stay in
  the table and a row index equals a ray id. Cost: a few hundred kB of dead rays per episode.
- Throughput on the bench scene (3 robots, 240x240 m at 2 m, `ray_follower`, machine idle):
  **full comms 191 dec/s** with the 64x64 local crop, **214 dec/s** with `tokens.local_size = 0`
  (`scripts/train.py`'s default unless `--use-local`); **range comms @ 200 m 82 dec/s**, 74 dec/s
  with a 32x32 per-robot BEV as well. The 200 dec/s target of section 11 is therefore missed by 4%
  in the local-crop configuration, and the cost is the per-decision bookkeeping the brief asks for,
  not the visited token block: peer tokens ~0.7%, the per-robot observation tracking ~1.9%, the
  visited raster stamping ~1.1%, `tokens.k_visited = 0` buys nothing back. Range comms is ~2.3x a
  full-comms decision: the gossip exchange itself is small, the cost is running the frontier,
  segment and ray extraction once per robot instead of once per team.
- Redundancy is a **fraction, averaged over the team** (`redundancy_cost * mean_r(redundant_r /
  observed_r)`), not a per-cell charge. Measured: `ray_follower` on the bench scene re-covers 74%
  of what it looks at (`nearest_frontier` 44%), so the term costs 0.037/decision against the
  0.01 time cost; on `data/scenes_v2/downtown_tornado_17.json` with 7 robots (384 cells observed
  per robot per decision, 252 of them redundant) 60 decisions with no find cost **-3.26** in total
  (-0.60 time, -2.19 redundancy, -0.50 one revisit). Counting cells instead cost -21.66 on the
  same run and grew with the team size and the cell size; the fraction does not.
- Intentional revisits on the synthetic 3-robot bench: `nearest_frontier` 0.0 per episode (it never
  targets a ray or a segment, so it writes no records), `ray_follower` 3.3 per episode = -1.67, and
  a 5-update (i.e. barely trained) policy 37 per episode = -19, because a quarter of its token
  slots are visited records and it still picks them at random. That is the term doing its job, but
  it is the biggest single number in an early episode; watch it in the sweep.
  `scripts/run_baselines.py` prints `redundancy_frac`, `intentional_revisits` and
  `revisit_penalties` per episode.
