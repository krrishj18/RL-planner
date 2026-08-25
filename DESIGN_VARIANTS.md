# Design variants and reward spec (2026-08-21)

Principle: the policy consumes RayFronts' raw outputs (embeddings, rays, frontiers) and *learns* what matters —
including which information from other drones matters. So the simulator exposes every sharing/observation option
behind flags, we train the variants as a sweep, and ablations decide. Execution is decentralized (each drone runs the
policy on its own map + gossip); training is centralized (critic sees the union).

## A. What a drone observes (actor input)
| flag | option | default |
|---|---|---|
| `obs.local_crop` | dense ego-centric crop (feature PCs, hits, observed, ray raster) | on, 64×64 cells |
| `obs.global_bev` | coarse BEV over the region of what this drone knows | on, 64×64 |
| `obs.tokens` | own frontiers / rays / segments, newest-first caps | 32 / 32 / 32 |
| `obs.query_tokens` | mission query + hint embeddings with weights | on |
| `obs.peer_tokens` | one token per peer (see B) | on |
Variants to train: tokens-only · tokens+local · tokens+local+global (full).
Segment granularity (`segment_scale`): 40 (class-scale regions; a single 2 m cell is always absorbed, so a near
person reaches the policy via the local crop / rays) · 4 (keeps single-cell items, over-segments surfaces) · both
(two segment budgets, coarse + fine). Ablate; the model decides what it uses.

## B. What drones share (gossip, range-limited with relay; everything persists on the receiver)
Always: position, current target (type + xy), last-contact age, reported coverage fraction.
| flag | payload | ablation values |
|---|---|---|
| `share.rays` | peer rays as tokens/raster | none · newest N · all (default all, capped) |
| `share.coverage` | coarse known/observed grid (feeds the receiver's global BEV + frontiers) | off · on (default on) |
| `share.segments` | peer segment tokens (mean feature, extent, hits) | off · all (default all, capped) |
| `share.visited` | **visited-target records** (see C) | off · on (default on) |
| `share.features` | raw voxel features for shared cells | off (bandwidth) · on (ablation only) |
| `comms.range_m` | link range, multi-hop relay | 100 · 200 · ∞ (randomized in training) |
Default = share everything within range ("include everything for now"); the sweep removes one payload at a time.

## C. Visited targets
A drone *visits* a target when it chose that token (ray / segment) as its current target and arrived within
`arrive_radius_m`. Record = (xy, token type, feature, time, casualties found there, drone id). Records are shared in
gossip and appear to peers as (a) visited tokens and (b) a BEV channel. This is bookkeeping of the drone's own
actions, not inference.

## D. Reward (team reward shared by all robots; per-robot terms credited to the robot that caused them)
| term | value | notes |
|---|---|---|
| casualty found | +1.0 | first time its cell is voxel-observed with the human row ≥ `found_hits` |
| time | −0.01 / decision | unchanged |
| **redundant coverage** | −`c_red` × mean over robots of (cells this robot observed this decision that a peer had already observed ÷ cells it observed) | `c_red` = 0.05: a decision in which the whole team re-covers known ground costs exactly 0.05, whatever the footprint, the cell size or the team size. **Refunded** (that robot's fraction set to 0) if it found a casualty in a redundant cell during the same decision. Ablation: off. |
| **intentional revisit** | −`c_revisit` | large: 0.5. Applies when a robot *arrives at a target it chose* that lies within `revisit_m` of a target another drone visited earlier. Not applied for incidental fly-bys (the robot's current target was elsewhere). Default `known_only=True`: only if the visiting drone had received that visited record (otherwise it is punished for ignorance it could not avoid); ablation: `known_only=False`. **Refunded** (`revisit_refund_on_find`, default on) if that arrival turns up a new casualty within `revisit_m` of the target — the same principle as the redundancy refund: a revisit that finds somebody was not a wasted journey. The charge stays open while the drone is still at the target (a find needs `found_hits` looks), so the refund can land a decision or two later; the episode total nets out. Ablation: off. |
Variants: with/without redundancy term · with/without revisit term · both.

## E. Variants to train first (each a config file under `configs/variants/`)
1. `central_full` — shared team map, full comms (today's reference)
2. `decentral_share_all` — per-drone maps, share everything, range randomized 100–∞
3. `decentral_share_pos_cov` — positions/targets + coverage only (no rays, no segments, no visited)
4. `decentral_share_rays` — positions/targets + rays only
5. `decentral_blackout` — positions only at spawn, no further sharing (lower bound)
6. `decentral_share_all_noreward` — as 2 without redundancy/revisit terms
7. `decentral_share_all_tokens_only` — as 2 without local crop / global BEV
Metrics per variant (held-out scenes, 16+ episodes, 95% CI): found, finds-AUC, time-to-first/half, redundancy rate,
intentional revisits, coverage, dist/find, by-container finds; plus the same under comms range 100 / 200 / ∞ at eval.

## F. Training plan
Synthetic 240×240 (3 robots, 600 s) for the sweep (≈30–40 min per variant at ~1k decisions/s), then the best 2 variants on
the v2 downtown set with area-scaled teams/horizons.

## G. As built (2026-08-21)
Everything above is implemented behind flags; defaults share everything. What the code decided
where the spec left a choice (all documented at the call site, and in CONTRACTS.md 5.1 / 6):
- `comms.mode = "full" | "range"`; in `range` each robot owns a `RobotBelief`. The feature map and
  the hit counts stay **global** and a robot reads them through its own `feat_known` mask (a
  per-robot copy of `vox_feat_sum` is 54 MB at 750x750); everything else — `known`, frontiers,
  segments, rays, peers, records — is private. Private state is 3.9 MB per robot at 750x750.
- Contact is evaluated **once per decision**, at the end-of-decision positions. `range_m <= 0` is a
  blackout (no link at any distance), `relay_hops = 0` is the whole connected component.
- Shared coverage is conservative: a coarse cell is sent only when *every* fine cell in it is known
  to the sender. On the receiver it sets `known` but not `feat_known`, so the cell stops pulling
  exploration while its hit and feature channels stay zero.
- Rays travel as **snapshots**: the receiver keeps the bin as it was at contact, and resolves it
  against its own map. Rays and segments are the sender's own; visited records are everything the
  sender knows (the epidemic is the point of a record).
- Visited records are written for **ray and segment** arrivals only (a frontier disappears when it
  is mapped, so it cannot be revisited); the record's `n_found` counts the casualties *that robot*
  found within `revisit_m` during the decision it arrived in.
- The revisit penalty is charged on arrival at the chosen target (any type), never on proximity,
  and only when the robot was outside `arrive_radius_m` of it when the decision started — sitting
  on a target and re-selecting it is not a journey (it would otherwise cost 0.5 per decision and
  fill the record list with copies). It is **refunded** when that arrival finds a new casualty
  within `revisit_m` (see D); `info` and the metrics keep the gross `intentional_revisits` and the
  `revisit_refunds` apart, and `revisit_penalties` is the net charge.
- `policy.sequential_decode` (CONTRACTS.md 6) follows the execution model: the same-decision claim
  mask is a centralised-execution device, so only `central_full` decodes sequentially. Every
  `decentral_*` variant decodes independently — two robots may pick the same token in one decision,
  and a peer's intent reaches them only through gossip, one decision later.
- The sweep evaluates every variant at the common ranges *and* at its own training comms setting
  (`eval.own`: blackout at range 0, `central_full` on one shared belief, the share_* variants on
  their randomised range). The common ranges keep the rows comparable but hand the blackout a radio
  it never had; the payload flags are always the variant's own, at every range.
- Both new terms are per robot and **added to the team reward** (MAPPO shares one return); `info`
  and the metrics carry them per robot.
- Token vocabulary grew a `visited` type (K = 129 slots, `TOKEN_FIXED` 28 -> 32); peer tokens are
  14 wide; the actor's own BEV is `tokens.robot_bev_size` = 32 cells (64 would be 36 MB per
  decision over the worker pipes at 32 envs x 3 robots).

**Redundancy is a fraction, not a cell count.** Counting cells made the term scale with the
footprint, the cell size and the team: 2000+ cells per robot per decision on a v2 scene put a
7-robot `ray_follower` at -21.66 after 60 decisions with no finds. It is now
`c_red x mean_r(redundant_r / observed_r)`, so one fully redundant decision costs `c_red` = 0.05
and never more, and the average (rather than the sum) keeps it independent of the team size.

**Measured magnitudes** (`ray_follower`, no learning):
- synthetic 240x240 m, 3 robots: redundant fraction 0.74 (`nearest_frontier` 0.44) => 0.037 per
  decision against the 0.01 time cost; 3.3 intentional revisits per 600 s episode => -1.67.
  `nearest_frontier` never writes a record (it only takes frontier tokens) and pays 0 revisits.
- a 5-update policy (essentially untrained) pays 37 revisits per episode = -19: a quarter of its
  token slots are visited records and it picks them at random. The signal is learnable and the
  ablation is `decentral_share_all_noreward`, but it is the term to watch first in the sweep.
- `downtown_tornado_17` v2, 7 robots, 60 decisions, 0 finds: **-3.26** total = -0.60 time,
  -2.19 redundancy (fraction 0.73), -0.50 one revisit.
So the two terms now sit a few times above the time cost and well below a find (+1.0).
`decentral_share_all_noreward` is the ablation that measures whether they help at all.

## H. QA pass (2026-08-21) — what changed and what the numbers say

Fixes (all in CONTRACTS.md 5 / 5.1 / 7 / 8, tests in `tests/test_sim_comms.py`,
`tests/test_sim_rewards.py`, `tests/test_train_decentral.py`):
- a token's `peer_dist_min` / `claimed_by_peer` came from the **true** robot list, so a robot under
  blackout still read its peers' exact positions off its own observation. They now come from its
  peer cache.
- the arbitration of the hand-coded baselines keys on the target **position** as well as on
  `(type, id)`. Under range comms the id key can never match across robots (per-robot id
  namespaces), so it did nothing: `ray_follower` with three robots from one spawn picked one token
  for the whole team in 35 of 40 decisions and covered 0.55 against 0.92 with the fix. The same
  hole was open for the learned policy — the sequential-decode claim mask is keyed on `(type, id)`
  too — and it is now closed the other way: the mask is a centralised-execution device, so it is
  **off** for every decentral variant (`policy.sequential_decode`). Under decentralised execution
  nothing de-conflicts the team except what the policy learns from `claimed_by_peer` and the peer
  tokens, and the training no longer pretends otherwise.
- a ray a robot received as a peer snapshot *and* later fed itself appeared twice, with one id.
- a payload could cross one hop more than `relay_hops` allows inside a single exchange.
- the ray and segment inboxes grew for the whole episode (5300 dead ray snapshots per robot by
  decision 180) and are now pruned.
- `link_frac` counted the spawn hand-over as a radio link (blackout read 1/n_decisions, not 0);
  the metric also lagged one decision behind the reward.
- `redundancy_frac` reported the pooled ratio of the totals, not the team mean of per-robot
  fractions the reward actually charges.
- the sweep scored every variant with **its own** reward, so `decentral_share_all_noreward` was not
  charged for the terms it ablates: it read -1.20 where the common yardstick says -48.23. Every row
  is now scored with the default reward.

**The revisit term is the thing to watch.** On synthetic 240x240, 3 robots, 600 s (6 episodes),
before and after `reward.revisit_refund_on_find` (`scripts/run_baselines.py --episodes 6`):

| policy | reward before -> after | found | revisits | penalty before -> after | redundancy frac |
|---|---|---|---|---|---|
| nearest_frontier | +0.95 -> +0.95 | 0.42 | 0.0 | 0.00 -> 0.00 | 0.48 |
| ray_follower | -1.12 -> **-1.03** | 0.49 | 3.0 | -1.50 -> **-1.42** | 0.72 |
| random | -3.24 -> -3.24 | 0.39 | 3.7 | -1.83 -> -1.83 | 0.82 |
| segment_seeker | -12.91 -> -12.91 | 0.00 | 13.0 | -6.50 -> -6.50 | 0.87 |
| **oracle** | **-35.11 -> -34.86** | **0.85** | **84.3** | **-42.17 -> -41.92** | 0.46 |

Under range comms at 200 m the same six episodes give oracle -45.65 -> -45.40 (penalty -52.25 ->
-52.00, 104.5 revisits), ray_follower -3.25 (unchanged, 5.8 revisits), random -10.75, nearest
+0.17, segment_seeker -7.72.

**The refund is right and it barely moves the number.** Over those 6 oracle episodes 506 revisits
are charged and 3 are refunded. Instrumenting every charged arrival: only 6 of 506 have a find by
the *arriving* robot within two decisions at any distance, and 2 within `revisit_m`. The reason is
in the token types of the charged arrivals — **354 of 506 are `visited` tokens**, 102 segments, 50
frontiers. The oracle picks the token closest to the nearest unfound casualty; a visited record
next to a casualty nobody can find (rubble, `p_observe` 0.15) *is* that token, so it flies to a
place the team already searched, finds nothing new, and pays 0.5. Those revisits are fruitless by
the term's own definition, and the refund correctly leaves them charged.

So the ordering of the table is unchanged and the term still outweighs the find budget ~7x for the
oracle. What is left, if the sweep confirms it hurts learning: scale `revisit_cost` to ~0.05, or
charge only when the record being repeated found nothing (both change the reward the pilot trained
on), or keep visited records out of the *candidate* set (they would stay an observation, not a
target) — the last one is a token-vocabulary change, not a reward change, and it is the one this
measurement points at.

## I. Sweep evaluation (2026-08-21)
Every variant is scored on the held-out split at the common ranges 100 / 200 / inf **and** at its
own training comms setting (`own` rows in `summary.md`, `eval` column in `summary.csv`):
`central_full` on one shared belief, `decentral_blackout` at range 0 (spawn exchange only), the
`share_*` variants on their randomised range. The common ranges keep the rows comparable but give
the blackout a radio it never had, so its `own` row is the one that describes the system that was
trained. At every range each variant keeps **its own payload flags** — `share_rays` never receives
coverage, `share_pos_cov` never receives rays or visited records, the blackout receives nothing
after spawn — and every row is scored with the default reward.


## H2. Sweep results (OSMO, 2026-08-24 — 7 variants × 300 PPO updates, synthetic, 16 held-out episodes/cell)

found @ comms range 100 / 200 / ∞ / own: central_full .57/.58/.58/.54 · share_all .56/.61/.60/.56 ·
pos_cov .56/.50/.55/.46 · share_rays .60/.60/.59/.63 · blackout .55/.50/.54/.53 ·
**share_all_noreward .67/.66/.66/.67** · tokens_only .57/.57/.54/.57. CI ≈ ±0.06/cell.

Findings: (1) the redundancy/revisit penalties cost ~0.06–0.10 found under a common yardstick — decide:
keep (deployment semantics), shrink (revisit 0.5 → 0.1), or refunds-only; (2) decentral ≈ central; blackout
only ~-0.05; (3) rays are the payload that matters (share_rays ≈ share_all > pos_cov); (4) dense inputs worth
a few points (tokens_only lowest of the share-all family); (5) noreward PPO-from-scratch (0.67) ≈ warm-start
(0.68) on synthetic — the earlier from-scratch plateau was partly the penalty terms.
Full tables with CI/AUC: /volume4/dsta/rl-planner/sweep_v2/runs/**/summary.md.

## H3 — v2 city-scale warm-start (`ws_v2_central`)

Warm-start on the v2 downtown distribution itself (DAgger 345k labels from `oracle_sweep`, BC 0.12–0.13
found, then PPO 200 updates). Held-out v2 eval, 12 episodes, identical protocol for all rows:

| policy | found | finds-AUC | t_first | reward |
|---|---|---|---|---|
| random | 0.02 | 0.02 | 584 s | −33 |
| segment_seeker | 0.01 | 0.01 | — | −29 |
| ray_follower | 0.11 ± 0.02 | 0.06 | 130 s | −5.7 |
| oracle (greedy, privileged) | 0.14 ± 0.05 | 0.12 | 40 s | −256 |
| nearest_frontier | 0.15 ± 0.03 | 0.08 | 106 s | +3.8 |
| **ft (DAgger→PPO)** | **0.19 ± 0.03** | **0.11 ± 0.02** | **68 s** | **+7.8** |

Reading: at city scale (0.3–2 km², coverage impossible in budget) the learned policy is the best method —
it clears nearest-frontier on found and AUC and beats the greedy privileged oracle, whose beeline behaviour
collapses (coverage 0.10, reward −256). Combined with H2's zero-shot collapse (0.06), the conclusion is that
the architecture transfers but the policy must be trained on the target scene distribution.
Artifacts: `/volume4/dsta/rl-planner/ws_v2_central/runs/ws_central_full/{bc,ft}/`.

## H4 — revisit penalty scoped to confirmed targets (2026-08-24)

H2/H3 showed the penalties suppress the lingering that confirming occluded casualties requires
(16–23 of 61 rubble-contained finds with penalties vs 31–33 without). Fix: the revisit charge now
applies only to visited records **whose visit confirmed a casualty** (`n_found > 0`,
`reward.revisit_confirmed_only`, default on). Rationale: a record marks "a drone went here", not
"this spot is done" — under stochastic occluded detection (p=0.15/sub-step, `found_hits=2`) a
teammate may have swept a debris pile and confirmed nothing, and going back is search, not waste;
returning to a *confirmed* find is unambiguously wasteful. `revisit_refund_on_find` stays as the
backstop for multi-casualty piles near a confirmed find. The gossip merge already prefers the
higher-`n_found` copy of a record, so confirmation propagates with the records themselves.

**Outcome** (`ws_v2_central2`, 36 CPU / 1 GPU, 63 min wall): BC 0.16 sampled (vs 0.12–0.13 under the
old rule), FT **0.21 ± 0.03 found / 0.12 AUC, reward +8.3** at update 200 vs 0.19 ± 0.03 / 0.11 / +7.8
before — found overlaps the 12-episode CI but the whole trajectory is stronger (the old final 0.19 was
matched by update 100). The greedy oracle's reward moves −256 → +0.9: its returns to unconfirmed
targets are search, not waste, and are no longer charged. Final 24-episode greedy/sampled CSVs:
`/volume4/dsta/rl-planner/ws_v2_central2/runs/ws_central_full/{bc,ft}/`.

## H5 — 8-robot v2 run, motion-only ideal, and the LLM hint ablation (2026-08-25)

`ws_v2_8robot`: 8 robots fixed, t_max auto ≤ 3000 s, dynamic queries ON, confirmed-only revisit.
Held-out v2, 24-episode finals (greedy): **learned 0.203 ± 0.017 found / 0.127 AUC, coverage 0.79,
reward +7.0** vs nearest 0.17, lawnmower 0.16 (lowest redundancy 0.39; finds every open casualty,
trails on containers), ray_follower 0.16, random 0.03. In-train peak 0.22 at update 150.

**`oracle_ideal`** (motion-only bound; obstacle-aware Dijkstra over the env's blocked mask,
arrival = visit): **1.00 found / 0.88 AUC, all casualties in ~11.1 km of the ~62 km budget** — the
horizon never binds; the whole gap below the ideal is search + confirmation. Progress = found/ideal
= found fraction on this bank. Token-space instant-confirm oracles (0.25–0.27) are bounded by the
token action interface, not time, and were dropped from headline charts in favour of the ideal.

**LLM hint-lift ablation** (6 heldout episodes/condition, live Claude backend, cadence 10):
none 0.20 · static 0.19 · scripted 0.20 · llm 0.19 — **no lift; the policy ignores the query
channel**. Expected on a closed 15-class sim: container correlations are fully learnable from raw
token features, so queries add no information (COS-POMDP: priors pay when semantics are open or
unreliable). The channel itself works end-to-end (189 list-changing LLM edits landed via
set_queries). To make hints bite: per-disaster casualty-container priors that vary across episodes
(the query channel becomes the only way to know which prior applies) and/or eval-time novel
container classes. CSVs: runs/llm_hints_8robot/, runs/oracle_ideal_8robot.csv.

## H6 — ws_v2_big: the PoC margin (2026-08-25)

600-scene bank (200 seeds x 3 disasters, severity 0.3-1.0, 0.3-2 km2), 8 robots, t_max auto <= 3000,
dynamic queries, confirmed-only revisit. DAgger 8x300 (614k labels, BC 0.20 sampled) -> PPO 500
updates (1.02M team decisions). 24 held-out episodes, identical (scene, seed) list for every row:

| policy | found | AUC | coverage |
|---|---|---|---|
| oracle_ideal (motion bound) | 1.00 | 0.88 | - |
| **learned (ws_v2_big)** | **0.293 +- 0.026** | **0.173** | 0.82 |
| lawnmower (waypoint) | 0.25 +- 0.03 | 0.15 | 0.80 |
| nearest_frontier | 0.22 +- 0.03 | 0.12 | 0.77 |
| ray_follower | 0.19 +- 0.02 | 0.10 | 0.48 |
| random | 0.06 | 0.04 | 0.12 |

**+33% found / +44% AUC over nearest-frontier with non-overlapping CIs**, and +17%/+15% over the
strongest scripted baseline (the strict vertical-strip waypoint lawnmower). BC alone 0.095 greedy:
the RL stage triples it. Wall: ~3.1 h on 32 CPU + 1 GPU. Artifacts:
/volume4/dsta/rl-planner/ws_v2_big/runs/; local bank data/scenes_v2_big (deterministic re-export).
