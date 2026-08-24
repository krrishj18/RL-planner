"""Hand-coded baselines (CONTRACTS.md 7). Each returns one token index per robot.

They are heuristics *over the observation*, not rules inside the environment: the similarity a
`ray_follower` ranks by is computed here, from the token's own `feat` and the query tokens the
observation carries. The simulator never scores anything against a query, so a baseline that wants
a query view has to take it itself — which is the point.

One line of arbitration each, no priority tree and no threshold: whichever candidate of the right
type answers the mission best wins, and if the type is not on offer the robot sweeps.
"""
from __future__ import annotations

import numpy as np

from ..scene import schema
from .config import DEFAULT_QUERIES
from .embeddings import get_embedding_table
from .sensor import human_visibility
from .state import (F_DIST, F_FEAT0, F_XABS, F_YABS, TOKEN_FRONTIER, TOKEN_HOLD, TOKEN_RAY,
                    TOKEN_SEGMENT, EnvState, TeamObs)

CLAIM_M = 15.0     # two robots flying to points this close are going to the same place
HUMAN_CLASSES = (schema.CLASS_ID["human_standing"], schema.CLASS_ID["human_prone"])


class _Claims:
    """What the lower-index robots already took this step (CONTRACTS.md 7).

    Keyed on the target *position* as well as the `(type, id)` pair: under range comms every robot
    numbers its own frontiers, rays and segments in its own namespace, so an id key can never match
    across robots and the arbitration would silently do nothing. With identical beliefs a
    threshold-free greedy baseline then picks one token for the whole team and the robots fly as
    one body (measured: `ray_follower`, 3 robots, one spawn, identical actions in 35 of 40
    decisions and 0.55 coverage against 0.82 under full comms).
    """
    __slots__ = ("ids", "pts")

    def __init__(self) -> None:
        self.ids: set[tuple[int, int]] = set()
        self.pts: list[tuple[float, float]] = []

    def has(self, key: tuple[int, int], xy) -> bool:
        if key in self.ids:
            return True
        x, y = float(xy[0]), float(xy[1])
        return any((x - u) ** 2 + (y - v) ** 2 <= CLAIM_M * CLAIM_M for u, v in self.pts)

    def add(self, key: tuple[int, int], xy) -> None:
        self.ids.add(key)
        self.pts.append((float(xy[0]), float(xy[1])))


class Policy:
    name = "policy"
    privileged = False

    min_travel_m: float | None = None

    def __init__(self, queries=DEFAULT_QUERIES, seed: int = 0):
        self.queries = tuple(queries)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: TeamObs, state: EnvState | None = None) -> np.ndarray:
        raise NotImplementedError

    # helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _valid(obs: TeamObs, r: int) -> np.ndarray:
        return np.flatnonzero(obs.token_mask[r])

    @staticmethod
    def _fallback(obs: TeamObs, r: int) -> int:
        """Slot 0 is `hold` only when `tokens.include_hold`; otherwise fall back to any valid slot."""
        v = np.flatnonzero(obs.token_mask[r])
        return int(v[0]) if v.size else 0

    def _min_travel(self, state) -> float:
        """Half the robot's own sensor footprint radius: a nearer target does not move the footprint.

        The wedge reaches nadir, but it is only `hfov_deg` wide, so the cells beside and behind a
        robot stay unobserved and always offer a frontier token a metre or two away. A greedy policy
        that takes it turns on the spot instead of sweeping (measured: 7% coverage in 600 s against
        99% with the guard). The full footprint radius is too strict — every frontier bordering the
        current footprint sits exactly on it — so half of it is the operating point.
        """
        if self.min_travel_m is not None:
            return self.min_travel_m
        if state is None:
            return 0.01        # fraction of the region diagonal, when only scaled features exist
        c = state.cfg
        return 0.5 * float(np.sqrt(max(c.sensor.depth_limit_m ** 2 - c.robot.flight_alt_m ** 2, 1.0)))

    @staticmethod
    def query_scores(obs: TeamObs, r: int) -> np.ndarray:
        """[K] best cosine of each token's feature against the observation's query tokens.

        The whole query-dependent part of a baseline lives here: the environment handed over
        embeddings, this is one matrix product on top of them.
        """
        q = obs.query_emb[obs.query_mask]
        if q.size == 0:
            return np.zeros(obs.tokens.shape[1], np.float32)
        f = obs.tokens[r, :, F_FEAT0:]
        n = np.linalg.norm(f, axis=1, keepdims=True)
        return ((f / np.maximum(n, 1e-12)) @ q.T).max(axis=1)

    def _dist(self, state, r, k, obs) -> float:
        if state is None:
            return float(obs.tokens[r, k, F_DIST])
        p = state.robots[r].pos
        return float(np.hypot(obs.token_xy[r, k, 0] - p[0], obs.token_xy[r, k, 1] - p[1]))

    def _nearest_frontier(self, obs, state, r, taken: "_Claims", lo: float) -> int:
        """Nearest frontier token at least `lo` away, else the farthest one, else hold."""
        best, best_d = -1, np.inf
        far, far_d = -1, -1.0
        for k in self._valid(obs, r):
            k = int(k)
            if int(obs.token_type[r, k]) != TOKEN_FRONTIER:
                continue
            if taken.has((TOKEN_FRONTIER, int(obs.token_id[r, k])), obs.token_xy[r, k]):
                continue
            d = self._dist(state, r, k, obs)
            if d > far_d:
                far, far_d = k, d
            if lo <= d < best_d:
                best, best_d = k, d
        pick = best if best >= 0 else far
        if pick >= 0:
            taken.add((TOKEN_FRONTIER, int(obs.token_id[r, pick])), obs.token_xy[r, pick])
            return pick
        return self._fallback(obs, r)

    def _best_of_type(self, obs, state, ttype: int, r: int, taken: "_Claims", lo: float) -> int:
        """Argmax of the query score over the selectable tokens of one type; sweep if there are
        none. Threshold-free by construction: with no token of that type there is nothing to score."""
        sc = self.query_scores(obs, r)
        best, best_s = -1, -np.inf
        for k in self._valid(obs, r):
            k = int(k)
            if int(obs.token_type[r, k]) != ttype:
                continue
            if taken.has((ttype, int(obs.token_id[r, k])), obs.token_xy[r, k]):
                continue
            if sc[k] > best_s:
                best, best_s = k, float(sc[k])
        if best >= 0:
            taken.add((ttype, int(obs.token_id[r, best])), obs.token_xy[r, best])
            return best
        return self._nearest_frontier(obs, state, r, taken, lo)


class RandomPolicy(Policy):
    name = "random"

    def act(self, obs, state=None):
        n = obs.n_robots
        out = np.zeros(n, np.int64)
        for r in range(n):
            v = self._valid(obs, r)
            out[r] = int(self.rng.choice(v)) if v.size else self._fallback(obs, r)
        return out


class NearestFrontierPolicy(Policy):
    """Nearest frontier the robot has not effectively already reached.

    The horizontal FoV leaves an unobserved wedge beside and behind each robot, so without
    `min_travel_m` the nearest frontier is always that wedge and the robot never sweeps.
    """
    name = "nearest_frontier"

    def __init__(self, queries=DEFAULT_QUERIES, seed: int = 0, min_travel_m: float | None = None):
        super().__init__(queries, seed)
        self.min_travel_m = min_travel_m

    def act(self, obs, state=None):
        n = obs.n_robots
        out = np.zeros(n, np.int64)
        taken = _Claims()
        lo = self._min_travel(state)
        for r in range(n):
            out[r] = self._nearest_frontier(obs, state, r, taken, lo)
        return out


class RayFollowerPolicy(Policy):
    """Fly down the ray whose feature answers the mission query best; sweep when no ray is offered.

    No merging or triangulation: the single best ray token wins, and its target point is wherever
    its own elevation says the ground is.
    """
    name = "ray_follower"

    def act(self, obs, state=None):
        n = obs.n_robots
        out = np.zeros(n, np.int64)
        taken = _Claims()
        lo = self._min_travel(state)
        for r in range(n):
            out[r] = self._best_of_type(obs, state, TOKEN_RAY, r, taken, lo)
        return out


class SegmentSeekerPolicy(Policy):
    """Go to the mapped segment whose mean feature answers the mission query best; else sweep."""
    name = "segment_seeker"

    def act(self, obs, state=None):
        n = obs.n_robots
        out = np.zeros(n, np.int64)
        taken = _Claims()
        lo = self._min_travel(state)
        for r in range(n):
            out[r] = self._best_of_type(obs, state, TOKEN_SEGMENT, r, taken, lo)
        return out


class LawnmowerPolicy(Policy):
    """Boustrophedon coverage that breaks off to investigate a human it actually sees.

    **Sweep.** The region is cut into `n_robots` contiguous horizontal bands, robot `r` owning band
    `r` (bands are disjoint, so the team de-conflicts by construction and the position claims only
    bite on the out-of-band fallback). Inside its band the robot follows a fixed serpentine order:
    lanes one sensor swath tall, +x along an even lane and -x along an odd one. The action is a
    token, so "go to the next sweep waypoint" is *the in-band frontier with the smallest sweep key*
    — frontiers vanish as the ground behind them is mapped, so the smallest remaining key is always
    the next unswept point of the band and an interrupted sweep resumes where it stopped instead of
    restarting. A frontier the robot has effectively already reached sorts last (the `min_travel`
    guard of CONTRACTS.md 12, folded into the key rather than filtering); an empty band falls back
    to the nearest frontier anywhere, and only a robot with no frontier at all holds.

    **Investigate.** A sweep that ignores what it sees is not a search baseline, so a live ray whose
    feature *is* a person diverts the robot. The classification is threshold-free: argmax cosine of
    the ray token's own `feat` over the **whole class-embedding set**, a human ray iff the argmax
    lands on `human_standing`/`human_prone`. No score is compared against a cutoff and no mission
    query is read — the queries would make the divert a function of the word list. The nearest such
    ray wins, and the robot stays on it until the ray resolves (it leaves the token set) or it has
    arrived (the target falls inside the `min_travel` guard), then the sweep resumes at its next
    waypoint. This runs on the robot's own view, so it works unchanged under range comms; a ray a
    peer gossiped is investigated like the robot's own.

    Only rays trigger a divert. Only an `open`-visibility human raises a far-field human ray at all
    (CONTRACTS.md 4): a casualty in a car, a building or under rubble is a `vehicle_toppled` /
    `building_damaged` / `debris` ray and this baseline sweeps past it, which is exactly the gap a
    learned policy has to close. Segments are not diverted to either — at `segment_scale = 40` a
    body is absorbed by its neighbours, so a human-classified segment would be a persistent region
    with no resolution rule and a threshold-free argmax over it livelocks (CONTRACTS.md 12);
    close-range human voxels accrue their find hits from the sweep's own footprint anyway.
    """
    name = "lawnmower"

    def __init__(self, queries=DEFAULT_QUERIES, seed: int = 0, min_travel_m: float | None = None):
        super().__init__(queries, seed)
        self.min_travel_m = min_travel_m
        self._chasing: dict[int, int] = {}      # robot -> ray token id it is investigating
        self._class_emb: dict[int, np.ndarray] = {}

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._chasing.clear()

    def act(self, obs, state=None):
        n = obs.n_robots
        out = np.zeros(n, np.int64)
        taken = _Claims()
        lo = self._min_travel(state)
        lane_h = self._lane_height(state)
        cls = self._classes(obs, state)
        for r in range(n):
            pick = self._investigate(obs, state, r, taken, lo, cls)
            out[r] = self._sweep(obs, state, r, n, taken, lo, lane_h) if pick < 0 else pick
        return out

    # -- investigate -----------------------------------------------------------------------
    def _classes(self, obs, state) -> np.ndarray:
        """[N_CLASSES, D] class embeddings — the robot's own text tower, not a belief the sim keeps.

        The env's table when there is one (so a custom similarity table or embedding cache is
        honoured), else the default table at the width of the token feature tail — which is the
        only D the argmax can use, and is not always the query block's (a hand-built observation,
        or a query set the env did not embed).
        """
        emb = getattr(state, "emb", None) if state is not None else None
        if emb is not None:
            return np.asarray(emb.class_emb, np.float32)
        d = int(obs.tokens.shape[2]) - F_FEAT0
        hit = self._class_emb.get(d)
        if hit is None:
            hit = np.asarray(get_embedding_table(self.queries, dim=d).class_emb, np.float32)
            self._class_emb[d] = hit
        return hit

    def _human_rays(self, obs, r: int, cls: np.ndarray) -> np.ndarray:
        """[K] True on the selectable ray tokens whose feature classifies as a person."""
        f = obs.tokens[r, :, F_FEAT0:]
        n = np.linalg.norm(f, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):     # a non-finite feature divides to nan, not a warning
            arg = ((f / np.maximum(n, 1e-12)) @ cls.T).argmax(axis=1)
        hit = np.isin(arg, HUMAN_CLASSES) & obs.token_mask[r] & (obs.token_type[r] == TOKEN_RAY)
        return hit & (n[:, 0] > 0)             # zero or non-finite: no argmax to trust

    def _investigate(self, obs, state, r: int, taken: "_Claims", lo: float,
                     cls: np.ndarray) -> int:
        """The ray token to fly at, or -1 to sweep. Sticky: the current one until it goes away."""
        hit = self._human_rays(obs, r, cls)
        live = {int(obs.token_id[r, k]): int(k) for k in np.flatnonzero(hit)
                if self._dist(state, r, int(k), obs) >= lo}
        held = self._chasing.get(r)
        if held is not None and held in live \
                and not taken.has((TOKEN_RAY, held), obs.token_xy[r, live[held]]):
            taken.add((TOKEN_RAY, held), obs.token_xy[r, live[held]])
            return live[held]
        # resolved, arrived, or a lower-index robot got there first: back to the sweep
        self._chasing.pop(r, None)
        best, best_d = -1, np.inf
        for rid, k in live.items():
            if taken.has((TOKEN_RAY, rid), obs.token_xy[r, k]):
                continue
            d = self._dist(state, r, k, obs)
            if d < best_d:
                best, best_d = k, d
        if best < 0:
            return -1
        self._chasing[r] = int(obs.token_id[r, best])
        taken.add((TOKEN_RAY, self._chasing[r]), obs.token_xy[r, best])
        return best

    # -- sweep -----------------------------------------------------------------------------
    @staticmethod
    def _lane_height(state) -> float:
        """One sensor swath, in the [-1, 1] region coordinates the tokens carry."""
        if state is None:
            return 2.0                          # no region to scale by: one lane per band
        c = state.cfg
        swath = 2.0 * float(np.sqrt(max(c.sensor.depth_limit_m ** 2
                                        - c.robot.flight_alt_m ** 2, 1.0)))
        y0, y1 = state.raster.region[1], state.raster.region[3]
        return float(max(2.0 * swath / max(y1 - y0, 1e-9), 1e-6))

    def _sweep(self, obs, state, r: int, n_robots: int, taken: "_Claims", lo: float,
               lane_h: float) -> int:
        band = 2.0 / max(int(n_robots), 1)
        y_lo = -1.0 + band * int(r)
        y_hi = 1.0 + 1e-6 if r == n_robots - 1 else y_lo + band
        best, best_key = -1, None
        for k in self._valid(obs, r):
            k = int(k)
            if int(obs.token_type[r, k]) != TOKEN_FRONTIER:
                continue
            yn = float(obs.tokens[r, k, F_YABS])
            if not (y_lo <= yn < y_hi):
                continue
            if taken.has((TOKEN_FRONTIER, int(obs.token_id[r, k])), obs.token_xy[r, k]):
                continue
            xn = float(obs.tokens[r, k, F_XABS])
            lane = int((yn - y_lo) / lane_h)
            key = (self._dist(state, r, k, obs) < lo, lane, xn if lane % 2 == 0 else -xn)
            if best_key is None or key < best_key:
                best, best_key = k, key
        if best >= 0:
            taken.add((TOKEN_FRONTIER, int(obs.token_id[r, best])), obs.token_xy[r, best])
            return best
        return self._nearest_frontier(obs, state, r, taken, lo)


class OraclePolicy(Policy):
    """Privileged: hold while a casualty is inside the mapping footprint (finding is by hit count,
    so dwelling is the productive action), otherwise the token closest to the nearest unfound one."""
    name = "oracle"
    privileged = True

    def __init__(self, queries=DEFAULT_QUERIES, seed: int = 0, min_travel_m: float = 6.0):
        super().__init__(queries, seed)
        self.min_travel_m = min_travel_m

    def act(self, obs, state=None):
        n = obs.n_robots
        out = np.zeros(n, np.int64)
        if state is None:
            raise ValueError("OraclePolicy needs the EnvState (privileged=True)")
        hs = state.raster.humans
        cas = (hs["role_id"] == schema.HUMAN_ROLES.index("casualty")) & ~state.human_found
        tx, ty = hs["x"][cas], hs["y"][cas]
        cf = state.cfg
        # the same LoS point the belief uses: the body top, never above the surface over it
        ras = state.raster
        hi, hj = ras.xy_to_ij(hs["x"][cas], hs["y"][cas])
        hz = ras.height[np.clip(hi, 0, ras.ny - 1), np.clip(hj, 0, ras.nx - 1)] if tx.size else None
        pts = np.stack([tx, ty, np.minimum(hs["z"][cas] + 0.5, hz)], axis=1) if tx.size else None
        taken: set[int] = set()
        for r in range(n):
            v = self._valid(obs, r)
            if tx.size == 0 or v.size == 0:
                out[r] = self._fallback(obs, r)
                continue
            rb = state.robots[r]
            cam = np.array([rb.pos[0], rb.pos[1], rb.alt], np.float64)
            iv, rr = human_visibility(state.raster, cf.sensor, cam, rb.heading, pts)
            if (iv & (rr <= cf.sensor.depth_limit_m)).any():
                out[r] = 0                # one is in the camera already: dwell until the hits land
                continue
            goal = self._goal(state, r, tx, ty, taken)
            gx, gy = tx[goal], ty[goal]
            xy = obs.token_xy[r][v]
            dd = np.hypot(xy[:, 0] - gx, xy[:, 1] - gy)
            dd[~np.isfinite(dd)] = np.inf
            dr = np.hypot(xy[:, 0] - rb.pos[0], xy[:, 1] - rb.pos[1])
            dd[dr < self.min_travel_m] = np.inf       # a token already reached is not progress
            hold = np.flatnonzero(obs.token_type[r][v] == TOKEN_HOLD)
            if hold.size:
                dd[hold] = np.inf     # holding never gets closer to a casualty
            out[r] = int(v[int(np.argmin(dd))]) if np.isfinite(dd).any() else int(v[0])
        return out

    def _goal(self, state, r: int, tx: np.ndarray, ty: np.ndarray, taken: set[int]) -> int:
        """Greedy claim: the nearest unfound casualty no lower-index robot took, else the nearest."""
        d = np.hypot(tx - state.robots[r].pos[0], ty - state.robots[r].pos[1])
        order = np.argsort(d, kind="stable")
        goal = next((int(c) for c in order if int(c) not in taken), int(order[0]))
        taken.add(goal)
        return goal


class OracleAssignPolicy(OraclePolicy):
    """Privileged: the optimal robot -> unfound-casualty matching instead of the oracle's greedy
    claims, re-solved from scratch every decision as casualties are found.

    Everything else is `OraclePolicy` — the same dwell rule and the same mechanical channel to a
    casualty (the token nearest to it, hold excluded, tokens the robot has effectively reached
    excluded) — so a row against `oracle` isolates the assignment and nothing else. The cost is the
    straight-line robot/casualty distance, matched by `scipy.optimize.linear_sum_assignment`.
    With more robots than casualties the matching leaves robots over; each of those takes the
    oracle's own no-claim line, its nearest unfound casualty.
    """
    name = "oracle_assign"
    privileged = True

    def __init__(self, queries=DEFAULT_QUERIES, seed: int = 0, min_travel_m: float = 6.0):
        super().__init__(queries, seed, min_travel_m)
        self._plan: np.ndarray | None = None

    def act(self, obs, state=None):
        self._plan = None                         # re-solve every decision
        return super().act(obs, state)

    def _goal(self, state, r: int, tx: np.ndarray, ty: np.ndarray, taken: set[int]) -> int:
        if self._plan is None:
            pos = np.array([rb.pos[:2] for rb in state.robots], np.float64)
            self._plan = assign_casualties(pos, tx, ty)
        g = int(self._plan[r]) if r < self._plan.size else -1
        if g < 0:
            return super()._goal(state, r, tx, ty, taken)
        taken.add(g)
        return g


def assign_casualties(pos: np.ndarray, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """Min-cost robot -> casualty matching over the straight-line distance matrix.

    -> int64 [n_robots], the casualty index per robot, -1 where the matching left the robot over
    (more robots than casualties). Deterministic: `linear_sum_assignment` is.
    """
    from scipy.optimize import linear_sum_assignment
    out = np.full(pos.shape[0], -1, np.int64)
    if pos.size == 0 or tx.size == 0:
        return out
    cost = np.hypot(pos[:, 0][:, None] - tx[None, :], pos[:, 1][:, None] - ty[None, :])
    rows, cols = linear_sum_assignment(cost)
    out[rows] = cols
    return out


POLICIES = {p.name: p for p in (RandomPolicy, NearestFrontierPolicy, LawnmowerPolicy,
                                RayFollowerPolicy, SegmentSeekerPolicy, OraclePolicy,
                                OracleAssignPolicy)}


def make_policy(name: str, queries=DEFAULT_QUERIES, seed: int = 0) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](queries=queries, seed=seed)


__all__ = ["Policy", "RandomPolicy", "NearestFrontierPolicy", "LawnmowerPolicy",
           "RayFollowerPolicy", "SegmentSeekerPolicy", "OraclePolicy", "OracleAssignPolicy",
           "assign_casualties", "POLICIES", "make_policy", "HUMAN_CLASSES"]
