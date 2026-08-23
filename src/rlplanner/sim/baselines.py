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
from .sensor import human_visibility
from .state import (F_DIST, F_FEAT0, TOKEN_FRONTIER, TOKEN_HOLD, TOKEN_RAY, TOKEN_SEGMENT,
                    EnvState, TeamObs)

CLAIM_M = 15.0     # two robots flying to points this close are going to the same place


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
            d = np.hypot(tx - rb.pos[0], ty - rb.pos[1])
            order = np.argsort(d, kind="stable")
            goal = None
            for c in order:
                if int(c) not in taken:
                    goal = int(c)
                    break
            if goal is None:
                goal = int(order[0])
            taken.add(goal)
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


POLICIES = {p.name: p for p in (RandomPolicy, NearestFrontierPolicy, RayFollowerPolicy,
                                SegmentSeekerPolicy, OraclePolicy)}


def make_policy(name: str, queries=DEFAULT_QUERIES, seed: int = 0) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](queries=queries, seed=seed)


__all__ = ["Policy", "RandomPolicy", "NearestFrontierPolicy", "RayFollowerPolicy",
           "SegmentSeekerPolicy", "OraclePolicy", "POLICIES", "make_policy"]
