"""Hand-coded baselines (CONTRACTS.md 7). Each returns one token index per robot — except
`lawnmower`, which returns waypoints (float [n_robots, 2]); a stripe pattern flies over ground
that is already mapped, and no token is ever offered there.

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
from .config import DEFAULT_QUERIES, EnvConfig
from .embeddings import get_embedding_table
from .sensor import human_visibility
from .state import (F_DIST, F_FEAT0, TOKEN_FRONTIER, TOKEN_HOLD, TOKEN_RAY, TOKEN_SEGMENT,
                    EnvState, TeamObs)

CLAIM_M = 15.0     # two robots flying to points this close are going to the same place
HUMAN_CLASSES = (schema.CLASS_ID["human_standing"], schema.CLASS_ID["human_prone"])

# `LawnmowerPolicy` lane geometry, when neither a constructor kwarg nor an `EnvState` supplies it
_DEFAULTS = EnvConfig()
DEPTH_LIMIT_M = float(_DEFAULTS.sensor.depth_limit_m)
FLIGHT_ALT_M = float(_DEFAULTS.robot.flight_alt_m)
ARRIVE_M = float(_DEFAULTS.robot.arrive_radius_m)
HFOV_DEG = float(_DEFAULTS.sensor.hfov_deg)
_MOVE_EPS_M = 0.25          # below this a robot did not move this decision
_STALL_DECISIONS = 2        # that many motionless sweep decisions = an unreachable waypoint


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
    """Textbook boustrophedon coverage that breaks off to investigate a human it actually sees.

    **Waypoint actions.** `act` returns float `[n_robots, 2]` world points, not token indices
    (CONTRACTS.md 6): a stripe pattern flies *over ground it has already mapped*, and a token is
    only ever offered at the edge of the unknown, so the token action space cannot express a
    lawnmower at all — steering by tokens zig-zags between frontiers. A row of NaNs is "hold".

    **Sweep.** The region — `TeamObs.region`, the deployment extent, not something guessed from the
    tokens the robot happens to hold — is cut into `n_robots` contiguous **vertical strips** of
    equal width (`strips`), robot `r` owning strip `r` for the whole episode. Its strip is filled
    with `ceil(strip_width / swath)` evenly spaced vertical lanes, the swath being the ground width
    one pass of the camera sweeps (`_swath`: the depth limit's horizontal reach, narrowed by the
    horizontal FoV), and the waypoint list is the lane ends in serpentine order: +y along an even
    lane, -y along an odd one, the ends inset half a swath so the footprint still reaches the
    boundary. A per-robot cursor walks that list, advancing when the robot is within
    `arrive_radius_m` of the point it is flying to; a point A* cannot reach (a lane end inside a
    building) leaves the robot parked, so `_STALL_DECISIONS` decisions without motion advance the
    cursor past it.

    **A robot never leaves its strip.** Every waypoint it emits — sweep or divert — is clamped into
    its own x range, and there is no fallback onto a neighbour's ground: strips are disjoint, so
    nothing has to de-conflict the sweep, each robot's cursor is its own, and the whole thing is
    unchanged under range comms and for any `n_robots` in 1..10. A robot with no strip to sweep (a
    degenerate region, or no region on offer) holds — a NaN row. One that reaches the end of its
    list wraps to lane 0, the lane it swept longest ago, and runs its own serpentine again — all a
    robot with a per-robot view can do, since it cannot see which of its lanes the stochastic hit
    model left unobserved.

    **Investigate.** A sweep that ignores what it sees is not a search baseline, so a live ray whose
    feature *is* a person diverts the robot: the waypoint becomes that ray's own target point. The
    classification is threshold-free — argmax cosine of the ray token's `feat` over the **whole
    class-embedding set**, a human ray iff the argmax lands on `human_standing`/`human_prone`. No
    score is compared against a cutoff and no mission query is read, which would make the divert a
    function of the word list. Only a ray whose target lands **inside the robot's own strip**
    counts: one pointing across the boundary is a teammate's to investigate, and chasing it would
    both break the partition and put two robots on one body. The nearest such ray wins, and the
    robot stays on it until the ray resolves (it leaves the token set) or it has arrived (the
    target falls inside the `min_travel` guard); the cursor is untouched while it is away, so the
    sweep resumes at the waypoint it left. This runs on the robot's own view, so a ray a peer
    gossiped is investigated like its own.

    Only rays trigger a divert. Only an `open`-visibility human raises a far-field human ray at all
    (CONTRACTS.md 4): a casualty in a car, a building or under rubble is a `vehicle_toppled` /
    `building_damaged` / `debris` ray and this baseline sweeps past it, which is exactly the gap a
    learned policy has to close. Segments are not diverted to either — at `segment_scale = 40` a
    body is absorbed by its neighbours, so a human-classified segment would be a persistent region
    with no resolution rule and a threshold-free argmax over it livelocks (CONTRACTS.md 12);
    close-range human voxels accrue their find hits from the sweep's own footprint anyway.
    """
    name = "lawnmower"
    waypoint_policy = True                  # `act` -> float [n_robots, 2] waypoints

    def __init__(self, queries=DEFAULT_QUERIES, seed: int = 0, min_travel_m: float | None = None,
                 depth_limit_m: float | None = None, flight_alt_m: float | None = None,
                 hfov_deg: float | None = None, arrive_radius_m: float | None = None,
                 swath_m: float | None = None):
        super().__init__(queries, seed)
        self.min_travel_m = min_travel_m
        # None = take it from the env when there is one, else the EnvConfig default
        self.depth_limit_m = depth_limit_m
        self.flight_alt_m = flight_alt_m
        self.hfov_deg = hfov_deg
        self.arrive_radius_m = arrive_radius_m
        self.swath_m = swath_m                  # pins the lane spacing outright
        self._chasing: dict[int, int] = {}      # robot -> ray token id it is investigating
        self._class_emb: dict[int, np.ndarray] = {}
        self._cursor: dict[int, int] = {}       # robot -> index into its own waypoint list
        self._prev_xy: dict[int, tuple[float, float]] = {}
        self._stalled: dict[int, int] = {}      # consecutive sweep decisions without motion
        self._lanes: tuple[tuple, dict[int, np.ndarray]] | None = None

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._chasing.clear()
        self._cursor.clear()
        self._prev_xy.clear()
        self._stalled.clear()
        self._lanes = None

    def act(self, obs, state=None) -> np.ndarray:
        n = obs.n_robots
        out = np.full((n, 2), np.nan, np.float64)
        taken = _Claims()
        lo = self._min_travel(state)
        cls = self._classes(obs, state)
        reg = self._region(obs, state)
        swath = self._swath(state)
        arrive = self._arrive(state)
        pos = self._positions(obs, state, reg)
        strips = self.strips(reg, n)
        for r in range(n):
            sx = strips[r] if strips is not None else None
            xy = self._investigate(obs, state, r, taken, lo, cls, sx)
            if xy is None:
                xy = self._sweep(r, n, reg, swath, arrive, pos)
            if xy is not None:
                out[r] = self._clamp(xy, sx)
        return out

    @staticmethod
    def strips(region, n_robots: int) -> list[tuple[float, float]] | None:
        """The x range each robot owns, low to high — `[(x0, x1)] * n_robots`, or None with no
        region. Public so a viewer can draw the partition without re-deriving it."""
        reg = LawnmowerPolicy._region_tuple(region)
        if reg is None:
            return None
        x0, _, x1, _ = reg
        n = max(int(n_robots), 1)
        w = (x1 - x0) / n
        return [(x0 + w * i, x0 + w * (i + 1)) for i in range(n)]

    @staticmethod
    def _clamp(xy, strip) -> np.ndarray:
        """A waypoint never leaves the robot's own strip, however it was chosen."""
        out = np.asarray(xy, np.float64).copy()
        if strip is not None:
            out[0] = min(max(out[0], strip[0]), strip[1])
        return out

    # -- geometry --------------------------------------------------------------------------
    def _cfg_val(self, state, pinned, path: tuple[str, str], default: float) -> float:
        """A constructor kwarg wins; otherwise the env's own value, otherwise the shipped one."""
        if pinned is not None:
            return float(pinned)
        cfg = getattr(state, "cfg", None) if state is not None else None
        if cfg is not None:
            return float(getattr(getattr(cfg, path[0]), path[1]))
        return float(default)

    def _swath(self, state) -> float:
        """Ground width one pass of the camera sweeps.

        `reach = sqrt(depth_limit^2 - flight_alt^2)` is how far in front of the robot the depth
        limit reaches along the ground, so the footprint spans `2 * reach` *across* the flight line
        only if the camera looks sideways as far as it looks forward. It does not: the wedge is
        `hfov_deg` wide, so the widest point of the swept sector sits at `reach * sin(hfov / 2)`
        off the line and lanes a full `2 * reach` apart leave a strip between them unobserved
        (measured on an empty 240 m scene, 3 robots, ample horizon: 0.89 coverage at `2 * reach`
        against 1.00 at `2 * reach * sin(hfov / 2)`). `swath_m` pins it if a caller wants the
        uncorrected width back.
        """
        if self.swath_m is not None:
            return max(float(self.swath_m), 1e-6)
        d = self._cfg_val(state, self.depth_limit_m, ("sensor", "depth_limit_m"), DEPTH_LIMIT_M)
        a = self._cfg_val(state, self.flight_alt_m, ("robot", "flight_alt_m"), FLIGHT_ALT_M)
        h = self._cfg_val(state, self.hfov_deg, ("sensor", "hfov_deg"), HFOV_DEG)
        reach = float(np.sqrt(max(d * d - a * a, 1.0)))
        wide = 1.0 if h >= 180.0 else float(np.sin(0.5 * np.radians(max(h, 1e-6))))
        return max(2.0 * reach * wide, 1e-6)

    def _arrive(self, state) -> float:
        return self._cfg_val(state, self.arrive_radius_m, ("robot", "arrive_radius_m"), ARRIVE_M)

    @staticmethod
    def _region_tuple(reg) -> tuple[float, float, float, float] | None:
        if reg is None:
            return None
        v = np.asarray(reg, np.float64).reshape(-1)
        if v.size < 4 or not np.isfinite(v[:4]).all() or v[2] <= v[0] or v[3] <= v[1]:
            return None
        return float(v[0]), float(v[1]), float(v[2]), float(v[3])

    @staticmethod
    def _region(obs, state) -> tuple[float, float, float, float] | None:
        """(x0, y0, x1, y1) from the observation's own metadata, else the env's raster."""
        reg = getattr(obs, "region", None)
        if reg is None and state is not None:
            reg = getattr(getattr(state, "raster", None), "region", None)
        return LawnmowerPolicy._region_tuple(reg)

    @staticmethod
    def _positions(obs, state, reg) -> np.ndarray | None:
        """World xy per robot: the env's when there is one, else `robot_feat`'s [-1, 1] pose
        un-scaled by the region — the sweep runs off the observation alone under range comms."""
        if state is not None and getattr(state, "robots", None):
            return np.array([[rb.pos[0], rb.pos[1]] for rb in state.robots], np.float64)
        rf = getattr(obs, "robot_feat", None)
        if reg is None or rf is None or np.asarray(rf).shape[1] < 2:
            return None
        x0, y0, x1, y1 = reg
        f = np.asarray(rf, np.float64)[:, :2]
        return np.stack([x0 + 0.5 * (f[:, 0] + 1.0) * (x1 - x0),
                         y0 + 0.5 * (f[:, 1] + 1.0) * (y1 - y0)], axis=1)

    def _strip_lanes(self, reg, n_robots: int, r: int, swath: float) -> np.ndarray:
        """[m, 2] serpentine waypoints of robot `r`'s strip; empty when it has no strip to fly."""
        key = (reg, int(n_robots), float(swath))
        if self._lanes is None or self._lanes[0] != key:
            self._lanes = (key, {})
        hit = self._lanes[1].get(r)
        if hit is not None:
            return hit
        x0, y0, x1, y1 = reg
        n = max(int(n_robots), 1)
        w_strip = (x1 - x0) / n
        w = max(float(swath), 1e-6)
        if w_strip <= 0.0 or y1 <= y0:
            out = np.zeros((0, 2), np.float64)
        else:
            n_lane = max(1, int(np.ceil(w_strip / w - 1e-9)))
            dx = w_strip / n_lane
            inset = min(0.5 * w, 0.5 * (y1 - y0) - 1e-6)
            ya, yb = y0 + inset, y1 - inset
            pts = []
            for i in range(n_lane):
                x = x0 + w_strip * r + (i + 0.5) * dx
                pts += [(x, ya), (x, yb)] if i % 2 == 0 else [(x, yb), (x, ya)]
            out = np.asarray(pts, np.float64)
        self._lanes[1][r] = out
        return out

    def _sweep(self, r: int, n_robots: int, reg, swath: float, arrive: float,
               pos) -> np.ndarray | None:
        """The next unreached waypoint of robot `r`'s strip, or None to hold."""
        if reg is None or pos is None or r >= pos.shape[0]:
            return None
        way = self._strip_lanes(reg, n_robots, r, swath)
        m = way.shape[0]
        if m == 0:
            return None
        p = pos[r]
        c = int(self._cursor.get(r, 0)) % m
        for _ in range(m):                      # skip the points the robot is already standing on
            if float(np.hypot(way[c, 0] - p[0], way[c, 1] - p[1])) > arrive:
                break
            c = (c + 1) % m
        prev = self._prev_xy.get(r)
        moved = prev is None or np.hypot(p[0] - prev[0], p[1] - prev[1]) > _MOVE_EPS_M
        self._stalled[r] = 0 if moved else int(self._stalled.get(r, 0)) + 1
        if self._stalled[r] >= _STALL_DECISIONS:
            c = (c + 1) % m                     # A* cannot reach it (a lane end inside a building)
            self._stalled[r] = 0
        self._prev_xy[r] = (float(p[0]), float(p[1]))
        self._cursor[r] = c
        return way[c].copy()

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
                     cls: np.ndarray, strip) -> np.ndarray | None:
        """The point to fly at, or None to sweep. Sticky: the current ray until it goes away.

        A ray whose target lands outside `strip` is a teammate's business and never diverts."""
        hit = self._human_rays(obs, r, cls)
        live = {int(obs.token_id[r, k]): int(k) for k in np.flatnonzero(hit)
                if self._dist(state, r, int(k), obs) >= lo
                and np.isfinite(obs.token_xy[r, int(k)]).all()
                and self._in_strip(obs.token_xy[r, int(k)], strip)}
        held = self._chasing.get(r)
        if held is not None and held in live \
                and not taken.has((TOKEN_RAY, held), obs.token_xy[r, live[held]]):
            taken.add((TOKEN_RAY, held), obs.token_xy[r, live[held]])
            return np.asarray(obs.token_xy[r, live[held]], np.float64)
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
            return None
        self._chasing[r] = int(obs.token_id[r, best])
        taken.add((TOKEN_RAY, self._chasing[r]), obs.token_xy[r, best])
        return np.asarray(obs.token_xy[r, best], np.float64)

    @staticmethod
    def _in_strip(xy, strip) -> bool:
        return strip is None or (strip[0] <= float(xy[0]) <= strip[1])


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
