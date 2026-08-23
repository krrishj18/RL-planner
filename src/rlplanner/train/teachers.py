"""Privileged teachers for imitation (torch-free: the env workers import this).

A teacher is a `sim.baselines.Policy` with `privileged = True`, so it reads `env.state` through the
interface the baselines already use and the simulator stays untouched. `oracle` is the reference
teacher; `oracle_sweep` is the same idea with two changes the student can actually reproduce:

- it never selects a `visited` token. 354 of 506 charged revisits in the oracle's own episodes are
  visited tokens next to a casualty nobody can find (DESIGN_VARIANTS.md H), so the reference
  teacher spends a third of its decisions demonstrating a move the reward punishes.
- when the nearest unfound casualty is further than `teacher_radius_m` from every token it could
  pick, there is nothing to home in on and it sweeps (nearest frontier) instead of flying at a
  casualty it has no observable reason to know about. That is the part of the behaviour a
  non-privileged student can learn; the homing part is the part it can learn *near* a target.
"""
from __future__ import annotations

import numpy as np

from ..scene import schema
from ..sim import baselines
from ..sim.baselines import Policy, _Claims
from ..sim.config import DEFAULT_QUERIES
from ..sim.sensor import human_visibility
from ..sim.state import TOKEN_HOLD, TOKEN_VISITED

TEACHER_RADIUS_M = 150.0


class OracleSweepPolicy(baselines.OraclePolicy):
    """Oracle that never targets a visited record and sweeps when no casualty is in reach."""
    name = "oracle_sweep"
    privileged = True

    def __init__(self, queries=DEFAULT_QUERIES, seed: int = 0, min_travel_m: float = 6.0,
                 teacher_radius_m: float = TEACHER_RADIUS_M):
        super().__init__(queries, seed, min_travel_m)
        self.teacher_radius_m = float(teacher_radius_m)

    def act(self, obs, state=None):
        if state is None:
            raise ValueError("OracleSweepPolicy needs the EnvState (privileged=True)")
        n = obs.n_robots
        out = np.zeros(n, np.int64)
        hs = state.raster.humans
        cas = (hs["role_id"] == schema.HUMAN_ROLES.index("casualty")) & ~state.human_found
        tx, ty = hs["x"][cas], hs["y"][cas]
        cf = state.cfg
        ras = state.raster
        pts = None
        if tx.size:
            hi, hj = ras.xy_to_ij(tx, ty)
            hz = ras.height[np.clip(hi, 0, ras.ny - 1), np.clip(hj, 0, ras.nx - 1)]
            pts = np.stack([tx, ty, np.minimum(hs["z"][cas] + 0.5, hz)], axis=1)
        claims = _Claims()
        lo = self._min_travel(state)
        taken: set[int] = set()
        for r in range(n):
            v = self._selectable(obs, r)
            if tx.size == 0 or v.size == 0:
                out[r] = self._nearest_frontier(obs, state, r, claims, lo)
                continue
            rb = state.robots[r]
            cam = np.array([rb.pos[0], rb.pos[1], rb.alt], np.float64)
            iv, rr = human_visibility(state.raster, cf.sensor, cam, rb.heading, pts)
            if (iv & (rr <= cf.sensor.depth_limit_m)).any():
                out[r] = self._hold(obs, r)          # dwell: finding is by hit count
                continue
            xy = obs.token_xy[r][v]
            # nothing within reach of any casualty -> this robot has no target to home in on
            near = np.hypot(xy[:, 0][:, None] - tx[None, :], xy[:, 1][:, None] - ty[None, :])
            near[~np.isfinite(near)] = np.inf
            if not np.isfinite(near).any() or near.min() > self.teacher_radius_m:
                out[r] = self._nearest_frontier(obs, state, r, claims, lo)
                continue
            d = np.hypot(tx - rb.pos[0], ty - rb.pos[1])
            order = np.argsort(d, kind="stable")
            goal = next((int(c) for c in order if int(c) not in taken), int(order[0]))
            taken.add(goal)
            dd = np.hypot(xy[:, 0] - tx[goal], xy[:, 1] - ty[goal])
            dd[~np.isfinite(dd)] = np.inf
            dr = np.hypot(xy[:, 0] - rb.pos[0], xy[:, 1] - rb.pos[1])
            dd[dr < self.min_travel_m] = np.inf      # a token already reached is not progress
            if not np.isfinite(dd).any():
                out[r] = self._nearest_frontier(obs, state, r, claims, lo)
                continue
            pick = int(v[int(np.argmin(dd))])
            claims.add((int(obs.token_type[r, pick]), int(obs.token_id[r, pick])),
                       obs.token_xy[r, pick])
            out[r] = pick
        return out

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _selectable(obs, r: int) -> np.ndarray:
        """Valid slots minus hold and minus every visited record."""
        tt = obs.token_type[r]
        return np.flatnonzero(obs.token_mask[r] & (tt != TOKEN_VISITED) & (tt != TOKEN_HOLD))

    @staticmethod
    def _hold(obs, r: int) -> int:
        if bool(obs.token_mask[r, 0]) and int(obs.token_type[r, 0]) == TOKEN_HOLD:
            return 0
        v = np.flatnonzero(obs.token_mask[r] & (obs.token_type[r] != TOKEN_VISITED))
        return int(v[0]) if v.size else 0


TEACHERS: dict[str, type[Policy]] = {"oracle": baselines.OraclePolicy,
                                     "oracle_sweep": OracleSweepPolicy}


def is_teacher(name: str) -> bool:
    return str(name) in TEACHERS


def make_teacher(name: str, queries=DEFAULT_QUERIES, seed: int = 0, **kw) -> Policy:
    if name not in TEACHERS:
        raise ValueError(f"unknown teacher {name!r}; known: {sorted(TEACHERS)}")
    cls = TEACHERS[name]
    if cls is baselines.OraclePolicy:
        kw.pop("teacher_radius_m", None)
    return cls(queries=queries, seed=int(seed), **kw)


def make_any(name: str, queries=DEFAULT_QUERIES, seed: int = 0, **kw) -> Policy:
    """A teacher if the name is one, otherwise a hand-coded baseline."""
    if is_teacher(name):
        return make_teacher(name, queries, seed, **kw)
    return baselines.make_policy(name, queries=queries, seed=int(seed))


__all__ = ["OracleSweepPolicy", "TEACHERS", "TEACHER_RADIUS_M", "is_teacher", "make_teacher",
           "make_any"]
