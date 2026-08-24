"""Multi-robot casualty-search environment (CONTRACTS.md 6)."""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy import ndimage

from ..scene import schema
from ..scene.schema import Scene
from .config import EnvConfig
from .geometry import PathPlanner
from .metrics import EpisodeMetrics
from .raster import Raster, rasterize
from .rayfronts_sim import RayFrontsSim
from .sensor import observable_mask
from .comms import CommsSim
from .state import (CASUALTY_ROLE_ID, F_FEAT0, TOKEN_HOLD, TOKEN_RAY, TOKEN_SEGMENT, EnvState,
                    Event, RobotState, TeamObs, VisitRecord)
from .tokens import TokenBuilder, team_view


class DisasterEnv:
    def __init__(self, scene: Scene, cfg: EnvConfig | None = None, seed: int = 0,
                 raster: Raster | None = None):
        self.scene = scene
        self.cfg = cfg or EnvConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("EnvConfig invalid: " + "; ".join(errs))
        self.seed = int(seed)
        self.raster = raster if raster is not None else rasterize(scene, self.cfg.raster.cell_m)
        self.planner = PathPlanner(self.raster.obstacle_mask(self.cfg.robot.flight_alt_m,
                                                             self.cfg.robot.clearance_m))
        self._free_idx = None
        self._observable_mask = None
        self._n_observable = 1
        self.builder: TokenBuilder | None = None
        self._n_casualties = int(sum(1 for h in scene.humans if h.role == "casualty"))
        self._cas = self.raster.humans["role_id"] == CASUALTY_ROLE_ID
        self._n_by_container = _by_container(self.raster.humans, self._cas)
        self.state: EnvState | None = None
        self.prof: dict[str, float] | None = None
        self.comms: CommsSim | None = None
        self.visits: list[VisitRecord] = []
        self._qsched = None
        self._q_rng: np.random.Generator | None = None
        self.reset(seed)

    # ---- reset --------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> TeamObs:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.planner.cache.clear()
        rf = RayFrontsSim(self.raster, self.cfg, self.rng)
        rf.prof = self.prof
        self.rf = rf
        if self.builder is None or self.builder.D != rf.D:
            self.builder = TokenBuilder(self.raster, self.cfg, rf.emb)
        robots = self._spawn()
        rf.set_observable(self._observable(robots))
        self.visits = []
        self._visit_seq = np.zeros(len(robots), np.int64)
        self._visits_at_decision = 0
        self._revisits = np.zeros(len(robots), np.int64)
        self._revisit_refunds = np.zeros(len(robots), np.int64)
        self._pending_revisits: list[tuple[int, np.ndarray]] = []
        self._dec_pos0 = np.array([rb.pos[:2] for rb in robots], np.float64)
        self._dec_visits: list[VisitRecord] = []
        self.comms = None
        self._last_views = None
        if self.cfg.comms.mode == "range":
            self.comms = CommsSim(self.raster, self.cfg, len(robots), rf.D, self.rng)
            rf.on_observe = self.comms.observe
            rf.keep_rays = True      # per-robot beliefs index ray rows (see sim/comms.py)
        self.metrics = EpisodeMetrics(t_max=self.cfg.t_max_s, n_casualties=self._n_casualties,
                                      decision_dt=self.cfg.decision_dt,
                                      n_by_container=dict(self._n_by_container))
        self.state = EnvState(
            t=0.0, decision_idx=0, scene=self.scene, raster=self.raster, cfg=self.cfg,
            robots=robots, observed=rf.observed, vox_cnt=rf.vox_cnt,
            last_seen_t=rf.last_seen_t, rays=rf.store(), ray_targets=rf.ray_targets,
            frontier_mask=rf.frontier_mask, frontier_clusters=rf.frontier_clusters,
            segments=rf.segments, seg_labels=rf.seg_labels, human_hits=rf.human_hits,
            human_found=rf.human_found, last_obs=None, last_actions=None, cum_reward=0.0, events=[],
            metrics=self.metrics.to_dict(), observable=self._observable_mask,
            vox_feat_sum=rf.vox_feat_sum, emb=rf.emb, queries=rf.queries, rf=rf)
        rf.begin_decision(len(robots))
        rf.update(robots, 0.0, self.rng)              # one free look so the first obs is not blank
        rf.end_of_decision(0.0, robots)
        rf.commit_decision()
        self._sync()
        self._init_query_schedule()
        obs = self.builder.build(rf, robots, 0.0, self.planner, views=self._views(0.0))
        self.state.last_obs = obs
        return obs

    def _init_query_schedule(self) -> None:
        """Build the per-episode query sampler and draw the initial subset (off by default)."""
        self._qsched = None
        self._q_rng = None
        qd = self.cfg.queries_dynamic
        if not qd.enabled:
            return
        from ..llm.schedule import QueryScheduleSampler       # numpy-only; imported on demand
        self._qsched = QueryScheduleSampler.from_config(qd, self.rf.emb, self.cfg)
        self._q_rng = np.random.default_rng([self.seed, 0x51EED])
        self._schedule_queries(initial=True)

    # ---- comms --------------------------------------------------------------------------------
    def _views(self, t: float):
        """One `RobotView` per robot, or None while comms are full (everyone shares the team map).

        Contact is evaluated once per decision, at the positions the robots hold now.
        """
        st = self.state
        st.visits = self.visits
        self._last_views = None
        if self.comms is None:
            # one shared belief, but the visited records still reach the tokens and the BEV
            self.builder.global_view = team_view(self.rf, self.visits)
            st.robot_views = None
            n = len(st.robots)
            st.comms_links = ~np.eye(n, dtype=np.bool_)     # full comms: everyone hears everyone
            return None
        t0 = perf_counter() if self.prof is not None else 0.0
        st.comms_links = self.comms.exchange(
            st.robots, self.rf, t,
            force_all=(st.decision_idx == 0 and self.cfg.comms.spawn_exchange))
        # the critic keeps the union belief and every visited record: CTDE, not a robot's view
        self.builder.global_view = team_view(self.rf, self.visits)
        out = self.comms.views(self.rf, st.robots, t, st.decision_idx)
        st.robot_views = self._last_views = out   # viewers read the per-robot beliefs from here
        self._tick("comms", t0)
        return out

    def _observable(self, robots) -> np.ndarray:
        """Cells any robot could ever observe: the free space its spawn component connects to, plus
        whatever is visible from it. Cached — it depends only on the scene, altitude and spawns."""
        if self._observable_mask is None:
            labels = {self.planner.label_at(*self.raster.xy_to_ij(rb.pos[0], rb.pos[1]))
                      for rb in robots}
            labels.discard(0)
            reach = np.isin(self.planner.labels, list(labels)) if labels else ~self.planner.obst
            self._observable_mask = observable_mask(self.raster, self.cfg.sensor, reach,
                                                    self.cfg.robot.flight_alt_m)
            self._n_observable = max(1, int(self._observable_mask.sum()))
        return self._observable_mask

    @property
    def observable(self) -> np.ndarray:
        return self._observable_mask

    def coverage(self) -> float:
        """Observed fraction of what is observable at all (roofs no robot can look down on and
        sealed-off space are excluded, so a complete sweep reaches 1.0)."""
        return float(self.rf.observed.sum()) / self._n_observable

    def _spawn(self) -> list[RobotState]:
        cfg = self.cfg.robot
        n = cfg.n_robots
        spawns = [tuple(s[:2]) for s in self.scene.robots_spawn]
        if not spawns:
            spawns = [self._default_spawn()]
        out: list[RobotState] = []
        for i in range(n):
            if i < len(spawns):
                x, y = spawns[i]
            else:
                bx, by = spawns[-1]
                ang = 2 * math.pi * (i - len(spawns) + 1) / max(1, n - len(spawns) + 1)
                x = bx + cfg.spawn_jitter_m * math.cos(ang)
                y = by + cfg.spawn_jitter_m * math.sin(ang)
            x, y = self.raster.clip_xy(x, y)
            x, y = self._snap_free(x, y)
            out.append(RobotState(idx=i, pos=np.array([x, y], np.float64), alt=cfg.flight_alt_m,
                                  heading=0.0, target_xy=None, target_token_type=TOKEN_HOLD,
                                  target_id=-1, trajectory=[(x, y)]))
        return out

    def _default_spawn(self) -> tuple[float, float]:
        road = (self.raster.cls == schema.CLASS_ID["road"]) & ~self.planner.obst
        cand = np.argwhere(road) if road.any() else np.argwhere(~self.planner.obst)
        if cand.size == 0:
            raise ValueError(f"scene {self.scene.meta.preset}/{self.scene.meta.seed}: no free cell to spawn in")
        k = np.lexsort((cand[:, 1], cand[:, 0]))[0]
        return self.raster.ij_to_xy(int(cand[k, 0]), int(cand[k, 1]))

    def _snap_free(self, x: float, y: float) -> tuple[float, float]:
        i, j = self.raster.xy_to_ij(x, y)
        if not self.planner.obst[i, j]:
            return x, y
        if self._free_idx is None:
            if self.planner.obst.all():
                raise ValueError(f"scene {self.scene.meta.preset}/{self.scene.meta.seed}: "
                                 f"every cell is an obstacle at alt {self.cfg.robot.flight_alt_m}")
            self._free_idx = ndimage.distance_transform_edt(self.planner.obst, return_distances=False,
                                                            return_indices=True)
        ni = int(self._free_idx[0][i, j]); nj = int(self._free_idx[1][i, j])
        return self.raster.ij_to_xy(ni, nj)

    # ---- queries ------------------------------------------------------------------------------
    def set_queries(self, names, weights=None) -> TeamObs:
        """Switch the *mission* queries of a running env.

        Only the query tokens move: the belief is stored as features, the token width is
        `TOKEN_FIXED + embedding_dim` and the rasters project onto a fixed basis, so nothing else
        in the observation depends on the query list and a trained network keeps working.
        Returns the refreshed observation.
        """
        self._set_queries(names, weights)
        st = self.state
        # the views of the decision that just ended, not a fresh extraction: nothing but the
        # query block may move (CONTRACTS.md 5)
        obs = self.builder.build(self.rf, st.robots, st.t, self.planner, views=self._last_views)
        st.last_obs = obs
        return obs

    def _set_queries(self, names, weights=None) -> None:
        """Swap the mission list without rebuilding the observation (the callers do that once)."""
        names = tuple(names)
        if len(names) > self.cfg.tokens.max_queries:
            raise ValueError(f"set_queries: {len(names)} queries exceed tokens.max_queries="
                             f"{self.cfg.tokens.max_queries}")
        self.rf.set_queries(names, weights)
        self.cfg.rayfronts.queries = names
        self._sync()

    def _schedule_queries(self, initial: bool) -> None:
        """Training-side query churn (`cfg.queries_dynamic`); a no-op while it is disabled."""
        if self._qsched is None:
            return
        idx = int(self.state.decision_idx)
        names, w = (self._qsched.initial(self._q_rng) if initial
                    else self._qsched.edit(self.state.query_names(), self.rf.query_w, idx,
                                           self._q_rng))
        if names is not None:
            self._set_queries(names, w)

    # ---- step ---------------------------------------------------------------------------------
    def step(self, actions):
        st = self.state
        if st is None:
            raise RuntimeError("DisasterEnv.step called before reset")
        a = np.asarray(actions).reshape(-1)
        robots = st.robots
        if a.shape[0] != len(robots):
            raise ValueError(f"step: expected {len(robots)} actions, got {a.shape[0]}")
        obs = st.last_obs
        ev: list[Event] = []
        assigned: list[np.ndarray | None] = []
        for r, rb in enumerate(robots):
            k = int(a[r])
            if k < 0 or k >= obs.token_mask.shape[1]:
                raise ValueError(f"step: robot {r} action {k} out of range [0, {obs.token_mask.shape[1]})")
            if not bool(obs.token_mask[r, k]):
                raise ValueError(f"step: robot {r} selected masked token {k} "
                                 f"(type={int(obs.token_type[r, k])}, id={int(obs.token_id[r, k])})")
            rb.last_action = k
            ttype = int(obs.token_type[r, k])
            rb.target_token_type = ttype
            rb.target_id = int(obs.token_id[r, k])
            rb.target_feat = np.asarray(obs.tokens[r, k, F_FEAT0:], np.float32).copy()
            assigned.append(None if ttype == TOKEN_HOLD else obs.token_xy[r, k].astype(np.float64))
            if ttype == TOKEN_HOLD:
                rb.target_xy = None
                rb.path = []
            else:
                rb.target_xy = obs.token_xy[r, k].astype(np.float64)
                self._plan(rb, ev, st.t)
        if self.cfg.record_events:
            ev.append(Event(st.t, "decision", {"actions": a.astype(int).tolist()}))

        n_found0 = self._n_found()
        self._begin_decision()
        for _ in range(self.cfg.substeps_per_decision):
            ev += self.rf.update(robots, st.t, self.rng)
            t0 = perf_counter() if self.prof is not None else 0.0
            self._move(robots, ev, st.t)
            self._tick("motion", t0)
            st.t += self.cfg.dt_sim
        ev += self.rf.end_of_decision(st.t, robots)
        self.rf.commit_decision()
        self._close_visits()
        self._settle_revisits()
        st.decision_idx += 1
        n_found = self._n_found()
        new_found = n_found - n_found0

        rw = self.cfg.reward
        n_r = len(robots)
        red_n = self.rf.redundant_cells[:n_r]
        obs_n = self.rf.observed_cells[:n_r]
        # a *fraction*, not a cell count: a robot that re-covered everything it looked at this
        # decision pays exactly `redundancy_cost`, whatever the footprint or the cell size
        red = red_n / np.maximum(obs_n, 1)
        refunds = self.rf.redundancy_refund[:n_r].copy()
        if rw.redundancy_refund:
            red[refunds] = 0.0        # the redundancy bought a find: no charge for this decision
        # ... and *averaged* over the team, not summed: one fully redundant decision costs
        # `redundancy_cost` whether the team is 3 robots or 8, so the term never outgrows a find
        red_cost = rw.redundancy_cost * red / max(1, n_r)
        # a refunded revisit is not charged; one refunded a decision or two after the arrival
        # (the find needs `found_hits` looks) is credited back then, so the episode total nets out
        rev_n = (self._revisits[:n_r] - self._revisit_refunds[:n_r]).astype(np.float64)
        rev_cost = rw.revisit_cost * rev_n
        # per-robot terms are credited to the robot that caused them and *added to the team
        # reward*: MAPPO gives every robot the same return, so the team pays for the redundancy
        # and the revisits any of its members produce (they are logged per robot for analysis)
        reward = rw.casualty_reward * new_found - rw.time_cost - float(red_cost.sum()
                                                                      + rev_cost.sum())
        if rw.potential_shaping:
            reward += self._shaping(n_found0, n_found)
        st.cum_reward += reward
        self._sync()
        # gossip first, so the metrics record *this* decision's links rather than the previous one
        views = self._views(st.t)          # timed separately as "comms"
        self.metrics.update(st.t, n_found, self.coverage(),
                            float(sum(r.dist_travelled for r in robots)), assigned,
                            _by_container(self.raster.humans, self._cas & self.rf.human_found),
                            extra={"redundant_cells": int(red_n.sum()),
                                   "observed_cells": int(obs_n.sum()),
                                   # the same team mean of per-robot fractions the reward charges
                                   # (before the refund; `redundancy_refunds` counts the waivers)
                                   "redundancy_frac": float(
                                       (red_n / np.maximum(obs_n, 1)).mean()),
                                   "redundancy_refunds": int(refunds.sum()),
                                   "intentional_revisits": int(self._revisits[:n_r].sum()),
                                   "revisit_refunds": int(self._revisit_refunds[:n_r].sum()),
                                   "revisit_penalties": float(rev_cost.sum()),
                                   "visits": len(self._dec_visits),
                                   "link_frac": (self.comms.stats.link_frac
                                                 if self.comms is not None else 1.0),
                                   "comms_range_m": (self.comms.range_m if self.comms is not None
                                                     else float("inf"))})
        # a scene with no casualties runs the full clock rather than ending on decision 1
        done = bool((self._n_casualties > 0 and n_found >= self._n_casualties)
                    or st.t >= self.cfg.t_max_s - 1e-9)
        if done:
            self.metrics.finalise(st.t)
        st.metrics = self.metrics.to_dict()
        if self.cfg.record_events:
            st.events += ev
        self._schedule_queries(initial=False)     # cfg.queries_dynamic; disabled by default
        t0 = perf_counter() if self.prof is not None else 0.0
        obs = self.builder.build(self.rf, robots, st.t, self.planner, views=views)
        self._tick("tokens", t0)
        st.last_obs = obs
        st.last_actions = a.astype(np.int64)
        info = {
            "new_found": int(new_found), "found_total": int(n_found),
            "n_casualties": self._n_casualties, "coverage": self.coverage(),
            "dist_travelled": np.array([r.dist_travelled for r in robots], np.float64),
            "events_this_step": ev, "metrics": st.metrics,
            # per-robot accounting of the two decentralised terms (CONTRACTS.md 6)
            "redundant_cells": red_n.copy(),
            "observed_cells": obs_n.copy(),
            "redundancy_cost": red_cost,
            "redundancy_refunds": refunds,
            "intentional_revisits": self._revisits[:n_r].copy(),
            "revisit_refunds": self._revisit_refunds[:n_r].copy(),
            "revisit_penalties": rev_cost,
            "comms_range_m": self.comms.range_m if self.comms is not None else float("inf"),
            "link_frac": self.comms.stats.link_frac if self.comms is not None else 1.0,
        }
        return obs, float(reward), done, info

    def _n_found(self) -> int:
        return int(np.count_nonzero(self._cas & self.rf.human_found))

    # ---- motion -------------------------------------------------------------------------------
    def _plan(self, rb: RobotState, ev: list[Event], t: float) -> None:
        ras = self.raster
        si, sj = ras.xy_to_ij(rb.pos[0], rb.pos[1])
        gx, gy = ras.clip_xy(rb.target_xy[0], rb.target_xy[1])
        gi, gj = ras.xy_to_ij(gx, gy)
        p = self.planner.path((si, sj), (gi, gj))
        if p is None:
            rb.path = []
            rb.target_xy = None
            rb.target_token_type = TOKEN_HOLD
            if self.cfg.record_events:
                ev.append(Event(t, "unreachable", {"robot": rb.idx, "goal": [float(gx), float(gy)]}))
            return
        xs, ys = ras.ij_to_xy(p[:, 0], p[:, 1])
        way = [(float(x), float(y)) for x, y in zip(xs[1:], ys[1:])]
        way.append((float(gx), float(gy)))
        rb.path = way

    def _move(self, robots, ev: list[Event], t: float) -> None:
        cfg = self.cfg.robot
        budget0 = cfg.speed_mps * self.cfg.dt_sim
        for rb in robots:
            if rb.target_xy is None:
                rb.trajectory.append((rb.pos[0], rb.pos[1]))
                continue
            budget = budget0
            while budget > 1e-9 and rb.path:
                wx, wy = rb.path[0]
                dx, dy = wx - rb.pos[0], wy - rb.pos[1]
                d = math.hypot(dx, dy)
                if d <= 1e-9:
                    rb.path.pop(0)
                    continue
                if d <= budget:
                    rb.pos[0], rb.pos[1] = wx, wy
                    rb.path.pop(0)
                    budget -= d
                    rb.dist_travelled += d
                else:
                    rb.pos[0] += dx * budget / d
                    rb.pos[1] += dy * budget / d
                    rb.dist_travelled += budget
                    budget = 0.0
                rb.heading = math.atan2(dy, dx)
            if math.hypot(rb.target_xy[0] - rb.pos[0], rb.target_xy[1] - rb.pos[1]) <= cfg.arrive_radius_m:
                if self.cfg.record_events:
                    ev.append(Event(t, "arrive", {"robot": rb.idx, "target_id": rb.target_id,
                                                  "type": rb.target_token_type}))
                self._on_arrive(rb, t)
                rb.target_xy = None
                rb.path = []
            rb.trajectory.append((rb.pos[0], rb.pos[1]))

    # ---- visited targets ----------------------------------------------------------------------
    def _begin_decision(self) -> None:
        n = len(self.state.robots)
        self._dec_pos0 = np.array([rb.pos[:2] for rb in self.state.robots], np.float64)
        self.rf.begin_decision(n)
        self._revisits = np.zeros(n, np.int64)
        self._revisit_refunds = np.zeros(n, np.int64)
        self._dec_visits = []
        self._visits_at_decision = len(self.visits)
        if self.comms is not None:
            self.comms.begin_decision()

    def _known_visits(self, ridx: int, known_only: bool):
        """The visited records this robot may be held responsible for knowing.

        `known_only` (the default) restricts them to the records it had *received* when it chose
        its target; otherwise every record made before now counts, even ones nobody told it about.
        """
        if self.comms is None:
            recs = self.visits[: self._visits_at_decision] if known_only else self.visits
            return list(recs)
        b = self.comms.beliefs[ridx]
        if known_only:
            return [b.visited[k] for k in b.visited_at_decision if k in b.visited]
        return list(self.visits)

    def _on_arrive(self, rb: RobotState, t: float) -> None:
        """A robot reached the target it chose: record the visit, and charge an intentional
        revisit if another robot had already been there."""
        rw = self.cfg.reward
        xy = np.asarray(rb.target_xy, np.float64).copy()
        p0 = self._dec_pos0[rb.idx]
        if math.hypot(p0[0] - xy[0], p0[1] - xy[1]) <= self.cfg.robot.arrive_radius_m:
            return      # it was already there when it chose: no journey, no visit, no revisit
        for rec in self._known_visits(rb.idx, rw.revisit_known_only):
            if int(rec.robot) == int(rb.idx) or rec.t >= t:
                continue
            if rw.revisit_confirmed_only and rec.n_found <= 0:
                continue     # unconfirmed: the target may still hold a casualty
            if math.hypot(rec.xy[0] - xy[0], rec.xy[1] - xy[1]) <= rw.revisit_m:
                self._revisits[rb.idx] += 1
                if rw.revisit_refund_on_find:
                    self._pending_revisits.append((int(rb.idx), xy.copy()))
                break
        if rb.target_token_type not in (TOKEN_RAY, TOKEN_SEGMENT):
            return       # a visit is a target the robot *chose to go and look at*
        feat = (rb.target_feat if rb.target_feat is not None
                else np.zeros(self.rf.D, np.float32))
        rec = VisitRecord(xy=xy, token_type=int(rb.target_token_type), token_id=int(rb.target_id),
                          feat=np.asarray(feat, np.float32).copy(), t=float(t), robot=int(rb.idx),
                          seq=int(self._visit_seq[rb.idx]), id=len(self.visits))
        self._visit_seq[rb.idx] += 1
        self.visits.append(rec)
        self._dec_visits.append(rec)
        if self.comms is not None:
            self.comms.beliefs[rb.idx].add_visit(rec)

    def _close_visits(self) -> None:
        """Fill in what each visit turned up: the casualties this robot found within `revisit_m`
        of the target during the decision it arrived in."""
        if not self._dec_visits or not self.rf.found_this_decision:
            return
        hs = self.raster.humans
        for rec in self._dec_visits:
            n = 0
            for ridx, k in self.rf.found_this_decision:
                if ridx != rec.robot or int(hs["role_id"][k]) != CASUALTY_ROLE_ID:
                    continue
                if math.hypot(float(hs["x"][k]) - rec.xy[0],
                              float(hs["y"][k]) - rec.xy[1]) <= self.cfg.reward.revisit_m:
                    n += 1
            rec.n_found = n

    def _settle_revisits(self) -> None:
        """Refund a revisit that turned up a new casualty, as the redundancy term already is:
        the arrival was not wasted. The charge stays open while the robot is still at the target
        (a find needs `found_hits` looks and can land a decision or two after the arrival)."""
        if not self._pending_revisits:
            return
        hs = self.raster.humans
        rw = self.cfg.reward
        finds = [(r, k) for r, k in self.rf.found_this_decision
                 if int(hs["role_id"][k]) == CASUALTY_ROLE_ID]
        keep: list[tuple[int, np.ndarray]] = []
        for ridx, xy in self._pending_revisits:
            if any(r == ridx and math.hypot(float(hs["x"][k]) - xy[0],
                                            float(hs["y"][k]) - xy[1]) <= rw.revisit_m
                   for r, k in finds):
                self._revisit_refunds[ridx] += 1
                continue
            pos = self.state.robots[ridx].pos
            if math.hypot(pos[0] - xy[0], pos[1] - xy[1]) <= self.cfg.robot.arrive_radius_m:
                keep.append((ridx, xy))      # still on the target: the visit is not over yet
        self._pending_revisits = keep

    # ---- misc ---------------------------------------------------------------------------------
    def _shaping(self, n0: int, n1: int) -> float:
        g = self.cfg.reward.shaping_gamma
        tot = max(1, self._n_casualties)
        return g * (n1 / tot) - (n0 / tot)

    def _tick(self, key: str, t0: float) -> None:
        if self.prof is not None:
            self.prof[key] = self.prof.get(key, 0.0) + (perf_counter() - t0)

    def _sync(self) -> None:
        st, rf = self.state, self.rf
        st.observed = rf.observed
        st.vox_cnt = rf.vox_cnt
        st.vox_feat_sum = rf.vox_feat_sum
        st.emb = rf.emb
        st.queries = rf.queries
        st.rf = rf
        st.last_seen_t = rf.last_seen_t
        st.rays = rf.store()
        st.ray_targets = rf.ray_targets
        st.frontier_mask = rf.frontier_mask
        st.frontier_clusters = rf.frontier_clusters
        st.segments = rf.segments
        st.seg_labels = rf.seg_labels
        st.human_hits = rf.human_hits
        st.human_found = rf.human_found

    @property
    def n_robots(self) -> int:
        return self.cfg.robot.n_robots

    @property
    def k_tokens(self) -> int:
        return self.cfg.k_tokens


def _by_container(humans, sel) -> dict[str, int]:
    """Casualty counts per container class ("open", "vehicle", "building", "rubble")."""
    ids = humans["container_id"][sel] if humans.shape[0] else np.zeros(0, np.int8)
    n = np.bincount(np.asarray(ids, np.int64), minlength=len(schema.HUMAN_CONTAINERS))
    return {c: int(n[i]) for i, c in enumerate(schema.HUMAN_CONTAINERS)}


__all__ = ["DisasterEnv"]
