"""Parallel environment execution (CONTRACTS.md 6 wrappers, torch-free).

Env slots own their scene: when an episode ends the slot draws the next scene from its own
`SceneBank` shard and seed stream, so no partial episode is ever discarded and the vector never
has to be rebuilt. `SubprocVecEnv` runs the shards in worker processes; `SerialVecEnv` runs the
identical shards in-process, so the same `(seed, n_envs, workers)` gives bit-identical
observations either way.

Padding follows `sim.vec_env.VecEnv`: R = the max robot count of the run, K = `cfg.k_tokens`,
empty robot slots masked out. With a fixed `--robots` the count of a slot is drawn once and kept;
with `--robots auto` it follows the scene area (train.scenes.auto_robots) and therefore changes
when a slot draws a new scene — R stays the bank-wide maximum, so the padded shapes never move.
"""
from __future__ import annotations

import copy
import os
import time
import traceback
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..sim.config import EnvConfig
from ..sim.vec_env import VecObs
from .scenes import AUTO, DEFAULT_REGION, SceneBank, SceneKey

STEP, RESET, CLOSE, SCENES, EPISODES, EVAL_START, EVAL_STEP, EVAL_END, DAGGER = range(9)
_OBS_FIELDS = ("tokens", "token_mask", "token_xy", "token_type", "token_id", "robot_feat",
               "bev", "robot_mask", "t", "query_emb", "query_w", "query_mask", "local",
               "peer_tokens", "robot_bev")
_BEV_IDX, _LOCAL_IDX = _OBS_FIELDS.index("bev"), _OBS_FIELDS.index("local")
_RECV_TIMEOUT = 900.0
_THREAD_VARS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_BASELINE_ALIAS = {"nearest": "nearest_frontier"}


def default_workers(n_envs: int) -> int:
    """Leave two cores for the trainer (policy forward/backward) and the OS."""
    return max(1, min(int(os.cpu_count() or 2) - 2, int(n_envs)))


# Measured on the largest v2 scene (downtown_*_40, 735 x 720 = 0.529 M cells, D = 24) with
# `scripts/bench_env.py`-style RSS accounting: a warm env is 92 MB under full comms and 133 MB with
# 8 per-robot beliefs, i.e. ~174 B per cell for the shared belief (the feature map alone is 4 D =
# 96 B) plus ~7 B per cell per robot for the private `known` / `feat_known` / frontier / segment
# grids. A parked env keeps all of it, so the cache has to budget in bytes and know about the team.
SHARED_B_PER_CELL = 174
PRIVATE_B_PER_CELL_PER_ROBOT = 7


def _env_cells(env) -> int:
    return int(env.raster.ny) * int(env.raster.nx)


def _env_mb(env) -> float:
    """What parking this env costs. Counting plain cells ignored both the size of the belief and
    the team, so an 8-robot decentralised env read as 1.0 when it is really ~1.3 envs."""
    b = SHARED_B_PER_CELL
    comms = getattr(env, "comms", None)
    if comms is not None:
        b += PRIVATE_B_PER_CELL_PER_ROBOT * len(comms.beliefs)
    return _env_cells(env) * b / 1e6


def run_episode(env, actor, seed: int, max_decisions: int | None = None) -> dict[str, float]:
    """One episode to termination; returns the metric dict plus the undiscounted return."""
    obs = env.reset(seed)
    if hasattr(actor, "reset"):
        actor.reset(seed)
    total, n = 0.0, 0
    while True:
        obs, r, done, info = env.step(actor.act(obs, env.state))
        total += float(r)
        n += 1
        if done or (max_decisions is not None and n >= max_decisions):
            break
    m = dict(info["metrics"])
    m["reward"] = total
    return m


# ---- samplers ----------------------------------------------------------------------------------
class RandomSampler:
    """Endless (scene, seed) stream for one training slot; `weights` is the --scene-mix pmf."""

    def __init__(self, keys: Sequence[SceneKey], robots: tuple[int, int], seed: int, slot: int,
                 weights: np.ndarray | None = None):
        self.rng = np.random.default_rng([int(seed), int(slot)])
        self.keys = list(keys)
        self.weights = None if weights is None else np.asarray(weights, np.float64)
        self.n_robots = (AUTO if int(robots[1]) <= 0
                         else int(self.rng.integers(int(robots[0]), int(robots[1]) + 1)))

    def next(self):
        i = (int(self.rng.integers(len(self.keys))) if self.weights is None
             else int(self.rng.choice(len(self.keys), p=self.weights)))
        return self.keys[i], int(self.rng.integers(0, 2 ** 31 - 1)), self.n_robots


class TaskSampler:
    """Fixed (scene, seed) work list for one evaluation slot; exhausts."""

    def __init__(self, tasks, n_robots: int):
        self.q = deque(tasks)
        self.n_robots = int(n_robots)

    def next(self):
        if not self.q:
            return None
        key, seed = self.q.popleft()
        return key, int(seed), self.n_robots


# ---- one shard's envs --------------------------------------------------------------------------
class EnvGroup:
    """Env slots living in one process: sampling, auto-reset and padded stacking."""

    ENV_CACHE = 16          # rebuilding an env costs ~5 ms; resetting a cached one ~1 ms
    ENV_CACHE_MB = 260      # ... but a parked env keeps its whole belief alive (92 MB shared,
                            # 133 MB with 8 private ones on the largest v2 scene), so the cache is
                            # budgeted in megabytes, not cells. This is the *parked* envs only:
                            # the slots' own envs sit on top, so a worker peaks at this plus
                            # `len(samplers)` live envs, and 10 workers x (260 + 4 x 133) MB is
                            # ~9 GB of the box.

    def __init__(self, bank: SceneBank, cfg: EnvConfig, samplers, R: int, K: int,
                 send_bev: bool = True, max_decisions: int | None = None,
                 t_max: float | None = None):
        self.bank = bank
        self._cache: dict[tuple[Any, int], Any] = {}
        self.cfg = cfg
        self.samplers = list(samplers)
        self.R, self.K = int(R), int(K)
        self.send_bev = bool(send_bev)
        self.max_decisions = max_decisions
        self.t_max = None if t_max is None else float(t_max)   # None = keep cfg.t_max_s
        n = len(self.samplers)
        self.envs: list[Any] = [None] * n
        self.last: list[Any] = [None] * n
        self.slot_key: list[Any] = [None] * n
        self.scene_ids: list[str] = [""] * n
        self.slot_meta: list[dict[str, Any]] = [{} for _ in range(n)]
        self.finished = np.zeros(n, np.bool_)
        self.n_dec = np.zeros(n, np.int64)
        self.ret = np.zeros(n, np.float64)
        self.rows: list[dict[str, Any]] = []
        self._buf: tuple | None = None

    @property
    def n(self) -> int:
        return len(self.samplers)

    # -- lifecycle -------------------------------------------------------------------------
    def _release(self, i: int) -> None:
        """Park the slot's previous env in the cache; only unused envs are ever cached."""
        env, ck = self.envs[i], self.slot_key[i]
        if env is None or ck is None:
            return
        self._cache[ck] = env
        mb = sum(_env_mb(e) for e in self._cache.values())
        while len(self._cache) > 1 and (len(self._cache) > self.ENV_CACHE
                                        or mb > self.ENV_CACHE_MB):
            mb -= _env_mb(self._cache.pop(next(iter(self._cache))))

    def _new(self, i: int) -> bool:
        t = self.samplers[i].next()
        if t is None:
            self.finished[i] = True
            return False
        key, seed, nr = t
        n_rob, t_max = self.bank.env_params(
            key, nr, self.cfg.t_max_s if self.t_max is None else self.t_max)
        ck = (key, int(n_rob), round(float(t_max), 6))
        env = self._cache.pop(ck, None)
        if env is None:
            cfg = copy.deepcopy(self.cfg)
            cfg.robot.n_robots = int(n_rob)
            cfg.t_max_s = float(t_max)
            env = self.bank.make_env(key, cfg, int(seed))
        else:
            env.reset(int(seed))          # identical to a fresh build: reset rebuilds all state
        self._release(i)
        self.envs[i] = env
        self.slot_key[i] = ck
        self.last[i] = env.state.last_obs
        self.scene_ids[i] = str(key)
        self.slot_meta[i] = {"scene": str(key), "area_km2": self.bank.area(key),
                             "bucket": self.bank.bucket(key), "disaster": self.bank.disaster(key),
                             "n_robots": int(n_rob), "t_max": float(t_max)}
        self.n_dec[i] = 0
        self.ret[i] = 0.0
        return True

    def reset(self):
        self.finished[:] = False
        self.rows = []
        self._buf = None
        for i in range(self.n):
            self._new(i)
        return self.stack()

    def close(self) -> None:
        self.envs = [None] * self.n
        self.last = [None] * self.n
        self._cache.clear()

    # -- stepping --------------------------------------------------------------------------
    def step(self, actions):
        a = np.asarray(actions)
        rewards = np.zeros(self.n, np.float64)
        dones = np.zeros(self.n, np.bool_)
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            if self.finished[i]:
                infos.append({"finished": True})
                continue
            nr = env.n_robots
            obs, r, done, info = env.step(a[i, :nr])
            self.n_dec[i] += 1
            self.ret[i] += float(r)
            rewards[i] = r
            out = {"new_found": int(info["new_found"]),
                   "found_total": int(info["found_total"]),
                   "coverage": float(info["coverage"])}
            if self.max_decisions is not None and self.n_dec[i] >= int(self.max_decisions):
                done = True
            if done:
                row = dict(info["metrics"])
                row["reward"] = float(self.ret[i])
                row["length"] = int(self.n_dec[i])
                row.update(self.slot_meta[i])
                self.rows.append(row)
                out["final_info"] = {"metrics": info["metrics"], "scene": self.scene_ids[i],
                                     "found_total": int(info["found_total"]),
                                     "coverage": float(info["coverage"]),
                                     "t": float(env.state.t)}
                self.last[i] = obs
                if self._new(i):
                    obs = self.last[i]
            self.last[i] = obs
            dones[i] = done
            infos.append(out)
        return self.stack(), rewards, dones, infos

    def drain(self) -> list[dict[str, Any]]:
        rows, self.rows = self.rows, []
        return rows

    # -- imitation -------------------------------------------------------------------------
    def _teacher(self, name: str, radius: float):
        """One privileged teacher per group; it is stateless across envs (no rng use)."""
        key = (str(name), float(radius))
        if getattr(self, "_teach_key", None) != key:
            from .teachers import make_any
            self._teach = make_any(name, queries=self.cfg.rayfronts.queries, seed=0,
                                   teacher_radius_m=float(radius))
            self._teach_key = key
        return self._teach

    def label(self, name: str, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Teacher action for the observation each live slot is currently holding."""
        pol = self._teacher(name, radius)
        lab = np.zeros((self.n, self.R), np.int64)
        val = np.zeros((self.n, self.R), np.bool_)
        for i, env in enumerate(self.envs):
            if self.finished[i] or env is None or self.last[i] is None:
                continue
            a = np.asarray(pol.act(self.last[i], env.state), np.int64)
            nr = min(int(a.shape[0]), self.R)
            lab[i, :nr] = a[:nr]
            val[i, :nr] = True
        return lab, val

    def dagger_step(self, payload):
        """Label the current states, then step with the teacher's or the student's actions.

        One round trip per decision: the labels do not depend on what is executed, so the parent
        sends the student's actions plus a per-slot "run the teacher" flag and gets the labels for
        the states it just saw back together with the transition.
        """
        actions, use_teacher, name, radius = payload
        lab, val = self.label(name, radius)
        a = np.where(np.asarray(use_teacher, np.bool_)[:, None], lab, np.asarray(actions))
        return self.step(a) + (lab, val, a)

    # -- stacking --------------------------------------------------------------------------
    def _alloc(self) -> None:
        o = next(x for x in self.last if x is not None)
        n, R, K = self.n, self.R, self.K
        F, D = o.tokens.shape[2], o.robot_feat.shape[1]
        Q, QD = o.query_emb.shape
        self.bev_shape = tuple(o.bev.shape)
        bev = np.zeros((n,) + self.bev_shape, np.float32) if self.send_bev else None
        loc = (np.zeros((n, R) + tuple(o.local.shape[1:]), np.float32)
               if o.local is not None else None)
        pshape = (0, 0) if o.peer_tokens is None else (max(R - 1, 0), o.peer_tokens.shape[2])
        # the actor's per-robot BEV is R times the size of the critic's, so it is shipped at
        # `tokens.robot_bev_size` (32 by default in the decentral variants) and skipped when 0
        rbev = (np.zeros((n, R) + tuple(o.robot_bev.shape[1:]), np.float32)
                if o.robot_bev is not None else None)
        self._buf = (np.zeros((n, R, K, F), np.float32), np.zeros((n, R, K), np.bool_),
                     np.full((n, R, K, 2), np.nan, np.float32), np.zeros((n, R, K), np.int8),
                     np.full((n, R, K), -1, np.int32), np.zeros((n, R, D), np.float32),
                     bev, np.zeros((n, R), np.bool_), np.zeros(n, np.float64),
                     np.zeros((n, Q, QD), np.float32), np.zeros((n, Q), np.float32),
                     np.zeros((n, Q), np.bool_), loc, np.zeros((n, R) + pshape, np.float32),
                     rbev)

    def stack(self) -> tuple:
        if self._buf is None:
            self._alloc()
        tok, mask, xy, tt, tid, rf, bev, rmask, t, qe, qw, qm, loc, peer, rbev = self._buf
        for i, o in enumerate(self.last):
            if o is None:
                continue
            nr = o.tokens.shape[0]
            if nr < self.R:               # the slot's robot count can shrink (--robots auto)
                tok[i, nr:] = 0.0
                mask[i, nr:] = False
                xy[i, nr:] = np.nan
                tt[i, nr:] = 0
                tid[i, nr:] = -1
                rf[i, nr:] = 0.0
                rmask[i, nr:] = False
            tok[i, :nr] = o.tokens
            mask[i, :nr] = o.token_mask
            xy[i, :nr] = o.token_xy
            tt[i, :nr] = o.token_type
            tid[i, :nr] = o.token_id
            rf[i, :nr] = o.robot_feat
            rmask[i, :nr] = True
            if bev is not None:
                bev[i] = o.bev
            t[i] = o.t
            qe[i], qw[i], qm[i] = o.query_emb, o.query_w, o.query_mask
            if loc is not None and o.local is not None:
                loc[i, nr:] = 0.0
                loc[i, :nr] = o.local
            if peer.shape[2] and o.peer_tokens is not None and o.peer_tokens.shape[1]:
                peer[i] = 0.0
                peer[i, :nr, : o.peer_tokens.shape[1]] = o.peer_tokens
            if rbev is not None and o.robot_bev is not None:
                rbev[i] = 0.0
                rbev[i, :nr] = o.robot_bev
        return self._buf + (self.bev_shape,)


# ---- worker ------------------------------------------------------------------------------------
@dataclass
class ShardSpec:
    scenes: str
    region_m: tuple[float, float]
    holdout_frac: float
    cfg: EnvConfig
    split: str
    seed: int
    robots: tuple[int, int]
    slots: tuple[int, ...]
    index: int
    n_shards: int
    R: int
    K: int
    send_bev: bool
    num_threads: int
    t_max: float | None = None
    scene_mix: dict[str, float] | None = None


def _build_group(spec: ShardSpec) -> tuple[SceneBank, EnvGroup]:
    bank = SceneBank(spec.scenes, spec.region_m, spec.holdout_frac)
    keys = bank.split(spec.split)
    shard_keys = keys[spec.index::spec.n_shards] or keys
    w = bank.mix_weights(shard_keys, spec.scene_mix)
    samplers = [RandomSampler(shard_keys, spec.robots, spec.seed, s, w) for s in spec.slots]
    return bank, EnvGroup(bank, spec.cfg, samplers, spec.R, spec.K, spec.send_bev,
                          t_max=spec.t_max)


def _worker_main(conn, spec: ShardSpec) -> None:
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # the parent owns Ctrl-C
    try:
        import numba
        numba.set_num_threads(max(1, int(spec.num_threads)))
    except Exception:                                  # numba may be absent/threading disabled
        pass
    try:
        bank, group = _build_group(spec)
        ev: EnvGroup | None = None
        while True:
            cmd, payload = conn.recv()
            if cmd == CLOSE:
                break
            elif cmd == RESET:
                conn.send(group.reset())
            elif cmd == STEP:
                conn.send(group.step(payload))
            elif cmd == SCENES:
                conn.send(list(group.scene_ids))
            elif cmd == EPISODES:
                conn.send(_run_episodes(bank, spec.cfg, *payload))
            elif cmd == DAGGER:
                conn.send(group.dagger_step(payload))
            elif cmd == EVAL_START:
                tasks, n_robots, max_decisions, R = payload
                ev = EnvGroup(bank, spec.cfg, [TaskSampler(tasks, n_robots)], R, spec.K,
                              spec.send_bev, max_decisions, t_max=spec.t_max)
                conn.send(ev.reset())
            elif cmd == EVAL_STEP:
                out = ev.step(payload)
                conn.send(out + (ev.drain(), bool(ev.finished.all())))
            elif cmd == EVAL_END:
                rows = ev.drain() if ev is not None else []
                if ev is not None:
                    ev.close()
                ev = None
                conn.send(rows)
            else:
                raise ValueError(f"env worker: unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_episodes(bank: SceneBank, cfg: EnvConfig, name: str, tasks, n_robots: int,
                  max_decisions: int | None, t_max: float | None = None) -> list[dict[str, Any]]:
    """Whole baseline episodes inside the worker: no per-decision round trip."""
    from .teachers import make_any
    rows = []
    for key, seed in tasks:
        n_rob, tm = bank.env_params(key, n_robots, cfg.t_max_s if t_max is None else t_max)
        ecfg = copy.deepcopy(cfg)
        ecfg.robot.n_robots = int(n_rob)
        ecfg.t_max_s = float(tm)
        env = bank.make_env(key, ecfg, int(seed))
        pol = make_any(_BASELINE_ALIAS.get(name, name), queries=cfg.rayfronts.queries,
                       seed=int(seed))
        row = run_episode(env, pol, int(seed), max_decisions)
        row.update({"scene": str(key), "area_km2": bank.area(key), "bucket": bank.bucket(key),
                    "disaster": bank.disaster(key), "n_robots": int(n_rob), "t_max": float(tm)})
        rows.append(row)
    return rows


# ---- parents -----------------------------------------------------------------------------------
class _ShardedVecEnv:
    """Shared parent logic: shard layout, VecObs assembly and the eval/episode helpers."""

    def __init__(self, scenes: str, cfg: EnvConfig, n_envs: int, robots: tuple[int, int] = (3, 3),
                 split: str = "train", seed: int = 0, n_workers: int | None = None,
                 send_bev: bool = True, region_m: Sequence[float] = DEFAULT_REGION,
                 holdout_frac: float = 0.1, num_threads: int = 1, t_max: float | None = None,
                 scene_mix: dict[str, float] | None = None):
        if int(n_envs) < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.n_envs = int(n_envs)
        self.cfg = cfg
        self.robots = (int(robots[0]), int(robots[1]))
        if self.robots[1] < self.robots[0]:
            raise ValueError(f"robots range {self.robots} is empty")
        self.t_max = None if t_max is None else float(t_max)
        self.scene_mix = scene_mix
        self.n_workers = max(1, min(int(n_workers or default_workers(self.n_envs)), self.n_envs))
        # auto robot counts vary with the scene, so R is the bank-wide maximum (meta only: the
        # parent reads scene headers, never whole scenes)
        self.R = (self.robots[1] if self.robots[1] > 0 else
                  SceneBank(str(scenes), region_m, holdout_frac).robot_bounds(self.robots)[1])
        self.K = int(cfg.k_tokens)
        self.send_bev = bool(send_bev)
        self.closed = False
        self.obs: VecObs | None = None
        self._bev_shape: tuple[int, ...] | None = None
        chunks = np.array_split(np.arange(self.n_envs), self.n_workers)
        self.slices = [slice(int(c[0]), int(c[-1]) + 1) for c in chunks]
        self.specs = [ShardSpec(scenes=str(scenes), region_m=(float(region_m[0]), float(region_m[1])),
                                holdout_frac=float(holdout_frac), cfg=cfg, split=split,
                                seed=int(seed), robots=self.robots, slots=tuple(int(x) for x in c),
                                index=w, n_shards=self.n_workers, R=self.R, K=self.K,
                                send_bev=self.send_bev, num_threads=int(num_threads),
                                t_max=self.t_max, scene_mix=self.scene_mix)
                      for w, c in enumerate(chunks)]

    # -- transport (subclass) --------------------------------------------------------------
    def _dispatch(self, cmd: int, payloads: list[Any], targets: list[int] | None = None) -> list[Any]:
        raise NotImplementedError

    def _broadcast(self, cmd: int, payload: Any = None) -> list[Any]:
        return self._dispatch(cmd, [payload] * self.n_workers)

    # -- api -------------------------------------------------------------------------------
    def reset_all(self) -> VecObs:
        self.obs = self._assemble(self._broadcast(RESET))
        return self.obs

    def reset(self, seeds=None) -> VecObs:
        return self.reset_all()

    def step(self, actions) -> tuple[VecObs, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        a = np.asarray(actions)
        if a.ndim != 2 or a.shape[0] != self.n_envs:
            raise ValueError(f"step: actions must be [{self.n_envs}, R], got {tuple(a.shape)}")
        out = self._dispatch(STEP, [np.ascontiguousarray(a[sl]) for sl in self.slices])
        rewards = np.zeros(self.n_envs, np.float64)
        dones = np.zeros(self.n_envs, np.bool_)
        infos: list[dict[str, Any]] = []
        for sl, (_, r, d, inf) in zip(self.slices, out):
            rewards[sl] = r
            dones[sl] = d
            infos += inf
        self.obs = self._assemble([o[0] for o in out])
        return self.obs, rewards, dones, infos

    def dagger_step(self, actions, use_teacher, teacher: str, radius: float):
        """-> (VecObs, rewards, dones, infos, labels [E, R], label_mask [E, R], executed [E, R])."""
        a = np.asarray(actions)
        ut = np.asarray(use_teacher, np.bool_)
        if a.ndim != 2 or a.shape[0] != self.n_envs:
            raise ValueError(f"dagger_step: actions must be [{self.n_envs}, R], "
                             f"got {tuple(a.shape)}")
        out = self._dispatch(DAGGER, [(np.ascontiguousarray(a[sl]), ut[sl], str(teacher),
                                       float(radius)) for sl in self.slices])
        rewards = np.zeros(self.n_envs, np.float64)
        dones = np.zeros(self.n_envs, np.bool_)
        lab = np.zeros((self.n_envs, self.R), np.int64)
        val = np.zeros((self.n_envs, self.R), np.bool_)
        exe = np.zeros((self.n_envs, self.R), np.int64)
        infos: list[dict[str, Any]] = []
        for sl, (_, r, d, inf, lb, vl, ex) in zip(self.slices, out):
            rewards[sl], dones[sl], lab[sl], val[sl], exe[sl] = r, d, lb, vl, ex
            infos += inf
        self.obs = self._assemble([o[0] for o in out])
        return self.obs, rewards, dones, infos, lab, val, exe

    def scene_ids(self) -> list[str]:
        out: list[str] = []
        for ids in self._broadcast(SCENES):
            out += ids
        return out

    def run_episodes(self, name: str, tasks: Sequence[tuple[SceneKey, int]], robots: int,
                     max_decisions: int | None = None) -> list[dict[str, Any]]:
        """Run whole episodes of a baseline inside the workers (no policy round trips)."""
        parts = [list(tasks)[w::self.n_workers] for w in range(self.n_workers)]
        res = self._dispatch(EPISODES, [(name, p, int(robots), max_decisions, self.t_max)
                                        for p in parts])
        rows: list[dict[str, Any]] = []
        for r in res:
            rows += r
        return rows

    def eval_start(self, tasks: Sequence[tuple[SceneKey, int]], robots: int,
                   max_decisions: int | None = None) -> VecObs:
        """Open one eval env per worker over a fixed work list (exhausts, then holds)."""
        tasks = list(tasks)
        n = max(1, min(len(tasks), self.n_workers))
        self._eval = list(range(n))
        self._eval_slices = [slice(w, w + 1) for w in range(n)]
        R = max(self.R, int(robots))
        parts = [tasks[w::n] for w in range(n)]
        res = self._dispatch(EVAL_START, [(p, int(robots), max_decisions, R) for p in parts],
                             self._eval)
        return self._assemble(res, slices=self._eval_slices, n=n, key="eval")

    def eval_step(self, actions):
        """-> (VecObs, finished_rows, all_done); one eval env per worker."""
        a = np.asarray(actions)
        out = self._dispatch(EVAL_STEP, [np.ascontiguousarray(a[sl]) for sl in self._eval_slices],
                             self._eval)
        rows: list[dict[str, Any]] = []
        alldone = True
        for _, _, _, _, r, fin in out:
            rows += r
            alldone = alldone and bool(fin)
        obs = self._assemble([o[0] for o in out], slices=self._eval_slices, n=len(self._eval),
                             key="eval")
        return obs, rows, alldone

    def eval_end(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self._dispatch(EVAL_END, [None] * len(self._eval), self._eval):
            rows += r
        return rows

    # -- assembly --------------------------------------------------------------------------
    def _assemble(self, parts: list[tuple], slices=None, n: int | None = None,
                  key: str = "train") -> VecObs:
        """Fresh arrays every step: a rollout keeps its observation history, so the parent must
        never hand out a buffer it will overwrite."""
        slices = self.slices if slices is None else slices
        n = self.n_envs if n is None else n
        p0 = parts[0]
        self._bev_shape = tuple(p0[len(_OBS_FIELDS)])
        buf: dict[str, np.ndarray | None] = {"bev": np.zeros((n,) + self._bev_shape, np.float32)}
        for k, name in enumerate(_OBS_FIELDS):
            if name == "bev":
                continue
            buf[name] = None if p0[k] is None else np.empty((n,) + p0[k].shape[1:], p0[k].dtype)
        for sl, p in zip(slices, parts):
            for k, name in enumerate(_OBS_FIELDS):
                if name == "bev":
                    if p[_BEV_IDX] is not None:
                        buf["bev"][sl] = p[_BEV_IDX]
                elif buf[name] is not None:
                    buf[name][sl] = p[k]
        return VecObs(**buf)

    @property
    def bev_shape(self) -> tuple[int, ...]:
        return self._bev_shape or (0, 0, 0)

    # -- shutdown --------------------------------------------------------------------------
    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SerialVecEnv(_ShardedVecEnv):
    """The same shards, in this process. Debugging / reference for the subprocess path."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        nt = self.specs[0].num_threads
        if nt:
            try:
                import numba
                numba.set_num_threads(max(1, int(nt)))
            except Exception:
                pass
        self.banks, self.groups = [], []
        for spec in self.specs:
            bank, group = _build_group(spec)
            self.banks.append(bank)
            self.groups.append(group)
        self.evals: list[EnvGroup | None] = [None] * self.n_workers

    def _dispatch(self, cmd: int, payloads: list[Any], targets: list[int] | None = None) -> list[Any]:
        if self.closed:
            raise RuntimeError("SerialVecEnv is closed")
        targets = list(range(self.n_workers)) if targets is None else list(targets)
        out = []
        for w, p in zip(targets, payloads):
            g = self.groups[w]
            if cmd == RESET:
                out.append(g.reset())
            elif cmd == STEP:
                out.append(g.step(p))
            elif cmd == SCENES:
                out.append(list(g.scene_ids))
            elif cmd == EPISODES:
                out.append(_run_episodes(self.banks[w], self.cfg, *p))
            elif cmd == DAGGER:
                out.append(g.dagger_step(p))
            elif cmd == EVAL_START:
                tasks, n_robots, max_dec, R = p
                ev = EnvGroup(self.banks[w], self.cfg, [TaskSampler(tasks, n_robots)],
                              R, self.K, self.send_bev, max_dec, t_max=self.t_max)
                self.evals[w] = ev
                out.append(ev.reset())
            elif cmd == EVAL_STEP:
                ev = self.evals[w]
                out.append(ev.step(p) + (ev.drain(), bool(ev.finished.all())))
            elif cmd == EVAL_END:
                ev = self.evals[w]
                out.append(ev.drain() if ev is not None else [])
                self.evals[w] = None
            else:
                raise ValueError(f"unknown command {cmd!r}")
        return out

    def close(self) -> None:
        for g in self.groups:
            g.close()
        self.groups = []
        self.closed = True


class SubprocVecEnv(_ShardedVecEnv):
    """One worker process per shard; only action/observation arrays cross the pipes."""

    def __init__(self, *a, start_method: str | None = None, **kw):
        super().__init__(*a, **kw)
        import multiprocessing as mp
        method = start_method or ("forkserver" if "forkserver" in mp.get_all_start_methods()
                                  else "spawn")
        ctx = mp.get_context(method)
        if method == "forkserver":
            try:
                ctx.set_forkserver_preload(["rlplanner.train.par_env"])
            except Exception:
                pass
        self.conns, self.procs = [], []
        # numba reads NUMBA_NUM_THREADS at import, so the cap has to be in the environment the
        # children inherit; each worker then lowers itself with set_num_threads.
        nt = str(max(1, int(self.specs[0].num_threads)))
        with _env_vars({k: nt for k in _THREAD_VARS}), _quiet_main():
            for w, spec in enumerate(self.specs):
                parent, child = ctx.Pipe(duplex=True)
                p = ctx.Process(target=_worker_main, args=(child, spec), daemon=True,
                                name=f"envworker-{w}")
                p.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(p)
        self._finalizer = weakref.finalize(self, _shutdown, self.conns, self.procs)

    # -- transport -------------------------------------------------------------------------
    def _dispatch(self, cmd: int, payloads: list[Any], targets: list[int] | None = None) -> list[Any]:
        if self.closed:
            raise RuntimeError("SubprocVecEnv is closed")
        targets = list(range(self.n_workers)) if targets is None else list(targets)
        for w, p in zip(targets, payloads):
            self.conns[w].send((cmd, p))
        return [self._recv(w) for w in targets]

    def _recv(self, i: int):
        conn, proc = self.conns[i], self.procs[i]
        deadline = time.monotonic() + _RECV_TIMEOUT
        while not conn.poll(0.1):
            if not proc.is_alive():
                raise RuntimeError(f"env worker {i} died (exitcode {proc.exitcode})")
            if time.monotonic() > deadline:
                raise RuntimeError(f"env worker {i} timed out after {_RECV_TIMEOUT:.0f}s")
        msg = conn.recv()
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
            raise RuntimeError(f"env worker {i} failed:\n{msg[1]}")
        return msg

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._finalizer()

    @property
    def n_alive(self) -> int:
        return sum(1 for p in self.procs if p.is_alive())


def _shutdown(conns, procs) -> None:
    for conn in conns:
        try:
            conn.send((CLOSE, None))
        except Exception:
            pass
    for p in procs:
        p.join(timeout=5.0)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
        if p.is_alive():
            p.kill()
            p.join(timeout=2.0)
    for conn in conns:
        try:
            conn.close()
        except Exception:
            pass


class _env_vars:
    """Temporarily set environment variables (children inherit them at spawn time)."""

    def __init__(self, values: dict[str, str]):
        self.values = values
        self.old: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self.values.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc) -> None:
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _quiet_main:
    """Hide `__main__` while starting workers.

    spawn/forkserver otherwise re-execute the parent's entry script in every child (multiprocessing
    `_fixup_main_from_path`), which would import torch once per worker. Nothing we send references
    `__main__`, so the children get a bare one.
    """

    def __enter__(self):
        import sys
        m = sys.modules.get("__main__")
        self.mod = m
        self.file = getattr(m, "__file__", None)
        self.spec = getattr(m, "__spec__", None)
        if m is not None:
            m.__spec__ = None
            if self.file is not None:
                del m.__file__
        return self

    def __exit__(self, *exc) -> None:
        if self.mod is not None:
            self.mod.__spec__ = self.spec
            if self.file is not None:
                self.mod.__file__ = self.file


BACKENDS = {"subproc": SubprocVecEnv, "serial": SerialVecEnv}


def make_vec_env(backend: str, scenes: str, cfg: EnvConfig, n_envs: int, **kw) -> _ShardedVecEnv:
    if backend not in BACKENDS:
        raise ValueError(f"unknown --env-backend {backend!r}; use one of {sorted(BACKENDS)}")
    return BACKENDS[backend](scenes, cfg, n_envs, **kw)


__all__ = ["SubprocVecEnv", "SerialVecEnv", "EnvGroup", "ShardSpec", "RandomSampler",
           "TaskSampler", "make_vec_env", "default_workers", "run_episode", "BACKENDS"]
