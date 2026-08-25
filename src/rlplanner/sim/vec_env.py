"""Vectorised wrapper: a list of independent `DisasterEnv`s stepped together (CONTRACTS.md 6).

Envs may have different scenes, seeds, robot counts and token counts; observations are stacked
with zero padding and a `robot_mask` marks the real robot slots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .env import DisasterEnv
from .state import PEER_FEAT_DIM, TeamObs


@dataclass
class VecObs:
    tokens: np.ndarray        # float32 [E, R, K, F]
    token_mask: np.ndarray    # bool    [E, R, K]
    token_xy: np.ndarray      # float32 [E, R, K, 2]
    token_type: np.ndarray    # int8    [E, R, K]
    token_id: np.ndarray      # int32   [E, R, K]
    robot_feat: np.ndarray    # float32 [E, R, D]
    bev: np.ndarray           # float32 [E, C, Hb, Wb] compressed global BEV (critic)
    robot_mask: np.ndarray    # bool    [E, R] True = real robot
    t: np.ndarray             # float64 [E]
    query_emb: np.ndarray | None = None     # float32 [E, Qmax, D]
    query_w: np.ndarray | None = None       # float32 [E, Qmax]
    query_mask: np.ndarray | None = None    # bool    [E, Qmax]
    local: np.ndarray | None = None         # float32 [E, R, Cl, S, S] ego-centric crops
    peer_tokens: np.ndarray | None = None   # float32 [E, R, R - 1, PEER_FEAT_DIM]
    robot_bev: np.ndarray | None = None     # float32 [E, R, C, Hr, Wr] per-robot BEV (actor)
    region: np.ndarray | None = None        # float32 [E, 4] (x0, y0, x1, y1) per env

    @property
    def n_envs(self) -> int:
        return int(self.tokens.shape[0])

    def env_obs(self, e: int) -> TeamObs:
        n = int(self.robot_mask[e].sum())
        return TeamObs(tokens=self.tokens[e, :n], token_mask=self.token_mask[e, :n],
                       token_xy=self.token_xy[e, :n], token_type=self.token_type[e, :n],
                       token_id=self.token_id[e, :n], robot_feat=self.robot_feat[e, :n],
                       bev=self.bev[e], query_emb=self.query_emb[e], query_w=self.query_w[e],
                       query_mask=self.query_mask[e], t=float(self.t[e]),
                       local=None if self.local is None else self.local[e, :n],
                       peer_tokens=None if self.peer_tokens is None
                       else self.peer_tokens[e, :n, : max(n - 1, 0)],
                       robot_bev=None if self.robot_bev is None else self.robot_bev[e, :n],
                       region=None if self.region is None else self.region[e])


class VecEnv:
    def __init__(self, envs: Sequence[DisasterEnv], auto_reset: bool = True):
        if not envs:
            raise ValueError("VecEnv: empty env list")
        self.envs = list(envs)
        self.auto_reset = bool(auto_reset)
        self.R = max(e.n_robots for e in self.envs)
        self.K = max(e.k_tokens for e in self.envs)
        o = self.envs[0].state.last_obs
        self.F = o.tokens.shape[2]
        self.D = o.robot_feat.shape[1]
        self.bev_shape = o.bev.shape
        self.qmax, self.qdim = o.query_emb.shape
        self.local_shape = None if o.local is None else o.local.shape[1:]
        self.rbev_shape = None if o.robot_bev is None else o.robot_bev.shape[1:]
        for e in self.envs:
            ob = e.state.last_obs
            if ob.tokens.shape[2] != self.F or ob.bev.shape != self.bev_shape:
                raise ValueError("VecEnv: envs must share the query set and BEV shape")
        self._last: list[TeamObs] = [e.state.last_obs for e in self.envs]

    # ---- api ---------------------------------------------------------------------------------
    def reset(self, seeds: Sequence[int] | None = None) -> VecObs:
        for i, e in enumerate(self.envs):
            self._last[i] = e.reset(None if seeds is None else int(seeds[i]))
        return self._stack()

    def step(self, actions) -> tuple[VecObs, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        a = np.asarray(actions)
        if a.ndim != 2 or a.shape[0] != len(self.envs):
            raise ValueError(f"VecEnv.step: actions must be [{len(self.envs)}, R], got {a.shape}")
        rewards = np.zeros(len(self.envs), np.float64)
        dones = np.zeros(len(self.envs), np.bool_)
        infos: list[dict[str, Any]] = []
        for i, e in enumerate(self.envs):
            n = e.n_robots
            obs, r, done, info = e.step(a[i, :n])
            rewards[i] = r
            dones[i] = done
            if done and self.auto_reset:
                info = dict(info)
                info["final_info"] = {"metrics": info["metrics"], "found_total": info["found_total"],
                                      "coverage": info["coverage"], "t": e.state.t}
                obs = e.reset()
            infos.append(info)
            self._last[i] = obs
        return self._stack(), rewards, dones, infos

    def close(self) -> None:
        self.envs = []

    # ---- stacking ----------------------------------------------------------------------------
    def _stack(self) -> VecObs:
        E, R, K, F = len(self.envs), self.R, self.K, self.F
        v = VecObs(
            tokens=np.zeros((E, R, K, F), np.float32),
            token_mask=np.zeros((E, R, K), np.bool_),
            token_xy=np.full((E, R, K, 2), np.nan, np.float32),
            token_type=np.zeros((E, R, K), np.int8),
            token_id=np.full((E, R, K), -1, np.int32),
            robot_feat=np.zeros((E, R, self.D), np.float32),
            bev=np.zeros((E,) + self.bev_shape, np.float32),
            robot_mask=np.zeros((E, R), np.bool_),
            t=np.zeros(E, np.float64),
            query_emb=np.zeros((E, self.qmax, self.qdim), np.float32),
            query_w=np.zeros((E, self.qmax), np.float32),
            query_mask=np.zeros((E, self.qmax), np.bool_),
            local=(None if self.local_shape is None
                   else np.zeros((E, R) + tuple(self.local_shape), np.float32)),
            peer_tokens=np.zeros((E, R, max(R - 1, 0), PEER_FEAT_DIM), np.float32),
            robot_bev=(None if self.rbev_shape is None
                       else np.zeros((E, R) + tuple(self.rbev_shape), np.float32)),
            region=np.zeros((E, 4), np.float32))
        for i, o in enumerate(self._last):
            n, k = o.tokens.shape[0], o.tokens.shape[1]
            v.tokens[i, :n, :k] = o.tokens
            v.token_mask[i, :n, :k] = o.token_mask
            v.token_xy[i, :n, :k] = o.token_xy
            v.token_type[i, :n, :k] = o.token_type
            v.token_id[i, :n, :k] = o.token_id
            v.robot_feat[i, :n] = o.robot_feat
            v.bev[i] = o.bev
            v.robot_mask[i, :n] = True
            v.t[i] = o.t
            v.query_emb[i] = o.query_emb
            v.query_w[i] = o.query_w
            v.query_mask[i] = o.query_mask
            if v.local is not None and o.local is not None:
                v.local[i, :n] = o.local
            if o.peer_tokens is not None and o.peer_tokens.shape[1]:
                v.peer_tokens[i, :n, : o.peer_tokens.shape[1]] = o.peer_tokens
            if v.robot_bev is not None and o.robot_bev is not None:
                v.robot_bev[i, :n] = o.robot_bev
            if o.region is not None:
                v.region[i] = o.region
        return v


__all__ = ["VecEnv", "VecObs"]
