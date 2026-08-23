"""Torch-side observation batching: TeamObs / VecObs -> padded tensors (E, n, K, F).

Carries the CTDE split through to torch: `tokens`/`local`/`peer_tokens` and the query block are the
actor's per-robot inputs, `bev` is the centralised critic's compressed global belief.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
import torch

from ..sim.state import TeamObs

_EMPTY_BEV = (0, 0, 0)
_EMPTY_LOCAL = (0, 0, 0, 0)


@dataclass
class ObsBatch:
    tokens: torch.Tensor      # float32 [E, R, K, F]
    token_mask: torch.Tensor  # bool    [E, R, K]
    token_type: torch.Tensor  # int64   [E, R, K]
    token_id: torch.Tensor    # int64   [E, R, K]
    robot_feat: torch.Tensor  # float32 [E, R, D]
    bev: torch.Tensor         # float32 [E, C, H, W] (may be [E,0,0,0] when unused)
    robot_mask: torch.Tensor  # bool    [E, R]
    query_emb: torch.Tensor   # float32 [E, Q, D] unit mission-query embeddings, zero-padded
    query_w: torch.Tensor     # float32 [E, Q]
    query_mask: torch.Tensor  # bool    [E, Q]
    local: torch.Tensor       # float32 [E, R, Cl, S, S] (may be [E,R,0,0] when unused)
    peer_tokens: torch.Tensor # float32 [E, R, P, PEER_FEAT_DIM] what gossip last said about a peer
    robot_bev: torch.Tensor   # float32 [E, R, C, Hr, Wr] per-robot BEV (may be [E,R,0,0])

    # ---- shape helpers ------------------------------------------------------------------
    @property
    def n_envs(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_robots(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def k_tokens(self) -> int:
        return int(self.tokens.shape[2])

    @property
    def token_dim(self) -> int:
        return int(self.tokens.shape[3])

    @property
    def robot_dim(self) -> int:
        return int(self.robot_feat.shape[2])

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def has_bev(self) -> bool:
        return self.bev.numel() > 0

    @property
    def has_local(self) -> bool:
        return self.local.numel() > 0

    @property
    def has_robot_bev(self) -> bool:
        return self.robot_bev.numel() > 0

    @property
    def has_peers(self) -> bool:
        return self.peer_tokens.numel() > 0

    @property
    def n_queries(self) -> int:
        return int(self.query_emb.shape[1])

    @property
    def feat_dim(self) -> int:
        return int(self.query_emb.shape[2])

    # ---- transforms ---------------------------------------------------------------------
    def to(self, device) -> "ObsBatch":
        return ObsBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def index(self, idx) -> "ObsBatch":
        return ObsBatch(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def drop_bev(self) -> "ObsBatch":
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["bev"] = torch.zeros((self.n_envs,) + _EMPTY_BEV, dtype=self.bev.dtype,
                               device=self.bev.device)
        return ObsBatch(**d)

    def drop_local(self) -> "ObsBatch":
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["local"] = torch.zeros((self.n_envs, self.n_robots) + _EMPTY_LOCAL[:2],
                                 dtype=self.local.dtype, device=self.local.device)
        return ObsBatch(**d)

    def drop_robot_bev(self) -> "ObsBatch":
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["robot_bev"] = torch.zeros((self.n_envs, self.n_robots, 0, 0),
                                     dtype=self.robot_bev.dtype, device=self.robot_bev.device)
        return ObsBatch(**d)

    @staticmethod
    def cat(batches: Sequence["ObsBatch"]) -> "ObsBatch":
        if not batches:
            raise ValueError("ObsBatch.cat: empty sequence")
        return ObsBatch(**{f.name: torch.cat([getattr(b, f.name) for b in batches], dim=0)
                           for f in fields(ObsBatch)})

    # ---- constructors -------------------------------------------------------------------
    @staticmethod
    def from_vec_obs(v, device=None, with_bev: bool = True, with_local: bool = True,
                     with_robot_bev: bool = True) -> "ObsBatch":
        """Stack a `sim.vec_env.VecObs` (already padded) into tensors."""
        E, R = v.tokens.shape[0], v.tokens.shape[1]
        bev = np.asarray(v.bev, np.float32) if with_bev else np.zeros((E,) + _EMPTY_BEV, np.float32)
        loc = (np.asarray(v.local, np.float32) if with_local and v.local is not None
               else np.zeros((E, R, 0, 0), np.float32))
        peer = (np.asarray(v.peer_tokens, np.float32) if v.peer_tokens is not None
                else np.zeros((E, R, 0, 0), np.float32))
        rbev = (np.asarray(v.robot_bev, np.float32)
                if with_robot_bev and getattr(v, "robot_bev", None) is not None
                else np.zeros((E, R, 0, 0), np.float32))
        return ObsBatch(
            tokens=_t(v.tokens, torch.float32, device),
            token_mask=_t(v.token_mask, torch.bool, device),
            token_type=_t(v.token_type, torch.int64, device),
            token_id=_t(v.token_id, torch.int64, device),
            robot_feat=_t(v.robot_feat, torch.float32, device),
            bev=_t(bev, torch.float32, device),
            robot_mask=_t(v.robot_mask, torch.bool, device),
            query_emb=_t(v.query_emb, torch.float32, device),
            query_w=_t(v.query_w, torch.float32, device),
            query_mask=_t(v.query_mask, torch.bool, device),
            local=_t(loc, torch.float32, device),
            peer_tokens=_t(peer, torch.float32, device),
            robot_bev=_t(rbev, torch.float32, device))

    @staticmethod
    def from_team_obs(obs: TeamObs | Sequence[TeamObs], device=None, with_bev: bool = True,
                      with_local: bool = True, with_robot_bev: bool = True) -> "ObsBatch":
        """Stack raw `TeamObs` (one per env) with zero padding over robots and tokens."""
        lst = [obs] if isinstance(obs, TeamObs) else list(obs)
        if not lst:
            raise ValueError("ObsBatch.from_team_obs: empty sequence")
        E = len(lst)
        R = max(o.tokens.shape[0] for o in lst)
        K = max(o.tokens.shape[1] for o in lst)
        F = lst[0].tokens.shape[2]
        D = lst[0].robot_feat.shape[1]
        Q, QD = lst[0].query_emb.shape
        for o in lst:
            if o.tokens.shape[2] != F or o.robot_feat.shape[1] != D:
                raise ValueError("ObsBatch.from_team_obs: inconsistent feature dims "
                                 f"({o.tokens.shape[2]} vs {F}, {o.robot_feat.shape[1]} vs {D})")
        tokens = np.zeros((E, R, K, F), np.float32)
        mask = np.zeros((E, R, K), np.bool_)
        ttype = np.zeros((E, R, K), np.int64)
        tid = np.full((E, R, K), -1, np.int64)
        rfeat = np.zeros((E, R, D), np.float32)
        rmask = np.zeros((E, R), np.bool_)
        qe = np.zeros((E, Q, QD), np.float32)
        qw = np.zeros((E, Q), np.float32)
        qm = np.zeros((E, Q), np.bool_)
        bev_shape = lst[0].bev.shape if with_bev else _EMPTY_BEV
        bev = np.zeros((E,) + tuple(bev_shape), np.float32)
        l0 = lst[0].local
        loc = (np.zeros((E, R) + tuple(l0.shape[1:]), np.float32)
               if with_local and l0 is not None else np.zeros((E, R, 0, 0), np.float32))
        p0 = lst[0].peer_tokens
        peer = (np.zeros((E, R, p0.shape[1], p0.shape[2]), np.float32) if p0 is not None
                else np.zeros((E, R, 0, 0), np.float32))
        b0 = lst[0].robot_bev
        rbev = (np.zeros((E, R) + tuple(b0.shape[1:]), np.float32)
                if with_robot_bev and b0 is not None else np.zeros((E, R, 0, 0), np.float32))
        for i, o in enumerate(lst):
            n, k = o.tokens.shape[0], o.tokens.shape[1]
            tokens[i, :n, :k] = o.tokens
            mask[i, :n, :k] = o.token_mask
            ttype[i, :n, :k] = o.token_type
            tid[i, :n, :k] = o.token_id
            rfeat[i, :n] = o.robot_feat
            rmask[i, :n] = True
            qe[i], qw[i], qm[i] = o.query_emb, o.query_w, o.query_mask
            if with_bev:
                bev[i] = o.bev
            if loc.shape[2] and o.local is not None:
                loc[i, :n] = o.local
            if peer.shape[2] and o.peer_tokens is not None and o.peer_tokens.shape[1]:
                peer[i, :n, : o.peer_tokens.shape[1]] = o.peer_tokens
            if rbev.shape[2] and o.robot_bev is not None:
                rbev[i, :n] = o.robot_bev
        return ObsBatch(tokens=_t(tokens, torch.float32, device),
                        token_mask=_t(mask, torch.bool, device),
                        token_type=_t(ttype, torch.int64, device),
                        token_id=_t(tid, torch.int64, device),
                        robot_feat=_t(rfeat, torch.float32, device),
                        bev=_t(bev, torch.float32, device),
                        robot_mask=_t(rmask, torch.bool, device),
                        query_emb=_t(qe, torch.float32, device),
                        query_w=_t(qw, torch.float32, device),
                        query_mask=_t(qm, torch.bool, device),
                        local=_t(loc, torch.float32, device),
                        peer_tokens=_t(peer, torch.float32, device),
                        robot_bev=_t(rbev, torch.float32, device))


def _t(a, dtype, device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


__all__ = ["ObsBatch"]
