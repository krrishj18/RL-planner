"""Shared-parameter per-robot actor over the K candidate tokens + centralised MAPPO critic.

The observation is open-set: a token carries the item's RayFronts *feature*, and the mission
queries arrive as their own tokens. The actor therefore never sees a similarity — it embeds the
feature through a shared linear, embeds each query (embedding + weight) the same way, puts both in
one attention set with the robot's query token, and lets cross-attention work out what is relevant.
Adding a query, dropping one or handing the policy an LLM-proposed hint changes the *input*, not
the network.

CTDE: the actor reads the per-robot tokens and the ego-centric local crop; the critic additionally
pools the compressed global BEV. `use_bev` is therefore a critic switch.

`sequential_decode` is the *execution* switch (CONTRACTS.md 6): with it on, robots decode in index
order and a token a lower-index robot just claimed is masked for the later ones — information that
only a centralised executor has. Every decentralised variant runs with it off: the robots decode
independently and a peer's intent reaches them one decision late, through gossip
(`claimed_by_peer`, the peer tokens), which is what the deployed system can actually do.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..sim.state import PEER_FEAT_DIM, PEER_VALID, TOKEN_HOLD, TOKEN_TYPE_NAMES
from ..sim.tokens import BEV_CHANNELS, LOCAL_CHANNELS
from .obs import ObsBatch

_HOLD_KEY = 0          # sequential-decode key that is never excluded
_TYPE_STRIDE = 1 << 24


def _mlp(din: int, dout: int, dh: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dout))


class BevCNN(nn.Module):
    """C x H x W -> out_dim."""

    def __init__(self, channels: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenPolicy(nn.Module):
    """Actor: per-robot [ROBOT] query attends over {item tokens} u {query tokens}; logits = scaled
    dot product against the item tokens only.

    Critic: attention-pool of the per-robot query embeddings (+ the global BEV) -> one V per
    env/decision, shared by every robot of that decision (MAPPO).
    """

    def __init__(self, token_dim: int, robot_dim: int, feat_dim: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dim_ff: int = 256,
                 n_token_types: int = len(TOKEN_TYPE_NAMES), use_bev: bool = False,
                 bev_channels: int = len(BEV_CHANNELS), bev_dim: int = 64,
                 use_local: bool = False, local_channels: int = len(LOCAL_CHANNELS),
                 local_dim: int = 64, dropout: float = 0.0, use_peers: bool = False,
                 peer_dim: int = PEER_FEAT_DIM, use_robot_bev: bool = False,
                 robot_bev_channels: int = len(BEV_CHANNELS), robot_bev_dim: int = 64,
                 sequential_decode: bool = True):
        super().__init__()
        self.sequential_decode = bool(sequential_decode)
        self.token_dim = int(token_dim)
        self.robot_dim = int(robot_dim)
        self.feat_dim = int(feat_dim)
        self.geom_dim = self.token_dim - self.feat_dim
        if self.geom_dim < 1:
            raise ValueError(f"TokenPolicy: feat_dim {feat_dim} >= token_dim {token_dim}")
        self.d_model = int(d_model)
        self.use_bev = bool(use_bev)
        self.bev_channels = int(bev_channels)
        self.bev_dim = int(bev_dim)
        self.use_local = bool(use_local)
        self.local_channels = int(local_channels)
        self.local_dim = int(local_dim)
        self.use_peers = bool(use_peers)
        self.peer_dim = int(peer_dim)
        self.use_robot_bev = bool(use_robot_bev)
        self.robot_bev_channels = int(robot_bev_channels)
        self.robot_bev_dim = int(robot_bev_dim)

        self.token_mlp = _mlp(self.geom_dim, d_model, d_model)
        self.feat_proj = nn.Linear(self.feat_dim, d_model)       # shared by items and queries
        self.type_emb = nn.Embedding(n_token_types, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.robot_mlp = _mlp(robot_dim, d_model, d_model)
        self.robot_norm = nn.LayerNorm(d_model)
        self.query_w_proj = nn.Linear(1, d_model)
        self.query_kind = nn.Parameter(torch.zeros(d_model))     # "this is a mission query" marker
        self.query_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=dim_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.logit_scale = float(d_model) ** -0.5

        if self.use_peers:
            # one token per peer, in the same attention set as the items and the queries: what the
            # policy does with a stale peer position is learned, not coded
            self.peer_proj = nn.Linear(self.peer_dim, d_model)
            self.peer_kind = nn.Parameter(torch.zeros(d_model))
            self.peer_norm = nn.LayerNorm(d_model)
        if self.use_local:
            self.local_cnn = BevCNN(local_channels, local_dim)
            self.local_proj = nn.Linear(local_dim, d_model)
        if self.use_robot_bev:
            self.rbev_cnn = BevCNN(robot_bev_channels, robot_bev_dim)
            self.rbev_proj = nn.Linear(robot_bev_dim, d_model)
        if self.use_bev:
            self.bev_cnn = BevCNN(bev_channels, bev_dim)
        self.value_attn = nn.Linear(d_model, 1)
        cin = d_model + (bev_dim if self.use_bev else 0)
        self.value_head = nn.Sequential(nn.Linear(cin, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self._init_heads()

    def _init_heads(self) -> None:
        """Orthogonal init on our own heads; the transformer keeps torch defaults."""
        for mod in (self.token_mlp, self.robot_mlp, self.value_head):
            _orth(mod[0], 2 ** 0.5)
            _orth(mod[2], 1.0)
        for mod in (self.q_proj, self.k_proj, self.value_attn, self.feat_proj, self.query_w_proj):
            _orth(mod, 1.0)
        if self.use_local:
            _orth(self.local_proj, 1.0)
        if self.use_robot_bev:
            _orth(self.rbev_proj, 1.0)
        if self.use_peers:
            _orth(self.peer_proj, 1.0)
            nn.init.normal_(self.peer_kind, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        nn.init.normal_(self.query_kind, std=0.02)

    # ---- config / checkpointing ---------------------------------------------------------
    def config(self) -> dict[str, Any]:
        layer0 = self.encoder.layers[0]
        return {"token_dim": self.token_dim, "robot_dim": self.robot_dim,
                "feat_dim": self.feat_dim, "d_model": self.d_model,
                "n_heads": layer0.self_attn.num_heads,
                "n_layers": len(self.encoder.layers),
                "dim_ff": layer0.linear1.out_features,
                "n_token_types": self.type_emb.num_embeddings,
                "use_bev": self.use_bev, "bev_channels": self.bev_channels,
                "bev_dim": self.bev_dim, "use_local": self.use_local,
                "local_channels": self.local_channels, "local_dim": self.local_dim,
                "use_peers": self.use_peers, "peer_dim": self.peer_dim,
                "use_robot_bev": self.use_robot_bev,
                "robot_bev_channels": self.robot_bev_channels,
                "robot_bev_dim": self.robot_bev_dim,
                "sequential_decode": self.sequential_decode}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TokenPolicy":
        return cls(**cfg)

    # ---- encoding -----------------------------------------------------------------------
    def encode(self, obs: ObsBatch, need_bev: bool = True):
        E, R, K = obs.tokens.shape[:3]
        tok = torch.nan_to_num(obs.tokens)
        geom, feat = tok[..., : self.geom_dim], tok[..., self.geom_dim:]
        h_tok = self.token_norm(self.token_mlp(geom) + self.feat_proj(feat)
                                + self.type_emb(obs.token_type.clamp_min(0)))
        h_q = self.robot_norm(self.robot_mlp(torch.nan_to_num(obs.robot_feat)))

        if self.use_local:
            if not obs.has_local:
                raise ValueError("TokenPolicy(use_local=True) needs a local crop in the ObsBatch")
            loc = torch.nan_to_num(obs.local)
            f = self.local_cnn(loc.reshape(E * R, *loc.shape[2:])).reshape(E, R, self.local_dim)
            h_q = h_q + self.local_proj(f)
        if self.use_robot_bev:
            if not obs.has_robot_bev:
                raise ValueError("TokenPolicy(use_robot_bev=True) needs robot_bev in the ObsBatch")
            rb = torch.nan_to_num(obs.robot_bev)
            f = self.rbev_cnn(rb.reshape(E * R, *rb.shape[2:])).reshape(E, R, self.robot_bev_dim)
            h_q = h_q + self.rbev_proj(f)

        # query tokens: the same feature projection as the items, plus the weight and a kind marker
        q_emb = torch.nan_to_num(obs.query_emb)
        h_qy = self.query_norm(self.feat_proj(q_emb)
                               + self.query_w_proj(torch.nan_to_num(obs.query_w).unsqueeze(-1))
                               + self.query_kind)
        Q = h_qy.shape[1]
        h_qy = h_qy.unsqueeze(1).expand(E, R, Q, self.d_model)

        parts = [h_q.unsqueeze(2), h_tok, h_qy]
        pads = [torch.zeros(E, R, 1, dtype=torch.bool, device=obs.device), ~obs.token_mask,
                (~obs.query_mask).unsqueeze(1).expand(E, R, Q)]
        if self.use_peers and obs.has_peers:
            pt = torch.nan_to_num(obs.peer_tokens)
            parts.append(self.peer_norm(self.peer_proj(pt) + self.peer_kind))
            pads.append(pt[..., PEER_VALID] <= 0)      # never heard from / padded team slot
        seq = torch.cat(parts, dim=2)
        n_seq = seq.shape[2]
        seq = seq.reshape(E * R, n_seq, self.d_model)
        pad = torch.cat(pads, dim=2)
        out = self.encoder(seq, src_key_padding_mask=pad.reshape(E * R, n_seq))
        q_out = out[:, 0].reshape(E, R, self.d_model)
        t_out = out[:, 1: 1 + K].reshape(E, R, K, self.d_model)
        return q_out, t_out, (self._bev_feat(obs) if need_bev else None)

    def _bev_feat(self, obs: ObsBatch):
        if not self.use_bev:
            return None
        if not obs.has_bev:
            raise ValueError("TokenPolicy(use_bev=True) needs a BEV in the ObsBatch")
        return self.bev_cnn(torch.nan_to_num(obs.bev))

    def forward(self, obs: ObsBatch):
        """-> logits [E, R, K] (env mask applied), value [E]."""
        q_out, t_out, bev_feat = self.encode(obs)
        return self._logits(q_out, t_out), self.value(q_out, bev_feat, obs.robot_mask)

    def actor_logits(self, obs: ObsBatch) -> torch.Tensor:
        """Actor head alone -> logits [E, R, K]. No critic, so no centralised BEV is needed: the
        imitation path never ships one (it is 377 kB per decision that nothing reads)."""
        q_out, t_out, _ = self.encode(obs, need_bev=False)
        return self._logits(q_out, t_out)

    def _logits(self, q_out: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
        return (self.q_proj(q_out).unsqueeze(2) * self.k_proj(t_out)).sum(-1) * self.logit_scale

    def value(self, q_out, bev_feat, robot_mask) -> torch.Tensor:
        w = self.value_attn(q_out).squeeze(-1)
        w = torch.softmax(_fill(w, ~robot_mask), dim=-1)
        pooled = (w.unsqueeze(-1) * q_out).sum(1)
        if self.use_bev:
            pooled = torch.cat([pooled, bev_feat], dim=-1)
        return self.value_head(pooled).squeeze(-1)

    # ---- sequential decode --------------------------------------------------------------
    @staticmethod
    def token_keys(obs: ObsBatch) -> torch.Tensor:
        """Claim key per token; 0 for hold/empty slots (never blocks a later robot)."""
        tid, tt = obs.token_id, obs.token_type
        key = tt * _TYPE_STRIDE + tid + 1
        return torch.where((tt == TOKEN_HOLD) | (tid < 0), torch.zeros_like(key), key)

    def _decode(self, logits: torch.Tensor, obs: ObsBatch, actions: torch.Tensor | None,
                deterministic: bool):
        """Robots act in index order; a non-hold token claimed earlier is masked for later robots.

        With `sequential_decode=False` there is no cross-robot mask: two robots may pick the same
        token in one decision, exactly as they can when each runs its own copy of the policy.
        """
        E, R, K = logits.shape
        keys = self.token_keys(obs)
        claimed = torch.zeros(E, R, dtype=keys.dtype, device=logits.device)
        acts, logps, ents = [], [], []
        for r in range(R):
            valid = obs.token_mask[:, r]
            if r > 0 and self.sequential_decode:
                busy = (keys[:, r].unsqueeze(-1) == claimed[:, :r].unsqueeze(1)).any(-1)
                valid = valid & ~(busy & (keys[:, r] != _HOLD_KEY))
            empty = ~valid.any(-1)
            if bool(empty.any()):
                valid = valid.clone()
                valid[empty, 0] = True              # degenerate slot: only the hold option remains
            lg = _fill(logits[:, r], ~valid)
            dist = torch.distributions.Categorical(logits=lg, validate_args=False)
            if actions is not None:
                a = actions[:, r]
            elif deterministic:
                a = lg.argmax(-1)
            else:
                a = dist.sample()
            acts.append(a)
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
            claimed[:, r] = keys[:, r].gather(-1, a.unsqueeze(-1)).squeeze(-1)
        return (torch.stack(acts, 1), torch.stack(logps, 1), torch.stack(ents, 1))

    # ---- api ----------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: ObsBatch, deterministic: bool = False):
        """-> actions [E, R] int64, logp [E, R], value [E]."""
        logits, value = self.forward(obs)
        a, logp, _ = self._decode(logits, obs, None, deterministic)
        return a, logp, value

    def valid_masks(self, obs: ObsBatch, actions: torch.Tensor | None = None) -> torch.Tensor:
        """[E, R, K] bool: the selectable set each robot decodes over.

        With `sequential_decode` the claims of the lower-index robots come from `actions` — the
        same tokens `_decode` would have masked, so re-evaluating a trajectory (PPO, BC) sees the
        distribution the rollout sampled from.
        """
        E, R, K = obs.token_mask.shape
        keys = self.token_keys(obs)
        out = torch.zeros(E, R, K, dtype=torch.bool, device=obs.token_mask.device)
        claimed = torch.zeros(E, R, dtype=keys.dtype, device=keys.device)
        for r in range(R):
            valid = obs.token_mask[:, r]
            if r > 0 and self.sequential_decode:
                busy = (keys[:, r].unsqueeze(-1) == claimed[:, :r].unsqueeze(1)).any(-1)
                valid = valid & ~(busy & (keys[:, r] != _HOLD_KEY))
            empty = ~valid.any(-1)
            if bool(empty.any()):
                valid = valid.clone()
                valid[empty, 0] = True
            out[:, r] = valid
            if actions is None:
                break
            claimed[:, r] = keys[:, r].gather(-1, actions[:, r].long().unsqueeze(-1)).squeeze(-1)
        return out

    def evaluate_full(self, obs: ObsBatch, actions: torch.Tensor):
        """-> logp [E, R], entropy [E, R], value [E], log-probs over every token [E, R, K]."""
        logits, value = self.forward(obs)
        lg = _fill(logits, ~self.valid_masks(obs, actions))
        dist = torch.distributions.Categorical(logits=lg, validate_args=False)
        a = actions.long()
        return dist.log_prob(a), dist.entropy(), value, torch.log_softmax(lg, dim=-1)

    def evaluate(self, obs: ObsBatch, actions: torch.Tensor):
        """-> logp [E, R], entropy [E, R], value [E] for the given actions."""
        logp, ent, value, _ = self.evaluate_full(obs, actions)
        return logp, ent, value

    @torch.no_grad()
    def act_tokens(self, obs: ObsBatch, deterministic: bool = False) -> torch.Tensor:
        """Actions alone, through the actor head only -> [E, R] int64.

        The imitation rollout never ships the centralised BEV (377 kB per decision that only the
        critic reads), so `act` — which computes V — cannot be used there.
        """
        a, _, _ = self._decode(self.actor_logits(obs), obs, None, deterministic)
        return a

    @torch.no_grad()
    def action_probs(self, obs: ObsBatch) -> torch.Tensor:
        """Per-robot categorical probabilities under the decode rule (diagnostics/tests)."""
        logits, _ = self.forward(obs)
        E, R, K = logits.shape
        keys = self.token_keys(obs)
        claimed = torch.zeros(E, R, dtype=keys.dtype, device=logits.device)
        out = torch.zeros(E, R, K, device=logits.device)
        for r in range(R):
            valid = obs.token_mask[:, r]
            if r > 0 and self.sequential_decode:
                busy = (keys[:, r].unsqueeze(-1) == claimed[:, :r].unsqueeze(1)).any(-1)
                valid = valid & ~(busy & (keys[:, r] != _HOLD_KEY))
            empty = ~valid.any(-1)
            if bool(empty.any()):
                valid = valid.clone()
                valid[empty, 0] = True
            p = torch.softmax(_fill(logits[:, r], ~valid), dim=-1)
            out[:, r] = p
            claimed[:, r] = keys[:, r].gather(-1, p.argmax(-1, keepdim=True)).squeeze(-1)
        return out


def _fill(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """masked_fill with the dtype minimum: softmax gives exactly 0, log_prob stays finite."""
    return x.masked_fill(mask, torch.finfo(x.dtype).min)


def _orth(m: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


__all__ = ["TokenPolicy", "BevCNN", "fill_invalid"]


fill_invalid = _fill
