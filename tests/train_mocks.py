"""Mock envs/observations for the training tests (no simulator involved)."""
from __future__ import annotations

import numpy as np
import torch

from rlplanner.sim.state import PEER_FEAT_DIM, TOKEN_HOLD
from rlplanner.sim.vec_env import VecObs
from rlplanner.train.obs import ObsBatch


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def random_vec_obs(n_envs: int = 4, n_robots: int = 3, k_tokens: int = 8, token_dim: int = 12,
                   robot_dim: int = 6, bev: tuple[int, int, int] = (2, 8, 8),
                   p_mask: float = 0.35, shared_tokens: bool = True, feat_dim: int = 4,
                   n_queries: int = 1, q_max: int = 4, local: tuple[int, int, int] = (0, 0, 0),
                   rng: np.random.Generator | None = None) -> VecObs:
    """A padded observation with a valid hold slot 0 and randomly masked candidate slots.

    The last `feat_dim` token columns are unit-norm item features, matching the real observation
    (`state.F_FEAT0:`); the query block holds `n_queries` unit query vectors of the same width.
    """
    rng = rng or np.random.default_rng(0)
    E, R, K = n_envs, n_robots, k_tokens
    tokens = rng.random((E, R, K, token_dim), np.float32)
    feats = _unit(rng.standard_normal((E, 1 if shared_tokens else R, K, feat_dim)).astype(np.float32))
    tokens[..., token_dim - feat_dim:] = np.broadcast_to(feats, (E, R, K, feat_dim))
    ttype = rng.integers(1, 4, size=(E, 1 if shared_tokens else R, K)).astype(np.int8)
    tid = rng.integers(0, 1000, size=(E, 1 if shared_tokens else R, K)).astype(np.int32)
    ttype = np.broadcast_to(ttype, (E, R, K)).copy()
    tid = np.broadcast_to(tid, (E, R, K)).copy()
    ttype[:, :, 0] = TOKEN_HOLD
    tid[:, :, 0] = -1
    mask = rng.random((E, 1 if shared_tokens else R, K)) > p_mask
    mask = np.broadcast_to(mask, (E, R, K)).copy()
    mask[:, :, 0] = True
    qe = np.zeros((E, q_max, feat_dim), np.float32)
    qw = np.zeros((E, q_max), np.float32)
    qm = np.zeros((E, q_max), np.bool_)
    qe[:, :n_queries] = _unit(rng.standard_normal((E, n_queries, feat_dim)).astype(np.float32))
    qw[:, :n_queries] = 1.0
    qm[:, :n_queries] = True
    loc = (np.zeros((E, R, 0, 0), np.float32) if local[0] == 0
           else rng.random((E, R) + local, np.float32))
    return VecObs(tokens=tokens, token_mask=mask, token_xy=np.zeros((E, R, K, 2), np.float32),
                  token_type=ttype, token_id=tid,
                  robot_feat=rng.random((E, R, robot_dim), np.float32),
                  bev=rng.random((E,) + bev, np.float32),
                  robot_mask=np.ones((E, R), np.bool_), t=np.zeros(E),
                  query_emb=qe, query_w=qw, query_mask=qm, local=loc,
                  peer_tokens=np.zeros((E, R, max(R - 1, 0), PEER_FEAT_DIM), np.float32),
                  robot_bev=None)


def random_obs_batch(device="cpu", with_bev: bool = False, **kw) -> ObsBatch:
    return ObsBatch.from_vec_obs(random_vec_obs(**kw), device, with_bev=with_bev)


class BanditVecEnv:
    """Contextual bandit over tokens: team reward = cos(token feature, mission query).

    The value of a token is *not* a column of the token vector — it only exists relative to the
    query token that comes with the observation, so a policy that ignores the query block cannot
    beat chance. That is exactly the relevance the real policy has to learn by attention.

    Quacks like `sim.vec_env.VecEnv` (step -> VecObs, rewards, dones, infos) so `Collector` can
    drive it. Every step terminates, so returns are the immediate reward.
    """

    def __init__(self, n_envs: int = 8, n_robots: int = 1, k_tokens: int = 8,
                 token_dim: int = 12, robot_dim: int = 6, feat_dim: int = 4,
                 n_queries: int = 1, p_mask: float = 0.25, seed: int = 0):
        self.n_envs, self.n_robots, self.k_tokens = n_envs, n_robots, k_tokens
        self.token_dim, self.robot_dim = token_dim, robot_dim
        self.feat_dim = int(feat_dim)
        self.n_queries = int(n_queries)
        self.p_mask = float(p_mask)
        self.rng = np.random.default_rng(seed)
        self.obs = self._draw()

    def _draw(self) -> VecObs:
        return random_vec_obs(self.n_envs, self.n_robots, self.k_tokens, self.token_dim,
                              self.robot_dim, bev=(1, 8, 8), p_mask=self.p_mask,
                              shared_tokens=True, feat_dim=self.feat_dim,
                              n_queries=self.n_queries, rng=self.rng)

    def reset(self, seeds=None) -> VecObs:
        self.obs = self._draw()
        return self.obs

    def values(self, obs: VecObs | None = None) -> np.ndarray:
        """[E, R, K] cosine of every token's feature against the best of the env's *real* queries."""
        o = self.obs if obs is None else obs
        f = o.tokens[..., self.token_dim - self.feat_dim:]
        c = np.einsum("erkd,eqd->erkq", f, o.query_emb)
        return np.where(o.query_mask[:, None, None, :], c, -np.inf).max(-1)

    def optimal(self, obs: VecObs | None = None) -> np.ndarray:
        o = self.obs if obs is None else obs
        return np.where(o.token_mask, self.values(o), -np.inf).argmax(-1)

    def step(self, actions):
        a = np.asarray(actions)[:, : self.n_robots]
        val = np.take_along_axis(self.values(), a[..., None], -1)[..., 0]
        rewards = val.mean(-1).astype(np.float64)
        dones = np.ones(self.n_envs, np.bool_)
        opt = (a == self.optimal()).mean(-1)
        infos = [{"metrics": {"frac_found": float(opt[i])},
                  "final_info": {"metrics": {"frac_found": float(opt[i])}}}
                 for i in range(self.n_envs)]
        self.obs = self._draw()
        return self.obs, rewards, dones, infos

    def close(self) -> None:
        pass


@torch.no_grad()
def optimal_fraction(env: BanditVecEnv, policy, n_batches: int = 20) -> float:
    """Fraction of deterministic picks that maximise the query cosine."""
    hits, tot = 0, 0
    for _ in range(n_batches):
        obs = env.reset()
        ob = ObsBatch.from_vec_obs(obs, with_bev=policy.use_bev, with_local=policy.use_local)
        a, _, _ = policy.act(ob, deterministic=True)
        hits += int((a.cpu().numpy() == env.optimal(obs)).sum())
        tot += a.numel()
    return hits / max(1, tot)


__all__ = ["random_vec_obs", "random_obs_batch", "BanditVecEnv", "optimal_fraction"]
