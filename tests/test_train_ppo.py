"""PPO/MAPPO: GAE, a real update on a fixed batch, and learnability on a token bandit."""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from rlplanner.train.obs import ObsBatch
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig
from rlplanner.train.rollout import Collector, RolloutBatch, gae

from train_mocks import BanditVecEnv, optimal_fraction, random_obs_batch

TOKEN_DIM, ROBOT_DIM, K, FEAT = 12, 6, 8, 4


def fixed_batch(policy: TokenPolicy, n: int = 128, n_robots: int = 3, seed: int = 0
                ) -> RolloutBatch:
    torch.manual_seed(seed)
    ob = random_obs_batch(n_envs=n, n_robots=n_robots, k_tokens=K, token_dim=TOKEN_DIM,
                          robot_dim=ROBOT_DIM, p_mask=0.3)
    a, lp, v = policy.act(ob)
    g = torch.Generator().manual_seed(seed)
    adv = torch.randn(n, generator=g)
    return RolloutBatch(obs=ob, actions=a, logp=lp, values=v.detach(),
                        returns=v.detach() + adv, advantages=adv,
                        robot_mask=ob.robot_mask, n_steps=1, n_envs=n)


def test_gae_matches_discounted_returns_without_lambda():
    T, E = 5, 2
    r = torch.ones(T, E)
    v = torch.zeros(T, E)
    d = torch.zeros(T, E)
    adv = gae(r, v, d, torch.zeros(E), gamma=0.5, lam=1.0)
    expect = torch.tensor([1 + 0.5 + 0.25 + 0.125 + 0.0625, 1 + 0.5 + 0.25 + 0.125, 1.75, 1.5, 1.0])
    assert torch.allclose(adv[:, 0], expect, atol=1e-6)


def test_gae_does_not_bootstrap_across_done():
    T, E = 3, 1
    r = torch.zeros(T, E)
    v = torch.full((T, E), 10.0)
    d = torch.ones(T, E)
    adv = gae(r, v, d, torch.full((E,), 10.0), gamma=0.99, lam=0.95)
    assert torch.allclose(adv, torch.full((T, E), -10.0), atol=1e-6)


def test_update_lowers_surrogate_on_fixed_batch():
    torch.manual_seed(0)
    pol = TokenPolicy(TOKEN_DIM, ROBOT_DIM, FEAT, d_model=64)
    batch = fixed_batch(pol)
    ppo = PPO(pol, PPOConfig(n_minibatches=4, epochs=4), "cpu")
    before = ppo.surrogate(batch)
    out = ppo.update(batch)
    after = ppo.surrogate(batch)
    assert np.isfinite(before) and np.isfinite(after)
    assert after < before, f"surrogate did not improve: {before} -> {after}"
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm"):
        assert np.isfinite(out[k]), k


def test_update_only_uses_real_robots():
    torch.manual_seed(0)
    pol = TokenPolicy(TOKEN_DIM, ROBOT_DIM, FEAT, d_model=32)
    batch = fixed_batch(pol, n=32, n_robots=3)
    batch.robot_mask[:, 2] = False
    ppo = PPO(pol, PPOConfig(epochs=1, n_minibatches=2), "cpu")
    out = ppo.update(batch)
    assert np.isfinite(out["policy_loss"]) and np.isfinite(out["entropy"])


@pytest.mark.parametrize("device", ["cpu"])
def test_learns_token_bandit(device):
    """Contextual bandit over tokens: reward = cos(token feature, the mission query token).

    This is the key learnability test of the open-set observation. The value of a token is not a
    column of the token vector — it exists only relative to the query token that arrives with the
    observation, so a policy that ignores the query block cannot beat chance and the relevance has
    to come out of the cross-attention.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    n_updates, lr = 150, 1e-3
    env = BanditVecEnv(n_envs=64, n_robots=1, k_tokens=K, token_dim=TOKEN_DIM,
                       robot_dim=ROBOT_DIM, feat_dim=FEAT, seed=0)
    pol = TokenPolicy(TOKEN_DIM, ROBOT_DIM, FEAT, d_model=128, n_heads=4, n_layers=2).to(device)
    # no target_kl: the update is a bandit, so an early stop only slows the fit down
    ppo = PPO(pol, PPOConfig(lr=lr, epochs=4, n_minibatches=4, ent_coef=0.003, target_kl=None),
              device)
    col = Collector(env, device, gamma=0.99, lam=0.95, obs=env.reset())
    t0 = time.perf_counter()
    for u in range(n_updates):
        ppo.set_lr(lr * max(0.1, 1.0 - u / n_updates))
        batch, _ = col.rollout(pol, 16)
        ppo.update(batch)
    wall = time.perf_counter() - t0
    frac = optimal_fraction(env, pol, n_batches=40)
    assert frac >= 0.9, f"only {frac:.2%} optimal picks after {n_updates} updates ({wall:.0f}s)"
    assert wall < 240.0, f"learnability run took {wall:.0f}s"


def test_the_bandit_is_unsolvable_without_the_query_token():
    """Sanity on the test itself: with the query zeroed out there is nothing to rank by, so a
    policy that scored tokens on their own features could not do better than chance."""
    env = BanditVecEnv(n_envs=8, n_robots=1, k_tokens=K, token_dim=TOKEN_DIM,
                       robot_dim=ROBOT_DIM, feat_dim=FEAT, seed=1)
    o = env.reset()
    best = env.optimal(o)
    o2 = env.reset()
    o2.tokens, o2.token_mask = o.tokens, o.token_mask       # same items, different query
    assert not np.array_equal(best, env.optimal(o2))
