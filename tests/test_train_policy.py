"""TokenPolicy: shapes, masking, sequential decode, act/evaluate consistency, checkpoints."""
from __future__ import annotations

import numpy as np

from rlplanner.sim.state import PEER_FEAT_DIM
import pytest
import torch

from rlplanner.sim.config import EnvConfig
from rlplanner.sim.state import TOKEN_HOLD, ROBOT_FEAT_DIM, TOKEN_FIXED
from rlplanner.sim.vec_env import VecObs
from rlplanner.train.obs import ObsBatch
from rlplanner.train.policy import TokenPolicy

from train_mocks import random_obs_batch, random_vec_obs

CFG = EnvConfig()
K = CFG.k_tokens
FEAT = 4                      # mock feature width (the real one is rayfronts.embedding_dim)
F = TOKEN_FIXED + FEAT
D = ROBOT_FEAT_DIM


def make_policy(**kw) -> TokenPolicy:
    torch.manual_seed(0)
    return TokenPolicy(token_dim=kw.pop("token_dim", F), robot_dim=kw.pop("robot_dim", D),
                       feat_dim=kw.pop("feat_dim", FEAT), **kw)


@pytest.mark.parametrize("n", [1, 3, 10])
def test_forward_shapes(n):
    pol = make_policy()
    ob = random_obs_batch(n_envs=5, n_robots=n, k_tokens=K, token_dim=F, robot_dim=D)
    logits, value = pol(ob)
    assert logits.shape == (5, n, K)
    assert value.shape == (5,)
    a, lp, v = pol.act(ob)
    assert a.shape == (5, n) and lp.shape == (5, n) and v.shape == (5,)
    assert torch.isfinite(lp).all() and torch.isfinite(v).all()
    assert a.dtype == torch.int64


def test_k_tokens_from_config():
    assert K == (1 + CFG.tokens.k_frontier + CFG.tokens.k_ray + CFG.tokens.k_segment
                 + CFG.tokens.k_visited)
    ob = random_obs_batch(n_envs=2, n_robots=2, k_tokens=K, token_dim=F, robot_dim=D)
    assert ob.k_tokens == K and ob.token_dim == F and ob.robot_dim == D


def test_bev_head():
    """The BEV is the *critic's* global input under the CTDE split."""
    pol = make_policy(use_bev=True, bev_channels=9)
    v = random_vec_obs(n_envs=3, n_robots=2, k_tokens=K, token_dim=F, robot_dim=D,
                       bev=(9, 64, 64))
    ob = ObsBatch.from_vec_obs(v, with_bev=True)
    logits, value = pol(ob)
    assert logits.shape == (3, 2, K) and value.shape == (3,)
    with pytest.raises(ValueError):
        pol(ob.drop_bev())


def test_masked_tokens_have_zero_probability():
    pol = make_policy()
    ob = random_obs_batch(n_envs=8, n_robots=3, k_tokens=K, token_dim=F, robot_dim=D, p_mask=0.6)
    p = pol.action_probs(ob)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5)
    assert float(p[~ob.token_mask].abs().max()) == 0.0


def test_masked_tokens_never_sampled_10k():
    pol = make_policy()
    torch.manual_seed(1)
    ob = random_obs_batch(n_envs=250, n_robots=4, k_tokens=K, token_dim=F, robot_dim=D,
                          p_mask=0.5)
    n = 0
    for _ in range(10):
        a, lp, _ = pol.act(ob)
        picked = torch.gather(ob.token_mask, 2, a.unsqueeze(-1)).squeeze(-1)
        assert bool(picked.all()), "sampled a masked token"
        assert torch.isfinite(lp).all()
        n += a.numel()
    assert n >= 10_000


def test_sequential_decode_no_duplicate_tokens():
    """Robots decode in index order; a claimed non-hold (type, id) is masked for later robots."""
    pol = make_policy()
    torch.manual_seed(2)
    ob = random_obs_batch(n_envs=64, n_robots=4, k_tokens=K, token_dim=F, robot_dim=D,
                          p_mask=0.2, shared_tokens=True)
    for _ in range(5):
        a, _, _ = pol.act(ob)
        tt = torch.gather(ob.token_type, 2, a.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        tid = torch.gather(ob.token_id, 2, a.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        for e in range(a.shape[0]):
            # ids are per-type counters in the sim, so the claim key is (type, id), not the id
            keys = [(int(t), int(i)) for t, i in zip(tt[e], tid[e]) if int(t) != TOKEN_HOLD]
            assert len(keys) == len(set(keys)), f"duplicate token claimed in env {e}: {keys}"


def test_independent_decode_lets_two_robots_pick_one_token():
    """`sequential_decode=False` is the decentralised-execution rule: nothing masks a token a
    peer chose in the *same* decision, because that choice has not been gossiped yet."""
    torch.manual_seed(2)
    ob = random_obs_batch(n_envs=64, n_robots=4, k_tokens=K, token_dim=F, robot_dim=D,
                          p_mask=0.2, shared_tokens=True)
    dup = {}
    for seq in (True, False):
        pol = make_policy(sequential_decode=seq)
        assert pol.sequential_decode is seq
        torch.manual_seed(3)
        n = 0
        for _ in range(5):
            a, lp, _ = pol.act(ob)
            assert torch.isfinite(lp).all()
            tt = torch.gather(ob.token_type, 2, a.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            tid = torch.gather(ob.token_id, 2, a.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for e in range(a.shape[0]):
                keys = [(int(t), int(i)) for t, i in zip(tt[e], tid[e]) if int(t) != TOKEN_HOLD]
                n += len(keys) - len(set(keys))
        dup[seq] = n
    assert dup[True] == 0, "the claim mask must still hold under centralised execution"
    assert dup[False] > 0, "independent decode must be able to collide on one token"


def test_independent_decode_is_carried_by_the_checkpoint():
    pol = make_policy(sequential_decode=False)
    assert pol.config()["sequential_decode"] is False
    clone = TokenPolicy.from_config(pol.config())
    assert clone.sequential_decode is False
    assert TokenPolicy.from_config({k: v for k, v in pol.config().items()
                                    if k != "sequential_decode"}).sequential_decode is True


def test_independent_decode_keeps_the_env_mask():
    """No cross-robot mask is not no mask: a masked token still has probability 0."""
    pol = make_policy(sequential_decode=False)
    torch.manual_seed(5)
    ob = random_obs_batch(n_envs=16, n_robots=3, k_tokens=K, token_dim=F, robot_dim=D, p_mask=0.6)
    p = pol.action_probs(ob)
    assert float(p[~ob.token_mask].abs().max()) == 0.0
    a, lp, _ = pol.act(ob)
    assert bool(torch.gather(ob.token_mask, 2, a.unsqueeze(-1)).all())
    lp2, ent, _ = pol.evaluate(ob, a)
    assert torch.allclose(lp, lp2, atol=1e-6) and (ent >= 0).all()


def test_sequential_decode_falls_back_to_hold():
    """More robots than distinct candidates -> the surplus robots can only hold."""
    pol = make_policy(token_dim=6, robot_dim=4, feat_dim=2)
    E, R, Kt = 3, 5, 3
    tokens = np.zeros((E, R, Kt, 6), np.float32)
    ttype = np.zeros((E, R, Kt), np.int8)
    tid = np.full((E, R, Kt), -1, np.int32)
    ttype[:, :, 1:] = 1
    tid[:, :, 1] = 7
    tid[:, :, 2] = 9
    mask = np.ones((E, R, Kt), np.bool_)
    v = VecObs(tokens=tokens, token_mask=mask, token_xy=np.zeros((E, R, Kt, 2), np.float32),
               token_type=ttype, token_id=tid, robot_feat=np.zeros((E, R, 4), np.float32),
               bev=np.zeros((E, 1, 4, 4), np.float32), robot_mask=np.ones((E, R), np.bool_),
               t=np.zeros(E), query_emb=np.zeros((E, 2, 2), np.float32),
               query_w=np.zeros((E, 2), np.float32), query_mask=np.zeros((E, 2), np.bool_),
               local=np.zeros((E, R, 0, 0), np.float32),
               peer_tokens=np.zeros((E, R, R - 1, PEER_FEAT_DIM), np.float32), robot_bev=None)
    ob = ObsBatch.from_vec_obs(v)
    a, lp, _ = pol.act(ob)
    assert torch.isfinite(lp).all()
    for e in range(E):
        picks = [int(x) for x in a[e]]
        non_hold = [p for p in picks if p != 0]
        assert len(non_hold) == len(set(non_hold)) <= 2


def test_padded_robots_are_finite():
    """A padded (non-existent) robot has no valid token; its logp must stay finite."""
    pol = make_policy()
    v = random_vec_obs(n_envs=4, n_robots=3, k_tokens=K, token_dim=F, robot_dim=D)
    v.robot_mask[:, 2] = False
    v.token_mask[:, 2] = False
    ob = ObsBatch.from_vec_obs(v)
    a, lp, val = pol.act(ob)
    assert torch.isfinite(lp).all() and torch.isfinite(val).all()
    assert float(lp[:, 2].abs().max()) == 0.0     # single forced option -> log p = 0
    lp2, ent, _ = pol.evaluate(ob, a)
    assert torch.isfinite(ent).all()


def test_evaluate_matches_act():
    pol = make_policy()
    torch.manual_seed(3)
    ob = random_obs_batch(n_envs=16, n_robots=3, k_tokens=K, token_dim=F, robot_dim=D)
    a, lp, v = pol.act(ob)
    lp2, ent, v2 = pol.evaluate(ob, a)
    assert torch.allclose(lp, lp2, atol=1e-6)
    assert torch.allclose(v, v2, atol=1e-6)
    assert (ent >= 0).all()


def test_checkpoint_roundtrip(tmp_path):
    pol = make_policy()
    ob = random_obs_batch(n_envs=6, n_robots=3, k_tokens=K, token_dim=F, robot_dim=D)
    a0, lp0, v0 = pol.act(ob, deterministic=True)
    p = tmp_path / "ck.pt"
    torch.save({"policy": pol.state_dict(), "policy_config": pol.config()}, p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    pol2 = TokenPolicy.from_config(ck["policy_config"])
    pol2.load_state_dict(ck["policy"])
    a1, lp1, v1 = pol2.act(ob, deterministic=True)
    assert torch.equal(a0, a1)
    assert torch.allclose(lp0, lp1, atol=1e-6) and torch.allclose(v0, v1, atol=1e-6)


def test_obs_batch_padding_and_indexing():
    from rlplanner.sim.state import TeamObs
    obs = [TeamObs(tokens=np.zeros((n, K, F), np.float32),
                   token_mask=np.ones((n, K), np.bool_),
                   token_xy=np.full((n, K, 2), np.nan, np.float32),
                   token_type=np.zeros((n, K), np.int8),
                   token_id=np.full((n, K), -1, np.int32),
                   robot_feat=np.zeros((n, D), np.float32),
                   bev=np.zeros((9, 8, 8), np.float32),
                   query_emb=np.zeros((4, FEAT), np.float32), query_w=np.zeros(4, np.float32),
                   query_mask=np.zeros(4, np.bool_), t=0.0) for n in (1, 3, 2)]
    b = ObsBatch.from_team_obs(obs)
    assert b.tokens.shape == (3, 3, K, F)
    assert b.robot_mask.sum(1).tolist() == [1, 3, 2]
    assert b.index(torch.tensor([0, 2])).n_envs == 2
    assert ObsBatch.cat([b, b]).n_envs == 6


def test_query_tokens_are_masked_and_change_the_logits():
    """The query block is an *input*: padded queries must not leak, and swapping the mission query
    must move the ranking (that is the whole point of learning relevance by attention)."""
    pol = make_policy()
    torch.manual_seed(4)
    v = random_vec_obs(n_envs=6, n_robots=2, k_tokens=K, token_dim=F, robot_dim=D, feat_dim=FEAT,
                       n_queries=1, q_max=8)
    ob = ObsBatch.from_vec_obs(v)
    l0, _ = pol(ob)
    v2 = random_vec_obs(n_envs=6, n_robots=2, k_tokens=K, token_dim=F, robot_dim=D, feat_dim=FEAT,
                        n_queries=1, q_max=8, rng=np.random.default_rng(7))
    v2.tokens, v2.token_mask, v2.token_type = v.tokens, v.token_mask, v.token_type
    v2.token_id, v2.robot_feat = v.token_id, v.robot_feat
    l1, _ = pol(ObsBatch.from_vec_obs(v2))
    assert not torch.allclose(l0, l1, atol=1e-5), "the mission query does not reach the logits"
    # garbage in the padded query slots must not change anything
    v3 = random_vec_obs(n_envs=6, n_robots=2, k_tokens=K, token_dim=F, robot_dim=D, feat_dim=FEAT,
                        n_queries=1, q_max=8)
    v3.query_emb[:, 1:] = 5.0
    l2, _ = pol(ObsBatch.from_vec_obs(v3))
    assert torch.allclose(l0, l2, atol=1e-5)


@pytest.mark.parametrize("nq", [1, 8])
def test_one_and_eight_queries_run(nq):
    pol = make_policy()
    ob = random_obs_batch(n_envs=3, n_robots=2, k_tokens=K, token_dim=F, robot_dim=D,
                          feat_dim=FEAT, n_queries=nq, q_max=8)
    assert int(ob.query_mask.sum()) == 3 * nq
    logits, value = pol(ob)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_local_crop_head():
    """The ego-centric crop is the *actor's* dense input."""
    pol = make_policy(use_local=True, local_channels=5)
    v = random_vec_obs(n_envs=3, n_robots=2, k_tokens=K, token_dim=F, robot_dim=D, feat_dim=FEAT,
                       local=(5, 32, 32))
    ob = ObsBatch.from_vec_obs(v)
    logits, value = pol(ob)
    assert logits.shape == (3, 2, K) and torch.isfinite(logits).all()
    with pytest.raises(ValueError):
        pol(ob.drop_local())
