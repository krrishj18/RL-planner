"""DAgger: the beta schedule, the labels, the BC loss, and the PPO fine-tune knobs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rlplanner.sim.config import EnvConfig
from rlplanner.sim.state import TOKEN_FRONTIER, TOKEN_HOLD, TOKEN_VISITED
from rlplanner.train.evaluate import load_checkpoint
from rlplanner.train.imitation import (DaggerConfig, Imitator, LabelBuffer, agreement,
                                       bc_losses, beta_schedule, type_weights)
from rlplanner.train.par_env import SerialVecEnv, run_episode
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig, bc_reference, split_params
from rlplanner.train.scenes import SceneBank
from rlplanner.train.teachers import OracleSweepPolicy, make_any, make_teacher

from train_mocks import random_obs_batch
from test_train_ppo import fixed_batch

ROOT = Path(__file__).resolve().parents[1]
SCENES = "synthetic:0-4"
TOKEN_DIM, ROBOT_DIM, K, FEAT = 12, 6, 8, 4


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_cfg(n_robots: int = 2, t_max: float = 40.0, local: int = 0) -> EnvConfig:
    cfg = EnvConfig()
    cfg.robot.n_robots = n_robots
    cfg.t_max_s = t_max
    cfg.tokens.local_size = int(local)
    cfg.tokens.robot_bev_size = 0
    assert not cfg.validate()
    return cfg


def tiny_policy(seq: bool = True, **kw) -> TokenPolicy:
    return TokenPolicy(token_dim=TOKEN_DIM, robot_dim=ROBOT_DIM, feat_dim=FEAT, d_model=32,
                       n_layers=1, n_heads=2, dim_ff=32, sequential_decode=seq, **kw)


def labelled_batch(n: int = 24, n_robots: int = 3, seed: int = 0):
    ob = random_obs_batch(n_envs=n, n_robots=n_robots, k_tokens=K, token_dim=TOKEN_DIM,
                          robot_dim=ROBOT_DIM, p_mask=0.3, rng=np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    lab = np.zeros((n, n_robots), np.int64)
    m = ob.token_mask.numpy()
    for e in range(n):
        for r in range(n_robots):
            v = np.flatnonzero(m[e, r])
            lab[e, r] = int(rng.choice(v)) if v.size else 0
    return ob, torch.as_tensor(lab), ob.robot_mask.clone()


# ---- schedule ----------------------------------------------------------------------------------
def test_beta_schedule_starts_at_the_teacher_and_ends_at_the_student():
    b = [beta_schedule(i, 4, 0.5) for i in range(4)]
    assert b == [1.0, 0.5, 0.25, 0.0]
    assert all(b[i] >= b[i + 1] for i in range(3))
    assert beta_schedule(0, 1, 0.5) == 1.0 and beta_schedule(1, 2, 0.5) == 0.0
    assert beta_schedule(1, 6, 0.8) == pytest.approx(0.8)
    assert beta_schedule(5, 6, 0.8) == 0.0


# ---- teachers ----------------------------------------------------------------------------------
@pytest.mark.parametrize("teacher", ["oracle", "oracle_sweep"])
def test_teacher_only_picks_selectable_tokens(teacher):
    bank = SceneBank(SCENES)
    env = bank.make_env(bank.keys[0], tiny_cfg(), 0)
    pol = make_any(teacher, queries=env.cfg.rayfronts.queries, seed=0)
    obs = env.reset(0)
    for _ in range(12):
        a = pol.act(obs, env.state)
        assert a.shape == (env.n_robots,)
        for r, k in enumerate(a):
            assert bool(obs.token_mask[r, int(k)]), f"{teacher} picked a masked slot"
        obs, _, done, _ = env.step(a)
        if done:
            break


def test_oracle_sweep_never_picks_a_visited_token():
    bank = SceneBank(SCENES)
    env = bank.make_env(bank.keys[0], tiny_cfg(t_max=120.0), 1)
    pol = OracleSweepPolicy(queries=env.cfg.rayfronts.queries)
    obs = env.reset(1)
    seen_visited, picks = 0, []
    for _ in range(40):
        a = pol.act(obs, env.state)
        for r, k in enumerate(a):
            picks.append(int(obs.token_type[r, int(k)]))
        seen_visited += int((obs.token_type == TOKEN_VISITED).sum())
        obs, _, done, _ = env.step(a)
        if done:
            break
    assert seen_visited > 0, "the episode never offered a visited token: test is vacuous"
    assert TOKEN_VISITED not in picks


def test_oracle_sweep_falls_back_to_sweeping_when_nothing_is_in_reach():
    bank = SceneBank(SCENES)
    env = bank.make_env(bank.keys[0], tiny_cfg(t_max=120.0), 2)
    near = OracleSweepPolicy(queries=env.cfg.rayfronts.queries, teacher_radius_m=1e9)
    far = OracleSweepPolicy(queries=env.cfg.rayfronts.queries, teacher_radius_m=0.0)
    obs = env.reset(2)
    swept, homed = [], []
    for _ in range(25):
        a_far = far.act(obs, env.state)
        a_near = near.act(obs, env.state)
        swept += [int(obs.token_type[r, int(k)]) for r, k in enumerate(a_far)]
        homed += [int(obs.token_type[r, int(k)]) for r, k in enumerate(a_near)]
        obs, _, done, _ = env.step(a_near)
        if done:
            break
    # radius 0: nothing is ever in reach, so every pick comes from the sweep (or the hold fallback)
    assert set(swept) <= {TOKEN_FRONTIER, TOKEN_HOLD}
    # radius inf: it homes in, so it takes token types the sweep never would
    assert set(homed) - {TOKEN_FRONTIER, TOKEN_HOLD}


def test_dagger_step_labels_match_the_teacher_and_are_valid():
    cfg = tiny_cfg()
    vec = SerialVecEnv(SCENES, cfg, n_envs=2, robots=(2, 2), split="train", seed=0, n_workers=1,
                       send_bev=False)
    try:
        obs = vec.reset_all()
        for _ in range(5):
            mask = obs.token_mask.copy()
            a = np.zeros((2, vec.R), np.int64)
            obs, _, _, _, lab, val, exe = vec.dagger_step(a, np.ones(2, np.bool_),
                                                          "oracle_sweep", 150.0)
            assert (exe == lab).all(), "beta = 1 must execute the label"
            for e in range(2):
                for r in range(vec.R):
                    if val[e, r]:
                        assert bool(mask[e, r, lab[e, r]])
    finally:
        vec.close()


def test_dagger_step_runs_the_student_when_beta_is_zero():
    cfg = tiny_cfg()
    vec = SerialVecEnv(SCENES, cfg, n_envs=2, robots=(2, 2), split="train", seed=0, n_workers=1,
                       send_bev=False)
    try:
        obs = vec.reset_all()
        a = np.zeros((2, vec.R), np.int64)
        _, _, _, _, lab, val, exe = vec.dagger_step(a, np.zeros(2, np.bool_), "oracle_sweep",
                                                    150.0)
        assert (exe == 0).all()
        assert val.any() and lab.any(), "labels are still produced when the student acts"
    finally:
        vec.close()


# ---- loss --------------------------------------------------------------------------------------
def test_bc_labels_outside_the_decode_mask_are_dropped():
    torch.manual_seed(0)
    pol = tiny_policy(seq=False)             # no cross-robot claims: only the bad label is dropped
    ob, lab, val = labelled_batch(8)
    ob.token_mask[:, :, K - 1] = False          # a slot no robot may take
    lab[lab == K - 1] = 0                       # slot 0 (hold) is always selectable
    bad = lab.clone()
    bad[:, 0] = K - 1
    out = bc_losses(pol, ob, bad, val)
    assert float(out["dropped"]) == 8.0
    assert float(out["n"]) == float(val.sum()) - 8.0
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("seq", [True, False])
def test_cross_entropy_decreases_on_a_fixed_batch(seq):
    torch.manual_seed(0)
    pol = tiny_policy(seq=seq)
    ob, lab, val = labelled_batch(32)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    first = float(bc_losses(pol, ob, lab, val)["ce"].detach())
    for _ in range(60):
        out = bc_losses(pol, ob, lab, val, label_smoothing=0.05, ent_coef=0.0)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
    last = float(bc_losses(pol, ob, lab, val)["ce"].detach())
    assert last < first - 0.2, (first, last)
    assert float(bc_losses(pol, ob, lab, val)["accuracy"]) > 0.5


def test_type_weights_balance_a_skewed_label_set():
    w = type_weights(np.array([1000, 10, 0, 0, 0]))
    assert w[0] * 1000 == pytest.approx(w[1] * 10)
    assert float(w[2]) == 0.0


def test_label_buffer_round_trip_and_caps():
    ob, lab, val = labelled_batch(16)
    buf = LabelBuffer(max_samples=10 ** 9, max_gb=10.0, rng=np.random.default_rng(0))
    buf.add(ob, lab, val)
    assert len(buf) == 16 and buf.n_labels == int(val.sum())
    got, glab, gval = buf.gather(np.arange(16))
    assert got.tokens.dtype == torch.float32
    assert torch.allclose(got.tokens, ob.tokens, atol=1e-2)
    assert (glab == lab).all() and (gval == val).all()
    assert buf.bytes_per_decision > 0
    small = LabelBuffer(max_samples=10 ** 9, max_gb=buf.nbytes / 4e9,
                        rng=np.random.default_rng(0))
    small.add(ob, lab, val)
    assert 0 < len(small) < 16 and small.seen_decisions == 16


def test_imitator_trains_and_reports_agreement():
    torch.manual_seed(0)
    pol = tiny_policy()
    imi = Imitator(pol, DaggerConfig(epochs=8, batch=8, lr=3e-3, balance=False), seed=0)
    ob, lab, val = labelled_batch(24)
    imi.buffer.add(ob, lab, val)
    a0 = agreement(pol, imi.buffer, "cpu")
    imi.train_epochs()
    assert agreement(pol, imi.buffer, "cpu") > a0
    assert "policy_config" in imi.state_dict()


# ---- fine-tune knobs ---------------------------------------------------------------------------
def test_frozen_actor_update_leaves_the_actor_bit_identical():
    torch.manual_seed(0)
    pol = tiny_policy()
    ppo = PPO(pol, PPOConfig(epochs=2, n_minibatches=2, target_kl=None))
    actor, critic = split_params(pol)
    before = [p.detach().clone() for p in actor]
    cbefore = [p.detach().clone() for p in critic]
    ppo.freeze_actor(True)
    ppo.update(fixed_batch(pol, n=32))
    assert all(torch.equal(a, b) for a, b in zip(actor, before))
    assert not all(torch.equal(a, b) for a, b in zip(critic, cbefore))
    ppo.freeze_actor(False)
    ppo.update(fixed_batch(pol, n=32))
    assert not all(torch.equal(a, b) for a, b in zip(actor, before))


def test_bc_kl_is_zero_against_itself_and_at_the_end_of_the_anneal():
    train = load_script("train")
    assert train.bc_kl_coef(1, 100, 0.05) == pytest.approx(0.05)
    assert train.bc_kl_coef(100, 100, 0.05) == 0.0
    assert train.bc_kl_coef(1, 1, 0.05) == 0.0
    assert train._actor_ramp(1, 5, 10) == 0.0 and train._actor_ramp(15, 5, 10) == 1.0

    torch.manual_seed(0)
    pol = tiny_policy()
    ppo = PPO(pol, PPOConfig(bc_kl_coef=0.05, target_kl=None))
    batch = fixed_batch(pol, n=16)
    batch.ref_logp = bc_reference(pol, batch)
    idx = torch.arange(len(batch))
    assert float(ppo._losses(batch, idx)["bc_kl"]) == pytest.approx(0.0, abs=1e-6)
    ppo.bc_kl_coef = 0.0
    assert float(ppo._losses(batch, idx)["bc_kl"]) == 0.0


def test_init_from_carries_the_execution_model(tmp_path):
    train = load_script("train")
    pol = tiny_policy(seq=False, use_local=True, local_channels=3)
    ck = {"policy": pol.state_dict(), "policy_config": pol.config(), "variant": "decentral_x"}
    torch.save(ck, tmp_path / "bc.pt")
    a = train.build_parser().parse_args(["--init-from", str(tmp_path / "bc.pt")])
    assert a.sequential_decode is True                    # the parser default
    train.load_init(a)
    assert a.sequential_decode is False and a.use_local is True and a.variant == "decentral_x"
    loaded, raw = load_checkpoint(tmp_path / "bc.pt")
    assert loaded.sequential_decode is False
    assert all(torch.equal(x, y) for x, y in zip(loaded.state_dict().values(),
                                                 pol.state_dict().values()))


def test_teacher_is_evaluable_like_a_baseline():
    bank = SceneBank(SCENES)
    cfg = tiny_cfg(t_max=40.0)
    env = bank.make_env(bank.keys[0], cfg, 0)
    row = run_episode(env, make_teacher("oracle_sweep", cfg.rayfronts.queries), 0,
                      max_decisions=6)
    assert np.isfinite(row["reward"]) and "frac_found" in row
