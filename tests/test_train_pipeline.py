"""End-to-end: scene bank, env pool, rollout on real envs, evaluation and the CLIs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rlplanner.sim.config import EnvConfig
from rlplanner.sim.state import TOKEN_FIXED
from rlplanner.train.evaluate import (EVAL_COLS, TorchActor, evaluate_policy, format_table,
                                      make_actor, summarise)
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig
from rlplanner.train.rollout import Collector, EnvPool
from rlplanner.train.scenes import SceneBank, parse_robots, parse_scenes

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_cfg(n_robots: int = 2, t_max: float = 60.0) -> EnvConfig:
    cfg = EnvConfig()
    cfg.robot.n_robots = n_robots
    cfg.t_max_s = t_max
    assert not cfg.validate()
    return cfg


# ---- scenes ------------------------------------------------------------------------------------
def test_parse_scenes_and_split():
    keys = parse_scenes("synthetic:0-20")
    assert len(keys) == 20 and keys[0].ref == "0" and keys[-1].ref == "19"
    assert parse_scenes("synthetic:5") == parse_scenes("synthetic:0-5")
    bank = SceneBank("synthetic:0-20")
    assert len(bank.train) == 18 and len(bank.heldout) == 2
    assert not set(bank.train) & set(bank.heldout)
    with pytest.raises(ValueError):
        parse_scenes("synthetic:5-5")
    with pytest.raises(ValueError):
        parse_scenes("no/such/dir/*.json")


def test_parse_robots():
    assert parse_robots("3") == (3, 3)
    assert parse_robots("2-4") == (2, 4)
    with pytest.raises(ValueError):
        parse_robots("many")


def test_scene_bank_caches_raster():
    bank = SceneBank("synthetic:0-2")
    k = bank.keys[0]
    assert bank.scene(k) is bank.scene(k)
    assert bank.raster(k, 2.0) is bank.raster(k, 2.0)
    e1 = bank.make_env(k, tiny_cfg(), 0)
    e2 = bank.make_env(k, tiny_cfg(), 1)
    assert e1.raster is e2.raster


def test_scene_files_glob(tmp_path):
    from rlplanner.scene import schema
    for s in range(3):
        schema.make_synthetic_scene(s).to_json(tmp_path / f"s{s}.json")
    bank = SceneBank(str(tmp_path / "*.json"))
    assert len(bank) == 3 and len(bank.heldout) == 1
    assert bank.scene(bank.keys[0]) is not None


# ---- rollout on real envs ----------------------------------------------------------------------
def test_env_pool_variable_robots():
    bank = SceneBank("synthetic:0-4")
    pool = EnvPool(bank, tiny_cfg(), n_envs=3, robots=(2, 4), seed=0)
    counts = [e.n_robots for e in pool.envs]
    assert all(2 <= c <= 4 for c in counts)
    assert pool.obs.robot_mask.sum(1).tolist() == counts
    pool.close()


def test_rollout_and_update_on_real_envs():
    torch.manual_seed(0)
    bank = SceneBank("synthetic:0-4")
    cfg = tiny_cfg(n_robots=2)
    pool = EnvPool(bank, cfg, n_envs=2, robots=(2, 3), seed=0)
    col = Collector(pool.vec, "cpu", obs=pool.obs)
    ob = col.obs
    policy = TokenPolicy(ob.tokens.shape[3], ob.robot_feat.shape[2], ob.query_emb.shape[2], d_model=64)
    ppo = PPO(policy, PPOConfig(epochs=2, n_minibatches=2), "cpu")
    for _ in range(3):
        batch, stats = col.rollout(policy, 4)
        assert len(batch) == 8 and batch.actions.shape[1] == pool.obs.tokens.shape[1]
        assert torch.isfinite(batch.logp).all() and torch.isfinite(batch.advantages).all()
        out = ppo.update(batch)
        assert np.isfinite(out["policy_loss"])
    pool.close()


def test_rollout_actions_are_always_legal():
    """The env raises on a masked action; a full rollout is the strongest mask test."""
    bank = SceneBank("synthetic:0-2")
    pool = EnvPool(bank, tiny_cfg(n_robots=3), n_envs=2, robots=(3, 3), seed=1)
    col = Collector(pool.vec, "cpu", obs=pool.obs)
    ob = col.obs
    policy = TokenPolicy(ob.tokens.shape[3], ob.robot_feat.shape[2], ob.query_emb.shape[2], d_model=32)
    col.rollout(policy, 12)
    pool.close()


# ---- evaluation --------------------------------------------------------------------------------
def test_evaluate_baseline_and_policy():
    bank = SceneBank("synthetic:0-4")
    cfg = tiny_cfg(n_robots=2)
    res = evaluate_policy(make_actor("ray_follower", cfg), bank, cfg, episodes=2, robots=2)
    assert set(res) == set(EVAL_COLS)
    assert all(len(v) == 2 and np.isfinite(v).all() for v in res.values())
    s = summarise(res)
    assert "frac_found" in format_table({"ray_follower": s})

    ob_dim = TOKEN_FIXED + cfg.rayfronts.embedding_dim
    torch.manual_seed(0)
    policy = TokenPolicy(ob_dim, 18, cfg.rayfronts.embedding_dim, d_model=32)
    res2 = evaluate_policy(TorchActor(policy, "cpu"), bank, cfg, episodes=1, robots=3)
    assert np.isfinite(res2["frac_found"]).all()


def test_make_actor_rejects_unknown():
    with pytest.raises(Exception):
        make_actor("nope", tiny_cfg())


# ---- CLIs --------------------------------------------------------------------------------------
def test_train_script_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = load_script("train")
    rc = train.main(["--name", "t", "--scenes", "synthetic:0-4", "--envs", "2", "--rollout", "4",
                     "--updates", "3", "--eval-every", "3", "--eval-episodes", "1",
                     "--robots", "2", "--no-baselines", "--device", "cpu", "--d-model", "32",
                     "--ckpt-every", "3"])
    assert rc == 0
    run = tmp_path / "runs" / "t"
    for f in ("log.csv", "eval.csv", "curve.png", "latest.pt", "ckpt_3.pt", "args.json"):
        assert (run / f).exists(), f
    assert len((run / "log.csv").read_text().strip().splitlines()) == 4

    rc = train.main(["--name", "t", "--scenes", "synthetic:0-4", "--envs", "2", "--rollout", "4",
                     "--updates", "5", "--eval-every", "5", "--eval-episodes", "1",
                     "--robots", "2", "--no-baselines", "--device", "cpu", "--d-model", "32",
                     "--resume"])
    assert rc == 0
    assert len((run / "log.csv").read_text().strip().splitlines()) == 6

    actor = make_actor(str(run / "latest.pt"), tiny_cfg(), device="cpu")
    assert isinstance(actor, TorchActor)


def test_eval_policy_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ev = load_script("eval_policy")
    out = tmp_path / "e.csv"
    rc = ev.main(["--policy", "ray_follower", "--policy", "random", "--scenes", "synthetic:0-4",
                  "--episodes", "1", "--robots", "2", "--device", "cpu", "--out", str(out)])
    assert rc == 0 and out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3 and lines[1].startswith("ray_follower,")
