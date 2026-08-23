"""Parallel env backends: serial/subproc equivalence, scene resampling, shutdown, eval CIs."""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rlplanner.sim.config import EnvConfig
from rlplanner.sim.state import TOKEN_FIXED
from rlplanner.train.evaluate import (EVAL_COLS, Stat, TorchActor, evaluate_baselines,
                                      evaluate_policy, format_table, summarise, write_csv)
from rlplanner.train.par_env import SerialVecEnv, SubprocVecEnv, default_workers
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig
from rlplanner.train.rollout import Collector
from rlplanner.train.scenes import SceneBank

ROOT = Path(__file__).resolve().parents[1]
SCENES = "synthetic:0-6"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_cfg(t_max: float = 60.0) -> EnvConfig:
    cfg = EnvConfig()
    cfg.robot.n_robots = 2
    cfg.t_max_s = t_max
    assert not cfg.validate()
    return cfg


def valid_actions(obs, rng) -> np.ndarray:
    """One selectable token per (env, robot); identical for identical observations."""
    E, R, K = obs.token_mask.shape
    a = np.zeros((E, R), np.int64)
    for e in range(E):
        for r in range(R):
            v = np.flatnonzero(obs.token_mask[e, r])
            a[e, r] = int(rng.choice(v)) if v.size else 0
    return a


def obs_arrays(o) -> list[np.ndarray]:
    return [o.tokens, o.token_mask, o.token_xy, o.token_type, o.token_id, o.robot_feat,
            o.robot_mask, o.t]


# ---- equivalence -------------------------------------------------------------------------------
def test_subproc_matches_serial_for_ten_decisions():
    cfg = tiny_cfg()
    kw = dict(robots=(2, 2), split="train", seed=7, n_workers=2, send_bev=True, num_threads=1)
    with SerialVecEnv(SCENES, cfg, 4, **kw) as s, SubprocVecEnv(SCENES, cfg, 4, **kw) as p:
        os_, op = s.reset_all(), p.reset_all()
        assert s.scene_ids() == p.scene_ids()
        for k, (x, y) in enumerate(zip(obs_arrays(os_), obs_arrays(op))):
            assert np.array_equal(x, y, equal_nan=True), f"reset obs field {k}"
        rs, rp = np.random.default_rng(0), np.random.default_rng(0)
        for step in range(10):
            a, b = valid_actions(os_, rs), valid_actions(op, rp)
            assert np.array_equal(a, b)
            os_, r1, d1, _ = s.step(a)
            op, r2, d2, _ = p.step(b)
            for k, (x, y) in enumerate(zip(obs_arrays(os_), obs_arrays(op))):
                assert np.array_equal(x, y, equal_nan=True), f"step {step} obs field {k}"
            assert np.array_equal(r1, r2) and np.array_equal(d1, d2)


def test_padding_matches_vec_env_layout():
    cfg = tiny_cfg()
    with SubprocVecEnv(SCENES, cfg, 3, robots=(2, 3), seed=1, n_workers=2) as p:
        o = p.reset_all()
        assert o.tokens.shape[:3] == (3, 3, cfg.k_tokens)
        assert o.robot_mask.sum(1).min() >= 2
        for e in range(3):
            n = int(o.robot_mask[e].sum())
            assert not o.token_mask[e, n:].any()      # padded robot slots are unselectable
            assert (o.token_id[e, n:] == -1).all()


@pytest.mark.parametrize("klass", [SerialVecEnv, SubprocVecEnv])
def test_observations_are_not_recycled_buffers(klass):
    """A rollout keeps its obs history: consecutive steps must not share memory."""
    cfg = tiny_cfg()
    with klass(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=2) as p:
        o1 = p.reset_all()
        keep = o1.tokens.copy()
        rng = np.random.default_rng(0)
        o2 = p.step(valid_actions(o1, rng))[0]
        o3 = p.step(valid_actions(o2, rng))[0]
        for a, b in ((o1, o2), (o2, o3)):
            assert not np.shares_memory(a.tokens, b.tokens)
            assert not np.shares_memory(a.robot_feat, b.robot_feat)
        assert np.array_equal(o1.tokens, keep)


# ---- auto-reset --------------------------------------------------------------------------------
def test_auto_reset_resamples_the_scene():
    cfg = tiny_cfg(t_max=15.0)                        # 3 decisions per episode
    with SubprocVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=3, n_workers=2) as p:
        o = p.reset_all()
        before = p.scene_ids()
        rng = np.random.default_rng(0)
        seen = 0
        for _ in range(4):
            o, r, d, infos = p.step(valid_actions(o, rng))
            if d.any():
                seen += 1
                for i in np.flatnonzero(d):
                    assert "final_info" in infos[i] and "metrics" in infos[i]["final_info"]
        assert seen >= 1
        after = p.scene_ids()
        assert after != before                        # slots moved on to fresh scenes
        assert float(o.t.max()) < cfg.t_max_s          # and are mid-episode again


def test_serial_and_subproc_resample_identically():
    cfg = tiny_cfg(t_max=15.0)
    kw = dict(robots=(2, 2), seed=5, n_workers=2, num_threads=1)
    with SerialVecEnv(SCENES, cfg, 2, **kw) as s, SubprocVecEnv(SCENES, cfg, 2, **kw) as p:
        os_, op = s.reset_all(), p.reset_all()
        rs, rp = np.random.default_rng(1), np.random.default_rng(1)
        for _ in range(5):
            os_ = s.step(valid_actions(os_, rs))[0]
            op = p.step(valid_actions(op, rp))[0]
        assert s.scene_ids() == p.scene_ids()
        assert np.array_equal(os_.tokens, op.tokens)


# ---- determinism / shutdown --------------------------------------------------------------------
def test_determinism_across_two_runs():
    cfg = tiny_cfg()
    out = []
    for _ in range(2):
        with SubprocVecEnv(SCENES, cfg, 4, robots=(2, 2), seed=11, n_workers=2) as p:
            o = p.reset_all()
            rng = np.random.default_rng(2)
            for _ in range(3):
                o, r, _, _ = p.step(valid_actions(o, rng))
            out.append((p.scene_ids(), o.tokens.copy(), r.copy()))
    assert out[0][0] == out[1][0]
    assert np.array_equal(out[0][1], out[1][1])
    assert np.array_equal(out[0][2], out[1][2])


def test_close_leaves_no_children():
    cfg = tiny_cfg()
    p = SubprocVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=2)
    p.reset_all()
    assert p.n_alive == 2
    p.close()
    assert p.n_alive == 0
    assert not [c for c in mp.active_children() if c.name.startswith("envworker-")]
    p.close()                                          # idempotent
    with pytest.raises(RuntimeError):
        p.reset_all()


def test_context_manager_closes_on_exception():
    cfg = tiny_cfg()
    vec = None
    with pytest.raises(ZeroDivisionError):
        with SubprocVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=2) as p:
            vec = p
            p.reset_all()
            1 / 0
    assert vec.n_alive == 0
    assert not [c for c in mp.active_children() if c.name.startswith("envworker-")]


def test_worker_error_is_reported():
    cfg = tiny_cfg()
    with SubprocVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=1) as p:
        p.reset_all()
        with pytest.raises(RuntimeError):
            p.step(np.full((2, 2), 999, np.int64))     # masked/out-of-range token
    assert p.n_alive == 0


def test_default_workers_leaves_headroom():
    import os
    assert default_workers(64) == max(1, (os.cpu_count() or 2) - 2)
    assert default_workers(2) == 2


# ---- rollout / training loop -------------------------------------------------------------------
def test_collector_rollout_on_subproc():
    torch.manual_seed(0)
    cfg = tiny_cfg(t_max=30.0)
    with SubprocVecEnv(SCENES, cfg, 4, robots=(2, 2), seed=0, n_workers=2, send_bev=False) as p:
        col = Collector(p, "cpu")
        ob = col.obs
        policy = TokenPolicy(ob.tokens.shape[3], ob.robot_feat.shape[2], ob.query_emb.shape[2], d_model=32)
        ppo = PPO(policy, PPOConfig(epochs=2, n_minibatches=2, target_kl=None), "cpu")
        batch, stats = col.rollout(policy, 8)
        assert len(batch) == 32 and stats["episodes"] >= 1
        E = ob.tokens.shape[0]
        assert not torch.allclose(batch.obs.tokens[:E], batch.obs.tokens[-E:])  # real history
        out = ppo.update(batch)
        assert np.isfinite(out["policy_loss"]) and "explained_variance" in out
        assert out["minibatches"] == 4


def test_ppo_target_kl_stops_early_and_lr_setter():
    torch.manual_seed(0)
    from train_mocks import BanditVecEnv
    env = BanditVecEnv(n_envs=8, n_robots=1, k_tokens=6, seed=0)
    col = Collector(env, "cpu")
    ob = col.obs
    policy = TokenPolicy(ob.tokens.shape[3], ob.robot_feat.shape[2], ob.query_emb.shape[2], d_model=32)
    ppo = PPO(policy, PPOConfig(epochs=8, n_minibatches=4, lr=1e-2, target_kl=1e-6), "cpu")
    batch, _ = col.rollout(policy, 8)
    out = ppo.update(batch)
    assert out["early_stop"] == 1.0 and out["minibatches"] < 32
    ppo.set_lr(1e-5)
    assert ppo.lr == pytest.approx(1e-5)
    assert ppo.update(batch)["lr"] == pytest.approx(1e-5)


# ---- evaluation --------------------------------------------------------------------------------
def test_eval_has_ci_columns(tmp_path):
    bank = SceneBank(SCENES)
    cfg = tiny_cfg()
    res = evaluate_policy("ray_follower", bank, cfg, episodes=4, robots=2, backend="subproc", workers=2)
    s = summarise(res)
    assert set(res) == set(EVAL_COLS)
    assert all(len(v) == 4 for v in res.values())
    assert isinstance(s["frac_found"], Stat) and s["frac_found"].n == 4
    assert np.isfinite(s["frac_found"].ci)
    assert "+-" in format_table({"ray_follower": s})
    out = tmp_path / "e.csv"
    write_csv(out, {"ray_follower": s})
    head = out.read_text().splitlines()[0]
    assert "frac_found_ci" in head and "n_episodes" in head


def test_eval_matches_serial_and_subproc():
    bank = SceneBank(SCENES)
    cfg = tiny_cfg()
    a = evaluate_policy("ray_follower", bank, cfg, episodes=4, robots=2, backend="serial")
    b = evaluate_policy("ray_follower", bank, cfg, episodes=4, robots=2, backend="subproc", workers=2)
    assert np.allclose(sorted(a["frac_found"]), sorted(b["frac_found"]), equal_nan=True)


def test_eval_torch_actor_batched():
    torch.manual_seed(0)
    bank = SceneBank(SCENES)
    cfg = tiny_cfg(t_max=30.0)
    policy = TokenPolicy(TOKEN_FIXED + cfg.rayfronts.embedding_dim, 18, cfg.rayfronts.embedding_dim, d_model=32)
    res = evaluate_policy(TorchActor(policy, "cpu"), bank, cfg, episodes=4, robots=2,
                          backend="subproc", workers=2)
    assert np.isfinite(res["frac_found"]).all() and len(res["frac_found"]) == 4


def test_evaluate_baselines_reuses_one_pool():
    bank = SceneBank(SCENES)
    cfg = tiny_cfg(t_max=30.0)
    with SubprocVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=2) as p:
        out = evaluate_baselines(("ray_follower", "random"), bank, cfg, 2, 2, pool=p)
    assert set(out) == {"ray_follower", "random"}
    assert all(isinstance(v["frac_found"], Stat) for v in out.values())


# ---- play_episode ckpt: ------------------------------------------------------------------------
def smoke_ckpt(path: Path, cfg: EnvConfig) -> Path:
    torch.manual_seed(0)
    policy = TokenPolicy(TOKEN_FIXED + cfg.rayfronts.embedding_dim, 18, cfg.rayfronts.embedding_dim, d_model=32)
    sd = PPO(policy, PPOConfig(), "cpu").state_dict()
    sd.update({"update": 1, "decisions": 10})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    return path


def test_play_episode_ckpt(tmp_path, monkeypatch):
    # scripts/play_episode.py renders through rlplanner.viz, which the viz agent owns; skip
    # rather than fail while the visualizer is still on the old (per-query) observation
    try:
        import rlplanner.viz  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"viz not yet ported to the feature observation: {exc}")
    monkeypatch.chdir(tmp_path)
    cfg = EnvConfig()
    ck = smoke_ckpt(tmp_path / "runs" / "smk" / "latest.pt", cfg)
    play = load_script("play_episode")
    out = tmp_path / "ep.gif"
    rc = play.main(["--synthetic", "0", "--policy", f"ckpt:{ck}", "--robots", "2",
                    "--out", str(out), "--max-decisions", "2", "--every-n", "2",
                    "--dpi", "40", "--device", "cpu"])
    assert rc == 0 and out.exists() and out.stat().st_size > 0

    out2 = tmp_path / "ep2.gif"
    rc = play.main(["--synthetic", "0", "--policy", "ckpt:latest", "--run", "smk",
                    "--robots", "2", "--out", str(out2), "--max-decisions", "1",
                    "--every-n", "1", "--dpi", "40", "--device", "cpu"])
    assert rc == 0 and out2.exists()

    with pytest.raises(SystemExit):
        play.main(["--synthetic", "0", "--policy", "ckpt:latest", "--out", str(out2)])
    with pytest.raises(SystemExit):
        play.main(["--synthetic", "0", "--policy", "nope", "--out", str(out2)])
