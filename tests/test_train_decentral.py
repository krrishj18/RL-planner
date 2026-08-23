"""Decentralised execution through the training stack: per-robot BEV and peer tokens over the
worker pipes, the actor that consumes them, and determinism under range comms."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rlplanner.sim.config import EnvConfig
from rlplanner.sim.state import PEER_FEAT_DIM, PEER_LINK, PEER_VALID
from rlplanner.sim.tokens import BEV_CHANNELS
from rlplanner.train.obs import ObsBatch
from rlplanner.train.par_env import SerialVecEnv, SubprocVecEnv
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig
from rlplanner.train.rollout import Collector

ROOT = Path(__file__).resolve().parents[1]
SCENES = "synthetic:0-6"
VARIANTS = ROOT / "configs" / "variants"


def dec_cfg(t_max: float = 60.0, robots: int = 2, rbev: int = 16, range_m: float = 150.0):
    cfg = EnvConfig()
    cfg.robot.n_robots = robots
    cfg.t_max_s = t_max
    cfg.comms.mode = "range"
    cfg.comms.randomize_range = False
    cfg.comms.range_m = range_m
    cfg.tokens.robot_bev_size = rbev
    cfg.tokens.local_size = 0
    assert not cfg.validate()
    return cfg


def valid_actions(obs, rng) -> np.ndarray:
    E, R, _ = obs.token_mask.shape
    a = np.zeros((E, R), np.int64)
    for e in range(E):
        for r in range(R):
            v = np.flatnonzero(obs.token_mask[e, r])
            a[e, r] = int(rng.choice(v)) if v.size else 0
    return a


def obs_arrays(o) -> list[np.ndarray]:
    return [o.tokens, o.token_mask, o.token_xy, o.token_type, o.token_id, o.robot_feat,
            o.robot_mask, o.t, o.peer_tokens, o.robot_bev]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- plumbing -------------------------------------------------------------------------------
def test_workers_ship_peer_tokens_and_the_per_robot_bev():
    cfg = dec_cfg()
    with SerialVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=1, num_threads=1) as v:
        o = v.reset_all()
        assert o.peer_tokens.shape == (2, 2, 1, PEER_FEAT_DIM)
        assert o.robot_bev.shape == (2, 2, len(BEV_CHANNELS), 16, 16)
        o = v.step(valid_actions(o, np.random.default_rng(0)))[0]
        assert np.isfinite(o.robot_bev).all() and np.isfinite(o.peer_tokens).all()
        assert o.robot_bev.max() > 0.0
        b = ObsBatch.from_vec_obs(o, "cpu")
        assert b.has_robot_bev and b.has_peers
        assert tuple(b.robot_bev.shape) == o.robot_bev.shape
        assert not ObsBatch.from_vec_obs(o, "cpu", with_robot_bev=False).has_robot_bev


def test_robot_bev_is_off_when_not_asked_for():
    cfg = dec_cfg(rbev=0)
    with SerialVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=1, num_threads=1) as v:
        o = v.reset_all()
        assert o.robot_bev is None
        assert not ObsBatch.from_vec_obs(o, "cpu").has_robot_bev


def test_subproc_matches_serial_under_range_comms():
    cfg = dec_cfg(t_max=40.0)
    kw = dict(robots=(2, 2), split="train", seed=7, n_workers=2, send_bev=True, num_threads=1)
    with SerialVecEnv(SCENES, cfg, 4, **kw) as s, SubprocVecEnv(SCENES, cfg, 4, **kw) as p:
        os_, op = s.reset_all(), p.reset_all()
        rs, rp = np.random.default_rng(0), np.random.default_rng(0)
        for step in range(10):
            a, b = valid_actions(os_, rs), valid_actions(op, rp)
            assert np.array_equal(a, b)
            os_, r1, d1, _ = s.step(a)
            op, r2, d2, _ = p.step(b)
            for k, (x, y) in enumerate(zip(obs_arrays(os_), obs_arrays(op))):
                assert np.array_equal(x, y, equal_nan=True), f"step {step} obs field {k}"
            assert np.array_equal(r1, r2) and np.array_equal(d1, d2)


def test_padded_robot_slots_stay_zero():
    cfg = dec_cfg(robots=2)
    with SerialVecEnv(SCENES, cfg, 2, robots=(2, 3), seed=3, n_workers=1, num_threads=1) as v:
        o = v.reset_all()
        assert o.peer_tokens.shape[1] == 3
        for e in range(o.tokens.shape[0]):
            for r in range(3):
                if not o.robot_mask[e, r]:
                    assert (o.peer_tokens[e, r] == 0.0).all()
                    if o.robot_bev is not None:
                        assert (o.robot_bev[e, r] == 0.0).all()


# ---- the actor ------------------------------------------------------------------------------
def _policy(ob: ObsBatch, **kw) -> TokenPolicy:
    kw.setdefault("robot_bev_channels",
                  ob.robot_bev.shape[2] if ob.has_robot_bev else len(BEV_CHANNELS))
    return TokenPolicy(ob.token_dim, ob.robot_dim, ob.feat_dim, d_model=32, n_layers=1,
                       peer_dim=ob.peer_tokens.shape[-1], **kw)


def test_actor_consumes_peers_and_its_own_bev():
    cfg = dec_cfg()
    with SerialVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=1, num_threads=1) as v:
        ob = ObsBatch.from_vec_obs(v.reset_all(), "cpu")
    torch.manual_seed(0)
    pol = _policy(ob, use_peers=True, use_robot_bev=True)
    logits, value = pol(ob)
    assert logits.shape == (ob.n_envs, ob.n_robots, ob.k_tokens)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    loss = logits.sum() + value.sum()
    loss.backward()
    assert pol.peer_proj.weight.grad is not None and pol.peer_proj.weight.grad.abs().sum() > 0
    assert pol.rbev_proj.weight.grad.abs().sum() > 0
    # the peer block changes the decision: a different peer state is a different observation
    torch.manual_seed(0)
    other = ObsBatch(**{f: getattr(ob, f) for f in ob.__dataclass_fields__})
    other.peer_tokens = torch.zeros_like(ob.peer_tokens)
    assert not torch.allclose(pol(other)[0], logits)


def test_policy_config_round_trip_keeps_the_decentral_switches():
    cfg = dec_cfg()
    with SerialVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=1, num_threads=1) as v:
        ob = ObsBatch.from_vec_obs(v.reset_all(), "cpu")
    pol = _policy(ob, use_peers=True, use_robot_bev=True, use_bev=True,
                  bev_channels=ob.bev.shape[1])
    clone = TokenPolicy.from_config(pol.config())
    clone.load_state_dict(pol.state_dict())
    assert clone.use_peers and clone.use_robot_bev and clone.peer_dim == pol.peer_dim
    with torch.no_grad():
        assert torch.allclose(pol(ob)[0], clone(ob)[0])


def test_actor_without_robot_bev_refuses_a_missing_one():
    cfg = dec_cfg(rbev=0)
    with SerialVecEnv(SCENES, cfg, 2, robots=(2, 2), seed=0, n_workers=1, num_threads=1) as v:
        ob = ObsBatch.from_vec_obs(v.reset_all(), "cpu")
    pol = _policy(ob, use_robot_bev=True)
    with pytest.raises(ValueError, match="robot_bev"):
        pol(ob)


def test_ppo_update_on_a_decentral_rollout():
    torch.manual_seed(0)
    cfg = dec_cfg(t_max=30.0)
    with SubprocVecEnv(SCENES, cfg, 4, robots=(2, 2), seed=0, n_workers=2, send_bev=True,
                       num_threads=1) as p:
        col = Collector(p, "cpu")
        ob = ObsBatch.from_vec_obs(col.obs, "cpu")
        pol = _policy(ob, use_peers=True, use_robot_bev=True, use_bev=True,
                      bev_channels=ob.bev.shape[1])
        ppo = PPO(pol, PPOConfig(epochs=1, n_minibatches=2, target_kl=None), "cpu")
        batch, stats = col.rollout(pol, 4)
        out = ppo.update(batch)
    assert batch.obs.has_robot_bev and batch.obs.has_peers
    assert np.isfinite(out["policy_loss"]) and np.isfinite(out["value_loss"])


# ---- variant configs ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(p.stem for p in (ROOT / "configs" / "variants").glob("*.yaml")))
def test_every_variant_config_is_valid_and_runs(name):
    import yaml
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.baselines import make_policy
    from rlplanner.sim.env import DisasterEnv
    d = dict(yaml.safe_load((VARIANTS / f"{name}.yaml").read_text()))
    train = d.pop("train", {})
    d.pop("notes", None)
    cfg = EnvConfig.from_dict(d)
    assert cfg.validate() == []
    assert set(train) <= {"use-local", "use-bev", "use-robot-bev", "peers", "sequential_decode"}
    cfg.robot.n_robots = 2
    cfg.t_max_s = 40.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(120.0, 120.0)), cfg, seed=0)
    pol = make_policy("nearest_frontier", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    for _ in range(6):
        obs, r, done, info = env.step(pol.act(obs, env.state))
        assert np.isfinite(obs.tokens).all() and np.isfinite(r)
    assert (obs.robot_bev is not None) == (cfg.tokens.robot_bev_size > 0)
    assert (obs.local is not None) == (cfg.tokens.local_size > 0)


def test_train_script_reads_a_variant():
    train = load_script("train")
    ap = train.build_parser()
    a = ap.parse_args(["--variant", "decentral_share_all"])
    cfg = train.apply_variant(ap, a)
    assert cfg.comms.mode == "range" and cfg.comms.share.visited
    assert a.use_local and a.use_robot_bev and a.peers and a.use_bev
    a2 = ap.parse_args(["--variant", "decentral_share_all", "--no-peers"])
    train.apply_variant(ap, a2)
    assert not a2.peers, "the command line wins over the variant file"


def test_sweep_script_builds_a_summary(tmp_path):
    sweep = load_script("sweep")
    a = sweep.build_parser().parse_args(["--variants", "central_full", "decentral_share_all",
                                         "--out", str(tmp_path)])
    res = {("central_full", 100.0): {"cols": {c: (0.5, 0.1) for c, _, _ in sweep.COLS},
                                     "by_container": {c: (1, 2) for c in sweep.CONTAINERS},
                                     "n": 4}}
    md = sweep.summary_md(tmp_path, a, res, [100.0, float("inf")])
    assert "eval comms range = 100 m" in md and "eval comms range = inf" in md
    assert "central_full" in md and "decentral_share_all" in md
    assert "found / total" in md


def test_the_summary_has_an_own_setting_row_per_variant(tmp_path):
    """The blackout must not be scored only with a radio it never had."""
    sweep = load_script("sweep")
    a = sweep.build_parser().parse_args(["--variants", "decentral_blackout", "central_full",
                                         "--out", str(tmp_path)])
    assert a.own_eval and sweep.OWN in ([float(r) for r in a.eval_ranges] + [sweep.OWN])
    res = {(v, sweep.OWN): {"cols": {c: (0.5, 0.1) for c, _, _ in sweep.COLS},
                            "by_container": {c: (1, 2) for c in sweep.CONTAINERS}, "n": 4,
                            "setting": sweep.comms_setting(sweep.variant_cfg(v, None)),
                            "sequential_decode": False}
           for v in a.variants}
    md = sweep.summary_md(tmp_path, a, res, [200.0, sweep.OWN])
    assert "own training comms setting" in md
    assert "| setting |" in md
    assert "blackout: spawn exchange only" in md and "one shared belief" in md
    assert "`central_full`=True" in md and "`decentral_blackout`=False" in md


def test_the_own_eval_uses_each_variants_training_comms():
    sweep = load_script("sweep")
    expect = {"central_full": ("full", None), "decentral_blackout": ("range", 0.0)}
    for v in sweep.ALL_VARIANTS:
        cfg = sweep.variant_cfg(v, None)
        assert cfg.validate() == []
        mode, rng = expect.get(v, ("range", None))
        assert cfg.comms.mode == mode
        if rng is not None:
            assert cfg.comms.range_m == rng and not cfg.comms.randomize_range
        elif mode == "range":
            assert cfg.comms.randomize_range, "the share_* variants train on a randomised range"


@pytest.mark.parametrize("rng", [100.0, 200.0, float("inf"), None])
def test_every_variant_is_evaluated_with_its_own_payload_flags(rng):
    """The common ranges change the *link range*, never what a contact hands over."""
    import yaml
    sweep = load_script("sweep")
    for v in sweep.ALL_VARIANTS:
        d = dict(yaml.safe_load((VARIANTS / f"{v}.yaml").read_text()))
        want = dict((d.get("comms") or {}).get("share") or {})
        share = sweep.variant_cfg(v, rng).comms.share
        for k, expected in want.items():
            assert getattr(share, k) == expected, (v, rng, k)
    pos = sweep.variant_cfg("decentral_share_pos_cov", rng).comms.share
    assert pos.rays == "none" and not pos.segments and not pos.visited and pos.coverage
    rays = sweep.variant_cfg("decentral_share_rays", rng).comms.share
    assert rays.rays == "all" and not rays.coverage and not rays.visited
    black = sweep.variant_cfg("decentral_blackout", rng).comms.share
    assert black.rays == "none" and not (black.coverage or black.segments or black.visited)


# ---- CTDE execution: no same-decision claim mask decentralised ---------------------------------
@pytest.mark.parametrize("name", sorted(p.stem for p in (ROOT / "configs" / "variants").glob("*.yaml")))
def test_only_the_central_variant_decodes_sequentially(name):
    import yaml
    train = dict(yaml.safe_load((VARIANTS / f"{name}.yaml").read_text()).get("train") or {})
    assert "sequential_decode" in train, f"{name}: the execution switch must be explicit"
    assert bool(train["sequential_decode"]) is (name == "central_full")


def test_train_script_plumbs_the_decode_switch_from_the_variant():
    train = load_script("train")
    ap = train.build_parser()
    a = ap.parse_args(["--variant", "decentral_share_all"])
    train.apply_variant(ap, a)
    assert a.sequential_decode is False
    b = ap.parse_args(["--variant", "central_full"])
    train.apply_variant(ap, b)
    assert b.sequential_decode is True


def test_two_robots_may_share_a_token_under_decentralised_execution():
    """Nothing in the simulator de-conflicts a decentralised team: the env accepts one token for
    two robots, which is exactly what `sequential_decode=False` can produce."""
    from rlplanner.sim.env import DisasterEnv
    from rlplanner.scene.schema import make_synthetic_scene
    cfg = dec_cfg(t_max=40.0, robots=2, rbev=0)
    cfg.comms.mode = "full"                  # one id namespace, so a shared slot is one target
    env = DisasterEnv(make_synthetic_scene(0, region_m=(120.0, 120.0)), cfg, seed=0)
    obs = env.state.last_obs
    shared = int(np.flatnonzero(obs.token_mask[0] & obs.token_mask[1])[-1])
    obs, r, done, info = env.step(np.array([shared, shared], np.int64))
    assert np.isfinite(r)
    a, b = env.state.robots
    assert (a.target_token_type, a.target_id) == (b.target_token_type, b.target_id)


# ---- variants / sweep / determinism (QA pass 2026-08-21) ----------------------------------------
@pytest.mark.parametrize("name", sorted(p.stem for p in (ROOT / "configs" / "variants").glob("*.yaml")))
def test_every_variant_steps_twenty_decisions(name):
    import yaml
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.baselines import make_policy
    from rlplanner.sim.env import DisasterEnv
    d = dict(yaml.safe_load((VARIANTS / f"{name}.yaml").read_text()))
    d.pop("train", None)
    d.pop("notes", None)
    cfg = EnvConfig.from_dict(d)
    assert cfg.validate() == []
    cfg.robot.n_robots = 3
    cfg.t_max_s = 150.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(160.0, 160.0)), cfg, seed=0)
    pol = make_policy("ray_follower", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    total = 0.0
    for _ in range(20):
        obs, r, done, info = env.step(pol.act(obs, env.state))
        total += r
        assert np.isfinite(obs.tokens).all() and np.isfinite(r)
        assert obs.token_mask.any(1).all(), "every robot always has at least `hold`"
        if done:
            break
    assert np.isfinite(total)
    assert (obs.local is not None) == (cfg.tokens.local_size > 0)
    assert (obs.robot_bev is not None) == (cfg.tokens.robot_bev_size > 0)
    if name == "decentral_share_all_tokens_only":
        assert cfg.tokens.local_size == 0 and cfg.tokens.robot_bev_size == 0
        assert obs.local is None and obs.robot_bev is None


def test_the_blackout_variant_shares_nothing_after_spawn():
    import yaml
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.baselines import make_policy
    from rlplanner.sim.env import DisasterEnv
    d = dict(yaml.safe_load((VARIANTS / "decentral_blackout.yaml").read_text()))
    d.pop("train", None)
    d.pop("notes", None)
    cfg = EnvConfig.from_dict(d)
    cfg.robot.n_robots = 3
    cfg.t_max_s = 150.0
    env = DisasterEnv(make_synthetic_scene(0, region_m=(160.0, 160.0)), cfg, seed=0)
    pol = make_policy("ray_follower", queries=cfg.rayfronts.queries, seed=0)
    obs = env.state.last_obs
    spawn = [dict(b.peers) for b in env.comms.beliefs]
    assert all(len(p) == 2 for p in spawn), "the spawn exchange hands over positions"
    for _ in range(20):
        obs, _, done, info = env.step(pol.act(obs, env.state))
        assert info["link_frac"] == 0.0, "the spawn hand-over is not a radio link"
        assert not env.state.comms_links.any()
        if done:
            break
    for r, b in enumerate(env.comms.beliefs):
        assert not b.rays_in and not b.segs_in
        assert np.array_equal(b.known, b.feat_known), "no coverage arrived"
        assert all(v.robot == r for v in b.visited.values())
        assert set(b.peers) == set(spawn[r])
        for j, pr in b.peers.items():
            assert not pr.linked and pr.t_contact == 0.0, "the spawn contact never refreshes"


def test_the_held_out_split_is_the_same_for_every_variant():
    """Identical seeds and scene bank ⇒ identical held-out sets, so the sweep rows differ only by
    the variant's own config."""
    from rlplanner.train.scenes import SceneBank
    sweep = load_script("sweep")
    banks = [SceneBank("synthetic:0-40") for _ in sweep.ALL_VARIANTS]
    ref = [str(k) for k in banks[0].split("heldout")]
    assert ref and set(ref).isdisjoint(str(k) for k in banks[0].split("train"))
    for b in banks[1:]:
        assert [str(k) for k in b.split("heldout")] == ref
    tasks = [sweep.__dict__ for _ in ()]      # (the eval task list is a pure function of the bank)
    from rlplanner.train.evaluate import eval_tasks
    a = eval_tasks(banks[0], "heldout", 8, 10_000)
    for b in banks[1:]:
        assert eval_tasks(b, "heldout", 8, 10_000) == a


def test_sweep_scores_every_variant_under_range_comms():
    sweep = load_script("sweep")
    for v in sweep.ALL_VARIANTS:
        cfg = sweep.variant_cfg(v, 150.0)
        assert cfg.comms.mode == "range" and cfg.comms.range_m == 150.0
        assert not cfg.comms.randomize_range
        assert cfg.validate() == []
    assert sweep.variant_cfg("central_full", None).comms.mode == "full"


def test_eval_policy_comms_range_switches_to_range_comms():
    ev = load_script("eval_policy")
    ap_args = ["--policy", "ray_follower", "--variant", "central_full"]
    a = ev.main.__globals__["argparse"]  # noqa: F841  (parser is built inside main)
    import argparse
    ns = argparse.Namespace(variant="central_full", config=None, comms=None, comms_range=None)
    assert ev._load_cfg(ns).comms.mode == "full"
    ns.comms_range = "100"
    cfg = ev._load_cfg(ns)
    assert cfg.comms.mode == "range" and cfg.comms.range_m == 100.0
    ns.comms, ns.comms_range = "full", "100"
    assert ev._load_cfg(ns).comms.mode == "full", "an explicit --comms still wins"
    assert ap_args


def test_range_comms_eval_changes_with_the_link_range():
    from rlplanner.sim.baselines import make_policy
    from rlplanner.sim.env import DisasterEnv
    from rlplanner.scene.schema import make_synthetic_scene
    out = {}
    for rng in (12.0, float("inf")):
        cfg = dec_cfg(t_max=200.0, robots=3, rbev=0, range_m=rng)
        cfg.tokens.local_size = 0
        env = DisasterEnv(make_synthetic_scene(0, region_m=(200.0, 200.0)), cfg, seed=0)
        pol = make_policy("ray_follower", queries=cfg.rayfronts.queries, seed=0)
        obs, total = env.state.last_obs, 0.0
        while True:
            obs, r, done, info = env.step(pol.act(obs, env.state))
            total += r
            if done:
                break
        out[rng] = (total, info["link_frac"], info["metrics"]["coverage_end"])
    assert out[12.0][1] < out[float("inf")][1] == pytest.approx(1.0)
    assert out[12.0] != out[float("inf")], "the link range has to change the episode"


def test_subproc_determinism_with_four_workers_under_range_comms():
    cfg = dec_cfg(t_max=40.0, robots=2, rbev=16, range_m=150.0)
    kw = dict(robots=(2, 2), split="train", seed=11, n_workers=4, send_bev=True, num_threads=1)
    with SubprocVecEnv(SCENES, cfg, 8, **kw) as a, SubprocVecEnv(SCENES, cfg, 8, **kw) as b:
        oa, ob = a.reset_all(), b.reset_all()
        ra, rb = np.random.default_rng(5), np.random.default_rng(5)
        for step in range(12):
            x, y = valid_actions(oa, ra), valid_actions(ob, rb)
            assert np.array_equal(x, y)
            oa, r1, d1, _ = a.step(x)
            ob, r2, d2, _ = b.step(y)
            for k, (u, v) in enumerate(zip(obs_arrays(oa), obs_arrays(ob))):
                assert np.array_equal(u, v, equal_nan=True), f"step {step} field {k}"
            assert np.array_equal(r1, r2) and np.array_equal(d1, d2)


def test_a_saved_checkpoint_reproduces_its_actions_with_peer_tokens_present(tmp_path):
    import torch as _t
    from rlplanner.train.evaluate import TorchActor, load_checkpoint
    from rlplanner.scene.schema import make_synthetic_scene
    from rlplanner.sim.env import DisasterEnv
    cfg = dec_cfg(t_max=80.0, robots=3, rbev=16, range_m=120.0)
    with SerialVecEnv(SCENES, cfg, 1, robots=(3, 3), seed=0, n_workers=1, num_threads=1) as v:
        ob = ObsBatch.from_vec_obs(v.reset_all(), "cpu")
    _t.manual_seed(3)
    pol = _policy(ob, use_peers=True, use_robot_bev=True, use_bev=True,
                  bev_channels=ob.bev.shape[1])
    ck = tmp_path / "ck.pt"
    _t.save({"policy": pol.state_dict(), "policy_config": pol.config()}, ck)
    clone, _ = load_checkpoint(ck, "cpu")
    env = DisasterEnv(make_synthetic_scene(0, region_m=(200.0, 200.0)), cfg, seed=0)
    a1, a2 = TorchActor(pol, "cpu", True), TorchActor(clone, "cpu", True)
    obs, acts = env.state.last_obs, []
    for _ in range(10):
        x = a1.act(obs, env.state)
        assert np.array_equal(x, a2.act(obs, env.state)), "the reload picks the same tokens"
        acts.append(x.copy())
        obs, _, done, _ = env.step(x)
        if done:
            break
    assert obs.peer_tokens is not None and obs.peer_tokens[..., PEER_VALID].max() == 1.0
    assert len({tuple(a) for a in acts}) > 1, "a constant action would prove nothing"
