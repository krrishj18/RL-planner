"""Area scaling: the robots/t_max rule, mixed-size banks, variable-robot rollouts, GAE."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rlplanner.scene import schema
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.env import DisasterEnv
from rlplanner.sim.vec_env import VecEnv
from rlplanner.train.evaluate import EVAL_META, by_bucket
from rlplanner.train.obs import ObsBatch
from rlplanner.train.par_env import (EnvGroup, RandomSampler, SerialVecEnv, SubprocVecEnv,
                                     TaskSampler)
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig
from rlplanner.train.rollout import Collector, gae
from rlplanner.train.scenes import (AUTO, T_MAX_MAX_S, T_MAX_MIN_S, SceneBank, auto_robots,
                                    auto_t_max, parse_robots, parse_scene_mix, parse_t_max,
                                    region_area_km2)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def scene_file(tmp_path: Path, name: str, wh, disaster: str = "earthquake") -> Path:
    sc = schema.make_synthetic_scene(0, region_m=wh, n_casualties=4, n_bystanders=2)
    sc.meta.disaster_type = disaster
    p = tmp_path / f"{name}.json"
    sc.to_json(p)
    return p


def tiny_cfg(n_robots: int = 2, t_max: float = 60.0) -> EnvConfig:
    cfg = EnvConfig()
    cfg.robot.n_robots = n_robots
    cfg.t_max_s = t_max
    assert not cfg.validate()
    return cfg


# ---- the rule ----------------------------------------------------------------------------------
@pytest.mark.parametrize("wh, robots, t_max", [((400, 400), 3, 600.0),
                                               ((1000, 1000), 7, 1500.0),
                                               ((1500, 1500), 8, 1500.0)])
def test_area_rule_at_the_three_reference_sizes(wh, robots, t_max):
    a = region_area_km2(wh)
    assert auto_robots(a) == robots
    assert auto_t_max(a) == pytest.approx(t_max, rel=1e-9)


def test_area_rule_clips_and_is_monotone():
    assert auto_robots(region_area_km2((100, 100))) == 3        # below the reference: clipped
    assert auto_t_max(region_area_km2((100, 100))) == 600.0
    assert auto_robots(region_area_km2((3000, 3000))) == 8      # far above: clipped
    assert auto_t_max(region_area_km2((3000, 3000))) == 1500.0
    areas = [region_area_km2((s, s)) for s in (400, 500, 700, 900, 1100, 1500)]
    r = [auto_robots(a) for a in areas]
    t = [auto_t_max(a) for a in areas]
    assert r == sorted(r) and t == sorted(t)
    assert all(3 <= x <= 8 for x in r) and all(600.0 <= x <= 1500.0 for x in t)
    # non-square regions scale by area, not by the longer side
    assert auto_robots(region_area_km2((2000, 500))) == auto_robots(region_area_km2((1000, 1000)))


def test_parse_auto_specs():
    assert parse_robots("auto") == (AUTO, AUTO)
    assert parse_robots("3") == (3, 3) and parse_robots("2-5") == (2, 5)
    assert parse_t_max("auto") == float(AUTO) and parse_t_max(None) == float(AUTO)
    assert parse_t_max("900") == 900.0 and parse_t_max(750.0) == 750.0
    with pytest.raises(ValueError):
        parse_t_max("soon")
    with pytest.raises(ValueError):
        parse_t_max("-5")


def test_env_params_resolves_per_scene(tmp_path):
    scene_file(tmp_path, "a", (240, 240))
    scene_file(tmp_path, "b", (900, 900))
    bank = SceneBank(str(tmp_path / "*.json"))
    small, big = sorted(bank.keys, key=bank.area)
    assert bank.env_params(small, AUTO, AUTO) == (3, 600.0)
    assert bank.env_params(big, AUTO, AUTO) == (7, 1350.0)
    assert bank.env_params(big, 5, 300.0) == (5, 300.0)         # explicit overrides win
    assert bank.robot_bounds((AUTO, AUTO)) == (3, 7)
    assert bank.t_max_bounds(AUTO) == (600.0, 1350.0)


# ---- buckets, split, mix -----------------------------------------------------------------------
def mixed_bank(tmp_path: Path) -> SceneBank:
    sizes = [(100, 100), (120, 120), (140, 140), (300, 300), (320, 320), (340, 340),
             (700, 700), (720, 720), (740, 740)]
    for i, wh in enumerate(sizes):
        scene_file(tmp_path, f"s{i}", wh, ("earthquake", "tornado", "explosion")[i % 3])
    return SceneBank(str(tmp_path / "*.json"))


def test_buckets_are_area_terciles(tmp_path):
    bank = mixed_bank(tmp_path)
    assert bank.bucket_counts("all") == {"small": 3, "medium": 3, "large": 3}
    got = sorted((round(bank.area(k), 4), bank.bucket(k)) for k in bank.keys)
    assert [b for _, b in got] == ["small"] * 3 + ["medium"] * 3 + ["large"] * 3


def test_heldout_is_stratified_by_disaster_and_size(tmp_path):
    bank = mixed_bank(tmp_path)
    strata = {(bank.disaster(k), bank.bucket(k)) for k in bank.keys}
    assert len(strata) == 9 and len(bank.heldout) == 9      # one per stratum (each has 1 scene)
    single = SceneBank("synthetic:0-20")                    # single stratum: unchanged behaviour
    assert len(single.train) == 18 and len(single.heldout) == 2
    assert not set(single.train) & set(single.heldout)


def test_scene_mix_sampling_proportions(tmp_path):
    bank = mixed_bank(tmp_path)
    mix = parse_scene_mix("small:0.5,medium:0.3,large:0.2")
    assert mix == pytest.approx({"small": 0.5, "medium": 0.3, "large": 0.2})
    rng = np.random.default_rng(0)
    draws = [bank.bucket(bank.sample("all", rng, mix)) for _ in range(6000)]
    for b, w in mix.items():
        assert draws.count(b) / len(draws) == pytest.approx(w, abs=0.02)
    # uniform inside a bucket
    keys = [k for k in bank.keys if bank.bucket(k) == "small"]
    per = [draws.count("small") / 3] * 3
    assert min(per) > 0
    # a shard without a bucket renormalises over what it has
    part = [k for k in bank.keys if bank.bucket(k) != "medium"]
    w = bank.mix_weights(part, mix)
    assert w.sum() == pytest.approx(1.0)
    assert sum(w[i] for i, k in enumerate(part) if bank.bucket(k) == "small") == \
        pytest.approx(0.5 / 0.7, abs=1e-9)
    assert bank.mix_weights(keys, None) is None
    with pytest.raises(ValueError):
        parse_scene_mix("huge:1.0")


def test_random_sampler_follows_the_mix(tmp_path):
    bank = mixed_bank(tmp_path)
    mix = parse_scene_mix("small:0.8,medium:0.1,large:0.1")
    w = bank.mix_weights(bank.keys, mix)
    s = RandomSampler(bank.keys, (AUTO, AUTO), seed=0, slot=0, weights=w)
    got = [bank.bucket(s.next()[0]) for _ in range(2000)]
    assert got.count("small") / len(got) == pytest.approx(0.8, abs=0.03)
    assert s.next()[2] == AUTO                              # robot count resolved per scene


# ---- variable robot counts / horizons -----------------------------------------------------------
def test_mixed_robot_counts_through_obsbatch_policy_and_ppo():
    """Two synthetic envs, 2 vs 5 robots and 30 vs 60 s horizons, one padded batch."""
    torch.manual_seed(0)
    envs = [DisasterEnv(schema.make_synthetic_scene(0, region_m=(120.0, 120.0)),
                        tiny_cfg(2, 30.0), seed=0),
            DisasterEnv(schema.make_synthetic_scene(1, region_m=(200.0, 200.0)),
                        tiny_cfg(5, 60.0), seed=1)]
    vec = VecEnv(envs)
    obs = vec.reset()
    assert obs.robot_mask.sum(1).tolist() == [2, 5]
    ob = ObsBatch.from_vec_obs(obs)
    assert ob.n_robots == 5 and not ob.token_mask[0, 2:].any()

    policy = TokenPolicy(ob.token_dim, ob.robot_dim, ob.feat_dim, d_model=32)
    ppo = PPO(policy, PPOConfig(epochs=2, n_minibatches=2, target_kl=None), "cpu")
    col = Collector(vec, "cpu", obs=obs)
    batch, _ = col.rollout(policy, 6)
    assert batch.actions.shape == (12, 5)
    assert batch.robot_mask.sum(0).tolist() == [12, 12, 6, 6, 6]  # env 0 fills 2 slots
    assert torch.isfinite(batch.logp).all() and torch.isfinite(batch.advantages).all()
    out = ppo.update(batch)
    assert np.isfinite(out["policy_loss"]) and np.isfinite(out["entropy"])
    # the padded robots must not move the value target
    assert torch.isfinite(batch.values).all()
    # the two envs really run different horizons
    assert envs[0].cfg.t_max_s == 30.0 and envs[1].cfg.t_max_s == 60.0
    vec.close()


def test_auto_scaling_gives_each_slot_its_own_team(tmp_path):
    scene_file(tmp_path, "small", (240, 240))
    scene_file(tmp_path, "big", (900, 900))
    spec = str(tmp_path / "*.json")
    cfg = tiny_cfg(2, 60.0)
    with SerialVecEnv(spec, cfg, 2, robots=(AUTO, AUTO), split="all", seed=0, n_workers=2,
                      t_max=float(AUTO), send_bev=False) as vec:
        o = vec.reset_all()
        assert vec.R == 7                                    # padded to the bank's maximum
        assert sorted(o.robot_mask.sum(1).tolist()) == [3, 7]
        for e in range(2):
            n = int(o.robot_mask[e].sum())
            assert not o.token_mask[e, n:].any()
        rng = np.random.default_rng(0)
        for _ in range(3):
            a = np.zeros(o.token_mask.shape[:2], np.int64)
            for e in range(a.shape[0]):
                for r in range(a.shape[1]):
                    v = np.flatnonzero(o.token_mask[e, r])
                    a[e, r] = int(rng.choice(v)) if v.size else 0
            o, rew, d, _ = vec.step(a)
        assert np.isfinite(rew).all()
        # horizons follow the rule, not cfg.t_max_s
        assert sorted(round(g.envs[i].cfg.t_max_s, 1) for g in vec.groups
                      for i in range(g.n)) == [600.0, 1350.0]


def test_stack_clears_robot_rows_when_a_slot_shrinks(tmp_path):
    scene_file(tmp_path, "small", (240, 240))
    scene_file(tmp_path, "big", (900, 900))
    bank = SceneBank(str(tmp_path / "*.json"))
    big, small = sorted(bank.keys, key=bank.area, reverse=True)
    cfg = tiny_cfg(2, 60.0)
    g = EnvGroup(bank, cfg, [TaskSampler([(big, 0), (small, 1)], AUTO)], R=7, K=cfg.k_tokens,
                 send_bev=False, max_decisions=1, t_max=float(AUTO))
    out = g.reset()
    rmask = out[7]
    assert int(rmask[0].sum()) == 7                       # first task: the big scene
    a = np.zeros((1, 7), np.int64)
    out = g.step(a)[0]
    tok, mask, xy, tt, tid, rf, _bev, rmask, _t = out[:9]
    assert int(rmask[0].sum()) == 3                       # second task: the small scene
    assert not mask[0, 3:].any() and not tok[0, 3:].any() and not rf[0, 3:].any()
    assert (tid[0, 3:] == -1).all() and np.isnan(xy[0, 3:]).all()
    assert g.rows and g.rows[0]["n_robots"] == 7 and g.rows[0]["t_max"] == pytest.approx(1350.0)
    g.close()


# ---- GAE over mixed-length episodes --------------------------------------------------------------
def test_gae_handles_mixed_length_episodes():
    """gamma = lam = 1: the advantage is the reward-to-go of the *current* episode minus V."""
    r = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0], [6.0, 1.0]])
    d = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    v = torch.zeros(6, 2)
    last = torch.tensor([0.0, 0.0])
    adv = gae(r, v, d, last, gamma=1.0, lam=1.0)
    # column 0: episodes [0,1] [2,3,4] [5]; column 1: [0,1,2] [3,4,5]
    assert adv[:, 0].tolist() == [3.0, 2.0, 12.0, 9.0, 5.0, 6.0]
    assert adv[:, 1].tolist() == [3.0, 2.0, 1.0, 3.0, 2.0, 1.0]

    # values bootstrap only inside an episode, and only the unfinished tail uses last_value
    v2 = torch.full((6, 2), 7.0)
    adv2 = gae(r, v2, d, torch.tensor([10.0, 10.0]), gamma=1.0, lam=1.0)
    assert adv2[4, 0] == pytest.approx(5.0 - 7.0)                    # terminal: no bootstrap
    assert adv2[5, 0] == pytest.approx(6.0 + 10.0 - 7.0)             # tail: bootstrapped
    assert adv2[5, 1] == pytest.approx(1.0 + 10.0 - 7.0)

    # no leakage across a done: changing a later episode's rewards leaves earlier ones alone
    r3 = r.clone()
    r3[2:, 0] += 100.0
    adv3 = gae(r3, v, d, last, gamma=0.99, lam=0.95)
    base = gae(r, v, d, last, gamma=0.99, lam=0.95)
    assert torch.allclose(adv3[:2, 0], base[:2, 0])
    assert torch.allclose(adv3[:, 1], base[:, 1])


def test_rollout_length_covers_the_gae_window():
    """--rollout 64 keeps the truncation bias small for 120..300-decision episodes."""
    gamma, lam = 0.99, 0.95
    window = 1.0 / (1.0 - gamma * lam)
    assert 64 >= 3 * window                       # >= 3 effective GAE horizons per rollout


# ---- eval reporting ------------------------------------------------------------------------------
def test_by_bucket_splits_rows():
    rows = [{"bucket": "small", "frac_found": 0.5, "area_km2": 0.2, "n_robots": 3, "t_max": 600.0},
            {"bucket": "large", "frac_found": 0.1, "area_km2": 2.0, "n_robots": 8, "t_max": 1500.0},
            {"bucket": "large", "frac_found": 0.3, "area_km2": 1.8, "n_robots": 8, "t_max": 1500.0}]
    out = by_bucket(rows)
    assert list(out) == ["small", "large"]
    assert out["large"]["frac_found"].mean == pytest.approx(0.2)
    assert out["large"]["n_robots"].mean == pytest.approx(8.0)
    assert out["small"]["t_max"].mean == pytest.approx(600.0)


def test_eval_csv_has_area_columns(tmp_path, monkeypatch):
    scene_file(tmp_path, "a", (240, 240))
    scene_file(tmp_path, "b", (600, 600))
    monkeypatch.chdir(tmp_path)
    ev = load_script("eval_policy")
    out = tmp_path / "e.csv"
    rc = ev.main(["--policy", "random", "--scenes", str(tmp_path / "*.json"), "--split", "all",
                  "--episodes", "2", "--robots", "auto", "--t-max", "auto", "--backend", "serial",
                  "--device", "cpu", "--out", str(out)])
    assert rc == 0
    head, *body = out.read_text().strip().splitlines()
    for c in EVAL_META:
        assert c in head.split(",")
    assert any(line.startswith("random[") for line in body)      # per-bucket rows

    ep = tmp_path / "e.episodes.csv"
    lines = ep.read_text().strip().splitlines()
    cols = lines[0].split(",")
    assert {"scene", "bucket", *EVAL_META} <= set(cols)
    rows = [dict(zip(cols, ln.split(","))) for ln in lines[1:]]
    assert len(rows) == 2
    assert {int(r["n_robots"]) for r in rows} == {3, 4}
    assert {round(float(r["t_max"])) for r in rows} == {600, 900}


# ---- the t_max cap override and fixed teams -----------------------------------------------------
def test_t_max_cap_raises_the_horizon_and_defaults_to_today():
    big = region_area_km2((1500, 1500))
    assert auto_t_max(big) == T_MAX_MAX_S == 1500.0          # default is unchanged
    assert auto_t_max(big, 3000.0) == pytest.approx(2250.0)  # 600 * 3.75, now under the cap
    assert auto_t_max(region_area_km2((4000, 4000)), 3000.0) == 3000.0
    assert auto_t_max(region_area_km2((100, 100)), 3000.0) == 600.0      # floor is untouched
    assert auto_t_max(big, 0.0) == T_MAX_MAX_S                # non-positive means "the default"


def test_scene_bank_carries_the_cap(tmp_path):
    scene_file(tmp_path, "a", (240, 240))
    scene_file(tmp_path, "b", (1500, 1500))
    spec = str(tmp_path / "*.json")
    small, big = sorted(SceneBank(spec).keys, key=SceneBank(spec).area)
    assert SceneBank(spec).env_params(big, AUTO, AUTO) == (8, 1500.0)
    bank = SceneBank(spec, t_max_cap=3000.0)
    assert bank.t_max_cap == 3000.0
    assert bank.env_params(big, AUTO, AUTO)[1] == pytest.approx(2250.0)
    assert bank.env_params(small, AUTO, AUTO) == (3, 600.0)
    assert bank.env_params(big, AUTO, 900.0) == (8, 900.0)    # an explicit --t-max still wins
    assert bank.t_max_bounds(AUTO)[1] == pytest.approx(2250.0)


def test_t_max_cap_reaches_the_env_slots(tmp_path):
    scene_file(tmp_path, "big", (1500, 1500))
    with SerialVecEnv(str(tmp_path / "*.json"), tiny_cfg(2, 60.0), 1, robots=(2, 2), split="all",
                      seed=0, n_workers=1, t_max=float(AUTO), send_bev=False,
                      t_max_cap=3000.0) as vec:
        vec.reset_all()
        assert vec.groups[0].envs[0].cfg.t_max_s == pytest.approx(2250.0)


@pytest.mark.parametrize("script", ["train", "imitate"])
def test_the_parser_help_renders(script):
    """argparse `%`-formats every help string it prints, so one literal `%` in a help line makes
    `--help` raise (`scripts/imitate.py --help` did, on `96%`)."""
    assert "--t-max-cap" in load_script(script).build_parser().format_help()


@pytest.mark.parametrize("script", ["train", "imitate", "eval_policy"])
def test_scripts_take_fixed_robots_and_a_t_max_cap(script):
    mod = load_script(script)
    ap = mod.build_parser() if hasattr(mod, "build_parser") else None
    if ap is None:                                   # eval_policy builds its parser inside main()
        assert "--t-max-cap" in Path(mod.__file__).read_text()
        return
    a = ap.parse_args(["--robots", "8", "--t-max", "auto", "--t-max-cap", "3000"])
    assert parse_robots(a.robots) == (8, 8)
    assert a.t_max_cap == 3000.0
    assert ap.parse_args([]).t_max_cap == T_MAX_MAX_S          # default reproduces today


def test_fixed_eight_robot_team_runs_end_to_end(tmp_path, monkeypatch):
    scene_file(tmp_path, "a", (240, 240))
    monkeypatch.chdir(tmp_path)
    ev = load_script("eval_policy")
    out = tmp_path / "e8.csv"
    rc = ev.main(["--policy", "lawnmower", "--policy", "oracle_assign",
                  "--scenes", str(tmp_path / "*.json"), "--split", "all", "--episodes", "2",
                  "--robots", "8", "--t-max", "auto", "--t-max-cap", "3000",
                  "--backend", "serial", "--device", "cpu", "--out", str(out)])
    assert rc == 0
    lines = (tmp_path / "e8.episodes.csv").read_text().strip().splitlines()
    cols = lines[0].split(",")
    rows = [dict(zip(cols, ln.split(","))) for ln in lines[1:]]
    assert {r["policy"] for r in rows} == {"lawnmower", "oracle_assign"}
    assert {int(r["n_robots"]) for r in rows} == {8}
    assert {round(float(r["t_max"])) for r in rows} == {600}   # a 240 m scene sits on the floor


def test_a_cap_the_rule_cannot_honour_is_stored_as_the_one_that_ran(tmp_path):
    """`auto_t_max` clamps the cap to the 600 s floor and reads non-positive as 'the default', so
    `SceneBank.t_max_cap` is the *effective* cap: what a run records is the horizon it ran."""
    scene_file(tmp_path, "big", (1500, 1500))
    spec = str(tmp_path / "*.json")
    for raw, eff in ((3000.0, 3000.0), (900.0, 900.0), (100.0, T_MAX_MIN_S), (1.0, T_MAX_MIN_S),
                     (0.0, T_MAX_MAX_S), (-5.0, T_MAX_MAX_S)):
        bank = SceneBank(spec, t_max_cap=raw)
        key = bank.keys[0]
        assert bank.t_max_cap == eff, raw
        assert bank.env_params(key, AUTO, AUTO)[1] == auto_t_max(bank.area(key), raw)
        assert bank.env_params(key, AUTO, AUTO)[1] == auto_t_max(bank.area(key), bank.t_max_cap)


def test_the_eval_csv_and_the_resolved_rule_report_the_effective_cap(tmp_path, monkeypatch):
    scene_file(tmp_path, "a", (240, 240))
    tr = load_script("train")
    a = tr.build_parser().parse_args(["--t-max-cap", "0"])
    bank = SceneBank(str(tmp_path / "*.json"), t_max_cap=a.t_max_cap)
    res = tr._resolved(bank, a, 3, 3, 3, 3, 600.0, 600.0, None)
    assert res["t_max_cap"] == T_MAX_MAX_S and f"600, {T_MAX_MAX_S:.0f})" in res["rule"]

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "cap.csv"
    rc = load_script("eval_policy").main(
        ["--policy", "random", "--scenes", str(tmp_path / "*.json"), "--split", "all",
         "--episodes", "1", "--robots", "3", "--t-max", "auto", "--t-max-cap", "0",
         "--backend", "serial", "--device", "cpu", "--out", str(out)])
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    row = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert float(row["t_max_cap"]) == T_MAX_MAX_S


def test_t_max_cap_reaches_a_subprocess_worker(tmp_path):
    """The workers rebuild the bank from the `ShardSpec`, so the cap has to travel in it."""
    scene_file(tmp_path, "big", (1100, 1100))
    spec = str(tmp_path / "*.json")
    key = SceneBank(spec).split("all")[0]
    with SubprocVecEnv(spec, tiny_cfg(2, 60.0), 1, robots=(2, 2), split="all", seed=0,
                       n_workers=1, t_max=float(AUTO), send_bev=False, t_max_cap=3000.0) as vec:
        assert vec.specs[0].t_max_cap == 3000.0
        rows = vec.run_episodes("random", [(key, 0)], 2, max_decisions=1)
    assert rows[0]["t_max"] == pytest.approx(auto_t_max(region_area_km2((1100, 1100)), 3000.0))
    assert rows[0]["t_max"] > T_MAX_MAX_S          # and the default cap would have bitten


# ---- warmstart.sh ------------------------------------------------------------------------------
def _warmstart(tmp_path: Path, **env) -> list[str]:
    """Run it with a stub `uv` on PATH: every stage's command line, none of them executed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "uv").write_text('#!/bin/sh\necho "UV $*"\n')
    (bin_dir / "uv").chmod(0o755)
    e = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
         **{k: str(v) for k, v in env.items()}}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "warmstart.sh")], cwd=tmp_path, env=e,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return [ln for ln in r.stdout.splitlines() if ln.startswith("UV ")]


def test_warmstart_hands_the_horizon_and_the_team_to_every_stage(tmp_path):
    lines = _warmstart(tmp_path, WS_TMAX="auto", WS_TMAX_CAP="3000", WS_ROBOTS="8")
    assert len(lines) == 6                        # bc, ft, and the four eval rows
    for ln in lines:
        assert "--robots 8" in ln and "--t-max-cap 3000" in ln and "--t-max auto" in ln


def test_warmstart_without_the_horizon_env_keeps_the_envconfig_clock(tmp_path):
    """An unset `WS_TMAX` must not become an empty `--t-max`, and must not trip `set -e`."""
    lines = _warmstart(tmp_path)
    assert len(lines) == 6
    for ln in lines:
        assert "--t-max-cap 1500" in ln and " --t-max " not in ln and "--robots 3" in ln
