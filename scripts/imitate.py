#!/usr/bin/env python
"""DAgger from a privileged teacher into the token policy (stage 1 of BC -> PPO).

    uv run python scripts/imitate.py --variant central_full --name bc_central \
        --scenes synthetic:0-200 --robots 3 --iters 4 --steps 96 --envs 16 --workers 4
    uv run python scripts/imitate.py --smoke --name bc_smoke

Writes `runs/<name>/bc.pt`, which `scripts/train.py --init-from` picks up: the checkpoint carries
`policy_config` (including `sequential_decode`) and the variant it was cloned under, so the PPO
stage cannot silently change the execution model.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import yaml

from rlplanner.sim.config import EnvConfig
from rlplanner.sim.tokens import BEV_CHANNELS
from rlplanner.train.evaluate import (EVAL_COLS, HEURISTICS, PRIVILEGED, TorchActor, append_csv,
                                      evaluate_baselines, evaluate_policy, format_table,
                                      summarise, write_csv)
from rlplanner.train.imitation import (DaggerConfig, Imitator, LabelBuffer, agreement,
                                       beta_schedule, collect, label_histogram, teacher_states)
from rlplanner.train.par_env import make_vec_env
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.rollout import EpisodeStats
from rlplanner.train.scenes import AUTO, T_MAX_MAX_S, SceneBank, parse_robots, parse_t_max
from rlplanner.train.teachers import TEACHER_RADIUS_M, TEACHERS

REPORT = ("frac_found", "finds_auc", "time_to_first", "redundancy_frac", "intentional_revisits")
ITER_COLS = ("iter", "beta", "decisions", "labels", "buffer_decisions", "buffer_labels",
             "buffer_gb", "kb_per_decision", "ce", "accuracy", "entropy", "dropped",
             "agreement", "mode_slot", "mode_frac", "t_collect_s", "t_train_s",
             "t_eval_s") + EVAL_COLS + tuple(f"{c}_stoch" for c in EVAL_COLS)
SMOKE = {"scenes": "synthetic:0-12", "iters": 2, "steps": 8, "envs": 4, "workers": 2,
         "eval_episodes": 2, "agree_steps": 4, "epochs": 1}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="bc")
    ap.add_argument("--variant", default=None, help="configs/variants/<name>.yaml")
    ap.add_argument("--config", default=None, help="EnvConfig yaml")
    ap.add_argument("--dynamic-queries", action="store_true",
                    help="sample and edit the mission queries per episode (EnvConfig.queries_dynamic)")
    ap.add_argument("--dq-every", type=int, default=10, help="decisions between query edit draws")
    ap.add_argument("--dq-noise", type=float, default=0.0,
                    help="per-dim noise on a drawn query embedding")
    ap.add_argument("--scenes", default="synthetic:0-200", nargs="+")
    ap.add_argument("--robots", default="3")
    ap.add_argument("--t-max", default=None)
    ap.add_argument("--t-max-cap", type=float, default=T_MAX_MAX_S,
                    help=f"upper clip of the area-scaled t_max in seconds (default "
                         f"{T_MAX_MAX_S:.0f}); only bites with --t-max auto")
    ap.add_argument("--teacher", default="oracle_sweep", choices=sorted(TEACHERS))
    ap.add_argument("--teacher-radius", type=float, default=TEACHER_RADIUS_M,
                    help="oracle_sweep sweeps when no token is this close to an unfound casualty")
    ap.add_argument("--iters", type=int, default=4, help="DAgger iterations (0 = teacher only)")
    ap.add_argument("--steps", type=int, default=96, help="decisions per env slot per iteration")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--env-backend", default="subproc", choices=("subproc", "serial"))
    ap.add_argument("--epochs", type=int, default=4, help="passes over the buffer per iteration")
    ap.add_argument("--batch", type=int, default=32, help="decisions per minibatch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=False,
                    help="inverse-frequency weighting over the label's token type. Off by "
                         "default: it multiplies the rare `hold` label by ~4 and an under-trained "
                         "student then holds position in 96%% of its greedy decisions")
    ap.add_argument("--max-samples", type=int, default=2_000_000, help="robot-decisions held")
    ap.add_argument("--max-gb", type=float, default=8.0, help="RAM cap of the label buffer")
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1, help="iterations between held-out evals")
    ap.add_argument("--agree-steps", type=int, default=64,
                    help="held-out decisions per slot in the teacher-agreement probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--use-local", action=argparse.BooleanOptionalAction, default=True,
                    help="dense ego-centric crop for the actor (on by default here)")
    ap.add_argument("--use-bev", action=argparse.BooleanOptionalAction, default=True,
                    help="critic switch; BC never ships a BEV, it only sizes the head for PPO")
    ap.add_argument("--use-robot-bev", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--robot-bev-size", type=int, default=32)
    ap.add_argument("--peers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sequential-decode", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


def apply_variant(ap: argparse.ArgumentParser, a: argparse.Namespace) -> EnvConfig | None:
    """The variant's EnvConfig plus its `train:` block, for the flags this script shares."""
    if not a.variant:
        return None
    path = Path(a.variant)
    if not path.exists():
        path = Path("configs/variants") / f"{a.variant}.yaml"
    d = dict(yaml.safe_load(path.read_text()) or {})
    train = dict(d.pop("train", {}) or {})
    d.pop("notes", None)
    for k, v in train.items():
        k = k.replace("-", "_")
        if hasattr(a, k) and getattr(a, k) == ap.get_default(k):
            setattr(a, k, v)
    return EnvConfig.from_dict(d)


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    a.scenes = " ".join(a.scenes) if isinstance(a.scenes, list) else a.scenes
    if a.smoke:
        for k, v in SMOKE.items():
            if getattr(a, k) == ap.get_default(k):
                setattr(a, k, v)
    cfg_variant = apply_variant(ap, a)
    if a.sequential_decode is None:
        a.sequential_decode = a.variant is None or "central" in str(a.variant)
    if a.use_robot_bev is None:
        a.use_robot_bev = False
    dev = (torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto"
           else torch.device(a.device))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)
    cfg = cfg_variant or (EnvConfig.from_yaml(a.config) if a.config else EnvConfig())
    if not a.use_local:
        cfg.tokens.local_size = 0
    cfg.tokens.robot_bev_size = int(a.robot_bev_size) if a.use_robot_bev else 0
    if a.dynamic_queries:
        cfg.queries_dynamic.enabled = True
        cfg.queries_dynamic.every = int(a.dq_every)
        cfg.queries_dynamic.noise_std = float(a.dq_noise)
    lo, hi = parse_robots(a.robots)
    t_spec = parse_t_max(a.t_max)
    bank = SceneBank(a.scenes, t_max_cap=a.t_max_cap)
    cfg.robot.n_robots = bank.robot_bounds((lo, hi))[0]
    if t_spec > 0:
        cfg.t_max_s = float(t_spec)
    vec_t_max = None if t_spec > 0 else float(AUTO)
    errs = cfg.validate()
    if errs:
        raise SystemExit(f"[imitate] invalid EnvConfig: {'; '.join(errs)}")
    (run / "args.json").write_text(json.dumps(vars(a), indent=2, default=str))

    dcfg = DaggerConfig(iters=a.iters, steps=a.steps, beta0=a.beta0, epochs=a.epochs,
                        batch=a.batch, lr=a.lr, label_smoothing=a.label_smoothing,
                        ent_coef=a.ent_coef, balance=a.balance, max_samples=a.max_samples,
                        max_gb=a.max_gb)
    t_start = time.perf_counter()

    base: dict = {}
    if not a.no_baselines:
        t0 = time.perf_counter()
        names = list(HEURISTICS) + list(PRIVILEGED) + \
            ([a.teacher] if a.teacher not in PRIVILEGED else [])
        base = evaluate_baselines(names, bank, _thin(cfg), a.eval_episodes, lo, seed=10_000,
                                  workers=a.workers, t_max=vec_t_max)
        write_csv(run / "baselines.csv", base, EVAL_COLS)
        print(f"[imitate] reference policies, {a.eval_episodes} held-out episodes "
              f"({time.perf_counter() - t0:.1f}s)\n" + format_table(base, REPORT))

    mk = dict(cfg=cfg, robots=(lo, hi), seed=a.seed, send_bev=False, t_max=vec_t_max,
              t_max_cap=a.t_max_cap)
    hold = make_vec_env(a.env_backend, a.scenes, n_envs=a.workers, split="heldout",
                        n_workers=a.workers, **mk)
    vec = make_vec_env(a.env_backend, a.scenes, n_envs=a.envs, split="train",
                       n_workers=a.workers, **mk)
    try:
        return _run(a, dcfg, run, cfg, bank, vec, hold, dev, lo, vec_t_max, base, t_start)
    finally:
        vec.close()
        hold.close()


def _thin(cfg: EnvConfig) -> EnvConfig:
    """A copy with no dense rasters: the reference policies never read them."""
    import copy
    c = copy.deepcopy(cfg)
    c.tokens.local_size = 0
    c.tokens.robot_bev_size = 0
    return c


def _run(a, dcfg: DaggerConfig, run: Path, cfg: EnvConfig, bank, vec, hold, dev, lo: int,
         vec_t_max, base: dict, t_start: float) -> int:
    probe = vec.reset_all()
    hold.reset_all()
    policy = TokenPolicy(
        token_dim=probe.tokens.shape[3], robot_dim=probe.robot_feat.shape[2],
        feat_dim=probe.query_emb.shape[2], d_model=a.d_model,
        use_bev=bool(a.use_bev), bev_channels=len(BEV_CHANNELS),
        use_local=bool(a.use_local),
        local_channels=(probe.local.shape[2] if probe.local is not None else 0),
        use_peers=bool(a.peers and probe.peer_tokens is not None
                       and probe.peer_tokens.shape[2] > 0),
        peer_dim=(probe.peer_tokens.shape[3] if probe.peer_tokens is not None
                  and probe.peer_tokens.ndim == 4 else 0),
        use_robot_bev=bool(a.use_robot_bev and probe.robot_bev is not None),
        robot_bev_channels=(probe.robot_bev.shape[2] if probe.robot_bev is not None else 0),
        sequential_decode=bool(a.sequential_decode)).to(dev)
    imi = Imitator(policy, dcfg, dev, seed=a.seed)
    print(f"[imitate] {a.name} teacher={a.teacher} r={a.teacher_radius:.0f}m device={dev} "
          f"variant={a.variant} sequential_decode={policy.sequential_decode} "
          f"use_local={policy.use_local} use_robot_bev={policy.use_robot_bev} "
          f"envs={a.envs}x{a.workers}w steps={a.steps} iters={a.iters} "
          f"params={sum(p.numel() for p in policy.parameters()):,}")

    t0 = time.perf_counter()
    probe_states = LabelBuffer(max_samples=10 ** 9, max_gb=1.5, rng=np.random.default_rng(7))
    teacher_states(hold, probe_states, a.agree_steps, a.teacher, a.teacher_radius, dev, policy)
    print(f"[imitate] agreement probe: {len(probe_states)} held-out decisions, "
          f"{probe_states.n_labels} labels, {probe_states.nbytes / 1e9:.2f} GB "
          f"({time.perf_counter() - t0:.1f}s)")

    stats = EpisodeStats(vec.n_envs)
    for it in range(int(a.iters)):
        beta = beta_schedule(it, a.iters, a.beta0)
        t0 = time.perf_counter()
        cs = collect(vec, policy, imi.buffer, a.steps, beta, a.teacher, a.teacher_radius, dev,
                     imi.rng, stats)
        t_col = time.perf_counter() - t0
        t0 = time.perf_counter()
        tr = imi.train_epochs()
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        agr = agreement(policy, probe_states, dev)
        res, sres = {}, {}
        do_eval = a.eval_every > 0 and ((it + 1) % a.eval_every == 0 or it == a.iters - 1)
        if do_eval:
            # both decode rules: the greedy row is the one comparable with the PPO sweep, the
            # sampled row says whether a weak argmax is hiding a usable distribution
            for det in (True, False):
                actor = TorchActor(policy, dev, deterministic=det, name=a.name)
                raw = evaluate_policy(actor, bank, cfg, a.eval_episodes, lo, seed=10_000,
                                      split="heldout", pool=hold, t_max=vec_t_max)
                (res if det else sres).update(summarise(raw))
            policy.train()
        t_eval = time.perf_counter() - t0
        ep = stats.drain()
        mslot, mfrac = imi.buffer.mode_slot()
        row = {"iter": it, "beta": beta, "decisions": cs["decisions"],
               "labels": float(imi.buffer.seen_labels),
               "buffer_decisions": float(len(imi.buffer)),
               "buffer_labels": float(imi.buffer.n_labels),
               "buffer_gb": imi.buffer.nbytes / 1e9,
               "kb_per_decision": imi.buffer.bytes_per_decision / 1e3,
               "agreement": agr, "mode_slot": float(mslot), "mode_frac": mfrac,
               "t_collect_s": t_col, "t_train_s": t_train, "t_eval_s": t_eval}
        row.update({k: tr.get(k, float("nan")) for k in ("ce", "accuracy", "entropy", "dropped")})
        row.update({c: (res[c].mean if c in res else float("nan")) for c in EVAL_COLS})
        row.update({f"{c}_stoch": (sres[c].mean if c in sres else float("nan"))
                    for c in EVAL_COLS})
        append_csv(run / "iters.csv", {k: row[k] for k in ITER_COLS})
        hist = label_histogram(imi.buffer.label_type_counts())
        print(f"[iter {it}] beta {beta:.2f}  ce {row['ce']:.3f}  acc {row['accuracy']:.3f}  "
              f"agree {agr:.3f} (mode slot {mslot} = {mfrac:.3f})  buffer {len(imi.buffer)} "
              f"dec / {imi.buffer.n_labels} lab / {row['buffer_gb']:.2f} GB "
              f"({row['kb_per_decision']:.0f} kB/dec)  labels[{hist}]  "
              f"train_ep R {ep.get('reward', float('nan')):+.2f} ff "
              f"{ep.get('frac_found', float('nan')):.3f}  "
              f"({t_col:.0f}s collect / {t_train:.0f}s train / {t_eval:.0f}s eval)")
        if do_eval:
            print(format_table({f"{a.name}@{it}": res, f"{a.name}@{it}~sampled": sres, **base},
                               REPORT))
        torch.save({**imi.state_dict(), "iter": it, "args": vars(a),
                    "variant": a.variant, "teacher": a.teacher,
                    "labels": int(imi.buffer.seen_labels)}, run / "bc.pt")

    print(f"[imitate] done: {run}/bc.pt  {imi.buffer.seen_labels} labelled robot-decisions, "
          f"{time.perf_counter() - t_start:.0f}s wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
