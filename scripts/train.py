#!/usr/bin/env python
"""MAPPO training for the token policy.

    uv run python scripts/train.py --name run1 --scenes synthetic:0-200 --robots 3
    uv run python scripts/train.py --smoke --name smoke
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
from rlplanner.train.evaluate import (BASELINES, EVAL_COLS, EVAL_META, TorchActor, append_csv,
                                      evaluate_baselines, evaluate_policy, format_table,
                                      meta_arrays, summarise, write_csv, write_episode_csv)
from rlplanner.train.par_env import default_workers, make_vec_env
from rlplanner.train.policy import TokenPolicy
from rlplanner.train.ppo import PPO, PPOConfig, bc_reference
from rlplanner.train.rollout import Collector
from rlplanner.train.scenes import (AUTO, T_MAX_MAX_S, SceneBank, auto_robots, auto_t_max,
                                    parse_robots, parse_scene_mix, parse_t_max)

LOG_COLS = ("update", "env_steps", "decisions", "ep_reward", "frac_found", "episodes",
            "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
            "explained_variance", "grad_norm", "lr", "actor_lr", "bc_kl", "bc_kl_coef",
            "minibatches", "fps")
CURVE_PANELS = (("reward", "episode reward"), ("frac_found", "fraction found"),
                ("finds_auc", "finds AUC"), ("time_to_first", "time to first (s)"))
SMOKE = {"envs": 8, "rollout": 16, "updates": 30, "eval_episodes": 4, "eval_every": 10,
         "scenes": "synthetic:0-20"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run", help="run directory under runs/")
    ap.add_argument("--scenes", default="synthetic:0-200", nargs="+",
                    help="synthetic:A-B (end exclusive) or scene-json globs/paths "
                         "(an unquoted shell glob is fine)")
    ap.add_argument("--robots", default="3",
                    help="N, LO-HI (randomised per env slot) or 'auto' (area rule: "
                         "clip(round(3*sqrt(area_km2/0.16)), 3, 8) per scene)")
    ap.add_argument("--t-max", default=None,
                    help="episode horizon in seconds, or 'auto' (area rule: "
                         "clip(600*sqrt(area_km2/0.16), 600, 1500)); default = the EnvConfig value")
    ap.add_argument("--t-max-cap", type=float, default=T_MAX_MAX_S,
                    help=f"upper clip of the area-scaled t_max in seconds (default "
                         f"{T_MAX_MAX_S:.0f}); only bites with --t-max auto")
    ap.add_argument("--scene-mix", default=None,
                    help="sampling weights over the bank's area terciles, e.g. "
                         "'small:0.5,medium:0.3,large:0.2' (default: uniform over scenes)")
    ap.add_argument("--envs", type=int, default=32,
                    help="env slots; more slots per worker hides straggler latency "
                         "(32 -> ~1.2k dec/s, 96 -> ~1.5k on 6 cores)")
    ap.add_argument("--env-backend", default="subproc", choices=("subproc", "serial"),
                    help="serial runs the same shards in-process (debugging)")
    ap.add_argument("--workers", type=int, default=None,
                    help="env worker processes (default min(nproc - 2, envs))")
    ap.add_argument("--rollout", type=int, default=64, help="decisions per env per update")
    ap.add_argument("--updates", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use-bev", action="store_true",
                    help="give the centralised critic the compressed global BEV")
    ap.add_argument("--use-local", action="store_true",
                    help="give the actor the ego-centric local crop (off: the env skips building it)")
    ap.add_argument("--use-robot-bev", action="store_true",
                    help="give the actor a BEV of its *own* belief (--robot-bev-size cells)")
    ap.add_argument("--robot-bev-size", type=int, default=32,
                    help="side of the per-robot actor BEV; 64 x 22ch x R is too heavy for the "
                         "worker pipes, 32 costs ~94 kB per robot per decision")
    ap.add_argument("--peers", action=argparse.BooleanOptionalAction, default=True,
                    help="actor attends over the peer tokens (what gossip last said about a peer)")
    ap.add_argument("--sequential-decode", action=argparse.BooleanOptionalAction, default=True,
                    help="centralised execution: robots decode in index order and a token a "
                         "lower-index robot just claimed is masked for the later ones. Off for "
                         "every decentral variant — at decentralised execution a peer's intent "
                         "only arrives through gossip, one decision later (CONTRACTS.md 6)")
    ap.add_argument("--comms", default=None, choices=("full", "range"),
                    help="override EnvConfig.comms.mode")
    ap.add_argument("--comms-range", default=None,
                    help="fixed link range in metres ('inf' allowed); disables the per-episode "
                         "randomisation over comms.range_choices")
    ap.add_argument("--dynamic-queries", action="store_true",
                    help="sample and edit the mission queries per episode (EnvConfig.queries_dynamic)")
    ap.add_argument("--dq-every", type=int, default=None,
                    help="decisions between query edit draws (default: whatever the config says)")
    ap.add_argument("--dq-noise", type=float, default=None,
                    help="per-dim noise on a drawn query embedding (default: the config's)")
    ap.add_argument("--variant", default=None,
                    help="configs/variants/<name>.yaml (EnvConfig + a train: block of flags, "
                         "including policy.sequential_decode)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (< 5 min)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rotate-every", type=int, default=0,
                    help="deprecated no-op: env slots resample their scene on episode end")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=True,
                    help="linear decay of the learning rate to 0 over --updates")
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="stop an update once approx_kl exceeds this (<= 0 disables)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--config", default=None, help="EnvConfig yaml")
    ap.add_argument("--no-baselines", action="store_true", help="skip the baseline evals")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--init-from", default=None,
                    help="behaviour-cloned checkpoint (scripts/imitate.py). Its policy_config — "
                         "architecture, use_local/use_bev/use_robot_bev and sequential_decode — "
                         "wins over the command line, so the fine-tune cannot change the "
                         "execution model the student was cloned under")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="updates with the actor frozen (value head only): PPO otherwise moves a "
                         "good actor with a critic that predicts nothing")
    ap.add_argument("--actor-warmup", type=int, default=0,
                    help="updates over which the actor learning rate ramps 0 -> --lr (after the "
                         "critic warm-up); the critic is at full lr throughout")
    ap.add_argument("--bc-kl", type=float, default=0.0,
                    help="initial KL(pi || pi_bc) coefficient, annealed linearly to 0 over "
                         "--updates (needs --init-from)")
    return ap


def load_init(a: argparse.Namespace) -> dict | None:
    """Adopt the BC checkpoint's architecture and execution model before anything is built."""
    if not a.init_from:
        return None
    return apply_init(a, torch.load(a.init_from, map_location="cpu", weights_only=False))


def apply_init(a: argparse.Namespace, ck: dict) -> dict:
    pc = ck["policy_config"]
    a.use_bev = bool(pc["use_bev"])
    a.use_local = bool(pc["use_local"])
    a.use_robot_bev = bool(pc["use_robot_bev"])
    a.peers = bool(pc["use_peers"])
    a.sequential_decode = bool(pc["sequential_decode"])
    a.d_model = int(pc["d_model"])
    if a.variant is None and ck.get("variant"):
        a.variant = ck["variant"]
    return ck


def apply_smoke(ap: argparse.ArgumentParser, a: argparse.Namespace) -> argparse.Namespace:
    for k, v in SMOKE.items():
        if getattr(a, k) == ap.get_default(k):
            setattr(a, k, v)
    return a


def apply_variant(ap: argparse.ArgumentParser, a: argparse.Namespace) -> EnvConfig | None:
    """Load `--variant`: an EnvConfig yaml with an extra `train:` block of flag defaults.

    Anything the command line set explicitly (i.e. differs from the parser default) wins over the
    file, so a sweep can override one knob of a variant without editing it.
    """
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
        if not hasattr(a, k):
            raise SystemExit(f"[train] {path}: unknown train flag {k!r}")
        if getattr(a, k) == ap.get_default(k):
            setattr(a, k, v)
    return EnvConfig.from_dict(d)


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def curve(out: Path, log: list[dict], evals: list[dict], base: dict) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = {"random": "tab:gray", "nearest": "tab:green", "lawnmower": "tab:brown",
              "ray_follower": "tab:orange", "segment_seeker": "tab:purple",
              "oracle": "tab:red", "oracle_assign": "tab:pink"}
    for ax, (key, title) in zip(axes.ravel(), CURVE_PANELS):
        if key == "reward":
            tr = [(r["decisions"], r["ep_reward"]) for r in log
                  if r.get("episodes", 0) > 0 and np.isfinite(r.get("ep_reward", np.nan))]
            if tr:
                ax.plot(*zip(*tr), color="tab:blue", alpha=0.3, lw=1, label="train")
        ev = [(r["decisions"], r[key], r.get(f"{key}_ci", np.nan)) for r in evals
              if np.isfinite(r.get(key, np.nan))]
        if ev:
            x, y, ci = (np.array(v, np.float64) for v in zip(*ev))
            ci = np.nan_to_num(ci)
            ax.fill_between(x, y - ci, y + ci, color="tab:blue", alpha=0.18, lw=0)
            ax.plot(x, y, "o-", color="tab:blue", lw=1.6, ms=3.5, label="policy (eval)")
        for name, s in base.items():
            if key not in s:
                continue
            m, c = _mean_ci(s[key])
            ax.axhline(m, ls="--", lw=1.1, color=colors.get(name, "k"), label=name)
            if np.isfinite(c) and c > 0:
                ax.axhspan(m - c, m + c, color=colors.get(name, "k"), alpha=0.10, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("decisions")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(out.parent.name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _mean_ci(s) -> tuple[float, float]:
    return (float(s[0]), float(s[2]) if len(s) > 2 else float("nan"))


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    a.scenes = " ".join(a.scenes) if isinstance(a.scenes, list) else a.scenes
    if a.smoke:
        a = apply_smoke(ap, a)
    if a.rotate_every:
        print("[train] --rotate-every is deprecated and ignored: every env slot draws a fresh "
              "scene when its episode ends, so no partial episode is discarded.")
    init = load_init(a)
    cfg_from_variant = apply_variant(ap, a)
    if init is not None:
        apply_init(a, init)              # the variant must not undo the checkpoint's own switches
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)
    cfg = cfg_from_variant or (EnvConfig.from_yaml(a.config) if a.config else EnvConfig())
    if not a.use_local:
        cfg.tokens.local_size = 0            # do not build (or ship) a crop nothing reads
    cfg.tokens.robot_bev_size = int(a.robot_bev_size) if a.use_robot_bev else 0
    if a.comms:
        cfg.comms.mode = a.comms
    if a.comms_range is not None:
        cfg.comms.range_m = float(a.comms_range)
        cfg.comms.randomize_range = False
    if a.dynamic_queries:                        # the flag switches it on; the yaml keeps the rest
        cfg.queries_dynamic.enabled = True
        if a.dq_every is not None:
            cfg.queries_dynamic.every = int(a.dq_every)
        if a.dq_noise is not None:
            cfg.queries_dynamic.noise_std = float(a.dq_noise)
    lo, hi = parse_robots(a.robots)
    t_spec = parse_t_max(a.t_max)
    mix = parse_scene_mix(a.scene_mix)
    bank = SceneBank(a.scenes, t_max_cap=a.t_max_cap)
    rlo, rhi = bank.robot_bounds((lo, hi))
    cfg.robot.n_robots = rlo                     # per-env override happens at env construction
    if t_spec > 0:
        cfg.t_max_s = float(t_spec)
    vec_t_max = None if t_spec > 0 else float(AUTO)
    tlo, thi = bank.t_max_bounds(cfg.t_max_s if vec_t_max is None else float(AUTO))
    errs = cfg.validate()
    if errs:
        raise SystemExit(f"[train] invalid EnvConfig: {'; '.join(errs)}")
    (run / "args.json").write_text(json.dumps(
        {**vars(a), "policy": {"sequential_decode": bool(a.sequential_decode)},
         "resolved": _resolved(bank, a, lo, hi, rlo, rhi, tlo, thi, mix)},
        indent=2, default=str))

    workers = int(a.workers or default_workers(a.envs))
    vec = make_vec_env(a.env_backend, a.scenes, cfg, a.envs, robots=(lo, hi), split="train",
                       seed=a.seed, n_workers=workers, send_bev=a.use_bev, t_max=vec_t_max,
                       scene_mix=mix, t_max_cap=a.t_max_cap)
    try:
        return _train(a, ap, run, cfg, bank, vec, dev, lo, hi, workers, vec_t_max, rlo, rhi, init)
    finally:
        vec.close()


def _resolved(bank: SceneBank, a, lo: int, hi: int, rlo: int, rhi: int, tlo: float, thi: float,
              mix) -> dict:
    """What the area rule actually resolved to, for the record in args.json."""
    areas = sorted(bank.area(k) for k in bank.keys)
    return {
        "rule": "n_robots = clip(round(3*sqrt(area_km2/0.16)), 3, 8); "
                f"t_max_s = clip(600*sqrt(area_km2/0.16), 600, {bank.t_max_cap:.0f})",
        "t_max_cap": float(bank.t_max_cap),      # the effective cap, not the raw flag
        "dynamic_queries": bool(a.dynamic_queries),
        "robots": "auto" if lo <= 0 else [lo, hi], "robots_range": [rlo, rhi],
        "t_max": "auto" if a.t_max and str(a.t_max).lower() == "auto" else a.t_max,
        "t_max_range": [round(tlo, 1), round(thi, 1)],
        "scenes": len(bank), "train": len(bank.train), "heldout": len(bank.heldout),
        "area_km2": {"min": round(areas[0], 4), "median": round(areas[len(areas) // 2], 4),
                     "max": round(areas[-1], 4)},
        "bucket_edges_km2": [round(v, 4) for v in bank.bucket_edges],
        "buckets_train": bank.bucket_counts("train"),
        "buckets_heldout": bank.bucket_counts("heldout"),
        "scene_mix": mix,
        "per_bucket_example": {b: {"area_km2": round(x, 4), "n_robots": auto_robots(x),
                                   "t_max_s": round(auto_t_max(x, bank.t_max_cap), 1)}
                               for b, x in _bucket_examples(bank).items()},
    }


def _bucket_examples(bank: SceneBank) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in bank.keys:
        out.setdefault(bank.bucket(k), bank.area(k))
    return out


def _train(a, ap, run: Path, cfg: EnvConfig, bank: SceneBank, vec, dev, lo: int, hi: int,
           workers: int, vec_t_max: float | None, rlo: int, rhi: int, init: dict | None = None
           ) -> int:
    col = Collector(vec, dev, gamma=a.gamma, lam=a.lam)
    probe = col.obs
    policy = TokenPolicy(token_dim=probe.tokens.shape[3], robot_dim=probe.robot_feat.shape[2],
                         feat_dim=probe.query_emb.shape[2], d_model=a.d_model, use_bev=a.use_bev,
                         bev_channels=probe.bev.shape[1], use_local=a.use_local,
                         local_channels=(probe.local.shape[2] if probe.local is not None
                                         else 0),
                         use_peers=bool(a.peers and probe.peer_tokens is not None
                                        and probe.peer_tokens.shape[2] > 0),
                         peer_dim=(probe.peer_tokens.shape[3]
                                   if probe.peer_tokens is not None
                                   and probe.peer_tokens.ndim == 4 else 0),
                         use_robot_bev=bool(a.use_robot_bev and probe.robot_bev is not None),
                         robot_bev_channels=(probe.robot_bev.shape[2]
                                             if probe.robot_bev is not None else 0),
                         sequential_decode=bool(a.sequential_decode)).to(dev)
    bc_ref = None
    if init is not None:
        policy = TokenPolicy.from_config(init["policy_config"]).to(dev)
        policy.load_state_dict(init["policy"])
        if probe.tokens.shape[3] != policy.token_dim:
            raise SystemExit(f"[train] --init-from: token width {policy.token_dim} does not match "
                             f"the environment's {probe.tokens.shape[3]}")
        if a.bc_kl > 0:
            bc_ref = TokenPolicy.from_config(init["policy_config"]).to(dev)
            bc_ref.load_state_dict(init["policy"])
            bc_ref.eval()
            for prm in bc_ref.parameters():
                prm.requires_grad_(False)
        print(f"[train] init from {a.init_from} (variant={init.get('variant')} "
              f"teacher={init.get('teacher')} labels={init.get('labels')}) "
              f"critic_warmup={a.critic_warmup} actor_warmup={a.actor_warmup} bc_kl={a.bc_kl}")
    ppo = PPO(policy, PPOConfig(lr=a.lr, clip=a.clip, ent_coef=a.ent_coef, epochs=a.epochs,
                                n_minibatches=a.minibatches, bc_kl_coef=a.bc_kl,
                                target_kl=a.target_kl if a.target_kl > 0 else None), dev)
    rob = f"auto({rlo}-{rhi})" if lo <= 0 else (f"{lo}" if lo == hi else f"{lo}-{hi}")
    tm = "auto" if vec_t_max is not None else f"{cfg.t_max_s:.0f}s"
    print(f"[train] comms={cfg.comms.mode} range="
          f"{'random' if cfg.comms.randomize_range and cfg.comms.mode == 'range' else cfg.comms.range_m}"
          f" share={cfg.comms.share} peers={policy.use_peers} robot_bev={policy.use_robot_bev} "
          f"redundancy={cfg.reward.redundancy_cost} revisit={cfg.reward.revisit_cost} "
          f"refund_on_find={cfg.reward.revisit_refund_on_find} "
          f"sequential_decode={policy.sequential_decode}")
    print(f"[train] {a.name} device={dev} scenes={len(bank)} (train {len(bank.train)} / "
          f"heldout {len(bank.heldout)}) buckets={bank.bucket_counts('train')} "
          f"mix={a.scene_mix or 'uniform'} robots={rob} t_max={tm} envs={a.envs} "
          f"backend={a.env_backend}x{workers} rollout={a.rollout} "
          f"params={sum(p.numel() for p in policy.parameters()):,}")

    start, decisions = 1, 0
    log_rows: list[dict] = []
    eval_rows: list[dict] = []
    if a.resume and (run / "latest.pt").exists():
        ck = torch.load(run / "latest.pt", map_location=dev, weights_only=False)
        ppo.load_state_dict(ck)
        start = int(ck.get("update", 0)) + 1
        decisions = int(ck.get("decisions", 0))
        col.env_steps = int(ck.get("env_steps", 0))
        log_rows = _read_rows(run / "log.csv")
        eval_rows = _read_rows(run / "eval.csv")
        print(f"[train] resumed from update {start - 1} ({decisions} decisions)")

    base: dict = {}
    bpath = run / "baselines.csv"
    last_ep = {"reward": float("nan"), "frac_found": float("nan")}
    for u in range(start, a.updates + 1):
        lr = a.lr * (max(0.0, 1.0 - (u - 1) / max(1, a.updates)) if a.lr_decay else 1.0)
        ppo.set_lr(lr)
        ppo.freeze_actor(u <= a.critic_warmup)
        ppo.set_actor_lr(lr * _actor_ramp(u, a.critic_warmup, a.actor_warmup))
        ppo.bc_kl_coef = bc_kl_coef(u, a.updates, a.bc_kl)
        t0 = time.perf_counter()
        batch, st = col.rollout(policy, a.rollout)
        if bc_ref is not None and ppo.bc_kl_coef > 0:
            batch.ref_logp = bc_reference(bc_ref, batch)
        losses = ppo.update(batch)
        dt = time.perf_counter() - t0
        decisions += int(st["decisions"])
        row = {"update": u, "env_steps": int(st["env_steps"]), "decisions": decisions,
               "ep_reward": st.get("reward", float("nan")),
               "frac_found": st.get("frac_found", float("nan")),
               "episodes": st.get("episodes", 0.0),
               "fps": st["decisions"] / max(dt, 1e-9)}
        row.update({k: losses.get(k, float("nan"))
                    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
                              "explained_variance", "grad_norm", "lr", "actor_lr", "bc_kl",
                              "bc_kl_coef", "minibatches")})
        log_rows.append(row)
        append_csv(run / "log.csv", {k: row[k] for k in LOG_COLS})
        if row["episodes"] > 0:
            last_ep = {"reward": row["ep_reward"], "frac_found": row["frac_found"]}
        if u % 5 == 0 or u == start:
            print(f"  u{u:>4} dec {decisions:>7}  R {last_ep['reward']:+.3f}  ff "
                  f"{last_ep['frac_found']:.3f}  pl {row['policy_loss']:+.4f}  vl "
                  f"{row['value_loss']:.4f}  H {row['entropy']:.3f}  kl {row['approx_kl']:.4f}"
                  f"  ev {row['explained_variance']:+.2f}  {row['fps']:.0f} dec/s")

        if a.eval_every > 0 and (u % a.eval_every == 0 or u == a.updates):
            t0 = time.perf_counter()
            actor = TorchActor(policy, dev, deterministic=True, name=a.name)
            raw, rows = evaluate_policy(actor, bank, cfg, a.eval_episodes, lo, seed=10_000,
                                        pool=vec, t_max=vec_t_max, return_rows=True)
            res = summarise({**raw, **meta_arrays(rows)})
            write_episode_csv(run / "eval_episodes.csv",
                              [{**r, "update": u} for r in rows], policy=a.name)
            policy.train()
            if not a.no_baselines:      # cheap now: re-measured against the current sim build
                base = evaluate_baselines(BASELINES, bank, cfg, a.eval_episodes, lo, seed=10_000,
                                          pool=vec, t_max=vec_t_max)
                write_csv(bpath, base, EVAL_COLS + EVAL_META,
                          extra={"episodes": a.eval_episodes, "robots": rob})
            erow = {"update": u, "decisions": decisions, "env_steps": int(st["env_steps"])}
            erow.update({c: res[c].mean for c in EVAL_COLS + EVAL_META})
            erow.update({f"{c}_std": res[c].std for c in EVAL_COLS})
            erow.update({f"{c}_ci": res[c].ci for c in EVAL_COLS})
            eval_rows.append(erow)
            append_csv(run / "eval.csv", erow)
            curve(run / "curve.png", log_rows, eval_rows, base)
            print(f"[eval] {a.eval_episodes} held-out episodes, "
                  f"{time.perf_counter() - t0:.1f}s (mean +- 95% CI)\n"
                  + format_table({a.name: res, **base}))
        if u % a.ckpt_every == 0 or u == a.updates:
            _save(run / f"ckpt_{u}.pt", ppo, a, u, col, decisions)
        _save(run / "latest.pt", ppo, a, u, col, decisions)

    if not (run / "curve.png").exists():
        curve(run / "curve.png", log_rows, eval_rows, base)
    print(f"[train] done: {run}/  ({decisions} decisions, {col.env_steps} env steps)")
    return 0


def bc_kl_coef(u: int, updates: int, bc_kl: float) -> float:
    """Linear anneal of the KL-to-BC weight; exactly 0 on the last update."""
    n = max(1, int(updates))
    return float(bc_kl) * max(0.0, (n - int(u)) / max(1, n - 1))


def _actor_ramp(u: int, critic_warmup: int, actor_warmup: int) -> float:
    """0 while the actor is frozen, then a linear 0 -> 1 ramp over `actor_warmup` updates."""
    if u <= int(critic_warmup):
        return 0.0
    if int(actor_warmup) <= 0:
        return 1.0
    return float(min(1.0, (u - int(critic_warmup)) / float(actor_warmup)))


def _save(path: Path, ppo: PPO, a, update: int, col: Collector, decisions: int) -> None:
    sd = ppo.state_dict()
    sd.update({"update": update, "decisions": decisions, "env_steps": col.env_steps,
               "args": vars(a)})
    torch.save(sd, path)


def _read_rows(path: Path) -> list[dict]:
    import csv as _csv
    if not path.exists():
        return []
    with path.open() as fh:
        return [{k: _num(v) for k, v in r.items()} for r in _csv.DictReader(fh)]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


if __name__ == "__main__":
    raise SystemExit(main())
