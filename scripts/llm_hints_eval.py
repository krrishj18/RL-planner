#!/usr/bin/env python
"""Evaluate a policy on held-out episodes under four query-channel conditions.

    uv run python scripts/llm_hints_eval.py --policy ray_follower --episodes 4
    uv run python scripts/llm_hints_eval.py --policy runs/run1/latest.pt --scenes 'data/scenes_v2/*.json' \
        --conditions none static llm scripted --cadence 5 --backend claude --model opus

Conditions (the *only* thing that differs between them is the query channel):
  `none`     the query block handed to the policy is zeroed and masked — no queries at all.
  `static`   the config's mission queries, fixed for the episode (today's behaviour).
  `llm`      a live `HintAgent` on the `claude` backend, updated every K decisions and on a find.
  `scripted` the same closed loop on the deterministic mock backend (no subprocess, CI-safe).

Runs serially in this process on CPU (the hint loop needs the live env between decisions), prints
the standard eval table and writes the per-condition query-edit log to CSV and JSONL.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from rlplanner.llm import DigestBuilder, HintAgent, HintController, QueryEmbedder
from rlplanner.sim.config import EnvConfig
from rlplanner.sim.embeddings import get_embedding_table
from rlplanner.train.evaluate import (EVAL_COLS, EVAL_META, TorchActor, eval_tasks, format_table,
                                      make_actor, meta_arrays, summarise, write_csv,
                                      write_episode_csv)
from rlplanner.train.scenes import AUTO, SceneBank, parse_robots, parse_t_max

CONDITIONS = ("none", "static", "llm", "scripted")
EDIT_COLS = ("condition", "episode", "scene", "decision", "t", "kind", "backend", "add", "remove",
             "reweight", "queries", "weights", "note", "warnings")


def load_cfg(a) -> EnvConfig:
    """EnvConfig from --config / --variant (the variant's EnvConfig block only)."""
    import yaml
    if a.variant:
        path = Path(a.variant)
        if not path.exists():
            path = Path("configs/variants") / f"{a.variant}.yaml"
        d = dict(yaml.safe_load(path.read_text()) or {})
        d.pop("train", None)
        d.pop("notes", None)
        return EnvConfig.from_dict(d)
    return EnvConfig.from_yaml(a.config) if a.config else EnvConfig()


def strip_queries(obs):
    """The `none` condition: the same observation with an empty query block.

    The simulator requires at least one mission query, so 'no queries' is expressed here, on the
    observation handed to the policy, rather than by asking the belief for an illegal state.
    """
    return replace(obs, query_emb=np.zeros_like(obs.query_emb),
                   query_w=np.zeros_like(obs.query_w),
                   query_mask=np.zeros_like(obs.query_mask))


def make_controller(cond: str, env, a) -> HintController | None:
    if cond not in ("llm", "scripted"):
        return None
    emb = QueryEmbedder.for_env(env)
    backend = a.backend if cond == "llm" else "scripted"
    agent = HintAgent(backend, embedder=emb, max_queries=env.cfg.tokens.max_queries,
                      default_queries=env.cfg.rayfronts.queries, model=a.model,
                      timeout_s=a.timeout, output_format=a.output_format)
    agent.reset()
    return HintController(agent, emb, every=a.cadence, on_events=not a.no_events,
                          digest=DigestBuilder(max_chars=a.digest_chars), condition=cond)


def run_condition(cond: str, actor, bank: SceneBank, cfg: EnvConfig, tasks, a):
    """One condition over the whole task list; -> (episode rows, edit-log records)."""
    rows: list[dict] = []
    edits: list[dict] = []
    pl = getattr(actor, "policy", None)
    for ep, (key, seed) in enumerate(tasks):
        nr, tm = bank.env_params(key, a.robots_lo, cfg.t_max_s if a.t_spec > 0 else AUTO)
        ecfg = copy.deepcopy(cfg)
        ecfg.robot.n_robots = int(nr)
        ecfg.t_max_s = float(tm)
        if not bool(getattr(pl, "use_local", False)):
            ecfg.tokens.local_size = 0
        if not bool(getattr(pl, "use_robot_bev", False)):
            ecfg.tokens.robot_bev_size = 0
        env = bank.make_env(key, ecfg, seed)
        obs = env.reset(seed)
        if hasattr(actor, "reset"):
            actor.reset(seed)
        ctl = make_controller(cond, env, a)
        if cond == "static":
            obs = env.set_queries(cfg.rayfronts.queries)
        elif ctl is not None:
            obs = ctl.start(env) or obs
        total, n, info = 0.0, 0, {}
        while True:
            fed = strip_queries(obs) if cond == "none" else obs
            obs, r, done, info = env.step(actor.act(fed, env.state))
            total += float(r)
            n += 1
            if ctl is not None:
                obs = ctl.after_step(env, info) or obs
            if done or (a.max_decisions and n >= a.max_decisions):
                break
        m = dict(info["metrics"])
        m.update({"reward": total, "scene": str(key), "area_km2": bank.area(key),
                  "bucket": bank.bucket(key), "disaster": bank.disaster(key),
                  "n_robots": int(nr), "t_max": float(tm), "length": n})
        rows.append(m)
        if ctl is not None:
            for rec in ctl.log:
                edits.append({**rec, "episode": ep, "scene": str(key)})
    return rows, edits


def write_edits(path: Path, records) -> None:
    """The full record (digest included) to JSONL; a flat view without the digest to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(EDIT_COLS), extrasaction="ignore")
        wr.writeheader()
        for r in records:
            wr.writerow({**r, "add": json.dumps(r.get("add", [])),
                         "reweight": json.dumps(r.get("reweight", {})),
                         "remove": "; ".join(r.get("remove", [])),
                         "queries": "; ".join(r.get("queries", [])),
                         "weights": "; ".join(f"{w:.2f}" for w in r.get("weights", [])),
                         "warnings": " | ".join(r.get("warnings", []))})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True,
                    help="baseline name (random|nearest|ray_follower|segment_seeker|oracle) "
                         "or a checkpoint path")
    ap.add_argument("--scenes", default="synthetic:0-200", nargs="+",
                    help="synthetic:A-B or scene-json globs/paths")
    ap.add_argument("--split", default="heldout", choices=("heldout", "train", "all"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--conditions", nargs="+", default=["none", "static", "scripted"],
                    choices=CONDITIONS,
                    help="'llm' shells out to the Claude CLI once per update, so it is opt-in")
    ap.add_argument("--cadence", type=int, default=5, metavar="K",
                    help="decisions between hint updates (0 = only on events)")
    ap.add_argument("--no-events", action="store_true",
                    help="update on the cadence only, never on a find")
    ap.add_argument("--backend", default="claude", choices=("claude", "scripted"),
                    help="backend of the 'llm' condition")
    ap.add_argument("--model", default="opus", help="model for the claude backend")
    ap.add_argument("--output-format", default="json", choices=("json", "text"),
                    help="claude CLI --output-format")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per CLI call")
    ap.add_argument("--digest-chars", type=int, default=2400, help="hard cap on the digest length")
    ap.add_argument("--robots", default="3", help="N or 'auto' (area rule per scene)")
    ap.add_argument("--t-max", default=None, help="seconds or 'auto'; default = the EnvConfig value")
    ap.add_argument("--max-decisions", type=int, default=0, help="0 = run to termination")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--stochastic", action="store_true", help="sample instead of argmax")
    ap.add_argument("--config", default=None, help="EnvConfig yaml")
    ap.add_argument("--variant", default=None, help="configs/variants/<name>.yaml")
    ap.add_argument("--out", default=None, help="CSV path (default runs/llm_hints_<stamp>.csv)")
    a = ap.parse_args(argv)
    a.scenes = " ".join(a.scenes) if isinstance(a.scenes, list) else a.scenes

    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    cfg = load_cfg(a)
    a.robots_lo, _ = parse_robots(a.robots)
    a.t_spec = parse_t_max(a.t_max)
    bank = SceneBank(a.scenes)
    cfg.robot.n_robots = bank.robot_bounds((a.robots_lo, a.robots_lo), a.split)[0]
    if a.t_spec > 0:
        cfg.t_max_s = float(a.t_spec)
    errs = cfg.validate()
    if errs:
        raise SystemExit(f"[llm_hints_eval] invalid EnvConfig: {'; '.join(errs)}")
    tasks = eval_tasks(bank, a.split, a.episodes, a.seed)
    actor = make_actor(a.policy, cfg, device=dev, seed=a.seed, deterministic=not a.stochastic)
    name = getattr(actor, "name", a.policy)
    if isinstance(actor, TorchActor):
        actor.policy.to(dev)

    probe = QueryEmbedder.build(get_embedding_table(
        cfg.rayfronts.queries, dim=cfg.rayfronts.embedding_dim,
        path=cfg.rayfronts.embeddings_path, sim_table_path=cfg.rayfronts.sim_table_path))
    print(f"[llm_hints_eval] policy={name} {len(bank.split(a.split))} {a.split} scene(s), "
          f"{a.episodes} episodes, cadence K={a.cadence}, device={dev}")
    print(f"[llm_hints_eval] {probe.describe()}")

    out = Path(a.out) if a.out else Path("runs") / f"llm_hints_{int(time.time())}.csv"
    ep_out = out.with_suffix(".episodes.csv")
    rows: dict[str, dict] = {}
    all_edits: list[dict] = []
    for cond in a.conditions:
        t0 = time.perf_counter()
        ep, edits = run_condition(cond, actor, bank, cfg, tasks, a)
        res = {c: np.array([r.get(c, np.nan) for r in ep], np.float64) for c in EVAL_COLS}
        rows[f"{name}[{cond}]"] = summarise({**res, **meta_arrays(ep)})
        write_episode_csv(ep_out, ep, policy=f"{name}[{cond}]")
        all_edits += edits
        n_edit = sum(1 for e in edits if e["add"] or e["remove"] or e["reweight"])
        print(f"  {cond}: {time.perf_counter() - t0:.1f}s, {len(edits)} hint turns, "
              f"{n_edit} of them changed the list")

    print(format_table(rows, EVAL_COLS + EVAL_META) + "\n  (mean +- 95% CI of the mean)")
    write_csv(out, rows, EVAL_COLS + EVAL_META,
              extra={"episodes": a.episodes, "split": a.split, "scenes": a.scenes,
                     "cadence": a.cadence, "backend": a.backend, "model": a.model,
                     "embedder": probe.mode, "policy_spec": a.policy})
    edit_path = out.with_suffix(".edits.csv")
    write_edits(edit_path, all_edits)
    print(f"[llm_hints_eval] wrote {out}, {ep_out}, {edit_path} and {edit_path.with_suffix('.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
