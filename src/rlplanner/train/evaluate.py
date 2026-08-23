"""Evaluation of a trained checkpoint or a hand-coded baseline on the held-out scenes.

Episodes run in the `par_env` workers: baselines are executed entirely inside a worker (no
per-decision round trip), a torch actor is driven from here with one eval env per worker so the
policy forward stays batched.
"""
from __future__ import annotations

import copy
import csv
import math
from collections import namedtuple
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..sim import baselines
from ..sim.config import EnvConfig
from ..sim.state import TeamObs
from .obs import ObsBatch
from .par_env import default_workers, make_vec_env, run_episode
from .policy import TokenPolicy
from .scenes import BUCKETS, SceneBank
from .teachers import TEACHERS, make_any

EVAL_COLS = ("frac_found", "finds_auc", "time_to_first", "time_to_half", "time_to_all",
             "coverage_end", "reward", "dist_per_find", "redundancy", "redundancy_frac",
             "intentional_revisits", "link_frac")
EVAL_META = ("area_km2", "n_robots", "t_max")     # per-scene columns of the area-scaling rule
EPISODE_COLS = ("policy", "scene", "disaster", "bucket") + EVAL_META + ("length",) + EVAL_COLS
BASELINES = ("random", "nearest", "ray_follower", "segment_seeker", "oracle")
HEURISTICS = ("random", "nearest", "ray_follower", "segment_seeker")
_BASELINE_ALIAS = {"nearest": "nearest_frontier"}
Z95 = 1.96

Stat = namedtuple("Stat", "mean std ci n")
NO_STAT = Stat(float("nan"), float("nan"), float("nan"), 0)


class TorchActor:
    """Adapts a TokenPolicy to the `baselines.Policy.act(obs, state)` interface."""
    privileged = False

    def __init__(self, policy: TokenPolicy, device="cpu", deterministic: bool = True,
                 name: str = "policy"):
        self.policy = policy.eval()
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.name = name

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            torch.manual_seed(int(seed))

    @torch.no_grad()
    def act(self, obs: TeamObs, state=None) -> np.ndarray:
        ob = ObsBatch.from_team_obs(obs, self.device, with_bev=self.policy.use_bev,
                                    with_local=self.policy.use_local,
                                    with_robot_bev=self.policy.use_robot_bev)
        a, _, _ = self.policy.act(ob, deterministic=self.deterministic)
        return a[0, : obs.n_robots].cpu().numpy()


def load_checkpoint(path: str | Path, device="cpu") -> tuple[TokenPolicy, dict[str, Any]]:
    ck = torch.load(str(path), map_location=device, weights_only=False)
    policy = TokenPolicy.from_config(ck["policy_config"]).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck


def make_actor(spec: str, cfg: EnvConfig, device="cpu", seed: int = 0, deterministic: bool = True):
    """`spec` is a baseline name or a path to a checkpoint."""
    if spec in BASELINES or spec in baselines.POLICIES or spec in TEACHERS:
        p = make_any(_BASELINE_ALIAS.get(spec, spec), queries=cfg.rayfronts.queries, seed=seed)
        p.name = spec
        return p
    policy, _ = load_checkpoint(spec, device)
    return TorchActor(policy, device, deterministic, name=Path(spec).stem)


# ---- evaluation --------------------------------------------------------------------------------
def eval_tasks(bank: SceneBank, split: str, episodes: int, seed: int) -> list[tuple[Any, int]]:
    """Fixed (scene, seed) work list: episode i uses held-out scene i % n and seed + i."""
    keys = bank.split(split)
    return [(keys[i % len(keys)], int(seed) + i) for i in range(int(episodes))]


def _baseline_name(actor) -> str | None:
    if isinstance(actor, str):
        known = actor in BASELINES or actor in baselines.POLICIES or actor in TEACHERS
        return actor if known else None
    if isinstance(actor, baselines.Policy):
        return getattr(actor, "name", None)
    return None


def _pool_for(bank: SceneBank, cfg: EnvConfig, episodes: int, robots: int, split: str, seed: int,
              backend: str, workers: int | None, use_bev: bool, t_max: float | None = None,
              use_local: bool = False, use_robot_bev: bool = False):
    if (not use_local and cfg.tokens.local_size) or (not use_robot_bev
                                                     and cfg.tokens.robot_bev_size):
        cfg = copy.deepcopy(cfg)      # nothing reads them: do not build or ship them
        if not use_local:
            cfg.tokens.local_size = 0
        if not use_robot_bev:
            cfg.tokens.robot_bev_size = 0
    w = max(1, min(int(workers or default_workers(episodes)), int(episodes)))
    if backend == "auto":
        backend = "subproc" if w > 1 and episodes >= 4 else "serial"
    return make_vec_env(backend, bank.spec, cfg, n_envs=w, robots=(robots, robots), split=split,
                        seed=seed, n_workers=w, send_bev=use_bev, region_m=bank.region_m,
                        holdout_frac=bank.holdout_frac, t_max=t_max)


def _actor_rows(pool, actor: TorchActor, tasks, robots: int,
                max_decisions: int | None) -> list[dict[str, float]]:
    """One eval env per worker, stepped in lock-step so the policy forward stays batched."""
    obs = pool.eval_start(tasks, robots, max_decisions)
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        while True:
            ob = ObsBatch.from_vec_obs(obs, actor.device, with_bev=actor.policy.use_bev,
                                       with_local=actor.policy.use_local,
                                       with_robot_bev=actor.policy.use_robot_bev)
            a, _, _ = actor.policy.act(ob, deterministic=actor.deterministic)
            obs, new, alldone = pool.eval_step(a.cpu().numpy())
            rows += new
            if alldone:
                break
    return rows + pool.eval_end()


def evaluate_policy(actor, bank: SceneBank, cfg: EnvConfig, episodes: int = 8,
                    robots: int = 3, seed: int = 10_000, split: str = "heldout",
                    max_decisions: int | None = None, pool=None, backend: str = "auto",
                    workers: int | None = None, t_max: float | None = None,
                    return_rows: bool = False):
    """-> per-episode arrays for EVAL_COLS (and the raw rows if `return_rows`).

    `robots`/`t_max` follow `SceneBank.env_params`: `AUTO` (or any value <= 0) applies the area
    scaling rule per scene, so a run over mixed-size scenes uses a different team and horizon per
    episode; `t_max=None` keeps `cfg.t_max_s`. `pool` reuses an existing vec env's workers.
    """
    tasks = eval_tasks(bank, split, episodes, seed)
    name = _baseline_name(actor)
    own = None
    if pool is None:
        pl = getattr(actor, "policy", None)
        own = pool = _pool_for(bank, cfg, episodes, robots, split, seed, backend, workers,
                               bool(getattr(pl, "use_bev", False)), t_max,
                               bool(getattr(pl, "use_local", False)),
                               bool(getattr(pl, "use_robot_bev", False)))
    try:
        if name is not None:
            rows = pool.run_episodes(name, tasks, robots, max_decisions)
        elif isinstance(actor, TorchActor):
            rows = _actor_rows(pool, actor, tasks, robots, max_decisions)
        else:
            rows = []
            for k, s in tasks:
                nr, tm = bank.env_params(k, robots, cfg.t_max_s if t_max is None else t_max)
                ecfg = copy.deepcopy(cfg)
                ecfg.robot.n_robots = int(nr)
                ecfg.t_max_s = float(tm)
                row = run_episode(bank.make_env(k, ecfg, s), actor, s, max_decisions)
                row.update({"scene": str(k), "area_km2": bank.area(k), "bucket": bank.bucket(k),
                            "disaster": bank.disaster(k), "n_robots": int(nr), "t_max": float(tm)})
                rows.append(row)
    finally:
        if own is not None:
            own.close()
    res = {c: np.array([r.get(c, np.nan) for r in rows], np.float64) for c in EVAL_COLS}
    return (res, rows) if return_rows else res


def meta_arrays(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Per-episode area / robot count / horizon columns (nan when the row predates them)."""
    return {c: np.array([r.get(c, np.nan) for r in rows], np.float64) for c in EVAL_META}


def by_bucket(rows: Sequence[dict[str, Any]], cols: Sequence[str] = EVAL_COLS
              ) -> dict[str, dict[str, Stat]]:
    """Summaries per size bucket, in small -> large order; empty buckets are omitted."""
    out: dict[str, dict[str, Stat]] = {}
    for b in BUCKETS:
        sub = [r for r in rows if r.get("bucket") == b]
        if sub:
            res = {c: np.array([r.get(c, np.nan) for r in sub], np.float64) for c in cols}
            res.update(meta_arrays(sub))
            out[b] = summarise(res)
    return out


def write_episode_csv(path: str | Path, rows: Sequence[dict[str, Any]],
                      policy: str | None = None) -> None:
    """One line per evaluation episode, with the scene's area / robots / horizon."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(EPISODE_COLS), extrasaction="ignore")
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({**{k: r.get(k, "") for k in EPISODE_COLS},
                         "policy": policy if policy is not None else r.get("policy", "")})


def summarise(res: dict[str, np.ndarray]) -> dict[str, Stat]:
    """mean, sample sd and the 95% CI half-width of the mean (normal approximation)."""
    out: dict[str, Stat] = {}
    for k, v in res.items():
        v = np.asarray(v, np.float64)
        n = int(np.isfinite(v).sum())
        if n == 0:
            out[k] = NO_STAT
            continue
        sd = float(np.nanstd(v, ddof=1)) if n > 1 else 0.0
        out[k] = Stat(float(np.nanmean(v)), sd, Z95 * sd / math.sqrt(n) if n > 1 else 0.0, n)
    return out


def evaluate_baselines(names: Sequence[str], bank: SceneBank, cfg: EnvConfig, episodes: int,
                       robots: int, seed: int = 10_000, pool=None, backend: str = "auto",
                       workers: int | None = None, split: str = "heldout",
                       t_max: float | None = None) -> dict[str, dict[str, Stat]]:
    own = None
    if pool is None:
        own = pool = _pool_for(bank, cfg, episodes, robots, split, seed, backend, workers, False,
                               t_max)
    try:
        out: dict[str, dict[str, Stat]] = {}
        for n in names:
            res, rows = evaluate_policy(n, bank, cfg, episodes, robots, seed, split, pool=pool,
                                        t_max=t_max, return_rows=True)
            out[n] = summarise({**res, **meta_arrays(rows)})
        return out
    finally:
        if own is not None:
            own.close()


# ---- reporting ---------------------------------------------------------------------------------
def format_table(rows: dict[str, dict[str, Stat]], cols: Sequence[str] = EVAL_COLS) -> str:
    """mean +- 95% CI half-width of the mean."""
    w = 17
    head = f"{'policy':<14}" + "".join(f"{c:>{w}}" for c in cols)
    out = [head, "-" * len(head)]
    for name, s in rows.items():
        cells = [f"{_stat(s, c).mean:8.2f}+-{_stat(s, c).ci:<6.2f}" for c in cols]
        out.append(f"{name:<14}" + "".join(f"{x:>{w}}" for x in cells))
    return "\n".join(out)


def _stat(s: dict[str, Any], c: str) -> Stat:
    v = s.get(c, NO_STAT)
    return v if isinstance(v, Stat) else Stat(v[0], v[1], float("nan"), 0)


def write_csv(path: str | Path, rows: dict[str, dict[str, Stat]],
              cols: Sequence[str] = EVAL_COLS, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["policy"] + [f"{c}{sfx}" for c in cols for sfx in ("", "_std", "_ci")] + \
             ["n_episodes"] + sorted(extra or {})
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for name, s in rows.items():
            row: dict[str, Any] = {"policy": name}
            n = 0
            for c in cols:
                st = _stat(s, c)
                row[c], row[f"{c}_std"], row[f"{c}_ci"] = st.mean, st.std, st.ci
                n = max(n, st.n)
            row["n_episodes"] = n
            row.update(extra or {})
            wr.writerow(row)


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)


__all__ = ["EVAL_COLS", "EVAL_META", "EPISODE_COLS", "BASELINES", "HEURISTICS", "Stat", "TorchActor",
           "make_actor", "load_checkpoint", "run_episode", "eval_tasks", "evaluate_policy",
           "evaluate_baselines", "summarise", "format_table", "write_csv", "append_csv",
           "by_bucket", "meta_arrays", "write_episode_csv"]
