"""Rollout collection + GAE. Every (env, robot) decision is one MAPPO sample sharing the
team reward, the team advantage and the centralised value of its decision.

Episodes have different lengths (`t_max / decision_dt` = 120..300 decisions under the area rule)
and env slots auto-reset, so a rollout of T decisions holds fragments of several episodes per
slot. GAE handles that per column through `dones`; the horizon end is treated as a terminal (no
bootstrap) rather than a truncation, which is consistent because the robot features carry
`t / t_max`, making the horizon part of the state. Keep `--rollout >= 64`: the advantage of a
decision is truncated at the rollout boundary and bootstrapped from V, so T much below the
~1/(1-gamma*lam) ~ 20-decision GAE window would bias short-horizon credit; T = 64 covers a
quarter to a half of a 120..300-decision episode.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..sim.config import EnvConfig
from ..sim.vec_env import VecEnv, VecObs
from .obs import ObsBatch
from .policy import TokenPolicy


@dataclass
class RolloutBatch:
    """Flattened (T*E) decisions. `actions`/`logp`/`robot_mask` are per (decision, robot)."""
    obs: ObsBatch
    actions: torch.Tensor      # int64   [N, R]
    logp: torch.Tensor         # float32 [N, R]
    values: torch.Tensor       # float32 [N]
    returns: torch.Tensor      # float32 [N]
    advantages: torch.Tensor   # float32 [N]
    robot_mask: torch.Tensor   # bool    [N, R]
    n_steps: int
    n_envs: int
    ref_logp: torch.Tensor | None = None   # float32 [N, R, K] frozen-BC log-probs (KL regulariser)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_decisions(self) -> int:
        return len(self)

    @property
    def n_samples(self) -> int:
        return int(self.robot_mask.sum())


@dataclass
class EpisodeStats:
    """Running per-env episode accumulators; episode summaries flushed on `done`."""
    n_envs: int
    ret: np.ndarray = field(init=False)
    length: np.ndarray = field(init=False)
    finished: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        self.ret = np.zeros(self.n_envs, np.float64)
        self.length = np.zeros(self.n_envs, np.int64)

    def add(self, rewards, dones, infos) -> None:
        self.ret += np.asarray(rewards, np.float64)
        self.length += 1
        for i, d in enumerate(np.asarray(dones)):
            if not d:
                continue
            m = dict(infos[i].get("final_info", {}).get("metrics", infos[i].get("metrics", {})))
            m["reward"] = float(self.ret[i])
            m["length"] = int(self.length[i])
            self.finished.append(m)
            self.ret[i] = 0.0
            self.length[i] = 0

    def drain(self) -> dict[str, float]:
        out: dict[str, float] = {"episodes": float(len(self.finished))}
        if self.finished:
            for k in ("reward", "frac_found", "finds_auc", "time_to_first", "coverage_end",
                      "length"):
                v = [e[k] for e in self.finished if k in e]
                if v:
                    out[k] = float(np.mean(v))
        self.finished = []
        return out


class Collector:
    """Owns the current VecObs and the episode accumulators across updates."""

    def __init__(self, vec, device: torch.device | str = "cpu", gamma: float = 0.99,
                 lam: float = 0.95, obs: VecObs | None = None):
        self.vec = vec
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.obs = vec.reset() if obs is None else obs
        self.stats = EpisodeStats(self.obs.tokens.shape[0])
        self.env_steps = 0

    def set_vec(self, vec, obs: VecObs) -> None:
        self.vec = vec
        self.obs = obs

    @torch.no_grad()
    def rollout(self, policy: TokenPolicy, n_steps: int) -> tuple[RolloutBatch, dict[str, float]]:
        obs_hist: list[ObsBatch] = []
        acts, logps, vals, rews, dones, rmasks = [], [], [], [], [], []
        for _ in range(int(n_steps)):
            ob = ObsBatch.from_vec_obs(self.obs, self.device, with_bev=policy.use_bev,
                                       with_local=policy.use_local,
                                       with_robot_bev=policy.use_robot_bev)
            a, lp, v = policy.act(ob)
            obs_hist.append(ob)
            acts.append(a)
            logps.append(lp)
            vals.append(v)
            rmasks.append(ob.robot_mask)
            self.obs, r, d, infos = self.vec.step(a.cpu().numpy())
            self.stats.add(r, d, infos)
            self.env_steps += ob.n_envs
            rews.append(torch.as_tensor(r, dtype=torch.float32, device=self.device))
            dones.append(torch.as_tensor(d, dtype=torch.float32, device=self.device))
        last_ob = ObsBatch.from_vec_obs(self.obs, self.device, with_bev=policy.use_bev,
                                        with_local=policy.use_local,
                                        with_robot_bev=policy.use_robot_bev)
        _, last_value = policy.forward(last_ob)

        values = torch.stack(vals)                     # [T, E]
        adv = gae(torch.stack(rews), values, torch.stack(dones), last_value, self.gamma, self.lam)
        ret = adv + values
        T, E = values.shape
        batch = RolloutBatch(
            obs=ObsBatch.cat(obs_hist),
            actions=torch.cat(acts).long(),
            logp=torch.cat(logps),
            values=values.reshape(T * E),
            returns=ret.reshape(T * E),
            advantages=adv.reshape(T * E),
            robot_mask=torch.cat(rmasks),
            n_steps=T, n_envs=E)
        stats = self.stats.drain()
        stats["decisions"] = float(T * E)
        stats["env_steps"] = float(self.env_steps)
        return batch, stats


def gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
        last_value: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Generalised advantage estimate over [T, E] team rewards.

    Mixed-length episodes: `dones[t, e]` cuts both the bootstrap and the GAE recursion for that
    column, so slots that reset at different steps never mix returns across episodes; the tail of
    an unfinished episode is bootstrapped from `last_value`.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    nxt = last_value
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
        nxt = values[t]
    return adv


# ---- env pool ----------------------------------------------------------------------------------
class EnvPool:
    """`n_envs` DisasterEnvs drawn from a SceneBank split, rotated onto fresh scenes on demand."""

    def __init__(self, bank, cfg: EnvConfig, n_envs: int, robots: tuple[int, int] = (3, 3),
                 split: str = "train", seed: int = 0, t_max: float | None = None,
                 scene_mix: dict[str, float] | None = None):
        from ..sim.env import DisasterEnv
        self._make = DisasterEnv
        self.bank = bank
        self.cfg = cfg
        self.n_envs = int(n_envs)
        self.robots = (int(robots[0]), int(robots[1]))
        self.t_max = None if t_max is None else float(t_max)
        self.scene_mix = scene_mix
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.envs: list[Any] = []
        self.vec: VecEnv | None = None
        self.obs: VecObs | None = None
        self.rotate()

    def _one(self):
        key = self.bank.sample(self.split, self.rng, self.scene_mix)
        n = (-1 if self.robots[1] <= 0
             else int(self.rng.integers(self.robots[0], self.robots[1] + 1)))
        nr, tm = self.bank.env_params(key, n, self.cfg.t_max_s if self.t_max is None
                                      else self.t_max)
        cfg = copy.deepcopy(self.cfg)
        cfg.robot.n_robots = int(nr)
        cfg.t_max_s = float(tm)
        seed = int(self.rng.integers(0, 2 ** 31 - 1))
        return self.bank.make_env(key, cfg, seed, self._make)

    def rotate(self) -> VecObs:
        """Rebuild every env on a freshly sampled scene / robot count and reset."""
        self.envs = [self._one() for _ in range(self.n_envs)]
        self.vec = VecEnv(self.envs)
        self.obs = self.vec.reset()
        return self.obs

    def close(self) -> None:
        if self.vec is not None:
            self.vec.close()
        self.envs = []


__all__ = ["RolloutBatch", "Collector", "EpisodeStats", "EnvPool", "gae"]
