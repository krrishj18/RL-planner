"""MAPPO-style PPO update over per-(env, robot) decision samples.

Fine-tuning from a behaviour-cloned actor needs three knobs that plain PPO does not have
(`scripts/train.py --init-from`): the actor can be **frozen** while the value head catches up on
its own return distribution (PPO's first updates otherwise push a good actor around with a critic
that predicts nothing), the actor carries its **own learning rate** so it can be warmed up from 0,
and the loss can carry a **KL to the frozen BC policy** that is annealed to 0 — the student is
allowed to leave the teacher, but not in one update.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .policy import TokenPolicy
from .rollout import RolloutBatch


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    vf_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 4
    n_minibatches: int = 4
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    bc_kl_coef: float = 0.0           # KL(pi || pi_bc); annealed to 0 by the caller
    eps: float = 1e-5
    target_kl: float | None = 0.02    # stop the update once the policy has moved this far

    def to_dict(self) -> dict:
        return asdict(self)


CRITIC_MODULES = ("value_head", "value_attn", "bev_cnn")


def split_params(policy: TokenPolicy) -> tuple[list, list]:
    """(actor, critic) parameter lists. The shared trunk counts as actor: freezing the actor during
    the critic warm-up is meant to hold the whole BC representation still."""
    actor, critic = [], []
    for name, prm in policy.named_parameters():
        (critic if name.split(".")[0] in CRITIC_MODULES else actor).append(prm)
    return actor, critic


class PPO:
    def __init__(self, policy: TokenPolicy, cfg: PPOConfig | None = None,
                 device: torch.device | str = "cpu"):
        self.policy = policy
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        actor, critic = split_params(policy)
        self.opt = torch.optim.Adam([{"params": actor, "lr": self.cfg.lr},
                                     {"params": critic, "lr": self.cfg.lr}],
                                    lr=self.cfg.lr, eps=self.cfg.eps)
        self.bc_kl_coef = float(self.cfg.bc_kl_coef)
        self._frozen = False

    # ---- fine-tuning knobs --------------------------------------------------------------
    @property
    def actor_frozen(self) -> bool:
        return self._frozen

    def freeze_actor(self, on: bool = True) -> None:
        """Critic warm-up: no actor gradient at all, so Adam never touches an actor weight."""
        for prm in self.opt.param_groups[0]["params"]:
            prm.requires_grad_(not on)
            if on:
                prm.grad = None
        self._frozen = bool(on)

    def set_actor_lr(self, lr: float) -> None:
        self.opt.param_groups[0]["lr"] = float(lr)

    @property
    def actor_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    # ---- losses -------------------------------------------------------------------------
    def _losses(self, batch: RolloutBatch, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.cfg
        obs = batch.obs.index(idx)
        rm = batch.robot_mask[idx]
        logp, ent, value, logp_all = self.policy.evaluate_full(obs, batch.actions[idx])
        adv = batch.advantages[idx].unsqueeze(1).expand_as(logp)
        if c.norm_adv:
            adv = _norm(adv, rm)
        ratio = torch.exp(logp - batch.logp[idx])
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - c.clip, 1.0 + c.clip) * adv
        pol_loss = -_mean(torch.min(unclipped, clipped), rm)

        ret, v_old = batch.returns[idx], batch.values[idx]
        v_clip = v_old + (value - v_old).clamp(-c.vf_clip, c.vf_clip)
        val_loss = 0.5 * torch.max((value - ret) ** 2, (v_clip - ret) ** 2).mean()
        entropy = _mean(ent, rm)
        bc_kl = self._bc_kl(batch, idx, logp_all, rm)
        loss = (c.vf_coef * val_loss if self._frozen
                else pol_loss + c.vf_coef * val_loss - c.ent_coef * entropy
                + self.bc_kl_coef * bc_kl)
        return {"policy_loss": pol_loss, "value_loss": val_loss, "entropy": entropy,
                "loss": loss, "bc_kl": bc_kl.detach(),
                "approx_kl": _mean((ratio - 1.0) - torch.log(ratio.clamp_min(1e-8)), rm).detach(),
                "clipfrac": _mean(((ratio - 1.0).abs() > c.clip).float(), rm).detach()}

    def _bc_kl(self, batch: RolloutBatch, idx: torch.Tensor, logp_all: torch.Tensor,
               rm: torch.Tensor) -> torch.Tensor:
        """KL(pi_theta || pi_bc) over the decode-masked token distribution.

        Masked slots hold `finfo.min` before the softmax, so their probability is exactly 0 and the
        clamped difference makes the product 0 rather than nan.
        """
        if batch.ref_logp is None or self.bc_kl_coef == 0.0:
            return torch.zeros((), device=logp_all.device)
        d = (logp_all - batch.ref_logp[idx]).clamp(-50.0, 50.0)
        return _mean((logp_all.exp() * d).sum(-1), rm)

    @torch.no_grad()
    def surrogate(self, batch: RolloutBatch) -> float:
        """Clipped policy loss of the whole batch under the current parameters."""
        idx = torch.arange(len(batch), device=self.device)
        return float(self._losses(batch, idx)["policy_loss"])

    # ---- update -------------------------------------------------------------------------
    @property
    def lr(self) -> float:
        """The critic group's rate: during the actor warm-up group 0 is ramping and is not the
        run's learning rate. `actor_lr` reports that one."""
        return float(self.opt.param_groups[-1]["lr"])

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = float(lr)

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        c = self.cfg
        n = len(batch)
        mb = max(1, n // max(1, c.n_minibatches))
        acc: dict[str, float] = {}
        nb, stop = 0, False
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                idx = perm[s: s + mb]
                if idx.numel() == 0:
                    continue
                out = self._losses(batch, idx)
                self.opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.opt.step()
                nb += 1
                for k, v in out.items():
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
                acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(gn.detach())
                if c.target_kl is not None and float(out["approx_kl"]) > float(c.target_kl):
                    stop = True
                    break
            if stop:
                break
        res = {k: v / max(1, nb) for k, v in acc.items()}
        res["minibatches"] = float(nb)
        res["early_stop"] = float(stop)
        res["lr"] = self.lr
        res["actor_lr"] = self.actor_lr
        res["bc_kl_coef"] = float(self.bc_kl_coef)
        res["explained_variance"] = explained_variance(batch.values, batch.returns)
        return res

    # ---- checkpointing ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "opt": self.opt.state_dict(),
                "policy_config": self.policy.config(), "ppo_config": self.cfg.to_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.policy.load_state_dict(sd["policy"])
        if "opt" in sd:
            try:
                self.opt.load_state_dict(sd["opt"])
            except ValueError:            # a checkpoint from before the actor/critic split
                pass


@torch.no_grad()
def bc_reference(policy: TokenPolicy, batch: RolloutBatch, chunk: int = 256) -> torch.Tensor:
    """[N, R, K] log-probs of a frozen policy on the batch's own states and decode masks."""
    was = policy.training
    policy.eval()
    out = []
    for s in range(0, len(batch), chunk):
        idx = torch.arange(s, min(s + chunk, len(batch)), device=batch.actions.device)
        _, _, _, lp = policy.evaluate_full(batch.obs.index(idx), batch.actions[idx])
        out.append(lp)
    policy.train(was)
    return torch.cat(out, 0)


@torch.no_grad()
def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """1 - Var(returns - V) / Var(returns) of the pre-update critic; 0 = no better than the mean."""
    var = returns.var(unbiased=False)
    if float(var) < 1e-12:
        return float("nan")
    return float(1.0 - (returns - values).var(unbiased=False) / var)


def _mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1)


def _norm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _mean(x, mask)
    var = _mean((x - m) ** 2, mask)
    return (x - m) / (var.sqrt() + 1e-8)


__all__ = ["PPO", "PPOConfig", "explained_variance", "split_params", "bc_reference",
           "CRITIC_MODULES"]
