"""DAgger from a privileged teacher into `TokenPolicy` (the BC stage of BC -> PPO).

Why this exists: PPO from scratch on the open-set observation plateaus at the heuristic level
(0.52-0.56 found, 0.38 AUC after 300 updates) while the privileged teacher gets 0.94 / 0.82. The
gap is not the reward, it is that a 129-way token choice over raw 24-D embeddings gives almost no
gradient before the first find. So the student is taught the teacher's *actions* first and only
then fine-tuned on the reward.

Loop (Ross et al., DAgger): iteration 0 executes the teacher, iteration i > 0 executes the student
with a beta-mixture (beta decaying 0.5 -> 0), and **every** visited state is labelled by the
teacher whether the teacher acted or not — that is the whole point: the student is trained on the
states its own mistakes take it to. Labelling happens inside the env worker (`par_env.dagger_step`),
which is where `env.state` lives, so one round trip per decision carries both the labels and the
transition.

The student imitates *actions*, not returns, so it is irrelevant that the teacher's own episode
reward is terrible (-35 on synthetic: it pays 84 revisits). What is copied is "fly at the thing
worth looking at"; the revisit and redundancy terms are the PPO stage's problem.

Balancing the label types is **off by default**, against the obvious reading of "the labels are
skewed, so balance them". Measured: the teacher's labels on synthetic are 59% segment, 16% ray,
16% frontier and 9% `hold`, and inverse-frequency weighting multiplies `hold` by ~4. `hold` is not
a token type like the others — it is the do-nothing action — and the extra weight is enough to make
it the argmax of an under-trained student everywhere: 345 of 360 greedy decisions, 0.15 coverage,
0.00 found, while the same weights *sampled* still pick rays 50% of the time. The flag stays
(`--balance`) because the skew is real; the default does not, because the cure was worse.

Memory. The dense actor input is what costs: an ego-centric 12 x 64 x 64 crop is 197 kB per robot
per decision in float32, so 200k labelled robot-decisions is ~40 GB with the crop against ~0.5 GB
for the tokens alone. The buffer therefore stores float16 and is capped in **bytes** as well as in
samples; on overflow it is thinned uniformly, so it stays a subsample of the aggregate rather than
the last iteration only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Sequence

import numpy as np
import torch

from ..sim.state import TOKEN_TYPE_NAMES
from .obs import ObsBatch
from .policy import TokenPolicy, fill_invalid

_HALF_FIELDS = ("tokens", "robot_feat", "query_emb", "query_w", "local", "peer_tokens",
                "robot_bev", "bev")


@dataclass
class DaggerConfig:
    iters: int = 4
    steps: int = 64                # decisions per env slot per iteration
    beta0: float = 0.5             # mixture weight of the teacher at iteration 1
    epochs: int = 4                # passes over the buffer per iteration
    batch: int = 32                # decisions per minibatch (x n_robots samples)
    lr: float = 3e-4
    label_smoothing: float = 0.05
    ent_coef: float = 0.001
    balance: bool = False          # inverse-frequency weight over the label's token type
    max_samples: int = 2_000_000   # robot-decisions
    max_gb: float = 8.0
    grad_clip: float = 0.5
    weight_decay: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def beta_schedule(i: int, iters: int, beta0: float = 0.5) -> float:
    """1.0 at iteration 0 (pure teacher), then linear beta0 -> 0 over the remaining iterations."""
    i, iters = int(i), int(iters)
    if i <= 0:
        return 1.0
    if iters <= 2:
        return 0.0
    return float(beta0) * (iters - 1 - i) / (iters - 2)


# ---- dataset -----------------------------------------------------------------------------------
class LabelBuffer:
    """Aggregated (state, teacher label) pairs, float16, capped in samples and in bytes.

    Chunk-per-collection rather than one growing tensor: concatenating an 8 GB buffer every
    iteration would peak at twice that. Sampling maps global row indices onto the chunks.
    """

    def __init__(self, max_samples: int = 2_000_000, max_gb: float = 8.0,
                 rng: np.random.Generator | None = None):
        self.max_samples = int(max_samples)
        self.max_bytes = int(float(max_gb) * 1e9)
        self.rng = rng or np.random.default_rng(0)
        self.chunks: list[tuple[ObsBatch, torch.Tensor, torch.Tensor]] = []
        self.seen_decisions = 0
        self.seen_labels = 0

    # -- size ------------------------------------------------------------------------------
    def __len__(self) -> int:
        return sum(int(c[1].shape[0]) for c in self.chunks)

    @property
    def n_labels(self) -> int:
        return sum(int(c[2].sum()) for c in self.chunks)

    @property
    def nbytes(self) -> int:
        tot = 0
        for ob, lab, val in self.chunks:
            tot += sum(getattr(ob, f.name).numel() * getattr(ob, f.name).element_size()
                       for f in fields(ObsBatch))
            tot += lab.numel() * lab.element_size() + val.numel() * val.element_size()
        return int(tot)

    @property
    def bytes_per_decision(self) -> float:
        n = len(self)
        return self.nbytes / n if n else 0.0

    # -- io --------------------------------------------------------------------------------
    def add(self, obs: ObsBatch, labels: torch.Tensor, valid: torch.Tensor) -> None:
        self.seen_decisions += int(labels.shape[0])
        self.seen_labels += int(valid.sum())
        self.chunks.append((_half(obs.to("cpu")), labels.cpu().to(torch.int16),
                            valid.cpu()))
        self._trim()

    def _trim(self) -> None:
        """Thin every chunk by the same factor until the buffer fits both caps."""
        while self.chunks:
            n = len(self)
            r_bytes = self.max_bytes / max(self.nbytes, 1)
            r_samp = self.max_samples / max(self.n_labels, 1)
            keep = min(1.0, r_bytes, r_samp)
            if keep >= 1.0:
                return
            keep = max(keep, 0.05)
            out = []
            for ob, lab, val in self.chunks:
                m = int(lab.shape[0])
                k = max(1, int(round(m * keep)))
                idx = torch.as_tensor(np.sort(self.rng.choice(m, size=k, replace=False)))
                out.append((ob.index(idx), lab[idx], val[idx]))
            self.chunks = out
            if len(self) >= n:            # no progress possible
                return

    def offsets(self) -> np.ndarray:
        return np.cumsum([0] + [int(c[1].shape[0]) for c in self.chunks])

    def gather(self, idx: np.ndarray, device=None) -> tuple[ObsBatch, torch.Tensor, torch.Tensor]:
        off = self.offsets()
        which = np.searchsorted(off, idx, side="right") - 1
        obs, labs, vals = [], [], []
        for c in np.unique(which):
            loc = torch.as_tensor(np.sort(idx[which == c] - off[c]))
            ob, lab, val = self.chunks[int(c)]
            obs.append(ob.index(loc))
            labs.append(lab[loc])
            vals.append(val[loc])
        ob = _float(ObsBatch.cat(obs))
        if device is not None:
            ob = ob.to(device)
        return (ob, torch.cat(labs).long().to(ob.device), torch.cat(vals).to(ob.device))

    def mode_slot(self) -> tuple[int, float]:
        """Most-labelled slot index and its share — the accuracy a constant predictor would get.

        Slot order is fixed (hold, newest frontier, newest ray, ...), so a student that has only
        learned "take the newest ray" scores this much: it is the floor an agreement number has to
        be read against.
        """
        n = len(TOKEN_TYPE_NAMES) * 64
        hist = np.zeros(max(n, 1), np.int64)
        tot = 0
        for _, lab, val in self.chunks:
            v = lab[val].numpy()
            if v.size:
                hist[: hist.size] += np.bincount(np.clip(v, 0, hist.size - 1),
                                                 minlength=hist.size)[: hist.size]
                tot += int(v.size)
        if tot == 0:
            return (-1, float("nan"))
        k = int(hist.argmax())
        return (k, float(hist[k]) / tot)

    def label_type_counts(self, n_types: int = len(TOKEN_TYPE_NAMES)) -> np.ndarray:
        out = np.zeros(n_types, np.int64)
        for ob, lab, val in self.chunks:
            tt = ob.token_type.gather(-1, lab.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
            for t, c in zip(*np.unique(tt[val].numpy(), return_counts=True)):
                if 0 <= int(t) < n_types:
                    out[int(t)] += int(c)
        return out


def _half(ob: ObsBatch) -> ObsBatch:
    d = {f.name: getattr(ob, f.name) for f in fields(ObsBatch)}
    for k in _HALF_FIELDS:
        if d[k].is_floating_point():
            d[k] = d[k].to(torch.float16)
    d["token_id"] = d["token_id"].to(torch.int32)
    d["token_type"] = d["token_type"].to(torch.int8)
    return ObsBatch(**d)


def _float(ob: ObsBatch) -> ObsBatch:
    d = {f.name: getattr(ob, f.name) for f in fields(ObsBatch)}
    for k in _HALF_FIELDS:
        if d[k].dtype == torch.float16:
            d[k] = d[k].to(torch.float32)
    d["token_id"] = d["token_id"].to(torch.int64)
    d["token_type"] = d["token_type"].to(torch.int64)
    return ObsBatch(**d)


# ---- loss --------------------------------------------------------------------------------------
def bc_losses(policy: TokenPolicy, obs: ObsBatch, labels: torch.Tensor, valid: torch.Tensor,
              label_smoothing: float = 0.05, ent_coef: float = 0.0,
              type_w: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Cross-entropy over the decode-masked token logits.

    The mask is the policy's own (`valid_masks`), so under `sequential_decode` the label of robot r
    is scored against exactly the set robot r would have decoded over given the teacher's earlier
    picks. A label the mask excludes (the teacher's arbitration is not the decode's) is dropped
    from the loss rather than scored at -inf.
    """
    logits = policy.actor_logits(obs)
    vm = policy.valid_masks(obs, labels)
    a = labels.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)
    ok = valid & vm.gather(-1, a).squeeze(-1)
    lg = fill_invalid(logits, ~vm)
    logp = torch.log_softmax(lg, dim=-1)
    nll = -logp.gather(-1, a).squeeze(-1)
    n_valid = vm.sum(-1).clamp_min(1)
    uni = -(logp * vm).sum(-1) / n_valid                  # smoothing over the selectable set only
    eps = float(label_smoothing)
    ce = (1.0 - eps) * nll + eps * uni
    ent = -(logp.exp() * logp * vm).sum(-1)
    w = ok.float()
    if type_w is not None:
        tt = obs.token_type.gather(-1, a).squeeze(-1).clamp(0, type_w.numel() - 1)
        w = w * type_w[tt]
    z = w.sum().clamp_min(1e-8)
    acc = ((logp.argmax(-1) == labels) & ok).float().sum() / ok.float().sum().clamp_min(1)
    return {"ce": (ce * w).sum() / z, "entropy": (ent * w).sum() / z,
            "loss": ((ce - ent_coef * ent) * w).sum() / z,
            "accuracy": acc.detach(), "n": ok.float().sum().detach(),
            "dropped": (valid & ~ok).float().sum().detach()}


def type_weights(counts: np.ndarray, device=None) -> torch.Tensor:
    """Inverse-frequency weights over the token types the teacher actually labels, mean 1."""
    c = np.asarray(counts, np.float64)
    w = np.where(c > 0, 1.0 / np.maximum(c, 1.0), 0.0)
    tot = (w * c).sum()
    w = w * (c.sum() / tot) if tot > 0 else np.ones_like(w)
    return torch.as_tensor(w, dtype=torch.float32, device=device)


# ---- training ----------------------------------------------------------------------------------
class Imitator:
    """Owns the student, its optimiser and the aggregated dataset."""

    def __init__(self, policy: TokenPolicy, cfg: DaggerConfig | None = None, device="cpu",
                 seed: int = 0):
        self.policy = policy
        self.cfg = cfg or DaggerConfig()
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.buffer = LabelBuffer(self.cfg.max_samples, self.cfg.max_gb, self.rng)
        self.opt = torch.optim.AdamW(policy.parameters(), lr=self.cfg.lr,
                                     weight_decay=self.cfg.weight_decay)

    def train_epochs(self, epochs: int | None = None) -> dict[str, float]:
        c = self.cfg
        n = len(self.buffer)
        if n == 0:
            return {}
        tw = (type_weights(self.buffer.label_type_counts(), self.device) if c.balance else None)
        self.policy.train()
        acc: dict[str, float] = {}
        nb = 0
        for _ in range(int(epochs if epochs is not None else c.epochs)):
            perm = self.rng.permutation(n)
            for s in range(0, n, c.batch):
                idx = perm[s: s + c.batch]
                if idx.size == 0:
                    continue
                obs, lab, val = self.buffer.gather(idx, self.device)
                out = bc_losses(self.policy, obs, lab, val, c.label_smoothing, c.ent_coef, tw)
                if float(out["n"]) == 0.0:
                    continue
                self.opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c.grad_clip)
                self.opt.step()
                nb += 1
                for k, v in out.items():
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
                acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(gn)
        res = {k: v / max(1, nb) for k, v in acc.items()}
        res["minibatches"] = float(nb)
        return res

    # -- checkpointing ---------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "opt": self.opt.state_dict(),
                "policy_config": self.policy.config(), "bc_config": self.cfg.to_dict()}


# ---- collection --------------------------------------------------------------------------------
@torch.no_grad()
def collect(vec, policy: TokenPolicy, buffer: LabelBuffer, steps: int, beta: float,
            teacher: str, radius: float, device, rng: np.random.Generator, stats=None,
            flush_every: int = 32) -> dict[str, float]:
    """`steps` decisions per env slot; every visited state is labelled by the teacher.

    `beta` is drawn per (env slot, decision): with probability beta the slot executes the teacher's
    action, otherwise the student's sample. The label is the teacher's action either way.

    Steps are flushed into the buffer in blocks: one chunk per decision would leave the buffer with
    thousands of 16-row pieces and turn every minibatch into as many tensor slices.
    """
    E = vec.n_envs
    n_teacher = 0
    pend: list[tuple[ObsBatch, torch.Tensor, torch.Tensor]] = []

    def flush() -> None:
        if pend:
            buffer.add(ObsBatch.cat([x[0] for x in pend]),
                       torch.cat([x[1] for x in pend]), torch.cat([x[2] for x in pend]))
            pend.clear()

    for t in range(int(steps)):
        ob = ObsBatch.from_vec_obs(vec.obs, device, with_bev=False, with_local=policy.use_local,
                                   with_robot_bev=policy.use_robot_bev)
        if beta >= 1.0:
            a = np.zeros((E, ob.n_robots), np.int64)
            use = np.ones(E, np.bool_)
        else:
            a = policy.act_tokens(ob, deterministic=False).cpu().numpy()
            use = rng.random(E) < float(beta)
        n_teacher += int(use.sum())
        _, r, d, infos, lab, val, _ = vec.dagger_step(a, use, teacher, radius)
        if stats is not None:
            stats.add(r, d, infos)
        pend.append((ob.to("cpu"), torch.as_tensor(lab), torch.as_tensor(val)
                     & ob.robot_mask.cpu()))
        if len(pend) >= int(flush_every):
            flush()
    flush()
    return {"teacher_frac": n_teacher / max(1, E * int(steps)), "decisions": float(E * int(steps))}


@torch.no_grad()
def agreement(policy: TokenPolicy, states: LabelBuffer, device, batch: int = 32) -> float:
    """Fraction of held-out (state, teacher label) pairs the student's argmax reproduces."""
    n = len(states)
    if n == 0:
        return float("nan")
    was = policy.training
    policy.eval()
    hits, tot = 0.0, 0.0
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        obs, lab, val = states.gather(idx, device)
        vm = policy.valid_masks(obs, lab)
        a = fill_invalid(policy.actor_logits(obs), ~vm).argmax(-1)
        ok = val & vm.gather(-1, lab.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        hits += float(((a == lab) & ok).sum())
        tot += float(ok.sum())
    policy.train(was)
    return hits / max(tot, 1.0)


@torch.no_grad()
def teacher_states(vec, buffer: LabelBuffer, steps: int, teacher: str, radius: float,
                   device, policy: TokenPolicy) -> None:
    """Fill `buffer` with pure-teacher states (the held-out agreement probe)."""
    collect(vec, policy, buffer, steps, 1.0, teacher, radius, device,
            np.random.default_rng(0), flush_every=max(1, int(steps)))


def label_histogram(counts: Sequence[int]) -> str:
    tot = max(1, int(sum(counts)))
    return " ".join(f"{n}={c / tot:.2f}" for n, c in zip(TOKEN_TYPE_NAMES, counts))


__all__ = ["DaggerConfig", "Imitator", "LabelBuffer", "beta_schedule", "bc_losses",
           "type_weights", "collect", "agreement", "teacher_states", "label_histogram"]
