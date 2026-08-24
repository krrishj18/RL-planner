"""Training-side query churn — the same movement an LLM hint causes, with no LLM in the loop.

A policy that only ever meets `("person lying on the ground", "person")` learns those two columns,
not the attention from items to queries. `QueryScheduleSampler` samples an initial subset of the
plausible queries per episode and edits it at decision boundaries, so the query block is a moving
input during training and a hint at evaluation time is not a distribution shift.

It is wired as `EnvConfig.queries_dynamic` and is **off by default**: with `enabled = False` the
env never calls it and every existing run reproduces bit-for-bit. Everything here is a function of
the `np.random.Generator` it is handed, so a seed fixes the whole schedule.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..sim.similarity_table import Q as KNOWN_QUERIES

NOISE_TAG = "~"     # a noised draw is registered as "<query>~<hash>" so the name stays traceable
BANK_CAP = 20_000   # process-lifetime guard: past this many bank entries, draws stop being noised


def default_pool(emb) -> tuple[str, ...]:
    """The hand-authored query bank, or whatever the live table knows if it is a different one.

    Names carrying `NOISE_TAG` are excluded: they are this sampler's own earlier draws, not
    vocabulary, and letting them back in would compound the noise over a run.
    """
    known = tuple(q for q in KNOWN_QUERIES if q in emb.bank)
    return known or tuple(sorted(n for n in emb.bank if NOISE_TAG not in n))


@dataclass
class QueryScheduleSampler:
    """Per-episode initial subset + per-interval add/remove/reweight draws over a query pool."""

    pool: tuple[str, ...]
    emb: object                     # sim.embeddings.EmbeddingTable (its bank receives noised draws)
    every: int = 10
    p_edit: float = 0.5
    n_init: tuple[int, int] = (1, 3)
    w_range: tuple[float, float] = (0.3, 1.0)
    noise_std: float = 0.0
    max_queries: int = 8

    def __post_init__(self) -> None:
        self.pool = tuple(dict.fromkeys(str(q) for q in self.pool))
        if not self.pool:
            raise ValueError("QueryScheduleSampler: empty query pool")
        lo, hi = int(self.n_init[0]), int(self.n_init[1])
        if not (1 <= lo <= hi):
            raise ValueError(f"QueryScheduleSampler: n_init must satisfy 1 <= lo <= hi, got {(lo, hi)}")
        self.n_init = (lo, min(hi, self.max_queries, len(self.pool)))
        self._noised = 0

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, qd, emb, cfg=None) -> "QueryScheduleSampler":
        """Build from `EnvConfig.queries_dynamic`; an empty pool means the table's whole bank."""
        cap = int(qd.max_queries or (cfg.tokens.max_queries if cfg is not None else 8))
        pool = tuple(qd.pool) or default_pool(emb)
        return cls(pool=pool, emb=emb, every=int(qd.every), p_edit=float(qd.p_edit),
                   n_init=(int(qd.n_init_min), int(qd.n_init_max)),
                   w_range=(float(qd.w_min), float(qd.w_max)), noise_std=float(qd.noise_std),
                   max_queries=cap)

    # ---- draws --------------------------------------------------------------------------------
    def _weight(self, rng: np.random.Generator) -> float:
        lo, hi = self.w_range
        return float(rng.uniform(lo, hi))

    def _draw(self, rng: np.random.Generator, exclude: set[str]) -> str | None:
        """One pool query not already active, optionally re-registered with a noised embedding."""
        free = [q for q in self.pool if q not in exclude]
        if not free:
            return None
        q = free[int(rng.integers(len(free)))]
        if self.noise_std <= 0.0 or len(self.emb.bank) >= BANK_CAP:
            return q
        base = np.asarray(self.emb.bank[q], np.float32)
        v = base + rng.normal(0.0, self.noise_std, size=base.shape).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        v = v.astype(np.float32)
        # the name is a hash of the vector, so two envs in one process (sharing the cached
        # EmbeddingTable) can never give one name two different meanings
        name = f"{q}{NOISE_TAG}{hashlib.blake2s(v.tobytes(), digest_size=4).hexdigest()}"
        self.emb.bank.setdefault(name, v)               # what makes set_queries accept the name
        self._noised += 1
        return name

    def initial(self, rng: np.random.Generator) -> tuple[list[str], list[float]]:
        lo, hi = self.n_init
        n = int(rng.integers(lo, hi + 1))
        names: list[str] = []
        for _ in range(n):
            q = self._draw(rng, set(names))
            if q is not None:
                names.append(q)
        if not names:                                   # the pool cannot be empty, but be explicit
            names = [self.pool[0]]
        return names, [self._weight(rng) for _ in names]

    def edit(self, active, weights, decision_idx: int, rng: np.random.Generator):
        """-> `(names, weights)` at an edit boundary, or `(None, None)` when nothing changes."""
        if self.every <= 0 or decision_idx <= 0 or decision_idx % self.every:
            return None, None
        if rng.random() >= self.p_edit:
            return None, None
        names = [str(q) for q in active]
        w = [float(x) for x in np.asarray(weights, np.float64).reshape(-1)[: len(names)]]
        w += [1.0] * (len(names) - len(w))
        kind = int(rng.integers(3))
        if kind == 0 and len(names) < self.max_queries:              # add
            q = self._draw(rng, set(names))
            if q is not None:
                names.append(q)
                w.append(self._weight(rng))
        elif kind == 1 and len(names) > 1:                           # remove
            k = int(rng.integers(len(names)))
            names.pop(k)
            w.pop(k)
        else:                                                        # reweight
            w[int(rng.integers(len(names)))] = self._weight(rng)
        return names, w


__all__ = ["QueryScheduleSampler", "default_pool", "NOISE_TAG", "BANK_CAP"]
