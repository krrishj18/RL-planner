"""Episode metrics (CONTRACTS.md 8), accumulated incrementally at decision boundaries."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

REDUNDANCY_M = 20.0


@dataclass
class EpisodeMetrics:
    t_max: float
    n_casualties: int
    decision_dt: float
    n_by_container: dict[str, int] = field(default_factory=dict)
    found_by_container: dict[str, int] = field(default_factory=dict)
    time_to_first: float = -1.0
    time_to_half: float = -1.0
    time_to_all: float = -1.0
    finds_auc: float = 0.0
    frac_found: float = 0.0
    n_found: int = 0
    coverage_end: float = 0.0
    dist_total: float = 0.0
    redundancy: int = 0
    n_decisions: int = 0
    finalised: bool = False
    # decentralised execution (DESIGN_VARIANTS.md D): totals over the episode, summed over robots
    redundant_cells: int = 0
    observed_cells: int = 0
    redundancy_frac_sum: float = 0.0
    redundancy_refunds: int = 0
    intentional_revisits: int = 0
    revisit_refunds: int = 0
    revisit_penalties: float = 0.0
    visits: int = 0
    link_frac: float = 1.0
    comms_range_m: float = float("inf")

    def update(self, t: float, n_found: int, coverage: float, dist_total: float,
               target_xy: np.ndarray, found_by_container: dict[str, int] | None = None,
               extra: dict[str, Any] | None = None) -> None:
        self.n_decisions += 1
        if extra:
            for k in ("redundant_cells", "observed_cells", "redundancy_refunds",
                      "intentional_revisits", "revisit_refunds", "visits"):
                if k in extra:
                    setattr(self, k, getattr(self, k) + int(extra[k]))
            for k, f in (("revisit_penalties", "revisit_penalties"),
                         ("redundancy_frac", "redundancy_frac_sum")):
                if k in extra:
                    setattr(self, f, getattr(self, f) + float(extra[k]))
            for k in ("link_frac", "comms_range_m"):
                if k in extra:
                    setattr(self, k, float(extra[k]))
        self.n_found = int(n_found)
        self.coverage_end = float(coverage)
        self.dist_total = float(dist_total)
        if found_by_container is not None:
            self.found_by_container = dict(found_by_container)
        self.frac_found = n_found / self.n_casualties if self.n_casualties else 1.0
        self.finds_auc += self.frac_found * (self.decision_dt / max(self.t_max, 1e-9))
        if self.time_to_first < 0 and n_found >= 1:
            self.time_to_first = t
        if self.time_to_half < 0 and self.n_casualties and n_found >= (self.n_casualties + 1) // 2:
            self.time_to_half = t
        if self.time_to_all < 0 and n_found >= self.n_casualties:
            self.time_to_all = t
        good = np.asarray([xy for xy in target_xy if xy is not None and np.all(np.isfinite(xy))],
                          dtype=np.float64).reshape(-1, 2)
        if good.shape[0] >= 2:
            d = np.hypot(good[:, None, 0] - good[None, :, 0], good[:, None, 1] - good[None, :, 1])
            np.fill_diagonal(d, np.inf)
            if (d <= REDUNDANCY_M).any():
                self.redundancy += 1

    def finalise(self, t: float) -> None:
        """Close the episode. The area under fraction-found is over the whole horizon, so an
        episode that ends early (everything found) is credited with the time it saved instead of
        being scored as if it had stopped finding people."""
        if self.finalised:
            return
        self.finalised = True
        self.finds_auc += self.frac_found * max(0.0, self.t_max - t) / max(self.t_max, 1e-9)
        for k in ("time_to_first", "time_to_half", "time_to_all"):
            if getattr(self, k) < 0:
                setattr(self, k, float(self.t_max))

    def to_dict(self) -> dict[str, Any]:
        d = max(1, self.n_decisions)
        return {
            "time_to_first": float(self.time_to_first if self.time_to_first >= 0 else self.t_max),
            "time_to_half": float(self.time_to_half if self.time_to_half >= 0 else self.t_max),
            "time_to_all": float(self.time_to_all if self.time_to_all >= 0 else self.t_max),
            "finds_auc": float(self.finds_auc),
            "frac_found": float(self.frac_found),
            "n_found": int(self.n_found),
            "dist_per_find": float(self.dist_total / max(1, self.n_found)),
            "dist_total": float(self.dist_total),
            "coverage_end": float(self.coverage_end),
            "redundancy": float(self.redundancy / d),
            "n_decisions": int(self.n_decisions),
            "found_by_container": dict(self.found_by_container),
            "n_by_container": dict(self.n_by_container),
            "redundant_cells": int(self.redundant_cells),
            "redundant_per_decision": float(self.redundant_cells / d),
            "redundancy_frac": float(self.redundancy_frac_sum / d),
            "observed_cells": int(self.observed_cells),
            "redundancy_refunds": int(self.redundancy_refunds),
            "intentional_revisits": int(self.intentional_revisits),
            "revisit_refunds": int(self.revisit_refunds),
            "revisit_penalties": float(self.revisit_penalties),
            "visits": int(self.visits),
            "link_frac": float(self.link_frac),
            "comms_range_m": float(self.comms_range_m),
        }


METRIC_KEYS = tuple(EpisodeMetrics(600.0, 1, 5.0).to_dict().keys())

__all__ = ["EpisodeMetrics", "METRIC_KEYS", "REDUNDANCY_M"]
