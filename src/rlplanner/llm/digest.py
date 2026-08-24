"""Bounded text digest of the live simulated-RayFronts belief, for the hint agent to read.

Two rules hold this file together:

  1. **The decode is for the LLM's eyes only.** A ray or a segment reaches the *policy* as its raw
     `feat[D]` embedding; here it is summarised by its nearest class (argmax cosine against
     `EmbeddingTable.class_emb`) purely so an English-speaking model can talk about it. Nothing
     produced here is fed back into the observation.
  2. **It is length-capped.** `max_chars` is enforced on the final string, and the per-interval
     class lists are cut to `top_k` first, so a busy city-scale scene digests to the same order of
     size as an empty one.

The digest is team-level: what the union belief holds this decision, and what is new since the
previous digest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.schema import CLASS_NAMES, HUMAN_CONTAINERS

DIGEST_MAX_CHARS = 2400
TOP_K = 6
TRUNC = "\n[...digest truncated]"


def nearest_class(feats: np.ndarray, emb) -> tuple[np.ndarray, np.ndarray]:
    """`feats [n, D]` -> `(class index [n], cosine [n])` against the table's class embeddings."""
    f = np.asarray(feats, np.float32).reshape(-1, int(emb.D))
    if f.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    cos = (f / np.maximum(n, 1e-12)) @ emb.class_emb.T
    k = cos.argmax(axis=1)
    return k.astype(np.int64), cos[np.arange(f.shape[0]), k].astype(np.float32)


def _class_lines(feats: np.ndarray, emb, top_k: int, weights=None, unit: str = "w") -> list[str]:
    """`class xN (cos 0.xx, <unit> S)` lines, most frequent first, cut to `top_k`."""
    k, cos = nearest_class(feats, emb)
    if k.size == 0:
        return []
    w = np.ones(k.shape[0]) if weights is None else np.asarray(weights, np.float64).reshape(-1)
    out = []
    for c in np.unique(k):
        m = k == c
        out.append((int(m.sum()), CLASS_NAMES[int(c)], float(cos[m].mean()), float(w[m].sum())))
    out.sort(key=lambda r: (-r[0], r[1]))
    return [f"  {name} x{n} (cos {c:.2f}, {unit} {s:.0f})" for n, name, c, s in out[:top_k]]


def scene_context(scene, cfg, n_robots: int | None = None) -> str:
    """The static mission brief: disaster metadata and the search setup, no belief."""
    m = scene.meta
    x0, y0, x1, y1 = scene.region
    nr = int(n_robots if n_robots is not None else cfg.robot.n_robots)
    return "\n".join([
        f"disaster: {m.disaster_type} severity {float(m.severity):.2f} locale {m.locale} "
        f"(preset {m.preset}, seed {m.seed})",
        f"area: {x1 - x0:.0f} x {y1 - y0:.0f} m, {nr} drones at {cfg.robot.flight_alt_m:.0f} m, "
        f"horizon {cfg.t_max_s:.0f} s",
        f"sensor: voxels within {cfg.sensor.depth_limit_m:.0f} m, semantic rays out to "
        f"{cfg.sensor.visual_range_m:.0f} m",
        f"mission queries now: {', '.join(repr(q) for q in cfg.rayfronts.queries)}",
        f"query-token capacity: {cfg.tokens.max_queries}",
    ])


def build_digest(state, since_t: float = 0.0, max_chars: int = DIGEST_MAX_CHARS,
                 top_k: int = TOP_K, metrics: dict | None = None) -> str:
    """One decision's digest of `state` (an `EnvState`), covering everything new since `since_t`."""
    emb = state.emb
    cfg = state.cfg
    names = state.query_names()
    w = getattr(state.rf, "query_w", None)
    wq = (np.ones(len(names), np.float32) if w is None else np.asarray(w, np.float32))
    met = metrics if metrics is not None else (state.metrics or {})
    found = met.get("found_by_container", {}) or {}
    n_found = int(sum(int(v) for v in found.values()))
    cont = ", ".join(f"{c} {int(found.get(c, 0))}" for c in HUMAN_CONTAINERS
                     if int(found.get(c, 0)) > 0) or "none yet"

    rays = list(state.ray_targets or [])
    segs = list(state.segments or [])
    new_rays = [r for r in rays if float(r.t_first) >= since_t]
    new_segs = [s for s in segs if float(s.t_first) >= since_t]

    lines = [
        f"t {state.t:.0f}/{cfg.t_max_s:.0f} s | decision {state.decision_idx} | "
        f"coverage {state.coverage:.2f}",
        "queries: " + " | ".join(f"{n!r} w{float(wq[i]):.2f}" for i, n in enumerate(names)),
        f"casualties found: {n_found} ({cont})",
        f"belief: {len(state.frontier_clusters or [])} frontier clusters, {len(rays)} live rays, "
        f"{len(segs)} segments",
        f"new semantic rays since t={since_t:.0f}: {len(new_rays)}",
    ]
    if new_rays:
        lines += _class_lines(np.stack([r.feat for r in new_rays]), emb, top_k,
                              weights=[r.conf for r in new_rays], unit="conf")
    lines.append(f"new segments since t={since_t:.0f}: {len(new_segs)}")
    if new_segs:
        lines += _class_lines(np.stack([s.feat for s in new_segs]), emb, top_k,
                              weights=[s.n_cells for s in new_segs], unit="cells")
    if rays:
        lines.append("strongest live rays: " + ", ".join(_strongest(rays, emb, top_k)))
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max(0, max_chars - len(TRUNC))] + TRUNC


def _strongest(rays, emb, top_k: int) -> list[str]:
    """The live rays whose peak feature decodes most confidently, newest-first among ties."""
    k, cos = nearest_class(np.stack([r.feat for r in rays]), emb)
    order = np.argsort(-(cos * np.asarray([min(float(r.conf), 5.0) for r in rays], np.float32)))
    return [f"{CLASS_NAMES[int(k[i])]}(cos {cos[i]:.2f}, conf {rays[i].conf:.1f}, "
            f"{rays[i].n_obs} look{'' if rays[i].n_obs == 1 else 's'})" for i in order[:top_k]]


@dataclass
class DigestBuilder:
    """Stateful wrapper: each `build` reports the interval since the previous one."""

    max_chars: int = DIGEST_MAX_CHARS
    top_k: int = TOP_K
    since_t: float = 0.0
    history: list[str] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        self.since_t = 0.0
        self.history.clear()

    def context(self, env) -> str:
        return scene_context(env.scene, env.cfg, n_robots=len(env.state.robots))

    def build(self, env_or_state, metrics: dict | None = None) -> str:
        state = getattr(env_or_state, "state", env_or_state)
        d = build_digest(state, since_t=self.since_t, max_chars=self.max_chars, top_k=self.top_k,
                         metrics=metrics)
        self.since_t = float(state.t)
        self.history.append(d)
        return d


__all__ = ["build_digest", "scene_context", "nearest_class", "DigestBuilder", "DIGEST_MAX_CHARS",
           "TOP_K"]
