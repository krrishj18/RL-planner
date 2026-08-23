#!/usr/bin/env python
"""Encode the class prompts and the default queries with a CLIP/SigLIP text tower, PCA-reduce them
and cache the result as JSON for `rlplanner.sim.embeddings`.

    uv sync --extra embed
    uv run python scripts/build_text_embeddings.py --out src/rlplanner/sim/data/text_embeddings.json

The cache stores the PCA basis (mean + components) so a query that is not in the file can still be
embedded at runtime by re-encoding it with the same model. Without the file the simulator
factorizes the hand-authored similarity table instead, so this script is optional.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rlplanner.scene.schema import CLASS_NAMES
from rlplanner.sim.config import DEFAULT_QUERIES
from rlplanner.sim.embeddings import CLASS_PROMPTS, DEFAULT_DIM, project_pca
from rlplanner.sim.similarity_table import Q as KNOWN_QUERIES
from rlplanner.sim.similarity_table import build_sim_table

DEFAULT_MODEL = "ViT-B-16-SigLIP"
DEFAULT_PRETRAINED = "webli"


def encode(model_name: str, pretrained: str, texts: list[str]) -> np.ndarray:
    import open_clip
    import torch

    model = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained,
                                                   return_transform=False)
    model.eval()
    tok = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        v = model.encode_text(tok(texts)).float().cpu().numpy()
    return np.asarray(v, np.float64)


def pca_basis(raw: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, components [dim, raw_dim], explained variance ratio)."""
    mean = raw.mean(axis=0)
    x = raw - mean[None, :]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(dim, vt.shape[0])
    var = s ** 2
    return mean, vt[:k], (var[:k] / max(var.sum(), 1e-12))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--out", default="src/rlplanner/sim/data/text_embeddings.json")
    ap.add_argument("--extra-queries", nargs="*", default=[],
                    help="additional query strings to bake into the cache")
    a = ap.parse_args(argv)

    queries = list(dict.fromkeys(list(DEFAULT_QUERIES) + list(KNOWN_QUERIES) + a.extra_queries))
    prompts = [CLASS_PROMPTS[c] for c in CLASS_NAMES]
    texts = prompts + queries
    t0 = time.perf_counter()
    try:
        raw = encode(a.model, a.pretrained, texts)
    except Exception as exc:                     # noqa: BLE001 - the whole point is to fall back
        print(f"[build_text_embeddings] FAILED to encode with {a.model}/{a.pretrained}: "
              f"{type(exc).__name__}: {exc}")
        print("[build_text_embeddings] no cache written; the simulator will keep factorizing "
              "the hand-authored similarity table (rlplanner.sim.embeddings.factorized_table).")
        return 1
    dt = time.perf_counter() - t0
    mean, comps, evr = pca_basis(raw, a.dim)
    emb = project_pca(raw, mean, comps)
    n_c = len(CLASS_NAMES)
    class_emb, query_emb = emb[:n_c], emb[n_c:]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": a.model, "pretrained": a.pretrained, "dim": int(comps.shape[0]),
        "raw_dim": int(raw.shape[1]), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encode_seconds": round(dt, 2),
        "class_names": list(CLASS_NAMES), "class_prompts": {c: CLASS_PROMPTS[c] for c in CLASS_NAMES},
        "queries": queries,
        "class_emb": class_emb.tolist(), "query_emb": query_emb.tolist(),
        "explained_variance_ratio": [float(v) for v in evr],
        "pca": {"mean": mean.tolist(), "components": comps.tolist()},
    }, indent=1))

    cos = class_emb @ query_emb.T
    hand = build_sim_table(KNOWN_QUERIES)
    hand_idx = [queries.index(q) for q in KNOWN_QUERIES]
    print(f"[build_text_embeddings] {out}  model={a.model}/{a.pretrained} raw_dim={raw.shape[1]} "
          f"-> D={comps.shape[0]}  encode {dt:.1f}s  explained variance "
          f"{float(evr.sum()) * 100:.1f}%")
    w = max(len(c) for c in CLASS_NAMES)
    head = "  ".join(f"{q[:9]:>9s}" for q in KNOWN_QUERIES)
    print(f"{'class':<{w}}  {head}")
    for i, c in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cos[i, j]:9.2f}" for j in hand_idx)
        print(f"{c:<{w}}  {row}")
    print(f"\nmax |cos - hand table| = {np.abs(cos[:, hand_idx] - hand).max():.3f}, "
          f"mean = {np.abs(cos[:, hand_idx] - hand).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
