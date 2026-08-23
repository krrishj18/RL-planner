"""Semantic embeddings for the RayFronts emulation.

Every observed cell and every semantic ray carries a `D`-dimensional unit feature vector derived
from its ground-truth class, the way RayFronts stores a CLIP/SigLIP feature per voxel and per ray.
Query similarities are cosines against text-query embeddings, i.e. a *derived view* of the stored
features rather than the stored quantity.

Sources, in order:
  (a) a cached JSON written by `scripts/build_text_embeddings.py` (a real text tower, PCA-reduced
      to D). The cache keeps the PCA basis so a query the cache does not know can still be
      embedded at runtime if the encoder is installed;
  (b) a factorization of the hand-authored `similarity_table.SIM_TABLE`: the joint
      (classes + queries) Gram matrix is eigendecomposed with negative eigenvalues clipped to get
      unit vectors, then refined by projected gradient until
      `cos(class_i, query_j) == table[i, j]`: exact (< 1e-6) for `embedding_dim >= 12`, which is
      where the shipped 15x11 table stops being compressible; smaller dims are rejected. No model
      download, so tests and the default configuration never touch the network.

The shipped cache `data/text_embeddings_siglip_vitb16.json` (open_clip ViT-B-16-SigLIP/webli, PCA
to 24 dims) is **opt-in** via `RayFrontsConfig.embeddings_path`: after mean-centring, its cosines
compress into roughly [-0.5, 0.8] and invert a few pairs the planner depends on (a toppled car
outscores a standing person on "person lying on the ground", nothing ever crosses
the hand table's), so it is offered for inspection rather than as the default belief.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..scene.schema import CLASS_NAMES, N_CLASSES
from .similarity_table import Q as KNOWN_QUERIES
from .similarity_table import build_sim_table, load_sim_table

DEFAULT_DIM = 24
FEAT_PC_DIM = 8          # principal components the BEV/local rasters project features onto
FACTOR_TOL = 0.05        # max |cos - table| a factorized table may miss by (D >= 12 fits exactly)
DATA_PATH = Path(__file__).resolve().parent / "data" / "text_embeddings.json"

# Natural-language prompt per raster class, used by scripts/build_text_embeddings.py.
CLASS_PROMPTS: dict[str, str] = {
    "ground":             "bare ground and patchy grass",
    "road":               "an asphalt road seen from above",
    "sidewalk":           "a concrete sidewalk along a street",
    "park":               "a green city park with grass and trees",
    "building_intact":    "an intact building with an undamaged roof",
    "building_damaged":   "a damaged building with a partly collapsed facade",
    "building_destroyed": "a collapsed building reduced to rubble",
    "debris":             "a pile of rubble and construction debris",
    "vehicle_intact":     "a car parked upright on the street",
    "vehicle_toppled":    "an overturned car lying on its side",
    "bus_stop":           "a bus stop shelter on the sidewalk",
    "tree":               "the canopy of a tree",
    "street_furniture":   "a street lamp post, a bench and a trash can",
    "human_standing":     "a person standing on the street",
    "human_prone":        "a person lying on the ground",
}
assert set(CLASS_PROMPTS) == set(CLASS_NAMES)


# ---- table ---------------------------------------------------------------------------------
@dataclass
class EmbeddingTable:
    """Unit-norm class and query embeddings plus every query the source knows about."""

    class_emb: np.ndarray                 # float32 [N_CLASSES, D]
    query_emb: np.ndarray                 # float32 [Q, D]
    names: tuple[str, ...]                # query names, aligned with query_emb
    source: str                           # "factorized" | "cache:<path>"
    bank: dict[str, np.ndarray] = field(default_factory=dict)   # name -> float32 [D]
    meta: dict[str, Any] = field(default_factory=dict)
    _pc: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.class_emb = _unit(np.ascontiguousarray(self.class_emb, np.float32))
        self.query_emb = _unit(np.ascontiguousarray(self.query_emb, np.float32))
        self.names = tuple(self.names)
        if self.class_emb.shape[0] != N_CLASSES:
            raise ValueError(f"EmbeddingTable: class_emb has {self.class_emb.shape[0]} rows, "
                             f"expected {N_CLASSES}")
        if self.query_emb.shape[0] != len(self.names):
            raise ValueError(f"EmbeddingTable: {self.query_emb.shape[0]} query vectors for "
                             f"{len(self.names)} names")
        if self.class_emb.shape[1] != self.query_emb.shape[1]:
            raise ValueError(f"EmbeddingTable: class dim {self.class_emb.shape[1]} != query dim "
                             f"{self.query_emb.shape[1]}")
        for n, v in zip(self.names, self.query_emb):
            self.bank.setdefault(n, v)

    @property
    def D(self) -> int:
        return int(self.class_emb.shape[1])

    def similarity(self, feat: np.ndarray, query_emb: np.ndarray | None = None) -> np.ndarray:
        """Cosine of `feat [..., D]` against `query_emb [Q, D]` (default: this table's queries).

        Returns `[..., Q]` float32. Zero-length features give zeros.
        """
        f = np.asarray(feat, np.float32)
        qe = self.query_emb if query_emb is None else _unit(np.asarray(query_emb, np.float32))
        if f.shape[-1] != qe.shape[-1]:
            raise ValueError(f"similarity: feature dim {f.shape[-1]} != query dim {qe.shape[-1]}")
        n = np.linalg.norm(f, axis=-1, keepdims=True)
        return ((f / np.maximum(n, 1e-12)) @ qe.T).astype(np.float32)

    def sim_table(self, clip: bool = True) -> np.ndarray:
        """[N_CLASSES, Q] class x query cosines (the emulated `SIM_TABLE`)."""
        t = (self.class_emb @ self.query_emb.T).astype(np.float32)
        return np.clip(t, 0.0, 1.0) if clip else t

    def knows(self, name: str) -> bool:
        return name in self.bank

    # -- fixed feature basis --------------------------------------------------------------------
    def pc_basis(self, k: int = FEAT_PC_DIM) -> tuple[np.ndarray, np.ndarray]:
        """`(mean [D], components [k, D])` — the PCA basis of the *class* vectors.

        Fitted once from `class_emb`, so it is a property of the embedding table and not of any
        episode: every env, every scene and every viewer projects features onto the same axes and
        the resulting raster channels are comparable across environments. Signs are fixed by
        making each component's largest-magnitude entry positive.
        """
        k = int(k)
        if k < 1:
            raise ValueError(f"pc_basis: k must be >= 1, got {k}")
        hit = self._pc.get(k)
        if hit is not None:
            return hit
        x = self.class_emb.astype(np.float64)
        mean = x.mean(axis=0)
        u, sv, vt = np.linalg.svd(x - mean[None, :], full_matrices=False)
        comps = np.zeros((k, self.D), np.float64)
        take = min(k, vt.shape[0])
        comps[:take] = vt[:take]
        sign = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
        comps *= np.where(sign == 0.0, 1.0, sign)[:, None]
        out = (np.ascontiguousarray(mean, np.float32), np.ascontiguousarray(comps, np.float32))
        self._pc[k] = out
        return out

    def project(self, feat: np.ndarray, k: int = FEAT_PC_DIM) -> np.ndarray:
        """`feat [..., D]` -> `[..., k]` coordinates in the fixed class-PCA basis."""
        mean, comps = self.pc_basis(k)
        f = np.asarray(feat, np.float32)
        if f.shape[-1] != self.D:
            raise ValueError(f"project: feature dim {f.shape[-1]} != table dim {self.D}")
        return ((f - mean) @ comps.T).astype(np.float32)

    def embed_queries(self, names: Sequence[str]) -> np.ndarray:
        """float32 [len(names), D] unit vectors. Unknown names are encoded with the cached text
        tower if one is available, otherwise this raises and names the offending query."""
        names = tuple(names)
        if not names:
            raise ValueError("embed_queries: empty query list")
        miss = [n for n in names if n not in self.bank]
        if miss:
            for n, v in zip(miss, self._encode(miss)):
                self.bank[n] = v
        return np.ascontiguousarray([self.bank[n] for n in names], np.float32)

    def with_queries(self, names: Sequence[str]) -> "EmbeddingTable":
        names = tuple(names)
        return EmbeddingTable(class_emb=self.class_emb, query_emb=self.embed_queries(names),
                              names=names, source=self.source, bank=self.bank, meta=self.meta)

    # -- runtime encoding of unseen queries ----------------------------------------------------
    def _encode(self, names: Sequence[str]) -> np.ndarray:
        pca = self.meta.get("pca")
        if not pca:
            raise ValueError(
                f"EmbeddingTable({self.source}): no embedding for queries {list(names)!r}. "
                f"This table only knows {sorted(self.bank)!r}; it was factorized from the "
                f"hand-authored similarity table, which has no text encoder behind it. Build the "
                f"cache first: `uv run --extra embed python scripts/build_text_embeddings.py`.")
        try:
            import open_clip
            import torch
        except ImportError as exc:                                   # pragma: no cover - optional
            raise ValueError(
                f"EmbeddingTable({self.source}): query {list(names)!r} is not in the cache and "
                f"open_clip is not installed ({exc}), so it cannot be encoded. Install the extra "
                f"(`uv sync --extra embed`) or add the query to the cache with "
                f"scripts/build_text_embeddings.py.") from exc
        model_name = self.meta.get("model", "")
        pretrained = self.meta.get("pretrained", "")
        model = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained,
                                                       return_transform=False)
        tok = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            raw = model.encode_text(tok(list(names))).float().cpu().numpy()
        return project_pca(raw, np.asarray(pca["mean"], np.float64),
                           np.asarray(pca["components"], np.float64))


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.ascontiguousarray(a / np.maximum(n, 1e-12), np.float32)


def project_pca(raw: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Centre raw text vectors, project onto the cached PCA basis and renormalise."""
    x = np.asarray(raw, np.float64).reshape(-1, mean.shape[0]) - mean[None, :]
    return _unit(x @ np.asarray(components, np.float64).T)


# ---- (b) factorization of the hand-authored table -------------------------------------------
def factorize_table(table: np.ndarray, dim: int = DEFAULT_DIM, iters: int = 400,
                    lr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors `C [N, dim]`, `Qv [Q, dim]` with `C @ Qv.T ~= table`.

    Initialised from the eigendecomposition of the joint (classes + queries) Gram matrix with
    negative eigenvalues clipped, then refined by projected gradient descent (the eigen-init alone
    leaves ~0.17 of error because clipping the indefinite completion perturbs the cross block).
    The joint Gram has rank <= N + Q, so a larger `dim` is filled with zero columns rather than
    returning vectors narrower than `rayfronts.embedding_dim` asked for.
    """
    t = np.asarray(table, np.float64)
    if t.ndim != 2:
        raise ValueError(f"factorize_table: table must be 2-D, got shape {t.shape}")
    n_c, n_q = t.shape
    if dim < 2:
        raise ValueError(f"factorize_table: dim must be >= 2, got {dim}")
    rank = min(int(dim), n_c + n_q)          # the joint Gram has at most n_c + n_q directions

    def corr(a: np.ndarray) -> np.ndarray:
        b = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        return b @ b.T

    n = n_c + n_q
    g = np.empty((n, n), np.float64)
    g[:n_c, :n_c] = corr(t)
    g[n_c:, n_c:] = corr(t.T)
    g[:n_c, n_c:] = t
    g[n_c:, :n_c] = t.T
    np.fill_diagonal(g, 1.0)
    w, v = np.linalg.eigh(g)
    keep = np.argsort(-np.clip(w, 0.0, None))[:rank]
    x = v[:, keep] * np.sqrt(np.clip(w[keep], 0.0, None))[None, :]
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    c, qv = x[:n_c].copy(), x[n_c:].copy()

    mc = np.zeros_like(c)
    mq = np.zeros_like(qv)
    for _ in range(int(iters)):
        e = c @ qv.T - t
        mc = 0.9 * mc + 2.0 * (e @ qv)
        mq = 0.9 * mq + 2.0 * (e.T @ c)
        c -= lr * mc / n_c
        qv -= lr * mq / n_q
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        qv /= np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    if rank < dim:                           # pad to the requested D: cosines and norms unchanged
        c = np.pad(c, ((0, 0), (0, dim - rank)))
        qv = np.pad(qv, ((0, 0), (0, dim - rank)))
    return c.astype(np.float32), qv.astype(np.float32)


def factorized_table(queries: Sequence[str] = KNOWN_QUERIES, dim: int = DEFAULT_DIM,
                     sim_table_path: str | None = None) -> EmbeddingTable:
    """Embeddings synthesised from the hand-authored (or JSON-overridden) similarity table.

    Every query the table knows is embedded, not only `queries`, so `set_queries` can switch to
    another hand-authored query without a text encoder.
    """
    all_q = tuple(KNOWN_QUERIES)
    table = (load_sim_table(sim_table_path, all_q) if sim_table_path else build_sim_table(all_q))
    c, qv = factorize_table(table, dim=dim)
    err = float(np.abs(c.astype(np.float64) @ qv.astype(np.float64).T - table).max())
    if err > FACTOR_TOL:
        raise ValueError(f"factorized_table: dim={dim} cannot represent the similarity table "
                         f"(max |cos - table| = {err:.3f} > {FACTOR_TOL}); raise "
                         f"rayfronts.embedding_dim (>= 12 fits the shipped table exactly)")
    bank = {n: qv[i] for i, n in enumerate(all_q)}
    names = tuple(queries)
    miss = [n for n in names if n not in bank]
    if miss:
        raise ValueError(f"factorized_table: unknown queries {miss!r}; the hand-authored table "
                         f"knows {list(all_q)!r}")
    src = "factorized" if not sim_table_path else f"factorized:{sim_table_path}"
    return EmbeddingTable(class_emb=c, query_emb=np.stack([bank[n] for n in names]), names=names,
                          source=src, bank=bank,
                          meta={"iters": 400, "table": sim_table_path or "similarity_table"})


# ---- (a) cached text embeddings ---------------------------------------------------------------
def load_embeddings(path: str | Path, queries: Sequence[str] | None = None) -> EmbeddingTable:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"load_embeddings: {p} does not exist; build it with "
                                f"`uv run --extra embed python scripts/build_text_embeddings.py`")
    d = json.loads(p.read_text())
    for k in ("class_names", "class_emb", "queries", "query_emb", "dim"):
        if k not in d:
            raise ValueError(f"load_embeddings({p}): missing key {k!r}")
    if tuple(d["class_names"]) != tuple(CLASS_NAMES):
        raise ValueError(f"load_embeddings({p}): class_names do not match schema.CLASS_NAMES")
    ce = np.asarray(d["class_emb"], np.float32)
    qe = np.asarray(d["query_emb"], np.float32)
    file_q = tuple(d["queries"])
    if ce.shape != (N_CLASSES, int(d["dim"])):
        raise ValueError(f"load_embeddings({p}): class_emb shape {ce.shape} != "
                         f"({N_CLASSES}, {d['dim']})")
    if qe.shape != (len(file_q), int(d["dim"])):
        raise ValueError(f"load_embeddings({p}): query_emb shape {qe.shape} != "
                         f"({len(file_q)}, {d['dim']})")
    bank = {n: v for n, v in zip(file_q, _unit(qe))}
    names = tuple(queries) if queries is not None else file_q
    meta = {k: v for k, v in d.items() if k not in ("class_emb", "query_emb")}
    t = EmbeddingTable(class_emb=ce, query_emb=qe, names=file_q, source=f"cache:{p}", bank=bank,
                       meta=meta)
    return t if names == file_q else t.with_queries(names)


# ---- resolution ---------------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _cached(queries: tuple[str, ...], dim: int, path: str | None, sim_table_path: str | None,
            stamp: float) -> EmbeddingTable:
    if path is not None:
        return load_embeddings(path, queries)
    if sim_table_path is None and DATA_PATH.exists():
        return load_embeddings(DATA_PATH, queries)
    return factorized_table(queries, dim=dim, sim_table_path=sim_table_path)


def get_embedding_table(queries: Iterable[str], dim: int = DEFAULT_DIM, path: str | None = None,
                        sim_table_path: str | None = None) -> EmbeddingTable:
    """Resolve the embedding source for a query list (cached: envs are rebuilt every reset).

    An explicit `path` wins; otherwise the packaged cache is used when present and no similarity
    table override is given; otherwise the hand-authored table is factorized.
    """
    src = path if path is not None else (None if sim_table_path else
                                         (str(DATA_PATH) if DATA_PATH.exists() else None))
    stamp = Path(src).stat().st_mtime if src and Path(src).exists() else 0.0
    t = _cached(tuple(queries), int(dim), path, sim_table_path, stamp)
    if t.D != int(dim):                      # a cache has its own D; do not run at another one silently
        raise ValueError(f"get_embedding_table: {t.source} has D={t.D} but rayfronts.embedding_dim "
                         f"is {int(dim)}; set embedding_dim to {t.D} or point embeddings_path at a "
                         f"cache of dim {int(dim)}")
    return t


__all__ = ["EmbeddingTable", "CLASS_PROMPTS", "DEFAULT_DIM", "FEAT_PC_DIM", "FACTOR_TOL",
           "DATA_PATH", "factorize_table",
           "factorized_table", "load_embeddings", "get_embedding_table", "project_pca"]
