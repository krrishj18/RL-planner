"""Text -> the simulator's D-dimensional query space.

The mission queries the policy consumes are unit vectors in the belief's own feature space
(`EmbeddingTable.query_emb`, `D = rayfronts.embedding_dim`). An LLM hint is a *string*, so it has
to be encoded into that space before `set_queries` can take it. Two routes:

  **projection** — fit a least-squares linear map from the cached SigLIP text embeddings of the
  known class prompts (and of the query bank) to the corresponding sim vectors, then
  `novel = normalize(W @ siglip(text))`. This needs the SigLIP cache *and* a live text tower
  (`open_clip`) to encode a string the cache has never seen.

  **lexicon** — no text tower installed: the vocabulary is exactly what already has a vector
  (the class prompts and the query/synonym bank), the prompt handed to the LLM is constrained to
  it, and anything outside is resolved by an alias/token match or rejected. The map is still fitted
  when the cache is present, and `class_roundtrip()` checks it, so switching the tower on later
  changes nothing else.

Whichever route produced it, a resolved query is registered in the live `EmbeddingTable.bank`
under the LLM's own wording, which is what makes `set_queries(("crushed vehicle", ...))` legal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..scene.schema import CLASS_NAMES
from ..sim.embeddings import CLASS_PROMPTS, EmbeddingTable, load_embeddings, project_pca

SIGLIP_CACHE = Path(__file__).resolve().parents[1] / "sim" / "data" / "text_embeddings_siglip_vitb16.json"
RIDGE = 1e-3          # ridge on the least-squares fit: 26 anchors for a DxD map is near-square
MATCH_MIN = 0.5       # token-overlap a fuzzy lexicon hit must reach (LLM side only, never the sim)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower().replace("_", " ")).strip()


STOP = frozenset(("a", "an", "the", "of", "on", "in", "with", "and", "its", "seen", "from",
                  "above", "along", "by"))

# Word-level normalisation for the *fuzzy* stage only (an exact or alias hit never reaches it).
# Purely an LLM-side reading aid: it lets "crushed vehicle" find the toppled-car entry instead of
# being dropped. Nothing in the simulator consults it.
SYNONYMS: dict[str, str] = {
    "vehicle": "car", "vehicles": "car", "cars": "car", "auto": "car", "automobile": "car",
    "truck": "car", "van": "car", "sedan": "car", "suv": "car",
    "crushed": "overturned", "wrecked": "overturned", "flipped": "overturned",
    "toppled": "overturned", "upturned": "overturned", "rolled": "overturned",
    "victim": "person", "victims": "person", "casualty": "person", "casualties": "person",
    "survivor": "person", "survivors": "person", "body": "person", "bodies": "person",
    "human": "person", "people": "person", "persons": "person", "pedestrian": "person",
    "injured": "person", "trapped": "person",
    "prone": "lying", "supine": "lying", "laying": "lying", "collapsed": "collapsed",
    "collapse": "collapsed", "destroyed": "collapsed", "flattened": "collapsed",
    "pancaked": "collapsed", "razed": "collapsed",
    "damage": "damaged", "cracked": "damaged", "partially": "damaged",
    "debris": "rubble", "wreckage": "rubble", "ruins": "rubble",
    "structure": "building", "buildings": "building", "apartment": "building",
    "trees": "tree", "canopy": "tree", "vegetation": "tree",
    "pavement": "sidewalk", "footpath": "sidewalk", "footway": "sidewalk",
    "shelter": "bus stop", "asphalt": "road", "street": "road", "carriageway": "road",
}


def _tokens(s: str) -> frozenset[str]:
    return frozenset(SYNONYMS.get(t, t) for t in _norm_text(s).split() if t not in STOP)


@lru_cache(maxsize=4)
def _load_siglip(path: str, stamp: float) -> EmbeddingTable:
    """The cache file is half a megabyte of JSON and an eval builds one embedder per episode."""
    return load_embeddings(path)


def fit_projection(src: EmbeddingTable, dst: EmbeddingTable) -> tuple[np.ndarray, float]:
    """Least-squares map `W [D_dst, D_src]` taking `src` vectors onto `dst` vectors.

    Anchors: the class rows (same class in both tables) plus every query name both banks know.
    Returns `(W, max residual cosine error over the anchors)`.
    """
    xs = [src.class_emb]
    ys = [dst.class_emb]
    shared = [n for n in src.bank if n in dst.bank]
    if shared:
        xs.append(np.stack([src.bank[n] for n in shared]))
        ys.append(np.stack([dst.bank[n] for n in shared]))
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    a = np.linalg.solve(x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ y)     # [D_src, D_dst]
    pred = x @ a
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    err = float(1.0 - np.einsum("ij,ij->i", pred, y).min())
    return np.ascontiguousarray(a.T, np.float32), err


@dataclass
class QueryEmbedder:
    """Resolves free text to a unit query vector in the sim's feature space."""

    emb: EmbeddingTable                                  # the live belief's table
    lexicon: dict[str, np.ndarray] = field(default_factory=dict)   # canonical name -> unit [D]
    aliases: dict[str, str] = field(default_factory=dict)          # normalised text -> canonical
    W: np.ndarray | None = None                          # SigLIP -> sim map, None without the cache
    siglip: EmbeddingTable | None = None
    fit_err: float = float("nan")
    encoder: str = ""                                    # "" = no live text tower
    notes: list[str] = field(default_factory=list)

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def build(cls, emb: EmbeddingTable, siglip_path: str | Path | None = None,
              allow_encoder: bool = True) -> "QueryEmbedder":
        """Lexicon from `emb`, plus the SigLIP projection when the cache (and a tower) are there."""
        lex: dict[str, np.ndarray] = {}
        alias: dict[str, str] = {}
        for i, c in enumerate(CLASS_NAMES):
            prompt = CLASS_PROMPTS[c]
            lex[prompt] = _unit(emb.class_emb[i])
            alias.setdefault(_norm_text(prompt), prompt)
            alias.setdefault(_norm_text(c), prompt)
            alias.setdefault(_norm_text(c.replace("_", " ")), prompt)
        for n, v in emb.bank.items():                    # the query / synonym bank wins on a clash
            lex[n] = _unit(v)
            alias[_norm_text(n)] = n
        out = cls(emb=emb, lexicon=lex, aliases=alias)
        p = Path(siglip_path) if siglip_path is not None else SIGLIP_CACHE
        if p.exists():
            try:
                sg = _load_siglip(str(p), p.stat().st_mtime)
                out.siglip = sg
                out.W, out.fit_err = fit_projection(sg, emb)
            except Exception as exc:                     # a broken cache must not kill an episode
                out.notes.append(f"siglip cache {p.name} unusable: {exc}")
        else:
            out.notes.append(f"no siglip cache at {p}")
        if allow_encoder and out.W is not None and out.siglip is not None:
            try:
                import open_clip                          # noqa: F401
                out.encoder = str(out.siglip.meta.get("model", "open_clip"))
            except ImportError:
                out.notes.append("open_clip not installed: novel text falls back to the lexicon")
        return out

    @classmethod
    def for_env(cls, env, **kw) -> "QueryEmbedder":
        return cls.build(env.rf.emb, **kw)

    # ---- properties ---------------------------------------------------------------------------
    @property
    def D(self) -> int:
        return int(self.emb.D)

    @property
    def mode(self) -> str:
        """"projection" only when novel strings can actually be encoded; else "lexicon"."""
        return "projection" if (self.W is not None and self.encoder) else "lexicon"

    def vocabulary(self) -> tuple[str, ...]:
        """The words the LLM prompt is constrained to under the lexicon fallback."""
        return tuple(sorted(self.lexicon))

    # ---- resolution ---------------------------------------------------------------------------
    def resolve(self, text: str) -> tuple[str | None, str]:
        """-> `(canonical lexicon name or None, how it was matched)`. Never raises."""
        t = str(text or "").strip()
        if not t:
            return None, "empty"
        if t in self.lexicon:
            return t, "exact"
        n = _norm_text(t)
        if n in self.aliases:
            return self.aliases[n], "alias"
        want = _tokens(t)
        if not want:
            return None, "unmatched"
        best, score = None, 0.0
        for name in self.lexicon:                        # LLM-side text matching, not a sim rule
            have = _tokens(name)
            if not have:
                continue
            s = len(want & have) / len(want | have)
            if s > score or (s == score and best is not None and name < best):
                best, score = name, s
        return (best, f"fuzzy:{score:.2f}") if score >= MATCH_MIN else (None, "unmatched")

    def project(self, raw: np.ndarray) -> np.ndarray:
        """Raw SigLIP-space vector -> unit vector in the sim's query space."""
        if self.W is None:
            raise ValueError("QueryEmbedder.project: no SigLIP cache, so no projection was fitted")
        return _unit(self.W @ np.asarray(raw, np.float32).reshape(-1))

    def encode(self, text: str) -> np.ndarray | None:
        """SigLIP-encode `text` and project it. None when no live text tower is installed."""
        if not self.encoder or self.siglip is None or self.W is None:
            return None
        pca = self.siglip.meta.get("pca")
        try:
            import open_clip
            import torch
            model = open_clip.create_model_from_pretrained(
                self.siglip.meta.get("model", ""), pretrained=self.siglip.meta.get("pretrained", ""),
                return_transform=False)
            tok = open_clip.get_tokenizer(self.siglip.meta.get("model", ""))
            with torch.no_grad():
                raw = model.encode_text(tok([str(text)])).float().cpu().numpy()
            v = project_pca(raw, np.asarray(pca["mean"], np.float64),
                            np.asarray(pca["components"], np.float64))[0]
        except Exception as exc:                         # pragma: no cover - optional dependency
            self.notes.append(f"encode({text!r}) failed: {exc}")
            return None
        return self.project(v)

    def embed(self, text: str) -> tuple[np.ndarray | None, str]:
        """-> `(unit [D] or None, provenance)`. The one entry point a hint goes through."""
        name, how = self.resolve(text)
        if name is not None:
            return self.lexicon[name].copy(), f"{how}:{name}"
        v = self.encode(text)
        if v is not None:
            return v, f"siglip:{self.encoder}"
        return None, how

    def register(self, text: str) -> tuple[str | None, str]:
        """Embed `text` and put it in the live table's bank under the caller's own wording.

        Returns `(the name usable in set_queries, provenance)`. The bank is what
        `EmbeddingTable.embed_queries` reads, so this is the step that makes an LLM-invented
        query legal input to the simulator.
        """
        t = str(text or "").strip()
        if t in self.emb.bank:
            return t, "bank"
        v, how = self.embed(t)
        if v is None:
            return None, how
        self.emb.bank[t] = _unit(v)
        self.lexicon.setdefault(t, self.emb.bank[t])
        self.aliases.setdefault(_norm_text(t), t)
        return t, how

    def register_vector(self, name: str, vec: np.ndarray) -> str:
        """Put an explicit vector in the bank (used by the training-side sampler's noise draws)."""
        self.emb.bank[str(name)] = _unit(vec)
        return str(name)

    # ---- sanity -------------------------------------------------------------------------------
    def class_roundtrip(self) -> dict[str, str]:
        """Project each class prompt's SigLIP vector and report the class it lands nearest.

        Every entry must map to itself: that is the check that the fitted map really lands a known
        name on its own class embedding.
        """
        if self.siglip is None or self.W is None:
            raise ValueError("class_roundtrip: no SigLIP cache, so there is no projection to check")
        v = np.stack([self.project(r) for r in self.siglip.class_emb])
        near = (v @ self.emb.class_emb.T).argmax(axis=1)
        return {c: CLASS_NAMES[int(k)] for c, k in zip(CLASS_NAMES, near)}

    def describe(self) -> str:
        bits = [f"mode={self.mode}", f"D={self.D}", f"lexicon={len(self.lexicon)}",
                f"table={self.emb.source}"]
        if self.W is not None:
            bits.append(f"siglip_fit_err={self.fit_err:.3f}")
        if self.encoder:
            bits.append(f"encoder={self.encoder}")
        return "QueryEmbedder(" + ", ".join(bits) + ")" + (
            "" if not self.notes else "  [" + "; ".join(self.notes) + "]")


__all__ = ["QueryEmbedder", "fit_projection", "SIGLIP_CACHE", "MATCH_MIN", "SYNONYMS"]
