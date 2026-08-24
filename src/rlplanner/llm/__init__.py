"""Closed-loop LLM hint component: the LLM edits the policy's *query list*, nothing else.

It never picks a target, a waypoint or a token. It reads the disaster context and a bounded text
digest of the live simulated-RayFronts semantics and returns query edits, which reach the policy
through `DisasterEnv.set_queries` — i.e. through the same open-set channel a mission query uses
(`TeamObs.query_emb / query_w / query_mask`). The nearest-class decode in the digest is *for the
LLM's eyes only*: the policy still receives raw feature embeddings.

Modules:
  `embed.py`     text -> the sim's D-dimensional query space (SigLIP projection or the lexicon).
  `digest.py`    live belief -> a length-capped text digest.
  `hint_agent.py` the agent, its backends (`claude` subprocess / `scripted` mock) and the
                 closed-loop controller that applies edits to a running env.
  `schedule.py`  training-side query churn with no LLM in the loop (`cfg.queries_dynamic`).
"""
from __future__ import annotations

from .digest import DIGEST_MAX_CHARS, DigestBuilder, build_digest, nearest_class, scene_context
from .embed import QueryEmbedder, SIGLIP_CACHE, fit_projection
from .hint_agent import (BackendError, ClaudeBackend, HintAgent, HintController, QueryEdits,
                         ScriptedBackend, make_backend)
from .schedule import QueryScheduleSampler

__all__ = ["QueryEdits", "HintAgent", "HintController", "ClaudeBackend", "ScriptedBackend",
           "BackendError", "make_backend", "DigestBuilder", "build_digest", "nearest_class",
           "scene_context", "DIGEST_MAX_CHARS", "QueryEmbedder", "fit_projection", "SIGLIP_CACHE",
           "QueryScheduleSampler"]
