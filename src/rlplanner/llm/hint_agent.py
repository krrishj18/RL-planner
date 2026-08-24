"""The hint agent: an LLM that edits the policy's query list and nothing else.

It cannot choose a target, a waypoint or a token — its whole output surface is
`{add, remove, reweight}` over query strings, which reach the policy as `TeamObs.query_emb /
query_w / query_mask` through `DisasterEnv.set_queries`. The simulator still ranks nothing and the
observation still carries raw embeddings.

Backends:
  `claude`   — one subprocess call per turn to the headless Claude Code CLI
               (`claude -p <prompt> --output-format json --model <model>`), strict JSON protocol
               with the schema in the prompt and one retry on malformed output.
  `scripted` — a deterministic rule-based mock for tests and CI: a fixed initial list, an add on
               the first vehicle-class sighting in the digest, a removal after N updates.

**Failure is never fatal.** A missing CLI, a timeout, a non-zero exit, an `is_error` result or
unparseable output all come back as no-op edits with a warning on `HintAgent.warnings`; the episode
keeps running on the query list it already had.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..sim.config import DEFAULT_QUERIES
from .digest import DigestBuilder

log = logging.getLogger("rlplanner.llm")

DEFAULT_MODEL = "opus"
DEFAULT_TIMEOUT_S = 120.0
W_LO, W_HI = 0.0, 1.0
VOCAB_IN_PROMPT = 40          # lexicon entries listed to the model (the whole bank is ~26)


class BackendError(RuntimeError):
    """Any failure of a hint backend. Caught by `HintAgent`, never raised into an episode."""


# ---- edits ---------------------------------------------------------------------------------
@dataclass
class QueryEdits:
    """What one turn of the agent proposes. Empty on every failure path."""

    add: list[tuple[str, float]] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    reweight: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def is_noop(self) -> bool:
        return not (self.add or self.remove or self.reweight)

    def to_dict(self) -> dict[str, Any]:
        return {"add": [[t, float(w)] for t, w in self.add], "remove": list(self.remove),
                "reweight": {k: float(v) for k, v in self.reweight.items()}, "note": self.note}

    @classmethod
    def parse(cls, d: Any) -> "QueryEdits":
        """Tolerant of the shapes a model actually emits; anything unusable is dropped."""
        if not isinstance(d, dict):
            raise BackendError(f"edits must be a JSON object, got {type(d).__name__}")
        return cls(add=_pairs(d.get("add")), remove=[str(x) for x in _listify(d.get("remove"))],
                   reweight={str(k): _w(v) for k, v in (d.get("reweight") or {}).items()},
                   note=str(d.get("note", ""))[:200])


def _w(v: Any) -> float:
    try:
        return min(W_HI, max(W_LO, float(v)))
    except (TypeError, ValueError):
        return 1.0


def _listify(v: Any) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _pairs(v: Any) -> list[tuple[str, float]]:
    """`[{"text":…, "weight":…}] | [["text", w]] | ["text"] | {"text": w}` -> [(text, weight)]."""
    if isinstance(v, dict):
        return [(str(k), _w(w)) for k, w in v.items()]
    out: list[tuple[str, float]] = []
    for item in _listify(v):
        if isinstance(item, dict):
            t = item.get("text", item.get("query", item.get("name")))
            if t:
                out.append((str(t), _w(item.get("weight", item.get("w", 1.0)))))
        elif isinstance(item, (list, tuple)) and item:
            out.append((str(item[0]), _w(item[1]) if len(item) > 1 else 1.0))
        elif isinstance(item, str):
            out.append((item, 1.0))
    return out


# ---- prompts -------------------------------------------------------------------------------
_ROLE = """You are the query-hint component of a multi-drone disaster-search planner.

The drones run an open-set semantic mapper (RayFronts): every mapped voxel and every long-range
semantic ray carries a text-embedding feature. The flight policy is a trained network that attends
from those features to a short list of MISSION QUERIES. You control that list and nothing else.
You never choose a target, a waypoint or a search area — you only decide which words the policy is
matching the map against, and how strongly (weight 0..1; a weight is a likelihood, not a priority).

A good hint names something the drones might actually see from above that is *evidence about where
casualties are* in this particular disaster (containers people end up in, signatures of the damage
that trapped them). A bad hint is a word the mapper cannot distinguish or a duplicate of a query
already active."""

_JSON_RULES = """Reply with ONE JSON object and NOTHING else: no prose, no markdown fence, no
explanation before or after."""


def initial_prompt(context: str, cap: int, vocabulary: Sequence[str] = ()) -> str:
    p = [_ROLE, "", "MISSION CONTEXT", context, ""]
    p += _vocab_block(vocabulary)
    p += [f"Choose at most {cap} mission queries for the start of this episode.", "", _JSON_RULES,
          'Schema: {"queries": [{"text": "<query>", "weight": <0..1>}], "note": "<=20 words"}']
    return "\n".join(p)


def update_prompt(digest: str, cap: int, vocabulary: Sequence[str] = ()) -> str:
    p = [_ROLE, "", "LIVE MAP DIGEST", digest, "",
         "Class names in the digest are a nearest-neighbour decode of the stored embeddings, shown "
         "to you only; the policy still receives the raw features.", ""]
    p += _vocab_block(vocabulary)
    p += [f"Edit the query list. Keep at most {cap} active queries. Return empty lists if nothing "
          f"should change — churn costs the policy context.", "", _JSON_RULES,
          'Schema: {"add": [{"text": "<query>", "weight": <0..1>}], "remove": ["<query>"], '
          '"reweight": {"<query>": <0..1>}, "note": "<=20 words"}']
    return "\n".join(p)


def _vocab_block(vocabulary: Sequence[str]) -> list[str]:
    if not vocabulary:
        return []
    v = list(vocabulary)[:VOCAB_IN_PROMPT]
    return ["This deployment has no live text encoder, so a query must come from this vocabulary "
            "(anything else is dropped):", "  " + "; ".join(v), ""]


# ---- backends ------------------------------------------------------------------------------
class ScriptedBackend:
    """Deterministic mock. Same digests in, same edits out — no network, no subprocess."""

    name = "scripted"

    def __init__(self, initial: Sequence[tuple[str, float]] = (),
                 add_on: Sequence[str] = ("vehicle_toppled", "vehicle_intact", "overturned car"),
                 add: tuple[str, float] = ("crushed vehicle", 0.6),
                 drop_after: int = 3, drop: str = "person"):
        self.initial_list = [(t, float(w)) for t, w in (initial or
                                                        [(q, 1.0) for q in DEFAULT_QUERIES])]
        self.add_on = tuple(add_on)
        self.add = (str(add[0]), float(add[1]))
        self.drop_after = int(drop_after)
        self.drop = str(drop)
        self.n_updates = 0
        self.added = False

    def reset(self) -> None:
        self.n_updates = 0
        self.added = False

    def initial(self, context: str) -> list[tuple[str, float]]:
        return list(self.initial_list)

    def update(self, digest: str) -> dict[str, Any]:
        self.n_updates += 1
        out: dict[str, Any] = {"add": [], "remove": [], "reweight": {},
                               "note": f"scripted update {self.n_updates}"}
        if not self.added and any(k in digest for k in self.add_on):
            out["add"] = [{"text": self.add[0], "weight": self.add[1]}]
            self.added = True
        if self.n_updates == self.drop_after:
            out["remove"] = [self.drop]
        return out


class ClaudeBackend:
    """One subprocess call per turn to the headless Claude Code CLI.

    `claude -p <prompt> --output-format json --model <model>` returns a JSON envelope whose
    `result` field holds the model's text; `--output-format text` returns the text itself.
    """

    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, cli: str = "claude",
                 output_format: str = "json", timeout_s: float = DEFAULT_TIMEOUT_S,
                 extra_args: Sequence[str] = (), cwd: str | None = None,
                 runner: Callable[[list[str], float], str] | None = None):
        self.model = str(model)
        self.cli = str(cli)
        self.output_format = str(output_format)
        self.timeout_s = float(timeout_s)
        self.extra_args = list(extra_args)
        self.cwd = cwd
        self.runner = runner
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    # -- transport ----------------------------------------------------------------------------
    def argv(self, prompt: str) -> list[str]:
        a = [self.cli, "-p", prompt, "--output-format", self.output_format]
        if self.model:
            a += ["--model", self.model]
        return a + self.extra_args

    def _run(self, prompt: str) -> str:
        self.calls += 1
        argv = self.argv(prompt)
        if self.runner is not None:
            return self._text(self.runner(argv, self.timeout_s))
        if shutil.which(self.cli) is None:
            raise BackendError(f"claude CLI {self.cli!r} not on PATH")
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_s,
                               cwd=self.cwd or os.getcwd())
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"claude CLI timed out after {self.timeout_s:.0f}s") from exc
        except OSError as exc:
            raise BackendError(f"claude CLI could not be run: {exc}") from exc
        if p.returncode != 0:
            raise BackendError(f"claude CLI exit {p.returncode}: {(p.stderr or '')[:300]}")
        return self._text(p.stdout)

    def _text(self, stdout: str) -> str:
        if self.output_format != "json":
            return stdout
        try:
            env = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise BackendError(f"claude CLI envelope is not JSON: {stdout[:200]!r}") from exc
        if isinstance(env, dict) and env.get("is_error"):
            raise BackendError(f"claude CLI reported an error: {str(env.get('result'))[:300]}")
        if isinstance(env, dict) and "result" in env:
            return str(env["result"])
        raise BackendError(f"claude CLI envelope has no 'result': {sorted(env)[:8]}")

    def ask(self, prompt: str, key: str) -> Any:
        """Ask, parse, and retry once on malformed output before giving up."""
        last = ""
        for attempt in range(2):
            p = prompt if attempt == 0 else (
                f"{prompt}\n\nYour previous reply was not usable ({last[:120]!r}). "
                f"Reply with the JSON object only, exactly matching the schema.")
            text = self._run(p)
            try:
                d = extract_json(text)
            except BackendError as exc:
                last = str(exc)
                continue
            if key in d or attempt == 1:
                return d
            last = f"no {key!r} key in {sorted(d)[:6]}"
        raise BackendError(f"claude CLI gave no usable JSON after a retry: {last}")

    # -- protocol -----------------------------------------------------------------------------
    def initial(self, context: str, cap: int = 8, vocabulary: Sequence[str] = ()
                ) -> list[tuple[str, float]]:
        return _pairs(self.ask(initial_prompt(context, cap, vocabulary), "queries").get("queries"))

    def update(self, digest: str, cap: int = 8, vocabulary: Sequence[str] = ()) -> dict[str, Any]:
        return self.ask(update_prompt(digest, cap, vocabulary), "add")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """First JSON object in `text`, unwrapping a markdown fence if the model added one."""
    s = str(text or "").strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            raise BackendError(f"no JSON object in reply: {s[:200]!r}") from None
        try:
            d = json.loads(s[i:j + 1])
        except json.JSONDecodeError as exc:
            raise BackendError(f"reply is not valid JSON: {s[:200]!r}") from exc
    if not isinstance(d, dict):
        raise BackendError(f"reply is a {type(d).__name__}, expected a JSON object")
    return d


def make_backend(kind: str = "scripted", **kw):
    """`"claude"` | `"scripted"`, or an object that already implements the two methods."""
    if not isinstance(kind, str):
        return kind
    if kind == "scripted":
        return ScriptedBackend(**{k: v for k, v in kw.items()
                                  if k in ("initial", "add_on", "add", "drop_after", "drop")})
    if kind == "claude":
        return ClaudeBackend(**{k: v for k, v in kw.items()
                                if k in ("model", "cli", "output_format", "timeout_s",
                                         "extra_args", "cwd", "runner")})
    raise ValueError(f"make_backend: unknown backend {kind!r} (claude | scripted)")


# ---- agent ---------------------------------------------------------------------------------
class HintAgent:
    """Holds the active query list and turns backend output into validated edits.

    `initial_queries(context)` and `update(digest)` are the whole API. Neither raises: a backend
    failure logs a warning and yields the previous list / no-op edits.
    """

    def __init__(self, backend: Any = "scripted", embedder=None, max_queries: int = 8,
                 default_queries: Sequence[str] = DEFAULT_QUERIES, constrain_vocab: bool = True,
                 **backend_kw):
        self.backend = make_backend(backend, **backend_kw)
        self.embedder = embedder
        self.max_queries = int(max_queries)
        self.default_queries = [(str(q), 1.0) for q in default_queries]
        self.constrain_vocab = bool(constrain_vocab)
        self.active: list[tuple[str, float]] = []
        self.warnings: list[str] = []
        self.dropped: list[tuple[str, str]] = []      # (text, why) — queries that never embedded

    @property
    def name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    def reset(self, active: Sequence[tuple[str, float]] | None = None) -> None:
        if hasattr(self.backend, "reset"):
            self.backend.reset()
        self.active = list(active or [])
        self.warnings.clear()
        self.dropped.clear()

    # -- vocabulary -----------------------------------------------------------------------------
    def vocabulary(self) -> tuple[str, ...]:
        """The words the prompt constrains the model to (empty = a live encoder is available)."""
        if self.embedder is None or not self.constrain_vocab:
            return ()
        return () if self.embedder.mode == "projection" else self.embedder.vocabulary()

    def _accept(self, text: str) -> str | None:
        """Resolve and register a proposed query; None (with a note) when it cannot be embedded."""
        t = str(text or "").strip()
        if not t:
            return None
        if self.embedder is None:
            return t
        name, how = self.embedder.register(t)
        if name is None:
            self.dropped.append((t, how))
            self._warn(f"dropped query {t!r}: {how}")
        return name

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("[hint] %s", msg)

    # -- turns ----------------------------------------------------------------------------------
    def initial_queries(self, context: str) -> list[tuple[str, float]]:
        """The query list to start the episode with. Falls back to the mission defaults."""
        try:
            got = self.backend.initial(context, self.max_queries, self.vocabulary()) \
                if _takes_cap(self.backend, "initial") else self.backend.initial(context)
            pairs = _pairs(got)
        except Exception as exc:                       # any backend failure: keep the mission list
            self._warn(f"initial_queries failed ({exc}); using the mission defaults")
            pairs = list(self.default_queries)
        out = self._validated(pairs)
        if not out:
            self._warn("no proposed query survived validation; using the mission defaults")
            out = self._validated(self.default_queries) or list(self.default_queries)
        self.active = self._cap(out)
        return list(self.active)

    def _validated(self, pairs) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for t, w in pairs:
            n = self._accept(t)
            if n is not None and n not in seen:
                seen.add(n)
                out.append((n, _w(w)))
        return out

    def update(self, digest: str) -> QueryEdits:
        """Validated edits for this interval. Any failure gives `QueryEdits()` (a no-op)."""
        try:
            raw = self.backend.update(digest, self.max_queries, self.vocabulary()) \
                if _takes_cap(self.backend, "update") else self.backend.update(digest)
            edits = QueryEdits.parse(raw)
        except Exception as exc:
            self._warn(f"update failed ({exc}); no query edits this interval")
            return QueryEdits(note="backend failure")
        return self._validate(edits)

    def _validate(self, e: QueryEdits) -> QueryEdits:
        active = {t for t, _ in self.active}
        add: list[tuple[str, float]] = []
        for t, w in e.add:
            n = self._accept(t)
            if n is None or n in active or n in [x for x, _ in add]:
                continue
            add.append((n, _w(w)))
        remove = [t for t in dict.fromkeys(e.remove) if t in active]
        # the mission list may never go empty: the sim rejects one, and an empty active list would
        # leave the agent describing a list the env does not have
        if remove and not add and len(remove) >= len(active):
            self._warn(f"kept {remove[-1]!r}: removing it would empty the query list")
            remove.pop()
        reweight = {t: _w(w) for t, w in e.reweight.items() if t in active and t not in remove}
        return QueryEdits(add=add, remove=remove, reweight=reweight, note=e.note)

    # -- list arithmetic --------------------------------------------------------------------------
    def apply(self, edits: QueryEdits) -> list[tuple[str, float]]:
        """Fold `edits` into the active list and cap it at the query-token capacity."""
        cur = {t: w for t, w in self.active}
        order = [t for t, _ in self.active]
        for t in edits.remove:
            cur.pop(t, None)
        for t, w in edits.reweight.items():
            if t in cur:
                cur[t] = w
        for t, w in edits.add:
            if t not in cur:
                order.append(t)
            cur[t] = w
        self.active = self._cap([(t, cur[t]) for t in order if t in cur])
        return list(self.active)

    def _cap(self, pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """At most `max_queries`, keeping the heaviest and, at equal weight, the oldest."""
        if len(pairs) <= self.max_queries:
            return pairs
        keep = sorted(range(len(pairs)), key=lambda i: (-pairs[i][1], i))[: self.max_queries]
        self._warn(f"query list capped at {self.max_queries}: dropped "
                   f"{[pairs[i][0] for i in range(len(pairs)) if i not in set(keep)]}")
        return [pairs[i] for i in sorted(keep)]


def _takes_cap(backend, method: str) -> bool:
    """Whether this backend's turn method accepts the `(cap, vocabulary)` prompt constraints."""
    try:
        return len(inspect.signature(getattr(backend, method)).parameters) >= 3
    except (TypeError, ValueError):                    # pragma: no cover - exotic callables
        return False


# ---- closed loop ---------------------------------------------------------------------------
class HintController:
    """Runs the agent against a live env: digest -> edits -> `env.set_queries`.

    `start(env)` sets the opening list; `after_step(env, info)` fires every `every` decisions and
    on an event (a new casualty), applies the edits and appends a record to `self.log`.
    """

    def __init__(self, agent: HintAgent, embedder=None, every: int = 5, on_events: bool = True,
                 digest: DigestBuilder | None = None, condition: str = ""):
        self.agent = agent
        self.embedder = embedder if embedder is not None else agent.embedder
        self.every = int(every)
        self.on_events = bool(on_events)
        self.digest = digest or DigestBuilder()
        self.condition = condition
        self.log: list[dict[str, Any]] = []
        self.n_since = 0

    def reset(self) -> None:
        self.agent.reset()
        self.digest.reset()
        self.log.clear()
        self.n_since = 0

    def start(self, env, context: str | None = None):
        # the cap is the env's query-token capacity, not whatever the agent was built with, or a
        # list the agent thinks is legal raises out of set_queries and into the episode
        self.agent.max_queries = min(self.agent.max_queries, int(env.cfg.tokens.max_queries))
        ctx = context if context is not None else self.digest.context(env)
        pairs = self.agent.initial_queries(ctx)
        obs = self._push(env, pairs)
        self._record(env, "initial", QueryEdits(add=list(pairs)), pairs)
        return obs

    def after_step(self, env, info: dict | None = None, force: bool = False):
        """-> the refreshed observation when the list changed, else None."""
        self.n_since += 1
        event = bool(self.on_events and info and int(info.get("new_found", 0)) > 0)
        if not (force or event or (self.every > 0 and self.n_since >= self.every)):
            return None
        self.n_since = 0
        d = self.digest.build(env, metrics=(info or {}).get("metrics"))
        edits = self.agent.update(d)
        before = list(self.agent.active)
        pairs = self.agent.apply(edits)
        obs = None if pairs == before else self._push(env, pairs)
        self._record(env, "event" if event else "cadence", edits, pairs, digest=d)
        return obs

    def _push(self, env, pairs):
        """Register the vectors, then hand the list to the env (which rebuilds the query block)."""
        if not pairs:
            return None
        if self.embedder is not None:
            pairs = [(n, w) for n, w in ((self.embedder.register(t)[0], w) for t, w in pairs)
                     if n is not None]
        if not pairs:
            return None
        return env.set_queries([t for t, _ in pairs], [w for _, w in pairs])

    def _record(self, env, kind: str, edits: QueryEdits, pairs, digest: str = "") -> None:
        self.log.append({"condition": self.condition, "kind": kind,
                         "decision": int(env.state.decision_idx), "t": float(env.state.t),
                         "backend": self.agent.name, **edits.to_dict(),
                         "queries": [t for t, _ in pairs], "weights": [float(w) for _, w in pairs],
                         "warnings": list(self.agent.warnings[-3:]), "digest": digest})


__all__ = ["QueryEdits", "HintAgent", "HintController", "ClaudeBackend", "ScriptedBackend",
           "BackendError", "make_backend", "extract_json", "initial_prompt", "update_prompt",
           "DEFAULT_MODEL"]
