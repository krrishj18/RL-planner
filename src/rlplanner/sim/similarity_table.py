"""Hand-authored CLASS x QUERY similarity table standing in for a CLIP/SigLIP text encoder.

Values are the cosine-similarity-like score in [0, 1] that RayFronts would report for a voxel of
that class against that text query. Rows are documented individually below; the numbers encode the
confusions that matter to the planner (rubble vs. collapsed building, a standing person answering a
"person lying" query, a toppled car still answering "car"), not photometric realism.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..scene.schema import CLASS_NAMES, N_CLASSES

Q = ("person lying on the ground", "person", "collapsed building", "damaged building", "rubble",
     "overturned car", "car", "bus stop", "road", "tree", "house")

# class -> {query: score}. Every class row must cover all of Q.
_ROWS: dict[str, tuple[float, ...]] = {
    # bare ground/grass: nothing scores high; slight "road" bleed from the flat grey look.
    "ground":             (0.05, 0.05, 0.05, 0.05, 0.12, 0.03, 0.03, 0.05, 0.30, 0.10, 0.05),
    # asphalt carriageway.
    "road":               (0.05, 0.05, 0.05, 0.05, 0.08, 0.05, 0.15, 0.10, 0.95, 0.05, 0.05),
    # paved footway: reads as "road" at reduced strength, some "bus stop" context.
    "sidewalk":           (0.06, 0.06, 0.05, 0.05, 0.08, 0.04, 0.08, 0.20, 0.60, 0.08, 0.08),
    # vegetated block interior.
    "park":               (0.05, 0.05, 0.03, 0.03, 0.05, 0.02, 0.03, 0.06, 0.10, 0.45, 0.05),
    # intact building: "house" strong, "damaged building" weakly (facade texture ambiguity).
    "building_intact":    (0.04, 0.05, 0.20, 0.35, 0.10, 0.03, 0.05, 0.15, 0.06, 0.08, 0.85),
    # damaged building: peak on "damaged building", strong on "collapsed"/"rubble".
    "building_damaged":   (0.05, 0.06, 0.60, 0.90, 0.55, 0.05, 0.06, 0.12, 0.06, 0.07, 0.55),
    # destroyed building: peak on "collapsed building"; a rubble mound reads as rubble.
    "building_destroyed": (0.05, 0.06, 0.92, 0.70, 0.80, 0.06, 0.06, 0.08, 0.06, 0.05, 0.30),
    # debris pile: peak on "rubble"; the contract's 0.5 confusion with "collapsed building".
    "debris":             (0.10, 0.10, 0.50, 0.35, 0.90, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10),
    # upright car.
    "vehicle_intact":     (0.05, 0.08, 0.05, 0.05, 0.06, 0.35, 0.92, 0.15, 0.25, 0.05, 0.06),
    # toppled car: peak on "overturned car", still 0.7 on plain "car" (contract).
    "vehicle_toppled":    (0.08, 0.10, 0.15, 0.12, 0.25, 0.90, 0.70, 0.12, 0.18, 0.05, 0.06),
    # bus shelter: some "house" bleed (small roofed structure).
    "bus_stop":           (0.06, 0.10, 0.08, 0.10, 0.08, 0.05, 0.10, 0.90, 0.25, 0.08, 0.20),
    "tree":               (0.05, 0.06, 0.04, 0.04, 0.05, 0.03, 0.04, 0.06, 0.06, 0.95, 0.08),
    # poles/benches/bins: thin vertical clutter, mildly confusable with a bus stop or a person.
    "street_furniture":   (0.05, 0.08, 0.05, 0.06, 0.06, 0.04, 0.08, 0.35, 0.20, 0.20, 0.08),
    # standing person: peak on "person"; 0.4 on "person lying" (contract).
    "human_standing":     (0.40, 0.95, 0.03, 0.03, 0.05, 0.03, 0.06, 0.10, 0.08, 0.10, 0.04),
    # prone person: peak on the primary casualty query; some "rubble" bleed when half-buried.
    "human_prone":        (0.92, 0.85, 0.05, 0.05, 0.25, 0.04, 0.05, 0.08, 0.10, 0.06, 0.04),
}

BASE_TABLE: dict[str, dict[str, float]] = {c: dict(zip(Q, v)) for c, v in _ROWS.items()}
assert set(BASE_TABLE) == set(CLASS_NAMES)


def build_sim_table(queries) -> np.ndarray:
    """float32 [N_CLASSES, len(queries)]. Unknown query names raise."""
    queries = tuple(queries)
    unknown = [q for q in queries if q not in Q]
    if unknown:
        raise ValueError(f"build_sim_table: unknown queries {unknown!r}; known: {list(Q)}")
    if not queries:
        raise ValueError("build_sim_table: empty query list")
    t = np.empty((N_CLASSES, len(queries)), np.float32)
    for i, c in enumerate(CLASS_NAMES):
        row = BASE_TABLE[c]
        for j, q in enumerate(queries):
            t[i, j] = row[q]
    return t


def load_sim_table(path, queries=None) -> np.ndarray:
    """Override the table from JSON: {"queries": [...], "values": [[...]]} (rows = CLASS_NAMES)
    or {"table": {class_name: {query: value}}}. Missing entries and out-of-range values raise."""
    d = json.loads(Path(path).read_text())
    if "table" in d:
        tbl = d["table"]
        qs = tuple(queries) if queries is not None else tuple(d.get("queries", Q))
        t = np.empty((N_CLASSES, len(qs)), np.float32)
        for i, c in enumerate(CLASS_NAMES):
            if c not in tbl:
                raise ValueError(f"load_sim_table({path}): missing class row {c!r}")
            for j, q in enumerate(qs):
                if q not in tbl[c]:
                    raise ValueError(f"load_sim_table({path}): class {c!r} missing query {q!r}")
                t[i, j] = float(tbl[c][q])
    else:
        file_q = tuple(d["queries"])
        vals = np.asarray(d["values"], np.float32)
        if vals.shape != (N_CLASSES, len(file_q)):
            raise ValueError(f"load_sim_table({path}): values shape {vals.shape} != "
                             f"({N_CLASSES}, {len(file_q)})")
        if queries is None:
            t = vals
        else:
            miss = [q for q in queries if q not in file_q]
            if miss:
                raise ValueError(f"load_sim_table({path}): unknown queries {miss!r}")
            t = vals[:, [file_q.index(q) for q in queries]]
    if not np.all((t >= 0.0) & (t <= 1.0)):
        raise ValueError(f"load_sim_table({path}): values outside [0, 1]")
    return np.ascontiguousarray(t, np.float32)


def query_index(queries, name: str, default: int = -1) -> int:
    qs = tuple(queries)
    return qs.index(name) if name in qs else default


__all__ = ["Q", "BASE_TABLE", "build_sim_table", "load_sim_table", "query_index"]
