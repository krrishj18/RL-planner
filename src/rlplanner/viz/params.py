"""Config / state lookups that survive the simulator's renames.

The sim renames fields and adds token types as it grows (segments replaced voxel candidates, a
"visited" topic is arriving). The visualizer must draw either side of that, so a field is fetched
by its current name and, failing that, by the *unique* field carrying the same suffix — never by a
hard-coded legacy name.
"""
from __future__ import annotations


def _field_names(obj) -> list[str]:
    f = getattr(obj, "__dataclass_fields__", None)
    if f:
        return list(f)
    return [k for k in vars(obj)] if hasattr(obj, "__dict__") else []


def field_by_suffix(obj, name: str, suffix: str, default=None):
    """`obj.name` if it exists, else the one field whose name ends in `suffix`, else `default`."""
    if hasattr(obj, name):
        return getattr(obj, name)
    cand = [k for k in _field_names(obj) if k.endswith(suffix)]
    if len(cand) == 1:
        return getattr(obj, cand[0])
    if default is not None:
        return default
    raise AttributeError(f"{type(obj).__name__} has no {name!r} and "
                         f"{len(cand)} fields ending in {suffix!r}: {cand}")


def ray_target_range_m(rf) -> float:
    """Fallback distance along the bearing at which a ray's target is placed (`el >= 0`)."""
    return float(field_by_suffix(rf, "ray_range_m", "_range_m", 50.0))


def token_type_names() -> tuple[str, ...]:
    """The sim's live token vocabulary; the viewer never hard-codes it."""
    from rlplanner.sim.state import TOKEN_TYPE_NAMES

    return tuple(TOKEN_TYPE_NAMES)


def token_slots(tokens_cfg, k_tokens: int) -> int:
    """How many ray token slots the builder reserves (derived when it is not a field)."""
    n = getattr(tokens_cfg, "k_ray", None)
    if n is not None:
        return int(n)
    used = int(bool(getattr(tokens_cfg, "include_hold", True)))
    for name in token_type_names():
        if name in ("hold", "ray"):
            continue
        used += int(getattr(tokens_cfg, f"k_{name}", 0) or 0)
    return max(0, int(k_tokens) - used)


def slot_ranges(tokens_cfg, k_tokens: int) -> dict[str, tuple[int, int]]:
    """`{token type: (first slot, last+1)}` in the builder's fixed order, skipping absent types."""
    out: dict[str, tuple[int, int]] = {}
    s = 0
    if getattr(tokens_cfg, "include_hold", True):
        out["hold"] = (0, 1)
        s = 1
    for name in token_type_names():
        if name == "hold":
            continue
        k = int(getattr(tokens_cfg, f"k_{name}", 0) or 0)
        if k <= 0:
            continue
        out[name] = (s, s + k)
        s += k
    return out


__all__ = ["field_by_suffix", "ray_target_range_m", "token_type_names", "token_slots",
           "slot_ranges"]
