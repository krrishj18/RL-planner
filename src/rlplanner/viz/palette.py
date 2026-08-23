"""Single source of colour for every visualizer panel.

All colours are chosen to stay legible on a white figure background; the belief panel uses
`UNOBSERVED` as its ground colour, which is dark enough to separate from any similarity value.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_rgb, to_rgba

from rlplanner.scene import schema

# ---- raster classes -------------------------------------------------------------------------
CLASS_COLORS: dict[str, str] = {
    "ground":             "#e9e3d3",
    "road":               "#9b9ba3",
    "sidewalk":           "#cbcbd2",
    "park":               "#b9d9a4",
    "building_intact":    "#d2b48c",
    "building_damaged":   "#e8912a",
    "building_destroyed": "#8b1a1a",
    "debris":             "#8b6b4a",
    "vehicle_intact":     "#2f6fdb",
    "vehicle_toppled":    "#a020c0",
    "bus_stop":           "#00a0a0",
    "tree":               "#2e8b3d",
    "street_furniture":   "#6b6b78",
    "human_standing":     "#ff2d95",
    "human_prone":        "#d40000",
}
assert set(CLASS_COLORS) == set(schema.CLASS_NAMES), "CLASS_COLORS out of sync with schema.CLASS_NAMES"


def class_color(cls: int | str) -> str:
    """Colour for a raster class id or name."""
    if isinstance(cls, str):
        if cls not in CLASS_COLORS:
            raise KeyError(f"unknown raster class name {cls!r}")
        return CLASS_COLORS[cls]
    i = int(cls)
    if not (0 <= i < schema.N_CLASSES):
        raise IndexError(f"raster class id {cls} outside [0, {schema.N_CLASSES})")
    return CLASS_COLORS[schema.CLASS_NAMES[i]]


def class_rgb_array() -> np.ndarray:
    """(N_CLASSES, 3) float array indexed by class id."""
    return np.array([to_rgb(CLASS_COLORS[n]) for n in schema.CLASS_NAMES], dtype=np.float64)


def class_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete colormap + norm mapping class id -> colour for `imshow`."""
    cmap = ListedColormap(class_rgb_array(), name="rlplanner_classes")
    norm = BoundaryNorm(np.arange(-0.5, schema.N_CLASSES + 0.5, 1.0), cmap.N)
    return cmap, norm


# ---- scene entities -------------------------------------------------------------------------
FATE_COLORS: dict[str, str] = {
    "intact":    CLASS_COLORS["building_intact"],
    "damaged":   CLASS_COLORS["building_damaged"],
    "destroyed": CLASS_COLORS["building_destroyed"],
}
VEHICLE_COLORS: dict[str, str] = {
    "intact":  CLASS_COLORS["vehicle_intact"],
    "toppled": CLASS_COLORS["vehicle_toppled"],
}
PROP_COLORS: dict[str, str] = {
    "bus_stop": CLASS_COLORS["bus_stop"],
    "tree":     CLASS_COLORS["tree"],
    "other":    CLASS_COLORS["street_furniture"],
}
DEBRIS_COLOR = CLASS_COLORS["debris"]
ROAD_COLOR = CLASS_COLORS["road"]
SIDEWALK_COLOR = CLASS_COLORS["sidewalk"]
PARK_COLOR = CLASS_COLORS["park"]
GROUND_COLOR = CLASS_COLORS["ground"]
BLOCK_EDGE = "#b0aa9a"

HUMAN_COLORS: dict[str, str] = {"casualty": "#d40000", "bystander": "#1a9e3f"}
# marker per container, per CONTRACTS.md 9
HUMAN_MARKERS: dict[str, str] = {"open": "o", "vehicle": "s", "building": "^", "rubble": "x"}
HUMAN_STATUS_EDGE: dict[str, str] = {
    "unfound": "#ffffff",
    "found":   "#00d0ff",
    "flash":   "#ff9500",      # a find, for the few frames the live viewer flashes it
}


def human_color(role: str) -> str:
    if role not in HUMAN_COLORS:
        raise KeyError(f"unknown human role {role!r}")
    return HUMAN_COLORS[role]


def human_marker(container: str) -> str:
    return HUMAN_MARKERS.get(container, "o")


def prop_color(category: str) -> str:
    return PROP_COLORS.get(category, PROP_COLORS["other"])


# ---- robots / belief overlays ----------------------------------------------------------------
ROBOT_COLORS: tuple[str, ...] = (
    "#0057ff", "#ff7f00", "#00b050", "#c000c0", "#00b8d4",
    "#b35c00", "#7a4fff", "#d0a000", "#e0004b", "#4d4d4d",
)


def robot_color(i: int) -> str:
    return ROBOT_COLORS[int(i) % len(ROBOT_COLORS)]


FRONTIER_COLOR = "#00e5ff"
RAY_TARGET_COLOR = "#ffd400"
SEGMENT_COLOR = "#a05cff"
VISITED_COLOR = "#8fb0c8"
FOUND_COLOR = "#00d0ff"
FLASH_COLOR = "#ff9500"
PEER_COLOR = "#ff70c0"          # peer-token arrows in a per-robot belief panel
LINK_COLOR = "#57ffb0"          # comms link between two robots in range
TRAJ_ALPHA = 0.85
FOV_ALPHA = 0.28
FOV_FAR_ALPHA = 0.10
TARGET_LINE = "#333333"

HOLD_COLOR = "#8a8a8a"
# markers per CONTRACTS.md 9: hold ., frontier o, ray *, segment D, visited s
TOKEN_COLORS: dict[str, str] = {
    "hold":      HOLD_COLOR,
    "frontier":  FRONTIER_COLOR,
    "ray":       RAY_TARGET_COLOR,
    "segment":   SEGMENT_COLOR,
    "visited":   VISITED_COLOR,
}
TOKEN_MARKERS: dict[str, str] = {"hold": ".", "frontier": "o", "ray": "*", "segment": "D",
                                 "visited": "s"}
# by slot, for a token type this palette does not know by name (the sim adds them)
TOKEN_SLOT_COLORS: tuple[str, ...] = (HOLD_COLOR, FRONTIER_COLOR, RAY_TARGET_COLOR, SEGMENT_COLOR,
                                      VISITED_COLOR, FLASH_COLOR, PEER_COLOR)
TOKEN_SLOT_MARKERS: tuple[str, ...] = (".", "o", "*", "D", "s", "P", "X")
TOKEN_CHOSEN_EDGE = "#ff0000"


def token_name(token_type: int | str) -> str:
    """The sim's own name for a token type id (`sim.state.TOKEN_TYPE_NAMES`)."""
    from rlplanner.sim.state import TOKEN_TYPE_NAMES

    if isinstance(token_type, str):
        return token_type
    i = int(token_type)
    if not (0 <= i < len(TOKEN_TYPE_NAMES)):
        raise IndexError(f"token type {token_type} outside [0, {len(TOKEN_TYPE_NAMES)})")
    return TOKEN_TYPE_NAMES[i]


def _slot_of(name: str) -> int | None:
    from rlplanner.sim.state import TOKEN_TYPE_NAMES

    return TOKEN_TYPE_NAMES.index(name) if name in TOKEN_TYPE_NAMES else None


def token_color(token_type: int | str) -> str:
    """Colour for a token type id or the sim's name for it; unknown names fall back to the slot."""
    name = token_name(token_type)
    if name in TOKEN_COLORS:
        return TOKEN_COLORS[name]
    slot = _slot_of(name)
    if slot is not None and slot < len(TOKEN_SLOT_COLORS):
        return TOKEN_SLOT_COLORS[slot]
    raise KeyError(f"unknown token type {name!r}")


def token_marker(token_type: int | str) -> str:
    name = token_name(token_type)
    if name in TOKEN_MARKERS:
        return TOKEN_MARKERS[name]
    slot = _slot_of(name)
    return TOKEN_SLOT_MARKERS[slot] if slot is not None and slot < len(TOKEN_SLOT_MARKERS) else "o"


# ---- fields ------------------------------------------------------------------------------------
UNOBSERVED = "#15171c"          # belief-panel ground: dark, never produced by SIM_CMAP
SIM_CMAP = LinearSegmentedColormap.from_list(
    "rlplanner_sim",
    ["#1b1170", "#5c04a5", "#9511a1", "#c53c74", "#e8654a", "#f9962b", "#f7d13d", "#fcffa4"],
    N=256)
SIM_VMIN, SIM_VMAX = 0.0, 1.0

DAMAGE_CMAP = LinearSegmentedColormap.from_list(
    "rlplanner_damage", ["#ffffcc", "#fdae61", "#d7191c"], N=256)
DAMAGE_CONTOUR_LEVELS = (0.25, 0.5, 0.75)
DAMAGE_CONTOUR_COLOR = "#7a0000"

HEIGHT_SHADE = (0.55, 1.0)      # (tallest, flat) multiplicative shade on the class colour


_PC_SCALE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def pc_scale(emb) -> tuple[np.ndarray, np.ndarray]:
    """(lo [3], hi [3]) of the *class* vectors on the table's fixed 3-PC basis.

    Fixing the range on the class rows makes a feature colour mean the same thing in every frame,
    scene and viewer — the same property `EmbeddingTable.pc_basis` gives the raster channels.
    """
    key = id(emb)
    hit = _PC_SCALE.get(key)
    if hit is not None:
        return hit
    p = np.asarray(emb.project(np.asarray(emb.class_emb, np.float32), 3), np.float64)
    lo, hi = p.min(0), p.max(0)
    out = (lo, np.maximum(hi, lo + 1e-6))
    _PC_SCALE[key] = out
    return out


def feat_rgb(feat: np.ndarray, emb) -> np.ndarray:
    """`feat [..., D]` -> RGB in [0,1] on the embedding table's fixed 3-PC basis.

    The colour of an item is a property of what it looks like, not of what else is on screen.
    """
    f = np.asarray(feat, np.float32)
    p = np.asarray(emb.project(f.reshape(-1, f.shape[-1]), 3), np.float64)
    lo, hi = pc_scale(emb)
    rgb = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    return rgb.reshape(f.shape[:-1] + (3,))


def sim_rgb(values: np.ndarray, observed: np.ndarray | None = None) -> np.ndarray:
    """Map similarity values in [0,1] to RGB, painting unobserved cells `UNOBSERVED`."""
    v = np.clip(np.asarray(values, dtype=np.float64), SIM_VMIN, SIM_VMAX)
    rgb = SIM_CMAP(v)[..., :3]
    if observed is not None:
        obs = np.asarray(observed, dtype=bool)
        if obs.shape != v.shape:
            raise ValueError(f"observed shape {obs.shape} != values shape {v.shape}")
        rgb[~obs] = to_rgb(UNOBSERVED)
    return rgb


def shade_by_height(rgb: np.ndarray, height: np.ndarray, hmax: float | None = None) -> np.ndarray:
    """Darken a class-colour image with extruded height; flat ground keeps its class colour."""
    h = np.asarray(height, dtype=np.float64)
    top = float(hmax) if hmax is not None else (float(np.nanmax(h)) if h.size else 0.0)
    lo, hi = HEIGHT_SHADE
    frac = np.sqrt(np.clip(h, 0.0, None) / top) if top > 1e-9 else np.zeros_like(h)
    f = np.where(h > 0.0, hi - (hi - lo) * frac, 1.0)
    return np.clip(rgb * f[..., None], 0.0, 1.0)


def alpha(color: str, a: float) -> tuple[float, float, float, float]:
    return to_rgba(color, a)


__all__ = [
    "CLASS_COLORS", "class_color", "class_rgb_array", "class_cmap",
    "FATE_COLORS", "VEHICLE_COLORS", "PROP_COLORS", "DEBRIS_COLOR", "ROAD_COLOR", "SIDEWALK_COLOR",
    "PARK_COLOR", "GROUND_COLOR", "BLOCK_EDGE", "HUMAN_COLORS", "HUMAN_MARKERS", "HUMAN_STATUS_EDGE",
    "human_color", "human_marker", "prop_color", "ROBOT_COLORS", "robot_color",
    "FRONTIER_COLOR", "RAY_TARGET_COLOR", "SEGMENT_COLOR", "VISITED_COLOR", "FOUND_COLOR",
    "FLASH_COLOR", "PEER_COLOR", "LINK_COLOR", "TOKEN_COLORS",
    "TOKEN_MARKERS", "TOKEN_SLOT_COLORS", "TOKEN_SLOT_MARKERS", "TOKEN_CHOSEN_EDGE", "token_color", "token_marker",
    "token_name",
    "UNOBSERVED", "SIM_CMAP", "DAMAGE_CMAP", "DAMAGE_CONTOUR_LEVELS", "DAMAGE_CONTOUR_COLOR",
    "sim_rgb", "shade_by_height", "alpha", "feat_rgb", "pc_scale",
]
