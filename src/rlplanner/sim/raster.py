"""2.5D raster of a Scene: per-cell height / class / damage / object id, plus hidden humans.

Grids are (ny, nx) row-major: row = y cell, col = x cell. Cell (i, j) covers
[x0 + j*cell, x0 + (j+1)*cell) x [y0 + i*cell, y0 + (i+1)*cell).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..scene import schema
from ..scene.schema import CLASS_ID, Scene

# Highest wins when footprints overlap (CONTRACTS.md 1).
_PRIORITY_BY_NAME: dict[str, int] = {
    "ground": 0, "park": 1, "road": 2, "sidewalk": 3, "tree": 4, "street_furniture": 5,
    "bus_stop": 6, "building_intact": 7, "building_damaged": 7, "building_destroyed": 7,
    "debris": 8, "vehicle_intact": 9, "vehicle_toppled": 9, "human_standing": 10, "human_prone": 10,
}
CLASS_PRIORITY = np.array([_PRIORITY_BY_NAME[n] for n in schema.CLASS_NAMES], dtype=np.int16)

HUMAN_DTYPE = np.dtype([
    ("x", "f8"), ("y", "f8"), ("z", "f8"),
    ("role_id", "i1"), ("pose_id", "i1"), ("container_id", "i1"), ("visibility_id", "i1"),
    ("scene_idx", "i4"),
])

_PROP_CLASS = {"bus_stop": "bus_stop", "tree": "tree"}


@dataclass
class Raster:
    cell_m: float
    origin: tuple[float, float]
    nx: int
    ny: int
    height: np.ndarray        # f32 [ny, nx] max extruded height
    cls: np.ndarray           # i8  [ny, nx] schema.CLASS_ID
    damage: np.ndarray        # f32 [ny, nx] in [0, 1]
    obj_id: np.ndarray        # i32 [ny, nx] index into `objects`, -1 = none
    objects: list[tuple[str, str]] = field(default_factory=list)   # (kind, scene id)
    humans: np.ndarray = field(default_factory=lambda: np.zeros(0, HUMAN_DTYPE))

    # -- geometry -----------------------------------------------------------------------------
    @property
    def region(self) -> tuple[float, float, float, float]:
        x0, y0 = self.origin
        return (x0, y0, x0 + self.nx * self.cell_m, y0 + self.ny * self.cell_m)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def diagonal_m(self) -> float:
        return float(math.hypot(self.nx * self.cell_m, self.ny * self.cell_m))

    def xy_to_ij(self, x, y):
        """World xy -> (row, col). Not clipped; use `in_bounds` to check."""
        x0, y0 = self.origin
        i = np.floor((np.asarray(y, dtype=np.float64) - y0) / self.cell_m).astype(np.int64)
        j = np.floor((np.asarray(x, dtype=np.float64) - x0) / self.cell_m).astype(np.int64)
        if np.isscalar(x) and np.isscalar(y):
            return int(i), int(j)
        return i, j

    def ij_to_xy(self, i, j):
        """(row, col) -> world xy of the cell centre."""
        x0, y0 = self.origin
        x = x0 + (np.asarray(j, dtype=np.float64) + 0.5) * self.cell_m
        y = y0 + (np.asarray(i, dtype=np.float64) + 0.5) * self.cell_m
        if np.isscalar(i) and np.isscalar(j):
            return float(x), float(y)
        return x, y

    def in_bounds(self, i, j):
        i = np.asarray(i); j = np.asarray(j)
        r = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        return bool(r) if r.ndim == 0 else r

    def clip_xy(self, x, y):
        x0, y0, x1, y1 = self.region
        eps = 1e-6
        return (float(np.clip(x, x0 + eps, x1 - eps)), float(np.clip(y, y0 + eps, y1 - eps)))

    def obstacle_mask(self, alt: float, clearance: float) -> np.ndarray:
        """Cells a robot flying at `alt` cannot cross with `clearance` margin."""
        return self.height >= (alt - clearance)

    def object_at(self, i: int, j: int) -> tuple[str, str] | None:
        k = int(self.obj_id[i, j])
        return self.objects[k] if k >= 0 else None


# ---- rasterisation ------------------------------------------------------------------------
class _Painter:
    def __init__(self, r: Raster):
        self.r = r
        self.prio = np.full(r.shape, CLASS_PRIORITY[CLASS_ID["ground"]], dtype=np.int16)

    def _apply(self, i0: int, i1: int, j0: int, j1: int, mask: np.ndarray, h: float,
               cls_id: int, oid: int) -> None:
        r = self.r
        sub = (slice(i0, i1), slice(j0, j1))
        if not mask.any():
            return
        hv = r.height[sub]
        np.maximum(hv, np.float32(h), out=hv, where=mask)
        p = CLASS_PRIORITY[cls_id]
        take = mask & (self.prio[sub] <= p)
        r.cls[sub][take] = cls_id
        self.prio[sub][take] = p
        if oid >= 0:
            r.obj_id[sub][take] = oid

    def _window(self, xa: float, ya: float, xb: float, yb: float):
        r = self.r
        x0, y0 = r.origin
        j0 = int(math.floor((xa - x0) / r.cell_m)); j1 = int(math.floor((xb - x0) / r.cell_m)) + 1
        i0 = int(math.floor((ya - y0) / r.cell_m)); i1 = int(math.floor((yb - y0) / r.cell_m)) + 1
        return max(i0, 0), min(i1, r.ny), max(j0, 0), min(j1, r.nx)

    def _centre_cell(self, cx: float, cy: float, h: float, cls_id: int, oid: int) -> None:
        r = self.r
        i, j = r.xy_to_ij(cx, cy)
        if r.in_bounds(i, j):
            self._apply(i, i + 1, j, j + 1, np.ones((1, 1), bool), h, cls_id, oid)

    def rect(self, rect, h: float, cls_id: int, oid: int = -1) -> None:
        """Axis-aligned rectangle (x0, y0, x1, y1)."""
        i0, i1, j0, j1 = self._window(*rect)
        if i1 <= i0 or j1 <= j0:
            return
        r = self.r
        xs, ys = self._centres(i0, i1, j0, j1)
        m = ((xs[None, :] >= rect[0]) & (xs[None, :] <= rect[2])
             & (ys[:, None] >= rect[1]) & (ys[:, None] <= rect[3]))
        self._apply(i0, i1, j0, j1, m, h, cls_id, oid)

    def obb(self, cx: float, cy: float, sx: float, sy: float, yaw_rad: float,
            h: float, cls_id: int, oid: int = -1) -> None:
        hx, hy = abs(sx) / 2.0, abs(sy) / 2.0
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        ex, ey = hx * abs(c) + hy * abs(s), hx * abs(s) + hy * abs(c)
        i0, i1, j0, j1 = self._window(cx - ex, cy - ey, cx + ex, cy + ey)
        if i1 <= i0 or j1 <= j0:
            self._centre_cell(cx, cy, h, cls_id, oid)
            return
        xs, ys = self._centres(i0, i1, j0, j1)
        dx = xs[None, :] - cx
        dy = ys[:, None] - cy
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        m = (np.abs(lx) <= hx) & (np.abs(ly) <= hy)
        if not m.any():
            self._centre_cell(cx, cy, h, cls_id, oid)
            return
        self._apply(i0, i1, j0, j1, m, h, cls_id, oid)

    def disc(self, cx: float, cy: float, rad: float, h: float, cls_id: int, oid: int = -1) -> None:
        i0, i1, j0, j1 = self._window(cx - rad, cy - rad, cx + rad, cy + rad)
        if i1 <= i0 or j1 <= j0:
            self._centre_cell(cx, cy, h, cls_id, oid)
            return
        xs, ys = self._centres(i0, i1, j0, j1)
        d2 = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2
        m = d2 <= rad * rad
        if not m.any():
            self._centre_cell(cx, cy, h, cls_id, oid)
            return
        self._apply(i0, i1, j0, j1, m, h, cls_id, oid)

    def _centres(self, i0, i1, j0, j1):
        r = self.r
        x0, y0 = r.origin
        xs = x0 + (np.arange(j0, j1, dtype=np.float64) + 0.5) * r.cell_m
        ys = y0 + (np.arange(i0, i1, dtype=np.float64) + 0.5) * r.cell_m
        return xs, ys


def rasterize(scene: Scene, cell_m: float) -> Raster:
    if cell_m <= 0:
        raise ValueError(f"rasterize: cell_m must be > 0, got {cell_m}")
    x0, y0, x1, y1 = scene.region
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"rasterize: scene {scene.meta.preset}/{scene.meta.seed} has degenerate region {scene.region}")
    nx = max(1, int(math.ceil((x1 - x0) / cell_m)))
    ny = max(1, int(math.ceil((y1 - y0) / cell_m)))
    r = Raster(cell_m=float(cell_m), origin=(float(x0), float(y0)), nx=nx, ny=ny,
               height=np.zeros((ny, nx), np.float32),
               cls=np.full((ny, nx), CLASS_ID["ground"], np.int8),
               damage=np.zeros((ny, nx), np.float32),
               obj_id=np.full((ny, nx), -1, np.int32),
               objects=[])
    p = _Painter(r)

    def reg(kind: str, oid: str) -> int:
        r.objects.append((kind, oid))
        return len(r.objects) - 1

    for b in scene.blocks:
        if b.typology == "park":
            p.rect(b.rect, 0.0, CLASS_ID["park"], reg("block", b.id))
    for rd in scene.roads:
        cid = CLASS_ID["road"] if rd.kind == "road" else CLASS_ID["sidewalk"]
        p.rect(rd.rect, 0.0, cid, reg("road", rd.id))
    for b in scene.buildings:
        p.obb(b.center[0], b.center[1], b.size[0], b.size[1], math.radians(b.yaw_deg),
              b.resolved_height(), CLASS_ID[f"building_{b.fate}"], reg("building", b.id))
    for d in scene.debris:
        p.disc(d.center[0], d.center[1], d.radius_m, d.resolved_height(),
               CLASS_ID["debris"], reg("debris", d.id))
    for v in scene.vehicles:
        p.obb(v.center[0], v.center[1], v.size[0], v.size[1], math.radians(v.yaw_deg),
              v.resolved_height(), CLASS_ID[f"vehicle_{v.state}"], reg("vehicle", v.id))
    for pr in scene.props:
        cid = CLASS_ID[_PROP_CLASS.get(pr.category, "street_furniture")]
        oid = reg("prop", pr.id)
        if pr.category == "tree":
            p.disc(pr.center[0], pr.center[1], max(pr.size) / 2.0, pr.resolved_height(), cid, oid)
        else:
            p.obb(pr.center[0], pr.center[1], pr.size[0], pr.size[1], math.radians(pr.yaw_deg),
                  pr.resolved_height(), cid, oid)

    r.damage = _damage_grid(scene, r)
    r.humans = _human_array(scene)
    return r


def _damage_grid(scene: Scene, r: Raster) -> np.ndarray:
    x0, y0 = r.origin
    xs = x0 + (np.arange(r.nx, dtype=np.float64) + 0.5) * r.cell_m
    ys = y0 + (np.arange(r.ny, dtype=np.float64) + 0.5) * r.cell_m
    g = scene.damage_field.grid
    if g is not None:
        vals = np.asarray(g["values"], dtype=np.float32)
        gny, gnx = int(g["ny"]), int(g["nx"])
        if vals.shape != (gny, gnx):
            raise ValueError(f"damage_field.grid values shape {vals.shape} != ({gny}, {gnx})")
        gc = float(g["cell_m"])
        gj = np.clip(((xs - x0) / gc).astype(np.int64), 0, gnx - 1)
        gi = np.clip(((ys - y0) / gc).astype(np.int64), 0, gny - 1)
        return np.ascontiguousarray(vals[np.ix_(gi, gj)], dtype=np.float32)
    return _damage_analytic(scene, xs, ys)


def _damage_analytic(scene: Scene, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    df = scene.damage_field
    pr = df.params
    inside = float(pr.get("inside", 1.0))
    outside = float(pr.get("outside", 0.0))
    X = xs[None, :] + np.zeros((ys.size, 1))
    Y = ys[:, None] + np.zeros((1, xs.size))
    if df.kind == "uniform":
        return np.full(X.shape, inside, np.float32)

    def ease(dist, full, fall):
        t = np.clip((dist - full) / max(fall, 1e-6), 0.0, 1.0)
        s = t * t * (3.0 - 2.0 * t)
        return np.where(dist <= full, inside, inside + (outside - inside) * s)

    if df.kind == "radial":
        cx, cy = pr.get("center", [0.0, 0.0])
        d = np.hypot(X - cx, Y - cy)
        return ease(d, float(pr.get("radius_m", 80.0)), float(pr.get("falloff_m", 120.0))).astype(np.float32)
    if df.kind == "path":
        pts = pr.get("points")
        if not pts:
            x0, y0, x1, y1 = scene.region
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            h = math.radians(float(pr.get("heading_deg", 0.0)))
            L = math.hypot(x1 - x0, y1 - y0)
            pts = [[cx - L * math.cos(h), cy - L * math.sin(h)], [cx + L * math.cos(h), cy + L * math.sin(h)]]
        d = None
        for k in range(len(pts) - 1):
            ax, ay = pts[k]; bx, by = pts[k + 1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 <= 1e-12 else np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0.0, 1.0)
            dk = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
            d = dk if d is None else np.minimum(d, dk)
        return ease(d, float(pr.get("width_m", 60.0)) / 2.0, float(pr.get("falloff_m", 40.0))).astype(np.float32)
    raise ValueError(f"unknown damage field kind {df.kind!r}")


def _human_array(scene: Scene) -> np.ndarray:
    hs = np.zeros(len(scene.humans), HUMAN_DTYPE)
    for k, h in enumerate(scene.humans):
        hs[k] = (float(h.pos[0]), float(h.pos[1]), float(h.pos[2]),
                 schema.HUMAN_ROLES.index(h.role), schema.HUMAN_POSES.index(h.pose),
                 schema.HUMAN_CONTAINERS.index(h.container), schema.HUMAN_VISIBILITY.index(h.visibility), k)
    return hs


__all__ = ["Raster", "rasterize", "CLASS_PRIORITY", "HUMAN_DTYPE"]
