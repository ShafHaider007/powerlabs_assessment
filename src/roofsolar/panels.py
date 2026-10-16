"""Step 6: solar panel placement on suitable roof faces.

Each face is worked in its own plane coordinates, so panel sizes and setbacks are true
distances on the slope:

    e1 = horizontal direction in the plane (along the eave),  e1 = normalise(z x n)
    e2 = up-slope direction in the plane,                     e2 = n x e1
    (u, v) = ((P - o) . e1, (P - o) . e2)  with P = (x, y, z_plane(x, y))

The map plan (x, y) -> plane (u, v) is affine, so polygons transform exactly.

Region = usable pixels of the face (step 4: obstructions and keep-out removed), intersected
with the face polygon shrunk by the edge setback. Panels are rectangles of the standard size
(portrait or landscape) in rows parallel to the eave, separated by a mounting gap.

Packing strategies (all evaluated, the configured one is used):
* "grid": one global (u, v) offset; columns aligned across rows (tidiest).
* "rows": one vertical offset for a fixed row pitch; each row packed greedily left to right.
  For identical items on a line, earliest-feasible greedy maximises the count (interval
  scheduling, exchange argument), so each row is optimal for its height.
* "dp": dynamic programming over row start heights (rows anywhere, one orientation).
* "dp_mixed": as "dp" but each row may be portrait or landscape. Exact optimum over all
  row-based layouts at the search resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box

from roofsolar.faces import Plane


@dataclass
class PlaneFrame:
    origin: np.ndarray  # 3D point on the plane
    e1: np.ndarray  # horizontal, in plane
    e2: np.ndarray  # up-slope, in plane
    normal: np.ndarray

    @classmethod
    def of(cls, plane: Plane, anchor_xy: np.ndarray) -> PlaneFrame:
        n = plane.normal / np.linalg.norm(plane.normal)
        e1 = np.cross([0.0, 0.0, 1.0], n)
        if np.linalg.norm(e1) < 1e-9:  # flat plane: any horizontal direction
            e1 = np.array([1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        if e2[2] < 0:  # make e2 point up-slope (and keep a right-handed frame)
            e1, e2 = -e1, -e2
        o = np.array([anchor_xy[0], anchor_xy[1], plane.z_at(*anchor_xy)])
        return cls(o, e1, e2, n)

    def plan_to_plane_affine(self, plane: Plane) -> list[float]:
        """shapely affine params [a, b, d, e, xoff, yoff] mapping plan (x, y) -> (u, v)."""
        nx, ny, nz = plane.normal
        # P(x, y) = (x, y, (d - nx x - ny y)/nz) is affine in (x, y): P = P0 + x Px + y Py
        px = np.array([1.0, 0.0, -nx / nz])
        py = np.array([0.0, 1.0, -ny / nz])
        p0 = np.array([0.0, 0.0, plane.d / nz]) - self.origin
        a, b = px @ self.e1, py @ self.e1
        d, e = px @ self.e2, py @ self.e2
        return [a, b, d, e, p0 @ self.e1, p0 @ self.e2]

    def to_3d(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(uv)
        return self.origin + uv[:, :1] * self.e1 + uv[:, 1:2] * self.e2


def to_plane(geom, frame: PlaneFrame, plane: Plane):
    return affinity.affine_transform(geom, frame.plan_to_plane_affine(plane))


@dataclass
class Layout:
    boxes: list[Polygon]  # panel rectangles in plane (u, v) coordinates
    orientation: str  # portrait | landscape
    strategy: str
    offset: tuple[float, float]
    candidates: dict = field(default_factory=dict)  # count for every (strategy, orientation)


def _valid(region_prep, u: np.ndarray, v0: float, w: float, h: float) -> np.ndarray:
    boxes = shapely.box(u, v0, u + w, v0 + h)
    return shapely.contains(region_prep, boxes)


def pack_rows(region, w: float, h: float, gap: float, step: float) -> tuple[list, tuple]:
    """Best vertical offset; each row packed greedily (optimal per row)."""
    shapely.prepare(region)
    umin, vmin, umax, vmax = region.bounds
    best: tuple[list, tuple] = ([], (0.0, 0.0))
    us = np.arange(umin, umax - w + 1e-9, step)
    if len(us) == 0:
        return best
    for off in np.arange(0, h + gap, step):
        placed = []
        v0 = vmin + off
        while v0 + h <= vmax + 1e-9:
            ok = _valid(region, us, v0, w, h)
            last_end = -np.inf
            for u, good in zip(us, ok, strict=True):
                if good and u >= last_end + gap - 1e-9:
                    placed.append(box(u, v0, u + w, v0 + h))
                    last_end = u + w
            v0 += h + gap
        if len(placed) > len(best[0]):
            best = (placed, (0.0, float(off)))
    return best


def pack_grid(region, w: float, h: float, gap: float, step: float) -> tuple[list, tuple]:
    """Best global (u, v) offset of an aligned grid."""
    shapely.prepare(region)
    umin, vmin, umax, vmax = region.bounds
    best: tuple[list, tuple] = ([], (0.0, 0.0))
    for du in np.arange(0, w + gap, step):
        cols = np.arange(umin + du, umax - w + 1e-9, w + gap)
        if len(cols) == 0:
            continue
        for dv in np.arange(0, h + gap, step):
            placed = []
            for v0 in np.arange(vmin + dv, vmax - h + 1e-9, h + gap):
                ok = _valid(region, cols, v0, w, h)
                placed += [box(u, v0, u + w, v0 + h) for u in cols[ok]]
            if len(placed) > len(best[0]):
                best = (placed, (float(du), float(dv)))
    return best


def _row_greedy(region, us: np.ndarray, v0: float, w: float, h: float, gap: float) -> list:
    """Optimal packing of one row: earliest feasible start first (interval scheduling)."""
    ok = _valid(region, us, v0, w, h)
    out, last_end = [], -np.inf
    for u, good in zip(us, ok, strict=True):
        if good and u >= last_end + gap - 1e-9:
            out.append(box(u, v0, u + w, v0 + h))
            last_end = u + w
    return out


def pack_rows_dp(
    region, sizes: list[tuple[float, float]], gap: float, step: float
) -> tuple[list, tuple]:
    """Optimal row-based layout by dynamic programming over row start heights.

    Rows are parallel to the eave; a row starting at height v with panel size (w, h)
    holds the greedy-optimal count R(v, size). Rows that do not overlap are independent,
    so the best total from height index i upwards is

        F[i] = max( F[i+1],  max_size R(v_i, size) + F[i + ceil((h + gap) / step)] )

    which is the exact optimum over all row layouts at this resolution (free row positions
    and, with several sizes, a per-row orientation).
    """
    shapely.prepare(region)
    umin, vmin, umax, vmax = region.bounds
    hmin = min(h for _, h in sizes)
    vs = np.arange(vmin, vmax - hmin + 1e-9, step)
    n = len(vs)
    if n == 0:
        return [], (0.0, 0.0)
    rows = {}
    for si, (w, h) in enumerate(sizes):
        us = np.arange(umin, umax - w + 1e-9, step)
        for i, v0 in enumerate(vs):
            rows[(i, si)] = (
                _row_greedy(region, us, v0, w, h, gap) if v0 + h <= vmax + 1e-9 and len(us) else []
            )
    best = np.zeros(n + 1, int)
    choice = [None] * (n + 1)
    for i in range(n - 1, -1, -1):
        best[i], choice[i] = best[i + 1], None
        for si, (_, h) in enumerate(sizes):
            j = min(n, i + int(np.ceil((h + gap) / step - 1e-9)))
            val = len(rows[(i, si)]) + best[j]
            if val > best[i]:
                best[i], choice[i] = val, (si, j)
    placed, i = [], 0
    while i < n:
        if choice[i] is None:
            i += 1
            continue
        si, j = choice[i]
        placed += rows[(i, si)]
        i = j
    return placed, (0.0, 0.0)


def place_on_region(
    region, panel_w: float, panel_h: float, gap: float, step: float, strategy: str
) -> Layout:
    """Evaluate every strategy and both orientations; return the configured strategy.

    strategies: "grid" (aligned columns, one global offset), "rows" (fixed row pitch, one
    offset, each row greedy), "dp" (free row positions, one orientation per face),
    "dp_mixed" (free row positions, portrait or landscape per row). All candidate counts
    are recorded for comparison.
    """
    portrait, landscape = (panel_w, panel_h), (panel_h, panel_w)
    cands, results = {}, {}
    empty = region.is_empty
    for orient, size in (("portrait", portrait), ("landscape", landscape)):
        for strat, fn in (("rows", pack_rows), ("grid", pack_grid)):
            res = fn(region, *size, gap, step) if not empty else ([], (0, 0))
            cands[f"{strat}/{orient}"], results[(strat, orient)] = len(res[0]), res
        res = pack_rows_dp(region, [size], gap, step) if not empty else ([], (0, 0))
        cands[f"dp/{orient}"], results[("dp", orient)] = len(res[0]), res
    res = pack_rows_dp(region, [portrait, landscape], gap, step) if not empty else ([], (0, 0))
    cands["dp_mixed"], results[("dp_mixed", "mixed")] = len(res[0]), res
    if strategy == "dp_mixed":
        pick = "mixed"
    else:
        pick = max(
            ("portrait", "landscape"),
            key=lambda o: (len(results[(strategy, o)][0]), o == "portrait"),
        )
    boxes, off = results[(strategy, pick)]
    return Layout(boxes, pick, strategy, off, cands)


def face_region(face_poly_plan, usable_poly_plan, frame: PlaneFrame, plane: Plane, setback: float):
    """Placement region in plane coordinates: usable area minus the edge setback."""
    face_uv = to_plane(face_poly_plan, frame, plane)
    usable_uv = to_plane(usable_poly_plan, frame, plane)
    inner = face_uv.buffer(-setback, join_style="mitre") if setback > 0 else face_uv
    region = shapely.make_valid(usable_uv.intersection(inner))
    if isinstance(region, (Polygon, MultiPolygon)):
        return region, face_uv
    polys = [g for g in getattr(region, "geoms", []) if isinstance(g, Polygon)]
    return (MultiPolygon(polys) if polys else Polygon()), face_uv


def compactness(boxes: list[Polygon]) -> float:
    """Panel area / area of their convex hull (1 = one solid rectangle of panels)."""
    if not boxes:
        return 0.0
    u = shapely.union_all(boxes)
    return float(sum(b.area for b in boxes) / u.convex_hull.area)


def check_layout(boxes: list[Polygon], region, face_uv, gap: float, setback: float) -> dict:
    """Geometric validity: inside region, no overlap, gap kept, setback kept."""
    if not boxes:
        return {"inside": True, "min_gap_m": None, "min_edge_distance_m": None, "overlaps": 0}
    inside = all(region.covers(b) for b in boxes)
    tree = shapely.STRtree(boxes)
    min_gap, overlaps = np.inf, 0
    for i, b in enumerate(boxes):
        for j in tree.query(b.buffer(gap * 2)):
            if j <= i:
                continue
            if b.intersection(boxes[j]).area > 1e-12:
                overlaps += 1
            min_gap = min(min_gap, b.distance(boxes[j]))
    edge = min(face_uv.boundary.distance(b) for b in boxes)  # includes hole boundaries
    return {
        "inside": bool(inside),
        "min_gap_m": None if np.isinf(min_gap) else round(float(min_gap), 4),
        "min_edge_distance_m": round(float(edge), 4),
        "overlaps": overlaps,
    }
