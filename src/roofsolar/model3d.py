"""Step 5: vector 3D roof model (planar faces + walls + simple objects) and mesh export.

1. ``face_polygons``   face labels -> polygons (a partition, so neighbours share edges
                       exactly), then topology-preserving simplification of the whole
                       coverage (shared edges are simplified once, so no gaps or overlaps).
2. ``consistent_vertices``  every polygon vertex gets a 3D position that both faces agree
                       on wherever the faces meet:
                       * on a crease shared by faces a, b: move the vertex (in plan) onto the
                         planes' intersection line, so z_a = z_b there;
                       * where >= 3 crease faces meet: the least-squares common point of
                         their planes (exact for 3 planes);
                       * on step edges (planes do not meet) and outer edges: z from each
                         face's own plane (a vertical wall closes the gap).
3. ``build_mesh``      triangulate each planar polygon (constrained Delaunay, holes kept),
                       lift with the face plane, add walls (eaves down to the ground, steps
                       between faces) and prisms for non-face building objects (dormers,
                       chimneys).

Coordinates: local metres, x east, y north (pixel corner (col, row) = (col*dx, -row*dy)),
the same frame as the planes from step 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon, shape

from roofsolar.faces import Edge, Plane

# ------------------------------------------------------------------------------ polygons


def polygonize(
    labels: np.ndarray, pixel: tuple[float, float], min_area_m2: float = 0.05
) -> dict[int, list[Polygon]]:
    """Polygons of each positive label (pixel-exact boundaries, holes included)."""
    tr = Affine(pixel[0], 0, 0, 0, -pixel[1], 0)
    out: dict[int, list[Polygon]] = {}
    for geom, val in shapes(labels.astype(np.int32), mask=labels > 0, transform=tr):
        poly = shape(geom)
        if poly.area >= min_area_m2:
            out.setdefault(int(val), []).append(poly)
    return out


def node_coverage(geoms: list[Polygon], tol: float = 1e-6) -> list[Polygon]:
    """Insert into every polygon the neighbours' vertices that lie on its edges.

    The polygonizer drops collinear vertices, so at a T-junction (faces B and C meeting
    along a straight edge of face A) the junction point is a vertex of B and C but not of
    A. Without it, A cannot follow when the junction is moved in 3D, and the junction is
    not recognised as shared by three faces. ``shapely.snap`` adds those vertices.
    """
    tree = shapely.STRtree(geoms)
    out = []
    for i, g in enumerate(geoms):
        near = [j for j in tree.query(g.buffer(tol)) if j != i]
        if near:
            ref = shapely.union_all([geoms[j].boundary for j in near])
            g = shapely.snap(g, ref, tol)
        out.append(g)
    return out


def face_polygons(
    labels: np.ndarray, pixel: tuple[float, float], tolerance_m: float, min_area_m2: float = 0.05
) -> dict[int, list[Polygon]]:
    """Face polygons simplified as one coverage (shared edges stay shared).

    Pixel boundaries are staircases with amplitude about one pixel; a tolerance of about
    1.5 pixels removes the stairs while keeping the shape. ``coverage_simplify`` simplifies
    each shared edge once, for both neighbours, so the result has no gaps or overlaps.
    """
    raw = polygonize(labels, pixel, min_area_m2)
    keys = [(k, i) for k, polys in raw.items() for i in range(len(polys))]
    geoms = node_coverage([raw[k][i] for k, i in keys])
    simp = shapely.coverage_simplify(geoms, tolerance_m)
    out: dict[int, list[Polygon]] = {}
    for (k, _), g in zip(keys, simp, strict=True):
        g = shapely.make_valid(g)
        parts = g.geoms if isinstance(g, MultiPolygon) or hasattr(g, "geoms") else [g]
        for p in parts:
            if isinstance(p, Polygon) and p.area >= min_area_m2:
                out.setdefault(k, []).append(p)
    return out


# ------------------------------------------------------------------------------ vertices


def plane_line_2d(pa: Plane, pb: Plane) -> tuple[np.ndarray, np.ndarray] | None:
    """Plan-view line where z_a(x, y) = z_b(x, y): returns (point, unit direction).

    z = (d - nx x - ny y) / nz, so z_a = z_b is linear: g . (x, y) = h with
    g = (nx_a/nz_a - nx_b/nz_b, ny_a/nz_a - ny_b/nz_b), h = d_a/nz_a - d_b/nz_b.
    """
    na, nb = pa.normal, pb.normal
    g = np.array([na[0] / na[2] - nb[0] / nb[2], na[1] / na[2] - nb[1] / nb[2]])
    gn = np.linalg.norm(g)
    if gn < 1e-9:
        return None  # parallel planes: no intersection line
    h = pa.d / na[2] - pb.d / nb[2]
    p0 = g * h / gn**2
    return p0, np.array([-g[1], g[0]]) / gn


def common_point(planes: list[Plane]) -> tuple[np.ndarray, float]:
    """Least-squares point closest to all planes (exact intersection for 3 planes).

    Returns (point, condition number of the normal matrix)."""
    n = np.array([p.normal for p in planes])
    d = np.array([p.d for p in planes])
    a = n.T @ n
    return np.linalg.solve(a, n.T @ d), float(np.linalg.cond(a))


@dataclass
class VertexInfo:
    xy: np.ndarray  # final plan position
    faces: set[int]
    crease_faces: list[int] = field(default_factory=list)  # faces meeting here in creases
    pair: tuple[int, int] | None = None  # crease pair whose intersection line was used
    moved_m: float = 0.0
    kind: str = "single"  # single | crease | junction | unresolved_junction | step


@dataclass
class Model:
    polygons: dict[int, list[Polygon]]  # final 2D polygons (vertices moved)
    vertices: dict[tuple[float, float], VertexInfo]
    planes: dict[int, Plane]
    stats: dict = field(default_factory=dict)


def _key(xy) -> tuple[float, float]:
    return (round(float(xy[0]), 6), round(float(xy[1]), 6))


def consistent_vertices(
    polys: dict[int, list[Polygon]],
    planes: dict[int, Plane],
    edges: list[Edge],
    max_snap_m: float,
    max_cond: float = 1e6,
) -> Model:
    """Make shared vertices agree in 3D (see module docstring)."""
    crease = {(min(e.a, e.b), max(e.a, e.b)) for e in edges if e.kind != "step"}
    incidence: dict[tuple[float, float], set[int]] = {}
    for k, plist in polys.items():
        for p in plist:
            for ring in [p.exterior, *p.interiors]:
                for xy in ring.coords[:-1]:
                    incidence.setdefault(_key(xy), set()).add(k)

    verts: dict[tuple[float, float], VertexInfo] = {}
    for key, fs in incidence.items():
        xy = np.array(key)
        info = VertexInfo(xy.copy(), fs)
        cf = sorted(fs)
        pairs = [(a, b) for i, a in enumerate(cf) for b in cf[i + 1 :] if (a, b) in crease]
        crease_faces = sorted({f for pr in pairs for f in pr})
        info.crease_faces = crease_faces
        if len(crease_faces) >= 3:
            pt, cond = common_point([planes[f] for f in crease_faces])
            if cond < max_cond and np.linalg.norm(pt[:2] - xy) <= max_snap_m:
                info.xy, info.kind = pt[:2], "junction"
        if info.kind == "single" and pairs:
            # snap onto the nearest intersection line among the crease pairs
            best = None
            for a, b in pairs:
                line = plane_line_2d(planes[a], planes[b])
                if line is None:
                    continue
                p0, u = line
                proj = p0 + u * ((xy - p0) @ u)
                dist = np.linalg.norm(proj - xy)
                if dist <= max_snap_m and (best is None or dist < best[0]):
                    best = (dist, proj, (a, b))
            if best is not None:
                info.xy, info.pair = best[1], best[2]
                info.kind = "unresolved_junction" if len(crease_faces) >= 3 else "crease"
        if info.kind == "single" and len(fs) >= 2:
            info.kind = "step"
        info.moved_m = float(np.linalg.norm(info.xy - xy))
        verts[key] = info

    moved: dict[int, list[Polygon]] = {}
    for k, plist in polys.items():
        for p in plist:

            def mv(ring):
                return [tuple(verts[_key(c)].xy) for c in ring.coords[:-1]]

            q = Polygon(mv(p.exterior), [mv(r) for r in p.interiors])
            if not q.is_valid:  # a snap folded the ring: keep the unsnapped polygon
                q = p
            moved.setdefault(k, []).append(q)
    return Model(moved, verts, planes)


def crease_gaps(model: Model) -> np.ndarray:
    """Height disagreement between the faces that meet in creases at each snapped vertex.

    0 means perfectly consistent. Step faces are excluded (a wall closes those gaps).
    For junctions of more than 3 planes the least-squares point is not on every plane, so
    small gaps remain there.
    """
    gaps = []
    for v in model.vertices.values():
        if v.kind not in ("crease", "junction"):
            continue
        fs = v.crease_faces if v.kind == "junction" else list(v.pair)
        z = [model.planes[f].z_at(np.array(v.xy[0]), np.array(v.xy[1])) for f in fs]
        gaps.append(float(np.max(z) - np.min(z)))
    return np.array(gaps)


# ------------------------------------------------------------------------------ mesh


@dataclass
class MeshParts:
    vertices: list[np.ndarray] = field(default_factory=list)
    faces: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)
    part: list[str] = field(default_factory=list)  # per triangle: roof | wall | object
    face_id: list[int] = field(default_factory=list)  # per triangle: roof face id or -1
    n: int = 0

    def add(self, v: np.ndarray, f: np.ndarray, color, part: str, face_id: int = -1):
        self.vertices.append(v)
        self.faces.append(f + self.n)
        self.colors.append(np.tile(np.asarray(color, np.uint8), (len(f), 1)))
        self.part += [part] * len(f)
        self.face_id += [face_id] * len(f)
        self.n += len(v)

    def quad(self, a, b, c, d, color, part: str):
        self.add(np.array([a, b, c, d], float), np.array([[0, 1, 2], [0, 2, 3]]), color, part)


def triangulate(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Constrained Delaunay triangulation of a polygon with holes -> (xy vertices, tris)."""
    tris = shapely.constrained_delaunay_triangles(poly)
    pts: dict[tuple[float, float], int] = {}
    faces = []
    for t in tris.geoms:
        idx = []
        for c in t.exterior.coords[:3]:
            k = _key(c)
            if k not in pts:
                pts[k] = len(pts)
            idx.append(pts[k])
        faces.append(idx)
    xy = np.array(list(pts.keys())) if pts else np.zeros((0, 2))
    return xy, np.array(faces, int).reshape(-1, 3)


def roof_mesh(model: Model, colors: dict[int, tuple]) -> MeshParts:
    """Triangulated roof faces, each lifted onto its own plane."""
    mp = MeshParts()
    for k, plist in model.polygons.items():
        pl = model.planes[k]
        for p in plist:
            xy, tri = triangulate(p)
            if len(tri) == 0:
                continue
            z = pl.z_at(xy[:, 0], xy[:, 1])
            mp.add(np.c_[xy, z], tri, colors[k], "roof", k)
    return mp


def add_walls(
    mp: MeshParts,
    model: Model,
    inside,
    ground_z,
    step_pairs: set,
    wall_color=(200, 200, 200),
    step_color=(170, 170, 170),
    probe_m: float = 0.15,
):
    """Vertical walls on face-polygon edges.

    * outer edge (the far side is outside the building): wall from the roof edge down to
      the ground (DTM);
    * step edge (the far side is a face it meets in a step): wall between the two planes
      (added once per edge).
    ``inside(x, y)`` returns the label / building state at a point; the far side of each
    edge is probed ``probe_m`` outwards along the edge normal.
    """
    done = set()
    for k, plist in model.polygons.items():
        pk = model.planes[k]
        for p in plist:
            p = shapely.geometry.polygon.orient(p, 1.0)  # CCW exterior: outward = right side
            for ring in [p.exterior, *p.interiors]:
                c = np.array(ring.coords)
                for a, b in zip(c[:-1], c[1:], strict=True):
                    d = b - a
                    length = np.linalg.norm(d)
                    if length < 1e-6:
                        continue
                    out = np.array([d[1], -d[0]]) / length  # right-hand normal
                    mid = (a + b) / 2 + out * probe_m
                    label_far, building_far = inside(mid)
                    za, zb = pk.z_at(*a), pk.z_at(*b)
                    if not building_far:
                        ga, gb = ground_z(a), ground_z(b)
                        mp.quad([*a, za], [*b, zb], [*b, gb], [*a, ga], wall_color, "wall")
                    elif (
                        label_far > 0
                        and label_far != k
                        and (min(k, label_far), max(k, label_far)) in step_pairs
                    ):
                        key = (min(k, label_far), max(k, label_far), _key(a), _key(b))
                        rkey = (min(k, label_far), max(k, label_far), _key(b), _key(a))
                        if key in done or rkey in done:
                            continue
                        done.add(key)
                        po = model.planes[label_far]
                        oa, ob = po.z_at(*a), po.z_at(*b)
                        mp.quad([*a, za], [*b, zb], [*b, ob], [*a, oa], step_color, "wall")


def add_prisms(
    mp: MeshParts, polys: list[Polygon], base_z, top_z: list[float], color=(120, 95, 80)
):
    """Simple prisms for non-face building objects: walls from the roof up to a flat top."""
    for p, zt in zip(polys, top_z, strict=True):
        xy, tri = triangulate(p)
        if len(tri) == 0:
            continue
        mp.add(np.c_[xy, np.full(len(xy), zt)], tri, color, "object")
        c = np.array(p.exterior.coords)
        for a, b in zip(c[:-1], c[1:], strict=True):
            mp.quad([*a, base_z(a)], [*b, base_z(b)], [*b, zt], [*a, zt], color, "object")


def to_arrays(mp: MeshParts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.vstack(mp.vertices), np.vstack(mp.faces), np.vstack(mp.colors)
