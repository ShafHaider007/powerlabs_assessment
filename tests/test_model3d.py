"""Tests for the vector 3D model against an analytic hip roof."""

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from roofsolar.faces import Plane, face_adjacency, fit_all, pixel_coords
from roofsolar.model3d import (
    common_point,
    consistent_vertices,
    crease_gaps,
    face_polygons,
    node_coverage,
    plane_line_2d,
    roof_mesh,
    triangulate,
)
from roofsolar.synthetic import hip_roof_scene

PX = (0.1, 0.1)


def _plane(gx, gy, z0):
    n = np.array([-gx, -gy, 1.0])
    n /= np.linalg.norm(n)
    return Plane(n, z0 * n[2])


def test_plane_line_and_common_point():
    a, b, c = _plane(0.5, 0, 10), _plane(-0.5, 0, 10), _plane(0, 0.5, 9)
    p0, u = plane_line_2d(a, b)
    for t in (-3.0, 0.0, 5.0):
        q = p0 + t * u
        assert a.z_at(*q) == pytest.approx(b.z_at(*q))
    pt, cond = common_point([a, b, c])
    for p in (a, b, c):
        assert p.normal @ pt == pytest.approx(p.d)
    assert plane_line_2d(a, _plane(0.5, 0, 12)) is None  # parallel planes


def test_node_coverage_inserts_t_junction():
    a = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    b = Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
    c = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    na, *_ = node_coverage([a, b, c])
    assert (1.0, 1.0) in [tuple(p) for p in na.exterior.coords]


def test_triangulate_polygon_with_hole():
    p = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)], [[(1, 1), (2, 1), (2, 2), (1, 2)]])
    xy, tri = triangulate(p)
    area = sum(Polygon(xy[t]).area for t in tri)
    assert area == pytest.approx(11.0)


@pytest.fixture(scope="module")
def hip_model():
    sc = hip_roof_scene()
    labels = sc.faces.copy()
    x, y = pixel_coords(labels.shape, PX)
    planes = fit_all(labels, sc.z, x, y)
    edges = face_adjacency(labels, planes, x, y, 0.1, 0.1)
    polys = face_polygons(labels, PX, 0.15)
    return sc, planes, edges, polys, consistent_vertices(polys, planes, edges, 0.3)


def test_hip_model_consistent_and_planar(hip_model):
    sc, planes, edges, polys, model = hip_model
    assert len(model.polygons) == 4
    assert crease_gaps(model).max() < 1e-9
    kinds = [v.kind for v in model.vertices.values()]
    assert kinds.count("junction") == 2  # the two ridge ends
    mp = roof_mesh(model, {k: (200, 0, 0) for k in planes})
    v = np.vstack(mp.vertices)
    f = np.vstack(mp.faces)
    fid = np.array(mp.face_id)
    for k, pl in planes.items():
        idx = np.unique(f[fid == k])
        assert np.abs(v[idx] @ pl.normal - pl.d).max() < 1e-9


def test_hip_model_ridge_ends_match_analytic(hip_model):
    """Ridge ends of a hip roof: (+-(L-W)/2, 0) in building axes, z = eave + tan(p) W/2."""
    sc, planes, edges, polys, model = hip_model
    L, W, rot, eave, pitch, size = 18.0, 12.0, np.radians(44.0), 5.0, np.radians(35.0), 30.0
    zr = eave + np.tan(pitch) * W / 2
    truth = []
    for s in (-1, 1):
        u, v = s * (L - W) / 2, 0.0
        x = u * np.cos(rot) + v * np.sin(rot)  # inverse of the scene's rotation
        y = -u * np.sin(rot) + v * np.cos(rot)
        truth.append(np.array([x + size / 2, y - size / 2, zr]))
    found = []
    for vtx in model.vertices.values():
        if vtx.kind == "junction":
            z = planes[vtx.crease_faces[0]].z_at(*vtx.xy)
            found.append(np.array([*vtx.xy, z]))
    for t in truth:
        assert min(np.linalg.norm(f - t) for f in found) < 0.05


def test_simplification_keeps_coverage(hip_model):
    sc, planes, edges, polys, model = hip_model
    allp = [p for v in polys.values() for p in v]
    assert all(p.is_valid for p in allp)
    assert sum(p.area for p in allp) - shapely.union_all(allp).area < 1e-6
