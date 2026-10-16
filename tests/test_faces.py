"""Tests for face segmentation and planar reconstruction against synthetic ground truth."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from roofsolar.cleaning import tgv_denoise
from roofsolar.faces import (
    Plane,
    SegmentParams,
    candidate_mask,
    face_adjacency,
    fit_plane,
    pixel_coords,
    planar_surface,
    segment_roof,
)
from roofsolar.synthetic import correlated_noise, hip_roof_scene, realistic_dsm
from roofsolar.terrain import horn_gradients, slope_deg, surface_normals

PX = (0.1, 0.1)


def test_fit_plane_exact_and_robust():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 10, (2, 500))
    n = np.array([0.3, -0.2, 1.0])
    n /= np.linalg.norm(n)
    z = (5.0 - n[0] * x - n[1] * y) / n[2]
    pl, inl = fit_plane(np.c_[x, y, z])
    assert pl.angle_to(Plane(n, 5.0)) < 1e-6 and pl.rmse < 1e-9
    z2 = z.copy()
    z2[:100] += rng.uniform(0.5, 2.0, 100)  # 20 % gross outliers (an object on the face)
    pl2, inl2 = fit_plane(np.c_[x, y, z2])
    assert pl2.angle_to(Plane(n, 5.0)) < 0.05
    assert not inl2[:100].any() and inl2[100:].all()


def test_tls_unbiased_on_steep_plane_with_noise():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 5, (2, 4000))
    z = np.tan(np.radians(60)) * x
    pts = np.c_[x, y, z] + rng.normal(0, 0.03, (4000, 3))  # noise in all directions
    pl, _ = fit_plane(pts, iterations=0)
    assert abs(pl.slope_deg - 60) < 0.2


def test_plane_aspect_and_z():
    t = np.tan(np.radians(30))
    n = np.array([0.0, t, 1.0]) / np.hypot(t, 1)  # rises to the south -> faces north
    pl = Plane(n, 10.0 * n[2])
    assert pl.aspect_deg() == pytest.approx(0.0, abs=1e-9)
    assert pl.slope_deg == pytest.approx(30.0)
    assert pl.z_at(np.array(0.0), np.array(-1.0)) == pytest.approx(10 + t)


def _run(z, sc, params=None):
    gx, gy = horn_gradients(z, *PX)
    normals, slope = surface_normals(gx, gy), slope_deg(gx, gy)
    ndsm = z - z[0, 0]
    cand = candidate_mask(ndsm, np.zeros(z.shape, bool), np.ones(z.shape, bool), 1.5)
    labels, planes = segment_roof(z, normals, slope, cand, PX, params or SegmentParams())
    return labels, planes, cand


def _match(labels, planes, sc):
    """Map each true face to the found face with the largest overlap."""
    out = {}
    for k in sc.normals:
        found, counts = np.unique(labels[sc.faces == k], return_counts=True)
        counts, found = counts[found > 0], found[found > 0]
        out[k] = int(found[np.argmax(counts)]) if len(found) else None
    return out


@pytest.fixture(scope="module")
def clean_scene():
    sc = hip_roof_scene()
    z = sc.z + correlated_noise(sc.z.shape, 0.01, 1.0, np.random.default_rng(0))
    return sc, z, *_run(z, sc)


def test_hip_roof_faces_found(clean_scene):
    sc, z, labels, planes, cand = clean_scene
    m = _match(labels, planes, sc)
    assert len(set(m.values())) == 4 and None not in m.values()
    for k, j in m.items():
        assert planes[j].angle_to(Plane(sc.normals[k], 0)) < 0.5
    # the chimney top is not a roof face of its own (too small) and not part of a face
    assert (labels[sc.chimney] == 0).mean() > 0.9


def test_labels_match_truth_and_creases_are_sharp(clean_scene):
    sc, z, labels, planes, cand = clean_scene
    m = _match(labels, planes, sc)
    inv = np.zeros(labels.max() + 1, int)
    for k, j in m.items():
        inv[j] = k
    roof = (sc.faces > 0) & ~ndi.binary_dilation(sc.chimney, iterations=3)
    acc = (inv[labels[roof]] == sc.faces[roof]).mean()
    assert acc > 0.98
    rec = planar_surface(labels, planes, PX)
    crease = sc.crease & roof
    err = np.abs(rec - sc.z)[crease]
    assert np.nanmedian(err) < 0.02


def test_adjacency_types(clean_scene):
    sc, z, labels, planes, cand = clean_scene
    x, y = pixel_coords(labels.shape, PX)
    edges = face_adjacency(labels, planes, x, y, 0.1, 0.1)
    kinds = sorted(e.kind for e in edges)
    assert kinds.count("hip") == 4 and kinds.count("ridge") == 1
    for e in edges:
        assert e.line_angle_deg < 3 and e.line_offset_m < 0.1


def test_realistic_degradation_rebuilds_edges():
    """Blurred, wavy roof (as the provided DSM) -> TGV -> faces: creases/walls restored."""
    sc = hip_roof_scene()
    raw = realistic_dsm(sc, np.random.default_rng(3))
    z, _ = tgv_denoise(raw, 0.05, 0.2, n_iter=3000)
    labels, planes, cand = _run(z, sc)
    m = _match(labels, planes, sc)
    assert len(set(m.values())) == 4
    for k, j in m.items():
        assert planes[j].angle_to(Plane(sc.normals[k], 0)) < 1.5
    rec = planar_surface(labels, planes, PX)
    roof = (sc.faces > 0) & ~ndi.binary_dilation(sc.chimney, iterations=3)
    crease = sc.crease & roof
    # step 2 alone left ~11 cm at creases; plane reconstruction must do much better
    assert np.nanmedian(np.abs(rec - sc.z)[crease]) < 0.05
    assert np.nanmedian(np.abs(z - sc.z)[crease]) > 0.05


def test_merge_does_not_chain_through_a_bridge():
    """A (z=0) and C (z=0.08) are parallel but offset; a small bridge B at z=0.04 is
    'coplanar enough' with each of them. Union-find would merge A-B-C; greedy merging
    merges B into one side and then refuses the non-coplanar union."""
    from roofsolar.faces import merge_coplanar

    rng = np.random.default_rng(0)
    shape = (60, 100)
    x, y = pixel_coords(shape, PX)
    z = rng.normal(0, 0.002, shape)
    labels = np.zeros(shape, np.int32)
    labels[:, :48] = 1
    labels[:, 52:] = 3
    labels[28:32, 48:52] = 2  # small bridge
    z[:, 52:] += 0.08
    z[28:32, 48:52] += 0.04
    out = merge_coplanar(labels, z, x, y, angle_deg=5, dist_m=0.04, gap_px=2)
    assert out[0, 0] != out[0, -1]
    assert len(np.unique(out[out > 0])) == 2


def test_merge_joins_one_face_split_by_a_gap():
    from roofsolar.faces import merge_coplanar

    rng = np.random.default_rng(1)
    shape = (60, 100)
    x, y = pixel_coords(shape, PX)
    z = 0.3 * x + rng.normal(0, 0.01, shape)
    labels = np.zeros(shape, np.int32)
    labels[:, :45] = 1
    labels[:, 49:] = 2  # same plane, 0.4 m gap (e.g. a dormer between)
    out = merge_coplanar(labels, z, x, y, angle_deg=5, dist_m=0.04, gap_px=5)
    assert len(np.unique(out[out > 0])) == 1
