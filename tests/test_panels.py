"""Tests for panel placement geometry."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from roofsolar.faces import Plane
from roofsolar.panels import (
    PlaneFrame,
    check_layout,
    compactness,
    face_region,
    pack_grid,
    pack_rows,
    place_on_region,
    to_plane,
)

W, H, GAP = 1.134, 1.722, 0.02


def _plane(slope_deg, aspect_deg, z0=100.0):
    t = np.tan(np.radians(slope_deg))
    a = np.radians(aspect_deg)
    # downslope towards the aspect -> gradient points the other way
    gx, gy = -t * np.sin(a), -t * np.cos(a)
    n = np.array([-gx, -gy, 1.0])
    n /= np.linalg.norm(n)
    return Plane(n, z0 * n[2])


@pytest.mark.parametrize("slope,aspect", [(35, 224), (0, 0), (60, 90), (20, 315)])
def test_frame_is_an_isometry_and_e1_is_horizontal(slope, aspect):
    pl = _plane(slope, aspect)
    fr = PlaneFrame.of(pl, np.array([3.0, -4.0]))
    assert abs(fr.e1[2]) < 1e-12 and fr.e2[2] >= -1e-12
    assert np.dot(fr.e1, fr.e2) == pytest.approx(0, abs=1e-12)
    uv = np.array([[0, 0], [2.0, 0], [2.0, 3.0]])
    p3 = fr.to_3d(uv)
    np.testing.assert_allclose(p3 @ pl.normal, pl.d, atol=1e-9)  # on the plane
    np.testing.assert_allclose(np.linalg.norm(p3[1] - p3[0]), 2.0)
    np.testing.assert_allclose(np.linalg.norm(p3[2] - p3[1]), 3.0)


def test_plan_to_plane_stretches_along_slope():
    pl = _plane(35, 180)  # faces south: up-slope is north
    fr = PlaneFrame.of(pl, np.array([0.0, 0.0]))
    sq = box(0, 0, 1, 1)  # 1 x 1 m in plan
    uv = to_plane(sq, fr, pl)
    assert uv.area == pytest.approx(1 / np.cos(np.radians(35)))  # true sloped area
    umin, vmin, umax, vmax = uv.bounds
    assert umax - umin == pytest.approx(1.0) and vmax - vmin == pytest.approx(
        1 / np.cos(np.radians(35))
    )


def test_rows_optimal_on_rectangle():
    region = box(0, 0, 10, 5)
    boxes, _ = pack_rows(region, W, H, GAP, 0.01)
    per_row = int((10 + GAP) // (W + GAP))
    rows = int((5 + GAP) // (H + GAP))
    assert len(boxes) == per_row * rows


def test_rows_beat_or_match_grid_on_irregular_region():
    region = Polygon([(0, 0), (10, 0), (10, 4), (6, 4), (6, 6), (0, 6)])  # L-shape
    rows, _ = pack_rows(region, W, H, GAP, 0.02)
    grid, _ = pack_grid(region, W, H, GAP, 0.02)
    assert len(rows) >= len(grid) > 0


def test_place_and_check_layout_with_hole_and_setback():
    face_plan = box(0, 0, 10, 6)
    usable_plan = face_plan.difference(box(4, 2, 5, 3))  # an obstruction
    pl = _plane(0, 180)
    fr = PlaneFrame.of(pl, np.array([0.0, 0.0]))
    region, face_uv = face_region(face_plan, usable_plan, fr, pl, setback=0.3)
    lay = place_on_region(region, W, H, GAP, 0.05, "rows")
    assert len(lay.boxes) > 0
    chk = check_layout(lay.boxes, region, face_uv, GAP, 0.3)
    assert chk["inside"] and chk["overlaps"] == 0
    assert chk["min_gap_m"] >= GAP - 1e-9
    assert chk["min_edge_distance_m"] >= 0.3 - 1e-9
    hole = to_plane(box(4, 2, 5, 3), fr, pl)
    assert all(not b.intersects(hole) for b in lay.boxes)
    assert 0 < compactness(lay.boxes) <= 1


def test_dp_matches_analytic_optimum_on_rectangle():
    from roofsolar.panels import pack_rows_dp

    region = box(0, 0, 10, 5)
    boxes, _ = pack_rows_dp(region, [(W, H)], GAP, 0.01)
    assert len(boxes) == int((10 + GAP) // (W + GAP)) * int((5 + GAP) // (H + GAP))


def test_dp_mixed_uses_both_orientations_when_it_pays():
    """Height fits one portrait row + one landscape row, but not two of either kind."""
    from roofsolar.panels import pack_rows_dp

    height = H + GAP + W + 0.01  # 2.876 m: 2 portrait rows need 3.464, 2 landscape fit
    region = box(0, 0, 6.0, height)
    single_p, _ = pack_rows_dp(region, [(W, H)], GAP, 0.01)
    single_l, _ = pack_rows_dp(region, [(H, W)], GAP, 0.01)
    mixed, _ = pack_rows_dp(region, [(W, H), (H, W)], GAP, 0.01)
    assert len(mixed) > max(len(single_p), len(single_l))
    # no overlaps and gaps respected
    for i, a in enumerate(mixed):
        for b in mixed[i + 1 :]:
            assert a.distance(b) >= GAP - 1e-9


def test_dp_never_worse_than_fixed_pitch_rows():
    from roofsolar.panels import pack_rows_dp

    region = Polygon([(0, 0), (12, 0), (9, 5), (3, 5)])  # trapezoid (a hip face)
    rows, _ = pack_rows(region, W, H, GAP, 0.05)
    dp, _ = pack_rows_dp(region, [(W, H)], GAP, 0.05)
    assert len(dp) >= len(rows)
