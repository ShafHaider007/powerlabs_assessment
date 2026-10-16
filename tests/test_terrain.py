"""Unit tests for terrain derivatives using synthetic surfaces with known answers."""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from roofsolar.terrain import (
    aspect_deg,
    estimate_ground,
    estimate_noise,
    grid_convergence_deg,
    horn_gradients,
    normal_scatter_deg,
    off_grid_mask,
    planar_tiles,
    slope_deg,
    surface_normals,
)

DX = DY = 0.1


def plane(slope: float, facing: float, shape=(50, 60), dx=DX, dy=DY) -> np.ndarray:
    """Plane with a given slope (deg) facing a given azimuth (deg from north, downslope)."""
    rows, cols = shape
    east = np.arange(cols) * dx
    north = -np.arange(rows) * dy  # row 0 is the northern edge
    e, n = np.meshgrid(east, north)
    g = np.tan(np.radians(slope))
    # height falls along the facing direction: z = -g * (e sin(facing) + n cos(facing))
    return 100 - g * (e * np.sin(np.radians(facing)) + n * np.cos(np.radians(facing)))


@pytest.mark.parametrize("facing", [0, 45, 90, 135, 180, 225, 270, 315, 10, 359])
@pytest.mark.parametrize("slope", [5, 30, 36, 60])
def test_plane_slope_and_aspect_exact(slope, facing):
    gx, gy = horn_gradients(plane(slope, facing), DX, DY)
    np.testing.assert_allclose(slope_deg(gx, gy), slope, atol=1e-9)
    asp = aspect_deg(gx, gy)
    diff = (asp - facing + 180) % 360 - 180  # circular difference
    np.testing.assert_allclose(diff, 0, atol=1e-9)


def test_non_square_pixels():
    z = plane(30, 135, dx=0.1, dy=0.25)
    gx, gy = horn_gradients(z, 0.1, 0.25)
    np.testing.assert_allclose(slope_deg(gx, gy), 30, atol=1e-9)
    np.testing.assert_allclose(aspect_deg(gx, gy), 135, atol=1e-9)


def test_flat_surface_has_undefined_aspect():
    gx, gy = horn_gradients(np.full((10, 10), 107.0), DX, DY)
    assert np.all(slope_deg(gx, gy) == 0)
    assert np.all(np.isnan(aspect_deg(gx, gy, flat_slope_deg=1.0)))


def test_rotation_is_added_and_wrapped():
    gx, gy = horn_gradients(plane(30, 359), DX, DY)
    np.testing.assert_allclose(aspect_deg(gx, gy, rotation_deg=2.0), 1.0, atol=1e-9)


def test_nan_propagates_only_to_its_8_neighbours():
    # Horn's operator never uses the centre pixel, so one missing pixel blanks the 8
    # pixels around it but not itself.
    z = plane(30, 180)
    z[20, 20] = np.nan
    gx, gy = horn_gradients(z, DX, DY)
    expected = np.zeros(z.shape, bool)
    expected[19:22, 19:22] = True
    expected[20, 20] = False
    np.testing.assert_array_equal(np.isnan(slope_deg(gx, gy)), expected)


def test_edge_modes():
    z = plane(30, 180)
    gx_nan, _ = horn_gradients(z, DX, DY, edge="nan")
    gx_ext, _ = horn_gradients(z, DX, DY, edge="extrapolate")
    assert np.isnan(gx_nan[0]).all() and not np.isnan(gx_ext).any()
    # Interior pixels are identical regardless of edge mode.
    np.testing.assert_array_equal(gx_nan[1:-1, 1:-1], gx_ext[1:-1, 1:-1])
    # Extrapolated edges are exact for a plane (no flattening on the outer ring).
    np.testing.assert_allclose(gx_ext, gx_ext[25, 25])


def test_normals_are_unit_and_match_slope():
    gx, gy = horn_gradients(plane(36, 225), DX, DY)
    n = surface_normals(gx, gy)
    np.testing.assert_allclose(np.linalg.norm(n, axis=0), 1)
    np.testing.assert_allclose(np.degrees(np.arccos(n[2])), 36, atol=1e-9)


def test_horn_is_less_noisy_than_central_difference():
    rng = np.random.default_rng(0)
    sigma = 0.02
    z = rng.normal(0, sigma, (400, 400))
    gx, _ = horn_gradients(z, DX, DY)
    central = (z[:, 2:] - z[:, :-2]) / (2 * DX)
    assert gx.std() == pytest.approx(sigma * np.sqrt(12) / (8 * DX), rel=0.05)
    assert central.std() == pytest.approx(sigma * np.sqrt(2) / (2 * DX), rel=0.05)


def test_grid_convergence_zero_on_central_meridian_and_positive_east():
    crs = CRS.from_epsg(32630)  # UTM 30N, central meridian 3 deg W at easting 500 km
    on_cm = grid_convergence_deg(Affine(0.1, 0, 500000 - 5, 0, -0.1, 5794760), crs, (100, 100))
    east = grid_convergence_deg(Affine(0.1, 0, 640499, 0, -0.1, 5794758), crs, (623, 757))
    assert abs(on_cm) < 1e-3
    assert east == pytest.approx(1.63, abs=0.01)  # ~ (lon - lon0) * sin(lat)


def test_ground_removes_building_and_keeps_slope():
    rows, cols = 400, 400
    ground = plane(2, 90, (rows, cols))
    z = ground.copy()
    z[150:250, 150:250] += 8.0  # 10 m x 10 m building
    dtm = estimate_ground(z, DX, cell_m=1.0, window_m=15.0, smooth_sigma_m=0.0)
    # Opening is only exact for sloping ground more than half a window (7.5 m) from the
    # raster edge; closer than that it is biased low by up to (window / 2) * tan(slope).
    inner = (slice(90, -90), slice(90, -90))
    assert np.abs(dtm - ground)[inner].max() < 0.1


def test_noise_estimate_recovers_sigma_on_plane():
    rng = np.random.default_rng(1)
    z = plane(30, 200, (300, 300)) + rng.normal(0, 0.02, (300, 300))
    assert estimate_noise(z, DX, DY, percentile=50) == pytest.approx(0.02, rel=0.1)


def test_off_grid_mask_finds_reprocessed_block():
    rng = np.random.default_rng(2)
    z = np.round(rng.uniform(100, 110, (100, 100)), 2)  # raw: every height on the 1 cm grid
    z[30:60, 40:80] += rng.uniform(0.001, 0.009, (30, 40))  # re-processed block
    mask = off_grid_mask(z, 0.01, window_px=5, min_density=0.5)
    assert mask[35:55, 45:75].all()
    assert not mask[:25].any() and not mask[65:].any()


def test_planar_tiles_and_scatter():
    z = plane(30, 135, (60, 60))
    tiles = planar_tiles(z, np.ones(z.shape, bool), window=15)
    assert len(tiles) == 16
    z2 = z.copy()
    z2[:, 30:] = z[:, 29:30] - (z[:, 30:] - z[:, 29:30])  # fold into a ridge at column 30
    assert len(planar_tiles(z2, np.ones(z.shape, bool), window=15)) == 16  # ridge on tile border
    gx, gy = horn_gradients(z, DX, DY)
    np.testing.assert_allclose(normal_scatter_deg(surface_normals(gx, gy), tiles), 0, atol=1e-6)
    noisy = z + np.random.default_rng(0).normal(0, 0.01, z.shape)
    gx, gy = horn_gradients(noisy, DX, DY)
    assert normal_scatter_deg(surface_normals(gx, gy), tiles).mean() > 1.0
