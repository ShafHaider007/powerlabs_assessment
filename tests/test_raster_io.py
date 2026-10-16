"""Tests for raster loading, grid checks, alignment and round-tripping."""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from roofsolar.config import PROJECT_ROOT
from roofsolar.raster_io import Raster, check_grid, crop, read_raster, window_of, write_raster

UTM = CRS.from_epsg(32630)
T = Affine(0.1, 0, 640495.0, 0, -0.1, 5794761.9)


def test_rejects_geographic_crs():
    with pytest.raises(ValueError, match="projected"):
        check_grid(Raster(np.zeros((2, 2)), T, CRS.from_epsg(4326)))


def test_rejects_rotated_grid():
    with pytest.raises(ValueError, match="north-up"):
        check_grid(Raster(np.zeros((2, 2)), Affine(0.1, 0.01, 0, 0, -0.1, 0), UTM))


def test_rejects_south_up_grid():
    with pytest.raises(ValueError, match="north-up"):
        check_grid(Raster(np.zeros((2, 2)), Affine(0.1, 0, 0, 0, 0.1, 0), UTM))


def test_window_and_crop_roundtrip():
    outer = Raster(np.arange(100.0).reshape(10, 10), T, UTM)
    inner_t = T @ Affine.translation(3, 2)
    inner = Raster(np.zeros((4, 5)), inner_t, UTM)
    win = window_of(inner, outer)
    assert (win.col_off, win.row_off, win.width, win.height) == (3, 2, 5, 4)
    cropped = crop(outer, win)
    assert cropped.transform == inner_t
    np.testing.assert_array_equal(cropped.data, outer.data[2:6, 3:8])


def test_window_rejects_half_pixel_offset():
    outer = Raster(np.zeros((10, 10)), T, UTM)
    inner = Raster(np.zeros((4, 4)), T @ Affine.translation(0.5, 0), UTM)
    with pytest.raises(ValueError, match="pixel-aligned"):
        window_of(inner, outer)


def test_write_read_roundtrip_keeps_nan_and_georef(tmp_path):
    data = np.array([[1.0, np.nan], [3.0, 4.0]])
    write_raster(tmp_path / "x.tif", Raster(data, T, UTM))
    back = read_raster(tmp_path / "x.tif")
    np.testing.assert_array_equal(back.data, data)
    assert back.transform == T and back.crs == UTM


@pytest.mark.skipif(not (PROJECT_ROOT / "data/raw/dsm_clipped.tif").exists(), reason="no data")
def test_provided_data_is_aligned():
    clip = read_raster(PROJECT_ROOT / "data/raw/dsm_clipped.tif")
    full = read_raster(PROJECT_ROOT / "data/raw/dsm_full.tif")
    win = window_of(clip, full)
    assert (win.col_off, win.row_off) == (40, 36)
    assert np.isnan(clip.data).any()  # -9999 converted to NaN
    sub = crop(full, win).data
    np.testing.assert_array_equal(sub[clip.valid], clip.data[clip.valid])
