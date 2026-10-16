"""Reading, validating, aligning and writing georeferenced rasters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window, from_bounds


@dataclass(frozen=True)
class Raster:
    """A raster held in memory together with its georeferencing.

    ``data`` has shape (rows, cols) or (bands, rows, cols). NoData is stored as NaN for
    float rasters, so every downstream operation can use NaN-aware maths.
    """

    data: np.ndarray
    transform: Affine
    crs: rasterio.crs.CRS

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[-2:]

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Pixel size (dx, dy) in CRS units, both positive."""
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def valid(self) -> np.ndarray:
        """True where the pixel holds data (single-band float rasters)."""
        return ~np.isnan(self.data)


def read_raster(path: str | Path, dtype: str = "float64") -> Raster:
    """Read a GeoTIFF, convert it to ``dtype`` and replace NoData with NaN (float only).

    Single-band rasters are returned as 2-D arrays, multi-band rasters as 3-D.
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(dtype)
        if src.nodata is not None and np.issubdtype(arr.dtype, np.floating):
            arr[arr == src.nodata] = np.nan
        if arr.shape[0] == 1:
            arr = arr[0]
        raster = Raster(arr, src.transform, src.crs)
    check_grid(raster)
    return raster


def check_grid(r: Raster) -> None:
    """Fail early if the raster breaks the assumptions of the geometry code.

    The derivative and area maths assume (1) a projected CRS in metres, so pixel size and
    height share units, and (2) a north-up grid with no rotation or shear, so rows run
    south and columns run east.
    """
    if r.crs is None or not r.crs.is_projected:
        raise ValueError(f"Expected a projected CRS, got {r.crs}")
    units = (r.crs.linear_units or "").lower()
    if units not in ("metre", "meter", "m"):
        raise ValueError(f"Expected CRS units in metres, got '{units}'")
    t = r.transform
    if t.b != 0 or t.d != 0:
        raise ValueError("Rotated or sheared rasters are not supported (need north-up)")
    if t.a <= 0 or t.e >= 0:
        raise ValueError("Expected a north-up raster (positive x pixel size, negative y)")


def window_of(inner: Raster, outer: Raster) -> Window:
    """Integer pixel window of ``inner``'s extent inside ``outer``.

    Raises if the two grids are not identical in CRS and resolution or are not aligned
    to whole pixels, because resampling would silently change the data.
    """
    if inner.crs != outer.crs:
        raise ValueError("CRS mismatch")
    if not np.allclose(inner.pixel_size, outer.pixel_size):
        raise ValueError("Pixel size mismatch")
    h, w = inner.shape
    left, top = inner.transform @ (0, 0)
    right, bottom = inner.transform @ (w, h)
    win = from_bounds(left, bottom, right, top, transform=outer.transform)
    offsets = np.array([win.col_off, win.row_off, win.width, win.height])
    if not np.allclose(offsets, np.round(offsets), atol=1e-3):
        raise ValueError(f"Grids are not pixel-aligned: {win}")
    return Window(*(int(v) for v in np.round(offsets)))


def crop(r: Raster, win: Window) -> Raster:
    """Crop to an integer window, updating the transform."""
    r0, c0 = win.row_off, win.col_off
    data = r.data[..., r0 : r0 + win.height, c0 : c0 + win.width]
    return replace(r, data=data, transform=rasterio.windows.transform(win, r.transform))


def write_raster(path: str | Path, r: Raster, nodata: float | None = np.nan) -> None:
    """Write a raster as a compressed GeoTIFF (float rasters use NaN as NoData)."""
    data = r.data if r.data.ndim == 3 else r.data[None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=r.crs,
        transform=r.transform,
        compress="deflate",
    )
    if np.issubdtype(data.dtype, np.floating):
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
