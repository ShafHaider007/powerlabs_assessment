"""Terrain derivatives: slope, aspect, surface normals, ground model and noise estimate.

Conventions used throughout the package
---------------------------------------
* Arrays are north-up: row index increases to the south, column index to the east.
* ``gx`` = dz/dx towards the east, ``gy`` = dz/dy towards the north (both dimensionless, m/m).
* Slope is the angle between the surface and the horizontal plane, in degrees [0, 90].
* Aspect is the compass direction the slope *faces* (the steepest-descent direction),
  in degrees clockwise from north [0, 360): 0 = N, 90 = E, 180 = S, 270 = W.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from scipy import ndimage as ndi


def horn_gradients(
    z: np.ndarray, dx: float, dy: float, edge: str = "extrapolate"
) -> tuple[np.ndarray, np.ndarray]:
    """Surface gradients with Horn's (1981) 3x3 finite-difference operator.

    For the 3x3 window::

        a b c        (north is up)
        d e f
        g h i

        dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 dx)
        dz/dy = ((a + 2b + c) - (g + 2h + i)) / (8 dy)

    This is a Sobel operator: a central difference smoothed with (1, 2, 1) weights across
    it. For independent noise of std sigma it gives a gradient error of
    sigma * sqrt(12) / (8 d) ~= 0.43 sigma / d, against 0.71 sigma / d for a plain 2-point
    central difference. It is also what GDAL, QGIS and ArcGIS use, so results can be checked
    against them.

    NaN anywhere in a 3x3 window makes that output pixel NaN (no made-up gradients next to
    missing data). ``edge`` sets what happens at the outer frame of the raster:
    ``"extrapolate"`` linearly extends the surface by one pixel (2*z0 - z1), which is exact
    for planes (the frame is the end of the tile, not missing data). Repeating the edge
    pixels instead would fake a flat strip and halve the gradient on the outer ring.
    ``"nan"`` leaves the outermost ring as NaN.
    """
    if edge == "extrapolate":
        p = np.pad(z, 1, mode="reflect", reflect_type="odd")
    elif edge == "nan":
        p = np.pad(z, 1, mode="constant", constant_values=np.nan)
    else:
        raise ValueError(f"Unknown edge mode '{edge}'")

    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]

    gx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    gy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * dy)
    return gx, gy


def slope_deg(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Slope angle in degrees: arctan(|grad z|)."""
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def aspect_deg(
    gx: np.ndarray,
    gy: np.ndarray,
    flat_slope_deg: float = 0.0,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Aspect (downslope direction) in degrees clockwise from north.

    The downslope direction is -grad z = (-gx east, -gy north), and its compass bearing is
    atan2(east, north). Where slope < ``flat_slope_deg`` the direction is dominated by noise,
    so aspect is NaN (undefined) instead of a misleading random angle.

    ``rotation_deg`` is added to turn grid azimuths into true azimuths (see
    :func:`grid_convergence_deg`).
    """
    asp = (np.degrees(np.arctan2(-gx, -gy)) + rotation_deg) % 360.0
    flat = slope_deg(gx, gy) < flat_slope_deg
    return np.where(flat | np.isnan(gx), np.nan, asp)


def surface_normals(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Unit upward surface normals, shape (3, rows, cols) in (east, north, up).

    For z = f(x, y) the normal is (-dz/dx, -dz/dy, 1) / norm. Normals are smooth across
    north (no 359 deg -> 0 deg jump like aspect), so they suit clustering roof faces.
    """
    n = np.stack([-gx, -gy, np.ones_like(gx)])
    return n / np.linalg.norm(n, axis=0)


def grid_convergence_deg(transform: Affine, crs: CRS, shape: tuple[int, int]) -> float:
    """True bearing of grid north at the raster centre, in degrees (positive = east of true N).

    A projected grid's "up" axis only matches true north on the projection's central
    meridian. Computed numerically: step 1 km up the grid, convert both points to lon/lat,
    and take the initial great-circle bearing between them. Then
    true_azimuth = grid_azimuth + convergence.
    """
    rows, cols = shape
    x0, y0 = transform @ (cols / 2, rows / 2)
    lon, lat = warp_transform(crs, "EPSG:4326", [x0, x0], [y0, y0 + 1000.0])
    phi1, phi2 = np.radians(lat)
    dlon = np.radians(lon[1] - lon[0])
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return float(np.degrees(np.arctan2(y, x)))


def hillshade(
    gx: np.ndarray, gy: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0
) -> np.ndarray:
    """Lambertian hillshade in [0, 1]: cosine of the angle between normal and light direction."""
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    n = surface_normals(gx, gy)
    return np.clip(np.tensordot(light, n, axes=1), 0, 1)


def estimate_ground(
    z: np.ndarray,
    pixel_size: float,
    cell_m: float = 1.0,
    window_m: float = 31.0,
    smooth_sigma_m: float = 2.0,
) -> np.ndarray:
    """Approximate bare-ground surface (DTM) by morphological opening of the DSM.

    1. Down-sample to ``cell_m`` cells using the block *minimum* (keeps ground showing
       through gaps in vegetation and speeds up the large filter).
    2. Grey-scale opening (erosion then dilation) with a ``window_m`` square: removes any
       raised object that this square cannot fit inside (buildings, trees) and keeps the
       slowly varying ground. ``window_m`` must be wider than the largest building.
    3. Gaussian smoothing, then bilinear interpolation back to full resolution.

    Height above ground (nDSM) = DSM - DTM. NaNs are ignored.
    """
    k = max(1, int(round(cell_m / pixel_size)))
    rows, cols = z.shape
    filled = np.where(np.isnan(z), np.nanmax(z), z)  # NaN never wins a minimum
    pad_r, pad_c = (-rows) % k, (-cols) % k
    filled = np.pad(filled, ((0, pad_r), (0, pad_c)), mode="edge")
    blocks = filled.reshape(filled.shape[0] // k, k, filled.shape[1] // k, k).min(axis=(1, 3))

    n = max(1, int(round(window_m / cell_m)))
    ground = ndi.grey_opening(blocks, size=(n, n), mode="nearest")
    if smooth_sigma_m > 0:
        ground = ndi.gaussian_filter(ground, smooth_sigma_m / cell_m, mode="nearest")

    # Block (i, j) is centred on full-res pixel (i*k + (k-1)/2, j*k + (k-1)/2).
    rr, cc = np.mgrid[0:rows, 0:cols].astype(float)
    coords = [(rr - (k - 1) / 2) / k, (cc - (k - 1) / 2) / k]
    return ndi.map_coordinates(ground, coords, order=1, mode="nearest")


def estimate_noise(
    z: np.ndarray, dx: float, dy: float, window: int = 15, percentile: float = 10.0
) -> float:
    """Estimate DSM noise (m) as the residual std of locally planar patches.

    Fits a least-squares plane to every non-overlapping ``window`` x ``window`` tile with
    no NaNs. Most tiles contain real shape (edges, ridges, trees), but the flattest tiles
    are planar, so their residual is sensor noise. The ``percentile`` of residual std over
    all tiles is the noise estimate.
    """
    yy, xx = np.mgrid[0:window, 0:window]
    design = np.c_[xx.ravel() * dx, -yy.ravel() * dy, np.ones(window * window)]
    pinv = np.linalg.pinv(design)
    stds = []
    for r in range(0, z.shape[0] - window + 1, window):
        for c in range(0, z.shape[1] - window + 1, window):
            tile = z[r : r + window, c : c + window].ravel()
            if np.isnan(tile).any():
                continue
            resid = tile - design @ (pinv @ tile)
            stds.append(resid.std())
    if not stds:
        raise ValueError("No complete tiles to estimate noise from")
    return float(np.percentile(stds, percentile))


def off_grid_mask(
    z: np.ndarray,
    quantum_m: float = 0.01,
    window_px: int = 11,
    min_density: float = 0.5,
) -> np.ndarray:
    """Regions whose heights are *not* multiples of ``quantum_m`` (data-provenance check).

    A raw photogrammetric DSM stored at 1 cm precision has every height on the 1 cm grid.
    Pixels that were later resampled, smoothed or interpolated by the data provider get
    off-grid values. Returns the filled regions where the local share of off-grid pixels
    (in a ``window_px`` box) is at least ``min_density``.
    """
    steps = np.round(z / quantum_m, 6)
    off = np.abs(steps - np.round(steps)) > 1e-3
    off &= ~np.isnan(z)
    density = ndi.uniform_filter(off.astype(float), window_px)
    return ndi.binary_fill_holes(density >= min_density)
