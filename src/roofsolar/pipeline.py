"""Pipeline stages. Each ``run_*`` function reads inputs, computes, writes outputs and
returns its results so later stages (and tests) can use them without re-reading files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from roofsolar import terrain, viz
from roofsolar.raster_io import Raster, crop, read_raster, window_of, write_raster

log = logging.getLogger(__name__)


@dataclass
class TerrainResult:
    """Step-1 outputs, all on the AOI grid (NaN outside the AOI)."""

    elevation: Raster
    slope: Raster
    aspect: Raster  # degrees clockwise from true (or grid) north
    normals: Raster  # (3, rows, cols): east, north, up
    dtm: Raster
    ndsm: Raster
    rgb: np.ndarray  # (rows, cols, 3) uint8, cropped to the AOI grid
    aoi: np.ndarray  # bool mask
    stats: dict[str, Any]


def run_terrain(cfg: dict[str, Any]) -> TerrainResult:
    """Step 1: elevation, slope, aspect, surface normals and height above ground."""
    paths, tcfg, gcfg, dcfg = cfg["paths"], cfg["terrain"], cfg["ground"], cfg["diagnostics"]
    out = Path(paths["output_dir"]) / "step01_terrain"
    out.mkdir(parents=True, exist_ok=True)

    # --- load and align -------------------------------------------------------------
    clip = read_raster(paths["dsm"])
    full = read_raster(paths["dsm_full"])
    rgb_full = read_raster(paths["rgb"], dtype="uint8")
    win = window_of(clip, full)
    if window_of(rgb_full, full) != window_of(full, full):
        raise ValueError("RGB is not on the same grid as the full DSM")
    aoi = clip.valid

    # The clipped DSM must be a pure subset of the full DSM, otherwise computing the
    # derivatives on the full grid would not describe the clipped data.
    full_in_aoi = crop(full, win).data[aoi]
    max_diff = float(np.max(np.abs(full_in_aoi - clip.data[aoi])))
    if max_diff > 1e-4:
        raise ValueError(f"Clipped DSM differs from full DSM by up to {max_diff:.4f} m")

    dx, dy = full.pixel_size
    log.info("Grid %s px, pixel %.2f x %.2f m, AOI window %s", full.shape, dx, dy, win)

    # --- derivatives on the full grid, then crop ------------------------------------
    gx_full, gy_full = terrain.horn_gradients(full.data, dx, dy, edge=tcfg["edge_mode"])
    gx = crop(replace(full, data=gx_full), win).data
    gy = crop(replace(full, data=gy_full), win).data

    convergence = terrain.grid_convergence_deg(clip.transform, clip.crs, clip.shape)
    rotation = convergence if tcfg["true_north"] else 0.0

    def masked(a: np.ndarray) -> np.ndarray:
        return np.where(aoi, a, np.nan)

    slope = masked(terrain.slope_deg(gx, gy))
    normals = np.where(aoi, terrain.surface_normals(gx, gy), np.nan)

    dtm_full = terrain.estimate_ground(
        full.data, dx, gcfg["cell_m"], gcfg["window_m"], gcfg["smooth_sigma_m"]
    )
    dtm = masked(crop(replace(full, data=dtm_full), win).data)
    ndsm = clip.data - dtm

    # Open flat ground: low nDSM and at least ground_buffer_m away from anything raised,
    # so wall bases and hedge edges (real slopes) do not count as noise.
    ground = aoi & (ndsm < dcfg["ground_max_height_m"])
    buffer_px = int(round(dcfg["ground_buffer_m"] / dx))
    ground_px = ndi.binary_erosion(ground, iterations=buffer_px) if buffer_px else ground
    flat_deg = tcfg["flat_slope_deg"]
    if flat_deg == "auto":
        flat_deg = round(
            float(np.nanpercentile(slope[ground_px], tcfg["flat_noise_percentile"])), 2
        )
    aspect = masked(terrain.aspect_deg(gx, gy, flat_deg, rotation))

    # --- diagnostics -----------------------------------------------------------------
    off_grid_full = terrain.off_grid_mask(
        full.data,
        dcfg["height_quantum_m"],
        dcfg["off_grid_window_px"],
        dcfg["off_grid_min_density"],
    )
    off_grid = crop(replace(full, data=off_grid_full), win).data & aoi

    def noise_in(mask: np.ndarray) -> float | None:
        try:
            return round(
                terrain.estimate_noise(
                    np.where(mask, clip.data, np.nan),
                    dx,
                    dy,
                    dcfg["noise_window_px"],
                    dcfg["noise_percentile"],
                ),
                4,
            )
        except ValueError:  # no complete tile inside the mask
            return None

    noise = noise_in(aoi)
    lo, hi = dcfg["roof_slope_range_deg"]
    roof_px = aoi & (ndsm >= dcfg["roof_min_height_m"]) & (slope >= lo) & (slope <= hi)

    stats = {
        "grid": {
            "shape_rows_cols": list(clip.shape),
            "pixel_size_m": [dx, dy],
            "crs": clip.crs.to_string(),
            "aoi_window_in_full": [win.col_off, win.row_off, win.width, win.height],
            "aoi_pixels": int(aoi.sum()),
            "aoi_area_m2": round(float(aoi.sum() * dx * dy), 1),
            "clipped_vs_full_max_abs_diff_m": max_diff,
        },
        "elevation_m": _summary(clip.data[aoi]),
        "ground_elevation_m": _summary(dtm[aoi]),
        "height_above_ground_m": _summary(ndsm[aoi]),
        "grid_convergence_deg": round(convergence, 3),
        "aspect_reference": "true north" if tcfg["true_north"] else "grid north",
        "dsm_noise_m": noise,
        "data_provenance": {
            "height_quantum_m": dcfg["height_quantum_m"],
            "off_grid_share_of_aoi": round(float(off_grid.sum() / aoi.sum()), 3),
            "off_grid_area_m2": round(float(off_grid.sum() * dx * dy), 1),
            "noise_m_off_grid_regions": noise_in(off_grid),
            "noise_m_raw_regions": noise_in(aoi & ~off_grid),
        },
        "flat_ground_slope_deg": _summary(slope[ground_px]),
        "flat_slope_threshold_deg": flat_deg,
        "flat_slope_threshold_source": (
            f"auto: p{tcfg['flat_noise_percentile']} of open-ground slope"
            if tcfg["flat_slope_deg"] == "auto"
            else "config"
        ),
        "aspect_undefined_fraction": round(float(np.isnan(aspect[aoi]).mean()), 3),
        "roof_candidates": {
            "pixels": int(roof_px.sum()),
            "area_m2_planimetric": round(float(roof_px.sum() * dx * dy), 1),
            "slope_deg": _summary(slope[roof_px]),
            "aspect_peaks_deg": _aspect_peaks(aspect[roof_px]),
        },
    }

    # --- write outputs ------------------------------------------------------------------
    rgb = np.moveaxis(crop(rgb_full, win).data, 0, -1)
    res = TerrainResult(
        elevation=replace(clip, data=masked(clip.data)),
        slope=replace(clip, data=slope),
        aspect=replace(clip, data=aspect),
        normals=replace(clip, data=normals),
        dtm=replace(clip, data=dtm),
        ndsm=replace(clip, data=masked(ndsm)),
        rgb=rgb,
        aoi=aoi,
        stats=stats,
    )
    for name in ("elevation", "slope", "aspect", "normals", "dtm", "ndsm"):
        r = getattr(res, name)
        write_raster(out / f"{name}.tif", replace(r, data=r.data.astype("float32")))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    hs = terrain.hillshade(gx, gy, azimuth_deg=315 - rotation)  # light from true NW
    viz.plot_terrain(res, hs, roof_px, ground_px, out)
    viz.plot_provenance(res, off_grid, out)
    log.info("Step 1 outputs written to %s", out)
    return res


def _summary(values: np.ndarray) -> dict[str, float]:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return {}
    p = np.percentile(v, [1, 5, 50, 95, 99])
    return {
        "min": round(float(v.min()), 3),
        "p1": round(float(p[0]), 3),
        "p5": round(float(p[1]), 3),
        "median": round(float(p[2]), 3),
        "p95": round(float(p[3]), 3),
        "p99": round(float(p[4]), 3),
        "max": round(float(v.max()), 3),
        "count": int(v.size),
    }


def _aspect_peaks(aspect: np.ndarray, bin_deg: float = 5.0, min_share: float = 0.05) -> list:
    """Dominant facing directions: local maxima of a circular aspect histogram.

    The histogram is smoothed with a circular [1, 2, 1] kernel. Peaks holding less than
    ``min_share`` of the pixels are dropped. Each peak's direction is the circular mean
    of the pixels within +-20 deg of it.
    """
    a = aspect[~np.isnan(aspect)]
    if a.size == 0:
        return []
    n_bins = int(360 / bin_deg)
    hist, edges = np.histogram(a, bins=n_bins, range=(0, 360))
    smooth = (np.roll(hist, 1) + 2 * hist + np.roll(hist, -1)) / 4
    peaks = []
    for i in range(n_bins):
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[(i + 1) % n_bins]:
            centre = (edges[i] + edges[i + 1]) / 2
            near = a[np.abs((a - centre + 180) % 360 - 180) <= 20]
            share = near.size / a.size
            if share >= min_share:
                mean = np.degrees(
                    np.arctan2(np.sin(np.radians(near)).mean(), np.cos(np.radians(near)).mean())
                )
                peaks.append(
                    {"azimuth_deg": round(float(mean % 360), 1), "share": round(float(share), 3)}
                )
    return sorted(peaks, key=lambda p: -p["share"])
