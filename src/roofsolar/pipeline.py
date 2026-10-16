"""Pipeline stages. Each ``run_*`` function reads inputs, computes, writes outputs and
returns its results so later stages (and tests) can use them without re-reading files."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.windows import Window
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from roofsolar import cleaning, terrain, viz
from roofsolar.raster_io import Raster, crop, read_raster, window_of, write_raster

log = logging.getLogger(__name__)


@dataclass
class Inputs:
    """Loaded and validated inputs. ``full`` is the larger DSM; ``win`` locates the AOI in it."""

    clip: Raster
    full: Raster
    rgb_full: Raster
    win: Window
    aoi: np.ndarray  # bool mask on the AOI grid
    clip_vs_full_max_diff_m: float = 0.0

    def to_aoi(self, a: np.ndarray) -> np.ndarray:
        """Crop a full-grid array to the AOI grid."""
        r0, c0 = self.win.row_off, self.win.col_off
        return a[..., r0 : r0 + self.win.height, c0 : c0 + self.win.width]

    def aoi_in_full(self) -> np.ndarray:
        """The AOI mask placed on the full grid."""
        m = np.zeros(self.full.shape, bool)
        r0, c0 = self.win.row_off, self.win.col_off
        m[r0 : r0 + self.win.height, c0 : c0 + self.win.width] = self.aoi
        return m


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
    inputs: Inputs | None = None
    ground_px: np.ndarray | None = None  # open flat ground used for noise statistics
    rotation_deg: float = 0.0  # grid -> true north


def load_inputs(cfg: dict[str, Any]) -> Inputs:
    """Read the rasters and check they share one grid and that the clip is a subset."""
    paths = cfg["paths"]
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
    log.info("Grid %s px, pixel %.2f x %.2f m, AOI window %s", full.shape, *full.pixel_size, win)
    return Inputs(clip, full, rgb_full, win, aoi, max_diff)


def surface_derivatives(
    z_full: np.ndarray, inp: Inputs, edge: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Horn gradients of a full-grid surface, cropped to the AOI.

    Returns (gx, gy, slope_deg, normals) on the AOI grid; slope/normals are NaN outside it.
    Computing on the full grid gives AOI-boundary pixels real neighbours.
    """
    dx, dy = inp.full.pixel_size
    gx_full, gy_full = terrain.horn_gradients(z_full, dx, dy, edge=edge)
    gx, gy = inp.to_aoi(gx_full), inp.to_aoi(gy_full)
    slope = np.where(inp.aoi, terrain.slope_deg(gx, gy), np.nan)
    normals = np.where(inp.aoi, terrain.surface_normals(gx, gy), np.nan)
    return gx, gy, slope, normals


def auto_flat_threshold(slope: np.ndarray, ground_px: np.ndarray, setting, pct: float) -> float:
    """Flat-slope threshold: a number from the config, or "auto" = pct of open-ground slope."""
    if setting == "auto":
        return round(float(np.nanpercentile(slope[ground_px], pct)), 2)
    return float(setting)


def run_terrain(cfg: dict[str, Any]) -> TerrainResult:
    """Step 1: elevation, slope, aspect, surface normals and height above ground."""
    paths, tcfg, gcfg, dcfg = cfg["paths"], cfg["terrain"], cfg["ground"], cfg["diagnostics"]
    out = Path(paths["output_dir"]) / "step01_terrain"
    out.mkdir(parents=True, exist_ok=True)

    inp = load_inputs(cfg)
    clip, full, rgb_full, win, aoi = inp.clip, inp.full, inp.rgb_full, inp.win, inp.aoi
    dx, dy = full.pixel_size

    # --- derivatives on the full grid, then crop ------------------------------------
    gx, gy, slope, normals = surface_derivatives(full.data, inp, tcfg["edge_mode"])

    convergence = terrain.grid_convergence_deg(clip.transform, clip.crs, clip.shape)
    rotation = convergence if tcfg["true_north"] else 0.0

    def masked(a: np.ndarray) -> np.ndarray:
        return np.where(aoi, a, np.nan)

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
    flat_deg = auto_flat_threshold(
        slope, ground_px, tcfg["flat_slope_deg"], tcfg["flat_noise_percentile"]
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
            "clipped_vs_full_max_abs_diff_m": inp.clip_vs_full_max_diff_m,
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
        inputs=inp,
        ground_px=ground_px,
        rotation_deg=rotation,
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


@dataclass
class CleaningResult:
    """Step-2 outputs on the AOI grid (NaN outside the AOI)."""

    dsm_clean: Raster
    dsm_clean_full: np.ndarray  # full-grid cleaned surface, for later full-grid derivatives
    slope: Raster
    aspect: Raster
    normals: Raster
    removed: Raster  # raw - clean (what the regulariser took out)
    outliers: np.ndarray  # bool: isolated spikes/pits, replaced before regularising
    filled: np.ndarray  # bool: every pixel whose value was interpolated (NaN + outliers)
    vegetation: np.ndarray  # bool: canopy (ExG above threshold)
    stats: dict[str, Any]


def run_cleaning(cfg: dict[str, Any], prev: TerrainResult | None = None) -> CleaningResult:
    """Step 2: remove isolated outliers, fill missing heights and regularise noise (TGV²).

    1. Missing pixels (NaN) are filled by biharmonic interpolation.
    2. TGV² regularisation on the full grid (AOI-boundary pixels keep real context).
    3. Isolated spikes/pits are detected in (data - TGV); if any, they are treated as
       missing, re-filled and the regularisation is run again on the corrected data.
    4. Slope/aspect/normals are recomputed from the cleaned surface (step-1 code).

    The cleaned surface is meant for *geometry* (faces, normals, planes). Obstruction
    detection (step 4) should compare the *raw* DSM with the fitted planes, so small real
    objects that the regulariser softens are never lost.
    """
    if prev is None or prev.inputs is None:
        prev = run_terrain(cfg)
    inp, tcfg, ccfg = prev.inputs, cfg["terrain"], cfg["cleaning"]
    out = Path(cfg["paths"]["output_dir"]) / "step02_cleaning"
    out.mkdir(parents=True, exist_ok=True)
    dx, dy = inp.full.pixel_size
    z = inp.full.data
    aoi, aoi_full = inp.aoi, inp.aoi_in_full()

    def masked(a: np.ndarray) -> np.ndarray:
        return np.where(aoi, a, np.nan)

    def tgv(f: np.ndarray) -> tuple[np.ndarray, dict]:
        return cleaning.tgv_denoise(
            f,
            ccfg["tgv_alpha1"],
            ccfg["tgv_alpha0"],
            n_iter=ccfg["tgv_max_iter"],
            tol=ccfg["tgv_tol_m"],
        )

    # --- 1-3: fill, regularise, outliers, (re-fill, regularise) -------------------------
    t0 = time.perf_counter()
    missing0 = np.isnan(z)
    f = cleaning.fill_biharmonic(z, missing0)
    u, info = tgv(f)
    passes = [info]
    max_blob_px = int(round(ccfg["outlier_max_blob_m2"] / (dx * dy)))
    outliers, sigma = cleaning.detect_outliers(
        f,
        u,
        aoi_full & ~missing0,
        k=ccfg["outlier_k"],
        window_px=ccfg["outlier_window_px"],
        max_blob_px=max_blob_px,
        background_px=ccfg["outlier_background_px"],
    )
    filled_full = missing0 | outliers
    if outliers.any():
        f = cleaning.fill_biharmonic(z, filled_full)
        u, info = tgv(f)
        passes.append(info)
    seconds = time.perf_counter() - t0
    log.info("Cleaning: %d outliers, %d TGV pass(es), %.1f s", outliers.sum(), len(passes), seconds)

    # --- 4: derivatives of the cleaned surface ------------------------------------------
    gx, gy, slope, normals = surface_derivatives(u, inp, tcfg["edge_mode"])
    flat_deg = auto_flat_threshold(
        slope, prev.ground_px, tcfg["flat_slope_deg"], tcfg["flat_noise_percentile"]
    )
    aspect = masked(terrain.aspect_deg(gx, gy, flat_deg, prev.rotation_deg))
    dsm_clean = masked(inp.to_aoi(u))
    removed = prev.elevation.data - dsm_clean

    # --- vegetation (reliability layer) -------------------------------------------------
    exg = cleaning.excess_green(inp.to_aoi(inp.rgb_full.data))
    exg_thr = ccfg["vegetation_exg_threshold"]
    if exg_thr == "auto":
        exg_thr = float(threshold_otsu(exg[aoi]))
    vegetation = aoi & (exg > exg_thr)

    # --- diagnostics: before vs after ---------------------------------------------------
    outl, filled = inp.to_aoi(outliers), inp.to_aoi(filled_full)
    roof = aoi & (np.nan_to_num(prev.ndsm.data) > cfg["diagnostics"]["roof_min_height_m"])
    roof &= ~vegetation
    tile = ccfg["check_tile_px"]
    tiles = terrain.planar_tiles(prev.elevation.data, roof, tile, ccfg["check_tile_max_resid_m"])
    scatter_before = terrain.normal_scatter_deg(prev.normals.data, tiles, tile)
    scatter_after = terrain.normal_scatter_deg(normals, tiles, tile)
    g = prev.ground_px
    stats = {
        "tgv": {
            "alpha1": ccfg["tgv_alpha1"],
            "alpha0": ccfg["tgv_alpha0"],
            "passes": passes,
            "seconds_total": round(seconds, 1),
        },
        "missing_pixels_in_aoi": int(inp.to_aoi(missing0)[aoi].sum()),
        "outliers": {
            "count": int(outl.sum()),
            "on_roof": int((outl & roof).sum()),
            "on_vegetation": int((outl & vegetation).sum()),
            "elsewhere": int((outl & ~roof & ~vegetation).sum()),
            "max_abs_deviation_m": round(float(np.max(np.abs(removed[outl]))), 3)
            if outl.any()
            else 0.0,
            "residual_robust_sigma_m": round(sigma, 4),
        },
        "filled_pixels_in_aoi": int(filled[aoi].sum()),
        "removed_m": {
            "robust_sigma": round(float(1.4826 * np.median(np.abs(removed[aoi]))), 4),
            "p99_abs": round(float(np.percentile(np.abs(removed[aoi]), 99)), 4),
            "roof_p99_abs": round(float(np.percentile(np.abs(removed[roof]), 99)), 4),
        },
        "open_ground_slope_deg": {
            "before_p50_p95": [
                round(float(v), 2) for v in np.percentile(prev.slope.data[g], [50, 95])
            ],
            "after_p50_p95": [round(float(v), 2) for v in np.percentile(slope[g], [50, 95])],
        },
        "roof_normal_scatter_deg": {
            "tiles": len(tiles),
            "before_median": round(float(np.median(scatter_before)), 2),
            "after_median": round(float(np.median(scatter_after)), 2),
        },
        "flat_slope_threshold_deg": {
            "before": prev.stats["flat_slope_threshold_deg"],
            "after": flat_deg,
        },
        "aspect_undefined_fraction": {
            "before": prev.stats["aspect_undefined_fraction"],
            "after": round(float(np.isnan(aspect[aoi]).mean()), 3),
        },
        "vegetation": {
            "exg_threshold": round(float(exg_thr), 4),
            "share_of_aoi": round(float(vegetation[aoi].mean()), 3),
            "tall_vegetation_m2": round(
                float((vegetation & (np.nan_to_num(prev.ndsm.data) > 2.5)).sum() * dx * dy), 1
            ),
        },
    }

    ref = prev.elevation
    res = CleaningResult(
        dsm_clean=replace(ref, data=dsm_clean),
        dsm_clean_full=u,
        slope=replace(ref, data=slope),
        aspect=replace(ref, data=aspect),
        normals=replace(ref, data=normals),
        removed=replace(ref, data=removed),
        outliers=outl,
        filled=filled,
        vegetation=vegetation,
        stats=stats,
    )
    for name in ("dsm_clean", "slope", "aspect", "normals", "removed"):
        r = getattr(res, name)
        write_raster(out / f"{name}.tif", replace(r, data=r.data.astype("float32")))
    for name in ("outliers", "filled", "vegetation"):
        write_raster(out / f"{name}.tif", replace(ref, data=getattr(res, name).astype("uint8")))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    hs_before = terrain.hillshade(
        *terrain.horn_gradients(prev.elevation.data, dx, dy), azimuth_deg=315 - prev.rotation_deg
    )
    hs_after = terrain.hillshade(gx, gy, azimuth_deg=315 - prev.rotation_deg)
    viz.plot_cleaning(
        prev,
        res,
        hs_before,
        hs_after,
        scatter_before,
        scatter_after,
        cfg["figures"]["profiles"],
        out,
    )
    log.info("Step 2 outputs written to %s", out)
    return res
