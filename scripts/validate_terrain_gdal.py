"""Cross-check our slope/aspect against GDAL's reference implementation (gdaldem).

gdaldem also uses Horn's method by default, so on the same DSM both must agree to
floating-point precision. Comparison is in grid azimuth (no true-north rotation) and on
interior pixels (the two tools handle the outermost raster ring differently).

gdaldem does its arithmetic in float32. With absolute heights of ~110 m that rounds at
~3e-5 m, which shows up as aspect differences of up to ~0.3 deg on near-flat pixels.
Slope and aspect do not change when a constant is added to every height, so the DSM's
minimum is subtracted before running gdaldem. That removes the rounding and leaves a
like-for-like check.

Usage:
    python scripts/validate_terrain_gdal.py [--gdaldem /path/to/gdaldem]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

from roofsolar.config import PROJECT_ROOT, load_config
from roofsolar.raster_io import read_raster
from roofsolar.terrain import aspect_deg, horn_gradients, slope_deg

QGIS_GDALDEM = "/Applications/QGIS.app/Contents/MacOS/gdaldem"


def find_gdaldem(explicit: str | None) -> str | None:
    for cand in (explicit, shutil.which("gdaldem"), QGIS_GDALDEM):
        if cand and Path(cand).exists():
            return cand
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdaldem")
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    gdaldem = find_gdaldem(args.gdaldem)
    if gdaldem is None:
        print("gdaldem not found; install GDAL or pass --gdaldem. Skipping.")
        return 0

    dsm_path = load_config(args.config)["paths"]["dsm_full"]
    dsm = read_raster(dsm_path)
    gx, gy = horn_gradients(dsm.data, *dsm.pixel_size)
    ours_slope, ours_aspect = slope_deg(gx, gy), aspect_deg(gx, gy)

    with tempfile.TemporaryDirectory() as tmp:
        z_path = Path(tmp) / "dsm_offset.tif"
        s_path, a_path = Path(tmp) / "slope.tif", Path(tmp) / "aspect.tif"
        with rasterio.open(dsm_path) as src:
            profile = src.profile
        with rasterio.open(z_path, "w", **profile) as dst:
            dst.write((dsm.data - np.nanmin(dsm.data)).astype(profile["dtype"]), 1)
        subprocess.run([gdaldem, "slope", "-q", str(z_path), str(s_path)], check=True)
        subprocess.run([gdaldem, "aspect", "-q", str(z_path), str(a_path)], check=True)
        with rasterio.open(s_path) as s, rasterio.open(a_path) as a:
            ref_slope, ref_aspect = s.read(1).astype(float), a.read(1).astype(float)

    inner = (slice(1, -1), slice(1, -1))
    ds = np.abs(ours_slope - ref_slope)[inner]
    # Aspect is only meaningful where there is a slope; compare circularly.
    sloped = (ref_slope[inner] > 1.0) & (ref_aspect[inner] >= 0)
    da = np.abs((ours_aspect[inner] - ref_aspect[inner] + 180) % 360 - 180)[sloped]

    print(f"gdaldem: {gdaldem}")
    print(f"pixels compared: {ds.size} (aspect: {da.size} with slope > 1 deg)")
    print(f"slope  |diff| max {ds.max():.2e} deg, mean {ds.mean():.2e} deg")
    print(f"aspect |diff| max {da.max():.2e} deg, mean {da.mean():.2e} deg")
    ok = ds.max() < 1e-4 and da.max() < 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
