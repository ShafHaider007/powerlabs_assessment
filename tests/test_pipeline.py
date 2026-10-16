"""End-to-end test: synthetic GeoTIFFs -> step 1 -> step 2, checking outputs against truth."""

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from roofsolar.config import PROJECT_ROOT, load_config
from roofsolar.faces import Plane
from roofsolar.pipeline import run_cleaning, run_faces, run_terrain
from roofsolar.synthetic import correlated_noise, hip_roof_scene, quantize

CRS_UTM = CRS.from_epsg(32630)
T = Affine(0.1, 0, 640000.0, 0, -0.1, 5795000.0)


def _write(path, data, nodata=None, dtype="float32"):
    data = data if data.ndim == 3 else data[None]
    profile = dict(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=dtype,
        crs=CRS_UTM,
        transform=T,
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    sc = hip_roof_scene(size_m=24, length_m=12, width_m=8, eave_m=4, pitch_deg=35)
    rng = np.random.default_rng(0)
    z = quantize(sc.z + 100 + correlated_noise(sc.z.shape, 0.01, 1.0, rng), 0.01)
    z[50, 60] += 0.8  # one spike on the ground
    clip = z.copy()
    clip[:10, :] = -9999  # AOI excludes the top strip
    rgb = np.where(sc.footprint, 100, 60)[None].repeat(3, 0)
    rgb[1] = np.where(sc.footprint, 100, 160)  # green lawn, grey roof
    _write(tmp / "full.tif", z)
    _write(tmp / "clip.tif", clip, nodata=-9999)
    _write(tmp / "rgb.tif", rgb, nodata=0, dtype="uint8")

    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    cfg["paths"] = {
        "dsm": tmp / "clip.tif",
        "dsm_full": tmp / "full.tif",
        "rgb": tmp / "rgb.tif",
        "output_dir": tmp / "out",
    }
    cfg["ground"]["window_m"] = 15.0
    cfg["figures"]["profiles"] = [[120, 120, 44, 20, "A: test"]]
    t = run_terrain(cfg)
    c = run_cleaning(cfg, t)
    f = run_faces(cfg, c)
    return sc, t, c, tmp / "out", f


def test_terrain_recovers_roof_pitch_and_orientation(run):
    _, t, _, _, _ = run
    peaks = [p["azimuth_deg"] for p in t.stats["roof_candidates"]["aspect_peaks_deg"]]
    # 44 deg rotated hip roof -> faces at ~44/134/224/314 grid deg (+ convergence ~0.2)
    for expected in (44, 134, 224, 314):
        assert min(abs((p - expected + 180) % 360 - 180) for p in peaks) < 3
    assert abs(t.stats["roof_candidates"]["slope_deg"]["median"] - 35) < 2


def test_cleaning_removes_spike_and_reduces_noise(run):
    sc, t, c, _, _ = run
    assert c.outliers[50, 60]  # clip and full share one grid here
    assert c.stats["outliers"]["count"] <= 5
    s = c.stats["open_ground_slope_deg"]
    assert s["after_p50_p95"][0] < s["before_p50_p95"][0]
    assert (
        c.stats["roof_normal_scatter_deg"]["after_median"]
        < c.stats["roof_normal_scatter_deg"]["before_median"]
    )
    # cleaned faces are closer to the truth than the raw data
    faces = sc.footprint & ~sc.wall & ~sc.crease & t.aoi
    raw_err = np.abs(t.elevation.data - (sc.z + 100))[faces].mean()
    cln_err = np.abs(c.dsm_clean.data - (sc.z + 100))[faces].mean()
    assert cln_err < raw_err


def test_outputs_written_and_nan_outside_aoi(run):
    _, t, c, out, _ = run
    for f in (
        "dsm_clean.tif",
        "slope.tif",
        "aspect.tif",
        "outliers.tif",
        "stats.json",
        "00_summary.png",
        "01_profiles.png",
    ):
        assert (out / "step02_cleaning" / f).exists()
    assert np.isnan(c.dsm_clean.data[~t.aoi]).all()
    assert not np.isnan(c.dsm_clean.data[t.aoi]).any()


def test_faces_recovered_end_to_end(run):
    sc, t, _, out, f = run
    assert f.stats["faces"] == 4
    for n in sc.normals.values():
        assert min(p.angle_to(Plane(n, 0)) for p in f.planes.values()) < 1.0
    kinds = f.stats["edges"]
    assert kinds["hip"] == 4 and kinds["ridge"] == 1
    for name in ("faces.tif", "planar.tif", "faces.json", "edges.json", "00_summary.png"):
        assert (out / "step03_faces" / name).exists()
