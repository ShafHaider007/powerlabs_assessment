"""Evidence for the step-2 method choices.

A. Smoothing: compare filters on a synthetic hip roof with known truth and noise matched
   to the measured DSM statistics (1 cm, lag-1 autocorrelation ~0.78, 1 cm quantisation).
   Each method gets a parameter grid; errors are reported per region (face interiors,
   ridges/hips, walls/eaves, ground).
B. Filling: punch holes into real roof pixels, fill them with each method and compare with
   the original heights (hold-out validation).

Usage:
    python scripts/benchmark_cleaning.py            # both parts
    python scripts/benchmark_cleaning.py --part A
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.fill import fillnodata  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from skimage.restoration import (  # noqa: E402
    denoise_bilateral,
    denoise_tv_chambolle,
    inpaint_biharmonic,
)

from roofsolar.cleaning import tgv_denoise  # noqa: E402
from roofsolar.config import PROJECT_ROOT, load_config  # noqa: E402
from roofsolar.synthetic import correlated_noise, hip_roof_scene, quantize, realistic_dsm  # noqa: E402
from roofsolar.terrain import horn_gradients, surface_normals  # noqa: E402

PX = 0.1


# ------------------------------------------------------------------- A: smoothing
def smoothing_methods():
    """name -> list of (param label, function)."""

    def bilateral(z, sc, ss):
        lo = z.min()
        return denoise_bilateral(z - lo, sigma_color=sc, sigma_spatial=ss, bins=20000) + lo

    m = {"raw": [("-", lambda z: z)]}
    m["gaussian"] = [
        (f"sigma={s}px", lambda z, s=s: ndi.gaussian_filter(z, s)) for s in (0.5, 1, 2, 3, 5)
    ]
    m["median"] = [(f"size={k}", lambda z, k=k: ndi.median_filter(z, k)) for k in (3, 5, 7, 9)]
    m["bilateral"] = [
        (f"sc={sc},ss={ss}", lambda z, sc=sc, ss=ss: bilateral(z, sc, ss))
        for sc in (0.02, 0.05, 0.1)
        for ss in (1, 2, 4)
    ]
    m["tv"] = [
        (f"w={w}", lambda z, w=w: denoise_tv_chambolle(z, weight=w))
        for w in (0.005, 0.01, 0.02, 0.05, 0.1)
    ]
    m["tgv"] = [
        (
            f"a1={a1:g},a0={r * a1:g}",
            lambda z, a1=a1, r=r: tgv_denoise(z, a1, r * a1, n_iter=4000)[0],
        )
        for a1 in (0.01, 0.02, 0.05, 0.1, 0.2)
        for r in (1, 2, 4)
    ]
    return m


def angle_err(u, truth):
    nu = surface_normals(*horn_gradients(u, PX, PX))
    nt = surface_normals(*horn_gradients(truth, PX, PX))
    return np.degrees(np.arccos(np.clip((nu * nt).sum(0), -1, 1)))


def part_a(out: Path, noise: str = "realistic", seed: int = 0) -> dict:
    """noise="simple": 1 cm, 1 px correlated, quantised (the step-1 lawn model).
    noise="realistic": blurred creases + 2.5 cm wavy roof + quantised ground (measured)."""
    sc = hip_roof_scene(pixel=PX)
    rng = np.random.default_rng(seed)
    if noise == "simple":
        noisy = quantize(sc.z + correlated_noise(sc.z.shape, 0.01, 1.0, rng), 0.01)
    else:
        noisy = realistic_dsm(sc, rng)

    crease_zone = ndi.binary_dilation(sc.crease, iterations=2) & sc.footprint & ~sc.wall
    chim = ndi.binary_dilation(sc.chimney, iterations=3)
    faces = sc.footprint & ~ndi.binary_dilation(sc.crease, iterations=3) & ~sc.wall & ~chim
    ground = ~ndi.binary_dilation(sc.footprint, iterations=25)
    regions = {"faces": faces, "creases": crease_zone, "walls": sc.wall, "ground": ground}

    rows = []
    for name, variants in smoothing_methods().items():
        for label, fn in variants:
            t0 = time.perf_counter()
            u = fn(noisy)
            dt = time.perf_counter() - t0
            err = u - sc.z
            ang = angle_err(u, sc.z)
            rec = {"method": name, "params": label, "seconds": round(dt, 2)}
            for r, m in regions.items():
                rec[f"rmse_cm_{r}"] = round(float(np.sqrt(np.mean(err[m] ** 2))) * 100, 2)
            rec["normal_err_deg_faces_median"] = round(float(np.median(ang[faces])), 2)
            rec["normal_err_deg_faces_p95"] = round(float(np.percentile(ang[faces], 95)), 2)
            rows.append(rec)

    # Best variant per method: smooth faces *and* sharp creases (lowest face-normal error in
    # deg + crease RMSE in cm), subject to not degrading the walls/eaves: wall RMSE no
    # worse than the unfiltered input (+0.1 cm). If nothing qualifies, lowest wall error.
    raw_walls = rows[0]["rmse_cm_walls"]

    def score(r):
        return r["normal_err_deg_faces_median"] + r["rmse_cm_creases"]

    best = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        cands = [r for r in rows if r["method"] == name]
        ok = [r for r in cands if r["rmse_cm_walls"] <= raw_walls + 0.1]
        best[name] = min(ok, key=score) if ok else min(cands, key=lambda r: r["rmse_cm_walls"])
    result = {"noise_model": noise, "all": rows, "best": list(best.values())}
    (out / f"benchmark_smoothing_{noise}.json").write_text(json.dumps(result, indent=2))
    _plot_a(sc, noisy, best, out, noise)
    return result


def _plot_a(sc, noisy, best, out, noise):
    fns = {n: dict(v) for n, v in smoothing_methods().items()}
    order = ["raw", "gaussian", "median", "bilateral", "tv", "tgv"]
    # profile across the roof perpendicular to the ridge, through the centre
    t = np.linspace(-9, 9, 721)
    a = np.radians(44.0)
    xs, ys = t * np.sin(a), t * np.cos(a)  # across-ridge direction (v axis)
    cols = (xs + 15) / PX - 0.5
    rows = (15 - ys) / PX - 0.5
    fig, ax = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    truth_p = ndi.map_coordinates(sc.z, [rows, cols], order=1)
    for axi, name in zip(ax.ravel(), order, strict=True):
        b = best[name]
        u = fns[name][b["params"]](noisy)
        p = ndi.map_coordinates(u, [rows, cols], order=1)
        axi.plot(t, truth_p, "k-", lw=2, label="truth")
        axi.plot(t, p, "r-", lw=0.9, label=name)
        axi.set_title(
            f"{name} ({b['params']}): face normal err {b['normal_err_deg_faces_median']}°, "
            f"crease {b['rmse_cm_creases']} cm",
            fontsize=9,
        )
        axi.set_xlim(-7.5, 7.5)
        axi.grid(alpha=0.3)
        ins = axi.inset_axes([0.62, 0.08, 0.35, 0.4])  # zoom on the ridge
        ins.plot(t, truth_p, "k-", lw=2)
        ins.plot(t, p, "r-", lw=1)
        ins.set_xlim(-1, 1)
        ins.set_ylim(truth_p.max() - 0.5, truth_p.max() + 0.08)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("ridge zoom", fontsize=7)
    ax[0, 0].legend(loc="upper left")
    fig.supxlabel("distance across the ridge (m)")
    fig.supylabel("height (m)")
    fig.tight_layout()
    fig.savefig(out / f"benchmark_smoothing_profiles_{noise}.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------- B: filling
def fill_methods(a1: float, a0: float):
    def nearest(f):
        idx = ndi.distance_transform_edt(np.isnan(f), return_distances=False, return_indices=True)
        return f[tuple(idx)]

    def idw(f):  # GDAL's inverse-distance fill
        return fillnodata(
            np.nan_to_num(f),
            mask=np.isfinite(f).astype("uint8"),
            max_search_distance=100,
            smoothing_iterations=0,
        )

    def linear(f):
        ok = np.isfinite(f)
        yy, xx = np.mgrid[: f.shape[0], : f.shape[1]]
        out = f.copy()
        out[~ok] = griddata((yy[ok], xx[ok]), f[ok], (yy[~ok], xx[~ok]), method="linear")
        return out

    def biharmonic(f):
        return inpaint_biharmonic(np.nan_to_num(f), np.isnan(f))

    def tgv(f):
        return tgv_denoise(f, a1, a0, n_iter=4000, tol=1e-7)[0]

    return {
        "nearest": nearest,
        "idw (GDAL)": idw,
        "linear (Delaunay)": linear,
        "biharmonic": biharmonic,
        "tgv": tgv,
    }


def part_b(out: Path, cfg: dict, a1: float, a0: float, seed: int = 1, per_radius: int = 25):
    step1 = Path(cfg["paths"]["output_dir"]) / "step01_terrain"
    with rasterio.open(step1 / "elevation.tif") as s:
        z = s.read(1).astype("f8")
    with rasterio.open(step1 / "ndsm.tif") as s:
        ndsm = s.read(1)
    with rasterio.open(step1 / "slope.tif") as s:
        slope = s.read(1)
    # Candidate centres: roof-like pixels (raised, sloped), at least 1 m from steep edges
    roof = np.nan_to_num(ndsm) > 2.5
    roof &= (np.nan_to_num(slope) > 15) & (np.nan_to_num(slope) < 60)
    roof = ndi.binary_erosion(roof, iterations=5)
    cand = np.argwhere(roof)
    rng = np.random.default_rng(seed)
    methods = fill_methods(a1, a0)
    rows = []
    for radius in (0.3, 0.75, 1.5):
        rpx = radius / PX
        margin = int(rpx + 20)  # 2 m of context around each hole
        picks = cand[rng.choice(len(cand), per_radius, replace=False)]
        for r0, c0 in picks:
            sl = (
                slice(max(r0 - margin, 0), r0 + margin + 1),
                slice(max(c0 - margin, 0), c0 + margin + 1),
            )
            crop = z[sl]
            if np.isnan(crop).any():
                continue
            yy, xx = np.mgrid[sl]
            hole = (yy - r0) ** 2 + (xx - c0) ** 2 <= rpx**2
            f = crop.copy()
            f[hole] = np.nan
            for name, fn in methods.items():
                filled = fn(f)
                e = filled[hole] - crop[hole]
                rows.append(
                    {
                        "radius_m": radius,
                        "method": name,
                        "rmse_cm": float(np.sqrt(np.nanmean(e**2)) * 100),
                        "max_cm": float(np.nanmax(np.abs(e)) * 100),
                    }
                )
    table = {}
    for radius in (0.3, 0.75, 1.5):
        for name in methods:
            v = [r["rmse_cm"] for r in rows if r["radius_m"] == radius and r["method"] == name]
            mx = [r["max_cm"] for r in rows if r["radius_m"] == radius and r["method"] == name]
            table.setdefault(name, {})[f"r={radius}m"] = {
                "rmse_cm_median": round(float(np.median(v)), 2),
                "rmse_cm_p90": round(float(np.percentile(v, 90)), 2),
                "max_cm_median": round(float(np.median(mx)), 2),
                "holes": len(v),
            }
    (out / "benchmark_filling.json").write_text(json.dumps(table, indent=2))
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B"], nargs="*", default=["A", "B"])
    parser.add_argument("--noise", choices=["realistic", "simple"], default="realistic")
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_dir"]) / "step02_cleaning" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)

    a1, a0 = cfg.get("cleaning", {}).get("tgv_alpha1", 0.01), None
    if "A" in args.part:
        res = part_a(out, args.noise)
        print(
            f"{'method':10s} {'params':18s} {'faces°':>7s} {'faces cm':>8s} "
            f"{'crease cm':>9s} {'walls cm':>8s} {'ground cm':>9s} {'s':>5s}"
        )
        for r in res["best"]:
            print(
                f"{r['method']:10s} {r['params']:18s} {r['normal_err_deg_faces_median']:7.2f} "
                f"{r['rmse_cm_faces']:8.2f} {r['rmse_cm_creases']:9.2f} {r['rmse_cm_walls']:8.2f} "
                f"{r['rmse_cm_ground']:9.2f} {r['seconds']:5.2f}"
            )
        tgv_best = next(r for r in res["best"] if r["method"] == "tgv")["params"]
        a1, a0 = (float(kv.split("=")[1]) for kv in tgv_best.split(","))
    if "B" in args.part:
        if a0 is None:
            a0 = cfg["cleaning"]["tgv_alpha0"]
        table = part_b(out, cfg, a1, a0)
        print(f"\nHold-out fill on real roof (TGV a1={a1}, a0={a0}); median RMSE cm [p90]")
        for name, d in table.items():
            cells = "  ".join(
                f"{k}: {v['rmse_cm_median']:6.2f} [{v['rmse_cm_p90']:6.2f}]" for k, v in d.items()
            )
            print(f"{name:18s} {cells}")


if __name__ == "__main__":
    main()
