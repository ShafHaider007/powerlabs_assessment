"""Evidence for step 3: accuracy against synthetic truth and stability on the real roof.

A. Synthetic hip roof degraded like the provided DSM (blurred edges, 2.5 cm waviness):
   raw -> TGV (step 2) -> faces (step 3). Reports face recovery, plane angle errors, label
   accuracy and height errors on faces / creases / walls for raw, cleaned, reconstructed.
B. Stability on the real roof: re-run segmentation with 1 mm perturbations (well below the
   2.5 cm waviness) and measure how much the result changes.

Usage:
    python scripts/validate_faces.py [--runs 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

from roofsolar import faces as F
from roofsolar.cleaning import tgv_denoise
from roofsolar.config import PROJECT_ROOT, load_config
from roofsolar.synthetic import hip_roof_scene, realistic_dsm
from roofsolar.terrain import horn_gradients, slope_deg, surface_normals

PX = (0.1, 0.1)


def _mae(a: np.ndarray, truth: np.ndarray, m: np.ndarray) -> float:
    """Median absolute height error (cm) inside mask m."""
    return round(float(np.nanmedian(np.abs(a - truth)[m])) * 100, 2)


def segment(z, cand, params):
    gx, gy = horn_gradients(z, *PX)
    return F.segment_roof(z, surface_normals(gx, gy), slope_deg(gx, gy), cand, PX, params)


def part_a(cfg, seeds=(0, 1, 2)) -> dict:
    ccfg = cfg["cleaning"]
    params = F.SegmentParams(**cfg["faces"]["segment"])
    out = []
    for seed in seeds:
        sc = hip_roof_scene()
        raw = realistic_dsm(sc, np.random.default_rng(seed))
        clean, _ = tgv_denoise(raw, ccfg["tgv_alpha1"], ccfg["tgv_alpha0"], n_iter=4000)
        cand = F.candidate_mask(
            clean - clean[0, 0],
            np.zeros(raw.shape, bool),
            np.ones(raw.shape, bool),
            cfg["faces"]["min_height_m"],
        )
        labels, planes = segment(clean, cand, params)
        rec = F.planar_surface(labels, planes, PX)
        # match true faces to found faces by overlap
        match, ang = {}, []
        for k, n in sc.normals.items():
            ids, c = np.unique(labels[sc.faces == k], return_counts=True)
            c, ids = c[ids > 0], ids[ids > 0]
            if len(ids):
                match[k] = int(ids[np.argmax(c)])
                ang.append(planes[match[k]].angle_to(F.Plane(n, 0)))
        inv = np.zeros(labels.max() + 1, int)
        for k, j in match.items():
            inv[j] = k
        chim = ndi.binary_dilation(sc.chimney, iterations=3)
        roof = (sc.faces > 0) & ~chim
        regions = {
            "faces": roof & ~ndi.binary_dilation(sc.crease, iterations=3) & ~sc.wall,
            "creases": ndi.binary_dilation(sc.crease, iterations=1) & roof & ~sc.wall,
            "walls/eaves (inside footprint)": sc.wall & sc.footprint & ~chim,
        }
        rec_err = {}
        for name, m in regions.items():
            rec_err[name] = {
                label: _mae(a, sc.z, m)
                for label, a in (("raw", raw), ("cleaned", clean), ("reconstructed", rec))
            }
        out.append(
            {
                "seed": seed,
                "faces_found": len(planes),
                "true_faces_matched": len(set(match.values())),
                "plane_angle_err_deg_max": round(float(max(ang)), 3),
                "label_accuracy": round(float((inv[labels[roof]] == sc.faces[roof]).mean()), 4),
                "footprint_iou": round(
                    float(
                        (sc.footprint & (labels > 0)).sum() / (sc.footprint | (labels > 0)).sum()
                    ),
                    4,
                ),
                "median_abs_height_err_cm": rec_err,
            }
        )
    return {"runs": out}


def part_b(cfg, runs: int) -> dict:
    o = Path(cfg["paths"]["output_dir"])
    with rasterio.open(o / "step02_cleaning" / "dsm_clean.tif") as s:
        zc = s.read(1).astype("f8")
    with rasterio.open(o / "step03_faces" / "candidate.tif") as s:
        cand = s.read(1).astype(bool)
    params = F.SegmentParams(**cfg["faces"]["segment"])
    l0, p0 = segment(zc, cand, params)
    big = [k for k in p0 if (l0 == k).sum() * PX[0] * PX[1] >= 15]
    rng = np.random.default_rng(0)
    counts, agree, intact, angles = [], [], [], []
    for _ in range(runs):
        lab, pl = segment(zc + rng.normal(0, 0.001, zc.shape), cand, params)
        counts.append(len(pl))
        same, ok = 0, 0
        for k in p0:
            m = l0 == k
            ids, c = np.unique(lab[m], return_counts=True)
            c, ids = c[ids > 0], ids[ids > 0]
            if not len(ids):
                continue
            same += c.max()
            if k in big:
                ok += c.max() / m.sum() > 0.95
                angles.append(p0[k].angle_to(pl[int(ids[np.argmax(c)])]))
        agree.append(same / (l0 > 0).sum())
        intact.append(int(ok))
    return {
        "perturbation_m": 0.001,
        "runs": runs,
        "reference_faces": len(p0),
        "large_faces_ge_15m2": len(big),
        "face_count_per_run": counts,
        "pixel_agreement_min": round(float(min(agree)), 3),
        "pixel_agreement_mean": round(float(np.mean(agree)), 3),
        "large_faces_intact_per_run": intact,
        "large_face_plane_change_deg_median": round(float(np.median(angles)), 4),
        "large_face_plane_change_deg_max": round(float(max(angles)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--config", default=PROJECT_ROOT / "configs" / "default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_dir"]) / "step03_faces" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    res = {"synthetic": part_a(cfg), "stability_real": part_b(cfg, args.runs)}
    (out / "validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
