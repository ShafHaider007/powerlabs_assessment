"""Sensitivity of step 4 to its main thresholds: reference recall vs usable roof area.

Re-runs detection on the step-3 outputs (already on disk) for a grid of colour threshold,
height threshold and keep-out buffer, one parameter at a time around the configured values.

Usage:
    python scripts/validate_obstructions.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scipy import ndimage as ndi

from roofsolar import obstructions as O
from roofsolar.config import PROJECT_ROOT, load_config
from roofsolar.pipeline import run_faces


def evaluate(fr, ocfg, ref) -> dict:
    c = fr.cleaning
    t = c.terrain
    pixel = t.inputs.full.pixel_size
    det = O.detect(t.elevation.data, t.rgb, fr.labels, fr.planes, fr.candidate, pixel, ocfg)
    cls, objects = O.classify(det, fr.labels, c.vegetation, pixel, ocfg)
    _, usable = O.usable_mask(fr.labels, cls > 0, pixel, ocfg["buffer_m"])
    dist = ndi.distance_transform_edt(cls == 0) * pixel[0]
    found = sum(dist[o["row"], o["col"]] <= ref["match_radius_m"] for o in ref["objects"])
    return {
        "recall": f"{found}/{len(ref['objects'])}",
        "objects": len(objects),
        "usable_share": round(float(usable.sum() / (fr.labels > 0).sum()), 3),
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    ref = json.loads(Path(cfg["paths"]["obstruction_reference"]).read_text())
    fr = run_faces(cfg)
    base = cfg["obstructions"]
    grid = {
        "colour_threshold": [4.0, 5.0, 6.0, 7.0],
        "min_height_m": [0.10, 0.15, 0.20, 0.30],
        "buffer_m": [0.0, 0.2, 0.3, 0.5],
    }
    rows = []
    for key, values in grid.items():
        for v in values:
            ocfg = copy.deepcopy(base)
            ocfg[key] = v
            r = evaluate(fr, ocfg, ref)
            rows.append({"parameter": key, "value": v, "default": v == base[key], **r})
            print(
                f"{key:18s} {v:5.2f}{' *' if v == base[key] else '  '}  recall {r['recall']:5s}"
                f"  objects {r['objects']:3d}  usable {100 * r['usable_share']:5.1f}%"
            )
    out = Path(cfg["paths"]["output_dir"]) / "step04_obstructions" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
