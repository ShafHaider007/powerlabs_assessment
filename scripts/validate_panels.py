"""Sensitivity of step 6 to its main assumptions: setback, suitability threshold, panel size.

Re-runs placement on the step-5 result (computed once) for one changed parameter at a time.

Usage:
    python scripts/validate_panels.py
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from roofsolar.config import PROJECT_ROOT, load_config
from roofsolar.pipeline import run_model3d, run_panels

VARIANTS = [
    ("default", {}),
    ("setback 0.2 m", {"edge_setback_m": 0.2}),
    ("setback 0.4 m", {"edge_setback_m": 0.4}),
    ("min relative yield 0.60 (allow NE/NW)", {"min_relative_yield": 0.60}),
    ("min relative yield 0.90", {"min_relative_yield": 0.90}),
    ("60-cell panel 1.65 x 0.99 m, 350 Wp", {"width_m": 0.99, "height_m": 1.65, "power_wp": 350}),
    ("strategy dp (one orientation per face)", {"strategy": "dp"}),
    ("strategy grid", {"strategy": "grid"}),
]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    model = run_model3d(cfg)
    rows = []
    for name, change in VARIANTS:
        c = copy.deepcopy(cfg)
        c["panels"].update(change)
        c["paths"]["output_dir"] = (
            Path(cfg["paths"]["output_dir"]) / "step06_panels" / "validation" / "tmp"
        )
        r = run_panels(c, model)
        s = r.stats
        rows.append(
            {
                "variant": name,
                "panels": s["panels"],
                "kwp": s["kwp"],
                "kwh_year": s["annual_kwh_estimate"],
                "faces_used": sum(f["panels"] > 0 for f in r.faces),
                "checks_ok": s["checks"]["overlaps"] == 0 and s["checks"]["raster_violations"] == 0,
            }
        )
        print(
            f"{name:42s} panels {s['panels']:3d}  {s['kwp']:5.2f} kWp  "
            f"~{s['annual_kwh_estimate']:6d} kWh/yr  faces {rows[-1]['faces_used']}  "
            f"checks ok {rows[-1]['checks_ok']}"
        )
    out = Path(cfg["paths"]["output_dir"]) / "step06_panels" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
