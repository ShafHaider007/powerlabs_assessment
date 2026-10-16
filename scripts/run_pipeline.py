"""Run the roof reconstruction and solar placement pipeline.

Usage:
    python scripts/run_pipeline.py                          # all implemented steps
    python scripts/run_pipeline.py --steps 1                # only step 1
    python scripts/run_pipeline.py --config my_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from roofsolar.config import PROJECT_ROOT, load_config
from roofsolar.pipeline import run_cleaning, run_faces, run_obstructions, run_terrain

STEPS = {
    1: ("terrain analysis", run_terrain),
    2: ("cleaning", run_cleaning),
    3: ("roof faces", run_faces),
    4: ("obstructions", run_obstructions),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=sorted(STEPS),
        choices=sorted(STEPS),
        help="which steps to run",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    prev = None  # each step receives the previous step's result (re-run if missing)
    for n in sorted(args.steps):
        name, fn = STEPS[n]
        logging.info("=== Step %d: %s ===", n, name)
        result = fn(cfg) if n == 1 else fn(cfg, prev if n - 1 in args.steps else None)
        print(json.dumps(result.stats, indent=2))
        prev = result


if __name__ == "__main__":
    main()
