#!/usr/bin/env python3
"""Generate the repaired MiDShip parameter set without starting Rhino.

The deterministic repair implementation and its published hyperparameters are
maintained in ``tools/run_repaired_random_design_pipeline.py``.  This entry
point exposes the parameter-only first stage from inside the
equation_repair_pipeline folder.
"""

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_repaired_random_design_pipeline import generate_candidates


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate the repaired MiDShip parameter-vector dataset."
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_arguments()
    generate_candidates()
