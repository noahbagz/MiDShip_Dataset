#!/usr/bin/env python3
"""Launch the shared resumable MiDShip Rhino structure generator.

The maintained supervisor/worker implementation lives in
``Constraint_Optimization_Pipeline/batched_structure_generation.py``.  This
small entry point preserves the familiar Rhino_Macros command while ensuring
that random, repaired, and SGLD generation all use the same restart,
checkpoint, and failed-index behavior.

Select the desired dataset in the clearly marked configuration blocks at the
top of the maintained script, or pass its command-line options explicitly.
"""

from pathlib import Path
import runpy


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SHARED_GENERATOR = (
    PROJECT_ROOT
    / "Constraint_Optimization_Pipeline"
    / "batched_structure_generation.py"
)


if __name__ == "__main__":
    runpy.run_path(str(SHARED_GENERATOR), run_name="__main__")
