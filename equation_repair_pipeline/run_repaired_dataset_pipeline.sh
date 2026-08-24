#!/bin/bash
# Run repaired parameter generation, resumable Rhino CAD generation, exact
# evaluation, publication into MiDShip_Dataset, and t-SNE graph generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/Autogluon_env/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ship-structures-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ship-structures-cache}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" tools/run_repaired_random_design_pipeline.py --stage full
