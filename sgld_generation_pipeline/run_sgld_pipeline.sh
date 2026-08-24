#!/bin/bash
# Run the notebook-faithful SGLD experiment with the established scientific
# Python environment. A new full-range experiment retrains both neural networks
# and resamples five candidate batches. Rerunning an interrupted experiment
# preserves that run's model, candidates, Rhino files, and evaluation outputs.

set -euo pipefail

# Resolve the repository from this script so the launcher works after cloning.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# A caller may select another compatible Python interpreter without editing
# this file: PYTHON_BIN=/path/to/python bash run_sgld_pipeline.sh
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/Autogluon_env/bin/python}"
PIPELINE_SCRIPT="$PROJECT_ROOT/sgld_generation_pipeline/run_sgld_pipeline.py"

export MPLCONFIGDIR="/private/tmp/ship-structures-matplotlib"
export XDG_CACHE_HOME="/private/tmp/ship-structures-cache"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$PIPELINE_SCRIPT" --stage full
