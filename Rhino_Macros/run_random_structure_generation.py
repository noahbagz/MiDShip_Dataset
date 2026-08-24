#!/usr/bin/env python3
"""Generate the random MiDShip structures with automatic Rhino restarts.

This is Stage 2 for the randomly sampled dataset.  It invokes the shared
Rhino supervisor repeatedly, preserves every per-design parameter checkpoint,
and skips one persistently failing design after three consecutive fresh Rhino
attempts so later designs can continue.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RANDOM_DIR = PROJECT_ROOT / "MiDShip_Dataset" / "Random_Structures"
RHINO_GENERATOR = (
    PROJECT_ROOT
    / "Constraint_Optimization_Pipeline"
    / "batched_structure_generation.py"
)

BATCH_ID = "random_test"
INPUT_CSV = RANDOM_DIR / "random_test_design_Parameters.csv"
UPDATED_PARAMETERS_CSV = RANDOM_DIR / "random_test_design_Parameters_All.csv"
ERROR_INDICES_CSV = RANDOM_DIR / "random_test_design_error_idx_All.csv"
STARTUP_MODEL = RANDOM_DIR / "random_test_design_0.3dm"
SKIPPED_INDICES_FILE = RANDOM_DIR / "skipped_rhino_indices.txt"
STOP_FILE = RANDOM_DIR / "rhino_watchdog.stop"

RHINO_BATCH_SIZE = 25
RESTART_DELAY_SECONDS = 5
MAX_CONSECUTIVE_FAILURES_PER_DESIGN = 3


def rhino_command(print_next_pending=False):
    """Build one invocation of the shared structure supervisor."""

    command = [
        sys.executable,
        str(RHINO_GENERATOR),
        "--batch-id",
        BATCH_ID,
        "--input-csv",
        str(INPUT_CSV),
        "--output-dir",
        str(RANDOM_DIR),
        "--updated-parameters-csv",
        str(UPDATED_PARAMETERS_CSV),
        "--error-indices-csv",
        str(ERROR_INDICES_CSV),
        "--startup-model",
        str(STARTUP_MODEL),
        "--batch-size",
        str(RHINO_BATCH_SIZE),
        "--skip-indices-file",
        str(SKIPPED_INDICES_FILE),
        "--checkpoint-every-design",
    ]

    if print_next_pending:
        command.append("--print-next-pending")

    return command


def next_pending_design():
    """Return the next incomplete, non-skipped random design index."""

    result = subprocess.run(
        rhino_command(print_next_pending=True),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def record_skipped_index(design_index):
    """Durably record a design that failed three fresh Rhino attempts."""

    with SKIPPED_INDICES_FILE.open("a") as handle:
        handle.write("{}\n".format(design_index))
        handle.flush()
        os.fsync(handle.fileno())


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate random MiDShip structures with Rhino restarts."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=RHINO_BATCH_SIZE,
        help="Number of designs submitted to each fresh Rhino process.",
    )
    return parser.parse_args()


def main():
    global RHINO_BATCH_SIZE

    args = parse_arguments()
    RHINO_BATCH_SIZE = args.batch_size

    if not INPUT_CSV.is_file():
        raise RuntimeError(
            "Generate random parameters before starting Rhino: {}".format(
                INPUT_CSV
            )
        )

    SKIPPED_INDICES_FILE.touch(exist_ok=True)
    last_failed_design = None
    consecutive_failures = 0

    for unused_restart_count in itertools.count(1):
        if STOP_FILE.is_file():
            raise RuntimeError(
                "Remove the stop file before resuming: {}".format(STOP_FILE)
            )

        pending_before = next_pending_design()

        if pending_before == "NONE":
            print("All non-skipped random structures are complete.")
            return

        exit_status = subprocess.run(rhino_command(), check=False).returncode
        pending_after = next_pending_design()

        if pending_after == "NONE":
            print("All non-skipped random structures are complete.")
            return

        # A successful batch may leave ordinary later work. Continue directly
        # and reset the failure tracking because the pipeline made progress.
        if exit_status == 0 or pending_after != pending_before:
            last_failed_design = None
            consecutive_failures = 0
            continue

        if pending_after == last_failed_design:
            consecutive_failures += 1
        else:
            last_failed_design = pending_after
            consecutive_failures = 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES_PER_DESIGN:
            record_skipped_index(pending_after)
            print(
                "Skipping random design {} after {} consecutive failures.".format(
                    pending_after,
                    consecutive_failures,
                )
            )
            last_failed_design = None
            consecutive_failures = 0
            continue

        print(
            "Restarting Rhino in {} seconds; completed designs remain "
            "checkpointed.".format(RESTART_DELAY_SECONDS)
        )
        time.sleep(RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    main()
