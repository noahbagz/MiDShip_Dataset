#!/usr/bin/env python3
"""
Publish the five retrained SGLD structure batches as one indexed dataset.

The local design indices in each source batch are converted into one global
index range:

    sgld_batch_001:   0--99
    sgld_batch_002: 100--199
    sgld_batch_003: 200--299
    sgld_batch_004: 300--399
    sgld_batch_005: 400--499

For example, ``sgld_batch_003_design_17.3dm`` becomes
``sgld_design_217.3dm``. Every companion file for that design receives the
same global index.

Only design artifacts directly inside each ``structures`` folder are copied.
The staged batch files and hidden completion markers remain in place so an
interrupted or audited experiment retains its original provenance.
"""

from __future__ import print_function

import argparse
import csv
import re
import shutil
from pathlib import Path


# Keep all source and destination paths relative to this script so the merge
# behaves consistently regardless of the terminal's current directory.
SGLD_RESULTS_DIR = Path(__file__).resolve().parent
SOURCE_BATCHES_DIR = (
    SGLD_RESULTS_DIR
    / "full_parameter_ranges_retrained"
    / "batches"
)
DESTINATION_DIR = (
    SGLD_RESULTS_DIR.parent
    / "MiDShip_Dataset"
    / "SGLD_Gen_Structures"
)

FIRST_BATCH_NUMBER = 1
LAST_BATCH_NUMBER = 5
DESIGNS_PER_BATCH = 100
REQUIRED_DESIGN_SUFFIXES = (
    "_Structural_Elements.csv",
    "_MeshElements.igs",
    ".3dm",
    ".igs",
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge and globally renumber the five SGLD structure batches."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and validate the publication plan without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching files from an earlier published SGLD run.",
    )

    return parser.parse_args()


def globally_indexed_name(file_name, batch_number):
    """Return the merged filename for one source-batch file."""

    batch_stem = "sgld_batch_{:03d}".format(batch_number)
    design_pattern = re.compile(
        r"^{}_design_(\d+)(.*)$".format(re.escape(batch_stem))
    )
    match = design_pattern.match(file_name)

    # Batch bookkeeping files do not contain a design index and are handled by
    # the separate aggregate-CSV publication script.
    if match is None:
        return file_name, None

    local_idx = int(match.group(1))
    file_suffix = match.group(2)

    if local_idx < 0 or local_idx >= DESIGNS_PER_BATCH:
        raise RuntimeError(
            "Design index {} is outside the expected 0--99 range in {}.".format(
                local_idx,
                file_name,
            )
        )

    global_idx = (batch_number - 1) * DESIGNS_PER_BATCH + local_idx
    merged_name = "sgld_design_{}{}".format(global_idx, file_suffix)

    return merged_name, global_idx


def build_copy_plan(overwrite=False):
    """Collect and validate every staged design artifact to publish."""

    move_plan = []
    planned_destinations = set()
    published_suffixes = {}
    global_error_indices = set()

    for batch_number in range(FIRST_BATCH_NUMBER, LAST_BATCH_NUMBER + 1):
        batch_name = "sgld_batch_{:03d}".format(batch_number)
        structures_dir = SOURCE_BATCHES_DIR / batch_name / "structures"

        if not structures_dir.is_dir():
            raise RuntimeError(
                "Could not find structures directory: {}".format(structures_dir)
            )

        error_path = structures_dir / (
            batch_name + "_design_error_idx_ALL.csv"
        )

        if error_path.is_file():
            with error_path.open("r", newline="") as handle:
                for row in csv.reader(handle):
                    if not row:
                        continue

                    try:
                        local_error_index = int(row[0])
                    except ValueError:
                        continue

                    global_error_indices.add(
                        (batch_number - 1) * DESIGNS_PER_BATCH
                        + local_error_index
                    )

        # Deliberately inspect direct children only and publish design
        # artifacts. Batch-local checkpoints are merged separately.
        for source_path in sorted(structures_dir.iterdir()):
            if not source_path.is_file():
                continue

            destination_name, global_idx = globally_indexed_name(
                source_path.name,
                batch_number,
            )

            if global_idx is None:
                continue

            matched_suffix = next(
                (
                    suffix
                    for suffix in REQUIRED_DESIGN_SUFFIXES
                    if source_path.name.endswith(suffix)
                ),
                None,
            )

            if matched_suffix is None:
                continue

            destination_path = DESTINATION_DIR / destination_name

            if destination_path in planned_destinations:
                raise RuntimeError(
                    "Multiple source files map to {}.".format(destination_path)
                )

            if destination_path.exists() and not overwrite:
                raise RuntimeError(
                    "Destination file already exists; pass --overwrite to "
                    "replace it: {}".format(destination_path)
                )

            planned_destinations.add(destination_path)
            published_suffixes.setdefault(global_idx, set()).add(
                matched_suffix
            )
            move_plan.append(
                {
                    "source": source_path,
                    "destination": destination_path,
                    "global_idx": global_idx,
                }
            )

    expected_design_indices = (
        set(range(LAST_BATCH_NUMBER * DESIGNS_PER_BATCH))
        - global_error_indices
    )
    incomplete_indices = sorted(
        design_index
        for design_index in expected_design_indices
        if published_suffixes.get(design_index, set())
        != set(REQUIRED_DESIGN_SUFFIXES)
    )

    if incomplete_indices:
        raise RuntimeError(
            "Cannot publish a partial SGLD run; {} non-error designs are "
            "incomplete. First indices: {}".format(
                len(incomplete_indices),
                incomplete_indices[:10],
            )
        )

    return move_plan


def main():
    args = parse_arguments()
    move_plan = build_copy_plan(overwrite=args.overwrite)

    design_indices = {
        item["global_idx"]
        for item in move_plan
        if item["global_idx"] is not None
    }
    design_file_count = sum(
        item["global_idx"] is not None
        for item in move_plan
    )

    print("Source batches: {}".format(SOURCE_BATCHES_DIR))
    print("Destination: {}".format(DESTINATION_DIR))
    print("Designs represented: {}".format(len(design_indices)))
    print("Design artifact files: {}".format(design_file_count))

    if args.dry_run:
        print("Dry run complete. No files were copied.")
        return

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

    # Copy only after the complete plan has passed collision and range checks.
    # Staged batches remain intact for resume and provenance.
    for item in move_plan:
        shutil.copy2(
            str(item["source"]),
            str(item["destination"]),
        )

    print("Published {} SGLD design files successfully.".format(len(move_plan)))


if __name__ == "__main__":
    main()
