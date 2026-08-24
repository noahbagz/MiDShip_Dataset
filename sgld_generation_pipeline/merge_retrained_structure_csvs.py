#!/usr/bin/env python3
"""Publish the five SGLD parameter batches as globally aligned CSV files."""

from __future__ import print_function

import csv
import os
from pathlib import Path


SGLD_RESULTS_DIR = Path(__file__).resolve().parent
SOURCE_BATCHES_DIR = (
    SGLD_RESULTS_DIR
    / "full_parameter_ranges_retrained"
    / "batches"
)
PUBLISHED_DIR = (
    SGLD_RESULTS_DIR.parent
    / "MiDShip_Dataset"
    / "SGLD_Gen_Structures"
)

FIRST_BATCH_NUMBER = 1
LAST_BATCH_NUMBER = 5
DESIGNS_PER_BATCH = 100

GENERATED_PARAMETERS_PATH = PUBLISHED_DIR / "sgld_design_parameters.csv"
UPDATED_PARAMETERS_PATH = PUBLISHED_DIR / "sgld_design_parameters_Updated.csv"
MERGED_ERROR_INDICES_PATH = PUBLISHED_DIR / "sgld_design_error_idx_ALL.csv"


def read_csv(file_path):
    """Read one complete CSV while retaining its original string values."""

    with open(file_path, "r", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise RuntimeError("CSV file is empty: {}".format(file_path))

    return rows[0], rows[1:]


def write_csv(file_path, header, rows):
    """Atomically replace one merged CSV after all validation passes."""

    temporary_path = str(file_path) + ".tmp"

    with open(temporary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    os.replace(temporary_path, file_path)


def main():
    parameter_header = None
    generated_parameter_rows = []
    updated_parameter_rows = []
    merged_error_indices = []

    for batch_number in range(FIRST_BATCH_NUMBER, LAST_BATCH_NUMBER + 1):
        batch_name = "sgld_batch_{:03d}".format(batch_number)
        batch_dir = SOURCE_BATCHES_DIR / batch_name
        generated_parameter_path = batch_dir / (
            batch_name + "_X_Results.csv"
        )
        updated_parameter_path = batch_dir / "structures" / (
            batch_name + "_X_Results_Updated.csv"
        )
        error_path = batch_dir / "structures" / (
            batch_name + "_design_error_idx_ALL.csv"
        )

        generated_header, generated_rows = read_csv(
            generated_parameter_path
        )
        updated_header, updated_rows = read_csv(updated_parameter_path)

        # Every batch must contain the same ordered parameter schema and all
        # 100 rows, including zero-filled rows for failed Rhino evaluations.
        if parameter_header is None:
            parameter_header = generated_header
        elif generated_header != parameter_header:
            raise RuntimeError(
                "Parameter headers do not match in {}.".format(
                    generated_parameter_path
                )
            )

        if updated_header != parameter_header:
            raise RuntimeError(
                "Updated parameter headers do not match in {}.".format(
                    updated_parameter_path
                )
            )

        if len(generated_rows) != DESIGNS_PER_BATCH:
            raise RuntimeError(
                "{} contains {} parameter rows; expected {}.".format(
                    generated_parameter_path,
                    len(generated_rows),
                    DESIGNS_PER_BATCH,
                )
            )

        if len(updated_rows) != DESIGNS_PER_BATCH:
            raise RuntimeError(
                "{} contains {} parameter rows; expected {}.".format(
                    updated_parameter_path,
                    len(updated_rows),
                    DESIGNS_PER_BATCH,
                )
            )

        generated_parameter_rows.extend(generated_rows)
        updated_parameter_rows.extend(updated_rows)

        error_header, error_rows = read_csv(error_path)

        if error_header != ["error_idx"]:
            raise RuntimeError(
                "Unexpected error-index header in {}.".format(error_path)
            )

        batch_offset = (batch_number - 1) * DESIGNS_PER_BATCH

        for row in error_rows:
            if not row:
                continue

            local_error_idx = int(row[0])

            if local_error_idx < 0 or local_error_idx >= DESIGNS_PER_BATCH:
                raise RuntimeError(
                    "Error index {} is outside 0--99 in {}.".format(
                        local_error_idx,
                        error_path,
                    )
                )

            merged_error_indices.append(batch_offset + local_error_idx)

    # Publish only after all five batches pass schema and row-count checks.
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        GENERATED_PARAMETERS_PATH,
        parameter_header,
        generated_parameter_rows,
    )
    write_csv(
        UPDATED_PARAMETERS_PATH,
        parameter_header,
        updated_parameter_rows,
    )
    write_csv(
        MERGED_ERROR_INDICES_PATH,
        ["error_idx"],
        [[idx] for idx in sorted(merged_error_indices)],
    )

    print(
        "Merged {} parameter rows into {}.".format(
            len(updated_parameter_rows),
            UPDATED_PARAMETERS_PATH,
        )
    )
    print(
        "Merged {} global error indices into {}: {}".format(
            len(merged_error_indices),
            MERGED_ERROR_INDICES_PATH,
            sorted(merged_error_indices),
        )
    )


if __name__ == "__main__":
    main()
