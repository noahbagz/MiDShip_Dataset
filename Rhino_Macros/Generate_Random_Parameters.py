#!/usr/bin/env python3
"""Generate the randomly sampled MiDShip parametric-design table.

This script performs only Stage 1 of the dataset workflow: parameter-vector
generation.  It does not start Rhino or create structural files.  The sampled
vectors reproduce the rules in ``Structure_3H.gen_rnd_Sturctures`` without
importing Rhino-only modules, so the script can run from an ordinary Python
environment containing NumPy and pandas.

The default output is consumed by ``Batched_Structure_Generation.py``:

    MiDShip_Dataset/Random_Structures/
        random_test_design_Parameters.csv

Example
-------
Generate the paper-sized set from the repository root:

    python Rhino_Macros/Generate_Random_Parameters.py \
        --num-samples 6050 --random-seed 41
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Resolve every default from this file instead of one developer's home folder.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RANDOM_DATASET_DIR = PROJECT_ROOT / "MiDShip_Dataset" / "Random_Structures"
PARAMETER_RANGE_CSV = (
    RANDOM_DATASET_DIR / "StructuralParameterList_V2_Updated_Ranges.csv"
)
DEFAULT_OUTPUT_CSV = (
    RANDOM_DATASET_DIR / "random_test_design_Parameters.csv"
)


# These index groups are the same groups used by Structure_3H.  Keeping them
# explicit makes it clear which values are binary, categorical, or integral.
BIT_INDICES = np.array(
    [27, 31, 35, 39, 43, 47, 51, 56, 74, 79, 86, 91, 96, 100, 104, 105, 111, 115, 116]
)
CATEGORY_INDICES = np.array([57, 58, 59])
INTEGER_INDICES = np.array(
    [15, 16, 17, 18, 19, 20, 21, 22, 70, 75, 82, 87, 92, 106]
)
THICKNESS_INDICES = np.array(
    [7, 8, 9, 10, 11, 12, 23, 29, 33, 37, 41, 45, 49, 52, 54,
     62, 65, 68, 69, 72, 77, 80, 81, 84, 89, 94, 98, 102, 107, 109, 113]
)
BRACKET_INDICES = np.array([60, 61, 63, 64])
CONTAINER_INDICES = np.array(
    [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
)
BULK_CARRIER_INDICES = np.array(
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
)


def clean_random_parameters(parameters, ranges):
    """Apply the same categorical and geometry cleanup used before Rhino."""

    cleaned = parameters.copy()
    lower = ranges["LL"].to_numpy(dtype=float)
    upper = ranges["UL"].to_numpy(dtype=float)
    strategic_lower = ranges["Strategic_LL"].to_numpy(dtype=float)
    strategic_upper = ranges["Strategic_UL"].to_numpy(dtype=float)

    # Rhino interprets these fields as discrete values.  Resolve them before
    # the parameter table is saved so Stage 1 has a stable, readable result.
    cleaned[:, BIT_INDICES] = cleaned[:, BIT_INDICES] >= 0.5
    cleaned[:, INTEGER_INDICES] = np.floor(
        cleaned[:, INTEGER_INDICES] + 0.5
    )
    cleaned[:, THICKNESS_INDICES] = np.floor(
        cleaned[:, THICKNESS_INDICES] + 0.5
    )

    for row_index in range(len(cleaned)):
        row = cleaned[row_index]

        # Exactly one ship class is active for every generated design.
        ship_class = int(np.argmax(row[CATEGORY_INDICES]))
        row[CATEGORY_INDICES] = 0.0
        row[CATEGORY_INDICES[ship_class]] = 1.0

        # The bilge radius cannot exceed the double-bottom height.
        if row[6] < 1000.0 * row[4]:
            row[4] = row[6] / 1000.0

        # Parameters belonging to inactive ship-class subsystems are zeroed.
        if row[57] == 1.0:
            row[BULK_CARRIER_INDICES] = 0.0
            row[CONTAINER_INDICES] = 0.0
            row[BRACKET_INDICES] = np.clip(
                row[BRACKET_INDICES],
                lower[BRACKET_INDICES],
                upper[BRACKET_INDICES],
            )
        elif row[58] == 1.0:
            row[BULK_CARRIER_INDICES] = 0.0
            row[BRACKET_INDICES] = 0.0
            row[66] = np.clip(row[66], lower[66], upper[66])
            row[67] = np.clip(row[67], lower[67], upper[67])
        else:
            row[CONTAINER_INDICES] = 0.0
            row[BRACKET_INDICES] = np.clip(
                row[BRACKET_INDICES],
                strategic_lower[BRACKET_INDICES],
                strategic_upper[BRACKET_INDICES],
            )

        # Match the optional-geometry cleanup in Structure_3H.
        if row[12] < 10.0:
            row[12] = 0.0
        else:
            row[50] = 0.0

        if row[105] <= 0.5:
            row[105] = 0.0
            row[106:116] = 0.0
        elif row[106] <= 0.5:
            row[105:116] = 0.0
        else:
            row[105] = 1.0

        if row[116] <= 0.5 or row[117] < 1.0e-2 or row[118] < 1.0e-2:
            row[116] = 0.0
            row[117:120] = 0.0
        else:
            row[116] = 1.0

    return cleaned


def generate_random_parameters(num_samples, random_seed=None):
    """Sample one complete random parameter table from the published ranges."""

    ranges = pd.read_csv(PARAMETER_RANGE_CSV, encoding="utf-8-sig")
    ranges["name"] = ranges["name"].str.strip()
    lower = ranges["LL"].to_numpy(dtype=float)
    upper = ranges["UL"].to_numpy(dtype=float)
    strategic_lower = ranges["Strategic_LL"].to_numpy(dtype=float)
    strategic_upper = ranges["Strategic_UL"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    parameters = np.zeros((num_samples, len(ranges)), dtype=float)

    # Sample the principal dimensions using the coupled ratios from the
    # original random-design generator.
    parameters[:, 0] = rng.uniform(lower[0], upper[0], num_samples)
    length_scale = rng.uniform(3.0, 4.5, num_samples)
    length_overall = length_scale * parameters[:, 0]
    parameters[:, 1] = length_overall * rng.uniform(
        strategic_lower[1], strategic_upper[1], num_samples
    )
    parameters[:, 3] = parameters[:, 1] * rng.uniform(
        strategic_lower[3], strategic_upper[3], num_samples
    )
    parameters[:, 2] = parameters[:, 3] * rng.uniform(
        strategic_lower[2], strategic_upper[2], num_samples
    )
    parameters[:, 6] = 1000.0 * parameters[:, 1] * rng.uniform(
        strategic_lower[6], strategic_upper[6], num_samples
    )
    parameters[:, 4] = 0.001 * parameters[:, 6] * rng.uniform(
        strategic_lower[4], strategic_upper[4], num_samples
    )
    parameters[:, 5] = rng.uniform(lower[5], upper[5], num_samples)

    # Sample all remaining continuous fields before replacing the discrete
    # fields with their correct binary, categorical, or integer distributions.
    parameters[:, 7:] = rng.uniform(
        lower[7:], upper[7:], (num_samples, len(ranges) - 7)
    )
    parameters[:, BIT_INDICES] = rng.integers(
        0, 2, (num_samples, len(BIT_INDICES))
    )
    parameters[:, 105] = rng.random(num_samples) < 0.20

    ship_classes = rng.integers(0, len(CATEGORY_INDICES), num_samples)
    parameters[:, CATEGORY_INDICES] = 0.0
    parameters[np.arange(num_samples), CATEGORY_INDICES[ship_classes]] = 1.0

    parameters[:, INTEGER_INDICES] = rng.integers(
        lower[INTEGER_INDICES].astype(int),
        upper[INTEGER_INDICES].astype(int) + 1,
        (num_samples, len(INTEGER_INDICES)),
    )
    parameters[:, THICKNESS_INDICES] = rng.integers(
        lower[THICKNESS_INDICES].astype(int),
        upper[THICKNESS_INDICES].astype(int) + 1,
        (num_samples, len(THICKNESS_INDICES)),
    )

    # Tankers and bulk carriers use different allowed bracket ranges.
    for row_index in range(num_samples):
        if parameters[row_index, 57] == 1.0:
            parameters[row_index, BRACKET_INDICES] = rng.uniform(
                lower[BRACKET_INDICES], upper[BRACKET_INDICES]
            )
        elif parameters[row_index, 59] == 1.0:
            parameters[row_index, BRACKET_INDICES] = rng.uniform(
                strategic_lower[BRACKET_INDICES],
                strategic_upper[BRACKET_INDICES],
            )

    parameters = clean_random_parameters(parameters, ranges)
    return pd.DataFrame(parameters, columns=ranges["name"].to_numpy())


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate the random MiDShip parameter-vector dataset."
    )
    parser.add_argument("--num-samples", type=int, default=6050)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated parameter table.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_csv = args.output_csv.resolve()

    if output_csv.exists() and not args.overwrite:
        raise RuntimeError(
            "Output already exists; pass --overwrite to replace it: {}".format(
                output_csv
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    parameters = generate_random_parameters(
        args.num_samples,
        random_seed=args.random_seed,
    )
    parameters.to_csv(output_csv, index=False)

    metadata_path = output_csv.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "method": "Structure_3H random parameter sampling",
                "num_samples": int(args.num_samples),
                "random_seed": args.random_seed,
                "parameter_range_csv": str(PARAMETER_RANGE_CSV),
                "output_csv": str(output_csv),
            },
            indent=2,
        )
    )

    print("Saved {} random parameter vectors to {}.".format(
        len(parameters), output_csv
    ))


if __name__ == "__main__":
    main()
