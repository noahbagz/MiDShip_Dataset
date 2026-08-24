#!/usr/bin/env python3
"""
Evaluate structural properties and ABS constraints for the MiDShip datasets.

This script reproduces the structural-evaluation calculation in the first
cells of ``Regression_Training_And_Optimization.ipynb`` for the randomly
generated, repaired, and SGLD design sets.

Run all three datasets from the project root with:

    /opt/anaconda3/envs/Autogluon_env/bin/python \
        tools/evaluate_midship_dataset.py

Evaluate only one dataset with:

    /opt/anaconda3/envs/Autogluon_env/bin/python \
        tools/evaluate_midship_dataset.py --dataset repaired

The same command accepts ``random``, ``repaired``, or ``sgld``.  The three
dataset configurations are grouped together below so a repository user can
see and edit every folder, filename stem, and aggregate output stem in one
place.

Each output table retains the complete source index range. Known error indices
and newly encountered evaluation errors receive NaN structural properties and
zero-valued constraint rows, matching the notebook's established convention.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    # Keep evaluation usable in a minimal compatible Python installation when
    # tqdm is unavailable without changing any calculated results.
    def tqdm(iterable, **unused_options):
        description = unused_options.get("desc")

        if description:
            print(description)

        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MIDSHIP_DATASET_DIR = PROJECT_ROOT / "MiDShip_Dataset"

# Make the project-level ``tools`` package importable when this file is run by
# its path from any terminal working directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.Parametric_Structure_Eval import StructureEval


# Retain the exact property names and ordering established by the notebook.
STRUCTURAL_PROPERTY_COLUMNS = [
    "Steel_Volume [m^3]",
    "Steel_Weight [tons]",
    "Unit_Steel_Weight [tons/m^3]",
    "Steel_LCG [m]",
    "Steel_YCG [m]",
    "Steel_VCG [m]",
    "Cross_Sectional_Area [m^2]",
    "z_Centroid_CX [m]",
    "y_Centroid_CX [m]",
    "I_11 [m^4]",
    "I_22 [m^4]",
    "Max_Bending_Moment [Nm]",
]


@dataclass(frozen=True)
class DatasetConfiguration:
    """Describe one aligned parametric-design and structure collection."""

    name: str
    directory: Path
    structure_stem: str
    output_stem: str
    parameter_file: str
    error_index_file: str
    design_data_file: str
    design_type: str


# The structure stem follows the individual structural-elements filenames.
# The output stem follows each dataset's aggregate parameter-table naming.
DATASETS = {
    "random": DatasetConfiguration(
        name="random",
        directory=MIDSHIP_DATASET_DIR / "Random_Structures",
        structure_stem="random_test_design",
        output_stem="random_test_design",
        parameter_file="random_test_design_Parameters_All.csv",
        error_index_file="random_test_design_error_idx_All.csv",
        design_data_file="Dataset_Design_Data.csv",
        design_type="Dataset",
    ),
    "repaired": DatasetConfiguration(
        name="repaired",
        directory=MIDSHIP_DATASET_DIR / "Repaired_Structures",
        structure_stem="repaired_random_design_design",
        output_stem="repaired_random_design",
        parameter_file="repaired_random_design_Parameters_Updated.csv",
        error_index_file="repaired_random_design_design_error_idx_ALL.csv",
        design_data_file="repaired_random_design_Design_Data.csv",
        design_type="Repaired",
    ),
    "sgld": DatasetConfiguration(
        name="sgld",
        directory=MIDSHIP_DATASET_DIR / "SGLD_Gen_Structures",
        structure_stem="sgld_design",
        output_stem="sgld",
        parameter_file="sgld_design_parameters_Updated.csv",
        error_index_file="sgld_design_error_idx_ALL.csv",
        design_data_file="sgld_Design_Data.csv",
        design_type="SGLD",
    ),
}


def read_error_indices(error_path: Path) -> set[int]:
    """Read either a headerless or ``error_idx``-headed index CSV."""

    error_indices = set()

    # A fully successful structural-generation run may not need to create an
    # error file. In that case every row remains eligible for evaluation.
    if not error_path.is_file():
        return error_indices

    with error_path.open("r", newline="") as handle:
        reader = csv.reader(handle)

        for row in reader:
            if not row:
                continue

            try:
                error_indices.add(int(row[0]))
            except ValueError:
                # The repaired and SGLD files include an ``error_idx`` header;
                # the original random-design error file does not.
                continue

    return error_indices


def evaluate_structure(structure_path: Path):
    """Return the notebook's 12 properties and exact constraint dictionaries."""

    structural_elements = pd.read_csv(structure_path)
    evaluator = StructureEval(structural_elements)

    volume = evaluator.Volume()
    weight, unit_weight = evaluator.Structural_Weight()
    volume_centroid = evaluator.Volume_Centroid()
    cross_section = evaluator.Effective_Longitudinal_CrossSection_Properties()

    properties = np.concatenate(
        (
            np.array([volume, weight, unit_weight]),
            np.asarray(volume_centroid),
            np.asarray(cross_section),
        )
    )

    thresholds, values = evaluator.Calculate_Transverse_Struct_Constraints()
    thresholds = dict(thresholds)
    values = dict(values)

    # Bottom-floor spacing is the only maximum constraint. Negating both
    # values converts it to the common passing rule ``value >= threshold``.
    thresholds["Bottom_Floor_Spacing"] *= -1.0
    values["Bottom_Floor_Spacing"] *= -1.0

    return properties, thresholds, values


def output_paths(configuration: DatasetConfiguration):
    """Return the three requested aggregate output paths."""

    output_prefix = configuration.directory / configuration.output_stem

    return {
        "properties": Path(str(output_prefix) + "_Structural_Properties.csv"),
        "thresholds": Path(str(output_prefix) + "_Constraint_Thresholds.csv"),
        "values": Path(str(output_prefix) + "_Constraint_Values.csv"),
    }


def evaluate_dataset(configuration: DatasetConfiguration):
    """Evaluate and save one complete, index-aligned MiDShip design set."""

    # The published parameter table defines the authoritative row count. This
    # keeps evaluation aligned if a future repository user generates a set of
    # a different size while retaining the same naming convention.
    parameters = pd.read_csv(
        configuration.directory / configuration.parameter_file
    )
    num_samples = len(parameters)
    known_error_indices = read_error_indices(
        configuration.directory / configuration.error_index_file
    )

    # NaN property rows and empty constraint records preserve the exact source
    # index of every requested design, including known Rhino failures.
    property_data = np.full(
        (num_samples, len(STRUCTURAL_PROPERTY_COLUMNS)),
        np.nan,
        dtype=float,
    )
    threshold_rows = [{} for _ in range(num_samples)]
    value_rows = [{} for _ in range(num_samples)]
    constraint_columns = None
    evaluation_errors = []

    for design_idx in tqdm(
        range(num_samples),
        total=num_samples,
        desc="Evaluating {} structures".format(configuration.name),
        unit="design",
        dynamic_ncols=True,
    ):
        structure_path = configuration.directory / (
            "{}_{}_Structural_Elements.csv".format(
                configuration.structure_stem,
                design_idx,
            )
        )

        # A persistent error index is skipped only when its structure is still
        # absent. If a later Rhino retry created the file, evaluate it and let
        # the aggregate tables reflect the recovered design.
        if design_idx in known_error_indices and not structure_path.is_file():
            continue

        try:
            properties, thresholds, values = evaluate_structure(structure_path)

            if constraint_columns is None:
                constraint_columns = list(thresholds.keys())

            if list(thresholds.keys()) != constraint_columns:
                raise RuntimeError(
                    "Constraint threshold columns changed at design {}.".format(
                        design_idx
                    )
                )

            if list(values.keys()) != constraint_columns:
                raise RuntimeError(
                    "Constraint value columns changed at design {}.".format(
                        design_idx
                    )
                )

            property_data[design_idx, :] = properties
            threshold_rows[design_idx] = thresholds
            value_rows[design_idx] = values

        except Exception as error:
            # One malformed or missing structure should not stop evaluation of
            # the remaining thousands of designs in the dataset.
            evaluation_errors.append(
                (
                    design_idx,
                    "{}: {}".format(type(error).__name__, error),
                )
            )

    if constraint_columns is None:
        raise RuntimeError(
            "No structures could be evaluated for the {} dataset.".format(
                configuration.name
            )
        )

    structural_properties = pd.DataFrame(
        property_data,
        columns=STRUCTURAL_PROPERTY_COLUMNS,
    )

    # Reindexing creates every constraint column in the consistent order from
    # StructureEval. Empty/error records then become zeros, as in the notebook.
    constraint_thresholds = pd.DataFrame(threshold_rows).reindex(
        columns=constraint_columns
    ).fillna(0.0)
    constraint_values = pd.DataFrame(value_rows).reindex(
        columns=constraint_columns
    ).fillna(0.0)

    paths = output_paths(configuration)
    structural_properties.to_csv(paths["properties"], index=False)
    constraint_thresholds.to_csv(paths["thresholds"], index=False)
    constraint_values.to_csv(paths["values"], index=False)

    # Rebuild the cleaned, analysis-ready table used by the neural-network and
    # plotting scripts. Only designs with a completed exact evaluation enter
    # this table; the three aligned raw tables above retain every source row.
    valid_rows = structural_properties.notna().all(axis=1)
    performance_columns = [
        "Unit_Steel_Weight [tons/m^3]",
        "Cross_Sectional_Area [m^2]",
        "Max_Bending_Moment [Nm]",
    ]
    constraint_pass = constraint_values >= constraint_thresholds
    design_data = pd.concat(
        [
            parameters.loc[valid_rows].reset_index(drop=True),
            structural_properties.loc[
                valid_rows,
                performance_columns,
            ].reset_index(drop=True),
            constraint_pass.loc[valid_rows].reset_index(drop=True),
        ],
        axis=1,
    )
    design_data["Design_Type"] = configuration.design_type
    design_data_path = configuration.directory / configuration.design_data_file
    design_data.to_csv(design_data_path, index=False)

    num_evaluated = int(structural_properties.notna().all(axis=1).sum())
    num_skipped = sum(
        not (
            configuration.directory
            / "{}_{}_Structural_Elements.csv".format(
                configuration.structure_stem,
                design_idx,
            )
        ).is_file()
        for design_idx in known_error_indices
    )

    print(
        "{}: evaluated {}, skipped {} known error indices, and encountered "
        "{} new evaluation errors.".format(
            configuration.name,
            num_evaluated,
            num_skipped,
            len(evaluation_errors),
        )
    )
    print("  Saved {}".format(paths["properties"]))
    print("  Saved {}".format(paths["thresholds"]))
    print("  Saved {}".format(paths["values"]))
    print("  Saved {}".format(design_data_path))

    if evaluation_errors:
        print("  New evaluation errors:")

        for design_idx, message in evaluation_errors:
            print("    {}: {}".format(design_idx, message))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate structural properties and ABS constraints for the "
            "MiDShip datasets."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to evaluate; the default evaluates all three sets.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    selected_datasets = (
        DATASETS.values()
        if args.dataset == "all"
        else [DATASETS[args.dataset]]
    )

    for configuration in selected_datasets:
        evaluate_dataset(configuration)


if __name__ == "__main__":
    main()
