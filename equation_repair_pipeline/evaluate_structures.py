#!/usr/bin/env python3
"""Evaluate one generated repair batch with the exact structural constraints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"

DEFAULT_BATCH_ID = "diverse_dataset_repair_002"
TARGET_FEASIBILITY_RATE = 0.95

PROPERTY_COLUMNS = [
    "Volume [m^3]",
    "Structural_Weight [tons]",
    "Unit_Steel_Weight [tons/m^3]",
    "Volume_Centroid_X [m]",
    "Volume_Centroid_Y [m]",
    "Volume_Centroid_Z [m]",
    "Cross_Sectional_Area [m^2]",
    "Cross_Section_Centroid_Z [m]",
    "Cross_Section_Centroid_Y [m]",
    "Cross_Section_I_11 [m^4]",
    "Cross_Section_I_22 [m^4]",
    "Max_Bending_Moment [Nm]",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate generated structural CSVs against the exact rule constraints."
    )
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--structures-dir", default=None)
    parser.add_argument("--candidate-csv", default=None)
    return parser.parse_args()


def evaluate_structure(structure_path):
    """Return structural properties, thresholds, and realized values."""

    # Import from the project only after fixing the module search path. This
    # keeps this script runnable from either the repository or pipeline folder.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from tools.Parametric_Structure_Eval import StructureEval

    structural_elements = pd.read_csv(structure_path)
    evaluator = StructureEval(structural_elements)

    volume = evaluator.Volume()
    weight, unit_weight = evaluator.Structural_Weight()
    volume_centroid = evaluator.Volume_Centroid()
    cross_section = evaluator.Effective_Longitudinal_CrossSection_Properties()

    properties = np.concatenate(
        (
            np.array([volume, weight, unit_weight]),
            volume_centroid,
            cross_section,
        )
    )

    thresholds, values = evaluator.Calculate_Transverse_Struct_Constraints()

    # Every rule is stored as a minimum requirement. Bottom floor spacing is
    # the one maximum rule, so reverse both sides before applying the same test.
    thresholds = dict(thresholds)
    values = dict(values)
    thresholds["Bottom_Floor_Spacing"] *= -1.0
    values["Bottom_Floor_Spacing"] *= -1.0

    return properties, thresholds, values


def main():
    args = parse_arguments()
    experiment_dir = EXPERIMENTS_DIR / args.batch_id

    if args.structures_dir is None:
        structures_dir = experiment_dir / "structures"
    else:
        structures_dir = Path(args.structures_dir).resolve()

    if args.candidate_csv is None:
        candidate_csv = experiment_dir / "{}_X_Results.csv".format(args.batch_id)
    else:
        candidate_csv = Path(args.candidate_csv).resolve()

    candidates = pd.read_csv(candidate_csv)
    num_requested = len(candidates)

    property_rows = []
    threshold_rows = []
    value_rows = []
    error_rows = []
    constraint_names = None

    for candidate_index in tqdm(
        range(num_requested),
        total=num_requested,
        desc="Evaluating structures",
        unit="design",
        dynamic_ncols=True,
    ):
        structure_path = structures_dir / (
            "{}_design_{}_Structural_Elements.csv".format(
                args.batch_id,
                candidate_index,
            )
        )

        try:
            properties, thresholds, values = evaluate_structure(structure_path)

            if constraint_names is None:
                constraint_names = list(thresholds.keys())

            property_rows.append(
                dict(candidate_index=candidate_index, **dict(zip(PROPERTY_COLUMNS, properties)))
            )
            threshold_rows.append(
                dict(candidate_index=candidate_index, **thresholds)
            )
            value_rows.append(
                dict(candidate_index=candidate_index, **values)
            )

        except Exception as exc:
            error_rows.append(
                {
                    "candidate_index": candidate_index,
                    "structure_file": str(structure_path),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )

    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    properties_frame = pd.DataFrame(property_rows)
    thresholds_frame = pd.DataFrame(threshold_rows)
    values_frame = pd.DataFrame(value_rows)
    errors_frame = pd.DataFrame(
        error_rows,
        columns=["candidate_index", "structure_file", "error"],
    )

    if constraint_names is None:
        constraint_names = []

    # Align rows on the candidate index before comparing realized values to
    # their requirements. A nonnegative margin denotes a passing constraint.
    indexed_thresholds = thresholds_frame.set_index("candidate_index")
    indexed_values = values_frame.set_index("candidate_index")
    margin_frame = indexed_values[constraint_names] - indexed_thresholds[constraint_names]
    pass_frame = margin_frame >= 0.0

    candidate_results = pd.DataFrame(
        {
            "candidate_index": indexed_values.index,
            "fully_feasible": pass_frame.all(axis=1).to_numpy(dtype=bool),
            "num_failed_constraints": (~pass_frame).sum(axis=1).to_numpy(dtype=int),
            "minimum_constraint_margin": margin_frame.min(axis=1).to_numpy(dtype=float),
        }
    )

    for constraint_name in constraint_names:
        candidate_results["pass_{}".format(constraint_name)] = (
            pass_frame[constraint_name].to_numpy(dtype=bool)
        )

    # Add the repair provenance so seed diversity and feasibility remain
    # inspectable together in one table.
    metadata_path = experiment_dir / "candidate_metadata.csv"

    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        candidate_results = metadata.merge(
            candidate_results,
            on="candidate_index",
            how="left",
        )

    num_evaluated = len(indexed_values)
    num_fully_feasible = int(pass_frame.all(axis=1).sum())
    feasibility_rate_requested = num_fully_feasible / num_requested
    feasibility_rate_evaluated = (
        num_fully_feasible / num_evaluated
        if num_evaluated
        else 0.0
    )

    failed_constraint_counts = {
        name: int((~pass_frame[name]).sum())
        for name in constraint_names
        if int((~pass_frame[name]).sum()) > 0
    }
    summary = {
        "batch_id": args.batch_id,
        "num_requested": num_requested,
        "num_evaluated": num_evaluated,
        "num_evaluation_errors": len(error_rows),
        "num_fully_feasible": num_fully_feasible,
        "feasibility_rate_over_requested": feasibility_rate_requested,
        "feasibility_rate_over_evaluated": feasibility_rate_evaluated,
        "target_feasibility_rate": TARGET_FEASIBILITY_RATE,
        "target_achieved": feasibility_rate_requested >= TARGET_FEASIBILITY_RATE,
        "failed_constraint_counts": failed_constraint_counts,
    }

    properties_frame.to_csv(evaluation_dir / "structural_properties.csv", index=False)
    thresholds_frame.to_csv(evaluation_dir / "constraint_thresholds.csv", index=False)
    values_frame.to_csv(evaluation_dir / "constraint_values.csv", index=False)
    margin_frame.reset_index().to_csv(
        evaluation_dir / "constraint_margins.csv",
        index=False,
    )
    pass_frame.reset_index().to_csv(
        evaluation_dir / "constraint_pass.csv",
        index=False,
    )
    errors_frame.to_csv(evaluation_dir / "evaluation_errors.csv", index=False)
    candidate_results.to_csv(evaluation_dir / "candidate_results.csv", index=False)

    with (evaluation_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
