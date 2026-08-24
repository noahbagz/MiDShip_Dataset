#!/usr/bin/env python3
"""Evaluate SGLD Rhino structures with the project's exact ABS constraints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_DIR = SCRIPT_DIR / "full_parameter_ranges_retrained"
BATCHES_DIR = RUN_DIR / "batches"

NUM_BATCHES = 5
POPULATION_SIZE = 100

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


def batch_id(batch_number):
    return f"sgld_batch_{batch_number:03d}"


def evaluate_structure(structure_path):
    """Return exact properties, thresholds, and realized constraint values."""

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
    thresholds = dict(thresholds)
    values = dict(values)

    # Bottom-floor spacing is the sole maximum constraint.  The established
    # evaluation pipeline negates both values so every pass remains value >=
    # threshold.
    thresholds["Bottom_Floor_Spacing"] *= -1.0
    values["Bottom_Floor_Spacing"] *= -1.0

    return properties, thresholds, values


def evaluate_batch(batch_number):
    """Evaluate every available structural-elements file in one 100-row batch."""

    current_batch_id = batch_id(batch_number)
    batch_dir = BATCHES_DIR / current_batch_id
    candidate_csv = batch_dir / f"{current_batch_id}_X_Results.csv"
    structures_dir = batch_dir / "structures"
    evaluation_dir = batch_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

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
        desc=f"Evaluating {current_batch_id}",
        unit="design",
        dynamic_ncols=True,
    ):
        structure_path = structures_dir / (
            f"{current_batch_id}_design_{candidate_index}_Structural_Elements.csv"
        )

        try:
            properties, thresholds, values = evaluate_structure(structure_path)

            if constraint_names is None:
                constraint_names = list(thresholds.keys())

            property_rows.append(
                {
                    "candidate_index": candidate_index,
                    **dict(zip(PROPERTY_COLUMNS, properties)),
                }
            )
            threshold_rows.append(
                {"candidate_index": candidate_index, **thresholds}
            )
            value_rows.append(
                {"candidate_index": candidate_index, **values}
            )

        except Exception as error:
            error_rows.append(
                {
                    "candidate_index": candidate_index,
                    "structure_file": str(structure_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    properties_frame = pd.DataFrame(property_rows)
    thresholds_frame = pd.DataFrame(threshold_rows)
    values_frame = pd.DataFrame(value_rows)
    errors_frame = pd.DataFrame(
        error_rows,
        columns=["candidate_index", "structure_file", "error"],
    )
    constraint_names = constraint_names or []

    indexed_thresholds = thresholds_frame.set_index("candidate_index")
    indexed_values = values_frame.set_index("candidate_index")
    margins = indexed_values[constraint_names] - indexed_thresholds[constraint_names]
    passes = margins >= 0.0
    fully_feasible = passes.all(axis=1)

    candidate_results = pd.DataFrame(
        {
            "candidate_index": indexed_values.index,
            "fully_feasible": fully_feasible.to_numpy(dtype=bool),
            "num_failed_constraints": (~passes).sum(axis=1).to_numpy(dtype=int),
            "minimum_constraint_margin": margins.min(axis=1).to_numpy(dtype=float),
        }
    )

    for constraint_name in constraint_names:
        candidate_results[f"pass_{constraint_name}"] = passes[
            constraint_name
        ].to_numpy(dtype=bool)

    num_evaluated = len(indexed_values)
    num_fully_feasible = int(fully_feasible.sum())
    summary = {
        "batch_id": current_batch_id,
        "num_requested": num_requested,
        "num_evaluated": num_evaluated,
        "num_evaluation_errors": len(error_rows),
        "num_fully_feasible": num_fully_feasible,
        "feasibility_rate_over_requested": num_fully_feasible / num_requested,
        "feasibility_rate_over_evaluated": (
            num_fully_feasible / num_evaluated if num_evaluated else 0.0
        ),
        "mean_num_failed_constraints_over_evaluated": (
            float((~passes).sum(axis=1).mean()) if num_evaluated else None
        ),
        "failed_constraint_counts": {
            name: int((~passes[name]).sum())
            for name in constraint_names
            if int((~passes[name]).sum()) > 0
        },
    }

    properties_frame.to_csv(
        evaluation_dir / "structural_properties.csv",
        index=False,
    )
    thresholds_frame.to_csv(
        evaluation_dir / "constraint_thresholds.csv",
        index=False,
    )
    values_frame.to_csv(
        evaluation_dir / "constraint_values.csv",
        index=False,
    )
    margins.reset_index().to_csv(
        evaluation_dir / "constraint_margins.csv",
        index=False,
    )
    passes.reset_index().to_csv(
        evaluation_dir / "constraint_pass.csv",
        index=False,
    )
    errors_frame.to_csv(
        evaluation_dir / "evaluation_errors.csv",
        index=False,
    )
    candidate_results.to_csv(
        evaluation_dir / "candidate_results.csv",
        index=False,
    )
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate exact constraints for one or all SGLD batches."
    )
    parser.add_argument(
        "--batch-number",
        type=int,
        default=None,
        help="Evaluate one 1-based batch number; default evaluates all five.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    batch_numbers = (
        [args.batch_number]
        if args.batch_number is not None
        else range(1, NUM_BATCHES + 1)
    )

    for batch_number in batch_numbers:
        evaluate_batch(batch_number)


if __name__ == "__main__":
    main()
