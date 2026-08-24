#!/usr/bin/env python3
"""Compare feasibility, violations, and mean-NN diversity across design sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MIDSHIP_DATASET_DIR = PROJECT_ROOT / "MiDShip_Dataset"
DATASET_DIR = MIDSHIP_DATASET_DIR / "Random_Structures"
RUN_DIR = SCRIPT_DIR / "full_parameter_ranges_retrained"
BATCHES_DIR = RUN_DIR / "batches"
REPAIRED_DIR = MIDSHIP_DATASET_DIR / "Repaired_Structures"
STATISTICS_DIR = RUN_DIR / "statistics"

DATASET_CSV = DATASET_DIR / "Dataset_Design_Data.csv"
REPAIRED_PARAMETER_CSV = (
    REPAIRED_DIR
    / "repaired_random_design_Parameters_Updated.csv"
)
REPAIRED_THRESHOLD_CSV = (
    REPAIRED_DIR / "repaired_random_design_Constraint_Thresholds.csv"
)
REPAIRED_VALUE_CSV = (
    REPAIRED_DIR / "repaired_random_design_Constraint_Values.csv"
)

NUM_PARAMETERS = 120
NUM_PERFORMANCE_COLUMNS = 3
NUM_CONSTRAINTS = 25
NUM_SGLD_BATCHES = 5
NUM_BASELINE_BATCHES = 30
SAMPLE_SIZE = 100
MAX_SUBSET_VIOLATIONS = 13
RANDOM_SEED = 41


def fit_dataset_minmax(parameter_values):
    """Return the original-dataset min/max scaling values and active columns."""

    parameter_minimum = parameter_values.min(axis=0)
    parameter_maximum = parameter_values.max(axis=0)
    nonconstant = parameter_maximum > parameter_minimum

    return parameter_minimum, parameter_maximum, nonconstant


def scale_parameters(
    parameter_values,
    parameter_minimum,
    parameter_maximum,
    nonconstant,
):
    """Scale every parameter to [0, 1] using only original-dataset bounds."""

    scaled = np.zeros_like(parameter_values, dtype=float)
    parameter_range = parameter_maximum - parameter_minimum
    scaled[:, nonconstant] = (
        parameter_values[:, nonconstant]
        - parameter_minimum[nonconstant]
    ) / parameter_range[nonconstant]

    return np.clip(scaled, 0.0, 1.0)


def mean_nearest_neighbor_distance(scaled_parameters):
    """Return mean Euclidean distance from each design to its nearest peer."""

    differences = (
        scaled_parameters[:, None, :]
        - scaled_parameters[None, :, :]
    )
    distances = np.sqrt(
        np.einsum("ijk,ijk->ij", differences, differences)
    )
    np.fill_diagonal(distances, np.inf)

    return float(distances.min(axis=1).mean())


def load_dataset_population():
    """Load the valid 6,020-row dataset and its exact constraint booleans."""

    dataset = pd.read_csv(DATASET_CSV)
    dataset.columns = dataset.columns.str.strip()
    parameter_columns = dataset.columns[:NUM_PARAMETERS].to_list()
    constraint_columns = dataset.columns[
        NUM_PARAMETERS + NUM_PERFORMANCE_COLUMNS:
        NUM_PARAMETERS + NUM_PERFORMANCE_COLUMNS + NUM_CONSTRAINTS
    ].to_list()
    constraint_pass = dataset[constraint_columns].to_numpy(dtype=bool)
    num_failed = (~constraint_pass).sum(axis=1)

    return {
        "parameters": dataset[parameter_columns].to_numpy(dtype=float),
        "fully_feasible": constraint_pass.all(axis=1),
        "num_failed": num_failed,
        "parameter_columns": parameter_columns,
    }


def load_repaired_population(parameter_columns):
    """Load repaired designs for which exact Rhino evaluation succeeded."""

    parameters = pd.read_csv(REPAIRED_PARAMETER_CSV)
    parameters.columns = parameters.columns.str.strip()
    thresholds = pd.read_csv(REPAIRED_THRESHOLD_CSV)
    values = pd.read_csv(REPAIRED_VALUE_CSV)

    # Evaluation-error rows are represented by zero parameters, thresholds,
    # and values in the aligned released tables. Exclude those rows before
    # calculating feasibility or diversity.
    valid_mask = ~(
        (parameters.to_numpy(dtype=float) == 0.0).all(axis=1)
        & (thresholds.to_numpy(dtype=float) == 0.0).all(axis=1)
        & (values.to_numpy(dtype=float) == 0.0).all(axis=1)
    )
    valid_indices = np.flatnonzero(valid_mask)
    passes = (
        values.iloc[valid_indices].to_numpy(dtype=float)
        >= thresholds.iloc[valid_indices].to_numpy(dtype=float)
    )

    return {
        "parameters": parameters.iloc[valid_indices][parameter_columns].to_numpy(
            dtype=float
        ),
        "fully_feasible": passes.all(axis=1),
        "num_failed": (~passes).sum(axis=1),
    }


def sgld_batch_id(batch_number):
    return f"sgld_batch_{batch_number:03d}"


def load_sgld_batch(batch_number, parameter_columns):
    """Load Rhino-updated vectors and exact outcomes for one SGLD batch."""

    current_batch_id = sgld_batch_id(batch_number)
    batch_dir = BATCHES_DIR / current_batch_id
    parameters = pd.read_csv(
        batch_dir
        / "structures"
        / f"{current_batch_id}_X_Results_Updated.csv"
    )
    parameters.columns = parameters.columns.str.strip()
    results = pd.read_csv(
        batch_dir / "evaluation" / "candidate_results.csv"
    )
    valid_indices = results["candidate_index"].to_numpy(dtype=int)

    return {
        "parameters": parameters.iloc[valid_indices][parameter_columns].to_numpy(
            dtype=float
        ),
        "all_parameters": parameters[parameter_columns].to_numpy(dtype=float),
        "fully_feasible": results["fully_feasible"].to_numpy(dtype=bool),
        "num_failed": results["num_failed_constraints"].to_numpy(dtype=int),
        "num_requested": len(parameters),
        "num_evaluated": len(results),
    }


def random_batch_metrics(
    method_name,
    population,
    num_batches,
    rng,
    scaling_values,
):
    """Draw independent 100-design sets and calculate all three metrics."""

    parameter_minimum, parameter_maximum, nonconstant = scaling_values
    metric_rows = []

    for batch_index in tqdm(
        range(num_batches),
        desc=f"Sampling {method_name}",
        unit="batch",
        dynamic_ncols=True,
    ):
        sampled_rows = rng.choice(
            len(population["parameters"]),
            size=SAMPLE_SIZE,
            replace=False,
        )
        scaled = scale_parameters(
            population["parameters"][sampled_rows],
            parameter_minimum,
            parameter_maximum,
            nonconstant,
        )
        metric_rows.append(
            {
                "method": method_name,
                "batch_index": batch_index,
                "sample_size": SAMPLE_SIZE,
                "feasibility_rate": float(
                    population["fully_feasible"][sampled_rows].mean()
                ),
                "mean_num_constraint_violations": float(
                    population["num_failed"][sampled_rows].mean()
                ),
                "mean_nearest_neighbor_distance": (
                    mean_nearest_neighbor_distance(scaled)
                ),
            }
        )

    return pd.DataFrame(metric_rows)


def native_sgld_batch_metrics(parameter_columns, scaling_values):
    """Calculate requested statistics on the five actual generated batches."""

    parameter_minimum, parameter_maximum, nonconstant = scaling_values
    metric_rows = []

    for batch_number in range(1, NUM_SGLD_BATCHES + 1):
        population = load_sgld_batch(batch_number, parameter_columns)
        scaled = scale_parameters(
            population["all_parameters"],
            parameter_minimum,
            parameter_maximum,
            nonconstant,
        )
        num_fully_feasible = int(population["fully_feasible"].sum())
        metric_rows.append(
            {
                "method": "SGLD generated designs",
                "batch_index": batch_number - 1,
                "sample_size": population["num_requested"],
                "num_evaluated": population["num_evaluated"],
                # Count an unevaluated requested design as not feasible, which
                # matches the existing pipeline's conservative rate.
                "feasibility_rate": (
                    num_fully_feasible / population["num_requested"]
                ),
                "mean_num_constraint_violations": float(
                    population["num_failed"].mean()
                ),
                "mean_nearest_neighbor_distance": (
                    mean_nearest_neighbor_distance(scaled)
                ),
            }
        )

    return pd.DataFrame(metric_rows)


def summarize_batches(batch_metrics):
    """Summarize batch means, standard deviations, and two-sigma intervals."""

    metric_columns = [
        "feasibility_rate",
        "mean_num_constraint_violations",
        "mean_nearest_neighbor_distance",
    ]
    summary_rows = []

    for method_name, method_rows in batch_metrics.groupby("method", sort=False):
        summary_row = {
            "method": method_name,
            "num_batches": len(method_rows),
            "batch_size": SAMPLE_SIZE,
        }

        for metric in metric_columns:
            mean = float(method_rows[metric].mean())
            standard_deviation = float(method_rows[metric].std(ddof=1))
            summary_row[f"mean_{metric}"] = mean
            summary_row[f"standard_deviation_{metric}"] = standard_deviation
            summary_row[f"two_sigma_{metric}"] = 2.0 * standard_deviation

        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def latex_mean_two_sigma(mean, two_sigma, decimals):
    return f"${mean:.{decimals}f} \\pm {two_sigma:.{decimals}f}$"


def write_latex_table(summary):
    """Write a paper-ready LaTeX table with mean plus/minus two sigma."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Feasibility and diversity of 100-design samples. Values are reported as the batch mean $\pm 2\sigma$. Mean nearest-neighbor distance is calculated after min--max scaling every design parameter using the original dataset bounds. SGLD feasibility treats exact-evaluation errors as infeasible, while its mean violation count is calculated over successfully evaluated designs.}",
        r"\label{tab:sgld_repair_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Design population & Feasibility rate & Mean violated constraints & Mean nearest-neighbor distance \\",
        r"\midrule",
    ]

    for unused_index, row in summary.iterrows():
        escaped_method = str(row["method"]).replace("<=", r"$\leq$")
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                escaped_method,
                latex_mean_two_sigma(
                    row["mean_feasibility_rate"],
                    row["two_sigma_feasibility_rate"],
                    3,
                ),
                latex_mean_two_sigma(
                    row["mean_mean_num_constraint_violations"],
                    row["two_sigma_mean_num_constraint_violations"],
                    2,
                ),
                latex_mean_two_sigma(
                    row["mean_mean_nearest_neighbor_distance"],
                    row["two_sigma_mean_nearest_neighbor_distance"],
                    3,
                ),
            )
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (STATISTICS_DIR / "method_comparison_table.tex").write_text(
        "\n".join(lines)
    )


def write_constraint_compatibility_audit():
    """Document why the reproduced run differs from its historical expectation."""

    pass_frames = []
    threshold_frames = []
    value_frames = []
    error_frames = []

    for batch_number in range(1, NUM_SGLD_BATCHES + 1):
        evaluation_dir = (
            BATCHES_DIR
            / sgld_batch_id(batch_number)
            / "evaluation"
        )
        pass_frames.append(pd.read_csv(evaluation_dir / "constraint_pass.csv"))
        threshold_frames.append(
            pd.read_csv(evaluation_dir / "constraint_thresholds.csv")
        )
        value_frames.append(
            pd.read_csv(evaluation_dir / "constraint_values.csv")
        )
        error_frames.append(
            pd.read_csv(evaluation_dir / "evaluation_errors.csv")
        )

    passes = pd.concat(pass_frames, ignore_index=True)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    values = pd.concat(value_frames, ignore_index=True)
    errors = pd.concat(error_frames, ignore_index=True)
    constraint_columns = [
        column for column in passes.columns if column != "candidate_index"
    ]
    failed_counts = {
        column: int((~passes[column].astype(bool)).sum())
        for column in constraint_columns
        if int((~passes[column].astype(bool)).sum()) > 0
    }

    # The present rules apply the expanded inner-bottom requirement to both
    # the inner-bottom deck and hopper plate.  This diagnostic does not replace
    # the official result; it shows why the historical 60--70% expectation is
    # recovered only when those two subsequently expanded checks are omitted.
    expanded_constraints = [
        "Inner_Bottom_Deck_Thickness",
        "Hopper_Plate_Thickness",
    ]
    comparison_columns = [
        column
        for column in constraint_columns
        if column not in expanded_constraints
    ]
    feasible_without_expanded_checks = passes[
        comparison_columns
    ].astype(bool).all(axis=1)
    inner_name = "Inner_Bottom_Deck_Thickness"

    audit = {
        "official_current_rule_result": {
            "num_requested": NUM_SGLD_BATCHES * SAMPLE_SIZE,
            "num_evaluated": len(passes),
            "num_evaluation_errors": len(errors),
            "num_fully_feasible": int(
                passes[constraint_columns].astype(bool).all(axis=1).sum()
            ),
            "failed_constraint_counts": failed_counts,
        },
        "systematic_inner_bottom_failure": {
            "generated_value_min_mm": float(values[inner_name].min()),
            "generated_value_max_mm": float(values[inner_name].max()),
            "current_threshold_min_mm": float(thresholds[inner_name].min()),
            "current_threshold_max_mm": float(thresholds[inner_name].max()),
            "notebook_bound_cause": (
                "The active notebook computes xl/xu with "
                "X_data.values[~error_idx]. Because error_idx is an integer "
                "array, this selects 30 negative positional rows and caps "
                "Inner Bottom Thickness at 21 mm."
            ),
        },
        "diagnostic_without_two_expanded_plate_checks": {
            "excluded_constraints": expanded_constraints,
            "num_feasible_over_evaluated": int(
                feasible_without_expanded_checks.sum()
            ),
            "feasibility_rate_over_evaluated": float(
                feasible_without_expanded_checks.mean()
            ),
            "feasibility_rate_over_requested": float(
                feasible_without_expanded_checks.sum()
                / (NUM_SGLD_BATCHES * SAMPLE_SIZE)
            ),
            "status": "diagnostic_only_not_the_reported_feasibility",
        },
        "evaluation_error_types": {
            name: int(count)
            for name, count in errors["error"].str.split(":").str[0].value_counts().items()
        },
    }
    (STATISTICS_DIR / "constraint_compatibility_audit.json").write_text(
        json.dumps(audit, indent=2)
    )

    return audit


def main():
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_population()
    scaling_values = fit_dataset_minmax(dataset["parameters"])
    repaired = load_repaired_population(dataset["parameter_columns"])
    dataset_leq_13_mask = dataset["num_failed"] <= MAX_SUBSET_VIOLATIONS
    dataset_leq_13 = {
        "parameters": dataset["parameters"][dataset_leq_13_mask],
        "fully_feasible": dataset["fully_feasible"][dataset_leq_13_mask],
        "num_failed": dataset["num_failed"][dataset_leq_13_mask],
    }
    rng = np.random.default_rng(RANDOM_SEED)

    batch_frames = [
        random_batch_metrics(
            "Full dataset",
            dataset,
            NUM_BASELINE_BATCHES,
            rng,
            scaling_values,
        ),
        random_batch_metrics(
            "Dataset with <=13 violations",
            dataset_leq_13,
            NUM_BASELINE_BATCHES,
            rng,
            scaling_values,
        ),
        native_sgld_batch_metrics(
            dataset["parameter_columns"],
            scaling_values,
        ),
        random_batch_metrics(
            "Repaired dataset",
            repaired,
            NUM_BASELINE_BATCHES,
            rng,
            scaling_values,
        ),
    ]
    batch_metrics = pd.concat(batch_frames, ignore_index=True)
    summary = summarize_batches(batch_metrics)

    batch_metrics.to_csv(
        STATISTICS_DIR / "batch_metrics.csv",
        index=False,
    )
    summary.to_csv(
        STATISTICS_DIR / "method_comparison_summary.csv",
        index=False,
    )
    batch_frames[2].to_csv(
        STATISTICS_DIR / "sgld_batch_summary.csv",
        index=False,
    )
    write_latex_table(summary)

    sgld_rows = batch_frames[2]
    sgld_summary = {
        "num_batches": NUM_SGLD_BATCHES,
        "designs_requested_per_batch": SAMPLE_SIZE,
        "mean_feasibility_rate": float(sgld_rows["feasibility_rate"].mean()),
        "mean_constraint_violations_per_batch": float(
            sgld_rows["mean_num_constraint_violations"].mean()
        ),
        "mean_nearest_neighbor_distance": float(
            sgld_rows["mean_nearest_neighbor_distance"].mean()
        ),
    }
    (STATISTICS_DIR / "sgld_experiment_summary.json").write_text(
        json.dumps(sgld_summary, indent=2)
    )

    print(summary.to_string(index=False))
    print(json.dumps(sgld_summary, indent=2))


if __name__ == "__main__":
    main()
