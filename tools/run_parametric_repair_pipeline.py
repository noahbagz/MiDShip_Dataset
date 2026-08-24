#!/usr/bin/env python3
"""Run the deterministic parametric-repair experiment from end to end.

The pipeline deliberately separates its work into resumable stages:

``candidates``
    Select the first 100 valid source designs below the configured initial
    violation limit, repair every violated constraint with
    ``repair_parametric_designs.py``, and save 100 unique repaired parameter
    vectors plus their source provenance and repair distances.

``rhino``
    Use the existing batched Rhino worker to generate four CAD/structure files
    per repaired vector.  Completed designs are discovered before each restart,
    so an interrupted run resumes instead of starting again.

``evaluate``
    Recalculate the true structural properties, thresholds, and constraint
    values from the generated structural-element CSV files.

``tsne``
    Create the existing class/bulkhead and constraint-feasibility t-SNE plots.

``full``
    Run all four stages in order.  Existing completed artifacts are reused.

This script lives in ``tools`` because its candidate-generation logic is a
general rule-based design tool.  It writes experiments into the established
``Constraint_Optimization_Pipeline/experiments`` directory so the existing
Rhino, evaluator, and plotting scripts can operate without modification.
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Ensure imports work whether this file is started from the repository root or
# directly from the tools directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "Constraint_Optimization_Pipeline"
EXPERIMENTS_DIR = PIPELINE_DIR / "experiments"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.repair_parametric_designs import (  # noqa: E402
    CONSTRAINT_NAMES,
    repair_parametric_design,
    violated_constraint_mask,
)


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# Change the batch ID before a new experiment.  The candidate stage will not
# overwrite an existing experiment with different saved settings.
BATCH_ID = "deterministic_parametric_repair_first_100_005"
NUM_SAMPLES = 100

# Only source designs with fewer than 12 initial violations are eligible for
# this experiment.  Storing the inclusive integer maximum makes the selection
# condition explicit: accepted violation counts are 0 through 11.
MAX_INITIAL_CONSTRAINT_VIOLATIONS = 11

# Each violated constraint receives a reproducible target margin based on its
# constraint family.  Direct depth/thickness and spacing repairs use 10--20%,
# while section-modulus repairs use 25--40% to provide more reserve against
# nonlinear section behavior and CAD geometry adjustments.
DEPTH_THICKNESS_EXCEEDANCE_LOWER = 0.10
DEPTH_THICKNESS_EXCEEDANCE_UPPER = 0.35
SECTION_MODULUS_EXCEEDANCE_LOWER = 0.35
SECTION_MODULUS_EXCEEDANCE_UPPER = 0.60
SPACING_EXCEEDANCE_LOWER = 0.20
SPACING_EXCEEDANCE_UPPER = 0.30
RANDOM_SEED = 41

# Candidate vectors must be distinct after every parameter is rounded to two
# decimals.  The source rows are the first 100 eligible aligned dataset rows.
DUPLICATE_DECIMALS = 2

# Rhino is launched in fresh batches.  Zero maximum restarts means the watchdog
# can continue indefinitely; the stop-file mechanism still permits a deliberate
# manual stop.  A repeatedly failing individual item is skipped after three
# consecutive failures so later items can proceed.
RHINO_BATCH_SIZE = 50
RHINO_MAX_RESTARTS = 0
RHINO_RESTART_DELAY_SECONDS = 5
MAX_CONSECUTIVE_FAILURES_PER_DESIGN = 3

# Plot settings match the existing generated-versus-dataset t-SNE workflow.
TSNE_PERPLEXITY = 45.0
TSNE_CATEGORICAL_WEIGHT = 2.5


# =============================================================================
# INPUT DATA AND EXISTING WORKERS
# =============================================================================

DATASET_DIR = (
    PROJECT_ROOT / "MiDShip_Dataset" / "Random_Structures"
)
PARAMETER_CSV = DATASET_DIR / "random_test_design_Parameters_All.csv"
THRESHOLD_CSV = DATASET_DIR / "random_test_design_Constraint_Thresholds.csv"
VALUE_CSV = DATASET_DIR / "random_test_design_Constraint_Values.csv"

RHINO_GENERATOR = PIPELINE_DIR / "batched_structure_generation.py"
EXACT_EVALUATOR = PIPELINE_DIR / "evaluate_structures.py"
TSNE_SCRIPT = PIPELINE_DIR / "plot_tsne_comparison.py"


def file_sha256(file_path):
    """Return a stable source-file fingerprint for the experiment manifest."""

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def experiment_settings(batch_id):
    """Return the complete candidate-generation and execution contract."""

    return {
        "batch_id": batch_id,
        "method": "deterministic_equation_based_parametric_repair",
        "num_samples": NUM_SAMPLES,
        "maximum_initial_constraint_violations": (
            MAX_INITIAL_CONSTRAINT_VIOLATIONS
        ),
        "depth_thickness_exceedance_lower": (
            DEPTH_THICKNESS_EXCEEDANCE_LOWER
        ),
        "depth_thickness_exceedance_upper": (
            DEPTH_THICKNESS_EXCEEDANCE_UPPER
        ),
        "section_modulus_exceedance_lower": (
            SECTION_MODULUS_EXCEEDANCE_LOWER
        ),
        "section_modulus_exceedance_upper": (
            SECTION_MODULUS_EXCEEDANCE_UPPER
        ),
        "spacing_exceedance_lower": SPACING_EXCEEDANCE_LOWER,
        "spacing_exceedance_upper": SPACING_EXCEEDANCE_UPPER,
        "random_seed": RANDOM_SEED,
        "duplicate_decimals": DUPLICATE_DECIMALS,
        "source_sampling": (
            "first_valid_rows_in_dataset_order_with_initial_violations_"
            "less_than_12"
        ),
        "zero_error_rows": "excluded_when_thresholds_and_values_are_all_zero",
        "rhino_batch_size": RHINO_BATCH_SIZE,
        "rhino_max_restarts": RHINO_MAX_RESTARTS,
        "rhino_restart_delay_seconds": RHINO_RESTART_DELAY_SECONDS,
        "max_consecutive_failures_per_design": (
            MAX_CONSECUTIVE_FAILURES_PER_DESIGN
        ),
        "tsne_perplexity": TSNE_PERPLEXITY,
        "tsne_categorical_weight": TSNE_CATEGORICAL_WEIGHT,
        "parameter_csv": str(PARAMETER_CSV),
        "threshold_csv": str(THRESHOLD_CSV),
        "value_csv": str(VALUE_CSV),
        "parameter_csv_sha256": file_sha256(PARAMETER_CSV),
        "threshold_csv_sha256": file_sha256(THRESHOLD_CSV),
        "value_csv_sha256": file_sha256(VALUE_CSV),
    }


def write_json(file_path, contents):
    """Write a JSON artifact atomically so interrupted writes remain valid."""

    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with temporary_path.open("w") as handle:
        json.dump(contents, handle, indent=2)

    os.replace(temporary_path, file_path)


def confirm_or_create_experiment(experiment_dir, batch_id):
    """Create a new experiment or confirm that saved settings still match."""

    settings = experiment_settings(batch_id)
    settings_path = experiment_dir / "pipeline_settings.json"

    if settings_path.is_file():
        with settings_path.open() as handle:
            saved_settings = json.load(handle)

        if saved_settings != settings:
            raise RuntimeError(
                "The existing experiment uses different settings: {}".format(
                    settings_path
                )
            )

        return

    experiment_dir.mkdir(parents=True, exist_ok=False)
    write_json(settings_path, settings)


def load_aligned_source_data():
    """Load the three aligned source tables and identify valid source rows."""

    parameters = pd.read_csv(PARAMETER_CSV)
    thresholds = pd.read_csv(THRESHOLD_CSV)
    values = pd.read_csv(VALUE_CSV)

    # Error rows were intentionally saved as all zeros in both constraint
    # tables.  Exclude them without reading or depending on an error-index file.
    threshold_is_zero = (thresholds.to_numpy(dtype=float) == 0.0).all(axis=1)
    value_is_zero = (values.to_numpy(dtype=float) == 0.0).all(axis=1)
    valid_source_rows = np.flatnonzero(
        ~(threshold_is_zero & value_is_zero)
    )

    return parameters, thresholds, values, valid_source_rows


def generate_candidates(batch_id):
    """Repair and save the first 100 valid, low-violation source vectors."""

    experiment_dir = EXPERIMENTS_DIR / batch_id
    candidate_csv = experiment_dir / "{}_X_Results.csv".format(batch_id)

    # Candidate generation is immutable once complete.  Later stages can resume
    # safely from the saved table and settings.
    if candidate_csv.is_file():
        print("Candidate CSV already exists; skipping candidate generation.")
        return experiment_dir

    confirm_or_create_experiment(experiment_dir, batch_id)
    parameters, thresholds, values, valid_source_rows = (
        load_aligned_source_data()
    )

    rng = np.random.default_rng(RANDOM_SEED)

    # Count source violations with the same active-constraint logic used by the
    # repair method.  Preserve dataset order after filtering so this remains a
    # reproducible "first 100" experiment rather than a ranked selection.
    source_violation_counts = np.asarray(
        [
            violated_constraint_mask(
                thresholds.iloc[source_row].to_numpy(dtype=float),
                values.iloc[source_row].to_numpy(dtype=float),
            ).sum()
            for source_row in valid_source_rows
        ],
        dtype=int,
    )
    eligible_source_rows = valid_source_rows[
        source_violation_counts <= MAX_INITIAL_CONSTRAINT_VIOLATIONS
    ]

    if len(eligible_source_rows) < NUM_SAMPLES:
        raise RuntimeError(
            "Only {} valid source designs have {} or fewer initial "
            "constraint violations; {} are required.".format(
                len(eligible_source_rows),
                MAX_INITIAL_CONSTRAINT_VIOLATIONS,
                NUM_SAMPLES,
            )
        )

    source_order = eligible_source_rows[:NUM_SAMPLES]
    repaired_rows = []
    metadata_rows = []
    selected_source_rows = []
    selected_threshold_rows = []
    selected_value_rows = []
    existing_candidate_keys = set()
    duplicate_candidates_rejected = 0

    # Repair the first 100 valid rows in their original dataset order.  If two
    # repaired results collapse to the same vector at two-decimal precision,
    # fail clearly rather than silently substituting a later source row.
    with tqdm(
        total=NUM_SAMPLES,
        desc="Repairing parametric designs",
        unit="design",
        dynamic_ncols=True,
    ) as progress:
        for source_dataset_row in source_order:
            source_parameters = parameters.iloc[source_dataset_row]
            source_thresholds = thresholds.iloc[source_dataset_row]
            source_values = values.iloc[source_dataset_row]
            row_seed = int(rng.integers(0, np.iinfo(np.uint32).max))

            repaired = repair_parametric_design(
                source_parameters.to_numpy(dtype=float),
                source_thresholds.to_numpy(dtype=float),
                source_values.to_numpy(dtype=float),
                depth_thickness_exceedance_lower=(
                    DEPTH_THICKNESS_EXCEEDANCE_LOWER
                ),
                depth_thickness_exceedance_upper=(
                    DEPTH_THICKNESS_EXCEEDANCE_UPPER
                ),
                section_modulus_exceedance_lower=(
                    SECTION_MODULUS_EXCEEDANCE_LOWER
                ),
                section_modulus_exceedance_upper=(
                    SECTION_MODULUS_EXCEEDANCE_UPPER
                ),
                spacing_exceedance_lower=SPACING_EXCEEDANCE_LOWER,
                spacing_exceedance_upper=SPACING_EXCEEDANCE_UPPER,
                random_seed=row_seed,
            ).to_numpy(dtype=float)

            candidate_key = tuple(
                np.round(repaired, DUPLICATE_DECIMALS)
            )

            if candidate_key in existing_candidate_keys:
                duplicate_candidates_rejected += 1
                raise RuntimeError(
                    "The repaired result for source row {} duplicates an "
                    "earlier result at {} decimal places.".format(
                        source_dataset_row,
                        DUPLICATE_DECIMALS,
                    )
                )

            existing_candidate_keys.add(candidate_key)
            candidate_index = len(repaired_rows)
            violated_mask = violated_constraint_mask(
                source_thresholds.to_numpy(dtype=float),
                source_values.to_numpy(dtype=float),
            )
            failed_names = [
                name
                for name, failed in zip(CONSTRAINT_NAMES, violated_mask)
                if failed
            ]

            repaired_rows.append(repaired)
            selected_source_rows.append(source_parameters.to_numpy(dtype=float))
            selected_threshold_rows.append(
                source_thresholds.to_numpy(dtype=float)
            )
            selected_value_rows.append(source_values.to_numpy(dtype=float))
            metadata_rows.append(
                {
                    "candidate_index": candidate_index,
                    "source_dataset_row": int(source_dataset_row),
                    "source_violation_count": int(violated_mask.sum()),
                    "source_failed_constraints": ";".join(failed_names),
                    "repair_random_seed": row_seed,
                }
            )
            progress.update(1)

    if len(repaired_rows) != NUM_SAMPLES:
        raise RuntimeError(
            "Only {} unique repaired designs could be created.".format(
                len(repaired_rows)
            )
        )

    repaired_frame = pd.DataFrame(
        repaired_rows,
        columns=parameters.columns,
    )
    source_parameter_frame = pd.DataFrame(
        selected_source_rows,
        columns=parameters.columns,
    )
    metadata_frame = pd.DataFrame(metadata_rows)

    # Save the raw signed change for every design parameter.  Positive values
    # mean the repair increased a parameter; negative values mean it decreased
    # it; zero means the parameter was left untouched.
    parameter_adjustments = repaired_frame - source_parameter_frame

    # Fit min/max scaling only on valid randomly generated dataset designs.
    # The same fixed bounds are then applied to both each source vector and its
    # repaired vector.  Repaired values are intentionally not clipped: a value
    # outside the original data range should contribute more than one range
    # unit to the distance instead of being hidden at the [0, 1] boundary.
    valid_parameters = parameters.iloc[valid_source_rows]
    parameter_minimum = valid_parameters.min(axis=0)
    parameter_maximum = valid_parameters.max(axis=0)
    parameter_range = parameter_maximum - parameter_minimum
    active_parameters = parameter_range > 0.0

    scaled_parameter_adjustments = pd.DataFrame(
        0.0,
        index=parameter_adjustments.index,
        columns=parameter_adjustments.columns,
    )
    scaled_parameter_adjustments.loc[:, active_parameters] = (
        parameter_adjustments.loc[:, active_parameters]
        / parameter_range.loc[active_parameters]
    )

    # The repair distance is the Euclidean length of the corresponding source
    # to repaired displacement in dataset-scaled 120-parameter space.
    repair_distance = np.linalg.norm(
        scaled_parameter_adjustments.to_numpy(dtype=float),
        axis=1,
    )
    number_parameters_changed = (
        np.abs(parameter_adjustments.to_numpy(dtype=float)) > 1.0e-12
    ).sum(axis=1)
    maximum_scaled_adjustment = np.abs(
        scaled_parameter_adjustments.to_numpy(dtype=float)
    ).max(axis=1)

    adjustment_identifiers = metadata_frame[
        ["candidate_index", "source_dataset_row"]
    ].copy()
    raw_adjustment_output = pd.concat(
        [adjustment_identifiers, parameter_adjustments],
        axis=1,
    )
    scaled_adjustment_output = pd.concat(
        [adjustment_identifiers, scaled_parameter_adjustments],
        axis=1,
    )
    distance_output = adjustment_identifiers.copy()
    distance_output["scaled_euclidean_repair_distance"] = repair_distance
    distance_output["num_parameters_changed"] = number_parameters_changed
    distance_output["maximum_absolute_scaled_adjustment"] = (
        maximum_scaled_adjustment
    )

    scaling_bounds = pd.DataFrame(
        {
            "parameter": parameters.columns,
            "dataset_minimum": parameter_minimum.to_numpy(dtype=float),
            "dataset_maximum": parameter_maximum.to_numpy(dtype=float),
            "dataset_range": parameter_range.to_numpy(dtype=float),
            "included_in_distance": active_parameters.to_numpy(dtype=bool),
        }
    )

    repaired_frame.to_csv(candidate_csv, index=False)
    metadata_frame.to_csv(
        experiment_dir / "candidate_metadata.csv",
        index=False,
    )
    source_parameter_frame.to_csv(
        experiment_dir / "source_parameters.csv",
        index=False,
    )
    pd.DataFrame(
        selected_threshold_rows,
        columns=thresholds.columns,
    ).to_csv(
        experiment_dir / "source_constraint_thresholds.csv",
        index=False,
    )
    pd.DataFrame(
        selected_value_rows,
        columns=values.columns,
    ).to_csv(
        experiment_dir / "source_constraint_values.csv",
        index=False,
    )
    raw_adjustment_output.to_csv(
        experiment_dir / "parameter_adjustments.csv",
        index=False,
    )
    scaled_adjustment_output.to_csv(
        experiment_dir / "scaled_parameter_adjustments.csv",
        index=False,
    )
    distance_output.to_csv(
        experiment_dir / "repair_distances.csv",
        index=False,
    )
    scaling_bounds.to_csv(
        experiment_dir / "parameter_scaling_bounds.csv",
        index=False,
    )

    manifest = {
        "batch_id": batch_id,
        "num_source_rows": int(len(parameters)),
        "num_valid_source_rows": int(len(valid_source_rows)),
        "num_source_rows_meeting_initial_violation_limit": int(
            len(eligible_source_rows)
        ),
        "num_candidates_written": int(len(repaired_frame)),
        "num_unique_candidates_at_configured_precision": int(
            len(existing_candidate_keys)
        ),
        "duplicate_candidates_rejected": int(
            duplicate_candidates_rejected
        ),
        "distinct_source_rows": int(
            metadata_frame["source_dataset_row"].nunique()
        ),
        "minimum_source_violations": int(
            metadata_frame["source_violation_count"].min()
        ),
        "maximum_source_violations": int(
            metadata_frame["source_violation_count"].max()
        ),
        "mean_source_violations": float(
            metadata_frame["source_violation_count"].mean()
        ),
        "repair_distance_definition": (
            "Euclidean source-to-repaired distance after scaling each "
            "parameter by the valid random-dataset min/max range"
        ),
        "mean_scaled_repair_distance": float(repair_distance.mean()),
        "median_scaled_repair_distance": float(
            np.median(repair_distance)
        ),
        "minimum_scaled_repair_distance": float(repair_distance.min()),
        "maximum_scaled_repair_distance": float(repair_distance.max()),
    }
    write_json(experiment_dir / "generation_manifest.json", manifest)

    print("Wrote {}".format(candidate_csv))
    return experiment_dir


# =============================================================================
# RESUMABLE RHINO WATCHDOG
# =============================================================================

def run_logged_command(command, log_path, environment=None):
    """Stream a worker process while retaining its complete experiment log."""

    with log_path.open("a") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )

        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()

        return process.wait()


def next_pending_design(batch_id, skip_file):
    """Ask the existing Rhino worker which incomplete design comes next."""

    result = subprocess.run(
        [
            sys.executable,
            str(RHINO_GENERATOR),
            "--batch-id",
            batch_id,
            "--batch-size",
            str(RHINO_BATCH_SIZE),
            "--skip-indices-file",
            str(skip_file),
            "--print-next-pending",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def run_rhino_watchdog(batch_id):
    """Generate structures and restart Rhino after recoverable failures."""

    experiment_dir = EXPERIMENTS_DIR / batch_id
    candidate_csv = experiment_dir / "{}_X_Results.csv".format(batch_id)

    if not candidate_csv.is_file():
        raise RuntimeError(
            "Generate candidates before starting Rhino: {}".format(
                candidate_csv
            )
        )

    skip_file = experiment_dir / "skipped_rhino_indices.txt"
    stop_file = experiment_dir / "rhino_watchdog.stop"
    log_path = experiment_dir / "rhino_watchdog.log"
    skip_file.touch(exist_ok=True)
    last_failed_item = None
    consecutive_failures = 0

    # An unbounded loop is intentional when RHINO_MAX_RESTARTS is zero.  Each
    # fresh worker discovers and skips every already-complete four-file package.
    for restart_count in itertools.count(1):
        if stop_file.is_file():
            raise RuntimeError(
                "Rhino stop file found: {}".format(stop_file)
            )

        pending_before = next_pending_design(batch_id, skip_file)

        if pending_before == "NONE":
            print("All Rhino structures are complete.")
            return

        command = [
            sys.executable,
            str(RHINO_GENERATOR),
            "--batch-id",
            batch_id,
            "--batch-size",
            str(RHINO_BATCH_SIZE),
            "--skip-indices-file",
            str(skip_file),
        ]
        exit_status = run_logged_command(command, log_path)
        pending_after = next_pending_design(batch_id, skip_file)

        if pending_after == "NONE":
            print("All Rhino structures are complete.")
            return

        if exit_status == 0:
            continue

        if pending_after == last_failed_item:
            consecutive_failures += 1
        else:
            last_failed_item = pending_after
            consecutive_failures = 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES_PER_DESIGN:
            with skip_file.open("a") as handle:
                handle.write("{}\n".format(pending_after))

            print(
                "Skipping design {} after {} consecutive failures.".format(
                    pending_after,
                    consecutive_failures,
                )
            )
            last_failed_item = None
            consecutive_failures = 0
            continue

        if RHINO_MAX_RESTARTS > 0 and restart_count >= RHINO_MAX_RESTARTS:
            raise RuntimeError("Rhino reached its configured restart limit.")

        print(
            "Restarting Rhino in {} seconds; completed designs will be "
            "resumed.".format(RHINO_RESTART_DELAY_SECONDS)
        )
        time.sleep(RHINO_RESTART_DELAY_SECONDS)


# =============================================================================
# EXACT EVALUATION AND PLOTTING
# =============================================================================

def run_exact_evaluation(batch_id):
    """Run the existing true structural constraint evaluator."""

    subprocess.run(
        [
            sys.executable,
            str(EXACT_EVALUATOR),
            "--batch-id",
            batch_id,
        ],
        check=True,
    )


def run_tsne(batch_id):
    """Create both existing generated-versus-dataset t-SNE figures."""

    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/private/tmp/ship-structures-matplotlib"
    subprocess.run(
        [
            sys.executable,
            str(TSNE_SCRIPT),
            "--batch-id",
            batch_id,
            "--perplexity",
            str(TSNE_PERPLEXITY),
            "--categorical-weight",
            str(TSNE_CATEGORICAL_WEIGHT),
            "--random-seed",
            str(RANDOM_SEED),
        ],
        check=True,
        env=environment,
    )


def parse_arguments():
    """Parse only execution stage and experiment identity."""

    parser = argparse.ArgumentParser(
        description="Run deterministic parametric design repair."
    )
    parser.add_argument(
        "--stage",
        choices=("candidates", "rhino", "evaluate", "tsne", "full"),
        default="full",
    )
    parser.add_argument(
        "--batch-id",
        default=BATCH_ID,
        help="Experiment directory name under Constraint_Optimization_Pipeline/experiments.",
    )
    return parser.parse_args()


def main():
    """Run the requested resumable pipeline stage or stages."""

    args = parse_arguments()
    experiment_dir = EXPERIMENTS_DIR / args.batch_id

    if args.stage in ("candidates", "full"):
        generate_candidates(args.batch_id)

    if args.stage in ("rhino", "full"):
        run_rhino_watchdog(args.batch_id)

    if args.stage in ("evaluate", "full"):
        summary_path = experiment_dir / "evaluation" / "summary.json"

        if args.stage == "full" and summary_path.is_file():
            print("Exact evaluation already exists; skipping evaluation.")
        else:
            run_exact_evaluation(args.batch_id)

    if args.stage in ("tsne", "full"):
        class_plot = (
            experiment_dir
            / "figures"
            / "tsne_class_bulkhead_comparison.png"
        )
        feasibility_plot = (
            experiment_dir
            / "figures"
            / "tsne_constraint_feasibility.png"
        )

        if (
            args.stage == "full"
            and class_plot.is_file()
            and feasibility_plot.is_file()
        ):
            print("t-SNE figures already exist; skipping plot generation.")
        else:
            run_tsne(args.batch_id)


if __name__ == "__main__":
    main()
