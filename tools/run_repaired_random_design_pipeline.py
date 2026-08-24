#!/usr/bin/env python3
"""Generate, checkpoint, evaluate, and plot repaired random ship designs.

This pipeline creates exactly two independently repaired variants for every
valid random-dataset design with 13 or fewer initial constraint violations.
The output size is determined from the current aligned random dataset rather
than a hard-coded historical row count.

The long Rhino stage is deliberately resumable.  Before Rhino starts,
``structures/repaired_random_design_X_Results_Updated.csv`` is created as an
output-sized table of zeros. After each successful structure, only that row is
replaced with ``hull.params`` and the complete CSV is atomically saved.  A
separate marker is written last.  If the process is killed before the marker,
that design is generated again on the next run; completed designs are skipped.

Stages
------
``candidates``
    Select every valid source with at most 13 violations, create two unique
    random-exceedance repairs per source, and initialize the zero checkpoint.
``rhino``
    Generate structures in restartable Rhino batches with per-design commits.
``evaluate``
    Recalculate the exact structural constraints from the generated files.
``publish``
    Copy the completed staged files into ``MiDShip_Dataset/Repaired_Structures``
    and write aligned aggregate evaluation tables there.
``tsne``
    Create the established class/bulkhead and feasibility t-SNE figures.
``full``
    Run every stage, reusing complete artifacts when it is safe to do so.
"""

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Make project imports independent of the caller's current working directory.
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
# FIXED EXPERIMENT SETTINGS
# =============================================================================

# The user-requested root name is used consistently for the experiment folder,
# candidate table, structure files, and Rhino-updated parameter checkpoint.
BATCH_ID = "repaired_random_design"

# Every currently valid aligned source row at or below this limit contributes
# two repaired variants. The count is intentionally derived at runtime because
# recalculated constraints can change which source rows meet this criterion.
MAX_INITIAL_CONSTRAINT_VIOLATIONS = 13
VARIANTS_PER_SOURCE = 2

# These are the same random target-exceedance ranges used by experiment 007.
DEPTH_THICKNESS_EXCEEDANCE_LOWER = 0.10
DEPTH_THICKNESS_EXCEEDANCE_UPPER = 0.35
SECTION_MODULUS_EXCEEDANCE_LOWER = 0.35
SECTION_MODULUS_EXCEEDANCE_UPPER = 0.60
SPACING_EXCEEDANCE_LOWER = 0.20
SPACING_EXCEEDANCE_UPPER = 0.30
RANDOM_SEED = 41

# No two output rows may be identical after all parameters are rounded to two
# decimal places.  A duplicate draw is repaired again with a fresh random seed.
DUPLICATE_DECIMALS = 2
MAX_UNIQUENESS_ATTEMPTS_PER_VARIANT = 10000

# Rhino restarts indefinitely unless the operator creates the stop file.
# A single persistently failing design is bypassed after three fresh attempts.
RHINO_BATCH_SIZE = 50
RHINO_MAX_RESTARTS = 0
RHINO_RESTART_DELAY_SECONDS = 5
MAX_CONSECUTIVE_FAILURES_PER_DESIGN = 3

# Plot settings match the existing generated-versus-dataset workflow.
TSNE_PERPLEXITY = 45.0
TSNE_CATEGORICAL_WEIGHT = 2.5


# =============================================================================
# INPUT DATA AND EXISTING PIPELINE WORKERS
# =============================================================================

MIDSHIP_DATASET_DIR = PROJECT_ROOT / "MiDShip_Dataset"
DATASET_DIR = MIDSHIP_DATASET_DIR / "Random_Structures"
PUBLISHED_DIR = MIDSHIP_DATASET_DIR / "Repaired_Structures"
PARAMETER_CSV = DATASET_DIR / "random_test_design_Parameters_All.csv"
THRESHOLD_CSV = DATASET_DIR / "random_test_design_Constraint_Thresholds.csv"
VALUE_CSV = DATASET_DIR / "random_test_design_Constraint_Values.csv"

RHINO_GENERATOR = PIPELINE_DIR / "batched_structure_generation.py"
EXACT_EVALUATOR = PIPELINE_DIR / "evaluate_structures.py"
TSNE_SCRIPT = PIPELINE_DIR / "plot_tsne_comparison.py"
MIDSHIP_EVALUATOR = SCRIPT_DIR / "evaluate_midship_dataset.py"
STARTUP_MODEL = DATASET_DIR / "random_test_design_0.3dm"


def experiment_dir():
    """Return the fixed output directory for this experiment."""

    return EXPERIMENTS_DIR / BATCH_ID


def candidate_csv_path():
    """Return the immutable input parameter table consumed by Rhino."""

    return experiment_dir() / "{}_X_Results.csv".format(BATCH_ID)


def structures_dir():
    """Return the directory containing Rhino outputs and checkpoints."""

    return experiment_dir() / "structures"


def updated_parameter_csv_path():
    """Return the zero-prefilled, per-design Rhino parameter checkpoint."""

    return structures_dir() / "{}_X_Results_Updated.csv".format(BATCH_ID)


def file_sha256(file_path):
    """Return a stable fingerprint for one immutable source table."""

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_json_atomic(file_path, contents):
    """Write JSON atomically so a kill cannot expose a partial document."""

    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with temporary_path.open("w") as handle:
        json.dump(contents, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_path, file_path)


def write_dataframe_atomic(frame, file_path):
    """Write one CSV through a sibling temporary file and atomically replace."""

    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)

    with temporary_path.open("rb") as handle:
        os.fsync(handle.fileno())

    os.replace(temporary_path, file_path)


def experiment_settings():
    """Describe the complete reproducible candidate and execution contract."""

    return {
        "batch_id": BATCH_ID,
        "method": "deterministic_equation_repair_with_random_exceedance",
        "maximum_initial_constraint_violations": (
            MAX_INITIAL_CONSTRAINT_VIOLATIONS
        ),
        "variants_per_source": VARIANTS_PER_SOURCE,
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
        "maximum_uniqueness_attempts_per_variant": (
            MAX_UNIQUENESS_ATTEMPTS_PER_VARIANT
        ),
        "source_order": "ascending_original_dataset_row_then_variant_0_1",
        "zero_error_rows": (
            "excluded_when_thresholds_and_values_are_both_all_zero"
        ),
        "rhino_batch_size": RHINO_BATCH_SIZE,
        "rhino_max_restarts": RHINO_MAX_RESTARTS,
        "rhino_restart_delay_seconds": RHINO_RESTART_DELAY_SECONDS,
        "max_consecutive_failures_per_design": (
            MAX_CONSECUTIVE_FAILURES_PER_DESIGN
        ),
        "rhino_checkpoint": (
            "zero_prefill_then_atomic_full_csv_save_after_each_design"
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


def confirm_or_create_experiment():
    """Create the experiment or verify that a resume uses identical settings."""

    output_dir = experiment_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "pipeline_settings.json"
    settings = experiment_settings()

    if settings_path.is_file():
        with settings_path.open() as handle:
            saved_settings = json.load(handle)

        if saved_settings != settings:
            raise RuntimeError(
                "Existing experiment settings do not match this script: {}".format(
                    settings_path
                )
            )

        return

    write_json_atomic(settings_path, settings)


def load_aligned_source_data():
    """Load aligned tables and exclude only their all-zero error rows."""

    parameters = pd.read_csv(PARAMETER_CSV)
    thresholds = pd.read_csv(THRESHOLD_CSV)
    values = pd.read_csv(VALUE_CSV)

    if not (len(parameters) == len(thresholds) == len(values)):
        raise RuntimeError("The three source CSV files are not row-aligned.")

    threshold_is_zero = (thresholds.to_numpy(dtype=float) == 0.0).all(axis=1)
    value_is_zero = (values.to_numpy(dtype=float) == 0.0).all(axis=1)
    valid_source_rows = np.flatnonzero(
        ~(threshold_is_zero & value_is_zero)
    )

    return parameters, thresholds, values, valid_source_rows


def select_eligible_source_rows(thresholds, values, valid_source_rows):
    """Return valid source rows having 13 or fewer active violations."""

    violation_counts = np.asarray(
        [
            violated_constraint_mask(
                thresholds.iloc[source_row].to_numpy(dtype=float),
                values.iloc[source_row].to_numpy(dtype=float),
            ).sum()
            for source_row in valid_source_rows
        ],
        dtype=int,
    )

    return valid_source_rows[
        violation_counts <= MAX_INITIAL_CONSTRAINT_VIOLATIONS
    ]


def rounded_candidate_key(candidate):
    """Return the two-decimal identity used to reject repeated designs."""

    return tuple(np.round(candidate, DUPLICATE_DECIMALS))


def repair_one_unique_variant(
    source_parameters,
    source_thresholds,
    source_values,
    rng,
    existing_candidate_keys,
):
    """Draw exceedances until one globally unique repaired vector is found."""

    for attempt in range(1, MAX_UNIQUENESS_ATTEMPTS_PER_VARIANT + 1):
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
        candidate_key = rounded_candidate_key(repaired)

        if candidate_key not in existing_candidate_keys:
            existing_candidate_keys.add(candidate_key)
            return repaired, row_seed, attempt

    raise RuntimeError(
        "Could not create a unique repair after {} draws for one source.".format(
            MAX_UNIQUENESS_ATTEMPTS_PER_VARIANT
        )
    )


def build_candidate_tables(
    parameters,
    thresholds,
    values,
    eligible_source_rows,
):
    """Create two ordered, independent, globally unique repairs per source."""

    rng = np.random.default_rng(RANDOM_SEED)
    existing_candidate_keys = set()
    repaired_rows = []
    source_parameter_rows = []
    source_threshold_rows = []
    source_value_rows = []
    metadata_rows = []
    duplicate_draws_rejected = 0

    with tqdm(
        total=len(eligible_source_rows) * VARIANTS_PER_SOURCE,
        desc="Repairing random dataset designs",
        unit="design",
        dynamic_ncols=True,
    ) as progress:
        for source_dataset_row in eligible_source_rows:
            source_parameters = parameters.iloc[source_dataset_row]
            source_thresholds = thresholds.iloc[source_dataset_row]
            source_values = values.iloc[source_dataset_row]
            violated_mask = violated_constraint_mask(
                source_thresholds.to_numpy(dtype=float),
                source_values.to_numpy(dtype=float),
            )
            failed_names = [
                name
                for name, failed in zip(CONSTRAINT_NAMES, violated_mask)
                if failed
            ]

            # Keep the two variants adjacent and preserve the original dataset
            # order.  Only their independent exceedance draws differ by design.
            for variant_index in range(VARIANTS_PER_SOURCE):
                repaired, row_seed, uniqueness_attempts = (
                    repair_one_unique_variant(
                        source_parameters,
                        source_thresholds,
                        source_values,
                        rng,
                        existing_candidate_keys,
                    )
                )
                duplicate_draws_rejected += uniqueness_attempts - 1
                candidate_index = len(repaired_rows)

                repaired_rows.append(repaired)
                source_parameter_rows.append(
                    source_parameters.to_numpy(dtype=float)
                )
                source_threshold_rows.append(
                    source_thresholds.to_numpy(dtype=float)
                )
                source_value_rows.append(
                    source_values.to_numpy(dtype=float)
                )
                metadata_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "source_dataset_row": int(source_dataset_row),
                        "source_variant_index": variant_index,
                        "source_violation_count": int(violated_mask.sum()),
                        "source_failed_constraints": ";".join(failed_names),
                        "repair_random_seed": row_seed,
                        "uniqueness_draw_attempts": uniqueness_attempts,
                    }
                )
                progress.update(1)

    frames = {
        "repaired": pd.DataFrame(repaired_rows, columns=parameters.columns),
        "source_parameters": pd.DataFrame(
            source_parameter_rows,
            columns=parameters.columns,
        ),
        "source_thresholds": pd.DataFrame(
            source_threshold_rows,
            columns=thresholds.columns,
        ),
        "source_values": pd.DataFrame(
            source_value_rows,
            columns=values.columns,
        ),
        "metadata": pd.DataFrame(metadata_rows),
    }

    return frames, existing_candidate_keys, duplicate_draws_rejected


def add_adjustment_and_distance_tables(frames, valid_parameters):
    """Calculate raw changes and dataset-min/max-scaled repair distances."""

    repaired = frames["repaired"]
    source_parameters = frames["source_parameters"]
    metadata = frames["metadata"]
    parameter_adjustments = repaired - source_parameters

    # Fit each range on valid random designs only.  Do not clip repaired values:
    # movement beyond the source range should remain visible in the distance.
    parameter_minimum = valid_parameters.min(axis=0)
    parameter_maximum = valid_parameters.max(axis=0)
    parameter_range = parameter_maximum - parameter_minimum
    active_parameters = parameter_range > 0.0
    scaled_adjustments = pd.DataFrame(
        0.0,
        index=parameter_adjustments.index,
        columns=parameter_adjustments.columns,
    )
    scaled_adjustments.loc[:, active_parameters] = (
        parameter_adjustments.loc[:, active_parameters]
        / parameter_range.loc[active_parameters]
    )

    identifiers = metadata[
        ["candidate_index", "source_dataset_row", "source_variant_index"]
    ].copy()
    raw_output = pd.concat([identifiers, parameter_adjustments], axis=1)
    scaled_output = pd.concat([identifiers, scaled_adjustments], axis=1)
    distance_output = identifiers.copy()
    distance_output["scaled_euclidean_repair_distance"] = np.linalg.norm(
        scaled_adjustments.to_numpy(dtype=float),
        axis=1,
    )
    distance_output["num_parameters_changed"] = (
        np.abs(parameter_adjustments.to_numpy(dtype=float)) > 1.0e-12
    ).sum(axis=1)
    distance_output["maximum_absolute_scaled_adjustment"] = np.abs(
        scaled_adjustments.to_numpy(dtype=float)
    ).max(axis=1)
    scaling_bounds = pd.DataFrame(
        {
            "parameter": repaired.columns,
            "dataset_minimum": parameter_minimum.to_numpy(dtype=float),
            "dataset_maximum": parameter_maximum.to_numpy(dtype=float),
            "dataset_range": parameter_range.to_numpy(dtype=float),
            "included_in_distance": active_parameters.to_numpy(dtype=bool),
        }
    )

    frames["parameter_adjustments"] = raw_output
    frames["scaled_parameter_adjustments"] = scaled_output
    frames["repair_distances"] = distance_output
    frames["parameter_scaling_bounds"] = scaling_bounds


def initialize_zero_parameter_checkpoint(candidate_frame):
    """Create the output-sized zero checkpoint without erasing resumed work."""

    output_dir = structures_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = updated_parameter_csv_path()

    if checkpoint_path.is_file():
        existing = pd.read_csv(checkpoint_path)

        if (
            list(existing.columns) != list(candidate_frame.columns)
            or len(existing) != len(candidate_frame)
        ):
            raise RuntimeError(
                "Existing Rhino checkpoint is incompatible: {}".format(
                    checkpoint_path
                )
            )

        return

    zero_checkpoint = pd.DataFrame(
        np.zeros(candidate_frame.shape, dtype=float),
        columns=candidate_frame.columns,
    )
    write_dataframe_atomic(zero_checkpoint, checkpoint_path)


def write_candidate_artifacts(
    frames,
    existing_candidate_keys,
    duplicate_draws_rejected,
    num_source_rows,
    num_valid_source_rows,
    num_eligible_source_rows,
):
    """Save the candidate table, provenance, adjustments, and manifest."""

    output_dir = experiment_dir()
    artifact_paths = {
        "repaired": candidate_csv_path(),
        "metadata": output_dir / "candidate_metadata.csv",
        "source_parameters": output_dir / "source_parameters.csv",
        "source_thresholds": output_dir / "source_constraint_thresholds.csv",
        "source_values": output_dir / "source_constraint_values.csv",
        "parameter_adjustments": output_dir / "parameter_adjustments.csv",
        "scaled_parameter_adjustments": (
            output_dir / "scaled_parameter_adjustments.csv"
        ),
        "repair_distances": output_dir / "repair_distances.csv",
        "parameter_scaling_bounds": (
            output_dir / "parameter_scaling_bounds.csv"
        ),
    }

    for frame_name, file_path in artifact_paths.items():
        write_dataframe_atomic(frames[frame_name], file_path)

    distances = frames["repair_distances"][
        "scaled_euclidean_repair_distance"
    ].to_numpy(dtype=float)
    metadata = frames["metadata"]
    manifest = {
        "batch_id": BATCH_ID,
        "num_source_rows": int(num_source_rows),
        "num_valid_source_rows": int(num_valid_source_rows),
        "num_eligible_source_rows": int(num_eligible_source_rows),
        "variants_per_source": VARIANTS_PER_SOURCE,
        "num_candidates_written": int(len(frames["repaired"])),
        "num_unique_candidates_at_configured_precision": int(
            len(existing_candidate_keys)
        ),
        "duplicate_draws_rejected_and_resampled": int(
            duplicate_draws_rejected
        ),
        "distinct_source_rows": int(
            metadata["source_dataset_row"].nunique()
        ),
        "minimum_source_violations": int(
            metadata["source_violation_count"].min()
        ),
        "maximum_source_violations": int(
            metadata["source_violation_count"].max()
        ),
        "mean_source_violations": float(
            metadata["source_violation_count"].mean()
        ),
        "repair_distance_definition": (
            "Euclidean source-to-repaired distance after scaling each "
            "parameter by the valid random-dataset min/max range"
        ),
        "minimum_scaled_repair_distance": float(distances.min()),
        "median_scaled_repair_distance": float(np.median(distances)),
        "mean_scaled_repair_distance": float(distances.mean()),
        "maximum_scaled_repair_distance": float(distances.max()),
    }
    write_json_atomic(output_dir / "generation_manifest.json", manifest)


def candidate_bundle_is_complete():
    """Return whether the candidate stage wrote its final commit marker."""

    return (experiment_dir() / "candidate_generation_complete.json").is_file()


def generate_candidates():
    """Generate two repairs per eligible source or reuse a committed bundle."""

    # A committed candidate bundle is immutable and remains valid when only
    # the repository's parent-directory path changes. Preserve it before
    # comparing the older manifest's absolute path strings.
    if candidate_bundle_is_complete():
        candidates = pd.read_csv(candidate_csv_path())
        initialize_zero_parameter_checkpoint(candidates)
        print("Candidate bundle already exists; preserving it unchanged.")
        return

    confirm_or_create_experiment()

    parameters, thresholds, values, valid_source_rows = (
        load_aligned_source_data()
    )
    eligible_source_rows = select_eligible_source_rows(
        thresholds,
        values,
        valid_source_rows,
    )

    frames, existing_candidate_keys, duplicate_draws_rejected = (
        build_candidate_tables(
            parameters,
            thresholds,
            values,
            eligible_source_rows,
        )
    )

    expected_num_candidates = (
        len(eligible_source_rows) * VARIANTS_PER_SOURCE
    )

    if len(frames["repaired"]) != expected_num_candidates:
        raise RuntimeError(
            "Generated {} candidates instead of the requested {}.".format(
                len(frames["repaired"]),
                expected_num_candidates,
            )
        )

    if len(existing_candidate_keys) != expected_num_candidates:
        raise RuntimeError("The completed candidate set contains repeats.")

    add_adjustment_and_distance_tables(
        frames,
        parameters.iloc[valid_source_rows],
    )
    write_candidate_artifacts(
        frames,
        existing_candidate_keys,
        duplicate_draws_rejected,
        num_source_rows=len(parameters),
        num_valid_source_rows=len(valid_source_rows),
        num_eligible_source_rows=len(eligible_source_rows),
    )
    initialize_zero_parameter_checkpoint(frames["repaired"])

    # Write this last so a resume never mistakes a partial candidate bundle for
    # an immutable Rhino input table.
    write_json_atomic(
        experiment_dir() / "candidate_generation_complete.json",
        {
            "status": "complete",
            "num_candidates": len(frames["repaired"]),
            "num_source_designs": len(eligible_source_rows),
        },
    )
    print("Wrote {} unique repaired designs.".format(len(frames["repaired"])))


# =============================================================================
# RESUMABLE RHINO WATCHDOG
# =============================================================================

def run_logged_command(command, log_path, environment=None):
    """Stream a child process while appending its complete output to a log."""

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


def rhino_worker_command(skip_file, print_next_pending=False):
    """Build the shared Rhino command with per-design checkpointing enabled."""

    command = [
        sys.executable,
        str(RHINO_GENERATOR),
        "--batch-id",
        BATCH_ID,
        "--batch-size",
        str(RHINO_BATCH_SIZE),
        "--input-csv",
        str(candidate_csv_path()),
        "--output-dir",
        str(structures_dir()),
        "--updated-parameters-csv",
        str(updated_parameter_csv_path()),
        "--error-indices-csv",
        str(
            structures_dir()
            / "{}_design_error_idx_ALL.csv".format(BATCH_ID)
        ),
        "--startup-model",
        str(STARTUP_MODEL),
        "--skip-indices-file",
        str(skip_file),
        "--checkpoint-every-design",
    ]

    if print_next_pending:
        command.append("--print-next-pending")

    return command


def next_pending_design(skip_file):
    """Ask the Rhino supervisor which non-skipped design comes next."""

    result = subprocess.run(
        rhino_worker_command(skip_file, print_next_pending=True),
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip().splitlines()[-1]


def append_skipped_index(skip_file, candidate_index):
    """Persist one three-time failure before continuing to later designs."""

    with skip_file.open("a") as handle:
        handle.write("{}\n".format(candidate_index))
        handle.flush()
        os.fsync(handle.fileno())


def run_rhino_watchdog():
    """Generate structures, restart after failures, and resume committed rows."""

    if not candidate_bundle_is_complete():
        raise RuntimeError("Run the candidate stage before starting Rhino.")

    candidates = pd.read_csv(candidate_csv_path())
    initialize_zero_parameter_checkpoint(candidates)
    skip_file = experiment_dir() / "skipped_rhino_indices.txt"
    stop_file = experiment_dir() / "rhino_watchdog.stop"
    log_path = experiment_dir() / "rhino_watchdog.log"
    skip_file.touch(exist_ok=True)
    last_failed_item = None
    consecutive_failures = 0

    # Zero configured restarts intentionally means no watchdog-level limit.
    for restart_count in itertools.count(1):
        if stop_file.is_file():
            raise RuntimeError(
                "Remove the stop file before resuming: {}".format(stop_file)
            )

        pending_before = next_pending_design(skip_file)

        if pending_before == "NONE":
            print("All non-skipped Rhino structures are complete.")
            return

        exit_status = run_logged_command(
            rhino_worker_command(skip_file),
            log_path,
        )
        pending_after = next_pending_design(skip_file)

        if pending_after == "NONE":
            print("All non-skipped Rhino structures are complete.")
            return

        # A zero status means the requested batch completed and more ordinary
        # pending work remains.  Launch the next fresh Rhino process directly.
        if exit_status == 0:
            continue

        if pending_after == last_failed_item:
            consecutive_failures += 1
        else:
            last_failed_item = pending_after
            consecutive_failures = 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES_PER_DESIGN:
            append_skipped_index(skip_file, pending_after)
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
            "Restarting Rhino in {} seconds; committed designs will be "
            "skipped.".format(RHINO_RESTART_DELAY_SECONDS)
        )
        time.sleep(RHINO_RESTART_DELAY_SECONDS)


# =============================================================================
# EXACT EVALUATION AND T-SNE
# =============================================================================

def run_exact_evaluation():
    """Calculate true constraints from every available structural-element CSV."""

    subprocess.run(
        [
            sys.executable,
            str(EXACT_EVALUATOR),
            "--batch-id",
            BATCH_ID,
        ],
        check=True,
    )


def publish_repaired_dataset():
    """Copy a completed staged repair run into the released dataset folder."""

    if not updated_parameter_csv_path().is_file():
        raise RuntimeError("Run the repaired Rhino stage before publication.")

    error_source = (
        structures_dir()
        / "{}_design_error_idx_ALL.csv".format(BATCH_ID)
    )
    error_indices = set()

    if error_source.is_file():
        error_rows = pd.read_csv(error_source)

        if not error_rows.empty:
            error_indices = set(
                error_rows.iloc[:, 0].to_numpy(dtype=int).tolist()
            )

    # Do not publish a partial Rhino run. Every row must either have all four
    # required structure artifacts or be explicitly recorded as an error.
    num_candidates = len(pd.read_csv(candidate_csv_path()))
    missing_indices = []

    for candidate_index in range(num_candidates):
        if candidate_index in error_indices:
            continue

        file_stem = "{}_design_{}".format(BATCH_ID, candidate_index)
        required_paths = [
            structures_dir() / (file_stem + ".3dm"),
            structures_dir() / (file_stem + ".igs"),
            structures_dir() / (file_stem + "_MeshElements.igs"),
            structures_dir() / (file_stem + "_Structural_Elements.csv"),
        ]

        if not all(path.is_file() and path.stat().st_size > 0 for path in required_paths):
            missing_indices.append(candidate_index)

    if missing_indices:
        raise RuntimeError(
            "Cannot publish a partial repaired run; {} non-error designs are "
            "incomplete. First indices: {}".format(
                len(missing_indices),
                missing_indices[:10],
            )
        )

    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)

    # Retain both the generated candidate vectors and Rhino's cleaned vectors.
    # The latter is the parameter table aligned with the released CAD files.
    shutil.copy2(
        candidate_csv_path(),
        PUBLISHED_DIR / "repaired_random_design_Parameters.csv",
    )
    shutil.copy2(
        updated_parameter_csv_path(),
        PUBLISHED_DIR / "repaired_random_design_Parameters_Updated.csv",
    )

    published_error_path = (
        PUBLISHED_DIR
        / "repaired_random_design_design_error_idx_ALL.csv"
    )

    if error_source.is_file():
        shutil.copy2(error_source, published_error_path)
    else:
        pd.DataFrame(columns=["error_idx"]).to_csv(
            published_error_path,
            index=False,
        )

    # Every staged CAD/structural artifact already has the canonical
    # ``repaired_random_design_design_<index>`` stem, so no renaming is needed.
    copied_files = 0
    for source_path in structures_dir().glob(
        "{}_design_*".format(BATCH_ID)
    ):
        if source_path.is_file():
            shutil.copy2(source_path, PUBLISHED_DIR / source_path.name)
            copied_files += 1

    # Recalculate the three aligned aggregate tables from the published files.
    subprocess.run(
        [
            sys.executable,
            str(MIDSHIP_EVALUATOR),
            "--dataset",
            "repaired",
        ],
        check=True,
    )

    print(
        "Published {} repaired structure files to {}.".format(
            copied_files,
            PUBLISHED_DIR,
        )
    )


def evaluation_is_current():
    """Return whether evaluation is newer than the Rhino parameter checkpoint."""

    summary_path = experiment_dir() / "evaluation" / "summary.json"

    if not summary_path.is_file() or not updated_parameter_csv_path().is_file():
        return False

    return (
        summary_path.stat().st_mtime
        >= updated_parameter_csv_path().stat().st_mtime
    )


def run_tsne():
    """Create both existing generated-versus-dataset t-SNE figures."""

    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/private/tmp/ship-structures-matplotlib"
    subprocess.run(
        [
            sys.executable,
            str(TSNE_SCRIPT),
            "--batch-id",
            BATCH_ID,
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


def tsne_is_current():
    """Return whether both figures are newer than exact candidate results."""

    results_path = experiment_dir() / "evaluation" / "candidate_results.csv"
    figure_paths = [
        experiment_dir() / "figures" / "tsne_class_bulkhead_comparison.png",
        experiment_dir() / "figures" / "tsne_constraint_feasibility.png",
    ]

    if not results_path.is_file() or not all(path.is_file() for path in figure_paths):
        return False

    source_time = max(
        results_path.stat().st_mtime,
        candidate_csv_path().stat().st_mtime,
    )

    return all(path.stat().st_mtime >= source_time for path in figure_paths)


def parse_arguments():
    """Select one resumable stage or the complete pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate two repaired variants for every valid random design "
            "with 13 or fewer initial constraint violations."
        )
    )
    parser.add_argument(
        "--stage",
        choices=(
            "candidates",
            "rhino",
            "evaluate",
            "publish",
            "tsne",
            "full",
        ),
        default="full",
    )

    return parser.parse_args()


def main():
    """Run only the requested stages and preserve every committed checkpoint."""

    args = parse_arguments()

    if args.stage in ("candidates", "full"):
        generate_candidates()

    if args.stage in ("rhino", "full"):
        run_rhino_watchdog()

    if args.stage in ("evaluate", "full"):
        if args.stage == "full" and evaluation_is_current():
            print("Exact evaluation is current; skipping it.")
        else:
            run_exact_evaluation()

    if args.stage in ("publish", "full"):
        publish_repaired_dataset()

    if args.stage in ("tsne", "full"):
        if args.stage == "full" and tsne_is_current():
            print("t-SNE figures are current; skipping them.")
        else:
            run_tsne()


if __name__ == "__main__":
    main()
