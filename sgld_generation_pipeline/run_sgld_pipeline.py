#!/usr/bin/env python3
"""Run the faithful SGLD experiment, resumable Rhino generation, and analysis."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_DIR = SCRIPT_DIR / "full_parameter_ranges_retrained"
BATCHES_DIR = RUN_DIR / "batches"
MODEL_TRAINING_MARKER = RUN_DIR / "model_training_complete.json"

SGLD_EXPERIMENT = SCRIPT_DIR / "sgld_experiment.py"
RHINO_GENERATOR = (
    PROJECT_ROOT
    / "equation_repair_pipeline"
    / "batched_structure_generation.py"
)
EXACT_EVALUATOR = SCRIPT_DIR / "evaluate_sgld_batches.py"
TSNE_SCRIPT = SCRIPT_DIR / "plot_shared_tsne.py"
STATISTICS_SCRIPT = SCRIPT_DIR / "summarize_design_methods.py"
PUBLISH_STRUCTURES_SCRIPT = (
    SCRIPT_DIR / "merge_full_parameter_range_structures.py"
)
PUBLISH_PARAMETERS_SCRIPT = (
    SCRIPT_DIR / "merge_retrained_structure_csvs.py"
)
MIDSHIP_EVALUATOR = PROJECT_ROOT / "tools" / "evaluate_midship_dataset.py"
STARTUP_MODEL = (
    PROJECT_ROOT
    / "MiDShip_Dataset"
    / "Random_Structures"
    / "random_test_design_0.3dm"
)

NUM_BATCHES = 5
POPULATION_SIZE = 100

# These settings govern only process supervision around Rhino.  They do not
# modify the notebook's SGLD sampling method or its fixed hyperparameters.
RHINO_BATCH_SIZE = 10
RHINO_RESTART_DELAY_SECONDS = 5
MAX_CONSECUTIVE_FAILURES_PER_DESIGN = 3


def batch_id(batch_number):
    return f"sgld_batch_{batch_number:03d}"


def batch_dir(batch_number):
    return BATCHES_DIR / batch_id(batch_number)


def candidate_csv(batch_number):
    current_batch_id = batch_id(batch_number)
    return batch_dir(batch_number) / f"{current_batch_id}_X_Results.csv"


def structures_dir(batch_number):
    return batch_dir(batch_number) / "structures"


def updated_parameter_csv(batch_number):
    current_batch_id = batch_id(batch_number)
    return structures_dir(batch_number) / f"{current_batch_id}_X_Results_Updated.csv"


def run_logged_command(command, log_path, environment=None):
    """Stream a subprocess to the terminal while preserving a complete log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

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


def rhino_command(batch_number, skip_file, print_next_pending=False):
    """Build the existing fresh-Rhino-process generator command."""

    command = [
        sys.executable,
        str(RHINO_GENERATOR),
        "--batch-id",
        batch_id(batch_number),
        "--input-csv",
        str(candidate_csv(batch_number)),
        "--output-dir",
        str(structures_dir(batch_number)),
        "--start-idx",
        "0",
        "--end-idx",
        str(POPULATION_SIZE),
        "--batch-size",
        str(RHINO_BATCH_SIZE),
        "--updated-parameters-csv",
        str(updated_parameter_csv(batch_number)),
        "--error-indices-csv",
        str(
            structures_dir(batch_number)
            / f"{batch_id(batch_number)}_design_error_idx_ALL.csv"
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


def next_pending_design(batch_number, skip_file):
    """Ask the shared Rhino supervisor for the next incomplete design index."""

    result = subprocess.run(
        rhino_command(
            batch_number,
            skip_file,
            print_next_pending=True,
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip().splitlines()[-1]


def append_skipped_index(skip_file, candidate_index):
    """Persist a repeatedly failing design so later structures can proceed."""

    with skip_file.open("a") as handle:
        handle.write(f"{candidate_index}\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_rhino_watchdog(batch_number):
    """Restart Rhino as needed and resume from per-design output checkpoints."""

    if not candidate_csv(batch_number).is_file():
        raise RuntimeError(
            f"Generate candidates before Rhino batch {batch_number}."
        )

    structures_dir(batch_number).mkdir(parents=True, exist_ok=True)
    skip_file = batch_dir(batch_number) / "skipped_rhino_indices.txt"
    stop_file = batch_dir(batch_number) / "rhino_watchdog.stop"
    log_path = batch_dir(batch_number) / "rhino_watchdog.log"
    skip_file.touch(exist_ok=True)
    last_failed_item = None
    consecutive_failures = 0

    for unused_restart_count in itertools.count(1):
        if stop_file.is_file():
            raise RuntimeError(
                f"Remove the stop file before resuming: {stop_file}"
            )

        pending_before = next_pending_design(batch_number, skip_file)

        if pending_before == "NONE":
            print(f"All non-skipped structures are complete for {batch_id(batch_number)}.")
            return

        exit_status = run_logged_command(
            rhino_command(batch_number, skip_file),
            log_path,
        )
        pending_after = next_pending_design(batch_number, skip_file)

        if pending_after == "NONE":
            print(f"All non-skipped structures are complete for {batch_id(batch_number)}.")
            return

        # A successful supervisor may still leave ordinary work only if the
        # process was intentionally bounded.  Continue with another fresh run.
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
                f"Skipping design {pending_after} after "
                f"{consecutive_failures} consecutive failures."
            )
            last_failed_item = None
            consecutive_failures = 0
            continue

        print(
            f"Restarting Rhino in {RHINO_RESTART_DELAY_SECONDS} seconds; "
            "completed designs remain checkpointed."
        )
        time.sleep(RHINO_RESTART_DELAY_SECONDS)


def evaluation_is_current(batch_number):
    """Return whether exact evaluation is newer than Rhino's parameter table."""

    summary_path = batch_dir(batch_number) / "evaluation" / "summary.json"
    updated_path = updated_parameter_csv(batch_number)

    if not summary_path.is_file() or not updated_path.is_file():
        return False

    return summary_path.stat().st_mtime >= updated_path.stat().st_mtime


def evaluate_batch(batch_number):
    """Run exact structural constraint evaluation for one completed batch."""

    subprocess.run(
        [
            sys.executable,
            str(EXACT_EVALUATOR),
            "--batch-number",
            str(batch_number),
        ],
        check=True,
    )


def run_candidate_stage():
    """Retrain once for this run, then sample five new SGLD populations.

    The run-specific marker distinguishes an ordinary resume from a new
    experiment. A new output folder always retrains both neural networks before
    sampling candidates. If Rhino or a later stage is interrupted, rerunning
    the pipeline preserves that model and those candidates so file checkpoints
    remain aligned.
    """

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_TRAINING_MARKER.is_file():
        subprocess.run(
            [
                sys.executable,
                str(SGLD_EXPERIMENT),
                "--stage",
                "train",
                "--force-retrain",
            ],
            check=True,
        )
        MODEL_TRAINING_MARKER.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model_checkpoint": str(
                        SCRIPT_DIR / "models" / "notebook_surrogate_models.pt"
                    ),
                    "behavior": (
                        "Both notebook neural networks were retrained before "
                        "candidate sampling for this run."
                    ),
                },
                indent=2,
            )
        )
    else:
        print(
            "This run's neural networks are already trained; preserving them "
            "to resume the same candidate and Rhino checkpoints."
        )

    subprocess.run(
        [sys.executable, str(SGLD_EXPERIMENT), "--stage", "generate"],
        check=True,
    )


def run_analysis_stage():
    """Create the shared t-SNE and the requested comparative statistics."""

    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/private/tmp/ship-structures-matplotlib"
    environment["XDG_CACHE_HOME"] = "/private/tmp/ship-structures-cache"

    subprocess.run(
        [sys.executable, str(TSNE_SCRIPT)],
        check=True,
        env=environment,
    )
    subprocess.run(
        [sys.executable, str(STATISTICS_SCRIPT)],
        check=True,
        env=environment,
    )


def run_publication_stage():
    """Publish staged batches and evaluate the aligned MiDShip SGLD set."""

    subprocess.run(
        [
            sys.executable,
            str(PUBLISH_STRUCTURES_SCRIPT),
            "--overwrite",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(PUBLISH_PARAMETERS_SCRIPT)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(MIDSHIP_EVALUATOR),
            "--dataset",
            "sgld",
        ],
        check=True,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the complete resumable SGLD/Rhino experiment."
    )
    parser.add_argument(
        "--stage",
        choices=(
            "candidates",
            "rhino",
            "evaluate",
            "publish",
            "analysis",
            "full",
        ),
        default="full",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.stage in ("candidates", "full"):
        run_candidate_stage()

    if args.stage in ("rhino", "full"):
        for batch_number in range(1, NUM_BATCHES + 1):
            run_rhino_watchdog(batch_number)

    if args.stage in ("evaluate", "full"):
        for batch_number in range(1, NUM_BATCHES + 1):
            if args.stage == "full" and evaluation_is_current(batch_number):
                print(f"Evaluation is current for {batch_id(batch_number)}.")
            else:
                evaluate_batch(batch_number)

    if args.stage in ("publish", "full"):
        run_publication_stage()

    if args.stage in ("analysis", "full"):
        run_analysis_stage()


if __name__ == "__main__":
    main()
