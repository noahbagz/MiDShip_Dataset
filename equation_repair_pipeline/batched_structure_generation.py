#!/usr/bin/env python3
"""
Generate MiDShip structures in fresh Rhino-process batches.

Run this file from normal Python, not from inside Rhino:

    python3 equation_repair_pipeline/batched_structure_generation.py \
        --checkpoint-every-design

Run a small smoke-test range with:

    python3 equation_repair_pipeline/batched_structure_generation.py \
        --batch-id random_test \
        --start-idx 0 --end-idx 10 --batch-size 1

The worker performs the same structure-generation operations as
Generate_Structures_After_Optimization.py. The supervisor skips rows whose
four expected structure files are already complete and restarts Rhino after
each batch.

The optional ``--checkpoint-every-design`` mode additionally initializes the
updated-parameter CSV with zeros, saves each successful ``hull.params`` row
immediately, and requires a per-design completion marker during resume.  The
default after-batch behavior remains unchanged for existing pipelines.

Rhino 8 note:
    `rhinocode` needs Rhino's script server running. If needed, add
    `StartScriptServer` to Rhino startup commands.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import traceback


SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
RHINO_MACROS_DIR = os.path.join(MAIN_DIR, "Rhino_Macros")
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "experiments")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "_batched_structure_config.json")

# Select the dataset by uncommenting one complete configuration block.  The
# random configuration is active by default; repaired and SGLD pipelines pass
# the same values explicitly when they invoke this shared worker.

# Randomly generated dataset.
DEFAULT_BATCH_ID = "random_test"
DEFAULT_INPUT_CSV = os.path.join(
    MAIN_DIR,
    "MiDShip_Dataset",
    "Random_Structures",
    "random_test_design_Parameters.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    MAIN_DIR,
    "MiDShip_Dataset",
    "Random_Structures",
)
DEFAULT_UPDATED_PARAMETERS_CSV = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "random_test_design_Parameters_All.csv",
)
DEFAULT_ERROR_INDICES_CSV = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "random_test_design_error_idx_All.csv",
)

# Repaired dataset.
# DEFAULT_BATCH_ID = "repaired_random_design"
# DEFAULT_INPUT_CSV = os.path.join(
#     MAIN_DIR, "MiDShip_Dataset", "Repaired_Structures",
#     "repaired_random_design_Parameters.csv",
# )
# DEFAULT_OUTPUT_DIR = os.path.join(
#     MAIN_DIR, "MiDShip_Dataset", "Repaired_Structures",
# )
# DEFAULT_UPDATED_PARAMETERS_CSV = os.path.join(
#     DEFAULT_OUTPUT_DIR, "repaired_random_design_Parameters_Updated.csv",
# )
# DEFAULT_ERROR_INDICES_CSV = os.path.join(
#     DEFAULT_OUTPUT_DIR, "repaired_random_design_design_error_idx_ALL.csv",
# )

# SGLD dataset.
# DEFAULT_BATCH_ID = "sgld"
# DEFAULT_INPUT_CSV = os.path.join(
#     MAIN_DIR, "MiDShip_Dataset", "SGLD_Gen_Structures",
#     "sgld_design_parameters.csv",
# )
# DEFAULT_OUTPUT_DIR = os.path.join(
#     MAIN_DIR, "MiDShip_Dataset", "SGLD_Gen_Structures",
# )
# DEFAULT_UPDATED_PARAMETERS_CSV = os.path.join(
#     DEFAULT_OUTPUT_DIR, "sgld_design_parameters_Updated.csv",
# )
# DEFAULT_ERROR_INDICES_CSV = os.path.join(
#     DEFAULT_OUTPUT_DIR, "sgld_design_error_idx_ALL.csv",
# )

START_IDX = 0
END_IDX = None
BATCH_SIZE = 10

# Opening an existing model gives Rhino's ScriptEditor an active document.
# Structure_3H clears that document before constructing each new structure.
STARTUP_MODEL_FILE = os.path.join(
    MAIN_DIR,
    "MiDShip_Dataset",
    "Random_Structures",
    "random_test_design_0.3dm",
)

RHINO_APP = "/Applications/Rhino 8.app"
RHINOCODE = os.path.join(RHINO_APP, "Contents", "Resources", "bin", "rhinocode")
RHINO_START_TIMEOUT_SECONDS = 30
RHINO_DOCUMENT_TIMEOUT_SECONDS = 30
RHINO_GRACEFUL_STOP_TIMEOUT_SECONDS = 10
RHINO_FORCED_STOP_TIMEOUT_SECONDS = 10
RHINO_POLL_SECONDS = 1
WORKER_TIMEOUT_SECONDS = 3600


def is_rhino():
    try:
        import Rhino  # noqa: F401
        import rhinoscriptsyntax  # noqa: F401
        return True
    except Exception:
        return False


def write_config(config):
    # The normal-Python supervisor and Rhino worker exchange one batch through
    # this small JSON file.  Write to a sibling temporary file first so the
    # polling reader never observes a truncated JSON document mid-write.
    temporary_path = CONFIG_PATH + ".tmp"

    with open(temporary_path, "w") as handle:
        json.dump(config, handle, indent=2)

    os.replace(temporary_path, CONFIG_PATH)


def read_config():
    with open(CONFIG_PATH, "r") as handle:
        return json.load(handle)


def read_csv_table(file_path):
    with open(file_path, "r", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise RuntimeError("Input CSV is empty: {}".format(file_path))

    return rows[0], rows[1:]


def write_csv_table(file_path, header, rows):
    # Write through a sibling temporary file.  The optional per-design mode
    # calls this after every successful structure, so flush the file before
    # exposing it as the current checkpoint.
    temporary_path = file_path + ".tmp"

    with open(temporary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_path, file_path)


def structure_file_stem(opt_test_id, idx):
    return "{}_design_{}".format(opt_test_id, idx)


def expected_structure_paths(output_dir, opt_test_id, idx):
    file_stem = structure_file_stem(opt_test_id, idx)

    return [
        os.path.join(output_dir, file_stem + ".3dm"),
        os.path.join(output_dir, file_stem + ".igs"),
        os.path.join(output_dir, file_stem + "_MeshElements.igs"),
        os.path.join(output_dir, file_stem + "_Structural_Elements.csv"),
    ]


def completion_marker_path(output_dir, opt_test_id, idx):
    """Return the commit marker written after one durable checkpoint."""

    return os.path.join(
        output_dir,
        ".completed",
        structure_file_stem(opt_test_id, idx) + ".json",
    )


def write_completion_marker(output_dir, opt_test_id, idx):
    """Commit one design after its files and updated parameters are saved."""

    marker_path = completion_marker_path(output_dir, opt_test_id, idx)
    marker_dir = os.path.dirname(marker_path)
    temporary_path = marker_path + ".tmp"
    os.makedirs(marker_dir, exist_ok=True)

    with open(temporary_path, "w") as handle:
        json.dump(
            {
                "candidate_index": int(idx),
                "status": "complete",
                "completed_unix_time": time.time(),
            },
            handle,
            indent=2,
        )
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_path, marker_path)


def structure_outputs_are_complete(
    output_dir,
    opt_test_id,
    idx,
    require_completion_marker=False,
):
    for output_path in expected_structure_paths(output_dir, opt_test_id, idx):
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            return False

    # In checkpoint mode the four output files alone are insufficient.  The
    # marker is created only after the matching row in Results_Updated.csv has
    # also been replaced and saved.
    if require_completion_marker:
        marker_path = completion_marker_path(output_dir, opt_test_id, idx)

        if not os.path.isfile(marker_path) or os.path.getsize(marker_path) == 0:
            return False

    return True


def find_pending_indices(
    start_idx,
    end_idx,
    output_dir,
    opt_test_id,
    skipped_indices=None,
    require_completion_marker=False,
):
    skipped_indices = skipped_indices or set()

    return [
        idx
        for idx in range(start_idx, end_idx)
        if idx not in skipped_indices
        and not structure_outputs_are_complete(
            output_dir,
            opt_test_id,
            idx,
            require_completion_marker=require_completion_marker,
        )
    ]


def load_skipped_indices(skip_indices_file):
    """Read persistent indices that the first-pass watchdog chose to bypass."""

    if skip_indices_file is None or not os.path.isfile(skip_indices_file):
        return set()

    with open(skip_indices_file, "r") as handle:
        return {
            int(line.strip())
            for line in handle
            if line.strip()
        }


def file_signature(file_path):
    if not os.path.isfile(file_path):
        return None

    stat_result = os.stat(file_path)
    modified_ns = getattr(
        stat_result,
        "st_mtime_ns",
        int(stat_result.st_mtime * 1000000000),
    )
    return (stat_result.st_size, modified_ns)


def list_rhino_instances():
    result = subprocess.run(
        [RHINOCODE, "list", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Could not list Rhino instances.")

    # RhinoCode can print a stale-pipe warning before the requested JSON.
    decoder = json.JSONDecoder()
    parsed_lists = []

    for position, character in enumerate(result.stdout):
        if character != "[":
            continue

        try:
            value, unused_end = decoder.raw_decode(result.stdout[position:])
        except ValueError:
            continue

        if isinstance(value, list):
            parsed_lists.append(value)

    if not parsed_lists:
        raise RuntimeError(
            "RhinoCode did not return a JSON instance list. Output: {}".format(
                result.stdout.strip()
            )
        )

    return parsed_lists[-1]


def instance_process_ids(instances):
    return {int(instance["processId"]) for instance in instances}


def launch_rhino(startup_model_file):
    # `open -na` starts a separate Rhino process and the model provides the
    # active document required by rhinoscriptsyntax and ScriptEditor.
    subprocess.Popen(["open", "-na", RHINO_APP, startup_model_file])


def wait_for_new_rhino_instance(existing_process_ids):
    deadline = time.time() + RHINO_START_TIMEOUT_SECONDS
    last_error = None

    while time.time() < deadline:
        try:
            instances = list_rhino_instances()

            for instance in instances:
                process_id = int(instance["processId"])

                if process_id not in existing_process_ids:
                    return instance

        except Exception as exc:
            last_error = exc

        time.sleep(RHINO_POLL_SECONDS)

    message = "A new Rhino script-server instance did not appear within {} seconds.".format(
        RHINO_START_TIMEOUT_SECONDS
    )

    if last_error is not None:
        message += " Last RhinoCode error: {}".format(last_error)

    raise RuntimeError(message)


def wait_for_rhino_document(process_id):
    deadline = time.time() + RHINO_DOCUMENT_TIMEOUT_SECONDS

    while time.time() < deadline:
        instances = list_rhino_instances()

        for instance in instances:
            if int(instance["processId"]) != process_id:
                continue

            if instance.get("activeDoc"):
                return

        time.sleep(RHINO_POLL_SECONDS)

    raise RuntimeError(
        "Rhino process {} did not create an active document within {} seconds.".format(
            process_id,
            RHINO_DOCUMENT_TIMEOUT_SECONDS,
        )
    )


def process_is_running(process_id):
    try:
        os.kill(int(process_id), 0)
        return True
    except OSError:
        return False


def wait_for_process_exit(process_id, timeout_seconds):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if not process_is_running(process_id):
            return True

        time.sleep(RHINO_POLL_SECONDS)

    return not process_is_running(process_id)


def stop_rhino(pipe_id, process_id):
    if not process_is_running(process_id):
        print("Rhino process {} was already stopped.".format(process_id))
        return

    try:
        result = subprocess.run(
            [RHINOCODE, "--rhino", pipe_id, "command", "_Exit"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=RHINO_GRACEFUL_STOP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        result = None

    if wait_for_process_exit(process_id, RHINO_GRACEFUL_STOP_TIMEOUT_SECONDS):
        print("Rhino process {} exited normally.".format(process_id))
        return

    if result is not None and result.returncode != 0:
        print(
            "Rhino's Exit command failed: {}".format(
                result.stderr.strip() or "unknown RhinoCode error"
            )
        )

    print("Rhino did not exit normally; sending SIGTERM to process {}.".format(process_id))

    try:
        os.kill(int(process_id), signal.SIGTERM)
    except OSError:
        return

    if wait_for_process_exit(process_id, RHINO_FORCED_STOP_TIMEOUT_SECONDS):
        return

    print("Rhino did not respond to SIGTERM; sending SIGKILL.")

    try:
        os.kill(int(process_id), signal.SIGKILL)
    except OSError:
        pass


def wait_for_worker_results(indices, process_id):
    deadline = time.time() + WORKER_TIMEOUT_SECONDS

    while time.time() < deadline:
        config = read_config()

        if config.get("worker_status") == "complete":
            results = config.get("results", [])

            if len(results) != len(indices):
                raise RuntimeError(
                    "Rhino returned {} result(s), but this batch requested {} structure(s).".format(
                        len(results),
                        len(indices),
                    )
                )

            return results

        if not process_is_running(process_id):
            raise RuntimeError(
                "Rhino process {} exited before completing structures {}.".format(
                    process_id,
                    indices,
                )
            )

        time.sleep(RHINO_POLL_SECONDS)

    raise RuntimeError(
        "Rhino did not finish structures {} within {} seconds.".format(
            indices,
            WORKER_TIMEOUT_SECONDS,
        )
    )


def run_batch_in_rhino(indices, worker_config, startup_model_file):
    print("Launching Rhino for structures {}...".format(indices))

    existing_process_ids = instance_process_ids(list_rhino_instances())

    batch_config = dict(worker_config)
    batch_config.update(
        {
            "indices": indices,
            "worker_status": "queued",
            "results": [],
        }
    )
    write_config(batch_config)

    launch_rhino(startup_model_file)

    instance = wait_for_new_rhino_instance(existing_process_ids)
    pipe_id = instance["pipeId"]
    process_id = int(instance["processId"])

    print("Using Rhino process {} for this batch.".format(process_id))

    try:
        wait_for_rhino_document(process_id)

        script_editor_command = '_-ScriptEditor _Run "{}"'.format(SCRIPT_PATH)
        result = subprocess.run(
            [
                RHINOCODE,
                "--rhino",
                pipe_id,
                "command",
                script_editor_command,
            ],
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "RhinoCode failed to submit structures {} with exit code {}.".format(
                    indices,
                    result.returncode,
                )
            )

        return wait_for_worker_results(indices, process_id)

    finally:
        print("Stopping Rhino process {}...".format(process_id))
        stop_rhino(pipe_id, process_id)


def parse_supervisor_arguments():
    parser = argparse.ArgumentParser(
        description="Generate MiDShip Rhino structures in fresh-process batches."
    )
    parser.add_argument(
        "--batch-id",
        default=DEFAULT_BATCH_ID,
        help="Candidate batch name inside the pipeline experiments directory.",
    )
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--updated-parameters-csv",
        default=None,
        help="Rhino-cleaned parameter checkpoint written after generation.",
    )
    parser.add_argument(
        "--error-indices-csv",
        default=None,
        help="Persistent CSV containing indices that failed CAD generation.",
    )
    parser.add_argument("--start-idx", type=int, default=START_IDX)
    parser.add_argument("--end-idx", type=int, default=END_IDX)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--startup-model", default=STARTUP_MODEL_FILE)
    parser.add_argument(
        "--skip-indices-file",
        default=None,
        help="Text file containing one persistently failed index per line.",
    )
    parser.add_argument(
        "--checkpoint-every-design",
        action="store_true",
        help=(
            "Initialize Results_Updated.csv with zeros, persist each "
            "successful Rhino row immediately, and require a per-design "
            "completion marker when resuming."
        ),
    )
    parser.add_argument(
        "--print-next-pending",
        action="store_true",
        help="Print the next non-skipped incomplete index and exit.",
    )
    return parser.parse_args()


def load_updated_rows(
    input_header,
    input_rows,
    updated_csv,
    initialize_with_zeros=False,
):
    if not os.path.isfile(updated_csv):
        if initialize_with_zeros:
            return [
                [0.0] * len(input_header)
                for unused_row in input_rows
            ]

        return [list(row) for row in input_rows]

    updated_header, updated_rows = read_csv_table(updated_csv)

    if updated_header != input_header or len(updated_rows) != len(input_rows):
        print(
            "Ignoring incompatible updated-parameter file: {}".format(
                updated_csv
            )
        )

        if initialize_with_zeros:
            return [
                [0.0] * len(input_header)
                for unused_row in input_rows
            ]

        return [list(row) for row in input_rows]

    return updated_rows


def write_error_indices(error_csv, error_indices):
    write_csv_table(
        error_csv,
        ["error_idx"],
        [[idx] for idx in sorted(error_indices)],
    )


def load_error_indices(error_csv):
    if not os.path.isfile(error_csv):
        return set()

    header, rows = read_csv_table(error_csv)

    if header != ["error_idx"]:
        print("Ignoring incompatible error-index file: {}".format(error_csv))
        return set()

    return {
        int(row[0])
        for row in rows
        if row
    }


def supervisor_main():
    from tqdm import tqdm

    if not os.path.exists(RHINOCODE):
        raise RuntimeError("Could not find rhinocode at {}".format(RHINOCODE))

    args = parse_supervisor_arguments()
    experiment_dir = os.path.join(EXPERIMENTS_DIR, args.batch_id)

    # Running without options uses the active dataset block above. Historical
    # constraint experiments that pass only a different batch ID retain their
    # established experiment-folder behavior.
    if args.output_dir is None:
        if args.batch_id == DEFAULT_BATCH_ID:
            output_dir = DEFAULT_OUTPUT_DIR
        else:
            output_dir = os.path.join(experiment_dir, "structures")
    else:
        output_dir = args.output_dir

    if args.input_csv is None:
        if args.batch_id == DEFAULT_BATCH_ID:
            input_csv = DEFAULT_INPUT_CSV
        else:
            input_csv = os.path.join(
                experiment_dir,
                "{}_X_Results.csv".format(args.batch_id),
            )
    else:
        input_csv = args.input_csv

    output_dir = os.path.abspath(output_dir)
    input_csv = os.path.abspath(input_csv)

    if not os.path.isfile(input_csv):
        raise RuntimeError("Could not find input CSV: {}".format(input_csv))

    if not os.path.isfile(args.startup_model):
        raise RuntimeError(
            "Could not find Rhino startup model: {}".format(args.startup_model)
        )

    os.makedirs(output_dir, exist_ok=True)

    input_header, input_rows = read_csv_table(input_csv)
    row_count = len(input_rows)
    end_idx = row_count if args.end_idx is None else min(args.end_idx, row_count)

    if args.start_idx < 0 or args.start_idx >= end_idx:
        raise RuntimeError(
            "Requested index range [{}, {}) is empty for {} input rows.".format(
                args.start_idx,
                end_idx,
                row_count,
            )
        )

    if args.updated_parameters_csv is None:
        if args.batch_id == DEFAULT_BATCH_ID:
            updated_csv = DEFAULT_UPDATED_PARAMETERS_CSV
        else:
            updated_csv = os.path.join(
                output_dir,
                "{}_X_Results_Updated.csv".format(args.batch_id),
            )
    else:
        updated_csv = args.updated_parameters_csv

    if args.error_indices_csv is None:
        if args.batch_id == DEFAULT_BATCH_ID:
            error_csv = DEFAULT_ERROR_INDICES_CSV
        else:
            error_csv = os.path.join(
                output_dir,
                "{}_design_error_idx_ALL.csv".format(args.batch_id),
            )
    else:
        error_csv = args.error_indices_csv

    updated_csv = os.path.abspath(updated_csv)
    error_csv = os.path.abspath(error_csv)
    updated_rows = load_updated_rows(
        input_header,
        input_rows,
        updated_csv,
        initialize_with_zeros=args.checkpoint_every_design,
    )

    # Make pending work explicit before Rhino starts.  A row remains all zeros
    # until that individual design has completed successfully.
    if args.checkpoint_every_design and not os.path.isfile(updated_csv):
        write_csv_table(updated_csv, input_header, updated_rows)

    skipped_indices = load_skipped_indices(args.skip_indices_file)

    pending_indices = find_pending_indices(
        args.start_idx,
        end_idx,
        output_dir,
        args.batch_id,
        skipped_indices,
        require_completion_marker=args.checkpoint_every_design,
    )

    if args.print_next_pending:
        print(pending_indices[0] if pending_indices else "NONE")
        return

    if not pending_indices:
        print(
            "No incomplete structures were found from {} to {}.".format(
                args.start_idx,
                end_idx - 1,
            )
        )

        if skipped_indices:
            print(
                "Skipped {} persistently failed structure(s).".format(
                    len(skipped_indices)
                )
            )

        return

    print(
        "Found {} incomplete structure(s). The next design is {}.".format(
            len(pending_indices),
            pending_indices[0],
        )
    )

    worker_config = {
        "input_csv": input_csv,
        "output_dir": output_dir,
        "opt_test_id": args.batch_id,
        "checkpoint_every_design": args.checkpoint_every_design,
        "updated_csv": updated_csv,
        "error_csv": error_csv,
    }
    all_results = []
    error_indices = load_error_indices(error_csv)

    with tqdm(
        total=len(pending_indices),
        desc="Generating {} structures".format(args.batch_id),
        unit="design",
        dynamic_ncols=True,
    ) as progress:
        for batch_start in range(0, len(pending_indices), args.batch_size):
            batch_indices = pending_indices[
                batch_start:batch_start + args.batch_size
            ]
            batch_results = run_batch_in_rhino(
                batch_indices,
                worker_config,
                os.path.abspath(args.startup_model),
            )
            all_results.extend(batch_results)

            # The Rhino worker owns persistence in per-design mode.  Keep the
            # established after-batch persistence unchanged for older callers.
            if not args.checkpoint_every_design:
                for result in batch_results:
                    idx = result["idx"]

                    if result["status"] == "success":
                        updated_rows[idx] = result["params"]
                        error_indices.discard(idx)
                    else:
                        updated_rows[idx] = [0.0] * len(input_header)
                        error_indices.add(idx)

                write_csv_table(updated_csv, input_header, updated_rows)
                write_error_indices(error_csv, error_indices)

            progress.update(len(batch_results))

    failures = [result for result in all_results if result["status"] != "success"]

    print(
        "Finished {} structures: {} succeeded and {} failed.".format(
            len(all_results),
            len(all_results) - len(failures),
            len(failures),
        )
    )

    for failure in failures:
        print("  Structure {} failed: {}".format(failure["idx"], failure["message"]))

    # Signal the watchdog when one or more designs remain incomplete. The next
    # supervisor run discovers complete designs from their four output files
    # and submits only the failed or incomplete indices to a fresh Rhino.
    if failures:
        raise RuntimeError(
            "{} structure(s) remain incomplete after this Rhino run.".format(
                len(failures)
            )
        )


def process_one_index(idx, vectors, output_dir, opt_test_id, Structure_3H):
    file_stem = structure_file_stem(opt_test_id, idx)
    output_paths = expected_structure_paths(output_dir, opt_test_id, idx)
    signatures_before = {
        output_path: file_signature(output_path)
        for output_path in output_paths
    }

    try:
        # Keep this sequence synchronized with
        # Generate_Structures_After_Optimization.py.
        hull = Structure_3H(
            vectors[idx],
            path=output_dir,
            id=file_stem,
        )
        hull.make_Structure()

    except Exception as exc:
        error_message = "{}: {}".format(type(exc).__name__, exc)
        print("Failed {}: {}".format(file_stem, error_message))
        traceback.print_exc()
        return {
            "idx": idx,
            "status": "failed",
            "message": error_message,
            "params": [],
            "files": output_paths,
        }

    missing_or_stale = []

    for output_path in output_paths:
        signature_after = file_signature(output_path)

        if signature_after is None or signature_after[0] == 0:
            missing_or_stale.append(output_path)
        elif signature_after == signatures_before[output_path]:
            missing_or_stale.append(output_path)

    if missing_or_stale:
        message = "{} expected output file(s) were missing, empty, or not rewritten.".format(
            len(missing_or_stale)
        )
        print("Failed {}: {}".format(file_stem, message))
        return {
            "idx": idx,
            "status": "failed",
            "message": message,
            "params": [],
            "files": missing_or_stale,
        }

    params = [float(value) for value in hull.params]
    print("Completed {} with {} output files.".format(file_stem, len(output_paths)))

    return {
        "idx": idx,
        "status": "success",
        "message": "",
        "params": params,
        "files": output_paths,
    }


def worker_main():
    import pandas as pd
    import Rhino
    import rhinoscriptsyntax as rs

    # Structure_3H remains in Rhino_Macros. This copy isolates only the batch
    # orchestration and its generated outputs inside the repair pipeline.
    if RHINO_MACROS_DIR not in sys.path:
        sys.path.append(RHINO_MACROS_DIR)

    from Parametric_Structure_V2 import Structure_3H

    config = read_config()
    indices = config["indices"]
    input_csv = config["input_csv"]
    output_dir = config["output_dir"]
    opt_test_id = config["opt_test_id"]
    checkpoint_every_design = config.get(
        "checkpoint_every_design",
        False,
    )

    config["worker_status"] = "running"
    write_config(config)

    # Match the original macro: read the complete parameter table and clean it
    # once before generating any structures in this Rhino process.
    parameter_data = pd.read_csv(input_csv)
    vectors = parameter_data.to_numpy().copy()
    vectors = Structure_3H.clean_Struct_Params(vectors)

    # This optional state belongs to the long-running repaired-random-design
    # workflow.  Its updated table begins as zeros; one row is replaced after
    # each successful Rhino structure.
    if checkpoint_every_design:
        input_header, input_rows = read_csv_table(input_csv)
        updated_csv = config["updated_csv"]
        error_csv = config["error_csv"]
        updated_rows = load_updated_rows(
            input_header,
            input_rows,
            updated_csv,
            initialize_with_zeros=True,
        )
        error_indices = load_error_indices(error_csv)

    print("Running Rhino structure batch {}...".format(indices))
    results = []

    for idx in indices:
        print("Processing structure {}...".format(idx))

        try:
            result = process_one_index(
                idx,
                vectors,
                output_dir,
                opt_test_id,
                Structure_3H,
            )
        except Exception as exc:
            error_message = "{}: {}".format(type(exc).__name__, exc)
            print("Skipping structure {}: {}".format(idx, error_message))
            traceback.print_exc()

            result = {
                "idx": idx,
                "status": "failed",
                "message": error_message,
                "params": [],
                "files": [],
            }

        results.append(result)

        if checkpoint_every_design:
            if result["status"] == "success":
                # Ensure all four generated files have reached the filesystem
                # before the matching parameter row is made visible.
                for output_path in result["files"]:
                    with open(output_path, "rb") as output_handle:
                        os.fsync(output_handle.fileno())

                updated_rows[idx] = result["params"]
                error_indices.discard(idx)
                write_csv_table(updated_csv, input_header, updated_rows)
                write_error_indices(error_csv, error_indices)

                # The marker is the final commit.  If Rhino is killed before
                # this point, the design remains pending and will be rerun.
                write_completion_marker(
                    output_dir,
                    opt_test_id,
                    idx,
                )

            else:
                # Failed and unprocessed rows remain visibly zero.  No marker
                # is written, so the watchdog retries this index in fresh Rhino.
                updated_rows[idx] = [0.0] * len(input_header)
                error_indices.add(idx)
                write_csv_table(updated_csv, input_header, updated_rows)
                write_error_indices(error_csv, error_indices)

            # Publish partial status for diagnostics.  Resume decisions use
            # the durable per-design marker rather than this shared JSON file.
            config["results"] = list(results)
            write_config(config)

    config["results"] = results
    config["worker_status"] = "complete"
    write_config(config)

    try:
        rs.Command("_-ClearUndo", False)
        rs.ClearCommandHistory()
    except Exception:
        pass

    Rhino.RhinoApp.Exit(False)


# RhinoCode does not guarantee that a submitted script uses `__main__` as its
# module name. Detect Rhino first so the worker always starts inside Rhino.
if is_rhino():
    worker_main()
elif __name__ == "__main__":
    supervisor_main()
