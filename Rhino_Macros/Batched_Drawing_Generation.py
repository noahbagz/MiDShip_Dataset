#!/usr/bin/env python3
"""
Run drawing generation in Rhino batches so the Rhino process is restarted
every `BATCH_SIZE` designs.

Run this file from normal Python, not from inside Rhino:

    python3 Rhino_Macros/Batched_Drawing_Generation.py

Use a smaller range for a repeatable smoke test:

    python3 Rhino_Macros/Batched_Drawing_Generation.py \
        --start-idx 0 --end-idx 10 --batch-size 1

The supervisor scans the requested model range and the output directory before
launching Rhino. Existing complete drawings are skipped, including gaps in the
numeric sequence. If Rhino fails, the supervisor rescans completed outputs and
continues in a fresh Rhino process. A design that prevents progress three times
is recorded in the output directory and skipped.

Rhino 8 note:
    `rhinocode` needs Rhino's script server running. If needed, add
    `StartScriptServer` to Rhino startup commands.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback


SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "_batched_drawing_config.json")

# Select exactly one dataset block.  The random dataset is active below;
# comment it and uncomment one complete alternative block for repaired or SGLD
# drawing generation.

# Generate drawings for the randomly generated structures.
MODEL_DIR = os.path.join(
    MAIN_DIR,
    "MiDShip_Dataset",
    "Random_Structures",
)
OUTPUT_DIR = os.path.join(MODEL_DIR, "Dataset_Drawings")
FILE_STEM = "random_test_design_{}"
PARAMETER_CSV = os.path.join(
    MODEL_DIR, "random_test_design_Parameters_All.csv"
)
END_IDX = None

# Generate drawings for the repaired structures.
# MODEL_DIR = os.path.join(
#     MAIN_DIR,
#     "MiDShip_Dataset",
#     "Repaired_Structures",
# )
# OUTPUT_DIR = os.path.join(MODEL_DIR, "Dataset_Drawings")
# FILE_STEM = "repaired_random_design_design_{}"
# PARAMETER_CSV = os.path.join(
#     MODEL_DIR, "repaired_random_design_Parameters_Updated.csv"
# )
# END_IDX = None

# Generate drawings for the SGLD structures.
# MODEL_DIR = os.path.join(
#     MAIN_DIR,
#     "MiDShip_Dataset",
#     "SGLD_Gen_Structures",
# )
# OUTPUT_DIR = os.path.join(MODEL_DIR, "Dataset_Drawings")
# FILE_STEM = "sgld_design_{}"
# PARAMETER_CSV = os.path.join(
#     MODEL_DIR, "sgld_design_parameters_Updated.csv"
# )
# END_IDX = None

FAILED_INDICES_PATH = os.path.join(
    OUTPUT_DIR,
    "_batched_drawing_failed_indices.csv",
)

START_IDX = 0
BATCH_SIZE = 50

RHINO_APP = "/Applications/Rhino 8.app"
RHINOCODE = os.path.join(RHINO_APP, "Contents", "Resources", "bin", "rhinocode")
RHINO_START_TIMEOUT_SECONDS = 10
RHINO_DOCUMENT_TIMEOUT_SECONDS = 10
RHINO_GRACEFUL_STOP_TIMEOUT_SECONDS = 10
RHINO_FORCED_STOP_TIMEOUT_SECONDS = 10
RHINO_POLL_SECONDS = 1
WORKER_TIMEOUT_SECONDS = 1800
EXPECTED_OUTPUT_FILE_COUNT = 8
MAX_CONSECUTIVE_NO_PROGRESS_FAILURES = 3

# Each drawing contains three views. The worker creates a normal PDF and a
# bounding-box PDF for each view, followed by two CSV annotation tables.
DRAWING_SLICE_NAMES = (
    "Midship Section IWO of Web Frame",
    "Transverse Bulkhead",
    "Midship Section of Long. Structure",
)


def file_stem_prefix():
    # FILE_STEM contains one numeric placeholder. Using the text before that
    # placeholder keeps the directory scan synchronized with model filenames.
    return FILE_STEM.split("{}", 1)[0]


def scan_model_indices(start_idx, end_idx):
    # Only .3dm files that exactly match the configured numeric naming pattern
    # are eligible for drawing generation.
    pattern = re.compile(
        r"^{}(\d+)\.3dm$".format(re.escape(file_stem_prefix()))
    )
    model_indices = set()

    for file_name in os.listdir(MODEL_DIR):
        match = pattern.match(file_name)

        if match is None:
            continue

        idx = int(match.group(1))

        if start_idx <= idx < end_idx:
            model_indices.add(idx)

    return model_indices


def expected_drawing_paths(idx):
    # List the exact eight outputs so unrelated files with the same design
    # prefix cannot make an incomplete drawing appear complete.
    file_name = FILE_STEM.format(idx)
    output_paths = [
        os.path.join(OUTPUT_DIR, file_name + "_Slice_Elements.csv"),
        os.path.join(OUTPUT_DIR, file_name + "_Drawing_Annotations.csv"),
    ]

    for slice_name in DRAWING_SLICE_NAMES:
        output_paths.extend(
            [
                os.path.join(
                    OUTPUT_DIR,
                    "{}_{}.pdf".format(file_name, slice_name),
                ),
                os.path.join(
                    OUTPUT_DIR,
                    "{}_{}_with_BBoxes.pdf".format(file_name, slice_name),
                ),
            ]
        )

    return output_paths


def drawing_outputs_are_complete(idx):
    # A design is complete only when all eight expected files exist and are
    # nonempty. Partial output from a crashed Rhino process will be repaired
    # automatically during the next supervisor attempt.
    output_paths = expected_drawing_paths(idx)

    if len(output_paths) != EXPECTED_OUTPUT_FILE_COUNT:
        raise RuntimeError(
            "Expected {} drawing paths, but constructed {}.".format(
                EXPECTED_OUTPUT_FILE_COUNT,
                len(output_paths),
            )
        )

    for output_path in output_paths:
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            return False

    return True


def scan_generated_indices(start_idx, end_idx):
    # Check every model index individually. An index is skipped only when all
    # eight of its expected output files already exist.
    generated_indices = set()

    for idx in sorted(scan_model_indices(start_idx, end_idx)):
        if drawing_outputs_are_complete(idx):
            generated_indices.add(idx)

    return generated_indices


def find_pending_indices(start_idx, end_idx, failed_indices=None):
    # Compare the numeric model sequence with the numeric output sequence.
    # Sorting guarantees that batches always advance from the earliest model
    # whose drawing output has not yet been generated.
    if failed_indices is None:
        failed_indices = set()

    model_indices = scan_model_indices(start_idx, end_idx)
    generated_indices = scan_generated_indices(start_idx, end_idx)

    return sorted(model_indices - generated_indices - set(failed_indices))


def load_failed_indices():
    # Previously skipped designs remain skipped when the supervisor is
    # restarted. Delete the CSV manually if those designs should be retried.
    if not os.path.isfile(FAILED_INDICES_PATH):
        return set()

    with open(FAILED_INDICES_PATH, "r", newline="") as handle:
        reader = csv.DictReader(handle)

        return {
            int(row["failed_idx"])
            for row in reader
            if row.get("failed_idx")
        }


def write_failed_indices(failed_indices):
    # Replace the complete file atomically so an interruption cannot leave a
    # partially written failed-index record.
    temporary_path = FAILED_INDICES_PATH + ".tmp"

    with open(temporary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["failed_idx"])

        for idx in sorted(failed_indices):
            writer.writerow([idx])

    os.replace(temporary_path, FAILED_INDICES_PATH)


def is_rhino():
    try:
        import Rhino  # noqa: F401
        import rhinoscriptsyntax  # noqa: F401
        return True
    except Exception:
        return False


def write_config(config):
    # The normal Python supervisor and Rhino worker use this small file to
    # exchange the requested indices and the result of each design.
    with open(CONFIG_PATH, "w") as handle:
        json.dump(config, handle, indent=2)


def read_config():
    with open(CONFIG_PATH, "r") as handle:
        return json.load(handle)


def list_rhino_instances():
    # Ask RhinoCode for instances whose Python script servers are available.
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
    # Scan for the last valid JSON list instead of assuming stdout is pure JSON.
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
    # Normalize process IDs because RhinoCode may serialize them as strings.
    return {int(instance["processId"]) for instance in instances}


def launch_rhino(initial_model_file):
    # `open -na` requests a separate Rhino process even if Rhino is running.
    # Opening the first model at launch also avoids Rhino's document-free
    # start screen, which cannot host rhinoscriptsyntax drawing commands.
    subprocess.Popen(["open", "-na", RHINO_APP, initial_model_file])


def wait_for_new_rhino_instance(existing_process_ids):
    # Rhino startup time varies. Polling avoids selecting an older Rhino
    # session or trying to submit the script before the server is ready.
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
    # Do not submit the worker until the model passed to `open` is active.
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
    # Ask Rhino itself to exit first. Operating-system signals make Rhino
    # treat an otherwise normal batch restart as a crash and can open its
    # error-reporting window.
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

    # A signal is retained only as a fallback for an unresponsive Rhino
    # process. Forced termination can still trigger Rhino's error reporter.
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
    # RhinoCode queues a script and can return before that script has run.
    # Wait for the Rhino worker's explicit completion marker before shutdown.
    deadline = time.time() + WORKER_TIMEOUT_SECONDS

    while time.time() < deadline:
        config = read_config()

        if config.get("worker_status") == "complete":
            results = config.get("results", [])
            expected_result_count = len(indices)

            if len(results) != expected_result_count:
                raise RuntimeError(
                    "Rhino returned {} result(s), but this batch requested {} design(s).".format(
                        len(results),
                        expected_result_count,
                    )
                )

            return results

        if not process_is_running(process_id):
            raise RuntimeError(
                "Rhino process {} exited before completing designs {}.".format(
                    process_id,
                    indices,
                )
            )

        time.sleep(RHINO_POLL_SECONDS)

    raise RuntimeError(
        "Rhino did not finish designs {} within {} seconds.".format(
            indices,
            WORKER_TIMEOUT_SECONDS,
        )
    )


def run_batch_in_rhino(indices):
    print("Launching Rhino for designs {}...".format(indices))

    # Capture existing instances before launch so this batch cannot attach to
    # an unrelated Rhino window that the user already has open.
    existing_process_ids = instance_process_ids(list_rhino_instances())

    write_config(
        {
            "indices": indices,
            "worker_status": "queued",
            "results": [],
        }
    )

    initial_model_file = os.path.join(
        MODEL_DIR,
        FILE_STEM.format(indices[0]) + ".3dm",
    )
    launch_rhino(initial_model_file)

    instance = wait_for_new_rhino_instance(existing_process_ids)
    pipe_id = instance["pipeId"]
    process_id = int(instance["processId"])

    print("Using Rhino process {} for this batch.".format(process_id))

    try:
        wait_for_rhino_document(process_id)

        # On macOS, `rhinocode script` can execute CPython without binding a
        # Rhino document. Run the same file through Rhino's ScriptEditor
        # command so rhinoscriptsyntax receives the current document context.
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
                "RhinoCode failed to submit designs {} with exit code {}.".format(
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
        description="Generate Rhino drawings in fresh-process batches."
    )
    parser.add_argument("--start-idx", type=int, default=START_IDX)
    parser.add_argument("--end-idx", type=int, default=END_IDX)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def supervisor_main():
    if not os.path.exists(RHINOCODE):
        raise RuntimeError("Could not find rhinocode at {}".format(RHINOCODE))

    args = parse_supervisor_arguments()

    # By default the selected subset's released parameter table defines the
    # drawing count. ``--end-idx`` can still limit a smoke test explicitly.
    if args.end_idx is None:
        with open(PARAMETER_CSV, "r", newline="") as handle:
            end_idx = sum(1 for unused_row in csv.reader(handle)) - 1
    else:
        end_idx = args.end_idx

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Designs that repeatedly prevent forward progress are stored separately
    # so a resumed run can continue without getting stuck on the same model.
    failed_indices = load_failed_indices()
    pending_indices = find_pending_indices(
        args.start_idx,
        end_idx,
        failed_indices,
    )

    if not pending_indices:
        print(
            "No ungenerated model drawings were found from {} to {}.".format(
                args.start_idx,
                end_idx - 1,
            )
        )
        return

    print(
        "Found {} ungenerated model(s). The next design is {}.".format(
            len(pending_indices),
            pending_indices[0],
        )
    )

    all_results = []
    failure_index = None
    consecutive_no_progress_failures = 0

    # Rebuild the pending queue after every Rhino run. This is important when
    # Rhino crashes partway through a batch: completed drawings are retained,
    # skipped during the rescan, and the next fresh Rhino process resumes at
    # the first genuinely incomplete design.
    while True:
        pending_before = find_pending_indices(
            args.start_idx,
            end_idx,
            failed_indices,
        )

        if not pending_before:
            break

        batch_indices = pending_before[:args.batch_size]
        first_incomplete_idx = batch_indices[0]
        generated_before = scan_generated_indices(
            args.start_idx,
            end_idx,
        )
        run_error = None

        try:
            batch_results = run_batch_in_rhino(batch_indices)
            all_results.extend(batch_results)

        except Exception as exc:
            # A Rhino crash, startup failure, submission failure, or worker
            # timeout should not terminate a long unattended run. The output
            # rescan below determines exactly where the next process resumes.
            run_error = "{}: {}".format(type(exc).__name__, exc)
            print("Rhino batch failed: {}".format(run_error))

        generated_after = scan_generated_indices(
            args.start_idx,
            end_idx,
        )
        newly_completed = generated_after - generated_before

        if newly_completed:
            run_description = "interrupted batch" if run_error else "Rhino batch"
            print(
                "The {} completed {} design(s). "
                "Restarting Rhino for the remaining designs.".format(
                    run_description,
                    len(newly_completed)
                )
            )
            failure_index = None
            consecutive_no_progress_failures = 0
            continue

        # No design became complete during this attempt. Count repeated
        # failures only while the same design remains first in the queue.
        if failure_index == first_incomplete_idx:
            consecutive_no_progress_failures += 1
        else:
            failure_index = first_incomplete_idx
            consecutive_no_progress_failures = 1

        print(
            "Design {} made no progress (attempt {} of {}).".format(
                first_incomplete_idx,
                consecutive_no_progress_failures,
                MAX_CONSECUTIVE_NO_PROGRESS_FAILURES,
            )
        )

        if run_error is not None:
            print("  Last batch error: {}".format(run_error))

        if (
            consecutive_no_progress_failures
            >= MAX_CONSECUTIVE_NO_PROGRESS_FAILURES
        ):
            failed_indices.add(first_incomplete_idx)
            write_failed_indices(failed_indices)
            print(
                "Recorded design {} in {} and continuing.".format(
                    first_incomplete_idx,
                    FAILED_INDICES_PATH,
                )
            )
            failure_index = None
            consecutive_no_progress_failures = 0

    failures = [result for result in all_results if result["status"] != "success"]
    generated_indices = scan_generated_indices(args.start_idx, end_idx)

    print(
        "Finished with {} complete drawing set(s), {} worker-reported "
        "failure(s), and {} persistently skipped design(s).".format(
            len(generated_indices),
            len(failures),
            len(failed_indices),
        )
    )

    for failure in failures:
        print("  Design {} skipped: {}".format(failure["idx"], failure["message"]))


def file_signature(path):
    # Size and nanosecond modification time let the worker distinguish a PDF
    # rewritten in this run from a stale file left by an earlier test.
    if not os.path.isfile(path):
        return None

    stat_result = os.stat(path)
    modified_ns = getattr(
        stat_result,
        "st_mtime_ns",
        int(stat_result.st_mtime * 1000000000),
    )
    return (stat_result.st_size, modified_ns)


def find_rhino_document(Rhino, file_path=None):
    # ActiveDoc is unreliable on macOS while a command is running. Resolve the
    # model from Rhino's open-document table and retain that explicit object.
    if file_path is not None:
        document = Rhino.RhinoDoc.FromFilePath(file_path)

        if document is not None:
            return document

    open_documents = list(Rhino.RhinoDoc.OpenDocuments(True))

    if open_documents:
        return open_documents[-1]

    raise RuntimeError("Rhino has no open document available to the worker.")


def process_one_index(idx):
    import Rhino
    import rhino_2D_Drawing as r2d
    import scriptcontext as sc

    file_name = FILE_STEM.format(idx)
    model_file = os.path.join(MODEL_DIR, file_name + ".3dm")
    struct_file = os.path.join(MODEL_DIR, file_name + "_Structural_Elements.csv")

    if not os.path.exists(model_file) or not os.path.exists(struct_file):
        message = "The model or structural-elements CSV is missing."
        print("Skipping {} because {}".format(file_name, message.lower()))
        return {"idx": idx, "status": "failed", "message": message, "pdfs": []}

    dwg_gen = r2d.Rhino2DDrawing(MODEL_DIR, file_name, OUTPUT_DIR)
    expected_pdf_paths = []
    signatures_before = {}
    error_message = None

    try:
        # Keep this drawing-generation sequence synchronized with
        # Test_Drawing_Generation.py. The supervisor only changes when Rhino
        # starts and stops; it does not replace or reorder these operations.

        # Scripts launched through RhinoCode do not automatically populate
        # scriptcontext.doc as Rhino's editor does. Bind the blank startup
        # document before load_Data uses rhinoscriptsyntax.
        sc.doc = find_rhino_document(Rhino)

        dwg_gen.load_Data()

        # load_Data opens the requested model, replacing the active document.
        # Refresh the binding so the remaining unchanged drawing operations
        # and PDF export use that newly opened model.
        sc.doc = find_rhino_document(Rhino, model_file)

        dwg_gen.rename_Structural_Element_Classes()
        dwg_gen.extract_X_slice_Positions()

        for j in range(len(dwg_gen.df_Slices)):
            slice_name = dwg_gen.df_Slices["Slice_Name"][j]

            pdf_path = os.path.join(
                OUTPUT_DIR,
                "{}_{}.pdf".format(file_name, slice_name),
            )
            bbox_pdf_path = os.path.join(
                OUTPUT_DIR,
                "{}_{}_with_BBoxes.pdf".format(file_name, slice_name),
            )
            expected_pdf_paths.extend([pdf_path, bbox_pdf_path])

            signatures_before[pdf_path] = file_signature(pdf_path)
            signatures_before[bbox_pdf_path] = file_signature(bbox_pdf_path)

            dwg_gen.create_Slice(slice_name)
            dwg_gen.create_Layout(slice_name)
            scale, origin_s = dwg_gen.scale_Slice(slice_name)
            dwg_gen.create_Title_Block(slice_name, scale)
            dwg_gen.create_Info_Blocks(slice_name)
            dwg_gen.create_Bounding_Boxes(slice_name, scale, origin_s)
            dwg_gen.export_DWG(slice_name)

    except Exception as exc:
        error_message = "{}: {}".format(type(exc).__name__, exc)
        print("Failed {}: {}".format(file_name, error_message))
        traceback.print_exc()

    finally:
        try:
            dwg_gen.close_Doc()
        except Exception as exc:
            close_message = "Document close failed with {}: {}".format(
                type(exc).__name__,
                exc,
            )
            print("{}: {}".format(file_name, close_message))

            if error_message is None:
                error_message = close_message

    if error_message is not None:
        return {
            "idx": idx,
            "status": "failed",
            "message": error_message,
            "pdfs": expected_pdf_paths,
        }

    missing_or_stale = []

    for pdf_path in expected_pdf_paths:
        signature_after = file_signature(pdf_path)

        if signature_after is None or signature_after[0] == 0:
            missing_or_stale.append(pdf_path)
        elif signature_after == signatures_before[pdf_path]:
            missing_or_stale.append(pdf_path)

    if missing_or_stale:
        message = "{} expected PDF(s) were missing, empty, or not rewritten.".format(
            len(missing_or_stale)
        )
        print("Failed {}: {}".format(file_name, message))
        return {
            "idx": idx,
            "status": "failed",
            "message": message,
            "pdfs": missing_or_stale,
        }

    print("Completed {} with {} PDFs.".format(file_name, len(expected_pdf_paths)))
    return {
        "idx": idx,
        "status": "success",
        "message": "",
        "pdfs": expected_pdf_paths,
    }


def worker_main():
    import Rhino
    import rhinoscriptsyntax as rs

    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)

    config = read_config()
    indices = config["indices"]

    config["worker_status"] = "running"
    write_config(config)

    print("Running Rhino batch {}...".format(indices))

    results = []

    for idx in indices:
        print("Processing drawing {}...".format(idx))

        try:
            result = process_one_index(idx)
        except Exception as exc:
            # A problem with one drawing must not stop the remaining batch.
            # Record unexpected errors here as a final safety net because
            # process_one_index also performs setup outside its own try block.
            error_message = "{}: {}".format(type(exc).__name__, exc)
            print("Skipping drawing {}: {}".format(idx, error_message))
            traceback.print_exc()

            result = {
                "idx": idx,
                "status": "failed",
                "message": error_message,
                "pdfs": [],
            }

        results.append(result)

    # Write results before Rhino is stopped so the supervisor can verify that
    # every requested design actually produced fresh, nonempty PDF files.
    config["results"] = results
    config["worker_status"] = "complete"
    write_config(config)

    try:
        rs.Command("_-ClearUndo", False)
        rs.ClearCommandHistory()
    except Exception:
        pass

    # Exit from inside Rhino's document-aware scripting context. On macOS,
    # the external `_Exit` command can be ignored after close_Doc returns
    # Rhino to its document-free start screen.
    Rhino.RhinoApp.Exit(False)


# RhinoCode does not guarantee that a submitted script uses `__main__` as its
# module name. Detect Rhino first so the worker always starts when RhinoCode
# executes this file. Normal Python still uses the conventional main guard.
if is_rhino():
    worker_main()
elif __name__ == "__main__":
    supervisor_main()
