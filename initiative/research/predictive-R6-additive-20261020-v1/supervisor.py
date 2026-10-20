"""Private sequential C/R6 execution; no model selection, network or account operations."""

import csv
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

R6_DRIVER_SHAS = (
    "affa44363c530eed37878d083707d4fc50dcf1571d8dfcce12fd1c5a7d4bc581",
    "b472091d7a323e0165958ca01e6f023b21db9c3d2b10d3f79ba8962fb83efc5b",
)
R6_CELLS = "300b34417bc1bc91018a4bb50fa9440403e904ab7abcc7236e432e61932df56f"
R6_NOTEBOOK = "fac1723c32de2d27f59e10ebca7102327bb33346117a3fec3a58937ecf2d3a0e"
PUBLIC_ROSTER = "e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b"
R6_PUBLIC_CSV = "5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b"
TOTAL_SECONDS = 32400
C_ACCEPTED = "predictive-competition-accepted.json"
R6_ACCEPTED = "exact8cell-script-accepted.json"
R6_OUTPUTS = (
    "submission.csv", "_public_stack.csv", "_own.csv", "ours_sub.csv",
    "raptor_sub.csv", "raptor224_sub.csv",
)
COMBINED_ACCEPTED = "predictive-R6-additive-accepted.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check_c_receipt(receipt, binding, *, roster_sha256, studies, csv_bytes):
    require(
        receipt["status"] == "actual_predictive_exact_test_roster_submission_created"
        and receipt["mode"] == "competition"
        and receipt["test_roster_sha256"] == roster_sha256
        and receipt["actual_test_studies"] == studies
        and receipt["submission_csv_absent"] is False
        and receipt["CSV_bytes"] == len(csv_bytes)
        and receipt["CSV_sha256"] == digest(csv_bytes),
        "Exact C submission receipt/CSV/roster required",
    )
    require(
        len(receipt["GPU_names"]) == 2 and all("T4" in name for name in receipt["GPU_names"]),
        "Actual C child must use both T4 devices",
    )
    require(
        len(receipt["models"]) == 1
        and receipt["models"][0]["arm"] == "C"
        and receipt["models"][0]["trained_export_sha256"] == binding["C_model_sha256"]
        and receipt["models"][0]["candidate_sha256"] == binding["C_candidate_sha256"]
        and receipt["models"][0]["precision_accepted"] is True
        and receipt["models"][0]["memory_accepted"] is True,
        "Exact complete trained C model required",
    )
    require(
        {key: receipt[key] for key in binding["C_runtime"]} == binding["C_runtime"],
        "C runtime versions differ from actual trained qualification",
    )
    require(
        {key: receipt[key] for key in binding["C_serving"]} == binding["C_serving"],
        "C serving source or Dataset differs from actual qualification",
    )
    if roster_sha256 == PUBLIC_ROSTER:
        require(
            receipt["CSV_sha256"] == binding["standalone_C_public_CSV_sha256"],
            "Combined C public CSV differs from its original standalone accepted bytes",
        )


def _roster(path):
    raw = path.read_bytes()
    reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
    header = next(reader, None)
    require(
        header and all(header) and len(set(header)) == len(header) and "StudyInstanceUID" in header,
        "Unique official test.csv header including StudyInstanceUID required",
    )
    rows = list(reader)
    require(rows and all(len(row) == len(header) for row in rows),
            "Complete official study roster rows required")
    column = header.index("StudyInstanceUID")
    uids = [row[column] for row in rows]
    require(
        all(uid and uid == uid.strip() for uid in uids) and len(set(uids)) == len(uids),
        "Unique complete official study roster required",
    )
    return uids, digest(raw)


def _official_roster(input_root="/kaggle/input"):
    input_root = Path(input_root)
    roots = (
        input_root / "rsna-knee-abnormality-detection",
        input_root / "competitions/rsna-knee-abnormality-detection",
    )
    files = {
        (root / "test.csv").resolve() for root in roots
        if (root / "test.csv").is_file()
    }
    require(len(files) == 1, "One exact official competition test.csv mount required")
    return next(iter(files))


def _write(path, data):
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(c_source, r6_source, fusion_source, binding, *, working=None, roster_path=None):
    """Run in a fresh Kaggle session; optional filesystem arguments support CPU lifecycle tests."""
    started = time.monotonic()
    require(binding["mode"] == "candidate", "Only one complete combined candidate mode is supported")
    require(binding["total_watchdog_seconds"] == TOTAL_SECONDS, "One shared nine-hour budget required")
    for key, source in (("C", c_source), ("R6", r6_source), ("fusion", fusion_source)):
        require(isinstance(source, bytes), "Exact child/helper source bytes required")
        require(digest(source) == binding["source_sha256"][key], "Child/helper source hash differs")
    require(digest(r6_source) in R6_DRIVER_SHAS, "Original exact admitted R6 driver required")
    working = Path("/kaggle/working") if working is None else Path(working)
    root = working / "predictive-R6-additive-v1"
    stale = (*R6_OUTPUTS, C_ACCEPTED, R6_ACCEPTED, COMBINED_ACCEPTED,
             "submission.csv.tmp", "submission_receipt.json", "blend_receipt.json",
             "predictive-final-serving-v1")
    require(
        working.is_dir() and not root.exists() and not root.is_symlink()
        and not any((working / name).exists() or (working / name).is_symlink() for name in stale),
        "Fresh combined root, child roots, receipts and outputs required",
    )
    root.mkdir(mode=0o700)
    active = {"pid": None, "watchdog_kill_error": None}
    lock = threading.Lock()
    expired = threading.Event()
    stages = []

    def expire():
        expired.set()
        with lock:
            if active["pid"] is not None:
                try:
                    _kill_group(active["pid"])
                except OSError as error:
                    # The main thread still performs mandatory group cleanup on
                    # the expired wait. Keep callback failures observable.
                    active["watchdog_kill_error"] = type(error).__name__

    def remaining():
        seconds = TOTAL_SECONDS - (time.monotonic() - started)
        if expired.is_set() or seconds <= 0:
            raise TimeoutError("Combined C/R6 total watchdog expired")
        return seconds

    timer = threading.Timer(max(0.001, TOTAL_SECONDS - (time.monotonic() - started)), expire)
    timer.daemon = True
    timer.start()
    try:
        uids, roster_file_sha = _roster(_official_roster() if roster_path is None else Path(roster_path))
        roster_sha = digest(canonical(uids))
        namespace = {"__name__": "_fixed_rank_fusion"}
        exec(compile(fusion_source, "bound_fixed_rank_fusion", "exec"), namespace)
        _write(root / "fusion.py", fusion_source)
        for arm, source in (("C", c_source), ("R6", r6_source)):
            remaining()
            source_path = root / (arm + ".py")
            _write(source_path, source)
            begin = time.monotonic()
            with (root / (arm + ".private.log")).open("xb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-B", str(source_path)],
                    cwd=working, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                )
                with lock:
                    active["pid"] = process.pid
                    if expired.is_set():
                        _kill_group(process.pid)
                try:
                    result = process.wait(timeout=remaining())
                    remaining()
                    require(result == 0, "Child execution failed; no fallback or next stage")
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        _kill_group(process.pid)
                        raise RuntimeError("Child left background processes; next stage refused")
                except BaseException:
                    _kill_group(process.pid)
                    process.wait()
                    raise
                finally:
                    with lock:
                        active["pid"] = None
            csv_bytes = (working / "submission.csv").read_bytes()
            namespace["_read_csv"](csv_bytes, tuple(uids))
            name = C_ACCEPTED if arm == "C" else R6_ACCEPTED
            acceptance_bytes = (working / name).read_bytes()
            proof = json.loads(acceptance_bytes)
            if arm == "C":
                check_c_receipt(
                    proof, binding, roster_sha256=roster_sha, studies=len(uids), csv_bytes=csv_bytes,
                )
            else:
                require(
                    proof["status"] == "actual_exact8cell_reference_script_MRI_and_final_reader_accepted"
                    and proof["all8code_cells_sha256"] == R6_CELLS
                    and proof["original_notebook_sha256"] == R6_NOTEBOOK
                    and proof["completed_original_cells"] == list(range(8))
                    and proof["R6_fallbacks"] == 0 and proof["R6_bundle_id"] == "fe055406fad6a12f"
                    and proof["full_final_reader_output_exists"] is True
                    and proof["rows"] == len(uids) and proof["columns"] == 13,
                    "Complete unchanged eight-cell R6 final-reader acceptance required",
                )
                for name in R6_OUTPUTS:
                    data = (working / name).read_bytes()
                    require(
                        proof["small_output_proofs"][name] == {
                            "bytes": len(data), "sha256": digest(data),
                        }, "Original R6 output proof differs",
                    )
                if roster_sha == PUBLIC_ROSTER:
                    require(
                        digest(csv_bytes) == R6_PUBLIC_CSV
                        and proof["original_public_example_exact_CSV_match"] is True,
                        "Original exact R6 public CSV required",
                    )
            os.replace(working / "submission.csv", root / (arm + ".csv"))
            stages.append({
                "arm": arm, "elapsed_seconds": time.monotonic() - begin, "exit_code": result,
                "source_sha256": digest(source), "CSV_sha256": digest(csv_bytes),
                "CSV_bytes": len(csv_bytes), "acceptance_sha256": digest(acceptance_bytes),
            })
            remaining()
        fused, fusion_receipt = namespace["fuse_csv"](
            (root / "R6.csv").read_bytes(), (root / "C.csv").read_bytes(), uids,
        )
        _write(root / "fusion.csv", fused)
        remaining()
        receipt = {
            "status": "actual_sequential_C_then_exact_R6_fixed_rank_fusion_completed",
            "mode": binding["mode"],
            "binding_sha256": digest(canonical(binding)),
            "source_sha256": binding["source_sha256"],
            "C_model_sha256": binding["C_model_sha256"],
            "C_runtime": binding["C_runtime"], "C_serving": binding["C_serving"],
            "official_test_csv_sha256": roster_file_sha,
            "test_roster_sha256": roster_sha, "studies": len(uids),
            "stage_order": ["C", "R6"], "stages": stages, "fusion": fusion_receipt,
            "elapsed_seconds": time.monotonic() - started, "total_watchdog_seconds": TOTAL_SECONDS,
            "runtime_guarantee": False, "clinical_or_hidden_score_claim": False,
            "submission_csv_created": True,
            "submission_csv_sha256": digest(fused), "submission_csv_bytes": len(fused),
            "entry_gate_accepted": False,
        }
        require(not (working / "submission.csv").exists(), "Final output already exists")
        _write(working / "submission.csv.tmp", fused)
        remaining()
        os.replace(working / "submission.csv.tmp", working / "submission.csv")
        _write(working / COMBINED_ACCEPTED, canonical(receipt) + b"\n")
        remaining()
        return receipt
    except BaseException as error:
        with lock:
            if active["pid"] is not None:
                _kill_group(active["pid"])
        for name in ("submission.csv", "submission.csv.tmp", COMBINED_ACCEPTED):
            try:
                (working / name).unlink()
            except FileNotFoundError:
                pass
        _write(root / "failed.json", canonical({
            "status": "combined_execution_failed_no_final_submission",
            "error_type": type(error).__name__, "stages": stages,
            "elapsed_seconds": time.monotonic() - started, "retries": 0,
            "watchdog_group_kill_error": active["watchdog_kill_error"],
        }) + b"\n")
        raise
    finally:
        timer.cancel()
