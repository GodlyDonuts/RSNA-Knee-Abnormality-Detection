"""Two exact C100 children in fresh processes, then fixed probability averaging.

No account operations, source modifications, retraining, or model fallback.
Intermediate outputs are archived before the second child reuses its paths.
"""

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

PUBLIC_ROSTER = "e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b"
CHILD_ROOT = "predictive-final-serving-v1"
CHILD_RECEIPT = "predictive-competition-accepted.json"
ROOT = "predictive-C100-two-seed-v1"
ACCEPTED = "predictive-C100-two-seed-accepted.json"
TOTAL_SECONDS = 32400
PUBLIC_SECONDS = 900
PUBLIC_TOLERANCE = 1e-4
ROLES = ("newton", "colab")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _roster(path):
    raw = path.read_bytes()
    reader = csv.reader(io.StringIO(raw.decode(), newline=""), strict=True)
    header = next(reader, None)
    require(
        header and all(header) and len(set(header)) == len(header) and "StudyInstanceUID" in header,
        "Unique official test.csv header required",
    )
    rows = list(reader)
    require(
        rows and all(len(row) == len(header) for row in rows), "Complete official rows required"
    )
    uids = [row[header.index("StudyInstanceUID")] for row in rows]
    require(
        all(uid and uid == uid.strip() for uid in uids) and len(set(uids)) == len(uids),
        "Unique complete official study roster required",
    )
    return uids, digest(raw)


def _official_roster():
    paths = {
        p.resolve()
        for p in (
            Path("/kaggle/input/rsna-knee-abnormality-detection/test.csv"),
            Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/test.csv"),
        )
        if p.is_file()
    }
    require(len(paths) == 1, "One exact official test.csv mount required")
    return next(iter(paths))


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


def check_child(proof, bound, csv_bytes, uids, public_csv, read_csv):
    roster_hash = digest(canonical(uids))
    is_public = roster_hash == PUBLIC_ROSTER
    require(
        proof["status"] == "actual_predictive_exact_test_roster_submission_created"
        and proof["mode"] == "competition"
        and proof["test_roster_sha256"] == roster_hash
        and proof["actual_test_studies"] == len(uids)
        and proof["submission_csv_absent"] is False
        and proof["CSV_sha256"] == digest(csv_bytes)
        and proof["CSV_bytes"] == len(csv_bytes),
        "Exact child roster/CSV proof required",
    )
    require(
        len(proof["GPU_names"]) == 2 and all("T4" in n for n in proof["GPU_names"]),
        "Actual two-T4 child required",
    )
    require(len(proof["models"]) == 1, "One complete child model required")
    model = proof["models"][0]
    require(
        model["arm"] == "C"
        and model["trained_export_sha256"] == bound["model_sha256"]
        and model["candidate_sha256"] == bound["candidate_sha256"]
        and model["precision_accepted"] is True
        and (not is_public or model["memory_accepted"] is True),
        "Exact trained child model/precision required",
    )
    require(
        all(proof[k] == v for k, v in bound["runtime_and_serving"].items()),
        "Child runtime/serving differs",
    )
    values = read_csv(csv_bytes, tuple(uids))
    difference = None
    if is_public:
        require(
            digest(public_csv) == bound["public_CSV_sha256"], "Original public reference differs"
        )
        original = read_csv(public_csv, tuple(uids))
        difference = max(
            abs(x - y)
            for a, b in zip(values, original, strict=True)
            for x, y in zip(a, b, strict=True)
        )
        require(difference <= PUBLIC_TOLERANCE, "Paired public child prediction difference")
    # The reference is never an input to hidden predictions or a fallback.
    return difference


def run(sources, fusion_source, references, binding, *, working=None, roster_path=None):
    started = time.monotonic()
    require(
        binding["mode"] == "candidate"
        and binding["total_watchdog_seconds"] == TOTAL_SECONDS
        and binding["public_watchdog_seconds"] == PUBLIC_SECONDS,
        "One bounded candidate execution required",
    )
    require(
        set(sources) == set(references) == set(ROLES), "Two complete independent children required"
    )
    for role in ROLES:
        require(
            digest(sources[role]) == binding["source_sha256"][role], "Child source hash differs"
        )
        require(
            digest(references[role]) == binding["children"][role]["public_CSV_sha256"],
            "Reference source hash differs",
        )
    require(
        digest(fusion_source) == binding["source_sha256"]["fusion"], "Fusion source hash differs"
    )
    require(
        binding["children"]["newton"]["model_sha256"]
        != binding["children"]["colab"]["model_sha256"],
        "Distinct complete learned models required",
    )
    working = Path("/kaggle/working") if working is None else Path(working)
    root = working / ROOT
    stale = (ROOT, CHILD_ROOT, CHILD_RECEIPT, "submission.csv", "submission.csv.tmp", ACCEPTED)
    require(
        working.is_dir()
        and not any((working / n).exists() or (working / n).is_symlink() for n in stale),
        "Fresh roots, receipts and outputs required",
    )
    uids, roster_file_hash = _roster(
        _official_roster() if roster_path is None else Path(roster_path)
    )
    roster_hash = digest(canonical(uids))
    seconds = PUBLIC_SECONDS if roster_hash == PUBLIC_ROSTER else TOTAL_SECONDS
    root.mkdir(mode=0o700)
    namespace = {"__name__": "_bound_probability_fusion"}
    exec(compile(fusion_source, "bound_probability_fusion", "exec"), namespace)
    expired, lock = threading.Event(), threading.Lock()
    active = {"pid": None, "signaled": False, "kill_error": None}
    stages = []

    def stop_active():
        # The timer and wait-timeout path can fire together. Signal a group only
        # once, before reaping its leader, rather than racing two killpg calls.
        with lock:
            if active["pid"] is not None and not active["signaled"]:
                try:
                    _kill_group(active["pid"])
                except OSError as error:
                    active["kill_error"] = type(error).__name__
                    raise
                active["signaled"] = True

    def expire():
        expired.set()
        try:
            stop_active()
        except OSError:
            # Main cleanup retains and reports this failure; never a successful
            # result with an unobserved timer-thread exception.
            pass

    def remaining():
        value = seconds - (time.monotonic() - started)
        if value <= 0 or expired.is_set():
            raise TimeoutError("Shared two-seed watchdog expired")
        return value

    timer = threading.Timer(max(0.001, seconds - (time.monotonic() - started)), expire)
    timer.daemon = True
    timer.start()
    try:
        for role in ROLES:
            remaining()
            archive = root / role
            archive.mkdir(mode=0o700)
            path = archive / "source.private.py"
            _write(path, sources[role])
            begin = time.monotonic()
            with (archive / "execution.private.log").open("xb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-B", str(path)],
                    cwd=working,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                )
                with lock:
                    active["pid"] = process.pid
                    active["signaled"] = False
                try:
                    if expired.is_set():
                        stop_active()
                    code = process.wait(timeout=remaining())
                    remaining()
                    require(code == 0, "Child failed; no next child or fallback")
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError("Child left background processes")
                except BaseException:
                    stop_active()
                    process.wait()
                    raise
                finally:
                    with lock:
                        active["pid"] = None
            csv_bytes = (working / "submission.csv").read_bytes()
            proof_bytes = (working / CHILD_RECEIPT).read_bytes()
            difference = check_child(
                json.loads(proof_bytes),
                binding["children"][role],
                csv_bytes,
                uids,
                references[role],
                namespace["_read_csv"],
            )
            require(
                (working / CHILD_ROOT).is_dir() and not (working / CHILD_ROOT).is_symlink(),
                "Original child working directory required",
            )
            # Preserve each original's bytes; permit exact unchanged second child paths.
            for name in (CHILD_ROOT, CHILD_RECEIPT, "submission.csv"):
                require(not (working / name).is_symlink(), "Child output symlink refused")
                os.replace(working / name, archive / name)
            stages.append(
                dict(
                    role=role,
                    source_sha256=digest(sources[role]),
                    model_sha256=binding["children"][role]["model_sha256"],
                    CSV_sha256=digest(csv_bytes),
                    acceptance_sha256=digest(proof_bytes),
                    elapsed_seconds=time.monotonic() - begin,
                    public_max_probability_difference=difference,
                )
            )
        fused, fusion = namespace["fuse_csv"](
            (root / "newton/submission.csv").read_bytes(),
            (root / "colab/submission.csv").read_bytes(),
            uids,
        )
        remaining()
        _write(working / "submission.csv.tmp", fused)
        os.replace(working / "submission.csv.tmp", working / "submission.csv")
        receipt = dict(
            status="actual_two_seed_complete_probability_ensemble",
            binding_sha256=digest(canonical(binding)),
            stage_order=list(ROLES),
            stages=stages,
            test_roster_sha256=roster_hash,
            official_test_csv_sha256=roster_file_hash,
            studies=len(uids),
            fusion=fusion,
            elapsed_seconds=time.monotonic() - started,
            effective_watchdog_seconds=seconds,
            total_watchdog_seconds=TOTAL_SECONDS,
            CSV_bytes=len(fused),
            CSV_sha256=digest(fused),
            submission_csv_created=True,
            runtime_guarantee=False,
            clinical_score_claim=False,
            entry_gate_accepted=False,
        )
        _write(working / ACCEPTED, canonical(receipt) + b"\n")
        remaining()
        return receipt
    except BaseException as error:
        for name in ("submission.csv", "submission.csv.tmp", ACCEPTED):
            try:
                (working / name).unlink()
            except FileNotFoundError:
                pass
        _write(
            root / "failed.json",
            canonical(
                dict(
                    status="no_final_submission",
                    error_type=type(error).__name__,
                    stages=stages,
                    retries=0,
                    group_kill_error=active["kill_error"],
                )
            )
            + b"\n",
        )
        raise
    finally:
        timer.cancel()
