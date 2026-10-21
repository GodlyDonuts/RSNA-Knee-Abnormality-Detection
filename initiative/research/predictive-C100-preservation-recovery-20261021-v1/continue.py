"""Continue the original C100 publisher after a proven pre-upload log-mode failure.

No numerical source, remote publisher, dataset path or GPU job is changed.
The failed original journal stays intact. Every successor mutation has its own
exclusive intent; any uncertain successor outcome stops for reconciliation.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = HERE.parent / "predictive-density100-full-20261020-v1"
SOURCE = "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"
OLD_PID = 67205
OLD_JOB = "996617"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def admit(observed, permissions, old_intent, old_transport):
    failed = observed["publisher_scheduler"]
    require(failed["job_id"] == OLD_JOB and failed["state"] == "FAILED"
            and failed["exit_code"] == "1:0" and failed["queue_empty"] is True,
            "Original publisher must actually be terminal")
    require(observed["scheduler"]["job_id"] == "995351"
            and observed["binding_sha256"] == "f0d73e6cc04cbf7b34f06499f5aef4ba60ea2c593662764b8f64ec5105a289cd"
            and observed["publication_root_exists"] is False
            and observed["funding_exists"] is False,
            "Original failed phase must precede publication and funding")
    require(observed["error"]["message"] == "Original owner-only regular file required"
            and observed["error"]["bad_paths"][0]["path"]
            == "/tmp/rsna-predictive-density100-full-20261020-v1/GPU-995351.out",
            "Exact observed log-mode failure required")
    require(old_intent["phase"] == "publish" and old_intent["epoch"] == 1
            and old_intent["source_git"] == SOURCE and old_transport["returncode"] == 1
            and old_transport["stdout"] == "",
            "Original first-epoch intent and failed transport required")
    require(permissions["status"] == "original_live_C100_logs_owner_permissions_corrected"
            and permissions["training_restarted"] is False
            and len(permissions["files"]) == 2
            and {x["name"] for x in permissions["files"]}
            == {"GPU-995351.out", "GPU-995351.err"}
            and all(x["previous_mode"] == "0644" and x["mode"] == "0600"
                    and x["content_written"] is False for x in permissions["files"]),
            "Only the two original log permissions may change")


def original():
    sys.path.insert(0, str(ORIGINAL))
    spec = importlib.util.spec_from_file_location("C100_original_preservation", ORIGINAL / "coordinator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pending(result, old_journal, new_journal):
    epochs = []
    for epoch in result["ready_epochs"]:
        prior = result["publications"].get(str(epoch))
        if prior:
            require(prior["status"].endswith("private_all_objects_download_size_SHA_verified"),
                    "Existing uncertain remote publication needs read-only reconciliation")
            continue
        require(not (new_journal / f"publish-{epoch:03d}-intent.json").exists(),
                "Never repeat a successor mutation")
        require(epoch == 1 or not (old_journal / f"publish-{epoch:03d}-intent.json").exists(),
                "Unreconciled original later-epoch intent")
        epochs.append(epoch)
    return epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    committed = subprocess.check_output(
        ["git", "show", args.source_git + ":" + Path(__file__).relative_to(REPO).as_posix()], cwd=REPO)
    require(committed == Path(__file__).read_bytes(), "Execute committed continuation")
    require(subprocess.run(["ps", "-p", str(OLD_PID), "-o", "pid="],
                          capture_output=True).returncode != 0, "Original owner is still live")
    c = original()
    old_journal = c.JOURNAL
    observed = json.loads((HERE / "actual-original-failure.json").read_text())
    permissions = json.loads((HERE / "actual-permissions-corrected.json").read_text())
    admit(observed, permissions,
          json.loads((old_journal / "publish-001-intent.json").read_text()),
          json.loads((old_journal / "publish-001-transport.private.json").read_text()))
    require(not (old_journal / "publish-001-result.json").exists(), "Reconcile returned original result")
    successor_source = args.source_git
    args.source_git = SOURCE  # Every original model/publisher binding remains unchanged.
    frozen = c.frozen(args)
    c.JOURNAL = REPO / "artifacts" / HERE.name / "continuation"
    c.JOURNAL.mkdir(mode=0o700, parents=True, exist_ok=True)
    c.a.once(c.JOURNAL / "owner-intent.json", {
        "source_git": successor_source, "original_source_git": SOURCE,
        "PID": os.getpid(), "original_owner_PID": OLD_PID, "GPU_job": "995351",
        "failed_publisher_job": OLD_JOB, "GPU_dispatches": 0, "automatic_mutation_retry": False,
        "original_failure_sha256": hashlib.sha256((HERE / "actual-original-failure.json").read_bytes()).hexdigest(),
        "permissions_sha256": hashlib.sha256((HERE / "actual-permissions-corrected.json").read_bytes()).hexdigest(),
    })
    deadline = c.a.datetime(2026, 10, 22, 12, tzinfo=c.a.timezone.utc).timestamp()
    while time.time() < deadline:
        result = c.observe(args, frozen, deadline)
        if result is None:
            break
        require(result["dispatch"]["GPU_job"] == "995351"
                and result["dispatch"]["source_git"] == SOURCE, "Original GPU/source differs")
        for epoch in pending(result, old_journal, c.JOURNAL):
            value = c.call(args, frozen, "publish", epoch=epoch)
            print(json.dumps({"epoch": epoch, "status": value["status"],
                              "revision": value["revision"]}), flush=True)
        if result["full_scheduler"]["state"] not in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING"}:
            final = c.observe(args, frozen, deadline)
            require(final is not None, "Final observation unavailable")
            c.a.once(c.JOURNAL / "owner-result.json", final)
            return
        time.sleep(min(120, max(0, deadline - time.time())))
    c.a.once(c.JOURNAL / "owner-result.json", {"status": "deadline_original_GPU_untouched"})


if __name__ == "__main__":
    os.umask(0o077)
    main()
