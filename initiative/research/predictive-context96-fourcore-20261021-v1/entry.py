"""Single-use entry for the already accepted dense96 notebook; never a new push.

An inference-variant key separates this record from the same checkpoint's original
24-context entry. Both retain the actual checkpoint SHA; no prior row is replaced.
"""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path("/tmp/rsna-context96-fourcore-20261021-v1")
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")
SOURCE = "5cc94175a2f2f3c3d5d0814201dda7fa154dbdc01e861c95bd999254a537571e"
MODEL = "e606d974f4aa8bcc2a4e07375353e41f1e4d31d7151f7d223380624c8116d8d6"
SLUG = "godlydonuts/rsna-context96-fourcore-" + SOURCE[:12]
KERNEL = 140075914
PROOF = "5f3ca5759757813ec8a9ce7b0862ad2f021bf029531679b8900ef367dd351441"
READY_KEY = "licensed-parent-full-epoch-serving:context96-fourcore-v1:" + SOURCE
ENTRY_KEY = hashlib.sha256(
    ("dense96-inference-variant-v1:" + SOURCE + ":" + MODEL).encode()
).hexdigest()
DESCRIPTION = "predictive-fixed96-fourcore-v1:" + ENTRY_KEY
START = datetime(2026, 10, 22, 0, 5, tzinfo=timezone.utc)
LAST = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def gate(w, db, remote, now):
    submissions = remote.submissions()
    original = [x for x in submissions if x.get("ref") == 57491555]
    assert len(original) == 1, "Exact original C100 outcome required"
    prior = original[0]
    if w.submission_outcome(prior) != "scored":
        return dict(
            status="waiting_original_C100_scored_outcome_no_mutation",
            original_submission=57491555,
            original_outcome=w.submission_outcome(prior),
            original_score=prior.get("publicScore"),
        )
    score = float(prior["publicScore"])
    assert math.isfinite(score) and 0 <= score <= 1
    if not START <= now <= LAST:
        return dict(
            status="outside_declared_entry_window_no_mutation",
            now=now.isoformat(),
            start=START.isoformat(),
            last=LAST.isoformat(),
            original_score=score,
        )
    account = remote.account()
    if w.allowance(db, account, now=now) < 2:
        return dict(status="preserving_Colab_or_final_daily_slot_no_mutation", account=account)
    assert not any(s.get("description") == DESCRIPTION for s in submissions), (
        "Existing exact entry needs reconciliation"
    )
    for (raw,) in db.execute("SELECT value FROM meta"):
        if raw.lstrip().startswith("{"):
            assert (
                not json.loads(raw)
                .get("state", "")
                .endswith(("_submit_intent", "_submit_uncertain"))
            ), "Uncertain other entry"
    return dict(status="ready_for_one_exact_code_entry", account=account, original_score=score)


def inspect_original(w, remote, db, root):
    compiled = json.loads((root / "compiled.json").read_text())
    proof = json.loads((root / "actual-complete.json").read_text())
    source = (root / "source.private.py").read_text()
    assert w.object_hash(proof) == PROOF
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE == compiled["source_sha256"]
    assert compiled["source_git"] == "f8d2380365e98666aa0d3afb3c3b754b8ebdeb10"
    assert proof["kernel_id"] == KERNEL and proof["version"] == 1 and proof["status"] == "COMPLETE"
    receipt = proof["receipt"]
    assert receipt["models"][0]["trained_export_sha256"] == MODEL
    assert receipt["runtime_planning"]["mean_below_9_hours"] is True
    assert receipt["runtime_planning"]["hidden_runtime_guaranteed"] is False
    assert len(receipt["private_paired_objects_verified"]) == 4
    row = json.loads(db.execute("SELECT value FROM meta WHERE key=?", (READY_KEY,)).fetchone()[0])
    assert row["state"] == "fixed96_complete" and row["kernel_id"] == KERNEL
    assert row["receipt_sha256"] == w.object_hash(receipt)
    actual = remote.kernel("GetKernel", SLUG, 1)
    meta = actual["metadata"]
    assert meta["id"] == KERNEL and actual["blob"]["source"] == source
    assert meta["isPrivate"] and meta["enableGpu"] and meta["machineShape"] == "NvidiaTeslaT4"
    assert not meta.get("enableInternet", False) and not meta.get("enableTpu", False)
    assert meta["dockerImage"] == compiled["binding"]["pinned_Docker"]
    datasets = {
        "godlydonuts/rsna-predictive-c100-final-3d49d387375a": 12762386,
        "nartaa/rsna-knee-publication-swa-weights-20261007": 12431568,
    }
    assert set(meta["datasetDataSources"]) == set(datasets) and not meta.get("kernelDataSources")
    assert meta["competitionDataSources"] == ["rsna-knee-abnormality-detection"]
    for slug, identity in datasets.items():
        data = remote.dataset(slug)
        assert data["id"] == identity and data["currentVersionNumber"] == 1
    assert remote.kernel("GetKernelSessionStatus", SLUG, 1).get("status") == "COMPLETE"
    fresh = remote.output_json(SLUG, 1, "predictive-competition-accepted.json")
    assert all(key in receipt and receipt[key] == value for key, value in fresh.items())
    assert fresh["status"] == "actual_fourcore_fixed96_public_comparison_complete"
    return proof


def execute(w, remote, db, phase, pin, *, root=ROOT, now=None):
    assert phase in ("check", "submit", "outcome")
    now = now or datetime.now(timezone.utc)
    found = db.execute("SELECT record FROM entries WHERE sha=?", (ENTRY_KEY,)).fetchone()
    existing = json.loads(found[0]) if found else None
    if existing:
        assert existing["entry_identity"] == ENTRY_KEY and existing["checkpoint_sha256"] == MODEL
        if phase != "outcome":
            return dict(
                status="existing_dense_entry_do_not_repeat",
                state=existing["state"],
                submission_id=existing.get("submission_id"),
                automatic_retry=False,
            )
        matches = [x for x in remote.submissions() if x.get("description") == DESCRIPTION]
        assert len(matches) == 1 and type(matches[0].get("ref")) is int, (
            "Uncertain response: preserve original intent"
        )
        item = matches[0]
        assert existing.get("submission_id") in (None, item["ref"])
        existing.update(
            state="submitted",
            submission_id=item["ref"],
            actual_outcome={
                k: item.get(k)
                for k in ("ref", "status", "publicScore", "errorDescription", "totalBytes")
            },
        )
        existing["actual_outcome"]["outcome"] = w.submission_outcome(item)
        db.execute("UPDATE entries SET record=? WHERE sha=?", (json.dumps(existing), ENTRY_KEY))
        db.commit()
        return existing
    assert phase != "outcome", "No dense competition entry has been attempted"
    assert not (root / "submit-intent.json").exists(), (
        "Original uncertain intent must be reconciled"
    )
    proof = inspect_original(w, remote, db, root)
    admission = gate(w, db, remote, now)
    if phase == "check" or admission["status"] != "ready_for_one_exact_code_entry":
        return {**admission, "exact_existing_notebook_verified": True, "new_Kaggle_mutations": 0}
    record = dict(
        state="submit_intent",
        entry_identity=ENTRY_KEY,
        record_kind="distinct_inference_of_existing_checkpoint",
        checkpoint_sha256=MODEL,
        source_sha256=SOURCE,
        source_git=pin,
        kernel_version=1,
        operation={"kernel_slug": SLUG},
        description=DESCRIPTION,
        attempted_at=now.isoformat(),
        quota_before=admission["account"],
        original_C100_score=admission["original_score"],
        accepted_proof_sha256=PROOF,
        runtime_planning=proof["receipt"]["runtime_planning"],
        inference_context_cap=96,
        training_context_cap=24,
        automatic_retry=False,
    )
    w.new_json(root / "submit-intent.json", record)
    # The table key is explicitly the inference-operation identity, not a false
    # checkpoint hash. The original MODEL-keyed 24-context record is untouched.
    db.execute("INSERT INTO entries(sha,record) VALUES (?,?)", (ENTRY_KEY, json.dumps(record)))
    db.commit()
    try:
        result = remote.competition(
            "CreateCodeSubmission",
            dict(
                fileName="submission.csv",
                kernelOwner="godlydonuts",
                kernelSlug=SLUG.split("/")[1],
                kernelVersion=1,
                submissionDescription=DESCRIPTION,
            ),
        )
        w.new_json(root / "submit-response.json", result)
        assert type(result.get("ref")) is int and result["ref"] > 0, "Uncertain entry response"
        record.update(state="submitted", submission_id=result["ref"], submitted_at=w.utcnow())
    except BaseException:
        record["state"] = "submit_uncertain"
        db.execute("UPDATE entries SET record=? WHERE sha=?", (json.dumps(record), ENTRY_KEY))
        db.commit()
        raise
    db.execute("UPDATE entries SET record=? WHERE sha=?", (json.dumps(record), ENTRY_KEY))
    db.commit()
    return record


def main(v):
    os.umask(0o077)
    assert os.uname().nodename == "evc27"
    assert not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.environ.get("SLURM_JOB_GPUS")
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    q = next(x.split() for x in quota.splitlines() if x.strip().startswith("/home/sa305415"))
    assert shutil.disk_usage("/tmp").free >= (518 + 20) * 2**30 + 176 * 2**20
    assert (int(q[2]) - int(q[1])) * 1024 >= 20 * 2**30 + 126 * 2**20
    assert int(q[6]) - int(q[5]) >= 5370
    assert ROOT.is_dir() and not ROOT.is_symlink()
    assert sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file()) < 32 * 2**20
    s = importlib.util.spec_from_file_location(
        "original_dense_entry_workflow", ROOT / "workflow.py"
    )
    w = importlib.util.module_from_spec(s)
    s.loader.exec_module(w)
    remote = w.Remote.__new__(w.Remote)
    remote.username = "godlydonuts"
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + v["token"]
    who = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    assert who.get("active") and who.get("username") == remote.username
    claim = LEDGER.parent / "coordinator-claim"
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        return {"status": "waiting_existing_canonical_claim_no_mutation"}
    owner = dict(
        kind="accepted_dense96_entry",
        CPU_job=os.environ["SLURM_JOB_ID"],
        phase=v["phase"],
        source_git=v["pin"],
    )
    w.new_json(claim / "owner.json", owner)
    retain = False
    try:
        with w.ledger(LEDGER, remote.username) as db:
            w.verify_account(db, remote)
            try:
                result = execute(w, remote, db, v["phase"], v["pin"])
                return {**result, "CPU_job": os.environ["SLURM_JOB_ID"]}
            finally:
                row = db.execute("SELECT record FROM entries WHERE sha=?", (ENTRY_KEY,)).fetchone()
                retain = bool(
                    row and json.loads(row[0])["state"] in {"submit_intent", "submit_uncertain"}
                )
    finally:
        if not retain:
            assert json.loads((claim / "owner.json").read_text()) == owner
            (claim / "owner.json").unlink()
            claim.rmdir()
