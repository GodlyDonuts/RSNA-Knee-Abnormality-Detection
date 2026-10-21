"""One conditional source-only SaveKernel; never starts a batch or submits.

The old R6 dataset slug now resolves differently. Native UI version pinning and
fresh runtime admission must follow this source-only step before any execution.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import requests

ROOT = Path("/tmp/rsna-C100-R6-20261021-v1")
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")
SOURCE_SHA = "fbb9808cd8969cdaad5435aa7c0b65854df95d95eab6e2a731074f385f723b41"
PREPARATION = "0e7ef257b04aef117f1b4173d1a09a824d243c2a"
SLUG = "godlydonuts/rsna-c100-r6-ensemble-" + SOURCE_SHA[:12]
KEY = "licensed-parent-full-epoch-serving:C100-complete-R6:" + SOURCE_SHA
MODEL_DATASET = "godlydonuts/rsna-predictive-c100-final-3d49d387375a"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
DATASETS = {
    "goodpjw2008/rsna-knee-2-5d-convnext-reader",
    "heliosli/rsna-knee-r6-frozen-fe055406",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
}
KERNEL = "goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958"


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def make_request(source, prepared, original):
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
    assert prepared["source_sha256"] == SOURCE_SHA and prepared["source_git"] == PREPARATION
    assert prepared["CPU_job"] == "1002213" and not prepared["actual_combined_GPU_run"]
    assert original["id"] == 139271768 and original["currentVersionNumber"] == 2
    assert set(original["datasetDataSources"]) == DATASETS
    assert original["kernelDataSources"] == [KERNEL]
    assert original["competitionDataSources"] == ["rsna-knee-abnormality-detection"]
    assert original["dockerImage"] == DOCKER
    return dict(
        slug=SLUG,
        newTitle=SLUG.split("/")[1].replace("-", " "),
        text=source,
        language="python",
        kernelType="script",
        isPrivate=True,
        enableGpu=True,
        enableInternet=False,
        enableTpu=False,
        machineShape="NvidiaTeslaT4",
        sessionTimeoutSeconds=32400,
        kernelExecutionType="QUICK_SAVE",
        dockerImage=DOCKER,
        dockerImagePinningType="original",
        datasetDataSources=sorted(DATASETS | {MODEL_DATASET}),
        kernelDataSources=[KERNEL],
        competitionDataSources=["rsna-knee-abnormality-detection"],
    )


def save_source(w, remote, db, request, compiled, pin, *, root=ROOT):
    """Caller owns the canonical claim. Intent precedes the one API mutation."""
    found = db.execute("SELECT value FROM meta WHERE key=?", (KEY,)).fetchone()
    if found:
        row = json.loads(found[0])
        return dict(
            status="existing_original_source_operation_do_not_repeat",
            original_state=row["state"],
            actual=row.get("actual_source_only"),
            automatic_retry=False,
            new_Kaggle_mutations=0,
        )
    assert not (root / "source-save-intent.json").exists()
    submissions = remote.submissions()
    selected = [x for x in submissions if x.get("ref") == 57491555]
    assert len(selected) == 1
    if not compiled.score_trigger(selected[0]):
        return dict(
            status="waiting_actual_C100_qualifying_score_no_mutation",
            submission_id=57491555,
            actual_status=selected[0].get("status"),
            public_score=selected[0].get("publicScore"),
            error_present=bool(selected[0].get("errorDescription")),
            new_Kaggle_mutations=0,
        )
    # The phase consumes neither batch time nor a competition slot. Requiring
    # spare time here avoids saving an unusable candidate late in the deadline.
    quota = remote.call("kernels.KernelsApiService", "GetAcceleratorQuotaStatistics", {})[
        "gpuQuota"
    ]

    def seconds(x):
        return float(str(x).removesuffix("s"))

    free = (
        seconds(quota["totalTimeAllowed"])
        - seconds(quota.get("timeUsed", "0s"))
        - seconds(quota.get("timeReserved", "0s"))
    )
    assert free >= 1020, "A future bounded public execution must remain fundable"
    account = remote.account()
    assert account["entered"] and account["account"] == "godlydonuts"
    # Check exactly the already-preserved C100 dataset, without repeating its bytes.
    data = remote.dataset(MODEL_DATASET)
    assert data["id"] == 12762386 and data["currentVersionNumber"] == 1
    absence = remote.require_absent_kernel(SLUG)
    row = dict(
        state="C100_R6_source_only_intent",
        slug=SLUG,
        source_sha256=SOURCE_SHA,
        source_git=pin,
        frozen_preparation_git=PREPARATION,
        request_sha256=w.object_hash(request),
        absence=absence,
        actual_C100_submission={
            k: selected[0].get(k) for k in ("ref", "status", "publicScore", "errorDescription")
        },
        free_GPU_VM_seconds=free,
        actual_account=account,
        required_before_batch="Native pin all original R6 input versions, fresh Colab reservation and GPU budget/capacity review, then one exact public run",
        GPU_batch_dispatches=0,
        competition_entries=0,
        automatic_retry=False,
    )
    w.new_json(root / "source-save-intent.json", row)
    db.execute("INSERT INTO meta VALUES (?,?)", (KEY, json.dumps(row, sort_keys=True)))
    db.commit()
    response = remote.call("kernels.KernelsApiService", "SaveKernel", request)
    w.new_json(root / "source-save-response.json", response)
    assert response.get("ref") == "/code/" + SLUG and response.get("versionNumber") == 1
    assert type(response.get("kernelId")) is int and response["kernelId"] > 0
    assert not any(
        response.get(k)
        for k in (
            "error",
            "invalidDatasetSources",
            "invalidKernelSources",
            "invalidCompetitionSources",
        )
    )
    actual = remote.kernel("GetKernel", SLUG, 1)
    metadata = actual["metadata"]
    assert metadata["id"] == response["kernelId"] and actual["blob"]["source"] == request["text"]
    for k in ("kernelType", "language", "isPrivate", "enableGpu", "machineShape", "dockerImage"):
        assert metadata.get(k) == request[k], k
    assert not metadata.get("enableInternet", False) and not metadata.get("enableTpu", False)
    for k in ("datasetDataSources", "kernelDataSources", "competitionDataSources"):
        assert sorted(metadata.get(k, [])) == sorted(request[k]), k
    result = dict(
        status="C100_R6_private_source_only_saved_requires_native_input_pins",
        kernel_id=metadata["id"],
        slug=SLUG,
        version=1,
        source_sha256=SOURCE_SHA,
        original_R6_Dataset_v1_required=True,
        input_versions_pinned=False,
        batch_GPU_execution=False,
        competition_entry=False,
        source_git=pin,
        original_C100_public_score=selected[0]["publicScore"],
        new_Kaggle_mutations=1,
    )
    w.new_json(root / "source-save-verified.json", result)
    row.update(state="C100_R6_source_only_saved", actual_source_only=result)
    db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(row, sort_keys=True), KEY))
    db.commit()
    return result


def main(v):
    os.umask(0o077)
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    q = next(x.split() for x in quota.splitlines() if x.strip().startswith("/home/sa305415"))
    assert (int(q[2]) - int(q[1])) * 1024 >= 20 * 2**30 + 125 * 2**20
    assert int(q[6]) - int(q[5]) >= 5370
    assert shutil.disk_usage("/tmp").free >= (518 + 20) * 2**30 + 144 * 2**20
    assert sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file()) + 4 * 2**20 < 16 * 2**20
    assert (ROOT / "compile.py").read_text() == v["compiler_source"]
    assert (ROOT / "workflow.py").read_text() == v["workflow_source"]
    c = load("original_C100_R6_compiler", ROOT / "compile.py")
    w = load("original_C100_R6_workflow", ROOT / "workflow.py")
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + v["token"]
    remote.username = "godlydonuts"
    who = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    assert who.get("active") and who.get("username") == "godlydonuts"
    request = make_request(
        (ROOT / "source.private.py").read_text(),
        json.loads((ROOT / "prepared.json").read_text()),
        json.loads((ROOT / "R6-original-metadata.private.json").read_text()),
    )
    claim = LEDGER.parent / "coordinator-claim"
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        return dict(status="waiting_original_canonical_claim_no_mutation", new_Kaggle_mutations=0)
    owner = dict(
        kind="C100_R6_source_only_save", CPU_job=os.environ["SLURM_JOB_ID"], source_git=v["pin"]
    )
    w.new_json(claim / "owner.json", owner)
    try:
        with w.ledger(LEDGER, "godlydonuts") as db:
            w.verify_account(db, remote)
            result = save_source(w, remote, db, request, c, v["pin"])
    except BaseException:
        # No automatic release following any uncertain mutation/result.
        if (ROOT / "source-save-intent.json").exists():
            raise
        assert json.loads((claim / "owner.json").read_text()) == owner
        (claim / "owner.json").unlink()
        claim.rmdir()
        raise
    assert json.loads((claim / "owner.json").read_text()) == owner
    (claim / "owner.json").unlink()
    claim.rmdir()
    return {**result, "CPU_job": os.environ["SLURM_JOB_ID"]}
