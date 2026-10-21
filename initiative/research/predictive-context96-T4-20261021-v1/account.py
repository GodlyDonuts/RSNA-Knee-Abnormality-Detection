"""Single canonical fixed96 diagnostic; existing model Dataset, no competition entry."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import requests

ROOT = Path("/tmp/rsna-predictive-context96-T4-20261021-v1")
ORIGINAL = Path("/tmp/rsna-predictive-final-deployment-20261020-v1-C100-selected-20261021-v1")
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")
MODEL_DATASET = "godlydonuts/rsna-predictive-c100-final-3d49d387375a"
PARENT_DATASET = "nartaa/rsna-knee-publication-swa-weights-20261007"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def accept(r, c):
    assert r["status"] == "actual_fixed96_public_T4_diagnostic_complete"
    assert r["inference_context_cap"] == 96 and r["training_context_cap"] == 24
    assert r["context_coverage_version"] == c.VERSION
    assert r["competition_admitted"] is False and r["submission_csv_absent"] is True
    assert r["GPU_names"] == ["Tesla T4", "Tesla T4"] and r["actual_test_studies"] == 3
    assert 0 < r["elapsed_seconds"] < 600 and len(r["models"]) == 1
    m = r["models"][0]
    assert m["trained_export_sha256"] == c.MODEL
    assert m["reference24_prediction_sha256"] == c.REFERENCE_PREDICTIONS
    assert m["memory_accepted"] is True and m["precision_accepted"] is True
    assert {x["study_index"] for x in m["measurements"]} == {0, 1, 2}
    for x in m["measurements"]:
        assert x["complete_descriptor_support_fields_exact"] is True
        assert x["original_context_bitwise_equal"] is True
        assert x["inference_context_cap"] == 96 and 48 < x["windows"] <= 192
        assert x["decoder_route"] == "original_exact" and x["prediction_precision"] == "fp16"
        assert x["maximum_probability_gap"] <= 0.005
    return r


def main(v):
    os.umask(0o077)
    assert os.uname().nodename == "evc27"
    assert not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.environ.get("SLURM_JOB_GPUS")
    assert v["phase"] in ("prepare", "push", "status")
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    q = next(x.split() for x in quota.splitlines() if x.strip().startswith("/home/sa305415"))
    funding = dict(
        node_free_bytes=shutil.disk_usage("/tmp").free,
        shared_free_bytes=(int(q[2]) - int(q[1])) * 1024,
        shared_free_files=int(q[6]) - int(q[5]),
        other_node_bytes=518 * 2**30 + 64 * 2**20,
        new_node_bytes=32 * 2**20,
        new_shared_bytes=2**20,
        new_shared_files=0,
        other_shared_bytes=123 * 2**20,
        other_shared_files=370,
    )
    assert funding["node_free_bytes"] >= (518 + 20) * 2**30 + 96 * 2**20
    assert funding["shared_free_bytes"] >= 20 * 2**30 + 124 * 2**20
    assert funding["shared_free_files"] >= 5370
    assert not ROOT.is_symlink()
    if v["phase"] == "prepare":
        assert not ROOT.exists(), "Original preparation requires read-only reconciliation"
        ROOT.mkdir(mode=0o700)
        for name, body in v["files"].items():
            assert name in ("compile.py", "workflow.py", "components.py", "gpu_capacity.py")
            with (ROOT / name).open("x") as f:
                f.write(body)
    else:
        assert ROOT.is_dir()
        assert all((ROOT / n).read_text() == body for n, body in v["files"].items())
    c = load("fixed96_compiler", ROOT / "compile.py")
    w = load("fixed96_workflow", ROOT / "workflow.py")
    capacity = load("fixed96_capacity", ROOT / "gpu_capacity.py")
    if v["phase"] == "prepare":
        source, compiled = c.build(
            (ORIGINAL / "compiled-qualification/source.private.py").read_text()
        )
        original_verified = json.loads((ORIGINAL / "dataset-verified.json").read_text())
        assert original_verified["id"] == 12762386 and original_verified["private"] is True
        assert original_verified["status"] == "all_payload_bytes_verified"
        compiled.update(source_git=v["pin"], funding=funding, CPU_job=os.environ["SLURM_JOB_ID"])
        with (ROOT / "source.private.py").open("x") as f:
            f.write(source)
        w.new_json(ROOT / "compiled.json", compiled)
        return {k: value for k, value in compiled.items() if k != "binding"}
    compiled = json.loads((ROOT / "compiled.json").read_text())
    source = (ROOT / "source.private.py").read_text()
    assert compiled["source_git"] == v["pin"] and sha(source) == compiled["source_sha256"]
    assert sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file()) < 32 * 2**20
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + v["token"]
    remote.username = "godlydonuts"
    who = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    assert who.get("active") and who.get("username") == remote.username
    slug = "godlydonuts/rsna-predictive-context96-" + compiled["source_sha256"][:12]
    request = dict(
        slug=slug,
        newTitle=slug.split("/")[1].replace("-", " "),
        text=source,
        language="python",
        kernelType="script",
        isPrivate=True,
        enableGpu=True,
        machineShape="NvidiaTeslaT4",
        enableInternet=False,
        enableTpu=False,
        sessionTimeoutSeconds=600,
        dockerImage=compiled["binding"]["pinned_Docker"],
        dockerImagePinningType="original",
        datasetDataSources=[PARENT_DATASET, MODEL_DATASET],
        kernelDataSources=[],
        competitionDataSources=["rsna-knee-abnormality-detection"],
    )
    claim = LEDGER.parent / "coordinator-claim"
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        return {"status": "waiting_existing_claim_no_mutation"}
    w.new_json(
        claim / "owner.json",
        {
            "kind": "fixed96_public_diagnostic",
            "phase": v["phase"],
            "CPU_job": os.environ["SLURM_JOB_ID"],
        },
    )
    retain = False
    try:
        with w.ledger(LEDGER, remote.username) as db:
            w.verify_account(db, remote)
            key = (
                "licensed-parent-full-epoch-serving:fixed96-diagnostic-v1:"
                + compiled["source_sha256"]
            )
            found = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            row = json.loads(found[0]) if found else {}

            def save():
                db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(row)))
                db.commit()

            if v["phase"] == "push":
                assert not row and not (ROOT / "push-intent.json").exists()
                original_slug = "godlydonuts/rsna-predictive-c100-qualification-e6-fe364a3c8a9b"
                original_kernel = remote.kernel("GetKernel", original_slug, 1)
                assert original_kernel["metadata"]["id"] == 140062137
                assert sha(original_kernel["blob"]["source"]) == c.PARENT
                assert (
                    remote.kernel("GetKernelSessionStatus", original_slug, 1).get("status")
                    == "COMPLETE"
                )
                for ds, expected in ((MODEL_DATASET, 12762386), (PARENT_DATASET, 12431568)):
                    metadata = remote.dataset(ds)
                    assert metadata["id"] == expected and metadata["currentVersionNumber"] == 1
                available = capacity.admission(db, remote, caller="serving")
                assert available["available"], "Keep original GPU work undisturbed"
                q = remote.call("kernels.KernelsApiService", "GetAcceleratorQuotaStatistics", {})[
                    "gpuQuota"
                ]

                def seconds(x):
                    return float(x.removesuffix("s"))

                free = (
                    seconds(q["totalTimeAllowed"])
                    - seconds(q.get("timeUsed", "0s"))
                    - seconds(q.get("timeReserved", "0s"))
                )
                assert free >= 1920 + 600 + 120, "Protect original Colab final deployment"
                absence = remote.require_absent_kernel(slug)
                row = dict(
                    state="fixed96_push_intent",
                    source_git=v["pin"],
                    source_sha256=sha(source),
                    slug=slug,
                    version=1,
                    request_sha256=w.object_hash(request),
                    free_VM_seconds=free,
                    protected_Colab_VM_seconds=1920,
                    own_timeout_seconds=600,
                    own_startup_reserve_seconds=120,
                    competition_entry=False,
                    new_Dataset=False,
                    funding=funding,
                    absence=absence,
                    CPU_job=os.environ["SLURM_JOB_ID"],
                    automatic_retry=False,
                )
                w.new_json(ROOT / "push-intent.json", row)
                save()
                retain = True
                response = remote.call("kernels.KernelsApiService", "SaveKernel", request)
                w.new_json(ROOT / "push-response.json", response)
                assert response.get("ref") == "/code/" + slug
                assert response.get("versionNumber") == 1 and type(response.get("kernelId")) is int
                assert not any(
                    response.get(k)
                    for k in (
                        "error",
                        "invalidDatasetSources",
                        "invalidKernelSources",
                        "invalidCompetitionSources",
                    )
                )
                row.update(state="fixed96_pushed", kernel_id=response["kernelId"])
                save()
                retain = False
            assert row.get("state") in ("fixed96_pushed", "fixed96_complete")
            actual = remote.kernel("GetKernel", slug, 1)
            metadata = actual["metadata"]
            assert metadata["id"] == row["kernel_id"] and actual["blob"]["source"] == source
            for name in ("isPrivate", "enableGpu", "machineShape", "dockerImage"):
                assert metadata.get(name) == request[name]
            assert not metadata.get("enableInternet", False) and not metadata.get(
                "enableTpu", False
            )
            for name in ("datasetDataSources", "kernelDataSources", "competitionDataSources"):
                assert sorted(metadata.get(name, [])) == sorted(request[name])
            status = remote.kernel("GetKernelSessionStatus", slug, 1).get("status")
            result = dict(
                status=status,
                slug=slug,
                version=1,
                kernel_id=row["kernel_id"],
                source_sha256=sha(source),
                competition_entry=False,
                new_Dataset=False,
                CPU_job=os.environ["SLURM_JOB_ID"],
            )
            if status == "COMPLETE":
                result["receipt"] = accept(
                    remote.output_json(slug, 1, "predictive-context96-diagnostic.json"), c
                )
                if row["state"] != "fixed96_complete":
                    w.new_json(ROOT / "actual-complete.json", result)
                    row.update(
                        state="fixed96_complete", receipt_sha256=w.object_hash(result["receipt"])
                    )
                    save()
            return result
    finally:
        if not retain:
            (claim / "owner.json").unlink()
            claim.rmdir()
