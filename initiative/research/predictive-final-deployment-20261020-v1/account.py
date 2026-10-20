"""Single canonical account phase per explicit call; no scheduler, owner or retry loop."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import requests

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("predictive_deployment_cpu", D / "cpu.py")
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
p = cpu.policy
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")


def observation(value):
    """GPU-capacity helper must not register observed unrelated notebooks as new jobs."""
    if isinstance(value, list):
        return [observation(v) for v in value]
    if isinstance(value, dict):
        return {
            ("observed_kernel_id" if k == "kernel_id" else k): observation(v)
            for k, v in value.items()
        }
    return value


def live_probe(remote, plan):
    ref = plan["probe"]
    kernel = remote.kernel("GetKernel", ref["slug"], ref["version"])
    p.require(
        kernel["metadata"]["id"] == ref["kernel_id"]
        and p.sha(kernel["blob"]["source"]) == ref["source_sha256"],
        "Exact original measured probe identity required",
    )
    p.require(
        remote.kernel("GetKernelSessionStatus", ref["slug"], ref["version"]).get("status")
        in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"),
        "Original measured probe must actually be terminal",
    )
    return remote.output_json(ref["slug"], ref["version"], ref["receipt_filename"])


def exact_kernel(remote, request, record):
    value = remote.kernel("GetKernel", request["slug"], 1)
    metadata = value["metadata"]
    p.require(
        value["blob"]["source"] == request["text"] and metadata["id"] == record["kernel_id"],
        "Exact source and notebook required",
    )
    for name in ("isPrivate", "enableGpu", "machineShape", "dockerImage"):
        p.require(metadata.get(name) == request[name], "Notebook metadata changed: " + name)
    p.require(not metadata.get("enableInternet", False), "Internet must remain disabled")
    for name in ("datasetDataSources", "kernelDataSources", "competitionDataSources"):
        p.require(
            sorted(metadata.get(name, [])) == sorted(request[name]),
            "Exact original sources required: " + name,
        )
    return remote.kernel("GetKernelSessionStatus", request["slug"], 1).get("status", "UNREPORTED")


def execute(v):
    os.umask(0o077)
    root = cpu.private_path(v["root"])
    stage = cpu.check_controls(root)
    p.require(stage["source_git"] == v["source_git"], "Frozen owner source required")
    cpu.funding(v, root)
    w = cpu.load("predictive_canonical_workflow", root / "control/common/workflow.py")
    publication = cpu.load(
        "predictive_canonical_publication", root / "control/common/publication.py"
    )
    capacity = cpu.load("predictive_canonical_capacity", root / "control/common/gpu_capacity.py")
    compiler = cpu.load("predictive_final_compiler", root / "control/serving/compile.py")
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + v["token"]
    remote.username = "godlydonuts"
    who = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    p.require(
        who.get("active") is True and who.get("username") == "godlydonuts",
        "Actual owner identity required",
    )
    prepared = cpu.read(root / "prepared.json")
    identity = prepared["bundle_identity"]
    phase, mode = v["phase"], v.get("mode", "qualification")
    p.require(
        phase in ("dataset", "dataset-reconcile", "push", "status", "submit", "outcome"),
        "One explicit canonical phase required",
    )
    p.require(mode == "qualification" or mode in p.ARMS, "Known exact serving mode required")
    claim = LEDGER.parent / "coordinator-claim"
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        return {"status": "waiting_existing_canonical_claim_no_mutation"}
    owner = dict(
        kind="predictive_final_deployment_v1",
        CPU_job=os.environ["SLURM_JOB_ID"],
        phase=phase,
        mode=mode,
        bundle_identity=identity,
    )
    w.new_json(claim / "owner.json", owner)
    retain = False
    try:
        with w.ledger(LEDGER, "godlydonuts") as db:
            w.verify_account(db, remote)
            key = p.FAMILY + identity + (":dataset" if phase.startswith("dataset") else ":" + mode)
            found = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            row = json.loads(found[0]) if found else {}

            def save():
                db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(row)))
                db.commit()

            parent = remote.dataset(p.PARENT_DATASET)
            p.require(
                parent["id"] == p.PARENT_ID and parent["currentVersionNumber"] == 1,
                "Exact existing parent Dataset 12431568 v1 required",
            )
            if phase.startswith("dataset"):
                if phase == "dataset":
                    p.no_prior_intent(root / "dataset-intent.json", row, "dataset")
                    p.require(not row, "Any previous Dataset state requires reconciliation")
                    absence = publication.require_absent(remote, prepared["dataset_slug"])
                    publication.validate_zip(root / "upload/bundle.zip", prepared["files"])
                    p.require(
                        cpu.digest(root / "upload/bundle.zip") == prepared["archive"]["sha256"],
                        "Original CPU-prepared archive changed",
                    )
                    row.update(
                        state="predictive_dataset_create_intent",
                        dataset=True,
                        dataset_slug=prepared["dataset_slug"],
                        version=1,
                        bundle_identity=identity,
                        absence=absence,
                        automatic_retry=False,
                    )
                    w.new_json(root / "dataset-intent.json", row)
                    save()
                    retain = True
                    response = publication.sdk_create_once(
                        v["token"],
                        prepared["dataset_slug"],
                        "RSNA predictive final B C C0",
                        root / "upload",
                    )
                    w.new_json(root / "dataset-response.json", response)
                    row["state"] = "predictive_dataset_create_returned"
                    save()
                    retain = False
                    # Ingestion may still be pending. Verification is a separate read-only phase.
                    return {"status": row["state"], "new_Dataset_calls": 1, "new_GPU_or_entries": 0}
                p.require(row.get("dataset") is True, "Original canonical Dataset intent required")
                if (root / "dataset-verified.json").exists():
                    result = cpu.read(root / "dataset-verified.json")
                    p.require(
                        publication.metadata(remote, prepared["dataset_slug"])
                        == {k: result[k] for k in ("id", "version", "slug", "private")},
                        "Verified immutable Dataset changed",
                    )
                    return result
                verified = publication.reconcile(
                    remote, prepared["dataset_slug"], prepared["files"]
                )
                w.new_json(root / "dataset-verified.json", verified)
                row.update(state="predictive_dataset_fully_verified", verified=verified)
                save()
                return verified

            dataset = cpu.read(root / "dataset-verified.json")
            p.require(
                dataset["status"] == "all_payload_bytes_verified"
                and dataset["downloaded_all_bytes"] is True
                and dataset["files"] == prepared["files"]
                and publication.metadata(remote, prepared["dataset_slug"])
                == {k: dataset[k] for k in ("id", "version", "slug", "private")},
                "Actual exact fully downloaded private Dataset required",
            )
            target = root / ("compiled-" + mode)
            compiled = cpu.read(target / "compiled.json")
            source = (target / "source.private.py").read_text()
            p.require(
                p.sha(source) == compiled["source_sha256"]
                and compiled["bundle_identity"] == identity
                and compiled["dataset"]["id"] == dataset["id"]
                and compiled["model_dataset_ref"] == dataset["slug"],
                "Exact CPU compilation required",
            )
            p.require(
                compiled["public_watchdog_seconds"] == 600,
                "Actual public-only 600-second watchdog must be compiled",
            )
            cap_seconds = v["runtime_plan"]["qualification_cap_seconds"]
            request = p.request(
                source, mode, identity, dataset["slug"], compiler.contract.DOCKER, cap_seconds
            )
            if row:
                p.require(
                    row["source_sha256"] == compiled["source_sha256"], "Canonical source changed"
                )
            if phase == "push":
                p.no_prior_intent(target / "push-intent.json", row, "push")
                p.require(not row, "Existing notebook canonical row requires reconciliation")
                plan = p.runtime_plan(v["runtime_plan"], live_probe(remote, v["runtime_plan"]))
                available = capacity.admission(db, remote, caller="serving")
                quota = remote.call(
                    "kernels.KernelsApiService", "GetAcceleratorQuotaStatistics", {}
                )["gpuQuota"]
                account = remote.account()
                remaining = sum(
                    not (root / ("compiled-" + a) / "accepted.json").exists() for a in p.ARMS
                )
                actual_budget = p.budget(
                    plan,
                    quota,
                    "qualification" if mode == "qualification" else "competition",
                    remaining,
                )
                if (
                    not available["available"]
                    or not actual_budget["available"]
                    or not w.allowance(db, account)
                ):
                    return dict(
                        status="waiting_live_capacity_budget_or_allowance_no_mutation",
                        capacity=observation(available),
                        budget=actual_budget,
                        allowance=account,
                    )
                projection = None
                if mode in p.ARMS:
                    qualification = cpu.read(root / "qualification-accepted.json")
                    candidate = next(c for c in prepared["models"] if c["arm"] == mode)
                    compiler.contract.validate_qualification(qualification, candidate)
                    measurements = next(
                        m["measurements"] for m in qualification["models"] if m["arm"] == mode
                    )
                    projection = p.hidden_projection(
                        plan, measurements, qualification_sha256=p.sha(p.canonical(qualification))
                    )
                absence = remote.require_absent_kernel(request["slug"])
                row.update(
                    state="predictive_final_push_intent",
                    push=True,
                    slug=request["slug"],
                    version=1,
                    source_sha256=compiled["source_sha256"],
                    bundle_identity=identity,
                    mode=mode,
                    model_files=compiled["model_files"],
                    request={k: z for k, z in request.items() if k != "text"},
                    runtime_plan=plan,
                    budget=actual_budget,
                    quota=quota,
                    capacity=observation(available),
                    allowance=account,
                    absence=absence,
                    projection=projection,
                    push_at=time.time(),
                    automatic_retry=False,
                )
                w.new_json(target / "push-intent.json", row)
                save()
                retain = True
                result = remote.call("kernels.KernelsApiService", "SaveKernel", request)
                w.new_json(target / "push-response.json", result)
                p.require(
                    result.get("ref") == "/code/" + request["slug"]
                    and result.get("versionNumber") == 1
                    and type(result.get("kernelId")) is int
                    and result["kernelId"] > 0
                    and not any(
                        result.get(k)
                        for k in (
                            "error",
                            "invalidDatasetSources",
                            "invalidKernelSources",
                            "invalidCompetitionSources",
                        )
                    ),
                    "Notebook push response uncertain",
                )
                row.update(state="predictive_final_pushed", kernel_id=result["kernelId"])
                save()
                exact_kernel(remote, request, row)
                retain = False
                w.new_json(target / "pushed.json", row)
                return {
                    k: row[k] for k in ("state", "slug", "version", "kernel_id", "source_sha256")
                }
            p.require(row.get("kernel_id"), "An actual original notebook push is required")
            request["sessionTimeoutSeconds"] = row["request"]["sessionTimeoutSeconds"]
            state = exact_kernel(remote, request, row)
            if phase == "status":
                result = dict(status=state, slug=row["slug"], version=1, kernel_id=row["kernel_id"])
                if state != "COMPLETE":
                    result["public_watchdog_exceeded"] = (
                        state not in capacity.TERMINAL_STATUSES
                        and (
                            time.time() - row["push_at"]
                            > row["runtime_plan"][
                                "qualification_cap_seconds"
                                if mode == "qualification"
                                else "public_watchdog_seconds"
                            ]
                        )
                    )
                    if state in capacity.TERMINAL_STATUSES:
                        row["state"] = "predictive_final_terminal_failure"
                        save()
                    return result
                receipt_name = (
                    "predictive-qualification-accepted.json"
                    if mode == "qualification"
                    else "predictive-competition-accepted.json"
                )
                receipt = remote.output_json(row["slug"], 1, receipt_name)
                if mode == "qualification":
                    for candidate in prepared["models"]:
                        compiler.contract.validate_qualification(receipt, candidate)
                    p.require(
                        receipt["serving_worker_sha256"] == compiled["worker_sha256"]
                        and receipt["serving_contract_sha256"] == compiled["contract_sha256"]
                        and receipt["serving_helpers_sha256"] == compiled["serving_helpers_sha256"]
                        and receipt["model_dataset_ref"] == dataset["slug"],
                        "Actual qualification used another worker",
                    )
                    destination = root / "qualification-accepted.json"
                else:
                    candidate = next(c for c in prepared["models"] if c["arm"] == mode)
                    p.require(
                        receipt["status"]
                        == "actual_predictive_exact_test_roster_submission_created"
                        and receipt["actual_test_studies"] == 3
                        and receipt["submission_csv_absent"] is False
                        and len(receipt["models"]) == 1
                        and receipt["models"][0]["trained_export_sha256"]
                        == candidate["model"]["sha256"]
                        and receipt["public_watchdog_started"] is True
                        and receipt["public_watchdog_seconds"] == 600,
                        "Exact public model and complete three-study CSV receipt required",
                    )
                    csv = cpu.load(
                        "predictive_public_output_csv", root / "control/common/output_csv.py"
                    )
                    checked = csv.verify_csv(
                        csv.download_csv(remote, row["slug"], 1),
                        compiler.contract.COLUMNS,
                        p.PUBLIC_ROSTER,
                    )
                    p.require(
                        checked["CSV_sha256"] == receipt["CSV_sha256"]
                        and checked["bytes"] == receipt["CSV_bytes"]
                        and receipt["test_roster_sha256"] == p.PUBLIC_ROSTER
                        and receipt["serving_worker_sha256"] == compiled["worker_sha256"]
                        and receipt["serving_contract_sha256"] == compiled["contract_sha256"]
                        and receipt["serving_helpers_sha256"] == compiled["serving_helpers_sha256"]
                        and receipt["model_dataset_ref"] == dataset["slug"],
                        "Actual versioned public CSV or serving code differs",
                    )
                    destination = target / "accepted.json"
                if destination.exists():
                    p.require(cpu.read(destination) == receipt, "Original accepted result changed")
                else:
                    w.new_json(destination, receipt)
                row.update(
                    state="predictive_final_public_accepted",
                    receipt_sha256=p.sha(p.canonical(receipt)),
                )
                save()
                return {**result, "receipt": receipt, "hidden_score_claim": False}
            p.require(
                mode in p.ARMS and state == "COMPLETE" and (target / "accepted.json").exists(),
                "Exact accepted competition notebook required",
            )
            candidate = next(c for c in prepared["models"] if c["arm"] == mode)
            model_sha = candidate["model"]["sha256"]
            description = "predictive-final-v1:" + p.sha(
                p.canonical([identity, mode, row["slug"], 1, row["source_sha256"]])
            )
            if phase == "submit":
                p.no_prior_intent(target / "submit-intent.json", row, "submit")
                account = remote.account()
                p.require(w.allowance(db, account) > 0, "Fresh live daily allowance required")
                p.require(
                    db.execute("SELECT record FROM entries WHERE sha=?", (model_sha,)).fetchone()
                    is None,
                    "Existing model entry requires reconciliation",
                )
                p.require(
                    not any(s.get("description") == description for s in remote.submissions()),
                    "Exact code entry already exists",
                )
                p.require(
                    not any(
                        json.loads(raw)
                        .get("state", "")
                        .endswith(("_submit_intent", "_submit_uncertain"))
                        for (raw,) in db.execute("SELECT value FROM meta")
                        if raw.lstrip().startswith("{")
                    ),
                    "Other uncertain account entry exists",
                )
                plan = p.runtime_plan(v["runtime_plan"], live_probe(remote, v["runtime_plan"]))
                q = cpu.read(root / "qualification-accepted.json")
                p.hidden_projection(
                    plan,
                    next(m["measurements"] for m in q["models"] if m["arm"] == mode),
                    qualification_sha256=p.sha(p.canonical(q)),
                )
                record = dict(
                    state="submit_intent",
                    checkpoint_sha256=model_sha,
                    attempted_at=w.utcnow(),
                    description=description,
                    quota_before=account,
                    operation={"kernel_slug": row["slug"]},
                    kernel_version=1,
                    source_sha256=row["source_sha256"],
                    bundle_identity=identity,
                )
                w.new_json(target / "submit-intent.json", record)
                w.save_entry(db, record, "predictive final exact code entry intent")
                row.update(submit=True)
                save()
                retain = True
                result = remote.competition(
                    "CreateCodeSubmission",
                    {
                        "fileName": "submission.csv",
                        "kernelOwner": "godlydonuts",
                        "kernelSlug": row["slug"].split("/")[1],
                        "kernelVersion": 1,
                        "submissionDescription": description,
                    },
                )
                w.new_json(target / "submit-response.json", result)
                p.require(
                    type(result.get("ref")) is int and result["ref"] > 0,
                    "Competition entry response uncertain",
                )
                record.update(
                    state="submitted", submission_id=result["ref"], submitted_at=w.utcnow()
                )
                w.save_entry(db, record, "one predictive final code entry accepted")
                row.update(state="predictive_final_submitted", submission_id=result["ref"])
                save()
                retain = False
                return record
            p.require(phase == "outcome", "Known outcome phase required")
            matches = [s for s in remote.submissions() if s.get("description") == description]
            p.require(
                len(matches) == 1 and matches[0].get("ref"), "Exact hidden result is ambiguous"
            )
            actual = matches[0]
            original = w.read_entry(db, model_sha)
            p.require(
                original.get("submission_id") in (None, actual["ref"]), "Entry identity changed"
            )
            result = {
                k: actual.get(k)
                for k in ("ref", "status", "publicScore", "errorDescription", "totalBytes")
            }
            result["outcome"] = w.submission_outcome(actual)
            original.update(state="submitted", submission_id=actual["ref"], actual_outcome=result)
            w.save_entry(db, original, "read-only predictive hidden outcome reconciliation")
            return result
    finally:
        if not retain:
            p.require(cpu.read(claim / "owner.json") == owner, "Claim owner changed")
            (claim / "owner.json").unlink()
            claim.rmdir()


if __name__ == "__main__":
    try:
        print(json.dumps(execute(json.load(sys.stdin)), allow_nan=False), flush=True)
    except Exception as error:
        print(
            json.dumps(
                {"status": "blocked", "error_type": type(error).__name__, "automatic_retry": False}
            ),
            flush=True,
        )
        raise SystemExit(1) from None
