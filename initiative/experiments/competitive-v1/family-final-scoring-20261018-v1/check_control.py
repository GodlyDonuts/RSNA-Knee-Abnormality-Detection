"""Synthetic transport/admission tests, never a model or hardware acceptance."""

import contextlib
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path.cwd()
SCRIPT = Path(__file__).with_name("program.py.txt")
ns = {"__name__": "synthetic_scoring_control_tests"}
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), ns)

spec = importlib.util.spec_from_file_location(
    "actual_workflow_for_pure_outcomes", ROOT / "scripts/kaggle/workflow.py"
)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


@pytest.fixture
def control(tmp_path, monkeypatch):
    state = {
        "capacity": True,
        "quota_used": "1000s",
        "save": "ok",
        "calls": [],
        "account": {
            "entered": True,
            "checked_at": workflow.utcnow(),
            "num_today": 5,
            "num_allowed_now": 0,
        },
        "submissions": [
            {"ref": 57332456, "status": "COMPLETE", "publicScore": 0.949, "description": "prior"}
        ],
    }
    P = tmp_path / "parent"
    R = P / "serving"
    M = R / "single"
    S = R / "code"
    M.mkdir(parents=True)
    S.mkdir()
    T = R / "family-template"
    T.mkdir()
    (T / "accepted.json").write_text(json.dumps({"CPU_job": "953320", "source_files": {}}))
    (R / "prepared.json").write_text(
        json.dumps(
            {
                "CPU_job": "946894",
                "source_files": {},
                "columns": [],
                "expected_public_roster_sha256": "fixture-roster",
            }
        )
    )
    (M / "program.py.txt").write_text("fixture-frozen-control")
    (M / "accepted.json").write_text(json.dumps({"code_source_git": ns["CODE_SOURCE"]}))
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);CREATE TABLE events(at TEXT,sha TEXT,event TEXT);CREATE TABLE entries(record TEXT);"
    )
    db.execute(
        "INSERT INTO meta VALUES(?,?)",
        (ns["REFERENCE_KEY"], json.dumps({"state": "public0958_UI_current_container_pushed"})),
    )
    row = None
    for i, candidate in enumerate(ns["CANDIDATES"]):
        D = R / candidate
        D.mkdir()
        model = hashlib.sha256(candidate.encode()).hexdigest()
        base = {
            "epoch": 3,
            "model_sha256": model,
            "training": {
                "role": "influence",
                "slug": "fixture/trained",
                "source_sha256": "f" * 64,
                "receipt": {"completed": 1458, "fixed_final_complete": True},
            },
            "publication": {
                "status": "actual_S6_label_epoch_private_full_download_size_SHA_verified"
            },
            "prepared": {},
        }
        for mode in ("qualification", "competition"):
            text = candidate + "-accepted-fixture-source-" + mode
            (D / (mode + ".private.py")).write_text(text)
            base["prepared"][mode] = {"source_sha256": hashlib.sha256(text.encode()).hexdigest()}
        (D / "prepared.json").write_text(json.dumps(base))
        row = {**base, "state": "importance_epoch_serving_prepared"}
        db.execute(
            "INSERT INTO meta VALUES(?,?)",
            ("licensed-parent-full-epoch-serving:" + model, json.dumps(row)),
        )
    whole = tmp_path / "whole"
    for e in (1, 2, 3):
        d = whole / ("segment-" + str(e))
        d.mkdir(parents=True)
        (d / "actual-accepted.json").write_text(
            json.dumps({"status": "COMPLETE", "receipt": {"completed": 486 * e}})
        )
        for candidate in ("uniform", "influence"):
            d = P / "full-comparison" / (candidate + "-epoch-" + str(e))
            d.mkdir(parents=True)
            (d / "actual-accepted.json").write_text(
                json.dumps({"status": "COMPLETE", "receipt": {"completed": 486 * e}})
            )
    contract = tmp_path / "contract.py.txt"
    contract.write_bytes(
        (
            ROOT
            / "initiative/experiments/competitive-v1/s6-source-label-contrast-v1/recovery_contract.py.txt"
        ).read_bytes()
    )
    claim = tmp_path / "claim"
    cap = tmp_path / "cap.py"
    cap.write_text("fixture capacity source")
    ext = P / "finding-strata-audit-v1/accepted-capacity.py"
    original_Path = Path

    def remap(path):
        value = str(path)
        if value.endswith("/coordinator-claim"):
            return claim
        if value.endswith("/GPU-admission-v1/gpu_capacity.py"):
            return cap
        if value.endswith("/production-template-v1/recovery_contract.py.txt"):
            return contract
        if value == "/tmp/rsna-licensed-parent-whole94-production-20261016-v1":
            return whole
        return original_Path(path)

    monkeypatch.setitem(ns, "Path", remap)
    for n, value in [("P", P), ("BASE", R), ("HOME", P), ("R", R), ("M", M), ("S", S), ("T", T)]:
        monkeypatch.setitem(ns, n, value)
    monkeypatch.setitem(ns, "bounds", lambda extra=0: None)
    old_sha = ns["sha"]
    monkeypatch.setitem(
        ns,
        "sha",
        lambda p: (
            "0df45fa6f4724f15fe819794ecade9b5b4e64cc68037457b8db5049cb2fd66a2"
            if p == cap
            else "1c915d09f5c39c341bf910818cdd45f234c054ccb3f1fe4225c352c7821a70ea"
            if p == ext
            else old_sha(p)
        ),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-CPU")

    def store(path, value):
        with path.open("x") as f:
            json.dump(value, f)

    class FakeRemote:
        def call(self, service, method, body):
            state["calls"].append((method, body))
            if method == "IntrospectToken":
                return {"active": True, "username": "godlydonuts"}
            if method == "GetAcceleratorQuotaStatistics":
                return {
                    "gpuQuota": {
                        "timeUsed": state["quota_used"],
                        "timeReserved": "0s",
                        "totalTimeAllowed": "162000s",
                        "minimumTimeAllowed": "108000s",
                        "isPayToScaleEnabled": True,
                    }
                }
            if method == "SaveKernel":
                if state["save"] == "exception":
                    raise TimeoutError("fixture uncertain SaveKernel")
                if state["save"] == "rejected":
                    return {"error": "Maximum batch GPU session count of 2 reached."}
                return {"ref": "/code/" + body["slug"], "versionNumber": 1, "kernelId": 999}
            raise AssertionError(method)

        def require_absent_kernel(self, slug):
            return {"absent": True}

        def kernel(self, method, slug, version):
            if method == "GetKernelSessionStatus":
                return {"status": "COMPLETE"}
            candidate = state.get("candidate", "family")
            mode = state.get("mode", "competition")
            D = R / candidate
            base = json.loads((D / "prepared.json").read_text())
            text = (D / (mode + ".private.py")).read_text()
            request = ns["request_for"](candidate, mode, base, text)
            return {"metadata": request, "blob": {"source": text}}

        def output_json(self, slug, version, name):
            return state.get("output", {"CSV_sha256": "fixture-CSV"})

        def account(self):
            return state["account"]

        def submissions(self):
            return state["submissions"]

        def competition(self, method, body):
            state["calls"].append((method, body))
            assert method == "CreateCodeSubmission"
            if state.get("submit_exception"):
                raise TimeoutError("fixture uncertain CreateCodeSubmission")
            return {"ref": 998}

    @contextlib.contextmanager
    def ledger(*args):
        yield db

    w = SimpleNamespace(
        Remote=FakeRemote,
        ledger=ledger,
        verify_account=lambda *args: None,
        new_json=store,
        object_hash=lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest(),
        utcnow=workflow.utcnow,
        allowance=workflow.allowance,
        submission_outcome=workflow.submission_outcome,
    )
    monkeypatch.setitem(
        ns,
        "load",
        lambda name, path: (
            SimpleNamespace(
                admission=lambda *args, **kwargs: {"available": state["capacity"]}, FAMILIES=()
            )
            if path in (cap, ext)
            else w
        ),
    )
    payload = {
        "phase": "push",
        "candidate": "family",
        "mode": "qualification",
        "token": "fixture-token",
        "program_sha256": hashlib.sha256(b"fixture-frozen-control").hexdigest(),
        "control_git": "fixture-control",
    }

    def get(candidate="family"):
        key = "licensed-parent-full-epoch-serving:" + hashlib.sha256(candidate.encode()).hexdigest()
        return key, json.loads(
            db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0]
        )

    def save_row(candidate, row):
        key, _ = get(candidate)
        db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(row), key))
        db.commit()

    def accepted_entry(candidate="family"):
        state["candidate"] = candidate
        D = R / candidate
        key, row = get(candidate)
        base = json.loads((D / "prepared.json").read_text())
        text = (D / "competition.private.py").read_text()
        request = ns["request_for"](candidate, "competition", base, text)
        row.update(
            state="full_parent_epoch_competition_accepted",
            competition={
                "source_sha256": base["prepared"]["competition"]["source_sha256"],
                "kernel_id": 999,
                "slug": request["slug"],
                "receipt": {"CSV_sha256": "fixture-CSV"},
            },
        )
        save_row(candidate, row)
        (D / "competition-actual-accepted.json").write_text(
            json.dumps(
                {
                    "receipt": row["competition"]["receipt"],
                    "CSV_verified_in_RAM": {"test_roster_sha256": "fixture-roster"},
                }
            )
        )
        return {**payload, "candidate": candidate, "mode": "competition"}

    return SimpleNamespace(
        ns=ns,
        state=state,
        R=R,
        P=P,
        M=M,
        claim=claim,
        db=db,
        payload=payload,
        get=get,
        save=save_row,
        accepted_entry=accepted_entry,
        whole=whole,
    )


def mutations(control):
    return [x for x in control.state["calls"] if x[0] in ("SaveKernel", "CreateCodeSubmission")]


def test_busy_claim_no_mutation(control):
    control.claim.mkdir()
    assert (
        control.ns["execute"](control.payload)["status"]
        == "waiting_existing_canonical_claim_no_mutation"
    )
    assert mutations(control) == []


@pytest.mark.parametrize("gate", ["capacity", "budget"])
def test_actual_resource_gate_never_creates_push_intent(control, gate):
    if gate == "capacity":
        control.state["capacity"] = False
    else:
        control.state["quota_used"] = "161000s"
    assert control.ns["execute"](control.payload)["status"].startswith(
        "waiting_actual_capacity_or_current_complete_VM_caps"
    )
    assert (
        mutations(control) == []
        and not (control.R / "family/qualification-push-intent.json").exists()
    )


def test_one_GPU_push_then_refuse_duplicate(control):
    assert (
        control.ns["execute"](control.payload)["status"]
        == "one_exact_matched_final_qualification_pushed"
    )
    with pytest.raises(AssertionError):
        control.ns["execute"](control.payload)
    assert len(mutations(control)) == 1 and not control.claim.exists()


@pytest.mark.parametrize("response", ["exception", "rejected"])
def test_uncertain_save_retains_original_claim_without_retry(control, response):
    control.state["save"] = response
    with pytest.raises((TimeoutError, AssertionError)):
        control.ns["execute"](control.payload)
    assert (
        control.claim.exists()
        and control.get()[1]["state"] == "full_parent_epoch_qualification_push_uncertain"
    )
    assert (
        control.ns["execute"](control.payload)["status"]
        == "waiting_existing_canonical_claim_no_mutation"
    )
    assert len(mutations(control)) == 1


def test_changed_staged_control_is_refused(control):
    (control.M / "program.py.txt").write_text("changed control")
    with pytest.raises(AssertionError):
        control.ns["execute"](control.payload)
    assert mutations(control) == []


def test_zero_live_slots_cannot_submit(control):
    v = control.accepted_entry()
    v["phase"] = "submit"
    with pytest.raises(AssertionError):
        control.ns["execute"](v)
    assert mutations(control) == [] and not (control.R / "family/entry-submit-intent.json").exists()


def test_one_exact_code_entry_and_no_public_csv_upload(control):
    v = control.accepted_entry()
    v["phase"] = "submit"
    control.state["account"].update(num_today=0, num_allowed_now=5)
    out = control.ns["execute"](v)
    assert out["submission_id"] == 998 and out["public_CSV_uploaded"] is False
    with pytest.raises(AssertionError):
        control.ns["execute"](v)
    calls = mutations(control)
    assert len(calls) == 1 and calls[0][0] == "CreateCodeSubmission"
    assert calls[0][1]["fileName"] == "submission.csv" and calls[0][1]["kernelVersion"] == 1


def test_uncertain_entry_is_not_retried(control):
    v = control.accepted_entry()
    v["phase"] = "submit"
    control.state["account"].update(num_today=0, num_allowed_now=5)
    control.state["submit_exception"] = True
    with pytest.raises(TimeoutError):
        control.ns["execute"](v)
    assert (
        control.claim.exists() and control.get()[1]["state"] == "full_parent_epoch_submit_uncertain"
    )
    assert control.ns["execute"](v)["status"] == "waiting_existing_canonical_claim_no_mutation"
    assert len(mutations(control)) == 1


def test_unrelated_reference_can_be_pending(control):
    v = control.accepted_entry()
    v["phase"] = "eligibility"
    control.state["account"].update(num_today=0, num_allowed_now=5)
    control.state["submissions"].append({"ref": 57333150, "status": "RUNNING"})
    out = control.ns["execute"](v)
    assert out["fresh"]["eligible"] and out["fresh"]["active_unresolved_submission_ids"] == [
        57333150
    ]
    assert mutations(control) == []


@pytest.mark.parametrize("kind", ["uncertain_entry", "model_alias", "description"])
def test_existing_or_uncertain_entry_needs_readonly_reconciliation(control, kind):
    v = control.accepted_entry()
    v["phase"] = "eligibility"
    control.state["account"].update(num_today=0, num_allowed_now=5)
    if kind == "uncertain_entry":
        row = {"state": "some_other_submit_uncertain"}
    elif kind == "model_alias":
        row = {"model_sha256": hashlib.sha256(b"family").hexdigest(), "submission_id": 123}
    else:
        row = {}
        control.state["submissions"].append(
            {
                "ref": 123,
                "description": "rsna-final-family-e3:" + hashlib.sha256(b"family").hexdigest(),
            }
        )
    control.db.execute("INSERT INTO meta VALUES(?,?)", ("alias", json.dumps(row)))
    with pytest.raises(AssertionError):
        control.ns["execute"](v)
    assert mutations(control) == []


def test_pending_hidden_is_not_a_score(control):
    v = control.accepted_entry()
    v["phase"] = "outcome"
    key, row = control.get()
    row.update(state="full_parent_epoch_submitted", submission_id=998)
    control.save("family", row)
    control.state["submissions"] = [{"ref": 998, "status": "RUNNING"}]
    out = control.ns["execute"](v)
    assert out["outcome"] == "pending_or_unknown" and out["public_score"] is None
    assert control.get()[1]["state"] == "full_parent_epoch_submitted" and mutations(control) == []


def test_nine_hour_offline_twoT4_public_request(control):
    base = json.loads((control.R / "family/prepared.json").read_text())
    text = (control.R / "family/competition.private.py").read_text()
    req = control.ns["request_for"]("family", "competition", base, text)
    assert (
        req["sessionTimeoutSeconds"] == 32400
        and req["enableInternet"] is False
        and req["machineShape"] == "NvidiaTeslaT4"
    )
    assert (
        req["kernelDataSources"] == ["fixture/trained"]
        and req["competitionDataSources"] == ["rsna-knee-abnormality-detection"]
        and req["datasetDataSources"] == []
    )


def test_COMPLETE_without_actual_model_qualification_receipt_is_refused(control):
    v = dict(control.payload, phase="status")
    key, row = control.get()
    source = (control.R / "family/qualification.private.py").read_text()
    request = control.ns["request_for"](
        "family",
        "qualification",
        json.loads((control.R / "family/prepared.json").read_text()),
        source,
    )
    assert request["kernelDataSources"][-1] == "godlydonuts/rsna-parent-exact-pixels32-09254e97913d"
    row.update(
        state="full_parent_epoch_qualification_pushed",
        qualification={
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "kernel_id": 999,
        },
    )
    control.save("family", row)
    control.state["mode"] = "qualification"
    with pytest.raises((AssertionError, KeyError)):
        control.ns["execute"](v)
    assert (
        not (control.R / "family/qualification-actual-accepted.json").exists()
        and mutations(control) == []
    )
