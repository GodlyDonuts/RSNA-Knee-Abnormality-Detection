"""Actual coordinator protocol tests with synthetic state; no network/model acceptance."""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_licensed_importance_serving_notebook import sources
from test_licensed_parent_serving_notebook import fixture_parent
from test_S6_twoT4_complete_epoch import (
    synthetic_complete_target_epoch as synthetic_complete_target_epoch,
)
from test_S6_twoT4_serving import D, compiler, load
from test_S6_twoT4_serving import new_epoch as new_epoch

policy = load("serving_policy")


@pytest.fixture
def synthetic_serving_source(monkeypatch, new_epoch):
    _, training, pub, _ = new_epoch
    training["slug"] = "godlydonuts/synthetic-trained"
    training["version"] = 1
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE", compiler.sha(parent))
    helper = sources()
    helper[1] = (D / "serving_validate.py.txt").read_text()
    source, meta = compiler.build(
        parent,
        *helper,
        training,
        pub,
        mode="competition",
        legacy_source=Path("scripts/kaggle/licensed_importance_serving.py").read_text(),
    )
    return source, meta, training


def test_public_deadline_keeps_all_image_and_prediction_functions(synthetic_serving_source):
    source, _, training = synthetic_serving_source
    bounded = policy.public_bounded_source(source, "competition")
    old = {
        n.name: ast.dump(n)
        for n in ast.parse(source).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    new = {
        n.name: ast.dump(n)
        for n in ast.parse(bounded).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    assert all(new[n] == value for n, value in old.items())
    assert set(new) - set(old) == {"_actual_public_only_watchdog"}
    assert policy.public_bounded_source(source, "qualification") == source
    request = policy.request(1, "competition", bounded, training)
    assert request["sessionTimeoutSeconds"] == 32400 and not request["enableInternet"]
    assert request["kernelDataSources"] == [policy.OFFLINE, training["slug"]]
    assert request["datasetDataSources"] == [] and request["competitionDataSources"] == [
        "rsna-knee-abnormality-detection"
    ]


@pytest.mark.parametrize("count,armed", [(3, True), (100, False)])
def test_deadline_only_bounds_the_real_small_example(synthetic_serving_source, count, armed):
    bounded = policy.public_bounded_source(synthetic_serving_source[0], "competition")
    fn = next(
        n
        for n in ast.parse(bounded).body
        if isinstance(n, ast.FunctionDef) and n.name == "_actual_public_only_watchdog"
    )
    events = []

    class Timer:
        def __init__(self, seconds, callback):
            events.append(seconds)

        def start(self):
            events.append("started")

    ns = {
        "find_test_root": lambda: "/synthetic-private-input",
        "threading": SimpleNamespace(Timer=Timer),
        "os": SimpleNamespace(_exit=lambda c: None),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "deadline", "exec"), ns)
    with patch.dict(
        sys.modules, {"pandas": SimpleNamespace(read_csv=lambda *a, **k: list(range(count)))}
    ):
        ns["_actual_public_only_watchdog"]()
    assert events == ([600, "started"] if armed else [])


@pytest.mark.parametrize(
    "epoch,mode,required",
    [
        (1, "qualification", 18000),
        (1, "competition", 16200),
        (3, "qualification", 3000),
        (3, "competition", 1200),
    ],
)
def test_budget_preserves_training_and_two_actual_serving_pairs(epoch, mode, required):
    quota = {
        "totalTimeAllowed": str(required) + "s",
        "timeUsed": "0s",
        "timeReserved": "0s",
        "minimumTimeAllowed": "9999999s",
    }
    assert policy.budget(quota, epoch, mode, latest_training_epoch=epoch)["available"]
    quota["timeReserved"] = "1s"
    assert not policy.budget(quota, epoch, mode, latest_training_epoch=epoch)["available"]


@pytest.mark.parametrize("quota", [{}, {"totalTimeAllowed": "NaNs"}, {"totalTimeAllowed": "-1s"}])
def test_unknown_or_nonfinite_budget_refused(quota):
    with pytest.raises(ValueError):
        policy.budget(quota, 1, "qualification", latest_training_epoch=1)


@pytest.mark.parametrize(
    "phase,uncertain", [("push", False), ("push", True), ("submit", False), ("submit", True)]
)
def test_real_state_machine_prepare_uses_existing_kernel_not_a_dataset(
    tmp_path, monkeypatch, synthetic_serving_source, phase, uncertain
):
    from scripts.kaggle import workflow

    cloud = load("serving_cloud")
    source, _, training = synthetic_serving_source
    modelsha = training["receipt"]["epoch"]["sha256"]
    d = tmp_path / "owner"
    f = d / "serving-account-control-v1"
    f.mkdir(parents=True)
    (f / "staged.json").write_text(json.dumps({"source_git": "a" * 40, "source_sha256": {}}))
    e = d / "epoch-1"
    s = e / "serving-v1"
    s.mkdir(parents=True)
    (e / "actual-accepted.json").write_text(json.dumps(training))
    (e / "publication-private-verified.json").write_text(
        json.dumps(
            {
                "model_sha256": modelsha,
                "epoch": 1,
                "kernel_id": training["kernel_id"],
                "original_source_sha256": training["source_sha256"],
            }
        )
    )
    outputs = {}
    for mode in policy.CAPS:
        (s / (mode + ".private.py")).write_text(source)
        outputs[mode] = {
            "bytes": len(source.encode()),
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
    (s / "compiled.json").write_text(
        json.dumps(
            {
                "status": "actual_privately_preserved_complete_target_twoT4_epoch_serving_sources_compiled",
                "epoch": 1,
                "model_sha256": modelsha,
                "outputs": outputs,
            }
        )
    )
    claimroot = tmp_path / "shared"
    claimroot.mkdir()
    originalPath = Path
    monkeypatch.setattr(
        cloud,
        "Path",
        lambda value: originalPath(
            str(value).replace("/home/sa305415/rsna-knee/kaggle", str(claimroot))
        ),
    )
    monkeypatch.setattr(cloud, "D", d)
    monkeypatch.setattr(cloud, "F", f)
    monkeypatch.setattr(cloud, "bounds", lambda: None)
    dbpath = tmp_path / "ledger.sqlite3"
    realledger = workflow.ledger
    w = SimpleNamespace(**{n: getattr(workflow, n) for n in ["new_json", "utcnow"]})

    class Remote:
        def call(self, *args):
            assert args[1] == "IntrospectToken"
            return {"active": True, "username": "godlydonuts"}

    w.Remote = Remote
    w.verify_account = lambda *a: None
    w.ledger = lambda path, user: realledger(dbpath, user)
    monkeypatch.setattr(cloud, "load", lambda name: {"workflow": w, "policy": policy}[name])
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-only")
    payload = {
        "phase": "prepare",
        "epoch": 1,
        "source_git": "a" * 40,
        "source_sha256": {},
        "token": "synthetic-only",
    }
    result = cloud.execute(payload)
    assert result["new_GPU_or_entry_mutations"] == 0
    assert not (claimroot / "coordinator-claim").exists()
    row = json.loads((s / "account-phase-v1/prepared.json").read_text())
    assert row["state"] == "full_parent_epoch_model_verified" and "dataset" not in row
    with pytest.raises(AssertionError):
        cloud.execute(payload)

    # Continue through the real mutation-intent protocol with a synthetic remote.
    mode = "qualification" if phase == "push" else "competition"
    runtime_source = (s / "account-phase-v1" / (mode + ".private.py")).read_text()
    req = policy.request(1, mode, runtime_source, training)
    calls = []

    def call(self, service, method, body):
        calls.append((method, body))
        if method == "IntrospectToken":
            return {"active": True, "username": "godlydonuts"}
        if method == "GetAcceleratorQuotaStatistics":
            return {"gpuQuota": {"totalTimeAllowed": "999999s"}}
        if method in ("SaveKernel", "CreateCodeSubmission"):
            if uncertain:
                raise RuntimeError("synthetic RPC uncertainty after original intent")
            return (
                {"ref": "/code/" + req["slug"], "versionNumber": 1, "kernelId": 321}
                if method == "SaveKernel"
                else {"ref": 123456}
            )
        raise AssertionError(method)

    Remote.call = call
    Remote.kernel = lambda self, method, slug, version: (
        {"status": "COMPLETE"}
        if method == "GetKernelSessionStatus"
        else {"metadata": {**req, "id": 321}, "blob": {"source": runtime_source}}
    )
    Remote.require_absent_kernel = lambda self, slug: {"synthetic_owned_absence": True}
    Remote.account = lambda self: {
        "entered": True,
        "checked_at": workflow.utcnow(),
        "num_today": 0,
        "num_allowed_now": 5,
    }
    Remote.submissions = lambda self: []
    Remote.competition = lambda self, method, body: call(self, "competition", method, body)
    for name in ("SAVE_KERNEL_INVALID", "object_hash", "allowance"):
        setattr(w, name, getattr(workflow, name))
    capacity = SimpleNamespace(admission=lambda *a, **k: {"available": True})
    monkeypatch.setattr(
        cloud,
        "load",
        lambda name: {"workflow": w, "policy": policy, "gpu_capacity": capacity}[name],
    )
    key = "licensed-parent-full-epoch-serving:" + modelsha
    with realledger(dbpath, "godlydonuts") as db:
        db.execute(
            "INSERT INTO meta VALUES (?,?)",
            (
                "licensed-importance-full-fit:complete-S6-cross:synthetic",
                json.dumps(
                    {"epoch": 1, "state": "complete_teacher_cross_full_fit_actual_complete"}
                ),
            ),
        )
        if phase == "submit":
            row["state"] = "full_parent_epoch_competition_accepted"
            row["competition"] = {
                "kernel_id": 321,
                "source_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
                "receipt": {"synthetic": True},
            }
            db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(row), key))
            (s / "account-phase-v1/competition-actual-accepted.json").write_text(
                json.dumps(
                    {
                        "receipt": row["competition"]["receipt"],
                        "CSV_verified_in_RAM": {"test_roster_sha256": policy.ROSTER},
                    }
                )
            )
        db.commit()
    payload.update(phase=phase, mode=mode)
    mutation = "SaveKernel" if phase == "push" else "CreateCodeSubmission"
    if uncertain:
        with pytest.raises(RuntimeError):
            cloud.execute(payload)
        assert (claimroot / "coordinator-claim/owner.json").exists()
        assert cloud.execute(payload)["status"] == "waiting_existing_canonical_claim_no_mutation"
    else:
        result = cloud.execute(payload)
        assert not (claimroot / "coordinator-claim").exists()
        if phase == "submit":
            assert (
                result["submission_id"] == 123456 and result["public_example_CSV_uploaded"] is False
            )
            actual = next(b for method, b in calls if method == mutation)
            assert actual["kernelVersion"] == 1 and actual["fileName"] == "submission.csv"
        with pytest.raises(AssertionError):
            cloud.execute(payload)
    assert sum(method == mutation for method, _ in calls) == 1
