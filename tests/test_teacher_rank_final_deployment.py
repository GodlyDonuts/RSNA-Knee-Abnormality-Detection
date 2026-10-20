"""Only role/provenance/runtime deltas over the frozen launched pointwise route."""

import hashlib
import importlib.util
import json
import sqlite3
import types
from pathlib import Path

import pytest

D = Path("initiative/research/teacher-rank-final-deployment-20261020-v1")
spec = importlib.util.spec_from_file_location("prepared_rank_transport", D / "run_once.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def admitted():
    config = json.loads((D / "config.json").read_text())
    decision = json.loads((D / "engineering-decision.json").read_text())
    decision.update(
        status="accepted_teacher_rank_final_runtime_and_storage_plan",
        actual_pointwise_qualification_passed=True,
        actual_pointwise_qualification_receipt_sha256="a" * 64,
        qualification_cap_seconds=500,
        public_watchdog_seconds=100,
    )
    text = json.dumps(decision)
    config["runtime_plan"].update(
        status="accepted_teacher_rank_final_runtime_budget",
        actual_same_graph_timing_receipt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        qualification_cap_seconds=500,
        public_watchdog_seconds=100,
    )
    return config, text


def generated(admitted):
    return runner._delta.sources(runner._parent, *admitted)


def test_pending_actual_timing_prevents_even_an_intent(monkeypatch):
    monkeypatch.setattr(runner, "git_bytes", lambda pin, path: path.read_bytes())
    original = runner._delta.accepted

    def still_pending(config, decision_text):
        config = json.loads(json.dumps(config))
        config["runtime_plan"]["status"] = "pending_actual_pointwise_timing_no_launch"
        return original(config, decision_text)

    monkeypatch.setattr(runner._delta, "accepted", still_pending)
    with pytest.raises(AssertionError, match="timing decision"):
        runner.controls("synthetic_no_remote")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("actual_pointwise_qualification_passed", False),
        ("actual_pointwise_qualification_receipt_sha256", None),
        ("qualification_cap_seconds", 501),
    ],
)
def test_incomplete_or_changed_runtime_decision_rejected(admitted, field, bad):
    config, text = admitted
    decision = json.loads(text)
    decision[field] = bad
    changed = json.dumps(decision)
    config["runtime_plan"]["actual_same_graph_timing_receipt_sha256"] = hashlib.sha256(
        changed.encode()
    ).hexdigest()
    with pytest.raises((AssertionError, TypeError)):
        runner._delta.sources(runner._parent, config, changed)


def test_exact_role_compilation_publication_and_model_bindings(admitted):
    sources = generated(admitted)
    ns = {"__name__": "synthetic_rank_only"}
    exec(sources["program.py"], ns)
    actual = json.loads(
        Path(
            "initiative/research/teacher-pair-final-serving-20261020-v2/actual-teacher_rank-compile.json"
        ).read_text()
    )
    assert ns["MODEL_SHA"] == actual["model_sha256"] == runner._delta.MODEL
    assert ns["COMPILED_SHA"] == {m: v["sha256"] for m, v in actual["outputs"].items()}
    assert ns["ORIGINAL_SERVING"].as_posix().endswith("/final-serving-v2/teacher_rank")
    assert (
        ns["MODEL_KEY"]
        == "licensed-parent-full-epoch-serving:teacher_rank-final:" + actual["model_sha256"]
    )
    assert ns["DATASET_SLUG"] == "godlydonuts/rsna-teacher-rank-final-e3-d478a502e6fd"
    assert ns["D"].name == "rsna-teacher-rank-final-deployment-20261020-v1"
    assert '"979767"' in sources["program.py"]
    assert "60a64c2be5d59f60e458caca1b6f475f82a95fa7" in sources["program.py"]
    assert runner._delta.OLD_MODEL not in sources["program.py"]
    for name in ("workflow", "components", "gpu_capacity", "output_csv", "publication"):
        assert sources[name + ".py"] == runner._parent["scripts/kaggle/" + name + ".py"]


def test_reviewed_caps_match_timer_request_and_fresh_budget(admitted):
    config, _ = admitted
    p = types.ModuleType("synthetic_rank_policy")
    exec(generated(admitted)["policy.py"], p.__dict__)
    p.PLAN = config["runtime_plan"]
    result = p.budget(
        dict(totalTimeAllowed="3600s", timeUsed="0s", timeReserved="0s"),
        3,
        "qualification",
        latest_training_epoch=3,
    )
    assert result["available"] and result["required_VM_seconds"] == 3600
    assert (
        result["session_timeout_seconds"] == 500
        and result["public_example_watchdog_seconds"] == 100
    )
    assert result["hidden_session_timeout_seconds"] == 32400 and not result["paid_credits_allowed"]
    original = "def main():\n    pass\nif __name__ == '__main__':\n    main()\n"
    assert p.public_bounded_source(original, "qualification") == original
    compiled = p.public_bounded_source(original, "competition")
    assert "Timer(100," in compiled and p.ROSTER in compiled and "Timer(300," not in compiled
    t = dict(
        epoch=3, status="COMPLETE", kernel_id=None, training_Slurm_job="975297", role="teacher_rank"
    )
    dataset = dict(
        status="all_payload_bytes_verified",
        private=True,
        version=1,
        downloaded_all_bytes=True,
        slug="godlydonuts/rsna-teacher-rank-final-e3-d478a502e6fd",
    )
    request = p.request(3, "qualification", original, t, dataset=dataset)
    assert request["slug"].startswith("godlydonuts/rsna-teacher-rank-e3-qual-")
    assert (
        request["sessionTimeoutSeconds"] == 500
        and len(request["dockerImage"].split("sha256:")[1]) == 64
    )
    t["role"] = "pointwise"
    with pytest.raises(ValueError):
        p.request(3, "qualification", original, t, dataset=dataset)


def test_existing_pointwise_node_reservation_is_preserved(admitted):
    text = generated(admitted)["program.py"]
    assert 'Path("/tmp/rsna-pointwise-final-deployment-20261020-v1")' in text
    assert "other_node_unpaid += max(0, 640 * 2**20 - pointwise_paid)" in text
    assert 'f["shared_other_liabilities_bytes"] >= 122 * 2**20' in text
    assert 'f["incremental_shared_files"] == 0' in text
    assert 'f["shared_future_files"] >= 376' in text


@pytest.mark.parametrize(
    "error,returned,identity_ok,wait",
    [
        ("Kaggle GetDatasetStatus HTTP 404", True, True, True),
        ("Kaggle GetDatasetStatus HTTP 403", True, True, False),
        ("Kaggle GetDatasetStatus HTTP 404", False, True, False),
        ("Kaggle GetDatasetStatus HTTP 404", True, False, False),
        ("Kaggle GetDatasetStatus HTTP 404", True, "404", True),
        ("Kaggle GetDatasetStatus HTTP 404", True, "403", False),
    ],
)
def test_dataset_status404_waits_only_after_returned_create_and_exact_identity(
    tmp_path, admitted, error, returned, identity_ok, wait
):
    ns = {"__name__": "synthetic_rank_reconcile"}
    exec(generated(admitted)["program.py"], ns)
    ns["D"] = tmp_path

    def put(path, obj):
        with path.open("x") as f:
            json.dump(obj, f)

    put(tmp_path / "bundle-prepared.json", {"files": {}})
    put(
        tmp_path / "dataset-intent.json",
        dict(
            slug=ns["DATASET_SLUG"], model_sha256=ns["MODEL_SHA"], version=1, automatic_retry=False
        ),
    )
    put(
        tmp_path / "dataset-response.json",
        {
            "returned": returned,
            "final_create_rpc_calls": 1,
            "status": "Ok",
            "private_requested": True,
        },
    )

    class RemoteError(RuntimeError):
        pass

    class Remote:
        def call(self, *a):
            raise RemoteError(error)

        def dataset(self, slug):
            if isinstance(identity_ok, str):
                raise RemoteError("Kaggle GetDataset HTTP " + identity_ok)
            return dict(
                ref=slug if identity_ok else "godlydonuts/wrong",
                id=7,
                isPrivate=True,
                currentVersionNumber=1,
            )

    ns["load"] = lambda name: object()
    w = types.SimpleNamespace(RemoteError=RemoteError, new_json=put)
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT)")
        row = dict(
            state="teacher_rank_dataset_create_returned",
            model_sha256=ns["MODEL_SHA"],
            slug=ns["DATASET_SLUG"],
            version=1,
        )
        db.execute("INSERT INTO meta VALUES (?,?)", (ns["MODEL_KEY"] + ":dataset", json.dumps(row)))
        if wait:
            result = ns["dataset_phase"]({"phase": "dataset-reconcile"}, db, Remote(), w)
            assert result["status"] == "waiting_original_dataset_ingestion_no_mutation"
            if identity_ok == "404":
                assert result["identity_verified"] is False
                assert not (tmp_path / "dataset-identity.json").exists()
            else:
                assert json.loads((tmp_path / "dataset-identity.json").read_text())["id"] == 7
        else:
            with pytest.raises((RemoteError, AssertionError)):
                ns["dataset_phase"]({"phase": "dataset-reconcile"}, db, Remote(), w)
            assert not (tmp_path / "dataset-identity.json").exists()
    assert json.loads((tmp_path / "dataset-intent.json").read_text())["automatic_retry"] is False


def test_owner_requires_original_pointwise_entry_but_not_its_hidden_completion():
    text = runner._delta.owner_source(runner._parent)
    assert (
        "actual-qualification-accepted.json" in text
        and "actual-competition-accepted.json" in text
        and "actual-entry.json" in text
    )
    assert "source-zero-final-serving-20261020-v2" not in text
    assert 'pointwise / "complete.json"' not in text
    assert "maximum_new_Datasets=1" in text and "maximum_new_entries=1" in text
    compile(text, "synthetic_finite_rank_owner", "exec")


def test_both_metadata_404s_wait_then_eventual_ready_verifies_without_create(tmp_path, admitted):
    ns = {"__name__": "synthetic_rank_propagation"}
    exec(generated(admitted)["program.py"], ns)
    ns["D"] = tmp_path

    def put(path, obj):
        with path.open("x") as f:
            json.dump(obj, f)

    files = {"epoch-003.pt": {"bytes": 1, "sha256": "a" * 64}}
    put(tmp_path / "bundle-prepared.json", {"files": files})
    intent = dict(
        slug=ns["DATASET_SLUG"], model_sha256=ns["MODEL_SHA"], version=1, automatic_retry=False
    )
    put(tmp_path / "dataset-intent.json", intent)
    put(
        tmp_path / "dataset-response.json",
        dict(returned=True, final_create_rpc_calls=1, status="Ok", private_requested=True),
    )
    original_intent = (tmp_path / "dataset-intent.json").read_bytes()
    rpc = []

    class RemoteError(RuntimeError):
        pass

    class Remote:
        ready = False

        def call(self, service, method, request):
            rpc.append(method)
            assert method == "GetDatasetStatus"
            if not self.ready:
                raise RemoteError("Kaggle GetDatasetStatus HTTP 404")
            return {"status": "READY"}

        def dataset(self, slug):
            rpc.append("GetDataset")
            assert slug == ns["DATASET_SLUG"]
            raise RemoteError("Kaggle GetDataset HTTP 404")

    remote = Remote()
    verification_calls = []

    def reconcile(actual_remote, slug, expected):
        assert (
            remote.ready
            and actual_remote is remote
            and slug == ns["DATASET_SLUG"]
            and expected == files
        )
        verification_calls.append(1)
        return dict(
            status="all_payload_bytes_verified",
            id=23,
            slug=slug,
            private=True,
            version=1,
            files=expected,
            remote_names={"epoch-003.pt": "epoch-003.pt"},
            downloaded_all_bytes=True,
        )

    ns["load"] = lambda name: types.SimpleNamespace(reconcile=reconcile)
    workflow = types.SimpleNamespace(RemoteError=RemoteError, new_json=put)
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT)")
        row = {**intent, "state": "teacher_rank_dataset_create_returned"}
        db.execute("INSERT INTO meta VALUES (?,?)", (ns["MODEL_KEY"] + ":dataset", json.dumps(row)))
        waiting = ns["dataset_phase"]({"phase": "dataset-reconcile"}, db, remote, workflow)
        assert waiting["status"] == "waiting_original_dataset_ingestion_no_mutation"
        assert waiting["identity_verified"] is False and waiting["new_Dataset_calls"] == 0
        assert waiting["bounded_by_finite_owner_seconds"] == 43200
        assert (
            not (tmp_path / "dataset-identity.json").exists()
            and not (tmp_path / "dataset-verified.json").exists()
        )
        remote.ready = True
        ready = ns["dataset_phase"]({"phase": "dataset-reconcile"}, db, remote, workflow)
        assert ready["status"] == "all_payload_bytes_verified" and ready["id"] == 23
        assert json.loads((tmp_path / "dataset-verified.json").read_text()) == ready
    assert rpc == ["GetDatasetStatus", "GetDataset", "GetDatasetStatus"]
    assert (
        verification_calls == [1]
        and (tmp_path / "dataset-intent.json").read_bytes() == original_intent
    )
