"""An uncertain slug must never be retried or treated as an accepted version."""

import copy
from types import SimpleNamespace

import pytest

from scripts.kaggle import coordinator as c
from scripts.kaggle.workflow import RemoteError


def case():
    old = {
        "kernel_slug": "user/original",
        "dataset_slug": "user/data",
        "manifest_sha256": "m",
        "kernel_source_sha256": "s",
        "fitting_admission_sha256": "f",
    }
    entry = {
        "state": "push_uncertain",
        "operation": old,
        "operation_sha256": "a" * 64,
        "dataset": {"id": 7},
        "kernel_absence_proof": {"count": 4},
    }
    new = dict(
        old,
        checkpoint_sha256="c" * 64,
        kernel_slug="user/distinct",
        dataset_publication_operation_sha256="a" * 64,
    )
    saved = []
    workflow = SimpleNamespace(
        verify_account=lambda *_: None,
        read_entry=lambda *_: copy.deepcopy(entry),
        save_entry=lambda db, record, event: saved.append(copy.deepcopy(record)),
        RemoteError=RemoteError,
        utcnow=lambda: "2026-10-04T20:40:00+00:00",
    )
    placeholder = {
        "kernels": [{"title": "[Private Notebook]", "lastRunTime": "2010-04-01T00:00:00Z"}]
    }

    def call(service, method, body):
        if method == "GetKernel":
            raise RemoteError("Kaggle GetKernel HTTP 403")
        assert method == "ListKernels" and body["search"] == "original"
        return placeholder

    return (
        entry,
        new,
        workflow,
        SimpleNamespace(call=call),
        SimpleNamespace(execute=lambda *_: None),
        saved,
    )


def test_distinct_identity_preserves_uncertainty_and_original_source_dataset():
    entry, new, workflow, remote, db, saved = case()
    c.supersede_uncertain_private_placeholder(workflow, remote, db, new, "a" * 64)
    record = saved[0]
    assert record["state"] == "queued"
    history = record["superseded_operations"][0]
    assert history["state"] == "push_uncertain_never_retry" and history["kernel_version"] is None
    assert history["operation"] == entry["operation"] and history["dataset"] == entry["dataset"]
    assert "operation" not in record and "operation_sha256" not in record


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "pushed"),
        ("kernel_version", 1),
        ("submission_id", 2),
        ("description", "attempted"),
        ("operation_sha256", "b" * 64),
    ],
)
def test_no_supersession_of_known_or_submitted_identity(field, value):
    entry, new, workflow, remote, db, saved = case()
    entry[field] = value
    with pytest.raises(ValueError, match="exact unsubmitted uncertain"):
        c.supersede_uncertain_private_placeholder(workflow, remote, db, new, "a" * 64)
    assert not saved


@pytest.mark.parametrize(
    "field",
    [
        "kernel_slug",
        "dataset_slug",
        "manifest_sha256",
        "kernel_source_sha256",
        "fitting_admission_sha256",
    ],
)
def test_same_slug_or_changed_payload_rejected(field):
    entry, new, workflow, remote, db, saved = case()
    new[field] = entry["operation"][field] if field == "kernel_slug" else "changed"
    with pytest.raises(ValueError, match="exact unsubmitted uncertain"):
        c.supersede_uncertain_private_placeholder(workflow, remote, db, new, "a" * 64)
    assert not saved


def test_accessible_original_requires_version_reconciliation():
    _, new, workflow, remote, db, saved = case()
    remote.call = lambda *_: {"metadata": {"currentVersionNumber": 1}}
    with pytest.raises(ValueError, match="reconcile its exact version"):
        c.supersede_uncertain_private_placeholder(workflow, remote, db, new, "a" * 64)
    assert not saved


def test_placeholder_must_be_the_exact_readonly_observation():
    _, new, workflow, remote, db, saved = case()
    original = remote.call
    remote.call = lambda service, method, body: (
        original(service, method, body) if method == "GetKernel" else {"kernels": []}
    )
    with pytest.raises(ValueError, match="placeholder observation differs"):
        c.supersede_uncertain_private_placeholder(workflow, remote, db, new, "a" * 64)
    assert not saved


def test_identity_reuse_accepts_original_already_pinned_runtime(tmp_path):
    import json

    old = {
        "kernel_slug": "user/old",
        "kernel_metadata_sha256": "old",
        "runtime_image": "pinned",
        "runtime_reference_checkpoint_sha256": "r",
        "manifest_sha256": "m",
        "kernel_source_sha256": "s",
    }
    original = tmp_path / "old.json"
    original.write_text(json.dumps(old))
    new = old | {
        "kernel_slug": "user/new",
        "kernel_metadata_sha256": "new",
        "dataset_publication_operation_sha256": c.p.digest(original),
    }
    assert c.dataset_publication_binding(new, tmp_path / "new.json", original) == c.p.digest(
        original
    )
    for key in (
        "runtime_image",
        "runtime_reference_checkpoint_sha256",
        "manifest_sha256",
        "kernel_source_sha256",
    ):
        with pytest.raises(ValueError):
            c.dataset_publication_binding(new | {key: "changed"}, tmp_path / "new.json", original)


def test_run_admits_bound_identity_correction_before_claim(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(c.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(c, "load_workflow", lambda *args: (object(), {}))
    monkeypatch.setattr(
        c,
        "validate_operation",
        lambda *args: ({"dataset_publication_operation_sha256": "old"}, {}, {}),
    )

    def binding(*args):
        raise RuntimeError("identity action reached original Dataset binding")

    monkeypatch.setattr(c, "dataset_publication_binding", binding)
    args = SimpleNamespace(
        projection=tmp_path,
        projection_manifest=tmp_path,
        projection_sha256="p",
        operation=tmp_path,
        action="push-after-uncertain-placeholder",
    )
    with pytest.raises(RuntimeError, match="reached original Dataset binding"):
        c.run(args)
