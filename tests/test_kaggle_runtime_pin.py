"""Pin a proven runtime without reuploading weights or losing a failed public attempt."""

import copy
import json

import pytest
from test_kaggle_coordinator_binding import c
from test_kaggle_submission import FakeRemote, entry, workflow

IMAGE = "gcr.io/kaggle-private-byod/python@sha256:" + "d" * 64


def operation():
    return {
        "checkpoint_sha256": "a" * 64,
        "kernel_slug": "owner/old",
        "kernel_metadata_sha256": "c" * 64,
        "dataset_slug": "owner/data",
        "manifest_sha256": "b" * 64,
        "kernel_source_sha256": "s" * 64,
        "shared_wheelhouse": {"id": 1},
    }


def corrected(tmp_path):
    old = operation()
    original = tmp_path / "original.json"
    original.write_text(json.dumps(old))
    new = old | {
        "kernel_slug": "owner/pinned",
        "kernel_metadata_sha256": "e" * 64,
        "runtime_image": IMAGE,
        "runtime_reference_checkpoint_sha256": "f" * 64,
        "dataset_publication_operation_sha256": c.p.digest(original),
    }
    return old, original, new


def test_verified_dataset_reuse_binds_every_original_non_kernel_field(tmp_path):
    _, original, new = corrected(tmp_path)
    assert c.dataset_publication_binding(new, tmp_path / "new.json", original) == c.p.digest(
        original
    )
    for key in [
        "dataset_slug",
        "checkpoint_sha256",
        "manifest_sha256",
        "kernel_source_sha256",
        "shared_wheelhouse",
    ]:
        changed = new | {key: "changed"}
        with pytest.raises(ValueError, match="Dataset or inference"):
            c.dataset_publication_binding(changed, tmp_path / "new.json", original)
    original.write_text("{}")
    with pytest.raises(ValueError, match="proof is missing or changed"):
        c.dataset_publication_binding(new, tmp_path / "new.json", original)


@pytest.mark.parametrize("mismatch", [None, "unbound", "latest", "arbitrary"])
def test_runtime_request_requires_consistent_immutable_approved_image(tmp_path, mismatch):
    d = tmp_path / "kernel"
    d.mkdir()
    (d / "run.py").write_text("source")
    meta = {
        "id": "owner/pinned",
        "title": "pinned",
        "is_private": True,
        "enable_internet": False,
        "dataset_sources": ["owner/data"],
        "competition_sources": [workflow.COMPETITION],
        "docker_image": IMAGE,
        "docker_image_pinning_type": "original",
    }
    op = {"kernel_slug": "owner/pinned", "dataset_slug": "owner/data", "runtime_image": IMAGE}
    if mismatch == "unbound":
        op["runtime_image"] = "another"
    if mismatch == "latest":
        meta["docker_image_pinning_type"] = "latest"
    if mismatch == "arbitrary":
        meta["docker_image"] = op["runtime_image"] = "untrusted/image:latest"
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    if mismatch is None:
        result = workflow.kernel_request(tmp_path / "operation.json", op)
        assert result["dockerImage"] == IMAGE and result["dockerImagePinningType"] == "original"
    else:
        with pytest.raises(ValueError, match="runtime pin"):
            workflow.kernel_request(tmp_path / "operation.json", op)


class Remote:
    username = "owner"

    def kernel(self, method, slug, version):
        return {"metadata": {"dockerImage": IMAGE}}

    def output_json(self, *args):
        return {
            "status": "offline_inference_complete",
            "checkpoint_sha256": "f" * 64,
            "bundle_manifest_sha256": "b" * 64,
        }


@pytest.mark.parametrize("failure", [None, "submitted", "running", "reference", "image", "history"])
def test_only_terminal_unsubmitted_public_error_can_be_preserved_for_runtime_pin(
    tmp_path, monkeypatch, failure
):
    old, original, new = corrected(tmp_path)
    remote = Remote()
    monkeypatch.setattr(
        workflow,
        "check_kernel",
        lambda r, e, v: {
            "status": "RUNNING"
            if failure == "running"
            else (
                "ERROR"
                if e["checkpoint_sha256"] == "a" * 64
                else ("ERROR" if failure == "reference" else "COMPLETE")
            )
        },
    )
    with workflow.ledger(tmp_path / "ledger.sqlite3", "owner") as db:
        entry = {
            "checkpoint_sha256": "a" * 64,
            "state": "pushed",
            "operation": old,
            "operation_sha256": c.p.digest(original),
            "kernel_version": 1,
            "dataset": {"id": 12, "version": 1},
        }
        if failure == "submitted":
            entry["submission_id"] = 123
        if failure == "history":
            entry["superseded_operations"] = [{"operation_sha256": c.p.digest(original)}]
        if failure == "image":
            new["runtime_image"] = "changed"
        workflow.save_entry(db, entry, "fixture")
        reference = {
            "checkpoint_sha256": "f" * 64,
            "state": "submitted",
            "kernel_version": 1,
            "operation": old | {"checkpoint_sha256": "f" * 64, "kernel_slug": "owner/reference"},
        }
        workflow.save_entry(db, reference, "reference fixture")
        before = copy.deepcopy(entry)
        if failure is not None:
            with pytest.raises(ValueError):
                c.supersede_failed_public_runtime(workflow, remote, db, new, c.p.digest(original))
            db.rollback()
            assert workflow.read_entry(db, "a" * 64) == before
        else:
            c.supersede_failed_public_runtime(workflow, remote, db, new, c.p.digest(original))
            result = workflow.read_entry(db, "a" * 64)
            assert result["state"] == "queued" and "operation" not in result
            history = result["superseded_operations"][0]
            assert (
                history["operation"] == old
                and history["state"] == "failed_public_runtime_unsubmitted"
            )


def test_exact_public_acceptance_rejects_runtime_image_drift():
    remote = FakeRemote()
    candidate = entry()
    candidate["operation"]["runtime_image"] = IMAGE
    original = remote.kernel
    observed = [IMAGE]

    def pinned(method, slug, version):
        result = original(method, slug, version)
        if method == "GetKernel":
            result["metadata"]["dockerImage"] = observed[0]
        return result

    remote.kernel = pinned
    assert workflow.check_kernel(remote, candidate, 1)["status"] == "COMPLETE"
    observed[0] = "changed"
    with pytest.raises(ValueError, match="exact prepared"):
        workflow.check_kernel(remote, candidate, 1)
