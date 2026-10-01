"""Real source reconstruction/private receipts and strict native training restoration."""

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import test_native_cache_publisher as publications
import test_native_cache_worker as workers

from scripts.newton import assemble_native_cache as assembly
from scripts.newton import run_native_cache_assembly as runner


@pytest.fixture
def bound(tmp_path):
    (root, shard, _), _ = publications.bundle.__wrapped__(tmp_path), None
    report = json.loads((root / "cache-verified.json").read_text())
    execution = {
        "job_id": "123",
        "cpus": 1,
        "gpus": 0,
        "worker_sha256": assembly.WORKERS[0],
        "plan_manifest_sha256": assembly.PLAN_SHA,
        "shard_file_sha256": "b" * 64,
    }
    report["execution_provenance"] = execution
    (root / "cache-verified.json").write_text(json.dumps(report))
    bundle = root, shard, execution
    receipt = publications.publish(bundle, tmp_path / "journal", publications.Api())
    proofs = [json.loads(line) for line in (root / "series-proofs.jsonl").read_text().splitlines()]
    return root, shard, receipt, report, proofs


def bind(bound):
    return assembly.bind_shard(*bound[1:])


def test_real_native_dicom_cache_can_be_bound_restored_and_used(bound, tmp_path):
    root, shard, receipt, _, _ = bound
    record = bind(bound)
    output = tmp_path / "assembly"
    count = assembly.restore_archive(
        root / "cache-side-384.tar",
        receipt["files"]["cache-side-384.tar"],
        record["members"],
        output,
    )
    assert count == 2
    assert assembly.check_metadata(output, shard["series"]) == 1
    loaders = workers.worker.check_loaders(
        shard, {256: root / "side-256/cache", 384: output / "cache"}
    )
    assert loaders["384"]["studies"] == 1
    assert loaders["384"]["valid_sampled_slices"] == 3
    path = output / "cache" / shard["series"][0]["cache_path"]
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["volume"].shape == (3, 384, 384)
        assert arrays["volume"].dtype == np.uint16
    with pytest.raises(FileExistsError):
        assembly.restore_archive(
            root / "cache-side-384.tar",
            receipt["files"]["cache-side-384.tar"],
            record["members"],
            output,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("repo_id", "someone/public"),
        ("revision", "main"),
        ("shard", 7),
        ("status", "commit_returned"),
        ("model_training_or_accuracy_claim", True),
    ],
)
def test_unverified_or_different_private_receipt_cannot_supply_training_cache(bound, field, value):
    changed = copy.deepcopy(bound)
    changed[2][field] = value
    with pytest.raises(ValueError, match="complete_exact_private"):
        bind(changed)


@pytest.mark.parametrize(
    "field,value", [("gpus", 1), ("cpus", 8), ("worker_sha256", "c" * 64), ("job_id", "fixture")]
)
def test_unbound_worker_execution_refused(bound, field, value):
    changed = copy.deepcopy(bound)
    changed[2]["execution_provenance"][field] = value
    changed[3]["execution_provenance"][field] = value
    with pytest.raises(ValueError, match="accepted_source_execution"):
        bind(changed)


def test_loader_and_source_membership_scope_are_not_downgraded(bound):
    changed = copy.deepcopy(bound)
    changed[3]["loaders"]["384"]["labels_read"] = True
    with pytest.raises(ValueError, match="actual_source_loader"):
        bind(changed)
    changed = copy.deepcopy(bound)
    changed[4][0]["study_uid"] = "another-study"
    with pytest.raises(ValueError, match="accepted_source_series_identity"):
        bind(changed)
    changed = copy.deepcopy(bound)
    changed[4].clear()
    with pytest.raises(ValueError, match="exact_source_proof_series"):
        bind(changed)


def tar_bytes(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as target:
        for name, kind, data in members:
            header = tarfile.TarInfo(name)
            if kind == "link":
                header.type = tarfile.SYMTYPE
                header.linkname = "/outside"
                target.addfile(header)
            else:
                header.size = len(data)
                target.addfile(header, io.BytesIO(data))
    return stream.getvalue()


def record(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize(
    "kind", ["traversal", "symlink", "duplicate", "missing", "payload_corruption"]
)
def test_even_hash_bound_archives_cannot_smuggle_invalid_members(tmp_path, kind):
    name = "cache/train_series/1.2/1.2.3.npz"
    members = {name: record(b"safe")}
    values = [(name, "file", b"safe")]
    if kind == "traversal":
        values = [("cache/train_series/../../outside", "file", b"safe")]
    elif kind == "symlink":
        values = [(name, "link", b"")]
    elif kind == "duplicate":
        values *= 2
    elif kind == "missing":
        members["cache/train_series/1.2/1.2.4.npz"] = record(b"missing")
    elif kind == "payload_corruption":
        values = [(name, "file", b"evil")]
    data = tar_bytes(values)
    path = tmp_path / "changed.tar"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        assembly.restore_archive(path, record(data), members, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


def test_download_refuses_truncation_and_changed_bytes_before_extraction(bound, tmp_path):
    root, _, receipt, _, _ = bound
    data = (root / "cache-side-384.tar").read_bytes()
    expected = receipt["files"]["cache-side-384.tar"]
    for n, candidate in enumerate((data[:-1], b"changed" + data[7:], data + b"oversized")):

        def get(*args, **kwargs):
            assert kwargs["stream"] and kwargs["headers"]["Authorization"] == "Bearer hf_fixture"
            return publications.Response(candidate)

        with pytest.raises(ValueError):
            assembly.download("private-url", "hf_fixture", tmp_path / f"bad-{n}.tar", expected, get)


def test_existing_attempt_refused_without_any_download(tmp_path):
    root = tmp_path / "keep"
    root.mkdir()
    with pytest.raises(FileExistsError, match="readonly_review"):
        assembly.restore({}, root, "hf_fixture", None, None, None)


def test_incomplete_real_cohort_plan_cannot_freeze_registry(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = (
        root
        / "initiative/experiments/competitive-v1/native-detail-v1/full-cache-shards384-v1/manifest.json"
    )
    assert assembly.digest(manifest) == assembly.PLAN_SHA
    with pytest.raises(FileNotFoundError):
        assembly.freeze_registry(manifest, tmp_path / "first", tmp_path / "incomplete-batch")


def test_both_actual_lane_jobs_must_be_terminal_successes():
    valid = "866426|COMPLETED|0:0\n866426.batch|COMPLETED|0:0\n866427|COMPLETED|0:0\n"
    runner.require_terminal_lanes(valid)
    for changed in (
        valid.replace("866427|COMPLETED", "866427|RUNNING"),
        valid.replace("866427|COMPLETED|0:0", "866427|FAILED|2:0"),
        valid.replace("866427|COMPLETED|0:0", "866428|COMPLETED|0:0"),
    ):
        with pytest.raises(ValueError, match="both_exact_source_lanes"):
            runner.require_terminal_lanes(changed)
