"""Real cache proofs, private publication, and uncertain-commit reconciliation."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_cache_worker import build
from test_native_cache_worker import inputs as worker_inputs

from scripts.hf import publish_native_cache_shard as publisher


@pytest.fixture
def bundle(tmp_path):
    inputs = worker_inputs.__wrapped__(tmp_path)
    inputs[0]["selection_sha256"] = "a" * 64
    output = tmp_path / "built"
    report = build(inputs, output)
    return output, inputs[0], report["execution_provenance"]


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for offset in range(0, len(self.data), min(size, 4096)):
            yield self.data[offset : offset + min(size, 4096)]


class Api:
    def __init__(self, private=True, lose_reply=False, corrupt=False):
        self.private, self.lose_reply, self.corrupt = private, lose_reply, corrupt
        self.values = {}
        self.commits = 0
        self.sha = "b" * 40

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private, sha=self.sha)

    def get_paths_info(self, repo, *, paths, **kwargs):
        return [SimpleNamespace(path=name) for name in paths if name in self.values]

    def create_commit(self, repo, *, operations, **kwargs):
        self.commits += 1
        for operation in operations:
            source = operation["path_or_fileobj"]
            self.values[operation["path_in_repo"]] = (
                source.read_bytes() if isinstance(source, Path) else source
            )
        self.sha = "c" * 40
        if self.lose_reply:
            raise TimeoutError("commit succeeded but its response was lost")
        return SimpleNamespace(oid=self.sha)

    def get(self, url, **kwargs):
        assert kwargs["stream"] and kwargs["headers"]["Authorization"] == "Bearer hf_fixture"
        data = self.values[url]
        return Response(data + b"changed" if self.corrupt else data)


def publish(bundle, journal, api):
    return publisher.publish(
        *bundle, journal, api, "hf_fixture", lambda **k: k, lambda repo, name, **k: name, api.get
    )


def rewrite_report(root, mutate):
    path = root / "cache-verified.json"
    report = json.loads(path.read_text())
    mutate(report)
    path.write_text(json.dumps(report))


def test_complete_private_publication_and_no_duplicate_mutation(bundle, tmp_path):
    journal = tmp_path / "journal"
    api = Api()
    receipt = publish(bundle, journal, api)
    assert receipt["status"] == "private_native_shard_full_download_size_sha_verified"
    assert len(receipt["files"]) == 6 and api.commits == 1
    assert not receipt["model_training_or_accuracy_claim"]
    with pytest.raises(FileExistsError, match="reconcile"):
        publish(bundle, journal, api)
    assert api.commits == 1
    recorded = publisher.reconcile(journal, api, "hf_fixture", lambda r, n, **k: n, api.get)
    assert recorded["revision"] == receipt["revision"] and api.commits == 1
    assert all("hf_fixture" not in p.read_text() for p in journal.iterdir())


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_values_or_reports_read", True),
        ("full_cohort_built", True),
        ("source_revision", "changed"),
        ("max_reference_uint16_difference", 2),
        ("slices", 4),
        ("execution_provenance", {"job_id": "wrong"}),
    ],
)
def test_changed_worker_binding_is_rejected_before_intent(bundle, tmp_path, field, value):
    rewrite_report(bundle[0], lambda r: r.update({field: value}))
    journal = tmp_path / "journal"
    api = Api()
    with pytest.raises(ValueError, match="binding_changed"):
        publish(bundle, journal, api)
    assert not journal.exists() and api.commits == 0


def test_altered_series_proof_even_with_updated_proof_hash_is_rejected(bundle):
    root, shard, execution = bundle
    path = root / "series-proofs.jsonl"
    proof = json.loads(path.read_text())
    proof["study_uid"] = "another-study"
    path.write_text(json.dumps(proof) + "\n")
    rewrite_report(
        root,
        lambda r: r.update(
            {
                "source_proofs": {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
        ),
    )
    with pytest.raises(ValueError, match="identity_or_reference"):
        publisher.validate_bundle(root, shard, execution)


def test_archive_byte_corruption_or_symlink_is_rejected(bundle, tmp_path):
    root, shard, execution = bundle
    path = root / "cache-side-384.tar"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="artifact_bytes_changed"):
        publisher.validate_bundle(root, shard, execution)
    target = tmp_path / "archive"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="artifact_bytes_changed"):
        publisher.validate_bundle(root, shard, execution)


def test_remote_corruption_is_never_marked_verified(bundle, tmp_path):
    journal = tmp_path / "journal"
    api = Api(corrupt=True)
    with pytest.raises(ValueError, match="download_bytes_changed"):
        publish(bundle, journal, api)
    assert (journal / "intent.json").exists() and (journal / "commit.json").exists()
    assert not (journal / "verified.json").exists() and api.commits == 1


def test_lost_commit_reply_reconciles_full_bytes_without_another_upload(bundle, tmp_path):
    journal = tmp_path / "journal"
    api = Api(lose_reply=True)
    with pytest.raises(TimeoutError):
        publish(bundle, journal, api)
    assert not (journal / "commit.json").exists()
    with pytest.raises(FileExistsError, match="reconcile"):
        publish(bundle, journal, api)
    receipt = publisher.reconcile(journal, api, "hf_fixture", lambda r, n, **k: n, api.get)
    assert receipt["reconciled_readonly"] and receipt["external_mutations"] == 0
    assert api.commits == 1 and len(receipt["files"]) == 6


def test_public_repository_and_existing_paths_are_rejected_before_intent(bundle, tmp_path):
    api = Api(private=False)
    with pytest.raises(ValueError, match="private_repository"):
        publish(bundle, tmp_path / "public", api)
    assert api.commits == 0 and not (tmp_path / "public").exists()
    api = Api()
    publish(bundle, tmp_path / "initial-journal", api)
    with pytest.raises(FileExistsError, match="already exist"):
        publish(bundle, tmp_path / "repeat", api)
    assert api.commits == 1 and not (tmp_path / "repeat").exists()


def test_partial_or_absent_uncertain_publication_never_causes_retry(bundle, tmp_path):
    journal = tmp_path / "journal"
    api = Api(lose_reply=True)
    with pytest.raises(TimeoutError):
        publish(bundle, journal, api)
    original = copy.copy(api.values)
    api.values = {}
    status = publisher.reconcile(journal, api, "hf_fixture", lambda r, n, **k: n, api.get)
    assert status["automatic_retry"] is False and status["external_mutations"] == 0
    name = next(iter(original))
    api.values[name] = original[name]
    with pytest.raises(ValueError, match="partial_remote"):
        publisher.reconcile(journal, api, "hf_fixture", lambda r, n, **k: n, api.get)
    assert api.commits == 1 and not (journal / "verified.json").exists()
