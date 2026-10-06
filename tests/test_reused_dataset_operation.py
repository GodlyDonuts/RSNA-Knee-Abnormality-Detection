"""A real previously byte-verified Dataset needs no new local weight copies."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location("reuse_workflow", "scripts/kaggle/workflow.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def original(tmp_path):
    audit = Path("initiative/audits/kaggle-prefetch-control-recovery-v1")
    root = tmp_path / "candidate"
    (root / "dataset").mkdir(parents=True)
    (root / "kernel").mkdir()
    op = json.loads((audit / "new-operation.json").read_text())
    (root / "dataset/bundle-manifest.json").write_bytes(
        (audit / "original-bundle-manifest.json").read_bytes()
    )
    (root / "dataset/dataset-metadata.json").write_bytes(
        (audit / "original-dataset-metadata.json").read_bytes()
    )
    (root / "original-operation.json").write_bytes((audit / "original-operation.json").read_bytes())
    (root / "original-dataset-verification.json").write_bytes(
        Path(
            "initiative/audits/kaggle-effusion-comparison-v1/seed1-control/reconcile-dataset-result.json"
        ).read_bytes()
    )
    (root / "kernel/run.py").write_text("new_inference_entry\n")
    (root / "kernel/kernel-metadata.json").write_text("{}\n")
    w = module()
    op.update(
        kernel_source_sha256=w.digest(root / "kernel/run.py"),
        kernel_metadata_sha256=w.digest(root / "kernel/kernel-metadata.json"),
    )
    (root / "operation.json").write_text(json.dumps(op))
    return w, root, op


def test_original_complete_56_file_byte_proof_admits_metadata_only(tmp_path):
    w, root, op = original(tmp_path)
    assert w.load_operation(root / "operation.json") == op
    assert not (root / "dataset/checkpoint.pt").exists()
    assert not list((root / "dataset").glob("*.safetensors"))


@pytest.mark.parametrize(
    "field,value",
    [("downloaded_all_bytes", False), ("private", False), ("version", 2), ("id", 123)],
)
def test_wrong_original_proof_scope_is_rejected_even_with_new_digest(tmp_path, field, value):
    w, root, op = original(tmp_path)
    path = root / "original-dataset-verification.json"
    proof = json.loads(path.read_text())
    proof[field] = value
    path.write_text(json.dumps(proof))
    op["reused_dataset_verification_sha256"] = w.digest(path)
    (root / "operation.json").write_text(json.dumps(op))
    with pytest.raises(ValueError, match="immutable Dataset"):
        w.load_operation(root / "operation.json")


def test_changed_checkpoint_member_cannot_reuse_original_manifest(tmp_path):
    w, root, op = original(tmp_path)
    path = root / "original-dataset-verification.json"
    proof = json.loads(path.read_text())
    proof["files"]["checkpoint.pt"]["sha256"] = "a" * 64
    path.write_text(json.dumps(proof))
    op["reused_dataset_verification_sha256"] = w.digest(path)
    (root / "operation.json").write_text(json.dumps(op))
    with pytest.raises(ValueError, match="immutable Dataset"):
        w.load_operation(root / "operation.json")
