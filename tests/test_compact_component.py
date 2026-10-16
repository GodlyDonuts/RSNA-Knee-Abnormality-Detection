import json

import pytest

from rsna.competitive.checkpoint import file_sha256
from scripts.kaggle.compact_component import KIND, SOURCE, admit_component


def component(root):
    (root / "checkpoint.pt").write_bytes(b"synthetic immutable bytes; never a model")
    sha = file_sha256(root / "checkpoint.pt")
    sidecar = {
        "checkpoint_sha256": sha,
        "checkpoint_bytes": (root / "checkpoint.pt").stat().st_size,
        "epoch": 4,
        "optimizer_steps": 1124,
        "binding": {"diagnostic_only": False},
    }
    (root / "epoch.json").write_text(json.dumps(sidecar))
    (root / "reference.json").write_text(json.dumps({"checkpoint_sha256": sha, "epoch": 4}))
    (root / "provenance.json").write_text("{}")
    value = {
        "kind": KIND,
        "source_training_git": SOURCE,
        "diagnostic_only": False,
        "competition_eligible": True,
        "epoch": 4,
        "optimizer_steps": 1124,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in root.iterdir()
        },
    }
    return value


def save(root, value):
    path = root / "component-manifest.json"
    path.write_text(json.dumps(value))
    return file_sha256(path)


def test_exact_immutable_component_and_reference(tmp_path):
    value = component(tmp_path)
    digest = save(tmp_path, value)
    assert admit_component(tmp_path, digest)["epoch"] == 4
    (tmp_path / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        admit_component(tmp_path, digest)


@pytest.mark.parametrize(
    ("key", "bad"),
    [("epoch", 3), ("epoch", True), ("diagnostic_only", True), ("optimizer_steps", 1123)],
)
def test_diagnostic_or_unscheduled_scope_rejected(tmp_path, key, bad):
    value = component(tmp_path)
    value[key] = bad
    with pytest.raises(ValueError, match="scope"):
        admit_component(tmp_path, save(tmp_path, value))


def test_wrong_reference_is_rejected_even_if_rehashed(tmp_path):
    value = component(tmp_path)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"checkpoint_sha256": "wrong", "epoch": 4}))
    value["files"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    with pytest.raises(ValueError, match="binding"):
        admit_component(tmp_path, save(tmp_path, value))
