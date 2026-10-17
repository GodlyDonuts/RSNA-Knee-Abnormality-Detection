import ast
import base64
import hashlib

import pytest

from scripts.kaggle import licensed_importance_recovery as sampled
from scripts.kaggle import licensed_parent_recovery as parent


def inputs(monkeypatch):
    monkeypatch.setattr(
        sampled, "PERMISSION_SHA", hashlib.sha256(b"synthetic permission").hexdigest()
    )
    source = "def build_backbone():pass\nclass RaptorClassifier:pass\ndef load_model():pass\n"
    monkeypatch.setattr(parent, "PARENT_SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    monkeypatch.setattr(
        parent,
        "pixel_build",
        lambda p, c: (
            "def _pixel_cache_main():pass\nif __name__=='__main__':\n    _pixel_cache_main()\n",
            {"CPU_only": True},
        ),
    )
    files = {
        n: "# Synthetic compiler fixture, not MRI/GPU acceptance\n"
        for n in (*sampled.MODULES, sampled.WHOLE_MODULE, *sampled.EXTRA_MODULES)
    }
    targets = {
        "uids": [f"synthetic-private-{i}" for i in range(32)],
        "labels": [[0.5] * 12 for _ in range(32)],
        "masks": [[True] * 12 for _ in range(32)],
        "eligible_fitting_only": True,
    }
    return source, "synthetic codec", files, targets


def test_recovery_has_one_bound_numerical_closure_and_no_cleartext_private_rows(monkeypatch):
    args = inputs(monkeypatch)
    source, metadata = sampled.build(*args, permission="synthetic permission")
    compile(source, "synthetic-compiler-only", "exec")
    assert metadata["competition_entries"] == 0
    assert metadata["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    literals = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "WORKER", "TARGETS")
    }
    assert literals["FILES"] == args[2]
    worker = base64.b64decode(literals["WORKER"], validate=True).decode()
    compile(worker, "synthetic-private-worker-only", "exec")
    assert metadata["worker_sha256"] == hashlib.sha256(worker.encode()).hexdigest()
    assert "Recipe(epochs=2,role=a.role)" in worker
    assert "Recipe(epochs=2,windows=94)" not in worker
    assert "importance-recovery" in worker and "parent-recovery" not in worker
    assert args[3]["uids"][0] not in source
    assert "same(identities[0],identities[1])" in source
    assert "same(full,resumed)" in source


def test_unadmitted_records_or_partial_closure_never_produce_source(monkeypatch):
    source, codec, files, target = inputs(monkeypatch)
    incomplete = dict(files)
    incomplete.pop(sampled.EXTRA_MODULES[0])
    with pytest.raises(ValueError):
        sampled.build(source, codec, incomplete, target, permission="synthetic permission")
    target["eligible_fitting_only"] = False
    with pytest.raises(ValueError):
        sampled.build(source, codec, files, target, permission="synthetic permission")
