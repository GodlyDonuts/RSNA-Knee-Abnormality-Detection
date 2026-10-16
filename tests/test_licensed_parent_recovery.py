import hashlib

import pytest

from scripts.kaggle import licensed_parent_recovery as recovery


def test_refuses_another_source_before_compiling_any_private_payload():
    with pytest.raises(ValueError, match="Exact scored"):
        recovery.build("unbound", "unbound", {}, {})


def sources(monkeypatch):
    # Synthetic compiler fixture; never an engineering or image-accuracy pass.
    parent = "def build_backbone():pass\nclass RaptorClassifier:pass\ndef load_model():pass\n"
    monkeypatch.setattr(
        recovery, "PARENT_SOURCE_SHA256", hashlib.sha256(parent.encode()).hexdigest()
    )
    monkeypatch.setattr(
        recovery,
        "pixel_build",
        lambda p, c: (
            "def _pixel_cache_main():pass\nif __name__=='__main__':\n    _pixel_cache_main()\n",
            {"CPU_only": True},
        ),
    )
    files = {name: "# Synthetic own-module compile fixture\n" for name in recovery.MODULES}
    targets = {
        "uids": [f"synthetic-{i}" for i in range(32)],
        "labels": [[0.5] * 12 for _ in range(32)],
        "masks": [[True] * 12 for _ in range(32)],
        "eligible_fitting_only": True,
    }
    return parent, files, targets


def test_source_closure_preserves_full_stop_resume_and_two_disjoint_engineering_sets(monkeypatch):
    parent, files, targets = sources(monkeypatch)
    source, binding = recovery.build(parent, "synthetic", files, targets)
    compile(source, "synthetic-compiler-only", "exec")
    assert binding["fitting_studies"] == binding["disjoint_engineering_studies"] == 16
    assert binding["GPU_world_size"] == 2 and binding["competition_entry"] is False
    assert binding["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert "for phase in ('full','stop','resume')" in source
    assert "same(full,resumed)" in source and "prediction_bytes_equal" in source
    assert "--nproc_per_node=2" in source and "timeout=900" in source
    assert "submission.csv" not in source
    assert targets["uids"][0] not in source


@pytest.mark.parametrize("bad", ["missing_module", "only31", "protected"])
def test_incomplete_or_unadmitted_private_payload_is_rejected(monkeypatch, bad):
    parent, files, targets = sources(monkeypatch)
    if bad == "missing_module":
        files.pop(recovery.MODULES[0])
    elif bad == "only31":
        targets["uids"].pop()
    else:
        targets["eligible_fitting_only"] = False
    with pytest.raises(ValueError):
        recovery.build(parent, "synthetic", files, targets)
