import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n = load("epoch2_control_test", HERE / "resume.py")


def test_same_numerical_setup_loader_and_exact_epoch_resume():
    t = load(
        "original_training_derivation",
        HERE.parent / "colab-private-kaggle-handoff-20261021-v1/training.py",
    )
    old = load(
        "original_epoch1_derivation",
        HERE.parent / "colab-C100-epoch1-recovery-20261021-v1/resume.py",
    )
    original = t.build()
    derived = n.transform(original, old)

    def classes(source):
        return {
            x.name: ast.dump(x) for x in ast.walk(ast.parse(source)) if isinstance(x, ast.ClassDef)
        }

    assert classes(original) == classes(derived)
    before = original[original.index("    recipe = Recipe") : original.index("    binding = ")]
    assert before in derived
    assert "resume=PRIOR/'checkpoints/epoch-002.pt'" in derived
    assert "'all_epochs_private_verified':False" in derived
    assert "publications = {1: done, 2: second}" in derived
    with pytest.raises(AssertionError):
        n.transform(original + "\n", old)


def test_local_preservation_never_creates_remote_proof_and_is_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(n.shutil, "disk_usage", lambda _: SimpleNamespace(free=40 * 2**30))
    p = tmp_path / "checkpoint.pt"
    p.write_bytes(b"original bytes")
    store = n.LocalStore(None, {"seed": 20261021, "slugs": {"epoch-003": "private/test"}})
    result = store.publish("epoch-003", {"full-state.private.pt": p}, tmp_path)
    assert result["new_account_mutations"] == 0 and not result["remote_preservation_verified"]
    assert json.loads((tmp_path / "local-preserved.json").read_text()) == result
    assert (
        not (tmp_path / "verified.json").exists() and not (tmp_path / "Kaggle-intent.json").exists()
    )
    with pytest.raises(FileExistsError):
        store.publish("epoch-003", {"full-state.private.pt": p}, tmp_path)
    with pytest.raises(AssertionError):
        n.LocalStore("not-used", {"seed": 20261021})
