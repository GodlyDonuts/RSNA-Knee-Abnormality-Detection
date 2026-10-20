"""Regression for the actual malformed digest; source/inference stays unchanged."""

import ast
import hashlib
from pathlib import Path

import pytest
from test_S6_twoT4_serving import D, load

fix = load("serving_runtime64")
CORE = (D / "serving_cloud.py.txt").read_text()


def canonical_runtime():
    tree = ast.parse(
        Path("initiative/research/S6-final-Kaggle-execution-20261019-v1/policy.py.txt").read_text()
    )
    return next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DOCKER" for t in n.targets)
    )


def test_runtime_is_the_existing_working64_hex_canonical_image():
    assert fix.DOCKER == canonical_runtime() and len(fix.DOCKER.rsplit(":", 1)[1]) == 64
    # This explicitly catches the previous61-character value, not a comparison to itself.
    assert (
        fix.DOCKER
        != "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f2f6b083ad2b2f"
    )


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_actual_policy_composition_changes_only_provenance_comment(tmp_path, mode):
    ns = fix.namespace(CORE, "a" * 40)
    f = tmp_path / "original-control"
    (f / "code").mkdir(parents=True)
    (f / "code/policy.py").write_text((D / "serving_policy.py.txt").read_text())
    ns["F"] = f
    policy = ns["load"]("policy")
    original = load("serving_policy")
    # Minimal executable public shape; no pixels, model or clinical claim.
    source = "def find_test_root(): return '/private'\nif __name__ == '__main__':\n    result='actual MRI path'\n"
    corrected = policy.public_bounded_source(source, mode)
    expected = original.public_bounded_source(source, mode)
    assert ast.dump(ast.parse(corrected)) == ast.dump(ast.parse(expected))
    training = {
        "epoch": 1,
        "version": 1,
        "status": "COMPLETE",
        "slug": "godlydonuts/synthetic-exact-training",
    }
    request = policy.request(1, mode, corrected, training)
    assert (
        request["dockerImage"] == canonical_runtime()
        and request["dockerImagePinningType"] == "original"
    )
    assert request["sessionTimeoutSeconds"] == (1800 if mode == "qualification" else 32400)
    assert not request["enableInternet"] and request["isPrivate"]
    assert request["slug"].endswith(hashlib.sha256(corrected.encode()).hexdigest()[:12])
    assert request["kernelDataSources"][1] == training["slug"]


def test_unknown_core_or_malformed_runtime_cannot_reach_a_remote_call(monkeypatch):
    with pytest.raises(ValueError):
        fix.namespace(CORE + "# unaccepted\n", "a" * 40)
    monkeypatch.setattr(fix, "DOCKER", "gcr.io/kaggle-private-byod/python@sha256:" + "a" * 61)
    with pytest.raises(ValueError):
        fix.namespace(CORE, "a" * 40)
