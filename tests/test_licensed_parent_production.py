import ast
import base64
import copy

import pytest

from scripts.kaggle import licensed_parent_production as production


@pytest.fixture
def synthetic_compiler(monkeypatch):
    worker = "def _whole_model(base):return base\nclass PixelCases:pass\ndef main():raise RuntimeError('original diagnostic main')\nif __name__=='__main__':main()\n"
    source = "WORKER=" + repr(base64.b64encode(worker.encode()).decode()) + "\n"
    monkeypatch.setattr(production, "RECOVERY_SHA", production.digest(source))
    monkeypatch.setattr(production, "WORKER_SHA", production.digest(worker))
    monkeypatch.setattr(production, "validate_proofs", lambda x: None)
    files = {name: "# synthetic compile-only fixture\n" for name in production.ACCEPTED_MODULES}
    monkeypatch.setattr(
        production, "ACCEPTED_MODULES", {n: production.digest(s) for n, s in files.items()}
    )
    files[production.POPULATION_MODULE] = "# synthetic population compile-only fixture\n"
    return source, files, [], "a" * 40


def value(source, name):
    return next(
        ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    )


def prior(source, binding, segment):
    return {
        "status": "actual_whole94_full_parent_epoch_session_complete",
        "segment": segment,
        "completed": 486 * segment,
        "worker_sha256": binding["worker_sha256"],
        "binding": binding["binding"],
        "checkpoint": {"bytes": 123, "sha256": "1" * 64},
        "run": {"bytes": 234, "sha256": "2" * 64},
    }


def test_three_sessions_have_identical_worker_recipe_and_population_binding(synthetic_compiler):
    inputs = synthetic_compiler
    first, b1 = production.build(*inputs, 1)
    previous = prior(first, b1, 1)
    second, b2 = production.build(*inputs, 2, previous=previous)
    third, b3 = production.build(*inputs, 3, previous=prior(second, b2, 2))
    assert b1["worker_sha256"] == b2["worker_sha256"] == b3["worker_sha256"]
    assert b1["binding"] == b2["binding"] == b3["binding"]
    for i, source in enumerate((first, second, third), 1):
        compile(source, "synthetic-no-GPU-acceptance", "exec")
        assert value(source, "SEGMENT") == i
        worker = base64.b64decode(value(source, "WORKER")).decode()
        assert "Recipe(epochs=3,windows=94)" in worker and "len(data)==3887" in worker
        assert "original diagnostic main" not in worker
        assert "def _whole_model(base):return base" in worker
        assert "--resume" in source and "PREVIOUS[key]['sha256']" in source
        assert "timeout=6.75*3600" in source
    assert value(first, "PREVIOUS") is None and value(second, "PREVIOUS") == previous


@pytest.mark.parametrize(
    "corruption",
    ["missing", "wrong_cursor", "worker", "binding", "missing_AdamW_bytes", "first_with_resume"],
)
def test_session_continuation_requires_original_complete_state_identity(
    synthetic_compiler, corruption
):
    inputs = synthetic_compiler
    source, binding = production.build(*inputs, 1)
    previous = prior(source, binding, 1)
    if corruption == "missing":
        previous = None
    elif corruption == "wrong_cursor":
        previous["completed"] -= 1
    elif corruption == "worker":
        previous["worker_sha256"] = "0" * 64
    elif corruption == "binding":
        previous["binding"] = copy.deepcopy(previous["binding"])
        previous["binding"]["source_git"] = "b" * 40
    elif corruption == "missing_AdamW_bytes":
        previous.pop("checkpoint")
    with pytest.raises(ValueError):
        production.build(*inputs, 1 if corruption == "first_with_resume" else 2, previous=previous)


def test_changed_actual_T4_numerics_are_refused(synthetic_compiler):
    source, files, proofs, git = synthetic_compiler
    files[next(iter(production.ACCEPTED_MODULES))] += "changed"
    with pytest.raises(ValueError, match="numerical modules"):
        production.build(source, files, proofs, git, 1)


@pytest.mark.parametrize("segment", [0, 4, True])
def test_production_session_cannot_change_fixed_three_epoch_design(synthetic_compiler, segment):
    with pytest.raises(ValueError):
        production.build(*synthetic_compiler, segment)


def test_json_proof_key_reordering_replays_the_exact_actual_worker(synthetic_compiler):
    accepted, files, _, git = synthetic_compiler
    proofs = [{"shard": 0, "files": {"bytes": 123, "sha256": "a" * 64}}]
    first, binding = production.build(accepted, files, proofs, git, 1)
    original_worker = base64.b64decode(value(first, "WORKER")).decode()
    previous = prior(first, binding, 1)
    reloaded = [{"files": {"sha256": "a" * 64, "bytes": 123}, "shard": 0}]
    # Real coordinator writes canonical JSON before the later session reload.
    with pytest.raises(ValueError):
        production.build(accepted, files, reloaded, git, 2, previous=previous)
    second, next_binding = production.build(
        accepted, files, reloaded, git, 2, previous=previous, previous_worker=original_worker
    )
    assert base64.b64decode(value(second, "WORKER")).decode() == original_worker
    assert binding["binding"] == next_binding["binding"]
    assert next_binding["worker_sha256"] == previous["worker_sha256"]
    assert value(second, "PREVIOUS") == previous


@pytest.mark.parametrize("change", ["proof_value", "numerical_code", "comment", "hash", "bool_int"])
def test_worker_replay_cannot_bypass_any_other_source_or_proof_change(change):
    original = "_PRODUCTION_PROOFS=[{'first':True,'second':2}]\nrate=3e-6\n"
    candidate = "_PRODUCTION_PROOFS=[{'second':2,'first':True}]\nrate=3e-6\n"
    sha = production.digest(original)
    if change == "proof_value":
        candidate = candidate.replace("'second':2", "'second':3")
    elif change == "numerical_code":
        candidate = candidate.replace("rate=3e-6", "rate=3e-5")
    elif change == "comment":
        candidate += "# even a harmless unrelated source change is refused\n"
    elif change == "hash":
        sha = "0" * 64
    else:
        candidate = candidate.replace("'first':True", "'first':1")
    with pytest.raises(ValueError):
        production.replay_previous_worker(candidate, original, sha)


def test_worker_replay_is_not_available_for_initialization(synthetic_compiler):
    source, _ = production.build(*synthetic_compiler, 1)
    worker = base64.b64decode(value(source, "WORKER")).decode()
    with pytest.raises(ValueError):
        production.build(*synthetic_compiler, 1, previous_worker=worker)
