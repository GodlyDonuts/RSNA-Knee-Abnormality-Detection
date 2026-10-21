import ast
import importlib.util
from pathlib import Path

import pytest

P = Path(__file__).with_name("continue.py")
spec = importlib.util.spec_from_file_location("status_continuation", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def compile_receipt(role, mode):
    return dict(
        status=(
            "C100_final_offline_source_compiled_CPU_only"
            if role == "Newton"
            else "Colab_C100_final_offline_source_compiled_CPU_only"
        ),
        branch="C100" if role == "Newton" else "Colab-C100-seed20261021",
        mode="qualification" if mode == "qualification" else "competition",
        source_sha256="a" * 64,
        actual_trained_GPU_execution=False,
        new_Dataset_notebook_or_entry_mutations=0,
    )


@pytest.mark.parametrize("role", ["Newton", "Colab"])
@pytest.mark.parametrize("mode", ["qualification", "C"])
def test_actual_role_receipts(role, mode):
    owner = m.make_owner(role, Path("/tmp/not-created-test-journal"))
    base = owner.original() if role == "Newton" else owner.base()[1]
    assert base.phase_result("compile", mode, compile_receipt(role, mode)) == "done"
    assert base.phase_result("submit", "C", dict(state="submitted", submission_id=123)) == "done"
    with pytest.raises(AssertionError):
        base.phase_result(
            "submit", "C", dict(state="predictive_final_submitted", submission_id=123)
        )
    assert (
        base.phase_result(
            "compile", mode, dict(status="waiting_existing_canonical_claim_no_mutation")
        )
        == "wait"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "predictive_final_offline_source_compiled_CPU_only"),
        ("branch", "B"),
        ("mode", "wrong"),
        ("source_sha256", "g" * 64),
        ("actual_trained_GPU_execution", True),
        ("new_Dataset_notebook_or_entry_mutations", 1),
    ],
)
def test_reject_unexpected_compile_receipt(key, value):
    actual = compile_receipt("Newton", "qualification")
    actual[key] = value
    with pytest.raises(AssertionError):
        m.phase_reader(lambda *_: None, "Newton")("compile", "qualification", actual)


def test_only_remove_already_completed_newton_phases():
    p = m.HERE.parent / "predictive-C100-deployment-owner-20261021-v1/owner.py"
    changed = m.newton_body(m.source(m.PIN, p))
    ast.parse(changed)
    assert '            _admit_existing(self)\n            self.action("push")' in changed
    for phase in ("stage", "prepare", "dataset", "dataset-reconcile", "compile"):
        assert f'self.action("{phase}")' not in changed
    assert 'self.action("compile", "C")' in changed
    assert 'self.action("submit", "C")' in changed


def test_original_observer_boundary_anchor():
    owner = m.make_owner("Newton", Path("/tmp/not-created-test-journal"))
    p = m.HERE.parent / "predictive-final-deployment-owner-20261020-v1/observe.py.txt"
    body = m.source(m.PIN, p)
    changed = m.replace(
        body,
        "    return result\n\n\nif __name__",
        m.BOUNDARY + "    return result\n\n\nif __name__",
    )
    ast.parse(changed)
    assert owner.HERE.name == "predictive-fast-newton-deployment-20261021-v1"
    assert "mode=ro" in changed
    assert "no_prior_push_or_submission" in changed
