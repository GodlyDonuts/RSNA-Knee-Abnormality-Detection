"""Compiler-boundary checks; the full private source is admitted on its owner."""

import ast
import base64
from pathlib import Path

import pytest
import recovery


def fixture():
    worker = """from rsna.competitive.licensed_importance_train import fit
def main():
    parser.add_argument('role',choices=('uniform','influence'))
    original_data=None
    initial_identity=_initial_identity(model,device);
    result=fit(model,data,output,binding,Recipe(epochs=2,role=a.role),device=device)
"""
    files = {"original.py": "# synthetic old closure\n"}
    source = (
        "FILES="
        + repr(files)
        + "\nWORKER="
        + repr(base64.b64encode(worker.encode()).decode())
        + "\nfor role in ('uniform','influence'):\n    pass\n"
        + "roles={};identities=[];finals=[]\nfor role in ('uniform','influence'):\n    pass\n"
    )
    receipt = {
        "same_initial_complete_model_and_allrank_RNG_between_roles": True,
        "roles": {"influence": {"prediction_SHA": "a" * 64}},
    }
    return source, receipt, files


def test_unaccepted_original_source_refused():
    source, receipt, _ = fixture()
    with pytest.raises(ValueError, match="Exact actually accepted"):
        recovery.build(source, "", "", receipt)


def test_preserve_every_original_module_and_bind_real_sampler_worker(monkeypatch):
    source, receipt, old = fixture()
    monkeypatch.setattr(recovery, "ORIGINAL", recovery.sha(source))
    d = Path(__file__).parent
    built, proof = recovery.build(
        source, (d / "sampling.py").read_text(), (d / "fit.py").read_text(), receipt
    )
    compile(built, "synthetic_private_template", "exec")
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(built).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "WORKER")
    }
    assert all(values["FILES"][n] == text for n, text in old.items())
    assert set(values["FILES"]) == {*old, "sampling.py", "finding_fit.py"}
    worker = base64.b64decode(values["WORKER"]).decode()
    assert "components=finding_components,allocation=a.role" in worker
    assert "Recipe(epochs=2,role='influence')" in worker
    assert proof["original_numerical_modules_unchanged"] is True
    assert "same(current,prior)" in built and "'a" + "a" * 63 + "'" in built


@pytest.mark.parametrize("failure", ["missing_worker_marker", "ambiguous_phase_roles"])
def test_source_drift_refused_before_gpu(monkeypatch, failure):
    source, receipt, _ = fixture()
    if failure == "missing_worker_marker":
        tree = ast.parse(source)
        worker = ast.literal_eval(tree.body[1].value)
        worker = base64.b64decode(worker).decode().replace("Recipe(epochs=2,role=a.role)", "None")
        source = source.replace(
            ast.get_source_segment(source, tree.body[1]),
            "WORKER=" + repr(base64.b64encode(worker.encode()).decode()),
        )
    else:
        source += "\nfor role in ('uniform','influence'):\n    pass\n"
    monkeypatch.setattr(recovery, "ORIGINAL", recovery.sha(source))
    with pytest.raises(ValueError):
        recovery.build(source, "# stub", "# stub", receipt)
