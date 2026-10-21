"""Actual staged final compiler with the executed Colab step972 lineage."""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_executed_recovery_survives_full_serving_capsule(monkeypatch, tmp_path):
    phase = load(
        "new_Colab_final_phase",
        HERE.parent / "predictive-fast-colab-deployment-20261021-v1/run_once.py",
    )
    fix = load(
        "actual_Colab_fixture", HERE.parent / "colab-C100-final-serving-20261021-v1/test_serving.py"
    )
    adapter, entry = fix.case.__wrapped__()
    fix.recovered2(entry)
    parent = "synthetic encoder source, no pixels or model weights"
    adapter._base.PARENT_SOURCE = fix.C.digest(parent)
    fix.rewrite_binding(entry, lambda b: b.update(parent_source_sha256=fix.C.digest(parent)))
    original_git = phase.git
    monkeypatch.setattr(
        phase,
        "git",
        lambda pin, path: path.read_text() if pin == "local-test" else original_git(pin, path),
    )
    files = phase.namespace()["controls"](
        "local-test",
        {
            "root": "/tmp/rsna-predictive-final-deployment-20261020-v1-Colab-C100-synthetic",
            "owning_node": "evc27",
            "Colab_evidence": {k: v for k, v in entry.items() if k != "license_bytes"},
        },
    )
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    compiler = load("actual_resilient_Colab_compiler", tmp_path / "serving/compile.py")
    compiler.adapter._base.LICENSE = adapter._base.LICENSE
    compiler.adapter._base.PARENT_SOURCE = adapter._base.PARENT_SOURCE
    monkeypatch.setattr(compiler.implementation.C, "Contract", lambda *_a, **_kw: compiler.adapter)
    dataset = entry["epoch_publications"][-1]["slug"]
    source, receipt = compiler.build(
        parent, [entry], mode="qualification", model_dataset_ref=dataset
    )
    ast.parse(source)
    assert 'precision="fp32"' in source and '"serving_resilience_version"' in source
    candidate = compiler.contract.candidate(**entry)
    assert candidate["execution_continuation"]["resume_update"] == 972
    q = fix.OLD.qualification(candidate, compiler.adapter._base)
    q.update(
        serving_worker_sha256=receipt["worker_sha256"],
        serving_contract_sha256=receipt["contract_sha256"],
        serving_helpers_sha256=receipt["serving_helpers_sha256"],
        model_dataset_ref=dataset,
        preparation_version="exact_selected_frames_and_batched_descriptors_v1",
        serving_resilience_version="exact_first_MRI_FP32_recovery_v1",
    )
    for m in q["models"][0]["measurements"]:
        m.update(
            original_context_bitwise_equal=True,
            context_equivalence_is_public_only=True,
            preparation_version=q["preparation_version"],
            original_render_comparison_seconds=10,
            render_seconds=5,
            decoder_route="original_exact",
            prediction_precision="fp16",
        )
    compiler.build(parent, [entry], mode="competition", model_dataset_ref=dataset, qualification=q)
    bad = copy.deepcopy(entry)
    bad["epoch_publications"][4]["status"] = "local_immutable_checkpoint_ready_publication_pending"
    with pytest.raises(ValueError):
        compiler.build(
            parent, [bad], mode="competition", model_dataset_ref=dataset, qualification=q
        )
