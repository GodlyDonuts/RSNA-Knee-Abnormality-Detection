import ast
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load("four_core_serving_test", HERE / "serving.py")
F = load("original_context_fixtures", HERE.parents[2] / "tests/test_predictive_descriptor_cache.py")


def test_real_context_comparator_allows_only_image_rounding():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        reference = F.assemble(
            F.cache.cache_descriptors_once(F.synthetic_groups()), side=32, context_cap=7
        )
        actual = copy.deepcopy(reference)
        actual[0]["images"][0, 0, 0, 0] += 1.1920928955078125e-7
        result = S.compare_contexts(reference, actual)
        assert result["thread1_reference_context_bitwise_equal"] is False
        assert result["thread1_reference_other_fields_exact"] is True
        for field, value in (("images", 1e-4), ("images", float("nan")), ("targets", 1e-7)):
            bad = copy.deepcopy(reference)
            bad[0][field].reshape(-1)[0] = value
            with pytest.raises(AssertionError):
                S.compare_contexts(reference, bad)
        bad = copy.deepcopy(reference)
        bad[0]["supports"].log_prior.reshape(-1)[0] += 1e-5
        with pytest.raises(AssertionError):
            S.compare_contexts(reference, bad)
    finally:
        torch.set_num_threads(previous)


def test_generated_full_capsule_requires_its_own_trained_comparison(monkeypatch, tmp_path):
    original = load(
        "resilience_fixture",
        HERE.parent / "predictive-inference-resilience-20261021-v1/test_capsule.py",
    )
    _, raw, entry, _, _ = original.F.capsule.__wrapped__(monkeypatch, tmp_path)
    resilience = load(
        "resilience_builder", HERE.parent / "predictive-inference-resilience-20261021-v1/serving.py"
    )
    files = S.capsule(resilience.capsule(original.U.capsule(raw)))
    assert {n for n in raw if raw[n] != files[n]} == {"serving/compile.py"}
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    facade = load("four_core_full_compiler", tmp_path / "serving/compile.py")
    fix = original.F.FIX
    facade.adapter._base.LICENSE = fix.C.digest(entry["license_bytes"])
    parent = "synthetic original encoder, no weights or clinical data"
    facade.adapter._base.PARENT_SOURCE = fix.C.digest(parent)
    fix.rewrite(entry, lambda p: p.update(parent_source_sha256=fix.C.digest(parent)))
    entry["full_terminal"]["parent_source_sha256"] = fix.C.digest(parent)
    facade.EVIDENCE = {k: entry[k] for k in original.F.CTL.EXTRA_KEYS}
    monkeypatch.setattr(facade.implementation.C, "Contract", lambda *_a, **_kw: facade.adapter)
    export = {k: v for k, v in entry.items() if k not in original.F.CTL.EXTRA_KEYS}
    source, receipt = facade.build(
        parent, [export], mode="qualification", model_dataset_ref="godlydonuts/synthetic-c100"
    )
    ast.parse(source)
    assert 'torch.set_num_threads(1 if BINDING["mode"] == "qualification" else 2)' in source
    assert (
        'reference = predict(models[device], original_prepared[index][0], precision="fp32")'
        in source
    )
    assert "assert thread_gap <= 1e-5" in source
    assert (
        source.index("original_prepared = list(pool.map")
        < source.index("torch.set_num_threads(2)")
        < source.index("prepared = list(pool.map", source.index("torch.set_num_threads(2)"))
    )
    q = fix.qualification(facade.contract.candidate(**export), facade.adapter._base)
    q.update(
        serving_worker_sha256=receipt["worker_sha256"],
        serving_contract_sha256=receipt["contract_sha256"],
        serving_helpers_sha256=receipt["serving_helpers_sha256"],
        model_dataset_ref=receipt["model_dataset_ref"],
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

    def compile_competition(value):
        return facade.build(
            parent,
            [export],
            mode="competition",
            model_dataset_ref=receipt["model_dataset_ref"],
            qualification=value,
        )

    with pytest.raises(ValueError, match="four-core"):
        compile_competition(q)
    q.update(thread_budget_version=S.VERSION, threads_per_lane=2)
    for m in q["models"][0]["measurements"]:
        m.update(
            thread_reference_count=1,
            threads_per_lane=2,
            thread1_reference_other_fields_exact=True,
            thread1_reference_context_bitwise_equal=False,
            thread1_reference_image_max_absolute_difference=1.1920928955078125e-7,
            thread_FP32_probability_gap=1e-7,
            thread_FP32_probability_tolerance=1e-5,
        )
    compile_competition(q)
    for key, value in (
        ("thread_FP32_probability_gap", 2e-5),
        ("thread1_reference_other_fields_exact", False),
        ("thread_reference_count", 2),
        ("thread1_reference_image_max_absolute_difference", float("nan")),
    ):
        bad = copy.deepcopy(q)
        bad["models"][0]["measurements"][0][key] = value
        with pytest.raises(ValueError):
            compile_competition(bad)
    bad = copy.deepcopy(q)
    bad["models"][0]["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        compile_competition(bad)
