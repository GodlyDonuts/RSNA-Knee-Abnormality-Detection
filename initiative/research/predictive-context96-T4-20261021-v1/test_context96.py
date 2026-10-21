import ast
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = load("dense96_compile_test", HERE / "compile.py")
F = load("dense96_synthetic_groups", HERE.parents[2] / "tests/test_predictive_descriptor_cache.py")


def test_more_images_preserve_every_region_and_support_and_catch_corruption():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        groups = F.cache.cache_descriptors_once(F.synthetic_groups(frames_per_group=12))
        sparse = F.assemble(groups, side=32, context_cap=24)
        dense = F.assemble(groups, side=32, context_cap=96)
        assert len(sparse[0]["images"]) == 24 and len(dense[0]["images"]) == 36
        C.compare_region_fields(sparse, dense)
        for key in ("targets", "observed", "region_metadata"):
            bad = copy.deepcopy(dense)
            target = bad[0][key].reshape(-1)
            target[0] = not bool(target[0]) if target.dtype == torch.bool else 123
            with pytest.raises(AssertionError):
                C.compare_region_fields(sparse, bad)
        bad = copy.deepcopy(dense)
        bad[0]["supports"].log_prior[0] += 0.01
        with pytest.raises(AssertionError):
            C.compare_region_fields(sparse, bad)
    finally:
        torch.set_num_threads(previous)


def test_actual_compiler_layers_keep_model_and_provenance_while_changing_only_context(
    monkeypatch, tmp_path
):
    original = load(
        "dense_resilience_fixture",
        HERE.parent / "predictive-inference-resilience-20261021-v1/test_capsule.py",
    )
    _, raw, entry, _, _ = original.F.capsule.__wrapped__(monkeypatch, tmp_path)
    resilience = load(
        "dense_resilience_builder",
        HERE.parent / "predictive-inference-resilience-20261021-v1/serving.py",
    )
    files = resilience.capsule(original.U.capsule(raw))
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    facade = load("dense_original_compiler", tmp_path / "serving/compile.py")
    fix = original.F.FIX
    facade.adapter._base.LICENSE = fix.C.digest(entry["license_bytes"])
    parent = "synthetic encoder; no private source or weights"
    facade.adapter._base.PARENT_SOURCE = fix.C.digest(parent)
    fix.rewrite(entry, lambda p: p.update(parent_source_sha256=fix.C.digest(parent)))
    entry["full_terminal"]["parent_source_sha256"] = fix.C.digest(parent)
    facade.EVIDENCE = {k: entry[k] for k in original.F.CTL.EXTRA_KEYS}
    monkeypatch.setattr(facade.implementation.C, "Contract", lambda *_a, **_kw: facade.adapter)
    export = {k: v for k, v in entry.items() if k not in original.F.CTL.EXTRA_KEYS}
    source, _ = facade.build(
        parent, [export], mode="qualification", model_dataset_ref="godlydonuts/synthetic-c100"
    )
    old, _ = C.split_source(source)
    monkeypatch.setattr(C, "PARENT", C.sha(source))
    monkeypatch.setattr(C, "MODEL", old["BINDING"]["models"][0]["model"]["sha256"])
    result, compiled = C.build(source)
    actual, worker = C.split_source(result)
    assert actual["BINDING"]["models"] == old["BINDING"]["models"]
    assert (
        actual["PARENT_SOURCE"] == old["PARENT_SOURCE"]
        and actual["PERMISSION"] == old["PERMISSION"]
    )
    changed = {k for k in old["FILES"] if old["FILES"][k] != actual["FILES"][k]}
    assert changed == {"_predictive_descriptor_cache.py"}
    assert actual["BINDING"]["inference_context_cap"] == 96
    assert actual["BINDING"]["models"][0]["architecture"]["context_cap"] == 24
    assert 'assert BINDING["mode"] == "qualification"' in worker
    assert "Original24 predictions changed" in worker
    assert compiled["competition_entry"] is False
    ast.parse(result)
    with pytest.raises(AssertionError):
        C.build(source + "\n")
