"""Execute the real final compiler in a staged isolated capsule."""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


U = load("capsule_upgrade", HERE / "serving_upgrade.py")
F = load(
    "original_C100_capsule_tests",
    HERE.parent / "predictive-density100-deployment-20261021-v1/test_controls.py",
)


def test_staged_compiler_preserves_final_contract_and_requires_exact_new_contexts(
    monkeypatch, tmp_path
):
    _, original, entry, _, _ = F.capsule.__wrapped__(monkeypatch, tmp_path)
    files = U.capsule(original)
    assert {n for n in original if original[n] != files[n]} == {"serving/compile.py"}
    for n, s in files.items():
        p = tmp_path / n
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text(s)
    facade = load("fast_C100_test", tmp_path / "serving/compile.py")
    facade.adapter._base.LICENSE = F.FIX.C.digest(entry["license_bytes"])
    parent = "synthetic parent; no clinical pixels"
    facade.adapter._base.PARENT_SOURCE = F.FIX.C.digest(parent)
    F.FIX.rewrite(entry, lambda p: p.update(parent_source_sha256=F.FIX.C.digest(parent)))
    entry["full_terminal"]["parent_source_sha256"] = F.FIX.C.digest(parent)
    facade.EVIDENCE = {k: entry[k] for k in F.CTL.EXTRA_KEYS}
    monkeypatch.setattr(facade.implementation.C, "Contract", lambda *_a, **_kw: facade.adapter)
    export = {k: v for k, v in entry.items() if k not in F.CTL.EXTRA_KEYS}
    source, receipt = facade.build(
        parent, [export], mode="qualification", model_dataset_ref="godlydonuts/synthetic-c100"
    )
    ast.parse(source)
    assert 'verify_original=BINDING["mode"] == "qualification"' in source
    assert receipt["worker_sha256"] != U.WORKER_SHA
    q = F.FIX.qualification(facade.contract.candidate(**export), facade.adapter._base)
    q.update(
        serving_worker_sha256=receipt["worker_sha256"],
        serving_contract_sha256=receipt["contract_sha256"],
        serving_helpers_sha256=receipt["serving_helpers_sha256"],
        model_dataset_ref=receipt["model_dataset_ref"],
    )
    with pytest.raises(ValueError, match="selected-frame"):
        facade.build(
            parent,
            [export],
            mode="competition",
            model_dataset_ref=receipt["model_dataset_ref"],
            qualification=q,
        )
    q["preparation_version"] = "exact_selected_frames_and_batched_descriptors_v1"
    for m in q["models"][0]["measurements"]:
        m.update(
            original_context_bitwise_equal=True,
            context_equivalence_is_public_only=True,
            preparation_version=q["preparation_version"],
            original_render_comparison_seconds=10,
            render_seconds=5,
        )
    facade.build(
        parent,
        [export],
        mode="competition",
        model_dataset_ref=receipt["model_dataset_ref"],
        qualification=q,
    )
    bad = copy.deepcopy(q)
    bad["models"][0]["measurements"][0]["original_context_bitwise_equal"] = False
    with pytest.raises(ValueError, match="equality"):
        facade.build(
            parent,
            [export],
            mode="competition",
            model_dataset_ref=receipt["model_dataset_ref"],
            qualification=bad,
        )
    bad = copy.deepcopy(q)
    bad["models"][0]["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        facade.build(
            parent,
            [export],
            mode="competition",
            model_dataset_ref=receipt["model_dataset_ref"],
            qualification=bad,
        )
