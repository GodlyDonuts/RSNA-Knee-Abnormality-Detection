import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "predictive-context96-T4-20261021-v1"


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = load("fourcore_test_original", OLD / "compile.py")
N = load("fourcore_test_compiler", HERE / "compile.py")
G = load("fourcore_test_control", HERE / "control.py")
F = load("fourcore_test_inputs", HERE.parents[2] / "tests/test_predictive_descriptor_cache.py")


def test_compiler_preserves_trained_model_and_adds_complete_direct_comparison(
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

    # Exercise the prospective competition derivation from the same synthetic
    # source graph, with a deliberately synthetic receipt/model binding.
    candidate = load("dense_candidate_test", OLD / "candidate.py")
    old_values, old_worker = C.split_source(source)
    old_values["BINDING"]["mode"] = "competition"
    for name, value in old_values.items():
        old_worker = C.replace(old_worker, "__" + name + "_VALUE__", repr(value))
    diagnostic = json.loads((OLD / "actual-complete.json").read_text())
    chosen = old_values["BINDING"]["models"][0]
    diagnostic["receipt"]["models"][0]["trained_export_sha256"] = chosen["model"]["sha256"]
    diagnostic["receipt"]["models"][0]["candidate_sha256"] = hashlib.sha256(
        candidate.canonical(chosen)
    ).hexdigest()
    monkeypatch.setattr(candidate, "PARENT_COMPETITION", C.sha(old_worker))
    monkeypatch.setattr(
        candidate, "RUNTIME_SHA", C.sha(old_values["FILES"]["_predictive_serving_contract.py"])
    )
    monkeypatch.setattr(candidate.D, "MODEL", chosen["model"]["sha256"])
    public, built = candidate.build(old_worker, diagnostic)
    monkeypatch.setattr(N, "PARENT_PUBLIC", C.sha(public))
    monkeypatch.setattr(N, "MODEL", chosen["model"]["sha256"])
    new, result = N.build(public)
    before, _ = C.split_source(public)
    after, worker = C.split_source(new)
    assert before["BINDING"]["models"] == after["BINDING"]["models"]
    assert before["BINDING"]["qualification"] == after["BINDING"]["qualification"]
    assert before["PARENT_SOURCE"] == after["PARENT_SOURCE"]
    assert before["PERMISSION"] == after["PERMISSION"]
    assert {k for k in before["FILES"] if before["FILES"][k] != after["FILES"][k]} == {
        "_predictive_descriptor_cache.py"
    }
    assert after["BINDING"]["public_watchdog_seconds"] == 480
    assert "if public_timer is not None:" in worker
    assert worker.index("prepared = list(pool.map(prepare, range(3)))") < worker.index(
        "torch.set_num_threads(1)"
    )
    assert (
        'if BINDING["mode"] == "competition":\n        path = Path("/kaggle/working/submission.csv")'
        in worker
    )
    assert "np.savez(paired_path" in worker
    assert "thread_FP32_probability_gap=fp32_gap" in worker
    assert result["competition_entry"] is False
    ast.parse(new)
    with pytest.raises(AssertionError):
        N.build(public + "\n")


def test_actual_contexts_thread_change_and_complete_field_corruption():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        groups = F.cache.cache_descriptors_once(F.synthetic_groups(frames_per_group=12))
        expected = F.assemble(groups, side=32, context_cap=96)
        torch.set_num_threads(2)
        actual = F.assemble(groups, side=32, context_cap=96)
        evidence = N.T.compare_contexts(expected, actual)
        assert evidence["thread1_reference_other_fields_exact"] is True
        for key in ("images", "token_metadata", "targets", "region_metadata"):
            bad = copy.deepcopy(actual)
            bad[0][key].reshape(-1)[0] += 0.01
            with pytest.raises(AssertionError):
                N.T.compare_contexts(expected, bad)
        bad = copy.deepcopy(actual)
        bad[0]["supports"].log_prior[0] += 0.01
        with pytest.raises(AssertionError):
            N.T.compare_contexts(expected, bad)
    finally:
        torch.set_num_threads(previous)


def test_controller_keeps_colab_reserve_and_cannot_submit():
    source = G.derive((OLD / "account.py").read_text())
    assert "free >= 1920 + 480 + 120" in source
    assert "sessionTimeoutSeconds=32400" in source
    assert "own_timeout_seconds=480" in source
    assert "context96-fourcore-v1:" in source
    assert 'assert not row and not (ROOT / "push-intent.json").exists()' in source
    assert "CreateSubmission" not in source and "submit(" not in source
    ast.parse(source)
    with pytest.raises(AssertionError):
        G.derive(
            (OLD / "account.py")
            .read_text()
            .replace("sessionTimeoutSeconds=600,", "sessionTimeoutSeconds=601,")
        )
    blobs = G.output_module((HERE.parents[2] / "scripts/kaggle/output_csv.py").read_text())
    assert '"filePath": filename' in blobs
    assert 'transport = remote.session if host == "api.kaggle.com" else requests' in blobs
    ast.parse(blobs)


def test_actual_paired_arrays_are_rechecked_and_bad_numerics_rejected():
    old = (
        json.loads((OLD / "actual-public-complete.json").read_text())
        if (OLD / "actual-public-complete.json").exists()
        else json.loads((OLD / "actual-complete.json").read_text())
    )
    # Use actual receipt shape but synthetic probabilities, never private MRI rows.
    r = old["receipt"]
    r.update(
        status="actual_fourcore_fixed96_public_comparison_complete",
        thread_comparison_version=N.VERSION,
        threads_per_lane=2,
        actual_test_studies=3,
        submission_csv_absent=False,
        public_watchdog_started=True,
        public_watchdog_seconds=480,
        elapsed_seconds=100,
        CSV_sha256="synthetic",
        CSV_bytes=123,
    )
    r["models"][0]["trained_export_sha256"] = N.MODEL
    blobs = {}
    records = []
    for index, device in [(0, 0), (1, 1), (2, 0), (0, 1)]:
        a = np.full(12, 0.5, dtype="<f4")
        stream = io.BytesIO()
        np.savez(stream, candidate_FP16=a, reference_FP16=a, candidate_FP32=a, reference_FP32=a)
        raw = stream.getvalue()
        name = f"thread-paired-study-{index}-gpu-{device}.npz"
        blobs[name] = raw
        records.append(
            dict(
                study_index=index,
                device=device,
                candidate_prepared_before_reference=True,
                threads_per_lane=2,
                thread_reference_count=1,
                thread1_reference_other_fields_exact=True,
                thread1_reference_image_max_absolute_difference=0,
                maximum_probability_gap=0,
                thread_FP32_probability_gap=0,
                thread_FP16_probability_gap=0,
                inference_context_cap=96,
                decoder_route="original_exact",
                prediction_precision="fp16",
                paired_prediction_file=name,
                paired_prediction_file_bytes=len(raw),
                paired_prediction_file_sha256=hashlib.sha256(raw).hexdigest(),
                decode_seconds=1,
                render_seconds=1,
                FP16_seconds=1,
            )
        )
    r["models"][0].update(
        measurements=records, memory_accepted=True, precision_accepted=True, cross_device_FP16_gap=0
    )
    remote = SimpleNamespace(output_json=lambda *a: r)
    output = SimpleNamespace(
        download_named=lambda *a: blobs[a[-1]],
        download_csv=lambda *a: b"synthetic",
        verify_csv=lambda *a: {"CSV_sha256": "synthetic", "bytes": 123},
    )
    binding = {
        k: r[k]
        for k in (
            "serving_worker_sha256",
            "serving_contract_sha256",
            "serving_helpers_sha256",
            "model_dataset_ref",
        )
    }
    binding["public_roster_sha256"] = "synthetic"
    accepted = G.accept_public(remote, "godlydonuts/synthetic", {"binding": binding}, N, output)
    assert len(accepted["private_paired_objects_verified"]) == 4
    assert accepted["runtime_planning"]["mean_below_9_hours"] is True
    name = records[0]["paired_prediction_file"]
    stream = io.BytesIO()
    np.savez(stream, candidate_FP16=a, reference_FP16=a, candidate_FP32=a + 0.01, reference_FP32=a)
    bad = stream.getvalue()
    blobs[name] = bad
    records[0].update(
        paired_prediction_file_bytes=len(bad),
        paired_prediction_file_sha256=hashlib.sha256(bad).hexdigest(),
    )
    with pytest.raises(AssertionError):
        G.accept_public(remote, "godlydonuts/synthetic", {"binding": binding}, N, output)


def test_stdlib_reader_accepts_original_numpy_format_and_rejects_object_or_wrong_shape():
    repair = load("fourcore_stdlib_reader_test", HERE / "reconcile.py")
    for dtype, shape, ok in (
        ("<f4", (12,), True),
        ("<f4", (13,), False),
        ("<f8", (12,), False),
        (object, (12,), False),
    ):
        stream = io.BytesIO()
        np.savez(
            stream,
            **{
                k: np.full(shape, 0.5, dtype=dtype)
                for k in ("candidate_FP16", "reference_FP16", "candidate_FP32", "reference_FP32")
            },
        )
        if ok:
            assert all(v == (0.5,) * 12 for v in repair.read_paired(stream.getvalue()).values())
        else:
            with pytest.raises(AssertionError):
                repair.read_paired(stream.getvalue())


def test_dependency_free_acceptance_keeps_actual_numerical_rejection(monkeypatch):
    repair = load("fourcore_stdlib_accept_test", HERE / "reconcile.py")
    ns = {"ast": ast, "read_paired": repair.read_paired, "probability_gap": repair.probability_gap}
    exec(compile(repair.accept_source(G.accept_public), "reconciled_actual_reader", "exec"), ns)
    monkeypatch.setattr(G, "accept_public", ns["accept_public"])
    test_actual_paired_arrays_are_rechecked_and_bad_numerics_rejected()
