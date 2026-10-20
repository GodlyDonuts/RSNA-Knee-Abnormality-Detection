"""Meaningful synthetic execution of the prepared delta; no GPU/account/model downloads."""

import ast
import copy
import hashlib
import importlib.util
import json
import threading
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "initiative/research/teacher-rank-delta-serving-20261020-v1"
SPEC = importlib.util.spec_from_file_location("rank_delta_adapter_test", HERE / "adapter.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def runtime(tmp_path):
    namespace = {"__name__": "synthetic_rank_delta", "np": np}
    for name in ("prelude.py", "publiccheck.py"):
        exec(compile((HERE / name).read_text(), name, "exec"), namespace)
    namespace["_RANK_DELTA_OUTPUT"] = tmp_path / "submission.csv"
    namespace["_RANK_DELTA_RECEIPT"] = tmp_path / "receipt.json"
    namespace["_RANK_DELTA_BINDING"] = {
        "public_roster_sha256": hashlib.sha256(b'["a","b","c"]').hexdigest(),
        "public_watchdog_seconds": 180,
        "model_sha256": adapter.MODEL_SHA,
        "training_source_sha256": adapter.WORKER_SHA,
    }
    namespace["_ADAPTED_TRAINING"] = {
        "role": "teacher_rank", "epoch": 3, "receipt": {"completed": 1458},
        "source_sha256": adapter.WORKER_SHA,
    }
    namespace["_ADAPTED_SHA"] = adapter.MODEL_SHA
    return namespace


def mount(tmp_path, name, ids):
    path = tmp_path / name
    path.mkdir()
    (path / "test.csv").write_text("StudyInstanceUID\n" + "\n".join(ids) + "\n")
    return path


class Timer:
    instances = []

    def __init__(self, seconds, callback):
        self.seconds, self.callback = seconds, callback
        self.started = self.cancelled = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


@pytest.mark.parametrize("nested", [False, True])
def test_stdlib_early_roster_arms_before_heavy_imports_and_rejects_ambiguity(tmp_path, monkeypatch, nested):
    ns = runtime(tmp_path)
    monkeypatch.setattr(threading, "Timer", Timer)
    p = mount(tmp_path, "nested" if nested else "flat", ["a", "b", "c"])
    absent = tmp_path / "absent"
    result = ns["_rank_delta_early"]((absent, p) if nested else (p, absent))
    assert result["timer"].started and result["timer"].seconds == 180
    with pytest.raises(AssertionError, match="unambiguous"):
        ns["_rank_delta_early"]((absent,))
    p2 = mount(tmp_path, "another", ["a", "b", "c"])
    with pytest.raises(AssertionError, match="unambiguous"):
        ns["_rank_delta_early"]((p, p2))
    # A different three-row roster is NOT the canonical public data.
    p3 = mount(tmp_path, "hidden", ["d", "e", "f"])
    assert ns["_rank_delta_early"]((p3,))["timer"] is None


def original_result(ns, ids):
    return {
        "status": "actual_teacher_pair_final_epoch_MRI_test_roster_CSV_complete",
        "test_roster_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "studies": len(ids), "model_sha256": adapter.MODEL_SHA,
        "prediction_counts": {
            "started": 2 * len(ids), "completed": 2 * len(ids),
            "failed": 0, "decode_errors": 0, "empty_views": 0,
        },
    }


@pytest.mark.parametrize("public", [True, False])
def test_one_original_competition_call_public_acceptance_hidden_no_selftest(tmp_path, monkeypatch, public):
    ns = runtime(tmp_path)
    monkeypatch.setattr(threading, "Timer", Timer)
    ids = ["a", "b", "c"] if public else ["h", "i", "j"]
    p = mount(tmp_path, "data", ids)
    ns["_RANK_DELTA_EARLY"] = ns["_rank_delta_early"]((p,))
    calls = []

    def boot():
        calls.append("boot")
        assert (ns["_RANK_DELTA_EARLY"]["timer"] is not None) == public

    def check(*args):
        calls.append("check")
        return {"synthetic": True}

    def competition():
        calls.append("competition")
        ns["_RANK_DELTA_OUTPUT"].write_text("synthetic CSV")
        return original_result(ns, ids)

    ns.update(_adapted_boot=boot, _rank_delta_public_check=check, _adapted_competition=competition)
    ns["_rank_delta_main"]()
    assert calls == (["boot", "check", "competition"] if public else ["boot", "competition"])
    actual = json.loads(ns["_RANK_DELTA_RECEIPT"].read_text())
    assert ("inline_new_weight_acceptance" in actual) == public
    assert ("public_total_elapsed_seconds" in actual) == public
    if public:
        assert ns["_RANK_DELTA_EARLY"]["timer"].cancelled


@pytest.mark.parametrize("failure", ["selftest", "fallback", "timeout"])
def test_failures_remove_csv_and_acceptance_and_never_continue_after_selftest(tmp_path, monkeypatch, failure):
    ns = runtime(tmp_path)
    monkeypatch.setattr(threading, "Timer", Timer)
    p = mount(tmp_path, "data", ["a", "b", "c"])
    ns["_RANK_DELTA_EARLY"] = ns["_rank_delta_early"]((p,))
    calls = []
    ns["_adapted_boot"] = lambda: None

    def check(*args):
        if failure == "selftest":
            raise AssertionError("different graph")
        return {}

    def competition():
        calls.append("competition")
        ns["_RANK_DELTA_OUTPUT"].write_text("must be removed")
        result = original_result(ns, ["a", "b", "c"])
        if failure == "fallback":
            result["prediction_counts"]["failed"] = 1
        if failure == "timeout":
            ns["_RANK_DELTA_RECEIPT"].write_text("partial")
            ns["_rank_delta_expired"]()
        return result

    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(ns["os"], "_exit", exit_)
    ns.update(_rank_delta_public_check=check, _adapted_competition=competition)
    with pytest.raises((AssertionError, SystemExit)):
        ns["_rank_delta_main"]()
    assert not ns["_RANK_DELTA_OUTPUT"].exists() and not ns["_RANK_DELTA_RECEIPT"].exists()
    assert calls == ([] if failure == "selftest" else ["competition"])


@pytest.mark.parametrize("bad", [None, "gpu_input", "trained_graph"])
def test_both_devices_three_studies_full_two_view_loop_and_exception_propagation(tmp_path, bad):
    ns = runtime(tmp_path)
    thread = threading.local()
    barrier = threading.Barrier(2)
    calls, lock = Counter(), threading.Lock()

    class CUDA:
        @contextmanager
        def device(self, i):
            thread.device = i
            yield

        def device_count(self):
            return 2

        def get_device_name(self, i):
            return "Tesla T4 synthetic"

        def synchronize(self, i):
            assert thread.device == i

        def max_memory_allocated(self, i):
            return 2 * 2**30

        def empty_cache(self):
            with lock:
                calls[("cleanup", thread.device)] += 1

    class Torch:
        cuda = CUDA()

        def __getattr__(self, name):
            return getattr(torch, name)

        def autocast(self, *args, **kwargs):
            return nullcontext()

    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.reference = bool(args)

        def to(self, *args, **kwargs):
            return self

        def forward(self, images):
            assert not torch.is_grad_enabled()
            with lock:
                calls[("reference" if self.reference else "served", thread.device)] += 1
            shift = .1 if bad == "trained_graph" and self.reference else 0.
            return (images.mean() * self.weight + shift).expand(1, 12)

    def load(*args):
        barrier.wait(timeout=5)
        return Model().eval(), 320

    def windows(volume, mask, k, res, **kwargs):
        assert (k, res) == (94, 320)
        return torch.full((94, 3, 2, 2), float(volume[0]) + int(kwargs["mirror"]))

    def gpu_windows(volume, mask, k, res, device, **kwargs):
        result = windows(volume, mask, k, res, **kwargs)
        if bad == "gpu_input":
            result += .1
        return result

    counts = {"started": 0, "completed": 0, "failed": 0, "decode_errors": 0, "empty_views": 0}
    ns.update(
        torch=Torch(), load_model=load, build_backbone=lambda *args, **kwargs: torch.nn.Identity(),
        RaptorClassifier=Model, ARMS=[{"arch": "synthetic"}], NORM="imagenet",
        _adapted_model_path=tmp_path / "unused.pt", eval_windows=windows, gpu_windows=gpu_windows,
        _ADAPTED_MIXIN="class CheckpointedEncode: pass",
        _ADAPTED_IMPORTANCE="class ImportanceReadout: pass",
        _adapted_counts=counts,
        _adapted_original_infer=lambda model, image, device: model(image[None]).sigmoid()[0].numpy(),
    )
    # Only shrink synthetic images. The production source remains exact94x3x320x320.
    text = (HERE / "publiccheck.py").read_text()
    assert text.count("(94, 3, 320, 320)") == 1
    exec(compile(text.replace("(94, 3, 320, 320)", "(94, 3, 2, 2)"), "synthetic_publiccheck", "exec"), ns)
    ns["_rank_delta_public_volumes"] = lambda *args: [(np.array([i]), np.ones(1)) for i in range(3)]
    if bad:
        with pytest.raises(AssertionError):
            ns["_rank_delta_public_check"](tmp_path, ["a", "b", "c"], ns["_RANK_DELTA_BINDING"]["public_roster_sha256"])
    else:
        actual = ns["_rank_delta_public_check"](tmp_path, ["a", "b", "c"], ns["_RANK_DELTA_BINDING"]["public_roster_sha256"])
        assert [d["MRI_studies"] for d in actual["devices"]] == [3, 3]
        assert all(d["maximum_original_class_vs_training_graph_probability_error"] == 0 for d in actual["devices"])
        assert actual["inherited32_reexecuted"] is False
        for i in (0, 1):
            assert calls[("reference", i)] == calls[("served", i)] == 6
    assert all(calls[("cleanup", i)] == 1 for i in (0, 1))
    assert counts == {"started": 0, "completed": 0, "failed": 0, "decode_errors": 0, "empty_views": 0}


def binding():
    return {
        "status": "prepared_teacher_rank_single_public_run_delta_source",
        "original_competition_sha256": adapter.ORIGINAL_SOURCE_SHA,
        "model_sha256": adapter.MODEL_SHA, "training_source_sha256": adapter.WORKER_SHA,
        "inherited32_receipt_sha256": adapter.INHERITED_SHA,
        "public_roster_sha256": adapter.PUBLIC_ROSTER_SHA,
        "adapter_sha256": "a" * 64, "publiccheck_sha256": "b" * 64, "prelude_sha256": "c" * 64,
        "public_watchdog_seconds": 180, "public_VM_setup_reserve_seconds": 80,
        "total_reserved_VM_seconds": 260, "hidden_session_timeout_seconds": 32400,
        "source_sha256": "d" * 64,
        "qualification_scope": "inherited32_same_code_plus_actual_public3_new_weights",
    }


def receipt():
    inline = {
        **binding(), "status": adapter.STATUS, "views_per_study": 2, "windows_per_view": 94,
        "actual_new_weight_studies_per_device": 3, "inherited_code_qualification_studies": 32,
        "inherited32_reexecuted": False, "elapsed_seconds": 80,
        "devices": [{
            "index": i, "GPU": "Tesla T4", "MRI_studies": 3,
            "both_original94_input_views_bit_equal": True,
            "complete_learned_graph_tensor_values_equal": True,
            "maximum_original_class_vs_training_graph_probability_error": 0,
            "peak_allocated_bytes": 2 * 2**30, "input_views_aggregate_sha256": "e" * 64,
        } for i in (0, 1)],
    }
    return {
        "status": "actual_teacher_pair_final_epoch_MRI_test_roster_CSV_complete",
        "role": "teacher_rank", "epoch": 3, "model_sha256": adapter.MODEL_SHA,
        "training_source_sha256": adapter.WORKER_SHA, "test_roster_sha256": adapter.PUBLIC_ROSTER_SHA,
        "studies": 3, "columns": 13, "CSV_bytes": 470, "CSV_sha256": "f" * 64,
        "prediction_counts": {"started": 6, "completed": 6, "failed": 0, "decode_errors": 0, "empty_views": 0},
        "inline_new_weight_acceptance": inline, "public_total_elapsed_seconds": 120,
    }


@pytest.mark.parametrize("field", ["model", "inputs", "coverage", "error", "fallback", "timeout", "nan", "source"])
def test_receipt_rejects_changed_state_coverage_error_fallback_or_budget(field):
    actual, metadata = receipt(), binding()
    assert adapter.validate_receipt(actual, metadata)["competition_entry_authority"] is False
    inline = actual["inline_new_weight_acceptance"]
    if field == "model":
        actual["model_sha256"] = "0" * 64
    elif field == "inputs":
        inline["devices"][1]["input_views_aggregate_sha256"] = "0" * 64
    elif field == "coverage":
        inline["devices"][1]["MRI_studies"] = 2
    elif field == "error":
        inline["devices"][1]["maximum_original_class_vs_training_graph_probability_error"] = 1e-5
    elif field == "fallback":
        actual["prediction_counts"]["empty_views"] = 1
    elif field == "timeout":
        actual["public_total_elapsed_seconds"] = 180.001
    elif field == "nan":
        actual["public_total_elapsed_seconds"] = float("nan")
    else:
        inline["publiccheck_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        adapter.validate_receipt(actual, metadata)


def test_exact_inherited_receipt_and_source_replacement_preserve_numeric_functions():
    text = (ROOT / "initiative/research/pointwise-final-deployment-20261020-v1/actual-qualification-accepted.json").read_text()
    assert adapter.inherited_evidence(text)["kernel_id"] == 139818636
    with pytest.raises(ValueError, match="inherited"):
        adapter.inherited_evidence(text + "\n")
    with pytest.raises(ValueError, match="competition source"):
        adapter.adapt("# fake", inherited_qualification_text=text)
    original = """import torch  # original heavy import
def _adapted_competition():
    return 'original inference'
if __name__ == '__main__':
    _adapted_boot()
    result = _adapted_qualify() if _ADAPTED_MODE == 'qualification' else _adapted_competition()
"""
    helper, prelude = ((HERE / n).read_text() for n in ("publiccheck.py", "prelude.py"))
    new = adapter._replace_entry(original, helper, prelude, binding())
    assert new.index("_RANK_DELTA_EARLY = _rank_delta_early(") < new.index("import torch")
    node = next(n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef) and n.name == "_adapted_competition")
    assert ast.get_source_segment(new, node) == "def _adapted_competition():\n    return 'original inference'"
    assert new.count("    _rank_delta_main()") == 1


def test_model_and_final_training_binding_are_explicit_even_before_original_boot():
    training = {
        "role": "teacher_rank", "epoch": 3, "status": "COMPLETE", "kernel_id": None,
        "training_Slurm_job": "975297", "source_sha256": adapter.WORKER_SHA,
        "receipt": {"completed": 1458, "epoch": {"bytes": 292867251, "sha256": adapter.MODEL_SHA}},
    }
    publication = {
        "model_sha256": adapter.MODEL_SHA, "role": "teacher_rank", "epoch": 3,
        "completed_updates": 1458, "objects": 6, "revision": "60a64c2be5d59f60e458caca1b6f475f82a95fa7",
    }

    def source(t):
        return "\n".join(f"{key}={value!r}" for key, value in {
            "_ADAPTED_MODE": "competition", "_ADAPTED_SHA": adapter.MODEL_SHA,
            "_ADAPTED_TRAINING": t, "_ADAPTED_PUBLICATION": publication,
        }.items())

    adapter._literal_bindings(source(training))
    bad = copy.deepcopy(training)
    bad["receipt"]["completed"] = 1457
    with pytest.raises(ValueError, match="ranking state"):
        adapter._literal_bindings(source(bad))
