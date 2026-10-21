"""Prepared four-core contingency; never edits a live owner or submits a model.

Apply after the existing exact preparation and resilience capsule. Qualification
retains original one-thread contexts and compares actual trained predictions.
"""

import ast
from pathlib import Path

VERSION = "four_core_two_lane_original_thread1_reference_v1"


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("One exact original source anchor required")
    return text.replace(old, new)


def compare_contexts(expected, actual):
    """Only finite FP32 image rounding <=1e-6; every other value byte-exact."""
    import torch

    assert len(expected) == len(actual) == 2
    maximum, exact = 0.0, True
    keys = {"images", "token_metadata", "region_metadata", "targets", "observed", "supports"}
    for e, a in zip(expected, actual, strict=True):
        assert set(e) == set(a) == keys
        for key in keys - {"supports"}:
            x, y = e[key], a[key]
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.device == y.device == torch.device("cpu")
            equal = torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
            exact &= equal
            if key == "images":
                assert (
                    x.dtype == torch.float32 and torch.isfinite(x).all() and torch.isfinite(y).all()
                )
                difference = float((x - y).abs().max()) if x.numel() else 0.0
                assert difference <= 1e-6, "Image rounding exceeds measured engineering bound"
                maximum = max(maximum, difference)
            else:
                assert equal, "Non-image context changed"
        es, ass = e["supports"], a["supports"]
        assert type(es) is type(ass) and es.region_count == ass.region_count
        for key in ("indptr", "regions", "log_prior"):
            x, y = getattr(es, key), getattr(ass, key)
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.device == y.device == torch.device("cpu")
            assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
    return {
        "thread1_reference_image_max_absolute_difference": maximum,
        "thread1_reference_context_bitwise_equal": bool(exact),
        "thread1_reference_other_fields_exact": True,
    }


def upgrade(original):
    worker = original["worker.py.txt"]
    worker = replace(worker, '    OMP_NUM_THREADS="1",', '    OMP_NUM_THREADS="2",')
    worker = replace(worker, '    MKL_NUM_THREADS="1",', '    MKL_NUM_THREADS="2",')
    worker = replace(
        worker,
        "    torch.set_num_threads(1)",
        '    torch.set_num_threads(1 if BINDING["mode"] == "qualification" else 2)',
    )
    worker = replace(
        worker,
        "from _predictive_descriptor_cache import prepare_contexts, load_study, predict_with_precision_recovery",
        "from _predictive_descriptor_cache import prepare_contexts, load_study, predict_with_precision_recovery, compare_thread_contexts",
    )
    old = """    prepared = None
    if BINDING["mode"] == "qualification":
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepared = list(pool.map(prepare, range(3)))"""
    new = """    prepared = None
    original_prepared = None
    if BINDING["mode"] == "qualification":
        # Thread settings change only after the whole prior pool has joined.
        with ThreadPoolExecutor(max_workers=2) as pool:
            original_prepared = list(pool.map(prepare, range(3)))
        torch.set_num_threads(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepared = list(pool.map(prepare, range(3)))
        for (reference_context, _), (candidate_context, stats) in zip(original_prepared, prepared, strict=True):
            stats.update(compare_thread_contexts(reference_context, candidate_context))
            stats["thread_reference_count"] = 1
            stats["threads_per_lane"] = 2"""
    worker = replace(worker, old, new)
    worker = replace(
        worker,
        '                    reference = predict(models[device], contexts, precision="fp32")',
        '''                    reference = predict(models[device], original_prepared[index][0], precision="fp32")
                    candidate_high = predict(models[device], contexts, precision="fp32")
                    thread_gap = float((candidate_high - reference).abs().max())
                    assert thread_gap <= 1e-5, "Trained thread-budget prediction difference exceeded bound"''',
    )
    worker = replace(
        worker,
        "                        maximum_probability_gap=float((reference - low).abs().max()),",
        """                        maximum_probability_gap=float((reference - low).abs().max()),
                        thread_FP32_probability_gap=thread_gap,
                        thread_FP32_probability_tolerance=1e-5,""",
    )
    worker = replace(
        worker,
        '        "clinical_score": None,',
        f'        "clinical_score": None,\n        "thread_budget_version": {VERSION!r},\n        "threads_per_lane": 2,',
    )
    ast.parse(worker)
    return {**original, "worker.py.txt": worker}


def require_qualification(receipt):
    if receipt.get("thread_budget_version") != VERSION or receipt.get("threads_per_lane") != 2:
        raise ValueError("Own actual four-core trained qualification required")
    records = receipt.get("models", [])
    if not records:
        raise ValueError("Actual trained records required")
    for row in records:
        measurements = row.get("measurements", [])
        if {m.get("study_index") for m in measurements} != {0, 1, 2}:
            raise ValueError("All three original public contexts required")
        for m in measurements:
            if not (
                m.get("thread_reference_count") == 1
                and m.get("threads_per_lane") == 2
                and m.get("thread1_reference_other_fields_exact") is True
                and type(m.get("thread1_reference_context_bitwise_equal")) is bool
                and 0
                <= m.get("thread1_reference_image_max_absolute_difference", float("inf"))
                <= 1e-6
                and 0 <= m.get("thread_FP32_probability_gap", float("inf")) <= 1e-5
                and m.get("thread_FP32_probability_tolerance") == 1e-5
            ):
                raise ValueError(
                    "Original one-thread input and trained prediction comparison required"
                )


def capsule(files):
    """Add a source-bound compiler layer; original graph and admissions remain."""
    import inspect

    body = Path(__file__).read_text()
    compare = inspect.getsource(compare_contexts).replace(
        "def compare_contexts(", "def compare_thread_contexts(", 1
    )
    portable = files["serving/preparation_upgrade.py"]
    portable += f"\n_THREAD_SOURCE = {body!r}\n_THREAD_COMPARE = {compare!r}\n"
    portable += """
import types as _thread_types
_thread_module = _thread_types.ModuleType("four_core_serving")
exec(compile(_THREAD_SOURCE, "four_core_serving", "exec"), _thread_module.__dict__)
_thread_previous_upgrade = upgrade
_thread_previous_qualification = require_qualification
def upgrade(original):
    files = _thread_module.upgrade(_thread_previous_upgrade(original))
    files["descriptor_cache.py"] += "\\n" + _THREAD_COMPARE
    return files
def require_qualification(receipt):
    _thread_previous_qualification(receipt)
    _thread_module.require_qualification(receipt)
"""
    ast.parse(portable)
    return {**files, "serving/preparation_upgrade.py": portable}
