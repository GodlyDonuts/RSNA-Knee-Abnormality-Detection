"""Distinct two-thread dense candidate with direct trained numerical comparison.

Pure source transformation. Original private source/model remain on owning node.
The old hash-only diagnostic remains historical, not retrospectively accepted.
"""

import ast
import copy
import importlib.util
import inspect
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load(
    "dense_fourcore_original",
    (HERE / "original_compile.py")
    if (HERE / "original_compile.py").exists()
    else HERE.parent / "predictive-context96-T4-20261021-v1/compile.py",
)
T = load(
    "dense_fourcore_compare",
    (HERE / "thread_compare.py")
    if (HERE / "thread_compare.py").exists()
    else HERE.parent / "predictive-thread-budget-serving-20261021-v1/serving.py",
)
PARENT = D.PARENT
PARENT_PUBLIC = "d4d26dd53300e0b28472b87473d5b4da9a333ded5e2cdd7a7b2819d5038e0e70"
MODEL = D.MODEL
VERSION = "fixed96_fourcore_direct_thread_reference_v1"


def build(original):
    assert D.sha(original) == PARENT_PUBLIC
    values, worker = D.split_source(original)
    files, binding = values["FILES"], values["BINDING"]
    before = copy.deepcopy(binding)
    assert binding["mode"] == "competition" and binding["inference_context_cap"] == 96
    assert binding["training_context_cap"] == 24 and len(binding["models"]) == 1
    assert binding["models"][0]["model"]["sha256"] == MODEL
    assert all(D.sha(body) == binding["module_sha256"][key] for key, body in files.items())
    helper = files["_predictive_descriptor_cache.py"]
    helper += "\n" + inspect.getsource(T.compare_contexts)
    files["_predictive_descriptor_cache.py"] = helper
    for old, new in (
        ('    OMP_NUM_THREADS="1",', '    OMP_NUM_THREADS="2",'),
        ('    MKL_NUM_THREADS="1",', '    MKL_NUM_THREADS="2",'),
        ("    torch.set_num_threads(1)", "    torch.set_num_threads(2)"),
        (
            "from _predictive_descriptor_cache import prepare_contexts, load_study, predict_with_precision_recovery",
            "from _predictive_descriptor_cache import prepare_contexts, load_study, predict_with_precision_recovery, compare_contexts",
        ),
    ):
        worker = D.replace(worker, old, new)
    worker = D.replace(
        worker,
        """    prepared = None
    if BINDING["mode"] == "qualification":
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepared = list(pool.map(prepare, range(3)))""",
        """    prepared = None
    original_prepared = None
    # Exact public roster only: candidate first/cold, then original thread1.
    # All pools join before changing global intra-op thread count.
    if public_timer is not None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepared = list(pool.map(prepare, range(3)))
        torch.set_num_threads(1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            original_prepared = list(pool.map(prepare, range(3)))
        torch.set_num_threads(2)
        for (expected, _), (actual, stats) in zip(original_prepared, prepared, strict=True):
            stats.update(compare_contexts(expected, actual))
            stats.update(threads_per_lane=2, thread_reference_count=1,
                         candidate_prepared_before_reference=True)""",
    )
    worker = D.replace(
        worker,
        '                    reference = predict(models[device], contexts, precision="fp32")',
        """                    reference = predict(models[device], original_prepared[index][0], precision="fp32")
                    candidate_high = predict(models[device], contexts, precision="fp32")
                    reference_low = predict(models[device], original_prepared[index][0], precision="fp16")
                    fp32_gap = float((candidate_high - reference).abs().max())
                    fp16_gap = float((low - reference_low).abs().max())
                    assert fp32_gap <= 1e-5 and fp16_gap <= 1e-4, "Thread prediction difference exceeds fixed tolerance"
                    # Retain actual arrays privately; the prior diagnostic only saved hashes.
                    paired_path = Path("/kaggle/working") / ("thread-paired-study-" + str(index) + "-gpu-" + str(device) + ".npz")
                    assert not paired_path.exists()
                    np.savez(paired_path, candidate_FP16=values,
                             reference_FP16=reference_low.cpu().numpy().astype("<f4"),
                             candidate_FP32=candidate_high.cpu().numpy().astype("<f4"),
                             reference_FP32=reference.cpu().numpy().astype("<f4"))""",
    )
    worker = D.replace(
        worker,
        "                        maximum_probability_gap=float((reference - low).abs().max()),",
        """                        maximum_probability_gap=float((candidate_high - low).abs().max()),
                        thread_FP32_probability_gap=fp32_gap,
                        thread_FP16_probability_gap=fp16_gap,
                        paired_prediction_file=paired_path.name,
                        paired_prediction_file_bytes=paired_path.stat().st_size,
                        paired_prediction_file_sha256=hashlib.sha256(paired_path.read_bytes()).hexdigest(),""",
    )
    worker = D.replace(
        worker,
        '            "actual_trained_predictive_public_T4_qualification_accepted"',
        '            "actual_fourcore_fixed96_public_comparison_complete"',
    )
    worker = D.replace(
        worker,
        '    if prepared is None:\n        path = Path("/kaggle/working/submission.csv")',
        '    if BINDING["mode"] == "competition":\n        path = Path("/kaggle/working/submission.csv")',
    )
    worker = D.replace(
        worker,
        '        "clinical_score": None,',
        f'        "clinical_score": None,\n        "thread_comparison_version": {VERSION!r},\n        "threads_per_lane": 2,',
    )
    binding.update(
        parent_context96_public_source_sha256=PARENT_PUBLIC,
        thread_comparison_version=VERSION,
        public_watchdog_seconds=480,
        threads_per_lane=2,
    )
    binding["module_sha256"]["_predictive_descriptor_cache.py"] = D.sha(helper)
    binding["serving_helpers_sha256"]["_predictive_descriptor_cache.py"] = D.sha(helper)
    binding["serving_worker_sha256"] = D.sha(worker)
    assert binding["models"] == before["models"]
    assert binding["qualification"] == before["qualification"]
    assert all(
        binding["module_sha256"][k] == v
        for k, v in before["module_sha256"].items()
        if k != "_predictive_descriptor_cache.py"
    )
    for name, value in values.items():
        worker = D.replace(worker, "__" + name + "_VALUE__", repr(value))
    ast.parse(worker)
    ast.parse(helper)
    return worker, dict(
        status="fourcore_fixed96_compiled_no_GPU_or_entry",
        source_sha256=D.sha(worker),
        source_bytes=len(worker.encode()),
        binding=binding,
        competition_entry=False,
        new_Dataset=False,
        actual_reference_arrays_required=True,
    )
