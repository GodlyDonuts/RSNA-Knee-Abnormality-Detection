"""Derive one fixed96 public diagnostic from the actually qualified final C100.

Executed on the original owning node. The private parent body never leaves it.
No training, account mutation, label reads or competition submission here.
"""

import ast
import copy
import hashlib

PARENT = "fe364a3c8a9b7ea2d0edd0ca6678302255aed8f83e709c3254af6172d0fe60c6"
HELPER = "d57af4b17bd879944d55cf30806b0a66c80d462a41141b56e29ea8de54ea3d20"
MODEL = "e606d974f4aa8bcc2a4e07375353e41f1e4d31d7151f7d223380624c8116d8d6"
REFERENCE_PREDICTIONS = "465e5da1d5f0f593412cb1bfee519555f61ea8b9ebbd5797f2c4ce4887bf9a1a"
VERSION = "fixed96_context_original24_prediction_reference_v1"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def replace(s, old, new):
    if s.count(old) != 1:
        raise ValueError("One exact original source anchor required")
    return s.replace(old, new)


def split_source(source):
    names = {"FILES", "BINDING", "PARENT_SOURCE", "PERMISSION"}
    values, spans = {}, []
    lines = source.splitlines(keepends=True)
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in names:
                assert name not in values
                values[name] = ast.literal_eval(n.value)
                spans.append((n.lineno - 1, n.end_lineno, name))
    assert set(values) == names
    for start, end, name in sorted(spans, reverse=True):
        lines[start:end] = [name + " = __" + name + "_VALUE__\n"]
    return values, "".join(lines)


def compare_region_fields(a, b):
    """The inference cap must not alter any withheld descriptor or support prior."""
    import torch

    assert len(a) == len(b) == 2
    for left, right in zip(a, b, strict=True):
        for key in ("region_metadata", "targets", "observed"):
            x, y = left[key], right[key]
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.device == y.device == torch.device("cpu")
            assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
        x, y = left["supports"], right["supports"]
        assert type(x) is type(y) and x.region_count == y.region_count
        for key in ("indptr", "regions", "log_prior"):
            xx, yy = getattr(x, key), getattr(y, key)
            assert xx.shape == yy.shape and xx.dtype == yy.dtype
            assert torch.equal(xx.contiguous().view(torch.uint8), yy.contiguous().view(torch.uint8))


def build(original):
    import inspect

    assert sha(original) == PARENT
    values, worker = split_source(original)
    files, binding = values["FILES"], values["BINDING"]
    old_binding = copy.deepcopy(binding)
    assert binding["mode"] == "qualification" and binding["branch"] == "C100"
    assert len(binding["models"]) == 1 and binding["models"][0]["model"]["sha256"] == MODEL
    assert binding["models"][0]["architecture"]["context_cap"] == 24
    assert all(sha(body) == binding["module_sha256"][name] for name, body in files.items())
    helper = files["_predictive_descriptor_cache.py"]
    assert sha(helper) == HELPER and helper.count("context_cap=24") == 2
    helper = helper.replace("context_cap=24", "context_cap=96")
    helper += "\n" + inspect.getsource(compare_region_fields)
    helper = replace(
        helper,
        "    return actual, {",
        """    reference24_start = time.monotonic()
    reference24 = selected.assemble(batched.cache_descriptors_once(groups), side=320, halo=8, context_cap=24, classes=12)
    reference24_seconds = time.monotonic() - reference24_start
    compare_region_fields(reference24, actual)
    return actual, {""",
    )
    helper = replace(
        helper,
        '        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",\n    }',
        f"""        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",
        "context_coverage_version": {VERSION!r},
        "inference_context_cap": 96,
        "training_context_cap": 24,
        "complete_descriptor_support_fields_exact": True,
        "reference24_render_seconds": reference24_seconds,
    }}, reference24""",
    )
    files["_predictive_descriptor_cache.py"] = helper
    worker = replace(
        worker,
        '    assert BINDING["mode"] in ("qualification", "competition")',
        '    assert BINDING["mode"] == "qualification", "Diagnostic only; no entry"',
    )
    worker = replace(
        worker,
        "        contexts, preparation = prepare_contexts(",
        "        contexts, preparation, reference24 = prepare_contexts(",
    )
    worker = replace(
        worker,
        'sum(len(c["images"]) for c in contexts) <= 48',
        'sum(len(c["images"]) for c in contexts) <= 192',
    )
    worker = replace(
        worker, "        return contexts, result", "        return contexts, result, reference24"
    )
    worker = replace(
        worker,
        "        rows = [None] * len(uids)",
        "        rows = [None] * len(uids)\n        reference24_rows = [None] * len(uids)",
    )
    worker = replace(
        worker,
        "                contexts, input_stats = prepared[index] if prepared is not None else prepare(index)",
        "                contexts, input_stats, reference24 = prepared[index]",
    )
    worker = replace(
        worker,
        "                local.append((index, values, metric))",
        """                reference24_start = time.monotonic()
                prediction24 = predict(models[device], reference24, precision="fp16")
                torch.cuda.synchronize(device)
                values24 = prediction24.cpu().numpy().astype("<f4")
                contract.check_predictions([uids[index]], [values24])
                metric.update(
                    reference24_FP16_seconds=time.monotonic() - reference24_start,
                    context96_vs24_max_probability_change=float(np.max(np.abs(values - values24))),
                    context96_vs24_mean_probability_change=float(np.mean(np.abs(values - values24))),
                    reference24_prediction_sha256=hashlib.sha256(values24.tobytes()).hexdigest(),
                )
                local.append((index, values, metric, values24))""",
    )
    worker = replace(
        worker,
        "            for index, values, metric in device_results:",
        "            for index, values, metric, values24 in device_results:",
    )
    worker = replace(
        worker,
        "                    rows[index] = values",
        "                    rows[index] = values\n                    reference24_rows[index] = values24",
    )
    worker = replace(
        worker,
        "        contract.check_predictions(uids, rows)",
        f"""        contract.check_predictions(uids, rows)
        reference24_sha = hashlib.sha256(np.asarray(reference24_rows, dtype="<f4").tobytes()).hexdigest()
        assert reference24_sha == {REFERENCE_PREDICTIONS!r}, "Original24 predictions changed"
""",
    )
    worker = replace(
        worker,
        '            "trained_export_sha256": chosen["model"]["sha256"],',
        '            "trained_export_sha256": chosen["model"]["sha256"],\n            "reference24_prediction_sha256": reference24_sha,',
    )
    worker = replace(
        worker,
        '        "clinical_score": None,',
        f'        "clinical_score": None,\n        "context_coverage_version": {VERSION!r},\n        "inference_context_cap": 96,\n        "training_context_cap": 24,\n        "competition_admitted": False,',
    )
    worker = replace(
        worker,
        '"actual_trained_predictive_public_T4_qualification_accepted"',
        '"actual_fixed96_public_T4_diagnostic_complete"',
    )
    worker = replace(
        worker,
        'Path("/kaggle/working/predictive-" + BINDING["mode"] + "-accepted.json")',
        'Path("/kaggle/working/predictive-context96-diagnostic.json")',
    )
    binding.update(
        parent_qualification_source_sha256=PARENT,
        context_coverage_version=VERSION,
        training_context_cap=24,
        inference_context_cap=96,
        maximum_encoder_windows_per_study=192,
        competition_admitted=False,
    )
    binding["module_sha256"]["_predictive_descriptor_cache.py"] = sha(helper)
    binding["serving_helpers_sha256"]["_predictive_descriptor_cache.py"] = sha(helper)
    binding["serving_worker_sha256"] = sha(worker)
    assert binding["models"] == old_binding["models"]
    assert all(
        binding["module_sha256"][k] == v
        for k, v in old_binding["module_sha256"].items()
        if k != "_predictive_descriptor_cache.py"
    )
    for name, value in values.items():
        worker = replace(worker, "__" + name + "_VALUE__", repr(value))
    ast.parse(worker)
    ast.parse(helper)
    return worker, {
        "status": "fixed96_public_diagnostic_compiled",
        "source_sha256": sha(worker),
        "source_bytes": len(worker.encode()),
        "binding": binding,
        "GPU": True,
        "competition_entry": False,
        "new_Dataset": False,
    }
