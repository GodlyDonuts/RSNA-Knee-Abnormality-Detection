"""Own bounded public test; no competition-submission method."""

import ast
import inspect


def accept_public(remote, slug, compiled, c, csv):
    import hashlib
    import io

    import numpy as np

    r = remote.output_json(slug, 1, "predictive-competition-accepted.json")
    assert r["status"] == "actual_fourcore_fixed96_public_comparison_complete"
    assert r["thread_comparison_version"] == c.VERSION and r["threads_per_lane"] == 2
    assert r["inference_context_cap"] == 96 and r["training_context_cap"] == 24
    assert r["actual_test_studies"] == 3 and r["submission_csv_absent"] is False
    assert r["GPU_names"] == ["Tesla T4", "Tesla T4"]
    assert r["public_watchdog_started"] is True and r["public_watchdog_seconds"] == 480
    assert 0 < r["elapsed_seconds"] < 480 and len(r["models"]) == 1
    m = r["models"][0]
    assert m["trained_export_sha256"] == c.MODEL
    assert m["memory_accepted"] is True and m["precision_accepted"] is True
    assert m["cross_device_FP16_gap"] <= 1e-4
    for key in (
        "serving_worker_sha256",
        "serving_contract_sha256",
        "serving_helpers_sha256",
        "model_dataset_ref",
    ):
        assert r[key] == compiled["binding"][key]
    records = m["measurements"]
    assert {(x["study_index"], x["device"]) for x in records} == {(0, 0), (1, 1), (2, 0), (0, 1)}
    verified = []
    per_study = {}
    for x in records:
        assert x["candidate_prepared_before_reference"] is True
        assert x["threads_per_lane"] == 2 and x["thread_reference_count"] == 1
        assert x["thread1_reference_other_fields_exact"] is True
        assert 0 <= x["thread1_reference_image_max_absolute_difference"] <= 1e-6
        assert x["maximum_probability_gap"] <= 0.005
        assert x["thread_FP32_probability_gap"] <= 1e-5 and x["thread_FP16_probability_gap"] <= 1e-4
        assert x["inference_context_cap"] == 96 and x["decoder_route"] == "original_exact"
        assert x["prediction_precision"] == "fp16"
        expected = f"thread-paired-study-{x['study_index']}-gpu-{x['device']}.npz"
        assert x["paired_prediction_file"] == expected and x["paired_prediction_file_bytes"] < 16384
        raw = csv.download_named(remote, slug, 1, expected)
        assert len(raw) == x["paired_prediction_file_bytes"]
        assert hashlib.sha256(raw).hexdigest() == x["paired_prediction_file_sha256"]
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "candidate_FP16",
                "reference_FP16",
                "candidate_FP32",
                "reference_FP32",
            }
            for a in arrays.values():
                assert a.shape == (12,) and a.dtype == np.dtype("<f4")
                assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
            gap32 = float(np.max(np.abs(arrays["candidate_FP32"] - arrays["reference_FP32"])))
            gap16 = float(np.max(np.abs(arrays["candidate_FP16"] - arrays["reference_FP16"])))
            assert gap32 <= 1e-5 and gap16 <= 1e-4
            assert (
                gap32 == x["thread_FP32_probability_gap"]
                and gap16 == x["thread_FP16_probability_gap"]
            )
        verified.append(
            dict(
                file=expected,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                independent_array_comparison=True,
            )
        )
        seconds = x["decode_seconds"] + x["render_seconds"] + x["FP16_seconds"]
        per_study[x["study_index"]] = max(per_study.get(x["study_index"], 0), seconds)
    columns = [
        "StudyInstanceUID",
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]
    checked = csv.verify_csv(
        csv.download_csv(remote, slug, 1), columns, compiled["binding"]["public_roster_sha256"]
    )
    assert checked["CSV_sha256"] == r["CSV_sha256"] and checked["bytes"] == r["CSV_bytes"]
    mean_plan = 600 + 1800 + 650 * float(np.mean(list(per_study.values())))
    max_plan = 600 + 1800 + 650 * max(per_study.values())
    return {
        **r,
        "public_CSV_verification": checked,
        "private_paired_objects_verified": verified,
        "runtime_planning": {
            "studies_assumed": 1300,
            "lanes": 2,
            "setup_seconds": 600,
            "tail_seconds": 1800,
            "mean_scenario_seconds": mean_plan,
            "max_scenario_seconds": max_plan,
            "mean_below_9_hours": mean_plan < 32400,
            "hidden_runtime_guaranteed": False,
        },
        "competition_entry": False,
        "raw_predictions_exported": False,
    }


def derive(original):
    def change(a, b):
        nonlocal original
        assert original.count(a) == 1, "Exact frozen coordinator anchor required"
        original = original.replace(a, b)

    change(
        'ROOT = Path("/tmp/rsna-predictive-context96-T4-20261021-v1")',
        'ROOT = Path("/tmp/rsna-context96-fourcore-20261021-v1")',
    )
    change(
        "other_node_bytes=518 * 2**30 + 64 * 2**20,", "other_node_bytes=518 * 2**30 + 144 * 2**20,"
    )
    change("other_shared_bytes=123 * 2**20,", "other_shared_bytes=125 * 2**20,")
    change("(518 + 20) * 2**30 + 96 * 2**20", "(518 + 20) * 2**30 + 176 * 2**20")
    change("20 * 2**30 + 124 * 2**20", "20 * 2**30 + 126 * 2**20")
    change(
        '("compile.py", "workflow.py", "components.py", "gpu_capacity.py")',
        '("compile.py", "original_compile.py", "thread_compare.py", "output_csv.py", "workflow.py", "components.py", "gpu_capacity.py")',
    )
    change(
        'c.build(\n            (ORIGINAL / "compiled-qualification/source.private.py").read_text()\n        )',
        'c.build(Path("/tmp/rsna-predictive-context96-ready-20261021-v1/source.private.py").read_text())',
    )
    change('"godlydonuts/rsna-predictive-context96-"', '"godlydonuts/rsna-context96-fourcore-"')
    change("sessionTimeoutSeconds=600,", "sessionTimeoutSeconds=32400,")
    change(
        '"licensed-parent-full-epoch-serving:fixed96-diagnostic-v1:"',
        '"licensed-parent-full-epoch-serving:context96-fourcore-v1:"',
    )
    change("free >= 1920 + 600 + 120", "free >= 1920 + 480 + 120")
    change("own_timeout_seconds=600,", "own_timeout_seconds=480,")
    start = original.index('                result["receipt"] = accept(')
    end = original.index('                if row["state"] != "fixed96_complete":', start)
    original = (
        original[:start]
        + '                result["receipt"] = accept_public(remote, slug, compiled, c, load("fourcore_public_outputs", ROOT / "output_csv.py"))\n'
        + original[end:]
    )
    original += "\n" + inspect.getsource(accept_public)
    ast.parse(original)
    return original


def output_module(original):
    """Reuse the bounded original signed-output transport for exact paired files."""
    tree = ast.parse(original)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "download_csv")
    source = ast.get_source_segment(original, node)
    source = source.replace(
        "def download_csv(remote, slug: str, version: int)",
        "def download_named(remote, slug: str, version: int, filename: str)",
        1,
    )
    assert source.count('"filePath": "submission.csv"') == 1
    source = source.replace('"filePath": "submission.csv"', '"filePath": filename')
    source = source.replace(
        '    owner, kernel = slug.split("/")',
        '    assert filename.startswith("thread-paired-study-") and filename.endswith(".npz") and "/" not in filename\n    owner, kernel = slug.split("/")',
    )
    result = original + "\n" + source
    ast.parse(result)
    return result
