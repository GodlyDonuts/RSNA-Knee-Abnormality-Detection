"""Derive a distinct public-output coordinator from the original diagnostic.

This coordinator cannot submit to the competition. It preserves the canonical
claim, durable intents, immutable version checks and protected Colab budget.
"""

import ast
import inspect


def accept_public(remote, slug, compiled, c, csv):
    r = remote.output_json(slug, 1, "predictive-competition-accepted.json")
    assert r["status"] == "actual_predictive_exact_test_roster_submission_created"
    assert r["inference_context_cap"] == 96 and r["training_context_cap"] == 24
    assert r["context_coverage_version"] == c.D.VERSION
    assert r["actual_test_studies"] == 3 and r["submission_csv_absent"] is False
    assert r["public_watchdog_started"] is True and r["public_watchdog_seconds"] == 600
    assert 0 < r["elapsed_seconds"] < 600 and len(r["models"]) == 1
    model = r["models"][0]
    assert model["trained_export_sha256"] == c.D.MODEL
    assert model["prediction_sha256"] == compiled["expected_public_prediction_sha256"]
    assert model["memory_accepted"] is True
    for key in (
        "serving_worker_sha256",
        "serving_contract_sha256",
        "serving_helpers_sha256",
        "model_dataset_ref",
    ):
        assert r[key] == compiled["binding"][key]
    assert all(
        x["inference_context_cap"] == 96
        and x["decoder_route"] == "original_exact"
        and x["prediction_precision"] == "fp16"
        for x in model["measurements"]
    )
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
    return {**r, "public_CSV_verification": checked, "competition_entry": False}


def derive(original):
    def change(a, b):
        nonlocal original
        assert original.count(a) == 1, "Exact frozen coordinator anchor required"
        original = original.replace(a, b)

    change(
        'ROOT = Path("/tmp/rsna-predictive-context96-T4-20261021-v1")',
        'ROOT = Path("/tmp/rsna-predictive-context96-ready-20261021-v1")',
    )
    change(
        "other_node_bytes=518 * 2**30 + 64 * 2**20,", "other_node_bytes=518 * 2**30 + 96 * 2**20,"
    )
    change("other_shared_bytes=123 * 2**20,", "other_shared_bytes=124 * 2**20,")
    change("(518 + 20) * 2**30 + 96 * 2**20", "(518 + 20) * 2**30 + 128 * 2**20")
    change("20 * 2**30 + 124 * 2**20", "20 * 2**30 + 125 * 2**20")
    change(
        '("compile.py", "workflow.py", "components.py", "gpu_capacity.py")',
        '("compile.py", "candidate.py", "output_csv.py", "workflow.py", "components.py", "gpu_capacity.py")',
    )
    change(
        'c = load("fixed96_compiler", ROOT / "compile.py")',
        'c = load("fixed96_ready_compiler", ROOT / "candidate.py")',
    )
    change(
        'c.build(\n            (ORIGINAL / "compiled-qualification/source.private.py").read_text()\n        )',
        'c.build((ORIGINAL / "compiled-C/source.private.py").read_text(), json.loads(Path("/tmp/rsna-predictive-context96-T4-20261021-v1/actual-complete.json").read_text()))',
    )
    change(
        '"godlydonuts/rsna-predictive-context96-"', '"godlydonuts/rsna-predictive-context96-ready-"'
    )
    change("sessionTimeoutSeconds=600,", "sessionTimeoutSeconds=32400,")
    change(
        '"licensed-parent-full-epoch-serving:fixed96-diagnostic-v1:"',
        '"licensed-parent-full-epoch-serving:fixed96-ready-v1:"',
    )
    change("== c.PARENT", "== c.D.PARENT")
    start = original.index('                result["receipt"] = accept(')
    end = original.index('                if row["state"] != "fixed96_complete":', start)
    original = (
        original[:start]
        + '                result["receipt"] = accept_public(remote, slug, compiled, c, load("fixed96_public_csv", ROOT / "output_csv.py"))\n'
        + original[end:]
    )
    original += "\n" + inspect.getsource(accept_public)
    ast.parse(original)
    return original
