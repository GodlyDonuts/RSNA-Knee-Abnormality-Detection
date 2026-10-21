import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("CPU_only_account_test", HERE / "account.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def compiled():
    return {
        "source_sha256": a.sha(b"own source"),
        "GPU": False,
        "competition_entry": False,
        "binding": {"pinned_Docker": "exact-original-image"},
    }


def test_request_has_no_GPU_dataset_or_submission():
    r = a.request_for("own source", compiled())
    assert r["enableGpu"] is False and r["enableTpu"] is False
    assert r["isPrivate"] is True and r["enableInternet"] is False
    assert r["datasetDataSources"] == r["kernelDataSources"] == []
    assert "machineShape" not in r
    assert r["sessionTimeoutSeconds"] == 1200
    with pytest.raises(AssertionError):
        a.request_for("changed source", compiled())


def receipt():
    rows = [
        {"index": i, "context_sha256": "exact" + str(i), "stats": {"frames": i + 1}}
        for i in range(3)
    ]
    return {
        "status": "actual_original_MRI_CPU_preparation_comparison_complete",
        "binding": compiled()["binding"],
        "studies": 3,
        "GPU_count": 0,
        "model_or_label_reads": False,
        "submission_csv_created": False,
        "clinical_score": None,
        "elapsed_seconds": 500,
        "peak_process_tree_RSS_bytes": 5 * 2**30,
        "trials": [
            {
                "kind": kind,
                "workers": n,
                "complete_original_input_equality": True,
                "wall_seconds": 50,
                "studies": copy.deepcopy(rows),
            }
            for kind, n in [
                ("thread", 2),
                ("process", 2),
                ("process", 3),
                ("thread", 2),
                ("process", 2),
            ]
        ],
    }


def test_complete_original_tensor_and_runtime_evidence_required():
    r = receipt()
    assert a.accept(r, compiled()) == r
    for key, value in [
        ("GPU_count", 1),
        ("elapsed_seconds", 901),
        ("peak_process_tree_RSS_bytes", 25 * 2**30),
        ("model_or_label_reads", True),
        ("submission_csv_created", True),
    ]:
        bad = copy.deepcopy(r)
        bad[key] = value
        with pytest.raises(AssertionError):
            a.accept(bad, compiled())
    for key, value in [("context_sha256", "different"), ("stats", {"frames": 999})]:
        bad = copy.deepcopy(r)
        bad["trials"][1]["studies"][0][key] = value
        with pytest.raises(AssertionError):
            a.accept(bad, compiled())
