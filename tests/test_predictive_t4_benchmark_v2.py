"""Receipt retention and resource admission for the distinct local T4 v2 control."""

import ast
import hashlib
import itertools
import json
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-T4-throughput-20261020-v2"
TEMPLATE = CONTROL / "benchmark.py.txt"


def functions(path, names, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_atomic_json_retains_previous_valid_receipt_on_invalid_value(tmp_path):
    ns = functions(TEMPLATE, {"atomic_json"}, {"Path": Path, "json": json, "os": os})
    output = tmp_path / "receipt.json"
    ns["atomic_json"](output, {"elapsed": 600.7})
    original = output.read_bytes()
    with pytest.raises(ValueError):
        ns["atomic_json"](output, {"elapsed": float("nan")})
    assert output.read_bytes() == original and not list(tmp_path.glob("*.tmp"))


def test_progress_is_atomic_serialized_and_flushed_for_both_devices(tmp_path, capsys):
    root = tmp_path / "private"
    root.mkdir()
    ticks = itertools.count()

    def mapped_path(value):
        value = str(value)
        return tmp_path / Path(value).name if value.startswith("/kaggle/working/") else Path(value)

    ns = functions(
        TEMPLATE,
        {"atomic_json", "stage"},
        {
            "Path": mapped_path,
            "json": json,
            "os": os,
            "time": types.SimpleNamespace(monotonic=lambda: next(ticks)),
            "start": 0,
            "root": root,
            "progress_lock": threading.Lock(),
            "progress": {"events": [], "competition_entry": False},
        },
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: ns["stage"]("device_timed", device=i, seconds=31.2), range(2)))
    first = json.loads((root / "progress.json").read_text())
    second = json.loads((tmp_path / "predictive-T4-progress-v2.json").read_text())
    assert first == second
    assert {e["metrics"]["device"] for e in first["events"]} == {0, 1}
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(ValueError):
        ns["stage"]("bad", seconds=float("nan"))
    ns["stage"]("device_failed", device=0, error_type="ValueError")
    assert len(json.loads((root / "progress.json").read_text())["events"]) == 3


@pytest.mark.parametrize(
    ("elapsed", "complete", "expected"),
    [
        (300.0, True, "measured_within_original_runtime_target"),
        (600.7, True, "measured_slow"),
        (1199.0, True, "measured_slow"),
        (300.0, False, "measured_incomplete_or_failed_checks"),
    ],
)
def test_slow_or_incomplete_results_are_classified_without_losing_measurements(
    elapsed, complete, expected
):
    ns = functions(TEMPLATE, {"classify"}, {})
    result = ns["classify"](
        {
            "completed_device_count": 2 if complete else 1,
            "device_errors": [] if complete else [{"device": 1, "error_type": "RuntimeError"}],
            "host22GiB_device13GiB_memory_checks": True,
            "submission_csv_absent": True,
            "elapsed_seconds": elapsed,
        }
    )
    assert result["status"] == expected
    assert result["measured_elapsed_seconds"] == elapsed
    assert not result["trained_model_acceptance_or_accuracy"]
    assert not result["hidden_size_runtime_acceptance"]


def test_timed_device_metrics_are_emitted_before_expensive_precision_checks():
    ticks = itertools.count()
    events = []
    cases = [
        [{"images": torch.zeros(2, 3, 16, 16)}, {"images": torch.zeros(2, 3, 16, 16)}]
        for _ in range(3)
    ]
    fake_cuda = types.SimpleNamespace(
        set_device=lambda device: None,
        synchronize=lambda device: None,
        reset_peak_memory_stats=lambda device: None,
        max_memory_allocated=lambda device: 1000,
        max_memory_reserved=lambda device: 2000,
        get_device_name=lambda device: "Tesla T4",
    )
    ns = functions(
        TEMPLATE,
        {"run"},
        {
            "torch": types.SimpleNamespace(cuda=fake_cuda, inference_mode=torch.inference_mode),
            "models": [object(), object()],
            "cases": cases,
            "np": np,
            "hashlib": hashlib,
            "time": types.SimpleNamespace(monotonic=lambda: next(ticks)),
            "predict": lambda model, case, precision: torch.full((12,), 0.5),
            "stage": lambda name, **metrics: events.append((name, metrics)),
            "nonzero_shift_precision_check": lambda model, case: {"synthetic": True},
        },
    )
    result = ns["run"](1)
    names = [name for name, _ in events]
    assert (
        names.index("device_FP16_timed_measurement_complete")
        < names.index("device_zero_shift_precision_complete")
        < names.index("device_nonzero_shift_precision_complete")
    )
    measured = dict(events)["device_FP16_timed_measurement_complete"]
    assert measured["studies"] == 3 and measured["encoder_windows"] == 12
    assert measured["seconds"] > 0 and len(measured["prediction_sha256"]) == 64
    assert result["device"] == 1 and result["finite_predictions"]


@pytest.mark.parametrize(
    ("state", "needed"),
    [("RUNNING", 4800), ("UNREPORTED", 4800), ("CANCEL_ACKNOWLEDGED", 4800), ("COMPLETE", 4200)],
)
def test_fresh_source_zero_status_controls_additional_budget(state, needed):
    requests = []

    class Remote:
        def kernel(self, method, slug, version):
            requests.append((method, slug, version))
            return {"metadata": {"id": 139786921}} if method == "GetKernel" else {"status": state}

    ns = functions(
        CONTROL / "cloud.py.txt", {"source_zero_observation", "required_free_seconds"}, {}
    )
    observed = ns["source_zero_observation"](Remote())
    assert ns["required_free_seconds"](observed) == needed
    assert [x[0] for x in requests] == ["GetKernel", "GetKernelSessionStatus"]
    assert all(x[1:] == ("godlydonuts/rsna-zero-full-e3-a5c5b1d7f43c", 1) for x in requests)


def test_observed_capacity_jobs_are_not_registered_as_new_probe_jobs():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "capacity_for_predictive_v2", ROOT / "scripts/kaggle/gpu_capacity.py"
    )
    capacity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capacity)
    ns = functions(CONTROL / "cloud.py.txt", {"capacity_observation"}, {})
    observed = ns["capacity_observation"](
        {
            "actual_active_registered_sessions": [
                {
                    "slug": "godlydonuts/original-source-zero",
                    "kernel_id": 100,
                    "version": 1,
                }
            ],
        }
    )
    row = {
        "state": "predictive_T4_v2_pushed",
        "slug": "godlydonuts/new-v2",
        "kernel_id": 200,
        "version": 1,
        "capacity": observed,
    }

    class Database:
        def execute(self, query):
            return [("licensed-importance-full-fit:predictive-T4-v2:test", json.dumps(row))]

    requests = []

    class Remote:
        username = "godlydonuts"

        def kernel(self, method, slug, version):
            requests.append((method, slug, version))
            return (
                {"metadata": {"enableGpu": True}}
                if method == "GetKernel"
                else {"status": "RUNNING"}
            )

    result = capacity.admission(Database(), Remote(), caller="sampling", now=0)
    assert len(result["actual_active_registered_sessions"]) == 1
    assert all(slug == "godlydonuts/new-v2" for _, slug, _ in requests)


def test_precision_protocol_is_identical_to_original_tested_protocol():
    original = ROOT / "initiative/research/predictive-T4-throughput-20261020-v1/benchmark.py.txt"
    bodies = [
        next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "nonzero_shift_precision_check"
        )
        for path in (original, TEMPLATE)
    ]
    assert ast.dump(bodies[0]) == ast.dump(bodies[1])


def test_receipts_precede_duration_classification_and_version_resources_are_distinct():
    source = TEMPLATE.read_text()
    tree = ast.parse(source)
    receipt_writes = [
        n.lineno
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "atomic_json"
        and any(isinstance(a, ast.Name) and a.id == "receipt" for a in n.value.args)
    ]
    classification = next(
        n.lineno
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "classification" for t in n.targets)
    )
    assert len(receipt_writes) == 2 and max(receipt_writes) < classification
    assert not any(isinstance(n, ast.Assert) and "550" in ast.unparse(n) for n in ast.walk(tree))
    assert "predictive-T4-engineering-v2" in source
    assert "predictive-T4-benchmark-v2.json" in source
    stages = {
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "stage"
        and n.args
        and isinstance(n.args[0], ast.Constant)
    }
    assert {
        "imports_complete",
        "model_load_complete",
        "study_decode_complete",
        "study_render_complete",
        "device_FP16_timed_measurement_complete",
        "device_zero_shift_precision_complete",
        "device_nonzero_shift_precision_complete",
    } <= stages
    cloud = (CONTROL / "cloud.py.txt").read_text()
    assert '"sessionTimeoutSeconds": 1200' in cloud
    assert "licensed-importance-full-fit:predictive-T4-v2:" in cloud
    assert "godlydonuts/rsna-predictive-t4-v2-" in cloud
    assert "12431568" in cloud and '"enableInternet": False' in cloud
    assert "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f" in cloud
    for path in (TEMPLATE, CONTROL / "cloud.py.txt", CONTROL / "run_once.py.txt"):
        compile(path.read_text(), str(path), "exec")
