"""Synthetic final-export/CSV/compiler checks; no actual trained GPU acceptance."""

import ast
import copy
import csv
import importlib.util
import json
import os
import threading
from collections import Counter, deque
from pathlib import Path

import pytest
import torch

from rsna.competitive.predictive.model import Predictive

D = Path("initiative/research/predictive-final-serving-20261020-v1")
spec = importlib.util.spec_from_file_location("predictive_final_compiler", D / "compile.py")
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)
c = compiler.contract
MODEL_DATASET = "godlydonuts/rsna-predictive-finals-synthetic"


@pytest.fixture
def entries(monkeypatch):
    parent = "def build_backbone(): pass\nclass RaptorClassifier: pass\ndef load_model(): pass\n"
    license_bytes = b"synthetic license, never actual attribution"
    monkeypatch.setattr(c, "PARENT_SOURCE", c.digest(parent))
    monkeypatch.setattr(c, "LICENSE", c.digest(license_bytes))
    source = {name: Path(name).read_text() for name in c.MODULES}
    source_hashes = {name: c.digest(text) for name, text in source.items()}
    own = json.dumps(source, sort_keys=True)
    result = []
    for arm in ("B", "C", "C0"):
        p = dict(
            arm=arm,
            epoch=6,
            completed_updates=2916,
            cohort_size=3887,
            recipe=c.recipe(arm),
            parent_weight_sha256=c.PARENT,
            parent_source_sha256=c.PARENT_SOURCE,
            source_revision=c.SOURCE_REVISION,
            targets_sha256=c.TARGETS,
            fitting_order_sha256=c.ROSTER,
            license_sha256=c.LICENSE,
            parameter_count=c.PARAMETERS[arm],
            model_schema_sha256=c.SCHEMAS[arm],
            publisher_sha256=c.PUBLISHER,
            original_publisher_core_sha256=c.PUBLISHER_CORE,
            original_numerical_source_git=c.ORIGINAL_SCIENCE,
            alias_regression_receipt_sha256=c.ALIAS_REGRESSION,
            accepted_renderer_sha256=[c.ORIGINAL_RENDERERS, c.CORRECTED_RENDERERS],
            renderer_provenance_counts={"original": 500, "corrected": 3387},
            model_source="public_licensed_MRI_parent_and_unchanged_RSNA_fitting_only",
            additional_Stanford_or_KneeCoT_or_teacher_probabilities_used=False,
            optimizer_RNG_private_paths_labels_IDs_exported=False,
            source_git="a" * 40,
            GPU_job="synthetic-test-only",
            code_sha256=source_hashes,
            renderer_sha256={
                n: source_hashes[n]
                for n in c.MODULES
                if n.endswith(("/inputs.py", "/cases.py", "/supports.py"))
            },
        )
        for key in (
            "binding_sha256",
            "context_index_sha256",
            "source_context_index_sha256",
            "runtime_receipt_sha256",
            "engineering_decision_sha256",
            "input_admission_sha256",
            "funding_sha256",
            "initial_shared_tensor_sha256",
            "checkpoint_sha256",
            "history_sha256",
            "sealed_sidecar_sha256",
        ):
            p[key] = "d" * 64
        text = json.dumps(p, sort_keys=True)
        files = {
            name: {"bytes": len(value), "sha256": c.digest(value)}
            for name, value in (
                ("provenance.json", text.encode()),
                ("own-source.json", own.encode()),
                ("source-permission.txt", license_bytes),
                ("manifest.json", b"synthetic"),
            )
        }
        files["model.pt"] = {
            "bytes": 293300000,
            "sha256": ("b" if arm == "B" else "c" if arm == "C" else "f") * 64,
        }
        pub = dict(
            status="actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified",
            objects=5,
            canonical_shared_publication_lock_used=True,
            files=files,
            revision="b" * 40,
            arm=arm,
            epoch=6,
            GPU_job=p["GPU_job"],
            source_git=p["source_git"],
            prefix=f"checkpoints/predictive-full3887-{p['source_git'][:12]}/{arm}/epoch-006",
        )
        result.append(
            dict(
                publication=pub,
                provenance_text=text,
                own_source_text=own,
                license_bytes=license_bytes,
            )
        )
    return parent, result


def rewrite_provenance(entry, mutate):
    p = json.loads(entry["provenance_text"])
    mutate(p)
    text = json.dumps(p, sort_keys=True)
    entry["provenance_text"] = text
    entry["publication"]["files"]["provenance.json"] = {
        "bytes": len(text.encode()),
        "sha256": c.digest(text),
    }


def accepted_qualification(chosen):
    row = dict(
        arm=chosen["arm"],
        candidate_sha256=c.digest(c.canonical(chosen)),
        trained_export_sha256=chosen["model"]["sha256"],
        precision_accepted=True,
        memory_accepted=True,
        FP16_FP32_tolerance=0.005,
        cross_device_FP16_tolerance=0.0001,
        cross_device_FP16_gap=0.00001,
        device_peak_bytes=[10000, 10000],
        host_peak_bytes=100000,
    )
    row["measurements"] = [
        dict(
            device=d,
            study_index=i,
            maximum_probability_gap=0.001,
            FP16_seconds=1.0,
            FP32_seconds=2.0,
            windows=48,
            regions=100,
        )
        for d, i in ((0, 0), (0, 2), (1, 1), (1, 0))
    ]
    return dict(
        status="actual_trained_predictive_public_T4_qualification_accepted",
        public_studies=3,
        GPU_names=["Tesla T4", "Tesla T4"],
        pinned_Docker=c.DOCKER,
        submission_csv_absent=True,
        hidden_runtime_acceptance=False,
        models=[row],
        serving_worker_sha256=c.digest((D / "worker.py.txt").read_text()),
        serving_contract_sha256=c.digest((D / "contract.py").read_text()),
        model_dataset_ref=MODEL_DATASET,
        serving_helpers_sha256={
            "_predictive_serving_mounts.py": c.digest((D / "mounts.py").read_text()),
            "_predictive_descriptor_cache.py": c.digest((D / "descriptor_cache.py").read_text()),
        },
    )


def test_combined_final_compiler_is_offline_and_preserves_exact_modules(entries):
    parent, candidates = entries
    source, meta = compiler.build(
        parent, candidates, mode="qualification", model_dataset_ref=MODEL_DATASET
    )
    assert meta["arms"] == ["B", "C", "C0"]
    assert meta["new_Dataset_notebook_or_entry_mutations"] == 0
    tree = ast.parse(source)
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "BINDING")
    }
    assert all(values["FILES"][name] == Path(name).read_text() for name in c.MODULES)
    assert values["BINDING"]["CPU_study_workers"] == 2
    assert values["BINDING"]["maximum_encoder_windows_per_study"] == 48
    assert values["BINDING"]["internet"] is False
    assert "CreateCodeSubmission" not in source and "hf_hub_download" not in source
    assert all(m["architecture"] == c.architecture(m["arm"]) for m in values["BINDING"]["models"])


@pytest.mark.parametrize(
    "problem",
    [
        "pilot",
        "steps",
        "Stanford",
        "teacher",
        "source",
        "license",
        "schema",
        "missing_source",
        "pub_sha",
        "private",
        "model_bytes",
        "duplicate",
    ],
)
def test_untrained_changed_or_unpreserved_inputs_rejected(entries, problem):
    parent, candidates = entries
    entry = candidates[0]
    if problem == "pilot":
        rewrite_provenance(entry, lambda p: p.update(cohort_size=12))
    elif problem == "steps":
        rewrite_provenance(entry, lambda p: p.update(completed_updates=24))
    elif problem in ("Stanford", "teacher"):
        rewrite_provenance(
            entry,
            lambda p: p.update(additional_Stanford_or_KneeCoT_or_teacher_probabilities_used=True),
        )
    elif problem == "source":
        rewrite_provenance(entry, lambda p: p["code_sha256"].update({c.MODULES[1]: "0" * 64}))
    elif problem == "license":
        entry["license_bytes"] += b"wrong"
    elif problem == "schema":
        rewrite_provenance(entry, lambda p: p.update(model_schema_sha256="0" * 64))
    elif problem == "missing_source":
        entry["own_source_text"] = "{}"
    elif problem == "pub_sha":
        entry["publication"]["files"]["provenance.json"]["sha256"] = "0" * 64
    elif problem == "private":
        entry["publication"]["canonical_shared_publication_lock_used"] = False
    elif problem == "model_bytes":
        entry["publication"]["files"]["model.pt"]["bytes"] = 12
    else:
        candidates = [entry, entry]
    with pytest.raises(ValueError):
        compiler.build(parent, candidates, mode="qualification", model_dataset_ref=MODEL_DATASET)


def test_competition_requires_exact_actual_trained_qualification(entries):
    parent, all_entries = entries
    entry = all_entries[1]
    chosen = c.candidate(**entry)
    receipt = accepted_qualification(chosen)
    source, meta = compiler.build(
        parent, [entry], mode="competition", qualification=receipt, model_dataset_ref=MODEL_DATASET
    )
    assert meta["arms"] == ["C"] and "__BINDING_VALUE__" not in source
    for change in (
        "missing",
        "different_model",
        "failed_precision",
        "missing_measure",
        "bad_gap",
        "worker",
        "helper",
        "dataset",
    ):
        wrong = copy.deepcopy(receipt)
        if change == "missing":
            wrong = None
        elif change == "different_model":
            wrong["models"][0]["trained_export_sha256"] = "0" * 64
        elif change == "failed_precision":
            wrong["models"][0]["precision_accepted"] = False
        elif change == "missing_measure":
            wrong["models"][0]["measurements"].pop()
        elif change == "bad_gap":
            wrong["models"][0]["measurements"][0]["maximum_probability_gap"] = 0.01
        elif change == "helper":
            wrong["serving_helpers_sha256"]["_predictive_serving_mounts.py"] = "0" * 64
        elif change == "dataset":
            wrong["model_dataset_ref"] = "someone/another-dataset"
        else:
            wrong["serving_worker_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            compiler.build(
                parent,
                [entry],
                mode="competition",
                qualification=wrong,
                model_dataset_ref=MODEL_DATASET,
            )


def tiny_export():
    encoder = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(12, 4), torch.nn.BatchNorm1d(4)
    )
    model = Predictive(encoder, 4, hidden=128)
    state = copy.deepcopy(model.state_dict())
    # Simulate actual learned parameters; loading must replace fresh initialization.
    state["shift.bias"].fill_(0.1)
    state["encoder.2.running_mean"].fill_(0.25)
    state["encoder.2.running_var"].fill_(1.5)
    state["encoder.2.num_batches_tracked"].fill_(17)
    provenance = {
        "model_schema_sha256": c.digest(
            c.canonical(
                {n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in state.items()}
            )
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    chosen = {"architecture": c.architecture("C"), "provenance": provenance}
    export = {"model": state, **chosen}
    return model, chosen, export


def test_strict_learned_full_graph_load_replaces_all_tensors():
    model, chosen, export = tiny_export()
    c.validate_export(export, chosen, model)
    assert torch.equal(model.shift.bias, export["model"]["shift.bias"])
    assert model.encoder[2].num_batches_tracked.item() == 17
    assert torch.equal(model.encoder[2].running_mean, export["model"]["encoder.2.running_mean"])
    assert torch.equal(model.encoder[2].running_var, export["model"]["encoder.2.running_var"])
    assert not model.training


@pytest.mark.parametrize(
    "problem", ["missing", "extra", "dtype", "shape", "nan", "old", "architecture", "provenance"]
)
def test_old_partial_or_nonfinite_model_rejected(problem):
    model, chosen, export = tiny_export()
    if problem == "missing":
        export["model"].pop("shift.bias")
    elif problem == "extra":
        export["model"]["clsW"] = torch.zeros(1)
    elif problem == "dtype":
        export["model"]["shift.bias"] = export["model"]["shift.bias"].half()
    elif problem == "shape":
        export["model"]["shift.bias"] = export["model"]["shift.bias"][:1]
    elif problem == "nan":
        export["model"]["shift.bias"][0] = float("nan")
    elif problem == "old":
        export["optimizer"] = {}
    elif problem == "architecture":
        export["architecture"] = {"arm": "B"}
    else:
        export["provenance"] = {}
    with pytest.raises(ValueError):
        c.validate_export(export, chosen, model)


def worker_function(name):
    tree = ast.parse((D / "worker.py.txt").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(Path=Path, csv=csv, os=os)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "isolated_worker_function", "exec"),
        ns,
    )
    return ns[name]


def test_CSV_exact_order_thirteen_columns_and_no_overwrite(tmp_path):
    write = worker_function("write_submission")
    path = tmp_path / "submission.csv"
    uids = ["1.2.9", "1.2.1", "1.2.4"]
    rows = [[(i + 1) / 10] * 12 for i in range(3)]
    write(path, uids, rows, c)
    with path.open() as handle:
        data = list(csv.reader(handle))
    assert data[0] == c.COLUMNS and [row[0] for row in data[1:]] == uids
    assert all(len(row) == 13 for row in data)
    with pytest.raises(AssertionError):
        write(path, uids, rows, c)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "nan", "range", "columns", "uid"])
def test_CSV_invalid_predictions_never_publish(tmp_path, problem):
    uids, rows = ["1.2.3", "1.2.4"], [[0.2] * 12, [0.3] * 12]
    if problem == "duplicate":
        uids[1] = uids[0]
    elif problem == "missing":
        rows.pop()
    elif problem == "nan":
        rows[0][0] = float("nan")
    elif problem == "range":
        rows[0][0] = 1.1
    elif problem == "columns":
        rows[0].pop()
    else:
        uids[0] = "../other"
    path = tmp_path / "submission.csv"
    with pytest.raises(ValueError):
        worker_function("write_submission")(path, uids, rows, c)
    assert not path.exists() and not path.with_name("submission.csv.tmp").exists()


def test_worker_preserves_original_fixed_encoder_and_two_context_graph():
    tree = ast.parse((D / "worker.py.txt").read_text())
    probe = ast.parse(
        Path(
            "initiative/research/predictive-T4-throughput-20261020-v2/benchmark.py.txt"
        ).read_text()
    )

    def find(t):
        return next(
            n for n in ast.walk(t) if isinstance(n, ast.ClassDef) and n.name == "FixedEncoder"
        )

    assert ast.dump(find(tree)) == ast.dump(find(probe))
    pools = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "ThreadPoolExecutor"
    ]
    assert pools and all(ast.literal_eval(n.keywords[0].value) == 2 for n in pools)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "assemble"
    ]
    assert len(calls) == 1 and {
        k.arg: ast.literal_eval(k.value) for k in calls[0].keywords
    } == dict(side=320, halo=8, context_cap=24, classes=12)
    assert (
        "load_state_dict" not in (D / "worker.py.txt").read_text()
    )  # Strict contract owns loading.


def test_hidden_scale_progress_is_bounded_with_linear_append_history(tmp_path, monkeypatch):
    tree = ast.parse((D / "worker.py.txt").read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("atomic_json", "stage")
    ]
    namespace = dict(
        Path=Path,
        json=json,
        os=os,
        time=type("FixedClock", (), {"monotonic": staticmethod(lambda: 42.0)}),
        START=0.0,
        LOCK=threading.Lock(),
        ROOT=tmp_path,
        EVENTS=deque(maxlen=32),
        EVENT_COUNTS=Counter(),
        BINDING={"mode": "competition"},
        print=lambda *a, **kw: None,
    )
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "bounded_progress", "exec"), namespace
    )
    write = namespace["stage"]
    for index in range(200):
        write("trained_study_measured", study_index=index)
    original_size = (tmp_path / "events.jsonl").stat().st_size
    for index in range(200, 400):
        write("trained_study_measured", study_index=index)
    progress = json.loads((tmp_path / "progress.json").read_text())
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == progress["total_events"] == 400
    assert progress["event_counts"] == {"trained_study_measured": 400}
    assert len(progress["latest_events"]) == 32
    assert progress["latest_events"][-1]["study_index"] == 399
    assert json.loads(lines[0])["study_index"] == 0
    assert json.loads(lines[-1]) == progress["latest_events"][-1]
    assert (tmp_path / "events.jsonl").stat().st_size < original_size * 2.1
    assert (tmp_path / "progress.json").stat().st_size < 8192


@pytest.mark.parametrize("mode,exact_public", [("qualification", True), ("competition", False)])
def test_watchdog_never_limits_qualification_or_a_different_hidden_roster(mode, exact_public):
    namespace, timers = watchdog_namespace(mode, exact_public)
    assert namespace["start_public_watchdog"](["1.2.3"], c) is None
    assert timers == []


def watchdog_namespace(mode, exact_public=True):
    tree = ast.parse((D / "worker.py.txt").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "start_public_watchdog"
    )
    timers = []

    class FakeTimer:
        def __init__(self, interval, callback):
            self.interval, self.callback, self.started = interval, callback, False
            timers.append(self)

        def start(self):
            self.started = True

    namespace = dict(
        BINDING=dict(
            mode=mode,
            public_roster_sha256=c.digest(c.canonical(["1.2.3"])) if exact_public else "0" * 64,
            public_watchdog_seconds=600,
        ),
        START=10,
        time=type("Clock", (), {"monotonic": staticmethod(lambda: 30)}),
        threading=type("Threads", (), {"Timer": FakeTimer}),
        os=type("OS", (), {"_exit": staticmethod(lambda code: code)}),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "watchdog", "exec"), namespace)
    return namespace, timers


def test_exact_public_watchdog_counts_elapsed_startup_and_cannot_emit_predictions():
    namespace, timers = watchdog_namespace("competition")
    timer = namespace["start_public_watchdog"](["1.2.3"], c)
    assert timer is timers[0] and timer.started and timer.daemon
    assert timer.interval == 580 and timer.callback() == 124
    source = (D / "worker.py.txt").read_text()
    assert source.index("public_timer = start_public_watchdog") < source.index(
        'parent, resolution = namespace["load_model"]'
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("alias_regression_receipt_sha256", "f" * 64),
        ("original_numerical_source_git", "f" * 40),
        ("accepted_renderer_sha256", [c.CORRECTED_RENDERERS]),
        ("renderer_provenance_counts", {"original": 500, "corrected": 3386}),
        ("renderer_provenance_counts", {"original": True, "corrected": 3886}),
        ("publisher_sha256", "6215f45748fdc4685565a94f31da7fc51f7ad0fa325192d004b362645bbf8d87"),
    ],
)
def test_final_export_cannot_erase_alias_recovery_lineage(entries, field, value):
    _, values = entries
    entry = values[0]
    rewrite_provenance(entry, lambda p: p.update({field: value}))
    with pytest.raises(ValueError):
        c.candidate(**entry)


def test_exact_alias_successor_publisher_is_pinned():
    publisher = Path("initiative/research/predictive-full-fitting-20261020-v2/publish_epoch.py.txt")
    assert c.digest(publisher.read_bytes()) == c.PUBLISHER
