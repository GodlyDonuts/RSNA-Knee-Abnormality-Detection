"""Synthetic final admission and actual generated serving-code tests; no model fitting."""

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BUILD = load("Colab_serving_test_builder", HERE / "compile.py")
C = BUILD.C
OLD = load(
    "Colab_fixture_original_helpers",
    HERE.parent / "predictive-density100-serving-20261020-v1/test_contract.py",
)


def encode(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n"


def fileproof(b):
    return {"bytes": len(b), "sha256": C.digest(b)}


@pytest.fixture
def case():
    adapter = C.Contract(BUILD.originals()["contract.py"], module_sha256=C.MODULE_SHA256)
    base = adapter._base
    license_bytes = b"synthetic source permission; no private data"
    base.LICENSE = C.digest(license_bytes)
    b = dict(
        branch=C.BRANCH,
        source_git=C.SCIENCE,
        base_science_git=C.BASE,
        original_binding_sha256=C.ORIGINAL_BINDING,
        input_manifest_sha256=C.MANIFEST,
        input_private_revision="Kaggle:12754716:v1+original_HF_revisions",
        fitting_order_sha256=base.ROSTER,
        recipe={**base.recipe("C"), "density_weight": 100.0, "seed": 20261021},
        parent_weight_sha256=base.PARENT,
        parent_source_sha256=base.PARENT_SOURCE,
        targets_sha256=base.TARGETS,
        license_sha256=base.LICENSE,
        torch="2.11.0+cu130",
        timm="1.0.29",
        cuda="13.0",
        GPU="NVIDIA A100-SXM4-40GB",
        storage_control_git="HEAD",
        storage_control_sha256=C.CODE,
        new_seed_and_runtime_not_strict_causal_comparison=True,
        protected_labels_or_external_source_data_used=False,
        clinical_accuracy_claim=False,
        initial_complete_state_sha256="a" * 64,
        numerical_code_sha256={"code/" + n: h for n, h in C.MODULE_SHA256.items()},
        started_UTC="2026-10-21T11:20:00+00:00",
    )
    binding_text = encode(b)
    terminal_text = encode(
        dict(
            status="actual_full_C100_seed20261021_complete",
            binding=b,
            updates=2916,
            epochs=6,
            all_epochs_private_verified=True,
            elapsed_seconds=27000.0,
            clinical_or_Kaggle_score=None,
        )
    )
    pubs = []
    for e in range(1, 7):
        key = f"epoch-{e:03d}"
        files = {
            "model.pt.bin": {"bytes": 293400000, "sha256": str(e) * 64},
            "full-state.private.pt.bin": {"bytes": 880000000, "sha256": str(e) * 64},
            "run-start.json.bin": fileproof(binding_text.encode()),
            "source-permission.txt.bin": fileproof(license_bytes),
            **{
                n + ".bin": {"bytes": 10000, "sha256": C.CODE[k]}
                for n, k in (
                    ("colab_train.py", "actual_colab_train.py"),
                    ("colab_storage.py", "colab_storage.py"),
                    ("kaggle_io.py", "kaggle_io.py"),
                )
            },
        }
        pubs.append(
            dict(
                status="private_epoch_full_download_size_SHA_verified",
                private=True,
                id=10000 + e,
                version=1,
                slug="godlydonuts/rsna-colab-c100-" + key + "-" + C.MANIFEST[:12],
                key=key,
                revision=f"Kaggle:{10000 + e}:v1",
                new_GPU_or_competition_calls=0,
                files=files,
            )
        )
    terminal_pub = dict(
        pubs[0],
        id=10007,
        key="terminal",
        revision="Kaggle:10007:v1",
        slug="godlydonuts/rsna-colab-c100-terminal-" + C.MANIFEST[:12],
        files={"terminal.json.bin": fileproof(terminal_text.encode())},
    )
    return adapter, dict(
        binding_text=binding_text,
        terminal_text=terminal_text,
        epoch_publications=pubs,
        terminal_publication=terminal_pub,
        own_source_text=encode(OLD.original_sources()),
        license_bytes=license_bytes,
    )


def rewrite_binding(entry, update):
    b = json.loads(entry["binding_text"])
    update(b)
    entry["binding_text"] = encode(b)
    t = json.loads(entry["terminal_text"])
    t["binding"] = b
    entry["terminal_text"] = encode(t)
    for r in entry["epoch_publications"]:
        r["files"]["run-start.json.bin"] = fileproof(entry["binding_text"].encode())
    entry["terminal_publication"]["files"]["terminal.json.bin"] = fileproof(
        entry["terminal_text"].encode()
    )


def test_final_reuses_actual_private_dataset_and_honest_lineage(case):
    a, entry = case
    chosen = a.candidate(**entry)
    assert chosen["model_filename"] == "model.pt.bin"
    assert chosen["model_dataset_ref"] == entry["epoch_publications"][-1]["slug"]
    assert chosen["provenance"]["recipe"]["seed"] == 20261021
    assert chosen["provenance"]["completed_updates"] == 2916
    assert "GPU_job" not in chosen["provenance"]
    assert chosen["branch"] == C.BRANCH


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_git", "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"),
        ("input_manifest_sha256", "0" * 64),
        ("targets_sha256", "0" * 64),
        ("parent_weight_sha256", "0" * 64),
        ("GPU_job", "995351"),
        ("storage_control_sha256", {}),
        ("protected_labels_or_external_source_data_used", True),
        ("initial_complete_state_sha256", "unknown"),
        ("timm", "1.0.30"),
    ],
)
def test_wrong_or_invented_lineage_rejected(case, field, value):
    a, entry = case
    rewrite_binding(entry, lambda b: b.update({field: value}))
    with pytest.raises(ValueError):
        a.candidate(**entry)


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "wrong_key",
        "public",
        "version2",
        "corrupt_small",
        "missing_full_state",
        "wrong_terminal",
        "missing_epoch",
        "recipe",
    ],
)
def test_incomplete_or_unpreserved_model_rejected(case, failure):
    a, entry = case
    if failure == "partial":
        t = json.loads(entry["terminal_text"])
        t["updates"] = 486
        entry["terminal_text"] = encode(t)
        entry["terminal_publication"]["files"]["terminal.json.bin"] = fileproof(
            entry["terminal_text"].encode()
        )
    elif failure == "wrong_key":
        entry["epoch_publications"][-1]["key"] = "epoch-001"
    elif failure == "public":
        entry["epoch_publications"][0]["private"] = False
    elif failure == "version2":
        entry["epoch_publications"][0]["version"] = 2
    elif failure == "corrupt_small":
        entry["binding_text"] += " "
    elif failure == "missing_full_state":
        del entry["epoch_publications"][0]["files"]["full-state.private.pt.bin"]
    elif failure == "wrong_terminal":
        entry["terminal_publication"]["files"]["terminal.json.bin"]["sha256"] = "0" * 64
    elif failure == "missing_epoch":
        entry["epoch_publications"].pop(0)
    else:
        rewrite_binding(entry, lambda b: b["recipe"].update(seed=20261020))
    with pytest.raises(ValueError):
        a.candidate(**entry)


def embedded(source):
    return {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"FILES", "BINDING", "PERMISSION", "PARENT_SOURCE"}
    }


def test_compiler_keeps_original_decoder_worker_and_requires_own_qualification(case, monkeypatch):
    a, entry = case
    parent = "synthetic MRI encoder source, no real weights"
    a._base.PARENT_SOURCE = C.digest(parent)
    rewrite_binding(entry, lambda b: b.update(parent_source_sha256=C.digest(parent)))
    monkeypatch.setattr(BUILD.C, "Contract", lambda *x, **kw: a)
    dataset = entry["epoch_publications"][-1]["slug"]
    options = dict(contract_options={"module_sha256": C.MODULE_SHA256}, model_dataset_ref=dataset)
    source, receipt = BUILD.build(parent, entry, mode="qualification", **options)
    values = embedded(source)
    expected = (BUILD.ORIGINAL / "worker.py.txt").read_text()
    for n, v in values.items():
        expected = expected.replace("__" + n + "_VALUE__", repr(v))
    assert source == expected
    assert values["BINDING"]["branch"] == C.BRANCH
    assert values["BINDING"]["internet"] is False
    assert values["BINDING"]["GPU_count"] == 2
    with pytest.raises(ValueError):
        BUILD.build(parent, entry, mode="competition", **options)
    chosen = a.candidate(**entry)
    q = OLD.qualification(chosen, a._base)
    q.update(
        serving_worker_sha256=receipt["worker_sha256"],
        serving_contract_sha256=receipt["contract_sha256"],
        serving_helpers_sha256=receipt["serving_helpers_sha256"],
        model_dataset_ref=dataset,
    )
    BUILD.build(parent, entry, mode="competition", qualification=q, **options)
    q["models"][0]["trained_export_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        BUILD.build(parent, entry, mode="competition", qualification=q, **options)
    with pytest.raises(ValueError):
        BUILD.build(
            parent,
            entry,
            mode="qualification",
            **{**options, "model_dataset_ref": "godlydonuts/unrelated"},
        )


def test_generated_runtime_validates_all_learned_tensors_and_buffers(case):
    a, entry = case
    text = BUILD.runtime_contract(
        BUILD.originals()["contract.py"],
        (HERE / "contract.py").read_text(),
        {"module_sha256": C.MODULE_SHA256},
    )
    ns = {}
    exec(compile(text, "generated_colab_runtime", "exec"), ns)
    expected = a.candidate(**entry)
    model = torch.nn.BatchNorm1d(2)
    state = {n: t.clone() for n, t in model.state_dict().items()}
    state["running_mean"].fill_(3)
    expected["model_schema_sha256"] = C.digest(
        C.canonical({n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in state.items()})
    )
    expected["parameter_count"] = sum(p.numel() for p in model.parameters())
    exported = dict(
        model=state, architecture=expected["architecture"], provenance=expected["provenance"]
    )
    ns["validate_export"](exported, expected, model)
    assert torch.equal(model.running_mean, torch.tensor([3.0, 3.0]))
    broken = copy.deepcopy(exported)
    broken["model"]["weight"][0] = float("nan")
    with pytest.raises(ValueError):
        ns["validate_export"](broken, expected, model)
    ns["check_predictions"](["1.2.3"], [[0.2] * 12])
    with pytest.raises(ValueError):
        ns["check_predictions"](["1.2.3"], [[float("nan")] * 12])
