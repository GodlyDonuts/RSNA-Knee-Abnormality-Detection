import ast
import base64
import copy
import dataclasses
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
from test_licensed_parent_epoch import completed as epoch_fixture

from rsna.competitive import licensed_parent_epoch
from rsna.competitive.paired_source_zero import VERSION, Recipe

D = Path("initiative/research/source-zero-full-T4-20261020-v1")


def module(name):
    loader = SourceFileLoader("new_full_source_zero_" + name, str(D / (name + ".py.txt")))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def synthetic(c, monkeypatch):
    engine = "# synthetic numerical closure, never actual acceptance"
    files = {
        "rsna/competitive/paired_source_zero_train.py": engine,
        **{
            "rsna/competitive/" + n + ".py": "# synthetic original inputs"
            for n in ("licensed_parent_population", "licensed_pixel_bags", "licensed_pixel_cache")
        },
    }
    worker = "def _whole_model(x):return x\n_RECOVERY_BINDING={}\nclass _FixedSubset:pass\ndef main():pass\n"
    old = "\n".join(
        f"{n}={v!r}"
        for n, v in {
            "FILES": files,
            "WORKER": base64.b64encode(worker.encode()).decode(),
            "PERMISSION": "fixture",
        }.items()
    )
    selected = "def _selected_original_windows(x):\n    return x"
    cpu = "\n".join(
        f"{n}={v!r}"
        for n, v in {
            "FILES": {
                **files,
                "rsna/competitive/licensed_selected_pixel_bags.py": "# fixture selected",
            },
            "WORKER": base64.b64encode(selected.encode()).decode(),
        }.items()
    )

    def sha(s):
        return hashlib.sha256(s.encode()).hexdigest()

    monkeypatch.setattr(c, "RECOVERY_SHA", sha(old))
    monkeypatch.setattr(c, "INPUT_SHA", sha(cpu))
    monkeypatch.setattr(c, "ENGINE_SHA", sha(engine))
    r = {
        "status": "COMPLETE",
        "kernel_id": 139736683,
        "receipt": {
            "complete_model_AdamW_LR_allrank_RNG_scaler_history_snapshots_exact_resume": True,
            "both_view_full94_15x12_CSV_bytes_equal": True,
            "finite_updates": 6,
            "binding": {"own_module_sha256": {n: sha(s) for n, s in files.items()}},
        },
    }
    i = {
        "status": "COMPLETE",
        "kernel_id": 139738563,
        "receipt": {
            "all_images_labels_masks_teachers_exact": True,
            "CPU_RNG_unchanged": True,
            "actual_plain_mirrored_pairs": 64,
            "binding": {
                "own_module_sha256": {
                    "rsna/competitive/licensed_selected_pixel_bags.py": sha("# fixture selected")
                },
                "selected_renderer_sha256": sha(ast.unparse(ast.parse(selected))),
            },
        },
    }
    return old, cpu, r, i


def test_source_worker_fixed_across_continuous_epochs_and_no_exported_manifest(monkeypatch):
    c = module("compile")
    args = synthetic(c, monkeypatch)
    main = (D / "worker.py.txt").read_text()
    boot = (D / "bootstrap.py.txt").read_text()
    s, a = c.build(*args, main, boot, "a" * 40, 1)
    previous = {
        "status": "actual_paired_source_zero_full_epoch_complete",
        "segment": 1,
        "completed": 486,
        "binding": a["binding"],
        "worker_sha256": a["worker_sha256"],
        "checkpoint": {"bytes": 100, "sha256": "0" * 64},
        "run": {"bytes": 10, "sha256": "1" * 64},
    }
    t, b = c.build(*args, main, boot, "a" * 40, 2, previous)
    assert a["worker_sha256"] == b["worker_sha256"] and a["binding"] == b["binding"]
    assert a["source_sha256"] != b["source_sha256"] and a["fixed_final_updates"] == 1458
    assert 'dir="/tmp"' in s and '"RSNA_ORIGINAL_WINDOW_INTEGRITY"' in s
    assert a["competition_entries"] == 0
    for changed in (
        None,
        {**previous, "completed": 1},
        {**previous, "binding": {}},
        {**previous, "worker_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError):
            c.build(*args, main, boot, "a" * 40, 2, changed)
    bad = copy.deepcopy(args)
    bad[2]["receipt"]["finite_updates"] = 0
    with pytest.raises(ValueError):
        c.build(*bad, main, boot, "a" * 40, 1)


@pytest.fixture
def admitted_epoch():
    return epoch_fixture.__wrapped__()


def zero_epoch(completed):
    state, snapshot, receipt = completed
    b = state["binding"]
    b.update(
        recipe=dataclasses.asdict(Recipe(epochs=3)),
        version=VERSION,
        source_loss_weight=0.0,
        anatomical_views_per_study=2,
        teacher_role="original_parent_only",
        protected_gold_confirmation_used=False,
        teacher_proposal_sha256="2d7515536e378b2775cbacbf4e296713e358ec8f70aba8cc9212a952bed70a77",
        actual_T4_recovery={"kernel_id": 139736683},
        actual_CPU_input={"kernel_id": 139738563},
        source_module_sha256={
            "rsna/competitive/paired_source_zero_train.py": "14f4b234eb6d549d7f066d6f353c65c7df525f9ff2aba02d2c76b29e07a853a2"
        },
        input_integrity_dataset_id=12722964,
        input_integrity_sha256="85c2fbb342c4018ba77a5027a22b266b12c4ff59a766fd2af517c7c9ff30763f",
    )
    for row in state["history"]:
        row["anatomical_views"] = 2 * row["studies"]
    receipt["status"] = "actual_paired_source_zero_full_epoch_complete"
    receipt["devices"] = [
        {
            "rank": rank,
            "GPU": "Tesla T4",
            "trainable_parameters": 73136096,
            "all_BN_buffers_equal_parent": True,
            "initial_identity": {
                "model": "a75cb18d1dec13c4fb2f547f017ed92e81bf768f580cee54da0ffa88b712fc78"
            },
            "peak_allocated_bytes": 10,
            "total_VRAM_bytes": 100,
            "completed": 486,
        }
        for rank in (0, 1)
    ]
    engine = {
        "kernel_id": 139736683,
        "receipt": {
            "complete_model_AdamW_LR_allrank_RNG_scaler_history_snapshots_exact_resume": True
        },
    }
    return state, snapshot, receipt, engine


@pytest.mark.parametrize(
    "fault", [None, "source_weight", "views", "teacher", "engine", "initial", "missing_AdamW"]
)
def test_new_publisher_checks_complete_numerics_and_ablation_binding(admitted_epoch, fault):
    completed = zero_epoch(admitted_epoch)
    ns = {}
    exec(module("publisher").CHECKER, ns)
    real = licensed_parent_epoch.verify
    ns["original_verify"] = lambda *a, **kw: real(*a, **kw, expected_parameters=3)
    state, snapshot, receipt, engine = completed
    if fault == "source_weight":
        state["binding"]["recipe"]["source_weight"] = 0.25
    if fault == "views":
        state["history"][0]["anatomical_views"] = 8
    if fault == "teacher":
        state["binding"]["teacher_role"] = "cross_lineage_equal_rank"
    if fault == "engine":
        state["binding"]["source_module_sha256"]["rsna/competitive/paired_source_zero_train.py"] = (
            "0" * 64
        )
    if fault == "initial":
        receipt["devices"][0]["initial_identity"]["model"] = "0" * 64
    if fault == "missing_AdamW":
        state["optimizer"]["state"].pop(0)
    if fault:
        with pytest.raises((ValueError, AssertionError)):
            ns["verify"](*completed)
    else:
        assert ns["verify"](*completed)["source_loss_weight"] == 0.0


def test_original_publisher_cannot_be_silently_changed():
    c = module("publisher")
    original = Path(
        "initiative/research/S6-Kaggle-production-route-20261019-v1/publish_epoch.py.txt"
    ).read_text()
    generated = c.build(original, hashlib.sha256(original.encode()).hexdigest())
    compile(generated, "new actual independent publisher", "exec")
    with pytest.raises(ValueError):
        c.build(original + "\n", c.ORIGINAL_SHA)
