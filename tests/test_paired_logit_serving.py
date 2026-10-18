"""New paired inference boundary and unchanged two-view/full94 serving."""

import copy
from pathlib import Path

import pytest
import torch
from test_licensed_importance_serving import accepted_epoch as old_fixture
from test_licensed_parent_serving_notebook import fixture_parent

from rsna.competitive import paired_logit_serving as scope
from scripts.kaggle import licensed_importance_serving as original
from scripts.kaggle import paired_logit_serving as compiler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def final():
    s, t, p, h = copy.deepcopy(old_fixture.__wrapped__())
    b = s["binding"]
    b.update(
        source_git=scope.SOURCE,
        production_plan=scope.PLAN,
        version=scope.VERSION,
        recipe=copy.deepcopy(scope.RECIPE),
        paired_recovery=copy.deepcopy(scope.RECOVERY),
        correction_table=copy.deepcopy(scope.TABLE),
        parent_correction_sha256=scope.TABLE["sha256"],
        original_target_sha256="48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
    )
    s.update(epoch=3, completed=1458)
    t.update(epoch=3, version=1)
    t["receipt"].update(
        status=scope.STATUS,
        segment=3,
        completed=1458,
        fixed_final_complete=True,
        worker_sha256=scope.WORKER,
        checkpoint={"bytes": 456, "sha256": "3" * 64},
        run={"bytes": 789, "sha256": "4" * 64},
        binding=copy.deepcopy(b),
    )
    for n in (
        "recipe",
        "version",
        "population",
        "population_sha256",
        "world_size",
        "total_steps",
        "parent_correction_sha256",
    ):
        t["receipt"]["binding"].pop(n, None)
    p.update(
        status=scope.PUB_STATUS,
        epoch=3,
        completed=1458,
        original_kernel=t["kernel_id"],
        original_source_sha256=t["source_sha256"],
        objects=6,
        full_optimizer_included_private_HF=True,
        all_HF_cache_and_Xet_paths_explicitly_RAM_before_import=True,
    )
    p["files"].update(
        {
            "optimizer-checkpoint.pt": t["receipt"]["checkpoint"],
            "run.json": t["receipt"]["run"],
            "source-permission.txt": {"bytes": 1, "sha256": "5" * 64},
            "own-source.json": {"bytes": 2, "sha256": "6" * 64},
            "provenance.json": {"bytes": 3, "sha256": "7" * 64},
        }
    )
    p["independent_CPU_state_verification"].update(
        role="influence",
        finite_contiguous_updates=1458,
        paired_correction_sha256=scope.TABLE["sha256"],
        exact_original_recipe_and_real_paired_recovery=True,
    )
    return s, t, p, h


def test_final_only_real_contract_without_runtime_fields_in_producer(final):
    before = copy.deepcopy(final[1:])
    out = scope.validate(*final)
    assert out["completed"] == 1458 and out["serving_windows"] == 94
    assert out["parent_logit_correction_applied_in_inference"] is False
    assert final[1:] == before


@pytest.mark.parametrize(
    "change",
    [
        "early",
        "ordinary",
        "wrong_recovery",
        "wrong_table",
        "lost_optimizer",
        "wrong_worker",
        "wrong_recipe",
        "wrong_source",
        "pending",
        "LR_mixture",
    ],
)
def test_rejects_unfinished_or_misbound_model(final, change):
    s, t, p, h = final
    if change == "early":
        t["epoch"] = 2
    elif change == "ordinary":
        s["binding"]["version"] = "licensed_full_encoder_uniform_or_teacher_influence_IID16_v1"
    elif change == "wrong_recovery":
        t["receipt"]["binding"]["paired_recovery"]["kernel_id"] = 0
    elif change == "wrong_table":
        s["binding"]["parent_correction_sha256"] = "0" * 64
    elif change == "lost_optimizer":
        p["files"].pop("optimizer-checkpoint.pt")
    elif change == "wrong_worker":
        t["receipt"]["worker_sha256"] = "0" * 64
    elif change == "wrong_recipe":
        s["binding"]["recipe"]["encoder_lr"] = 3e-5
    elif change == "wrong_source":
        p["original_source_sha256"] = "0" * 64
    elif change == "pending":
        t["status"] = "RUNNING"
    else:
        s["binding"]["single_factor"] = {"encoder_lr_treatment": 3e-5}
    with pytest.raises(ValueError):
        scope.validate(*final)


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_compiler_preserves_inference_and_includes_actual_training_graph(monkeypatch, final, mode):
    _, training, pub, _ = final
    parent = fixture_parent()
    monkeypatch.setattr(original, "PARENT_SOURCE_SHA256", original.digest(parent))
    helpers = [
        (ROOT / n).read_text()
        for n in (
            "rsna/competitive/bounded_preparation.py",
            "rsna/competitive/paired_logit_serving.py",
            "rsna/competitive/licensed_pixel_cache.py",
            "rsna/competitive/licensed_full_bag.py",
            "rsna/competitive/licensed_importance.py",
            "rsna/competitive/paired_logit_control.py",
        )
    ]
    source, meta = compiler.build(parent, *helpers, training, pub, mode=mode)
    ns = {"__name__": "synthetic_compilation_only"}
    exec(compile(source, "synthetic_paired_full94", "exec"), ns)
    assert ns["ARMS"][0]["file"] == "epoch-003.pt"
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert meta["qualification_reference_includes_actual_PairedLogitReadout"]
    assert meta["qualification_cannot_submit"] == (mode == "qualification")
    assert not meta["new_weight_Dataset_required"]
    assert (
        "class TrainingGraph(importance['PairedLogitReadout'],importance['ImportanceReadout']"
        in source
    )
    embedded = {}
    exec(compiler.literal(source, "_ADAPTED_IMPORTANCE"), embedded)

    class Parent(torch.nn.Module):
        def forward(self, images):
            return images.sum(-1)

    class Qualified(embedded["PairedLogitReadout"], embedded["ImportanceReadout"], Parent):
        pass

    x = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    assert torch.equal(Qualified()(x), Parent()(x))
    with pytest.raises(ValueError):
        Qualified()(x, baseline_shift=torch.zeros(2, 12))
