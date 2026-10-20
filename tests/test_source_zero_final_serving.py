"""Final-only provenance and unchanged inference; synthetic tests are not GPU acceptance."""

import copy
import dataclasses
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
from test_licensed_importance_serving import accepted_epoch as old_epoch
from test_licensed_importance_serving_notebook import sources
from test_licensed_parent_serving_notebook import fixture_parent

from rsna.competitive.paired_source_zero import Recipe

D = Path("initiative/research/source-zero-final-serving-20261020-v1")


def load(name):
    loader = SourceFileLoader("source_zero_final_" + name, str(D / (name + ".py.txt")))
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(m)
    return m


helper = load("validate")
compiler = load("compile")


@pytest.fixture
def final_epoch():
    snapshot, training, pub, sha = copy.deepcopy(old_epoch.__wrapped__())
    binding = {
        "source_git": helper.SCIENCE,
        "experiment": "full3887_paired_parent_teacher_zero_source_fixed3_v1",
        "parent_checkpoint_sha256": helper.PARENT,
        "parent_source_sha256": helper.PARENT_SOURCE,
        "teacher_role": helper.ROLE,
        "source_loss_weight": 0.0,
        "anatomical_views_per_study": 2,
        "source_module_sha256": {"rsna/competitive/paired_source_zero_train.py": helper.ENGINE},
        "actual_T4_recovery": {"kernel_id": 139736683},
        "actual_CPU_input": {"kernel_id": 139738563},
        "input_integrity_dataset_id": 12722964,
        "input_integrity_sha256": "85c2fbb342c4018ba77a5027a22b266b12c4ff59a766fd2af517c7c9ff30763f",
        "protected_gold_confirmation_used": False,
    }
    training.update(epoch=3, version=1)
    training["receipt"].update(
        status="actual_paired_source_zero_full_epoch_complete",
        segment=3,
        completed=1458,
        total=1458,
        fixed_final_complete=True,
        teacher_target_role=helper.ROLE,
        binding=copy.deepcopy(binding),
    )
    snapshot.update(epoch=3, completed=1458)
    snapshot["binding"] = {
        **binding,
        "version": "paired_IID16_zero_source_one_or_two_rank_v1",
        "recipe": dataclasses.asdict(Recipe(epochs=3)),
        "population": 3887,
        "population_sha256": helper.POPULATION,
        "world_size": 2,
        "total_steps": 1458,
        "teacher_proposal_sha256": "2d7515536e378b2775cbacbf4e296713e358ec8f70aba8cc9212a952bed70a77",
    }
    pub.update(
        status="actual_source_zero_full_epoch_private_all_objects_download_size_SHA_verified",
        source_git=helper.SCIENCE,
        epoch=3,
        completed=1458,
        version=1,
        kernel_id=training["kernel_id"],
        teacher_target_role=helper.ROLE,
        original_source_sha256=training["source_sha256"],
        objects=6,
        canonical_shared_publication_lock_used=True,
        model_sha256=sha,
        prefix="checkpoints/source-zero-full-twoT4-509db5b93aa3/original_parent_only/epoch-003",
    )
    pub["independent_CPU_state_verification"].update(
        full_learned_graph_exact=True,
        exact_original_model_graph_shapes_dtypes=True,
        finite_contiguous_updates=1458,
        actual_AdamW_parameters=73136096,
        all_two_rank_RNG_and_scaler_records_present=True,
        source_loss_weight=0.0,
        anatomical_views_per_study=2,
        original_parent_teacher=True,
        actual_input_acceptance=139738563,
    )
    return snapshot, training, pub, sha


def test_predeclared_full_final_and_zero_source_recipe(final_epoch):
    assert helper.RECIPE == dataclasses.asdict(Recipe(epochs=3))
    z = helper.validate(*final_epoch)
    assert z["completed"] == 1458 and z["source_loss_weight"] == 0.0
    assert z["serving_windows"] == 94 and not z["clinical_accuracy_claim"]


@pytest.mark.parametrize(
    "change",
    [
        "running",
        "early",
        "model",
        "publication",
        "source",
        "positive",
        "one_view",
        "teacher",
        "recipe",
        "population",
        "rank",
        "engine",
        "input",
        "head",
        "protected",
        "NaN",
    ],
)
def test_wrong_lineage_incomplete_fit_or_graph_refused(final_epoch, change):
    s, t, p, sha = final_epoch
    if change == "running":
        t["status"] = "RUNNING"
    elif change == "early":
        t["epoch"] = 1
    elif change == "model":
        p["model_sha256"] = "0" * 64
    elif change == "publication":
        p["canonical_shared_publication_lock_used"] = False
    elif change == "source":
        p["original_source_sha256"] = "0" * 64
    elif change == "positive":
        t["receipt"]["binding"]["source_loss_weight"] = 0.25
    elif change == "one_view":
        t["receipt"]["binding"]["anatomical_views_per_study"] = 1
    elif change == "teacher":
        t["receipt"]["binding"]["teacher_role"] = "cross_lineage_equal_rank"
    elif change == "recipe":
        s["binding"]["recipe"]["encoder_lr"] *= 2
    elif change == "population":
        s["binding"]["population"] = 17
    elif change == "rank":
        s["binding"]["world_size"] = 1
    elif change == "engine":
        t["receipt"]["binding"]["source_module_sha256"][
            "rsna/competitive/paired_source_zero_train.py"
        ] = "0" * 64
    elif change == "input":
        p["independent_CPU_state_verification"]["actual_input_acceptance"] = 0
    elif change == "head":
        s["model"]["clsW"] = s["model"]["clsW"][:1]
    elif change == "protected":
        t["receipt"]["binding"]["protected_gold_confirmation_used"] = True
    else:
        s["model"]["clsW"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        helper.validate(s, t, p, sha)


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_exact_existing_full94_inference_for_new_final_model(monkeypatch, final_epoch, mode):
    _, training, publication, _ = final_epoch
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE", compiler.digest(parent))
    inputs = sources()
    inputs[1] = (D / "validate.py.txt").read_text()
    source, meta = compiler.build(
        parent,
        *inputs,
        training,
        publication,
        mode=mode,
        legacy_source=Path("scripts/kaggle/licensed_importance_serving.py").read_text(),
    )
    ns = {"__name__": "synthetic_not_actual_serving"}
    exec(compile(source, "unchanged_original_inference", "exec"), ns)
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert ns["ARMS"][0]["file"] == "epoch-003.pt"
    assert meta["source_loss_weight"] == 0.0 and meta["teacher_target_role"] == helper.ROLE
    assert not meta["actual_new_weight_T4_serving_accepted"]
    assert meta["qualification_cannot_submit"] == (mode == "qualification")
    assert "actual_source_zero_final_epoch_" in source
    with pytest.raises(ValueError):
        compiler.namespace("changed", inputs[1])


def test_final_CPU_compiler_preserves_original_controls_and_waits_without_intent():
    p = load("prepare")
    old = Path(
        "initiative/research/S6-Kaggle-production-route-20261019-v1/serving_prepare.py.txt"
    ).read_text()
    derived = p.build(old)
    compile(derived, "actual_final_CPU_compiler", "exec")
    assert "source-zero-full-v1" in derived
    assert "type(epoch) is int and epoch == 3" in derived
    assert derived.index('h["provenance"]') < derived.index("output.mkdir")
    assert "actual_source_zero_fixed_final_serving_sources_compiled" in derived
    with pytest.raises(ValueError):
        p.build(old + "# modified\n")


def test_account_is_distinct_final_only_and_preserves_original_mutation_protocol():
    b = load("account")
    old = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")
    source = b.core((old / "serving_cloud.py.txt").read_text())
    ns = {"__name__": "unit_test_no_remote_mutations"}
    exec(compile(source, "source_zero_final_account", "exec"), ns)
    assert str(ns["D"]).endswith("/source-zero-full-v1")
    assert source.count('if __name__ == "__main__":') == 1
    assert "and epoch == 3" in source
    assert "licensed-importance-full-fit:source-zero-full:" in source
    assert "source_zero_full_fit_actual_complete" in source
    assert "CreateCodeSubmission" in source and "fresh_account" in source
    assert "claim.mkdir" in source and "automatic_retry" in source
    assert "paid_bytes" in source and "5000+128+128" in source
    with pytest.raises(ValueError):
        b.core(source)
    policy = b.policy((old / "serving_policy.py.txt").read_text())
    p = {"__name__": "unit_test_no_remote_mutations"}
    exec(compile(policy, "source_zero_final_policy", "exec"), p)
    assert p["EPOCHS"] == (3,) and p["DOCKER"] == b.DOCKER
    assert p["budget"]({"totalTimeAllowed": "3000s"}, 3, "qualification", latest_training_epoch=3)[
        "available"
    ]
    assert not p["budget"](
        {"totalTimeAllowed": "2999s"}, 3, "qualification", latest_training_epoch=3
    )["available"]
    with pytest.raises(ValueError):
        p["budget"]({"totalTimeAllowed": "99999s"}, 1, "qualification", latest_training_epoch=3)
    training = {
        "epoch": 3,
        "status": "COMPLETE",
        "version": 1,
        "slug": "godlydonuts/synthetic-only",
    }
    request = p["request"](3, "competition", "synthetic-only", training)
    assert request["sessionTimeoutSeconds"] == 32400 and not request["enableInternet"]
    assert request["slug"].startswith("godlydonuts/rsna-zero-final-e3-offline-")
    assert request["kernelDataSources"][-1] == training["slug"]
    assert request["dockerImage"] == b.DOCKER
    assert "actual_source_zero_final_epoch_" in policy
