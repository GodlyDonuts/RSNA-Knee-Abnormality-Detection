import copy

import pytest
from test_licensed_importance_serving import accepted_epoch as sampled_epoch
from test_licensed_importance_serving_notebook import sources
from test_licensed_parent_serving_notebook import fixture_parent

from rsna.competitive import licensed_importance_serving as original_scope
from rsna.competitive import licensed_label_serving as scope
from scripts.kaggle import licensed_importance_serving as original_compiler
from scripts.kaggle import licensed_label_serving as compiler


@pytest.fixture
def final_label_epoch():
    # Format fixture only; never actual model, resource or GPU acceptance.
    snapshot, training, publication, sha = copy.deepcopy(sampled_epoch.__wrapped__())
    snapshot.update(epoch=3, completed=1458)
    binding = snapshot["binding"]
    binding.update(
        production_plan=scope.PLAN,
        source_git=scope.SOURCE_GIT,
        role="uniform",
        source_label_role="released_S6_source",
        source_label_target_sha256=scope.TARGET,
        actual_label_recovery=dict(scope.RECOVERY),
        source_label_contrast=dict(scope.LABELS),
    )
    binding["recipe"]["role"] = "uniform"
    training.update(epoch=3, role="uniform")
    training["receipt"].update(
        status="actual_S6_label_only_uniform16_full_parent_epoch_session_complete",
        role="uniform",
        segment=3,
        completed=1458,
        fixed_final_complete=True,
        binding=copy.deepcopy({k: v for k, v in binding.items() if k != "source_label_contrast"}),
    )
    publication.update(
        status="actual_S6_label_epoch_private_full_download_size_SHA_verified",
        role="uniform",
        epoch=3,
        completed=1458,
        source_git=scope.SOURCE_GIT,
    )
    publication["independent_CPU_state_verification"].update(
        role="uniform",
        finite_contiguous_updates=1458,
        label_role="released_S6_source",
        actual_label_recovery_kernel=139230981,
        actual_S6_label_bytes_and_original_teacher_mask_binding_exact=True,
    )
    return snapshot, training, publication, sha


def test_new_label_scope_keeps_original_model_and_records_untouched(final_label_epoch):
    snapshot, training, publication, sha = final_label_epoch
    before = (
        copy.deepcopy(snapshot["binding"]),
        copy.deepcopy(training),
        copy.deepcopy(publication),
    )
    graph = snapshot["model"]
    checked = scope.validate(snapshot, training, publication, sha)
    assert checked["completed"] == 1458 and checked["serving_windows"] == 94
    assert checked["actual_label_recovery_kernel"] == 139230981
    assert checked["source_label_target_sha256"] == scope.TARGET
    assert snapshot["model"] is graph
    assert before == (snapshot["binding"], training, publication)
    with pytest.raises(ValueError):
        original_scope.validate(*final_label_epoch)


@pytest.mark.parametrize(
    "change",
    [
        "not_final",
        "pending",
        "old_receipt",
        "unpreserved",
        "source",
        "label_values",
        "original_labels",
        "mask",
        "teacher",
        "diagnostic",
        "recovery",
        "bool_version",
        "recipe",
        "incomplete_model",
        "private_proof",
    ],
)
def test_incomplete_or_mixed_label_model_cannot_become_served(final_label_epoch, change):
    snapshot, training, pub, sha = final_label_epoch
    if change == "not_final":
        training["epoch"] = 2
    elif change == "pending":
        training["status"] = "RUNNING"
    elif change == "old_receipt":
        training["receipt"]["status"] = "actual_matched_IID16_full_parent_epoch_session_complete"
    elif change == "unpreserved":
        pub["status"] = "publication_uncertain"
    elif change == "source":
        training["receipt"]["binding"]["source_git"] = "a" * 40
    elif change in ("label_values", "original_labels", "mask", "teacher"):
        key = {
            "label_values": "new_label_array_sha256",
            "original_labels": "original_label_array_sha256",
            "mask": "mask_array_sha256",
            "teacher": "parent_teacher_array_sha256",
        }[change]
        snapshot["binding"]["source_label_contrast"][key] = "0" * 64
    elif change == "diagnostic":
        snapshot["binding"]["source_label_contrast"]["diagnostic_only"] = True
    elif change == "recovery":
        snapshot["binding"]["actual_label_recovery"]["source_sha256"] = "0" * 64
    elif change == "bool_version":
        snapshot["binding"]["actual_label_recovery"]["version"] = True
    elif change == "recipe":
        snapshot["binding"]["recipe"]["source_weight"] = 0.75
    elif change == "incomplete_model":
        snapshot["model"].pop("backbone.weight")
    else:
        pub["independent_CPU_state_verification"][
            "actual_S6_label_bytes_and_original_teacher_mask_binding_exact"
        ] = False
    with pytest.raises(ValueError):
        scope.validate(snapshot, training, pub, sha)


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_label_notebook_retains_original_full94_graph_and_inputs(
    monkeypatch, final_label_epoch, mode
):
    _, training, pub, sha = final_label_epoch
    parent = fixture_parent()
    monkeypatch.setattr(original_compiler, "PARENT_SOURCE_SHA256", original_compiler.digest(parent))
    helpers = sources()
    label = compiler.Path(scope.__file__).read_text()
    old_globals = (original_compiler.SERVING_SHA, original_compiler.provenance)
    source, binding = compiler.build(
        parent, *helpers[:2], label, *helpers[2:], training, pub, mode=mode
    )
    ns = {"__name__": "synthetic_only_no_GPU_or_accuracy"}
    exec(compile(source, "label_full94_format_fixture", "exec"), ns)
    helper_ns = {}
    exec(compile(ns["_ADAPTED_HELPER"], "standalone_actual_label_scope", "exec"), helper_ns)
    assert helper_ns["validate"](*final_label_epoch)["completed"] == 1458
    assert ns["_ADAPTED_TRAINING"] == training and ns["_ADAPTED_PUBLICATION"] == pub
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert binding["model_sha256"] == sha and binding["serving_windows"] == 94
    assert binding["source_label_target_sha256"] == scope.TARGET and binding["fixed_final_only"]
    assert binding["new_weight_Dataset_required"] is False
    assert binding["qualification_cannot_submit"] == (mode == "qualification")
    assert old_globals == (original_compiler.SERVING_SHA, original_compiler.provenance)


@pytest.mark.parametrize("change", ["old_helper", "label_helper", "mode"])
def test_unknown_label_or_old_helper_code_cannot_compile(monkeypatch, final_label_epoch, change):
    _, training, pub, _ = final_label_epoch
    parent = fixture_parent()
    monkeypatch.setattr(original_compiler, "PARENT_SOURCE_SHA256", original_compiler.digest(parent))
    helpers = sources()
    label = compiler.Path(scope.__file__).read_text()
    mode = "qualification"
    if change == "old_helper":
        helpers[1] += "changed"
    elif change == "label_helper":
        label += "changed"
    else:
        mode = "tiny_diagnostic"
    with pytest.raises(ValueError):
        compiler.build(parent, *helpers[:2], label, *helpers[2:], training, pub, mode=mode)
