"""New timing instrumentation on actual small optimizer loops, not a GPU test."""

import copy

import pytest
import torch
from test_licensed_importance import Model, equal
from test_paired_logit_train import Corrected, dataset

from rsna.competitive import licensed_importance_train as ordinary
from rsna.competitive import paired_logit_train as corrected
from rsna.competitive.licensed_parent_train import PARENT_SHA
from rsna.competitive.paired_step_timing import _existing_output, timed_fit


@pytest.mark.parametrize("arm", ["ordinary", "controlled"])
def test_observation_preserves_complete_actual_numerical_trajectory(tmp_path, arm):
    torch.set_num_threads(1)
    paired, recipe, original = dataset(tmp_path, nonzero=True)
    engine, data = (ordinary, original) if arm == "ordinary" else (corrected, paired)
    model_class = Model if arm == "ordinary" else Corrected
    torch.manual_seed(7)
    initial = copy.deepcopy(model_class().state_dict())
    outputs = []
    before_hooks = (engine.lr_factor, engine.atomic_json)
    for mode in ("uninstrumented", "timed"):
        model = model_class()
        model.load_state_dict(initial)
        torch.manual_seed(11)
        output = tmp_path / mode
        binding = {"parent_checkpoint_sha256": PARENT_SHA}
        if mode == "timed":
            record = timed_fit(
                engine, model, data, output, binding, recipe, device=torch.device("cpu")
            )
        else:
            engine.fit(model, data, output, binding, recipe, device=torch.device("cpu"))
        outputs.append(torch.load(output / "checkpoint.pt", weights_only=True))
    equal(outputs[0], outputs[1])
    assert (engine.lr_factor, engine.atomic_json) == before_hooks
    assert record["actual_optimizer_updates"] == 6 and len(record["step_timings"]) == 6
    assert [v["update"] for v in record["step_timings"]] == list(range(1, 7))
    assert all(v["seconds"] > 0 for v in record["step_timings"])
    assert record["numerical_engine_body_changed"] is False
    assert record["new_GPU_or_accuracy_acceptance"] is False


def test_failure_restores_original_engine_hooks_and_does_not_retry(tmp_path, monkeypatch):
    paired, recipe, _ = dataset(tmp_path, nonzero=True)
    factor, write = corrected.lr_factor, corrected.atomic_json
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("synthetic new I/O failure")

    monkeypatch.setattr(corrected, "atomic_json", fail)
    with pytest.raises(RuntimeError, match="synthetic new I/O failure"):
        timed_fit(
            corrected,
            Corrected(),
            paired,
            tmp_path / "new",
            {"parent_checkpoint_sha256": PARENT_SHA},
            recipe,
            device=torch.device("cpu"),
        )
    assert len(calls) == 1 and corrected.lr_factor is factor and corrected.atomic_json is fail
    monkeypatch.setattr(corrected, "atomic_json", write)


def test_existing_output_refused_before_any_fit(tmp_path):
    paired, recipe, _ = dataset(tmp_path, nonzero=True)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="Fresh original six-update"):
        timed_fit(
            corrected,
            Corrected(),
            paired,
            output,
            {"parent_checkpoint_sha256": PARENT_SHA},
            recipe,
            device=torch.device("cpu"),
        )


def test_peer_uses_shared_prefit_directory_decision_not_a_racing_second_stat(tmp_path, monkeypatch):
    from rsna.competitive import paired_step_timing as timing

    output = tmp_path / "peer_created_after_shared_admission"
    output.mkdir()
    monkeypatch.setattr(timing.dist, "is_initialized", lambda: True)

    def admitted_before_creation(value, src):
        assert src == 0
        value.fill_(0)

    monkeypatch.setattr(timing.dist, "broadcast", admitted_before_creation)
    assert _existing_output(output, 1, torch.device("cpu")) is False
