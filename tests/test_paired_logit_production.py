"""New production template compilation only; no real GPU/fit acceptance implied."""

import base64
import copy
import json
from pathlib import Path

import pytest
from test_licensed_importance_production import inputs as original_fixture

from scripts.kaggle import licensed_importance_production as baseline
from scripts.kaggle import paired_logit_production as m


@pytest.fixture
def inputs(monkeypatch):
    values = list(original_fixture.__wrapped__(monkeypatch))
    worker = """from rsna.competitive.licensed_importance_train import fit
class WholeStudyRaptor(ImportanceReadout,CheckpointedEncode,RaptorClassifier):pass
def _whole_model(base):return base
def main():raise RuntimeError('synthetic diagnostic')
if __name__=='__main__':main()
"""
    source = (
        "WORKER="
        + repr(base64.b64encode(worker.encode()).decode())
        + "\nPERMISSION="
        + repr(base64.b64encode(b"synthetic permission").decode())
        + "\n"
    )
    monkeypatch.setattr(baseline, "RECOVERY_SHA", baseline.digest(source))
    monkeypatch.setattr(baseline, "WORKER_SHA", baseline.digest(worker))
    values[0] = source
    values[1]["source_sha256"] = baseline.RECOVERY_SHA
    values[1]["receipt"]["worker_source_sha256"] = baseline.WORKER_SHA
    values[2].update({n: Path(n).read_text() for n in m.EXTRA})
    role = copy.deepcopy(values[1]["receipt"]["roles"]["influence"])
    recovery = {
        "status": "COMPLETE",
        "kernel_id": 139417155,
        "version": 1,
        "source_sha256": m.RECOVERY_SOURCE,
        "receipt": {
            "status": "actual_paired_logit_encoder_FP16_dropout_recovery_accepted",
            "worker_source_sha256": m.RECOVERY_WORKER,
            "source_module_sha256": {**baseline.ACCEPTED_MODULES, **m.EXTRA},
            "fit_studies": 16,
            "disjoint_engineering_studies": 16,
            "epochs": 3,
            "updates_per_role": 6,
            "roles": {"influence": role},
        },
    }
    correction = json.loads(
        Path("initiative/research/paired-logit-fixed-schedule-20261018-v1/actual.json").read_text()
    )
    return values, recovery, correction


def build(inputs, segment=1, previous=None):
    values, actual, correction = inputs
    return m.build(
        *values, segment, paired_recovery=actual, correction=correction, previous=previous
    )


def previous(meta, epoch):
    return {
        "status": m.STATUS,
        "role": "influence",
        "segment": epoch,
        "completed": 486 * epoch,
        "worker_sha256": meta["worker_sha256"],
        "binding": meta["binding"],
        "checkpoint": {"bytes": 100, "sha256": "a" * 64},
        "run": {"bytes": 50, "sha256": "b" * 64},
    }


def test_same_fixed_recipe_draws_and_worker_across_all_three_continuous_epochs(inputs):
    source, first = build(inputs)
    _, second = build(inputs, 2, previous(first, 1))
    _, third = build(inputs, 3, previous(second, 2))
    assert first["worker_sha256"] == second["worker_sha256"] == third["worker_sha256"]
    assert first["binding"] == second["binding"] == third["binding"]
    assert third["completed_target"] == 1458 and third["maximum_GPU_seconds"] == 6000
    assert first["production_execution_admitted_by_compilation"] is False
    assert first["full_production_fit"] is False
    assert first["actual_noise_runtime_funding_and_dispatch_decision_still_required"] is True
    assert first["competition_entries"] == 0
    _, encoded = m._literal(source, "WORKER")
    worker = base64.b64decode(encoded).decode()
    assert "from rsna.competitive.paired_logit_train import fit" in worker
    assert "data=PairedCases(raw,table" in worker
    assert "PairedLogitReadout,ImportanceReadout,CheckpointedEncode" in worker
    assert "Recipe(epochs=3,role=_ROLE)" in worker
    assert "len(original)==3887" in worker and "73136096" in worker
    assert "source_weight" not in worker.split("def main():", 1)[1]
    assert "paired_parent_logit_IID16_training_v1" in source
    assert "paired-logit-full-session-accepted.json" in source
    assert m._literal(source, "FILES")[1] == inputs[0][2]
    compile(source, "prepared-only-new-full-template", "exec")


@pytest.mark.parametrize(
    "bad",
    [
        "pending",
        "incomplete_encoder",
        "table",
        "recipe",
        "source",
        "ordinary_resume",
        "missing_optimizer",
        "first_resume",
    ],
)
def test_unaccepted_or_incompatible_inputs_stop_before_new_template(inputs, bad):
    values, recovery, correction = copy.deepcopy(inputs)
    _, meta = build((values, recovery, correction))
    prior = previous(meta, 1)
    segment = 2
    if bad == "pending":
        recovery["status"] = "RUNNING"
    elif bad == "incomplete_encoder":
        recovery["receipt"]["roles"]["influence"]["devices"][0]["trainable_parameters"] = 100
    elif bad == "table":
        correction["table"]["sha256"] = "a" * 64
    elif bad == "recipe":
        correction["recipe"]["encoder_lr"] = 3e-5
    elif bad == "source":
        values[2][next(iter(m.EXTRA))] += "\n# changed numerical source\n"
    elif bad == "ordinary_resume":
        prior["status"] = "actual_matched_IID16_full_parent_epoch_session_complete"
    elif bad == "missing_optimizer":
        prior.pop("checkpoint")
    else:
        segment = 1
    with pytest.raises(ValueError):
        build((values, recovery, correction), segment, prior)


def test_literal_replacement_preserves_other_unicode_source_and_only_named_assignment():
    source = "# naïve λ\nX = 'é'\nY = None\nprint('None remains ordinary text')\n"
    changed = m._set_literal(source, "Y", {"value": "α"})
    assert m._literal(changed, "Y")[1] == {"value": "α"}
    assert "# naïve λ\nX = 'é'\n" in changed and "print('None remains ordinary text')" in changed
