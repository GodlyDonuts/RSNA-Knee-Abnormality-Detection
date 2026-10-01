import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch

from rsna.competitive.train import _uid_hash
from rsna.schema import TARGETS
from scripts.newton.measure_relation_gradients import matched_configs
from scripts.newton.probe_relation_trainer_recovery import (
    admit_gradient,
    compare_completed,
    execute,
    invocation_values,
    state_record,
    validate_predictions,
)


def checkpoint_and_trajectory():
    policy = dict(weight=0.05, target_mode="physical", max_pairs=32, hidden_dim=128, head_seed=19)
    checkpoint = {
        "schema_version": 2,
        "optimizer_boundary": True,
        "epoch": 2,
        "next_update": 0,
        "step": 4,
        "binding": {"world_size": 1, "config": {"relation": policy}},
        "model_storage": {"format": "frozen_base_delta_v1", "frozen_tensors": {"base": "hash"}},
        "model": {"encoder.adapter": torch.tensor([0.2, -0.3])},
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([0.01, 0.02]), "step": torch.tensor(4.0)}},
            "param_groups": [{"params": [0], "lr": 0.0001}],
        },
        "rng_by_rank": [{"torch": torch.arange(5, dtype=torch.uint8), "python": (1, 2, None)}],
    }
    rows = [
        dict(
            epoch=(index - 1) // 2,
            update=(index - 1) % 2 + 1,
            optimizer_steps=index,
            loss=0.4,
            gradient_norm=0.5,
            observed_cells=30.0,
            classification_loss=0.39,
            relation_loss=0.2,
            relation_studies=8.0,
            relation_weight=0.05,
            elapsed_seconds=float(index),
            peak_gpu_allocated_bytes=99,
        )
        for index in range(1, 5)
    ]
    partial = copy.deepcopy(checkpoint) | dict(epoch=0, next_update=1, step=1)
    return checkpoint, partial, rows, policy


def compare_fixture(checkpoint, partial, rows, policy, resumed=None, replay=b"MRI-only CSV"):
    return compare_completed(
        checkpoint,
        copy.deepcopy(checkpoint) if resumed is None else resumed,
        partial,
        {"updates": rows, "status": "completed", "exit_code": 0},
        {"updates": rows[:1], "status": "checkpointed_for_resume", "exit_code": 75},
        {"updates": rows[1:], "status": "completed", "exit_code": 0},
        b"MRI-only CSV",
        replay,
        policy,
    )


def test_decoded_checkpoint_and_trajectory_equality_survives_real_torch_serialization(tmp_path):
    checkpoint, partial, rows, policy = checkpoint_and_trajectory()
    path = tmp_path / "state.pt"
    torch.save(checkpoint, path)
    decoded = torch.load(path, map_location="cpu", weights_only=True)
    assert len(compare_fixture(checkpoint, partial, rows, policy, decoded)) == 64
    altered_timing = copy.deepcopy(rows)
    for row in altered_timing:
        row["elapsed_seconds"] += 100
        row["peak_gpu_allocated_bytes"] += 100
    assert invocation_values({"updates": altered_timing}) == invocation_values({"updates": rows})


@pytest.mark.parametrize(
    "kind", ["model", "optimizer", "rng", "frozen", "lr", "binding", "counter"]
)
def test_changed_decoded_recovery_envelope_is_rejected(kind):
    checkpoint, partial, rows, policy = checkpoint_and_trajectory()
    replay = copy.deepcopy(checkpoint)
    if kind == "model":
        replay["model"]["encoder.adapter"][0] += 0.001
    elif kind == "optimizer":
        replay["optimizer"]["state"][0]["exp_avg"][0] += 0.001
    elif kind == "rng":
        replay["rng_by_rank"][0]["torch"][0] += 1
    elif kind == "frozen":
        replay["model_storage"]["frozen_tensors"]["base"] = "other"
    elif kind == "lr":
        replay["optimizer"]["param_groups"][0]["lr"] *= 2
    elif kind == "binding":
        replay["binding"]["foreign_source"] = True
    else:
        replay["step"] = 5
    with pytest.raises(ValueError):
        compare_fixture(checkpoint, partial, rows, policy, replay)


@pytest.mark.parametrize("kind", ["partial", "prediction", "nonfinite", "missing_optimizer"])
def test_incomplete_boundary_or_invalid_outputs_reject_recovery(kind):
    checkpoint, partial, rows, policy = checkpoint_and_trajectory()
    csv = b"MRI-only CSV"
    if kind == "partial":
        partial["next_update"] = 0
    elif kind == "prediction":
        csv = b"a different MRI prediction"
    elif kind == "nonfinite":
        checkpoint["model"]["encoder.adapter"][0] = float("nan")
    else:
        checkpoint["optimizer"]["state"] = {}
    with pytest.raises(ValueError):
        compare_fixture(checkpoint, partial, rows, policy, replay=csv)


def gradient_fixture():
    configs = matched_configs(Path("initiative/experiments/competitive-v1/slice-relations-v1"))
    row = {
        "classification_gradient_sha256": "c" * 64,
        "classification_logits_sha256": "l" * 64,
        "pair_roster_sha256": "p" * 64,
        "observed_target_counts": [1] * 12,
        "microbatches": 8,
        "represented_studies": 8,
        "relation_pairs": 256,
        "relation_loss": 0.2,
        "encoder_gradients": {
            "classification_norm": 0.3,
            "unweighted_relation_norm": 0.2,
            "combined_norm": 0.3,
        },
    }
    results = {}
    for name, config in configs.items():
        results[name] = [
            copy.deepcopy(row) | dict(weighted_relation_loss=config.relation["weight"] * 0.2)
            for _ in range(2)
        ]
    expected = {name: asdict(config) for name, config in configs.items()}
    binding = {"exact_actual_training_input": "sha"}
    receipt = dict(
        status="actual_official_weight_full224_BF16_training_gradient_measurement_complete",
        optimizer_updates=0,
        clinical_accuracy_evidence=False,
        input_binding=binding,
        CPU_acceptance_sha256="a" * 64,
        configs=expected,
        same_classification_gradients_logits_pairs_across_arms=True,
        MRI_only_12_output_inference=True,
        results=results,
    )
    decision = dict(
        status="training_only_gradient_scale_review_accepted_for_bounded_pilot",
        configs=expected,
        reason="Training-only real measurement reviewed before pilot; no accuracy claim.",
    )
    return receipt, decision, binding, configs


def test_completed_real_gradient_contract_requires_matching_input_and_explicit_scale_decision():
    receipt, decision, binding, configs = gradient_fixture()
    admit_gradient(receipt, decision, "a" * 64, binding, configs)
    with pytest.raises(ValueError):
        admit_gradient(receipt, decision, "b" * 64, binding, configs)
    with pytest.raises(ValueError):
        admit_gradient(receipt, {}, "a" * 64, binding, configs)


@pytest.mark.parametrize(
    "kind", ["CPU_only", "input", "recipe", "decision", "objective", "gradient", "roster", "count"]
)
def test_premature_or_mismatched_gradient_admission_is_rejected(kind):
    receipt, decision, binding, configs = gradient_fixture()
    if kind == "CPU_only":
        receipt["status"] = "actual_full224_training_inputs_CPU_verified"
    elif kind == "input":
        receipt["input_binding"] = {"different": "training pixels"}
    elif kind == "recipe":
        receipt["configs"]["physical"]["relation"]["weight"] = 0.5
    elif kind == "decision":
        decision["reason"] = " "
    elif kind == "objective":
        receipt["results"]["physical"][0]["weighted_relation_loss"] = 1
    elif kind == "gradient":
        receipt["results"]["physical"][0]["encoder_gradients"]["classification_norm"] = 0
    elif kind == "roster":
        receipt["results"]["shuffled"][0]["pair_roster_sha256"] = "different"
    else:
        receipt["results"]["physical"] = receipt["results"]["physical"][:1]
    with pytest.raises(ValueError):
        admit_gradient(receipt, decision, "a" * 64, binding, configs)


def test_real_process_expected_stop_and_stderr_are_preserved_without_retry(tmp_path):
    output = tmp_path / "stop"
    command = [sys.executable, "-c", "import sys; print('boundary'); sys.exit(75)"]
    execute(command, output, 75)
    assert output.with_suffix(".stdout.txt").read_text() == "boundary\n"
    with pytest.raises(FileExistsError):
        execute(command, output, 75)
    with pytest.raises(ValueError, match="expected0"):
        execute(command, tmp_path / "wrong", 0)


def test_state_record_preserves_types_and_rejects_nonfinite_or_unsupported_values():
    assert state_record((1,)) != state_record([1])
    assert state_record(True) != state_record(1)
    for value in (float("inf"), float("nan"), object()):
        with pytest.raises(ValueError):
            state_record(value)
    assert json.dumps(state_record({0: torch.tensor([1]), "0": torch.tensor([2])}))


@pytest.mark.parametrize("altered", [None, "train", "outer", "code", "base", "nan", "range", "uid"])
def test_prediction_acceptance_checks_exact_original_source_and_exam_membership(altered):
    train = pd.DataFrame({"study_uid": [f"training-{i}" for i in range(16)]})
    outer = pd.DataFrame({"study_uid": [f"outer-{i}" for i in range(16)]})
    frame = outer.copy()
    for target in TARGETS:
        frame[target] = 0.5
    provenance = {"pretrained": True, "weights_sha256": "w" * 64}
    source = {"files": {"rsna/competitive/train.py": {"sha256": "c" * 64}}}
    checkpoint = {
        "binding": {
            "training_uids_sha256": _uid_hash(train),
            "outer_uids_sha256": _uid_hash(outer),
            "code_sha256": {"competitive/train.py": "c" * 64},
            "config": {"encoder": provenance, "precision": "bf16"},
            "device_type": "cuda",
        },
        "model_storage": {"encoder_provenance": provenance},
    }
    if altered in {"train", "outer"}:
        key = "training_uids_sha256" if altered == "train" else "outer_uids_sha256"
        checkpoint["binding"][key] = "other"
    elif altered == "code":
        checkpoint["binding"]["code_sha256"]["competitive/train.py"] = "modified"
    elif altered == "base":
        checkpoint["model_storage"]["encoder_provenance"] = {"different_weight": "yes"}
    elif altered == "nan":
        frame.loc[0, TARGETS[0]] = float("nan")
    elif altered == "range":
        frame.loc[0, TARGETS[0]] = 1.1
    elif altered == "uid":
        frame.loc[0, "study_uid"] = frame.loc[1, "study_uid"]
    if altered is None:
        validate_predictions(checkpoint, frame, train, outer, source, provenance)
    else:
        with pytest.raises(ValueError):
            validate_predictions(checkpoint, frame, train, outer, source, provenance)
