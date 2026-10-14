"""Predeclared single-run epoch-weight averaging as an inference-only candidate.

Epochs 6..10 of ONE completed fixed-ten-epoch frozen-base-delta run are averaged
uniformly; epoch 10 remains the primary comparison. Nothing here reads labels,
predictions or evaluation scores, so no epoch or weight can be selected by them.

Rank-stabilized LoRA residuals are averaged in effective-delta space, never by
averaging their factors: with ``s = alpha / sqrt(r)``,

    mean_k(s B_k A_k) = (s / K) [B_1 .. B_K] [A_1; ..; A_K]

which is stored exactly as concatenated factors of rank ``K r`` with scaling
``s / K`` inside the unchanged ``ResidualLinear`` forward (no SVD truncation).
Dense suffix deltas and learned head parameters are linear in the weights and
are averaged directly. Frozen encoder tensors, buffers and register tokens are
never stored here; they must reproduce the bound pretrained hashes. The output
has no optimizer, RNG, scaler or counter state and refuses training resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rsna.competitive.adapters import ResidualLinear
from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.delta_checkpoint import FrozenBaseDelta, _tensor_record

FORMAT = "epoch_weight_average_inference_v1"
FIXED_EPOCHS = 10
AVERAGED_EPOCHS = (6, 7, 8, 9, 10)
PRIMARY_COMPARISON_EPOCH = 10
# Single-epoch pipeline control: reproduces epoch 10 exactly through this code path.
IDENTITY_CONTROL_EPOCHS = (10,)
ROLES = {AVERAGED_EPOCHS: "uniform_epoch6_10_average", IDENTITY_CONTROL_EPOCHS: "epoch10_identity"}
LORA_METHODS = {"rs_lora_v1", "suffix_rs_lora_v1"}
DENSE_METHOD = "dense_suffix_delta_v1"
UNSUPPORTED_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
SELF = "competitive/weight_average.py"
_ATTRIBUTE = "_rsna_epoch_weight_average"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return _hash({name: _tensor_record(state[name]) for name in sorted(state)})


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing, incomplete or non-regular snapshot file: {path.name}")


def encoder_policy(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The single supported residual policy of a verified pretrained encoder."""
    policies = [provenance[key] for key in ("adapter", "suffix_adapter") if key in provenance]
    if provenance.get("pretrained") is not True or len(policies) != 1:
        raise ValueError("Averaging requires exactly one residual adapter on a pretrained base")
    policy = dict(policies[0])
    if policy.get("method") not in LORA_METHODS | {DENSE_METHOD}:
        raise ValueError("Unsupported residual adapter graph for weight averaging")
    if policy["method"] in LORA_METHODS and (
        type(policy.get("rank")) is not int or policy.get("scaling") != "alpha/sqrt(rank)"
    ):
        raise ValueError("LoRA averaging requires the bound alpha/sqrt(rank) scaling")
    if policy["method"] == DENSE_METHOD and policy.get("scaling") != 1.0:
        raise ValueError("Dense suffix averaging requires unit residual scaling")
    return policy


def _validate_binding(binding: Mapping[str, Any]) -> None:
    config = binding["config"]
    if (
        config.get("epochs") != FIXED_EPOCHS
        or config.get("checkpoint_format") != "frozen_base_delta"
        or config.get("freeze_encoder") is not True
        or config.get("submission_every_epochs") != 1
        or config.get("diagnostic_study_limit") is not None
    ):
        raise ValueError(
            "Averaging requires one fixed-ten-epoch frozen-base-delta run with every epoch kept"
        )
    if binding.get("external_initialization") is not None or binding.get("private_research_only"):
        raise ValueError("Private external-initialization weights are not averaged")


def read_snapshot(directory: Path, epoch: int) -> tuple[dict, dict]:
    """Read one immutable completed snapshot after verifying its recorded bytes."""
    path = directory / f"epoch-{epoch:03d}.pt"
    metadata = path.with_suffix(".json")
    _regular(path)
    _regular(metadata)
    record = json.loads(metadata.read_text())
    sha = file_sha256(path)
    if (
        record.get("schema_version") != 1
        or record.get("status") != "scheduled_epoch_snapshot"
        or record.get("epoch") != epoch
        or record.get("checkpoint_file") != path.name
        or record.get("diagnostic_only") is not False
        or record.get("checkpoint_sha256") != sha
        or record.get("checkpoint_bytes") != path.stat().st_size
    ):
        raise ValueError(f"Epoch {epoch} snapshot is incomplete, tampered or not completed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != sha:
        raise ValueError(f"Epoch {epoch} snapshot changed while it was read")
    binding = checkpoint.get("binding")
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("optimizer_boundary") is not True
        or checkpoint.get("scheduled_epoch") is not True
        or checkpoint.get("epoch") != epoch
        or checkpoint.get("next_update") != 0
        or checkpoint.get("step") != record["optimizer_steps"]
        or not isinstance(binding, dict)
        or _hash(binding) != record["binding_sha256"]
        or binding["release"]["binding_sha256"] != record["release_binding_sha256"]
        or binding["config"]["fold"] != record["fold"]
        or binding["config"]["seed"] != record["seed"]
    ):
        raise ValueError(f"Epoch {epoch} snapshot is not its bound completed epoch boundary")
    return record, checkpoint


def _learned_layout(storage: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    schema = storage["learned_tensors"]
    modules = {f"encoder.backbone.{name}" for name in policy["target_modules"]}
    suffixes = (".dense_delta",) if policy["method"] == DENSE_METHOD else (".lora_A", ".lora_B")
    residual = {module + suffix for module in modules for suffix in suffixes}
    encoder = {name for name in schema if name.startswith("encoder.")}
    if encoder != residual:
        raise ValueError("Learned encoder graph differs from the bound residual targets")
    head = sorted(set(schema) - encoder)
    if not head or any(name.endswith(UNSUPPORTED_BUFFER_SUFFIXES) for name in schema):
        raise ValueError("Unsupported learned buffer or empty classifier graph")
    if not storage["frozen_tensors"]:
        raise ValueError("Frozen encoder base is missing")
    if policy["method"] in LORA_METHODS:
        rank = policy["rank"]
        for module in modules:
            a, b = schema[module + ".lora_A"]["shape"], schema[module + ".lora_B"]["shape"]
            if len(a) != 2 or len(b) != 2 or a[0] != rank or b[1] != rank:
                raise ValueError("LoRA factor shapes differ from the bound rank")
    return {"modules": sorted(modules), "head": head}


def _mean(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    first = tensors[0]
    if not first.is_floating_point():
        if any(not torch.equal(first, value) for value in tensors[1:]):
            raise ValueError("A non-floating learned invariant differs between epochs")
        return first.clone()
    return torch.stack([value.to(torch.float64) for value in tensors]).mean(0).to(first.dtype)


def average_snapshots(
    run_directory: Path, epochs: tuple[int, ...] = AVERAGED_EPOCHS
) -> dict[str, Any]:
    """Build the inference-only candidate; refuses anything but the predeclared sets."""
    if epochs not in ROLES:
        raise ValueError("Only the predeclared epoch6-10 average or epoch10 identity is allowed")
    run_path = run_directory / "run.json"
    _regular(run_path)
    run = json.loads(run_path.read_text())
    if (
        run.get("status") != "completed"
        or run.get("diagnostic_run") is not False
        or run.get("held_out_confirmation_and_gold_used") is not False
    ):
        raise ValueError("Only one completed non-diagnostic run without gold can be averaged")
    binding = run["binding"]
    _validate_binding(binding)
    policy = encoder_policy(binding["config"]["encoder"])
    directory = run_directory / "epoch-checkpoints"
    snapshots = [read_snapshot(directory, epoch) for epoch in epochs]
    storage = snapshots[0][1]["model_storage"]
    for _, checkpoint in snapshots:
        if checkpoint["binding"] != binding:
            raise ValueError("Snapshots differ in run/source/config/fold/supervision binding")
        if checkpoint["model_storage"] != storage:
            raise ValueError("Snapshots differ in pretrained base or learned parameter graph")
    if (
        storage.get("format") != "frozen_base_delta_v1"
        or storage["encoder_provenance"] != binding["config"]["encoder"]
    ):
        raise ValueError("Snapshot base provenance differs from the run binding")
    steps = [record["optimizer_steps"] for record, _ in snapshots]
    per_epoch = steps[-1] // epochs[-1]
    if per_epoch <= 0 or steps != [epoch * per_epoch for epoch in epochs]:
        raise ValueError("Snapshot optimizer counters are not one uninterrupted fixed schedule")
    layout = _learned_layout(storage, policy)
    states = [checkpoint["model"] for _, checkpoint in snapshots]
    for state in states:
        if set(state) != set(storage["learned_tensors"]):
            raise ValueError("Snapshot learned tensor membership differs from its schema")
        for name, value in state.items():
            expected = storage["learned_tensors"][name]
            if (
                not isinstance(value, torch.Tensor)
                or list(value.shape) != expected["shape"]
                or str(value.dtype) != expected["dtype"]
                or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
            ):
                raise ValueError("Invalid snapshot learned tensor")
    count = len(epochs)
    model: dict[str, torch.Tensor] = {}
    modules: dict[str, dict[str, Any]] = {}
    for module in layout["modules"]:
        if policy["method"] in LORA_METHODS:
            scaling = policy["alpha"] / math.sqrt(policy["rank"])
            model[module + ".lora_A"] = torch.cat([s[module + ".lora_A"] for s in states], 0)
            model[module + ".lora_B"] = torch.cat([s[module + ".lora_B"] for s in states], 1)
            modules[module] = {
                "source_rank": policy["rank"],
                "rank": count * policy["rank"],
                "source_scaling": scaling,
                "scaling": scaling / count,
            }
        else:
            model[module + ".dense_delta"] = _mean([s[module + ".dense_delta"] for s in states])
            modules[module] = {"source_scaling": 1.0, "scaling": 1.0}
    for name in layout["head"]:
        model[name] = _mean([s[name] for s in states])
    return {
        "format": FORMAT,
        "schema_version": FORMAT,
        "inference_only": True,
        "optimizer_boundary": False,
        "resumable": False,
        "role": ROLES[epochs],
        "source_binding": binding,
        "source_binding_sha256": _hash(binding),
        "source_model_storage": storage,
        "source_snapshots": [
            {
                key: record[key]
                for key in ("epoch", "optimizer_steps", "checkpoint_sha256", "checkpoint_bytes")
            }
            for record, _ in snapshots
        ],
        "averaging": {
            "epochs": list(epochs),
            "weights": [1.0 / count] * count,
            "primary_comparison_epoch": PRIMARY_COMPARISON_EPOCH,
            "selection": "predeclared; no labels, predictions or evaluation scores read",
            "single_run": True,
            "prediction_ensemble": False,
            "backbone_forwards_per_view": 1,
        },
        "representation": {
            "method": policy["method"],
            "lora": "concatenated_factors_exact_effective_delta"
            if policy["method"] in LORA_METHODS
            else None,
            "svd_truncation": False,
            "modules": modules,
        },
        "model": model,
        "model_sha256": _state_sha256(model),
    }


def storage_report(candidate: Mapping[str, Any]) -> dict[str, int]:
    """Parameter counts of the stored candidate versus source and dense equivalents."""
    schema = candidate["source_model_storage"]["learned_tensors"]
    model = candidate["model"]
    residual = [name for name in model if name.startswith("encoder.")]
    head = [name for name in model if not name.startswith("encoder.")]
    dense = 0
    for module in candidate["representation"]["modules"]:
        if candidate["representation"]["method"] == DENSE_METHOD:
            dense += math.prod(schema[module + ".dense_delta"]["shape"])
        else:
            dense += schema[module + ".lora_B"]["shape"][0] * schema[module + ".lora_A"]["shape"][1]
    return {
        "source_residual_parameters": sum(
            math.prod(schema[name]["shape"]) for name in schema if name.startswith("encoder.")
        ),
        "candidate_residual_parameters": sum(model[name].numel() for name in residual),
        "dense_equivalent_parameters": dense,
        "head_parameters": sum(model[name].numel() for name in head),
        "candidate_tensor_bytes": sum(v.numel() * v.element_size() for v in model.values()),
    }


def write_candidate(candidate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Write once next to (never over) the sources; re-verifies source bytes afterwards."""
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Refusing to replace an existing averaged candidate")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("xb") as handle:
        torch.save(dict(candidate), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    temporary.unlink()
    receipt = {
        "schema_version": 1,
        "status": "inference_only_epoch_weight_average_candidate",
        "candidate_file": output.name,
        "candidate_sha256": file_sha256(output),
        "candidate_bytes": output.stat().st_size,
        "role": candidate["role"],
        "source_binding_sha256": candidate["source_binding_sha256"],
        "source_snapshots": candidate["source_snapshots"],
        "averaging": candidate["averaging"],
        "storage": storage_report(candidate),
        "model_sha256": candidate["model_sha256"],
        "clinical_or_Kaggle_gain_claimed": False,
    }
    with output.with_suffix(".json").open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def verify_sources_intact(run_directory: Path, candidate: Mapping[str, Any]) -> None:
    for record in candidate["source_snapshots"]:
        path = run_directory / "epoch-checkpoints" / f"epoch-{record['epoch']:03d}.pt"
        _regular(path)
        if (
            file_sha256(path) != record["checkpoint_sha256"]
            or path.stat().st_size != record["checkpoint_bytes"]
        ):
            raise ValueError("A source epoch snapshot changed")


def load_candidate(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("Averaged candidate bytes differ from the expected receipt")
    candidate = torch.load(path, map_location="cpu", weights_only=True)
    if (
        candidate.get("format") != FORMAT
        or candidate.get("inference_only") is not True
        or candidate.get("optimizer_boundary") is not False
        or any(key in candidate for key in ("optimizer", "rng_by_rank", "amp_scaler", "binding"))
        or _state_sha256(candidate["model"]) != candidate["model_sha256"]
        or _hash(candidate["source_binding"]) != candidate["source_binding_sha256"]
    ):
        raise ValueError("Not an intact inference-only epoch-weight average candidate")
    return candidate


def verify_source_code(binding: Mapping[str, Any], current: Mapping[str, str]) -> None:
    """Training code must be byte-identical; only this module may be newly added."""
    bound = binding["code_sha256"]
    if set(current) - set(bound) - {SELF} or any(current.get(k) != v for k, v in bound.items()):
        raise ValueError("Inference architecture/preprocessing code differs from training")


def apply_candidate(model: nn.Module, candidate: Mapping[str, Any]) -> None:
    """Load into a freshly built source-graph model and freeze it for inference only."""
    binding = candidate["source_binding"]
    provenance = binding["config"]["encoder"]
    guard = FrozenBaseDelta(model, provenance)
    if guard.metadata() != candidate["source_model_storage"]:
        raise ValueError("Model graph, frozen base or encoder provenance differs from the sources")
    if any(
        not name.startswith("encoder.") or name.endswith(UNSUPPORTED_BUFFER_SUFFIXES)
        for name, _ in model.named_buffers()
    ):
        raise ValueError("Unsupported buffer outside the immutable encoder base")
    learned = candidate["model"]
    if set(learned) != set(guard.trainable):
        raise ValueError("Averaged tensor membership differs from the learned graph")
    method = candidate["representation"]["method"]
    modules = dict(model.named_modules())
    targets = candidate["representation"]["modules"]
    for name, info in targets.items():
        module = modules.get(name)
        if method == DENSE_METHOD:
            # Older immutable R32 source trees predate the dense adapter class.
            # Import it only for the separately supported dense source graph.
            from rsna.competitive.adapters import DenseDeltaLinear

            if type(module) is not DenseDeltaLinear:
                raise ValueError("Dense averaged target is not a dense suffix projection")
            continue
        if type(module) is not ResidualLinear or module.scaling != info["source_scaling"]:
            raise ValueError("LoRA averaged target scaling or module differs")
        a, b = learned[name + ".lora_A"], learned[name + ".lora_B"]
        if a.shape != (info["rank"], module.in_features) or b.shape != (
            module.out_features,
            info["rank"],
        ):
            raise ValueError("Concatenated factor shapes differ from the bound projection")
    state = guard._check_partition(model)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in learned.items():
            target = parameters[name]
            if value.dtype != target.dtype:
                raise ValueError("Averaged tensor dtype differs from the learned graph")
            if value.shape == target.shape:
                target.copy_(value)
            else:
                module_name, attribute = name.rsplit(".", 1)
                setattr(
                    modules[module_name],
                    attribute,
                    nn.Parameter(value.to(target.device).clone(), requires_grad=False),
                )
        for name, info in targets.items():
            if method in LORA_METHODS:
                modules[name].scaling = info["scaling"]
    model.requires_grad_(False)
    model.eval()
    after = model.state_dict()
    if {n: _tensor_record(after[n]) for n in guard.frozen} != guard.initial_frozen or set(
        after
    ) != set(state):
        raise ValueError("Applying the average changed the frozen base")
    setattr(
        model,
        _ATTRIBUTE,
        {
            "role": candidate["role"],
            "source_binding_sha256": candidate["source_binding_sha256"],
            "model_sha256": candidate["model_sha256"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed training output root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--epoch10-identity-control",
        action="store_true",
        help="write the single-epoch control instead of the predeclared epoch6-10 average",
    )
    args = parser.parse_args()
    epochs = IDENTITY_CONTROL_EPOCHS if args.epoch10_identity_control else AVERAGED_EPOCHS
    candidate = average_snapshots(args.run, epochs)
    receipt = write_candidate(candidate, args.output)
    verify_sources_intact(args.run, candidate)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
