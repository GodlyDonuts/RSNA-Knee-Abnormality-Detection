"""Training-only full-input relation gradient diagnostic; never a scored model.

Use one allocated H100 after source/cache/runtime admission. All three policies
share the same parameters, observations and dropout stream. No optimizer step,
outer-fold prediction, checkpoint initialization or submission is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tarfile
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.checkpoint import capture_rng, file_sha256, restore_rng
from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
from rsna.competitive.delta_checkpoint import FrozenBaseDelta, _tensor_record
from rsna.competitive.folds import load_folds
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.relations import (
    effective_relation_loss,
    relation_model_kwargs,
    relation_study_count,
    relations_for_batch,
)
from rsna.competitive.train import TrainConfig, _model_inputs, _uid_hash, select_development
from rsna.release import load_verified_release
from scripts.newton.audit_relation_geometry import digest, signature


def matched_configs(root):
    configs = {
        name: TrainConfig(**json.loads((root / f"{name}-pilot-config.json").read_text()))
        for name in ("zero", "physical", "shuffled")
    }
    for config in configs.values():
        config.validate()
    if any(config.relation is None for config in configs.values()):
        raise ValueError("All matched arms require the same auxiliary head")
    reference = asdict(configs["physical"])
    reference.pop("relation")
    policy = configs["physical"].relation
    for name, config in configs.items():
        common = asdict(config)
        other_policy = common.pop("relation")
        expected = {
            **policy,
            "weight": 0 if name == "zero" else policy["weight"],
            "target_mode": "shuffled" if name == "shuffled" else "physical",
        }
        if common != reference or other_policy != expected:
            raise ValueError("Matched arms may differ only in relation weight/target mode")
    config = configs["physical"]
    if (
        policy["weight"] <= 0
        or config.precision != "bf16"
        or (config.side, config.slices, config.max_series) != (224, 16, 6)
        or (config.batch_size, config.accumulation_steps, config.diagnostic_study_limit)
        != (1, 8, 16)
        or config.augment
        or config.slice_sampling != "uniform"
        or config.series_sampling != "selected"
        or config.fit_scope != "outer_fold"
        or config.encoder["architecture"] != "dinov3_l16"
    ):
        raise ValueError("This diagnostic requires the admitted full224 BF16 paired pilot")
    return configs


def gradient_fingerprint(values):
    return digest([_tensor_record(value) for value in values])


def validate_cache(cache, archive, proof, uids, series):
    """Bind local pixels/geometry to the previously whole-byte-verified archive.

    The dataset verifies its metadata/volume envelopes too, but a self-consistent
    replacement envelope must not be able to impersonate the original source.
    Inspect all series metadata for these training studies, including alternates
    the dataset admits but does not select. No outer-study metadata is read.
    """
    if signature(archive) != proof["assets"][str(archive)]["stat"]:
        raise ValueError("Original accepted cache archive changed")
    entries = series[series.StudyInstanceUID.isin(uids)]
    expected = {
        f"cache/train_series/{row.StudyInstanceUID}/{row.SeriesInstanceUID}.json"
        for row in entries.itertuples()
    }
    if len(expected) != len(entries) or set(entries.StudyInstanceUID) != set(uids):
        raise ValueError("Missing/duplicate series for admitted training studies")
    seen, members = set(), {}
    with tarfile.open(archive, "r:") as source:
        for member in source:
            if member.name not in expected:
                continue
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or member.name in seen
                or member.size > 2 * 1024**2
                or ".." in name.parts
                or name.is_absolute()
            ):
                raise ValueError("Invalid original diagnostic cache metadata member")
            raw = source.extractfile(member).read()
            meta = json.loads(raw)
            relative = member.name.removeprefix("cache/")
            path = cache / relative
            volume = path.with_suffix(".npz")
            if (
                path.is_symlink()
                or volume.is_symlink()
                or path.read_bytes() != raw
                or meta["cache_path"] != relative.removesuffix(".json") + ".npz"
                or volume.stat().st_size != meta["cache_bytes"]
                or file_sha256(volume) != meta["cache_sha256"]
            ):
                raise ValueError("Node-local pixels/geometry differ from original verified cache")
            members[relative] = {"bytes": len(raw), "sha256": file_sha256(path)}
            members[meta["cache_path"]] = {
                "bytes": meta["cache_bytes"],
                "sha256": meta["cache_sha256"],
            }
            seen.add(member.name)
    if seen != expected or signature(archive) != proof["assets"][str(archive)]["stat"]:
        raise ValueError("Incomplete original diagnostic cache membership")
    return {
        "original_archive_sha256_previously_verified": proof["assets"][str(archive)]["sha256"],
        "members": members,
        "members_sha256": digest(members),
        "training_only_studies": len(uids),
    }


def gradient_summary(classification, auxiliary, weight, clip):
    if (
        not classification
        or len(classification) != len(auxiliary)
        or not math.isfinite(weight)
        or weight < 0
        or not math.isfinite(clip)
        or clip <= 0
    ):
        raise ValueError("Aligned gradients and finite nonnegative weight/positive clip required")
    squares = [0.0, 0.0]
    dot = 0.0
    for first, second in zip(classification, auxiliary):
        if first.shape != second.shape or not all(torch.isfinite(v).all() for v in (first, second)):
            raise ValueError("Gradient shape/finite-value mismatch")
        a, b = first.detach().double(), second.detach().double()
        squares[0] += a.square().sum().item()
        squares[1] += b.square().sum().item()
        dot += (a * b).sum().item()
    first_norm, second_norm = map(math.sqrt, squares)
    combined = math.sqrt(max(0.0, squares[0] + weight**2 * squares[1] + 2 * weight * dot))
    return {
        "classification_norm": first_norm,
        "unweighted_relation_norm": second_norm,
        "weighted_relation_norm": weight * second_norm,
        "weighted_relation_to_classification_ratio": weight * second_norm / first_norm
        if first_norm
        else None,
        "unweighted_gradient_cosine": max(-1.0, min(1.0, dot / (first_norm * second_norm)))
        if first_norm and second_norm
        else None,
        "combined_norm": combined,
        # This only concerns the shared encoder. Real trainer clipping includes
        # classification and relation heads too; do not present this as its clip.
        "encoder_only_clip_multiplier": min(1.0, clip / combined) if combined else 1.0,
    }


def measure(model, batches, config, device):
    """Measure accumulated encoder gradients with the trainer's normalization."""
    named = [(name, p) for name, p in model.encoder.named_parameters() if p.requires_grad]
    if not batches or not named:
        raise ValueError("Real effective batches and learned encoder parameters required")
    parameters = [p for _, p in named]
    accumulated = [[torch.zeros_like(p, device="cpu") for p in parameters] for _ in range(2)]
    relations = [
        relations_for_batch(batch, config.relation, seed=config.seed, epoch=0) for batch in batches
    ]
    counts = observed_counts([batch["label_mask"].to(device) for batch in batches])
    relation_count = relation_study_count(relations, device)
    model.train()
    started = time.monotonic()
    losses = [0.0, 0.0]
    logits_records = []
    for batch, relation in zip(batches, relations):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
            logits, relation_sum = model(*_model_inputs(batch, device), relations=relation)
            classification = effective_batch_loss(
                logits, batch["labels"].to(device), batch["label_mask"].to(device), counts
            )
            auxiliary = effective_relation_loss(relation_sum, relation_count)
        for objective, loss in enumerate((classification, auxiliary)):
            grads = torch.autograd.grad(
                loss, parameters, retain_graph=objective == 0, allow_unused=False
            )
            for total, grad in zip(accumulated[objective], grads):
                if not torch.isfinite(grad).all():
                    raise ValueError("Nonfinite real-image diagnostic encoder gradient")
                total.add_(grad.detach().cpu())
            losses[objective] += loss.detach().item()
        logits_records.append(_tensor_record(logits.detach().float()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "classification_loss": losses[0],
        "relation_loss": losses[1],
        "weighted_relation_loss": config.relation["weight"] * losses[1],
        "encoder_gradients": gradient_summary(
            *accumulated, config.relation["weight"], config.clip_grad_norm
        ),
        "classification_gradient_sha256": gradient_fingerprint(accumulated[0]),
        "classification_logits_sha256": digest(logits_records),
        "pair_roster_sha256": digest([_tensor_record(r.indices) for r in relations]),
        "target_roster_sha256": digest([_tensor_record(r.targets) for r in relations]),
        "observed_target_counts": counts.cpu().tolist(),
        "represented_studies": int(relation_count.item()),
        "relation_pairs": sum(len(r.indices) for r in relations),
        "microbatches": len(batches),
        "elapsed_seconds_including_two_gradient_passes": time.monotonic() - started,
    }
    if not all(math.isfinite(value) for value in losses):
        raise ValueError("Nonfinite diagnostic objective")
    return result


def admitted_batches(args, config):
    proof, request, run = (
        json.loads(path.read_text()) for path in (args.proof, args.request, args.run)
    )
    for name, row in proof["assets"].items():
        if signature(Path(name)) != row["stat"]:
            raise ValueError("An originally byte-verified asset has changed")
    manifest, binding = load_verified_release(args.studies, args.artifact_root)
    folds = load_folds(args.folds, manifest, binding)
    train, _ = select_development(manifest, folds, config.fold, config.fit_scope)
    if (
        _uid_hash(train) != request["training_uids_sha256"]
        or _uid_hash(train) != run["binding"]["training_uids_sha256"]
        or _uid_hash(train) != digest(sorted(request["training_uids"]))
        or binding != proof["data"]["release_binding"]
        or binding != run["binding"]["release"]
        or file_sha256(args.series) != binding["series_sha256"]
        or len(train) != run["training_studies"]
        or len(set(request["training_uids"])) != len(train)
    ):
        raise ValueError("Diagnostic labels/roster differ from the accepted training pool")
    selected = train.head(config.diagnostic_study_limit)
    if len(selected) != 16:
        raise ValueError("Exactly sixteen actual training studies are required")
    series = pd.read_csv(args.series)
    cache_binding = validate_cache(
        args.cache, args.archive, proof, selected.study_uid.tolist(), series
    )
    dataset = CompetitiveStudyDataset(
        selected,
        series,
        args.cache,
        **{
            name: getattr(config, name)
            for name in (
                "slices",
                "side",
                "max_series",
                "augment",
                "slice_sampling",
                "series_sampling",
            )
        },
    )
    if dataset.cache_side != 256 or dataset.source_revision != binding["source_revision"]:
        raise ValueError("Diagnostic image transform/source differs from the accepted cache")
    batches = [collate_studies([dataset[i]]) for i in range(len(dataset))]
    for batch in batches:
        batch["sample_valid"] = torch.ones(1, dtype=torch.bool)
    input_binding = {
        "training_uids_sha256": _uid_hash(train),
        "probe_uids_sha256": _uid_hash(selected),
        "node_local_original_cache_binding": cache_binding,
        "source_probe_sha256": file_sha256(Path(__file__)),
        "input_file_sha256": {
            name: file_sha256(getattr(args, name))
            for name in ("proof", "request", "run", "studies", "series", "source_manifest")
        },
        "batch_input_sha256": [
            digest(
                {
                    name: _tensor_record(batch[name])
                    for name in (
                        "images",
                        "positions",
                        "features",
                        "series_mask",
                        "slice_mask",
                        "labels",
                        "label_mask",
                    )
                }
            )
            for batch in batches
        ],
        "torch_version": str(torch.__version__),
    }
    return batches, input_binding


def admit_cpu_receipt(receipt, input_binding, configs):
    if (
        receipt.get("status") != "actual_full224_training_inputs_CPU_verified"
        or receipt.get("optimizer_updates") != 0
        or receipt.get("GPU_acceptance") is not False
        or receipt.get("input_binding") != input_binding
        or receipt.get("configs") != {name: asdict(value) for name, value in configs.items()}
    ):
        raise ValueError("GPU diagnostic inputs/source differ from actual CPU admission")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config-root",
        "proof",
        "request",
        "run",
        "studies",
        "artifact-root",
        "folds",
        "series",
        "cache",
        "archive",
        "output",
        "source-manifest",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu-input-only", action="store_true")
    parser.add_argument("--cpu-acceptance", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Diagnostic output already exists; reconcile instead of overwriting")
    if args.cpu_input_only:
        if (
            not os.environ.get("SLURM_JOB_ID", "").isdecimal()
            or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or os.environ.get("SLURM_JOB_GPUS")
            or torch.cuda.device_count() != 0
        ):
            raise ValueError("CPU input admission requires one CPU and no GPU allocation")
        torch.set_num_threads(1)
    else:
        if (
            not os.environ.get("SLURM_JOB_ID", "").isdecimal()
            or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
            or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or not torch.cuda.is_bf16_supported()
            or "H100" not in torch.cuda.get_device_name(0)
            or int(os.environ.get("WORLD_SIZE", "1")) != 1
        ):
            raise ValueError("One real H100/two-CPU allocation with BF16 is required")
        torch.set_num_threads(2)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    configs = matched_configs(args.config_root)
    config = configs["physical"]
    batches, input_binding = admitted_batches(args, config)
    if args.cpu_input_only:
        output = {
            "status": "actual_full224_training_inputs_CPU_verified",
            "diagnostic_only": True,
            "optimizer_updates": 0,
            "GPU_acceptance": False,
            "clinical_accuracy_evidence": False,
            "input_binding": input_binding,
            "configs": {name: asdict(value) for name, value in configs.items()},
        }
        with args.output.open("x") as handle:
            json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": output["status"], "studies": len(batches)}), flush=True)
        return
    if args.cpu_acceptance is None:
        raise ValueError("Actual CPU input admission is required before GPU measurement")
    admit_cpu_receipt(json.loads(args.cpu_acceptance.read_text()), input_binding, configs)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    encoder, dimension, provenance = create_encoder(config.encoder)
    configure_encoder_training(encoder, config.freeze_encoder)
    model = CompetitiveStudyNet(
        encoder,
        dimension,
        pooling=config.pooling,
        hidden_dim=config.hidden_dim,
        encode_chunk_size=config.encode_chunk_size,
        slice_context=config.slice_context,
        **relation_model_kwargs(config.relation),
    ).to(device)
    guard = FrozenBaseDelta(model, provenance)
    initial = {
        name: _tensor_record(value)
        for name, value in guard.checkpoint_model(model)["model"].items()
    }
    rng = capture_rng(device)
    results = {}
    for name, policy_config in configs.items():
        restore_rng(rng, device)
        torch.cuda.reset_peak_memory_stats(device)
        results[name] = [
            measure(model, batches[start : start + 8], policy_config, device)
            for start in range(0, len(batches), 8)
        ]
        learned = guard.checkpoint_model(model)["model"]
        if {key: _tensor_record(value) for key, value in learned.items()} != initial:
            raise ValueError("Diagnostic gradient measurement mutated learned parameters")
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Diagnostic gradient measurement contaminated trainer gradients")
        for previous, current in zip(results["zero"], results[name]):
            if any(
                previous[key] != current[key]
                for key in (
                    "classification_gradient_sha256",
                    "classification_logits_sha256",
                    "pair_roster_sha256",
                    "observed_target_counts",
                )
            ):
                raise ValueError(
                    "Matched arms changed classification initialization/dropout/observations"
                )
    # Normal classification must not execute the training-only auxiliary head.
    model.eval()
    model.relation_head.forward = lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("Auxiliary inference")
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(*_model_inputs(batches[0], device))
    if logits.shape != (1, 12) or not torch.isfinite(logits).all():
        raise ValueError("MRI-only full-input classification failed")
    output = {
        "status": "actual_official_weight_full224_BF16_training_gradient_measurement_complete",
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "optimizer_updates": 0,
        "recovery_acceptance": False,
        "input_binding": input_binding,
        "CPU_acceptance_sha256": file_sha256(args.cpu_acceptance),
        "encoder_provenance": provenance,
        "frozen_tensor_count": len(guard.frozen),
        "initial_learned_state_sha256": digest(initial),
        "configs": {name: asdict(value) for name, value in configs.items()},
        "results": results,
        "peak_gpu_allocated_bytes_last_arm": torch.cuda.max_memory_allocated(device),
        "GPU": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "same_classification_gradients_logits_pairs_across_arms": True,
        "MRI_only_12_output_inference": True,
        "timing_is_extra_diagnostic_backward_work_not_training_throughput": True,
    }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": output["status"], "optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
