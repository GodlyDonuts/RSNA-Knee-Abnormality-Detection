"""CPU-only official-weight geometry diagnostic; no disease labels or accuracy claims."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import random
import tarfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.checkpoint import (
    atomic_json,
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from rsna.competitive.data import _indices, _positions
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.relations import (
    effective_relation_loss,
    relation_model_kwargs,
    relation_study_count,
    relations_for_batch,
)
from rsna.competitive.train import TrainConfig, _model_inputs, learning_rate_multiplier
from rsna.data import PLANES
from scripts.newton.audit_relation_geometry import digest, signature


def validate_source(request, proof, run, archive, series):
    if (
        signature(archive) != proof["assets"][str(archive)]["stat"]
        or request["training_uids_sha256"] != digest(sorted(request["training_uids"]))
        or request["training_uids_sha256"] != run["binding"]["training_uids_sha256"]
        or len(request["training_uids"]) != run["training_studies"]
        or len(set(request["training_uids"])) != len(request["training_uids"])
        or request["source_revision"] != proof["data"]["release_binding"]["source_revision"]
        or request["series_table_sha256"] != run["binding"]["release"]["series_sha256"]
        or hashlib.sha256(series.read_bytes()).hexdigest() != request["series_table_sha256"]
    ):
        raise ValueError("Diagnostic inputs differ from the accepted training-only source")
    selected = request["series"][0]
    if selected["study_uid"] not in request["training_uids"] or selected["n_slices"] < 6:
        raise ValueError("A genuine training series with at least six slices is required")
    return selected


def volume_from_archive(archive, selected, source_revision):
    stem = f"cache/train_series/{selected['study_uid']}/{selected['series_uid']}"
    found = {}
    with tarfile.open(archive, "r:") as source:
        for member in source:
            if member.name not in {stem + ".json", stem + ".npz"}:
                continue
            if (
                member.name in found
                or not member.isfile()
                or member.size > (2 * 1024**2 if member.name.endswith(".json") else 64 * 1024**2)
            ):
                raise ValueError("Invalid diagnostic metadata/volume member")
            found[member.name] = source.extractfile(member).read()
            if len(found) == 2:
                break
    if len(found) != 2:
        raise ValueError("Accepted diagnostic series is missing")
    metadata = json.loads(found[stem + ".json"])
    payload = found[stem + ".npz"]
    if (
        metadata.get("study_uid") != selected["study_uid"]
        or metadata.get("series_uid") != selected["series_uid"]
        or metadata.get("source_revision") != source_revision
        or metadata.get("n_slices") != selected["n_slices"]
        or metadata.get("side") != 256
        or metadata.get("split") != "train_series"
        or metadata.get("cache_path") != stem.removeprefix("cache/") + ".npz"
        or metadata.get("cache_bytes") != len(payload)
        or metadata.get("cache_sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError("Diagnostic cache source/identity/bytes differ")
    with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
        volume = cached["volume"]
    if volume.dtype != np.uint16 or volume.shape != (selected["n_slices"], 256, 256):
        raise ValueError("Invalid diagnostic volume shape/dtype")
    return volume, metadata, hashlib.sha256(found[stem + ".json"]).hexdigest()


def real_batch(volume, metadata, selected, table):
    rows = table[
        table.StudyInstanceUID.eq(selected["study_uid"])
        & table.SeriesInstanceUID.eq(selected["series_uid"])
    ]
    if len(rows) != 1:
        raise ValueError("Exactly one accepted series-table row required")
    row = rows.iloc[0]
    if row.Anatomical_Plane != selected["plane"] or any(
        row[key] not in (0, 1) for key in ("Fluid_Sensitive", "Fat_Suppression")
    ):
        raise ValueError("Diagnostic protocol differs from the accepted selected series")
    indices = _indices(len(volume), 6, False)
    positions = _positions(metadata, len(volume))[indices]
    if not positions[:, 1].eq(1).all():
        raise ValueError("Known physical coordinates are required for this diagnostic")
    images = torch.zeros(1, 3, 6, 1, 32, 32)
    sampled = torch.from_numpy(volume[indices].astype(np.float32))[:, None] / 65535
    images[0, 0] = F.interpolate(
        sampled, (32, 32), mode="bilinear", align_corners=False, antialias=True
    ).clamp_(0, 1)
    physical = torch.zeros(1, 3, 6, 2)
    physical[0, 0] = positions
    features = torch.zeros(1, 3, 5)
    features[0, 0, PLANES[row.Anatomical_Plane]] = 1
    features[0, 0, 3:] = torch.tensor([row.Fluid_Sensitive, row.Fat_Suppression])
    return {
        "images": images,
        "positions": physical,
        "features": features,
        "series_mask": torch.tensor([[True, False, False]]),
        "slice_mask": torch.tensor([[[True] * 6, [False] * 6, [False] * 6]]),
        "sample_valid": torch.tensor([True]),
        "study_uid": [selected["study_uid"]],
        # No disease values are read or fabricated; every disease cell is masked.
        "labels": torch.zeros(1, 12),
        "label_mask": torch.zeros(1, 12),
        "source_slice_indices": indices.tolist(),
    }


def tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(tree_equal(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(tree_equal(a, b) for a, b in zip(first, second))
    return first == second


def initialized(config):
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    encoder, dimension, provenance = create_encoder(config.encoder)
    configure_encoder_training(encoder, config.freeze_encoder)
    network = CompetitiveStudyNet(
        encoder,
        dimension,
        pooling=config.pooling,
        hidden_dim=config.hidden_dim,
        encode_chunk_size=config.encode_chunk_size,
        slice_context=config.slice_context,
        **relation_model_kwargs(config.relation),
    )
    encoder_parameters = {id(p) for p in encoder.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for p in network.parameters()
                    if p.requires_grad and id(p) not in encoder_parameters
                ],
                "lr": config.head_lr,
            },
            {
                "params": [p for p in encoder.parameters() if p.requires_grad],
                "lr": config.encoder_lr,
            },
        ],
        weight_decay=config.weight_decay,
    )
    return network, optimizer, FrozenBaseDelta(network, provenance)


def update(network, optimizer, data, config, step):
    network.train()
    optimizer.zero_grad(set_to_none=True)
    relation = relations_for_batch(data, config.relation, seed=config.seed, epoch=step)
    logits, relation_sum = network(*_model_inputs(data, torch.device("cpu")), relations=relation)
    classification = effective_batch_loss(
        logits, data["labels"], data["label_mask"], observed_counts([data["label_mask"]])
    )
    auxiliary = effective_relation_loss(
        relation_sum, relation_study_count([relation], torch.device("cpu"))
    )
    loss = classification + config.relation["weight"] * auxiliary
    loss.backward()
    gradients = [p.grad for p in network.encoder.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in gradients) or not any(
        g.abs().sum() > 0 for g in gradients
    ):
        raise ValueError("Real-image relation loss did not produce finite adapter gradients")
    norm = torch.nn.utils.clip_grad_norm_(
        network.parameters(), config.clip_grad_norm, error_if_nonfinite=True
    )
    multiplier = learning_rate_multiplier(step, 2, config.warmup_updates, config.lr_schedule)
    for group, base_lr in zip(optimizer.param_groups, (config.head_lr, config.encoder_lr)):
        group["lr"] = base_lr * multiplier
    optimizer.step()
    return {
        "step": step + 1,
        "classification_loss": classification.item(),
        "relation_loss": auxiliary.item(),
        "weighted_loss": loss.item(),
        "gradient_norm": norm.item(),
        "relation_pairs": len(relation.indices),
        "represented_studies": relation.study_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "request", "proof", "run", "archive", "series", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a distinct diagnostic output; never overwrite a prior attempt")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    request, proof, run = (json.loads(p.read_text()) for p in (args.request, args.proof, args.run))
    selected = validate_source(request, proof, run, args.archive, args.series)
    volume, metadata, metadata_sha = volume_from_archive(
        args.archive, selected, request["source_revision"]
    )
    data = real_batch(volume, metadata, selected, pd.read_csv(args.series))
    del volume
    config = replace(
        TrainConfig(**json.loads(args.config.read_text())),
        side=32,
        slices=6,
        max_series=3,
        accumulation_steps=1,
        encode_chunk_size=6,
        precision="fp32",
        diagnostic_study_limit=1,
    )
    config.validate()
    if (
        config.relation is None
        or config.relation["target_mode"] != "physical"
        or config.relation["weight"] <= 0
    ):
        raise ValueError("This diagnostic requires a positive physical policy")
    args.output.mkdir(mode=0o700)
    network, optimizer, guard = initialized(config)
    print(
        json.dumps(
            {
                "status": "official_model_initialized",
                "frozen_tensors": len(guard.frozen),
                "learned_tensors": len(guard.trainable),
            }
        ),
        flush=True,
    )
    original_b = {
        name: p.detach().clone()
        for name, p in network.named_parameters()
        if name.endswith(".lora_B")
    }
    record1 = update(network, optimizer, data, config, 0)
    print(json.dumps({"phase": "before_boundary", **record1}), flush=True)
    if not any(
        not torch.equal(dict(network.named_parameters())[name], old)
        for name, old in original_b.items()
    ):
        raise ValueError("No adapter residual changed after actual optimization")
    binding = {
        "world_size": 1,
        "device_type": "cpu",
        "config": asdict(config),
        "training_uids_sha256": request["training_uids_sha256"],
        "metadata_sha256": metadata_sha,
        "diagnostic_only": True,
        "source_slice_indices": data["source_slice_indices"],
    }
    boundary = {
        **guard.checkpoint_model(network),
        "binding": binding,
        "optimizer_boundary": True,
        "optimizer": optimizer.state_dict(),
        "rng_by_rank": [capture_rng(device)],
    }
    save_checkpoint(args.output / "boundary.pt", boundary, min_free_gib=20)
    del boundary, original_b
    record2 = update(network, optimizer, data, config, 1)
    print(json.dumps({"phase": "uninterrupted_final", **record2}), flush=True)
    expected = copy.deepcopy(guard.checkpoint_model(network))
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_rng = capture_rng(device)
    network.eval()
    with torch.inference_mode():
        expected_logits = network(*_model_inputs(data, device)).clone()
    del network, optimizer, guard
    gc.collect()
    network, optimizer, guard = initialized(config)
    loaded = load_checkpoint(args.output / "boundary.pt", binding)
    guard.restore(network, loaded)
    optimizer.load_state_dict(loaded["optimizer"])
    restore_rng(loaded["rng_by_rank"][0], device)
    replay = update(network, optimizer, data, config, 1)
    print(json.dumps({"phase": "resumed_final", **replay}), flush=True)
    if not (
        tree_equal(expected, guard.checkpoint_model(network))
        and tree_equal(expected_optimizer, optimizer.state_dict())
        and tree_equal(expected_rng, capture_rng(device))
        and tree_equal(record2, replay)
    ):
        raise ValueError("Official-weight real-MRI decoded recovery differs")
    network.eval()
    network.relation_head.forward = lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("Auxiliary head executed during classification")
    )
    with torch.inference_mode():
        actual_logits = network(*_model_inputs(data, device))
    if (
        actual_logits.shape != (1, 12)
        or not torch.isfinite(actual_logits).all()
        or not torch.equal(actual_logits, expected_logits)
    ):
        raise ValueError("MRI-only diagnostic inference differs after recovery")
    if signature(args.archive) != proof["assets"][str(args.archive)]["stat"]:
        raise ValueError("Source archive changed during the diagnostic")
    atomic_json(
        args.output / "verified.json",
        {
            "status": "official_weight_real_MRI_CPU_relation_recovery_verified",
            "diagnostic_only": True,
            "disease_labels_read": False,
            "clinical_accuracy_evidence": False,
            "GPU_BF16_full_input_acceptance": False,
            "config": asdict(config),
            "binding": binding,
            "updates": [record1, record2],
            "frozen_tensor_count": len(guard.frozen),
            "learned_tensor_count": len(guard.trainable),
            "exact_decoded_model_optimizer_rng_recovery": True,
            "MRI_only_logits_exact": True,
            "checkpoint_bytes": (args.output / "boundary.pt").stat().st_size,
            "checkpoint_sha256": hashlib.sha256(
                (args.output / "boundary.pt").read_bytes()
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
