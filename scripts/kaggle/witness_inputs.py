"""Actual complete-release and original-MRI CPU admission for private T4 training.

Runs against the independently bound frozen scientific source, not the mounted
historical inference package. No GPU, optimizer or clinical evaluation occurs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.folds import load_folds
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig, _uid_hash, select_development
from rsna.preprocess import Source, cache_series
from rsna.release import load_verified_release


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 2**20), b""):
            h.update(b)
    return h.hexdigest()


def kaggle_storage_budget(path, pending_cache_bytes=2 * 2**30):
    """Kaggle scratch has a separate budget from retained cluster storage."""
    usage = shutil.disk_usage(path)
    if usage.free < 3 * 2**30 + pending_cache_bytes:
        raise OSError("Kaggle scratch cannot fund the cache while preserving3GiB")
    return {
        "filesystem_total_bytes": usage.total,
        "free_before_bytes": usage.free,
        "scratch_reserve_bytes": 3 * 2**30,
        "cache_cap_bytes": pending_cache_bytes,
        "cluster_shared_and_node_reserve_changed": False,
    }


def extract_inputs(directory, destination, expected_archive, expected_admission):
    admission = directory / "CPU-admission.json"
    if file_sha(admission) != expected_admission:
        raise ValueError("Original complete-release CPU admission differs")
    proof = json.loads(admission.read_text())
    archive = directory / "training-inputs.bin"
    if (
        proof["status"]
        != "actual_complete_original_release_and_scientific_source_portable_CPU_replay_accepted"
        or proof["archive_sha256"] != expected_archive
        or archive.stat().st_size != proof["archive_bytes"]
        or file_sha(archive) != expected_archive
        or proof["no_pixels_or_model_weights_in_archive"] is not True
    ):
        raise ValueError("Bounded original source archive differs")
    expected = proof["files"]
    if len(expected) > 1536 or sum(v["bytes"] for v in expected.values()) > 512 * 2**20:
        raise ValueError("Complete metadata archive exceeds its budget")
    destination.mkdir(mode=0o700)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Exact archive membership required")
        for member in members:
            path = PurePosixPath(member.name)
            row = expected[member.name]
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != member.name
                or not member.isfile()
                or member.size != row["bytes"]
            ):
                raise ValueError("Unsafe source member")
            raw = handle.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Source member bytes differ")
            target = destination / member.name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(raw)
    return proof


def main():
    parser = argparse.ArgumentParser()
    for n in ("control", "encoder-spec", "competition", "output"):
        parser.add_argument("--" + n, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    storage = kaggle_storage_budget(args.output)
    proof = json.loads((args.control / "original-input-proof.json").read_text())
    artifact = args.control / "artifacts"
    manifest, binding = load_verified_release(
        artifact / "initiative/releases/report-supervision-v1/studies.parquet", artifact
    )
    folds = load_folds(artifact / "initiative/experiments/competitive-v1/folds", manifest, binding)
    train, outer = select_development(manifest, folds, 0)
    if len(train) != 2247 or len(outer) != 1156:
        raise ValueError("Original fitting/development membership differs")
    train, outer = train.head(16), outer.head(16)
    if (
        _uid_hash(train) != proof["training_16_sha256"]
        or _uid_hash(outer) != proof["outer_16_sha256"]
    ):
        raise ValueError("Original engineering roster differs")
    series_path = (
        artifact / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    )
    if file_sha(series_path) != binding["series_sha256"]:
        raise ValueError("Original source sampling differs")
    series = pd.read_csv(series_path)
    ids = set(train.study_uid) | set(outer.study_uid)
    selected = series.loc[series.StudyInstanceUID.isin(ids)]
    cache = args.output / "cache"
    source = Source(args.competition)
    for i, row in enumerate(selected.itertuples(index=False)):
        folder = args.competition / "train_series" / row.StudyInstanceUID / row.SeriesInstanceUID
        paths = sorted(str(p.relative_to(args.competition)) for p in folder.glob("*.dcm"))
        if not paths:
            raise ValueError("An original engineering MRI series is absent")
        written = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
        next_upper = len(paths) * 384 * 384 * 2 + 2**20
        if written + next_upper > storage["cache_cap_bytes"]:
            raise OSError("Original MRI cache would exceed its funded2GiB cap")
        cache_series(source, paths, cache, 384, binding["source_revision"], min_free_gib=3)
        print(json.dumps({"stage": "original_MRI_cached", "series_completed": i + 1}), flush=True)
    tensor_hashes, other_hashes = {}, {}
    for role, frame in [("training", train), ("outer_engineering", outer)]:
        dataset = CompetitiveStudyDataset(
            frame, series, cache, slices=16, side=384, max_series=6, augment=False
        )
        image, other = hashlib.sha256(), hashlib.sha256()
        for i in range(16):
            item = dataset[i]
            image.update(item["images"].numpy().tobytes())
            for name in (
                "features",
                "positions",
                "slice_mask",
                "series_mask",
                "labels",
                "label_mask",
            ):
                other.update(item[name].numpy().tobytes())
        tensor_hashes[role], other_hashes[role] = image.hexdigest(), other.hexdigest()
    if (
        tensor_hashes != proof["input_tensor_sha256"]
        or other_hashes != proof["nonimage_tensor_sha256"]
    ):
        raise ValueError("Actual original MRI tensors or labels/masks differ from Newton admission")
    original_spec = json.loads(args.encoder_spec.read_text())
    initial = {}
    for mode in ("control", "candidate"):
        values = json.loads((args.control / (mode + ".json")).read_text())
        values["encoder"] = original_spec
        c = TrainConfig(**values)
        c.validate()
        if c.precision != "fp32" or c.encode_chunk_size != 1:
            raise ValueError("Explicit FP32 T4 recovery recipe required")
        random.seed(c.seed)
        np.random.seed(c.seed % 2**32)
        torch.manual_seed(c.seed)
        encoder, dim, provenance = create_encoder(c.encoder)
        configure_encoder_training(encoder, c.freeze_encoder)
        model = CompetitiveStudyNet(
            encoder,
            dim,
            pooling=c.pooling,
            hidden_dim=c.hidden_dim,
            encode_chunk_size=c.encode_chunk_size,
            slice_context=c.slice_context,
            witness=c.witness,
        )
        records = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                a = p.detach().contiguous().numpy()
                records[n] = {
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                }
        value = {
            "learned_graph_and_values_sha256": digest(records),
            "post_factory_torch_RNG_sha256": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            "learned_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_provenance_sha256": digest(provenance),
        }
        if value != proof["initial_learned_state"][mode]:
            raise ValueError("Actual official initial model/RNG differs")
        initial[mode] = value
        with (args.output / (mode + "-runtime-config.json")).open("x") as f:
            json.dump(values, f, sort_keys=True, indent=2)
            f.write("\n")
        del model, encoder, records
        gc.collect()
    result = {
        "status": "actual_private_Kaggle_original32_MRI_full_release_and_official_initial_state_CPU_accepted",
        "training_studies": 16,
        "engineering_outer_studies": 16,
        "complete_fitting_membership": 2247,
        "complete_outer_membership": 1156,
        "input_tensors_equal_to_original": True,
        "labels_masks_equal_to_original": True,
        "initial_learned_state": initial,
        "release_binding": binding,
        "source_git": "d3cf58242455bf96e46b0c2071e8e9fe16cf4aaf",
        "precision": "fp32",
        "encode_chunk_size": 1,
        "elapsed_seconds": time.monotonic() - started,
        "Kaggle_scratch_storage": storage,
        "GPU_or_optimizer_executed": False,
        "clinical_accuracy_evidence": False,
        "protected_gold_or_confirmation_used_for_training_or_selection": False,
        "competition_submission_created": False,
    }
    (args.output / "CPU-accepted.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    Path("/kaggle/working/witness-inputs-CPU-accepted.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--extract-inputs"]:
        parser = argparse.ArgumentParser()
        for name in ("input-directory", "destination"):
            parser.add_argument("--" + name, type=Path, required=True)
        for name in ("archive-sha256", "admission-sha256"):
            parser.add_argument("--" + name, required=True)
        options = parser.parse_args(sys.argv[2:])
        proof = extract_inputs(
            options.input_directory,
            options.destination,
            options.archive_sha256,
            options.admission_sha256,
        )
        print(
            json.dumps(
                {"stage": "exact_complete_source_archive_extracted", "files": len(proof["files"])}
            ),
            flush=True,
        )
    else:
        main()
