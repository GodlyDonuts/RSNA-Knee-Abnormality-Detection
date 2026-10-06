"""Reuse admitted MRI inputs; accept official zero/one memory initialization on CPU."""

import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.folds import load_folds
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.plane_memory import planes_from_features
from rsna.competitive.train import TrainConfig, _uid_hash, select_development
from rsna.release import load_verified_release
from rsna.semantic_supervision import digest


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    if os.environ.get("SLURM_JOB_GPUS") or os.uname().nodename != "evc31":
        raise ValueError("CPU-only original owning-node inputs required")
    torch.set_num_threads(1)
    original = Path("/tmp/rsna-patch-witness-H100-recovery-20261006-v2/cpu-accepted.json")
    proof = json.loads(original.read_text())
    full = json.loads(
        Path("/tmp/rsna-smooth-witness-full-fit-20261006-v2/CPU-accepted.json").read_text()
    )
    inp = full["trainer_inputs"]
    frame, binding = load_verified_release(Path(inp["studies"]), Path(inp["artifact-root"]))
    folds = load_folds(Path(inp["folds"]), frame, binding)
    train, outer = select_development(frame, folds, 0)
    if (len(train), len(outer)) != (2247, 1156) or set(train.study_uid) & set(outer.study_uid):
        raise ValueError("Original disjoint fitting/development membership required")
    if (
        _uid_hash(train) != full["training_uids_sha256"]
        or _uid_hash(outer) != full["outer_uids_sha256"]
    ):
        raise ValueError("Original full rosters changed")
    series = pd.read_csv(inp["series"])
    if file_sha(inp["series"]) != binding["series_sha256"]:
        raise ValueError("Original sequence selection changed")
    image_hashes, other_hashes, counts = {}, {}, {}
    for role, part in [("training", train.head(16)), ("outer_engineering", outer.head(16))]:
        if (
            _uid_hash(part)
            != proof["training_16_sha256" if role == "training" else "outer_16_sha256"]
        ):
            raise ValueError("Original engineering rows changed")
        data = CompetitiveStudyDataset(
            part, series, Path(inp["cache"]), slices=16, side=384, max_series=6
        )
        image, other = hashlib.sha256(), hashlib.sha256()
        genuine_slices, genuine_series = 0, 0
        for index in range(16):
            item = data[index]
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
            planes_from_features(item["features"], item["series_mask"])
            genuine_slices += int(item["slice_mask"].sum())
            genuine_series += int(item["series_mask"].sum())
        image_hashes[role], other_hashes[role] = image.hexdigest(), other.hexdigest()
        counts[role] = {"genuine_slices": genuine_slices, "genuine_series": genuine_series}
        del data, item
        gc.collect()
    if (
        image_hashes != proof["input_tensor_sha256"]
        or other_hashes != proof["nonimage_tensor_sha256"]
    ):
        raise ValueError("Original32 MRI tensors, labels, masks or metadata differ")
    base = json.loads(args.base_config.read_text())
    initial, provenance, configs = {}, {}, {}
    for role, weight in [("control", 0), ("candidate", 1)]:
        values = dict(
            base, witness=None, plane_memory={"weight": weight, "seed": 20261006, "hidden_dim": 128}
        )
        config = TrainConfig(**values)
        config.validate()
        random.seed(config.seed)
        np.random.seed(config.seed % 2**32)
        torch.manual_seed(config.seed)
        encoder, dimension, ep = create_encoder(config.encoder)
        configure_encoder_training(encoder, config.freeze_encoder)
        model = CompetitiveStudyNet(
            encoder,
            dimension,
            hidden_dim=config.hidden_dim,
            encode_chunk_size=config.encode_chunk_size,
            plane_memory=config.plane_memory,
        )
        records = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                array = parameter.detach().contiguous().numpy()
                records[name] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
        initial[role] = {
            "learned_graph_and_values_sha256": digest(records),
            "post_factory_torch_RNG_sha256": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            "learned_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        provenance[role] = digest(ep)
        configs[role] = values
        del model, encoder, records
        gc.collect()
    if (
        initial["control"] != initial["candidate"]
        or provenance["control"] != provenance["candidate"]
    ):
        raise ValueError("Actual official initial graphs/values/RNG differ")
    result = {
        "status": "actual_original32_MRI_and_official_memory_initialization_CPU_accepted",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "original_CPU_receipt_sha256": file_sha(original),
        "source_git": args.source_git,
        "full_training_studies": 2247,
        "full_development_studies": 1156,
        "training_uids_sha256": _uid_hash(train),
        "outer_uids_sha256": _uid_hash(outer),
        "engineering_input_tensors": image_hashes,
        "engineering_other_tensors": other_hashes,
        "engineering_counts": counts,
        "initial_learned_state": initial,
        "encoder_provenance_sha256": provenance,
        "trainer_inputs": inp,
        "config_sha256": {r: digest(c) for r, c in configs.items()},
        "GPU_encoder_optimizer_or_clinical_inference_executed": False,
        "clinical_accuracy_evidence": False,
    }
    for role, values in configs.items():
        with (args.output / (role + ".json")).open("x") as stream:
            json.dump(values, stream, indent=2, sort_keys=True)
            stream.write("\n")
    with (args.output / "CPU-accepted.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git", required=True)
    run(parser.parse_args())
