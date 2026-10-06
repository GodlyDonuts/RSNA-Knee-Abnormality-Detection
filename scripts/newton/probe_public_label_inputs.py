"""Actual owning-node MRI tensor admission for an isolated label-source contrast."""

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

from rsna.competitive import backbones
from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.folds import load_folds
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.supervision import apply_fitting_supervision
from rsna.competitive.train import TrainConfig, select_development
from rsna.release import load_verified_release
from rsna.semantic_supervision import digest


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def run(root):
    if os.uname().nodename != "evc31" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Original owning-node CPU-only admission required")
    torch.set_num_threads(1)
    previous = json.loads(
        Path("/tmp/rsna-cls-patch-summary-CPU-20261006-v2/CPU-accepted.json").read_text()
    )
    inp = previous["trainer_inputs"]
    frame, binding = load_verified_release(Path(inp["studies"]), Path(inp["artifact-root"]))
    folds = load_folds(Path(inp["folds"]), frame, binding)
    original, outer = select_development(frame, folds, 0)
    if (len(original), len(outer)) != (2247, 1156):
        raise ValueError("Original full fitting/evaluation population changed")
    configs = {
        role: json.loads((root / (role + ".json")).read_text()) for role in ["control", "public"]
    }
    rows = {}
    receipts = {}
    for role, values in configs.items():
        TrainConfig(**values).validate()
        rows[role], receipts[role] = apply_fitting_supervision(
            original,
            outer,
            root / "overlay.json",
            values["fitting_supervision"],
            release_sha256=binding["manifest_sha256"],
            folds_sha256=sha(Path(inp["folds"]) / "fold.csv"),
            fold=0,
        )
    pd.testing.assert_frame_equal(rows["control"], original)
    if not rows["public"].study_uid.equals(original.study_uid) or set(original.leakage_group) & set(
        outer.leakage_group
    ):
        raise ValueError("Fitting membership or leakage-component boundary changed")
    series = pd.read_csv(inp["series"])
    images = {}
    other = {}
    label_proofs = {}
    for name, part, soft in [
        ("control", rows["control"].head(16), True),
        ("public", rows["public"].head(16), True),
        ("outer_engineering", outer.head(16), False),
    ]:
        dataset = CompetitiveStudyDataset(
            part, series, Path(inp["cache"]), slices=16, side=384, max_series=6, soft_targets=soft
        )
        ih, nh, lh = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        for i in range(16):
            item = dataset[i]
            ih.update(item["images"].numpy().tobytes())
            for key in [
                "features",
                "positions",
                "slice_mask",
                "series_mask",
                "labels",
                "label_mask",
            ]:
                nh.update(item[key].numpy().tobytes())
            for key in ["labels", "label_mask"]:
                lh.update(item[key].numpy().tobytes())
            expected = part.iloc[i]
            labels = np.asarray(
                [
                    expected[t] if expected[t + "__mask"] else 0
                    for t in __import__("rsna.schema", fromlist=["TARGETS"]).TARGETS
                ],
                dtype=np.float32,
            )
            if not np.array_equal(labels, item["labels"].numpy()):
                raise ValueError(
                    "Actual fitting/evaluation label tensor differs from admitted source"
                )
        images[name], other[name], label_proofs[name] = (
            ih.hexdigest(),
            nh.hexdigest(),
            lh.hexdigest(),
        )
        del dataset, item
        gc.collect()
    if (
        images["control"] != images["public"]
        or images["control"] != previous["original_input_tensor_hashes"]["training"]
        or images["outer_engineering"]
        != previous["original_input_tensor_hashes"]["outer_engineering"]
        or other["control"] != previous["original_other_tensor_hashes"]["training"]
        or other["outer_engineering"]
        != previous["original_other_tensor_hashes"]["outer_engineering"]
    ):
        raise ValueError("Original MRI/metadata/control/evaluation tensor proofs differ")
    proof_path = Path(
        "/home/sa305415/rsna-knee/production-launch-native384-a32-c9-v1/cpu-preflight.json"
    )
    if sha(proof_path) != "fa9e8a52ce05e81c08c2c3ae3a5eec28999109be74af2d4c19b8ddef0e76e9b5":
        raise ValueError("Original whole-byte scientific asset proof changed")
    proof = json.loads(proof_path.read_text())
    check = backbones._checked_file

    def original_file(spec, key):
        if key == "weights":
            p = Path(spec["weights_path"])
            record = proof["scientific_assets"][str(p)]
            if signature(p) != record["signature"] or spec["weights_sha256"] != record["sha256"]:
                raise ValueError("Original whole-byte-verified official base stat binding changed")
            return p
        return check(spec, key)

    backbones._checked_file = original_file
    config = TrainConfig(**configs["control"])
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    encoder, dim, ep = backbones.create_encoder(config.encoder)
    configure_encoder_training(encoder, config.freeze_encoder)
    rng = torch.get_rng_state().clone()
    initial = {}
    for role in ["control", "public"]:
        torch.set_rng_state(rng)
        model = CompetitiveStudyNet(
            encoder, dim, hidden_dim=config.hidden_dim, encode_chunk_size=config.encode_chunk_size
        )
        records = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                array = p.detach().numpy()
                records[name] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
        initial[role] = {
            "learned_graph_values_sha256": digest(records),
            "post_factory_RNG_sha256": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            "learned_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        del model, records
        gc.collect()
    if initial["control"] != initial["public"]:
        raise ValueError("Official common learned graph/values/RNG differ")
    value = {
        "status": "actual_public_label_source_original_MRI_control_outer_and_official_initialization_CPU_accepted",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": json.loads((root / "source.json").read_text())["source_git"],
        "trainer_inputs": inp,
        "configs": configs,
        "initial_learned_state": initial,
        "encoder_provenance": ep,
        "full_fitting_studies": 2247,
        "full_outer_studies": 1156,
        "image_tensor_sha256": images,
        "other_tensor_sha256": other,
        "label_tensor_sha256": label_proofs,
        "supervision": receipts,
        "GPU_or_optimizer_or_clinical_inference_executed": False,
        "original_large_base_proof_and_stats_reused": True,
        "protected_labels_used_for_recipe_or_metrics": False,
        "new_production_checkpoints_funded": False,
    }
    with (root / "CPU-accepted.json").open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
