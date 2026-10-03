"""Admit actual MRI inputs, then verify each supervision arm's trainer recovery.

Engineering predictions are never used to choose a recipe or submitted. The
caller verifies original cache/assets and materializes pinned source/runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.checkpoint import capture_rng, file_sha256
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.delta_checkpoint import _tensor_record
from rsna.competitive.folds import load_folds
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.supervision import apply_fitting_supervision
from rsna.competitive.train import TrainConfig, _uid_hash, select_development
from rsna.release import load_verified_release
from rsna.schema import TARGETS


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def state_record(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite decoded checkpoint tensor")
        return {"tensor": _tensor_record(value)}
    if isinstance(value, dict):
        return {
            "dict": [
                [state_record(k), state_record(v)]
                for k, v in sorted(value.items(), key=lambda r: repr(r[0]))
            ]
        }
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [state_record(v) for v in value]}
    if value is None or type(value) in (str, int, bool):
        return {type(value).__name__: value}
    if type(value) is float and math.isfinite(value):
        return {"float": value}
    raise ValueError("Unsupported decoded checkpoint value")


def numerical_updates(record):
    keys = ("epoch", "update", "optimizer_steps", "loss", "gradient_norm", "observed_cells")
    rows = record["updates"]
    if not rows or any(
        not all(type(r[k]) in (int, float) and math.isfinite(r[k]) for k in keys) for r in rows
    ):
        raise ValueError("Finite original trainer updates required")
    return [{k: r[k] for k in keys} for r in rows]


def compare_recovery(full, resumed, partial, invocations, predictions):
    complete, first, last = invocations
    for c in (full, resumed):
        if (c.get("schema_version"), c.get("epoch"), c.get("next_update"), c.get("step")) != (
            2,
            2,
            0,
            4,
        ):
            raise ValueError("Exact final four-update boundary required")
        if (
            not c.get("optimizer_boundary")
            or not c.get("optimizer", {}).get("state")
            or len(c.get("rng_by_rank", [])) != 1
        ):
            raise ValueError("Complete optimizer and RNG required")
        if c.get("model_storage", {}).get("format") != "frozen_base_delta_v1" or not c.get("model"):
            raise ValueError("Exact frozen-base delta payload required")
    if (partial.get("epoch"), partial.get("next_update"), partial.get("step")) != (0, 1, 1):
        raise ValueError("Exact interrupted boundary required")
    if partial["binding"] != full["binding"] or state_record(full) != state_record(resumed):
        raise ValueError("Complete decoded model/base/optimizer/LR/counters/RNG differ")
    for r, status, code in (
        (complete, "completed", 0),
        (first, "checkpointed_for_resume", 75),
        (last, "completed", 0),
    ):
        if r.get("status") != status or r.get("exit_code") != code:
            raise ValueError("Actual invocation outcomes differ")
    rows = numerical_updates(complete)
    if [(r["epoch"], r["update"], r["optimizer_steps"]) for r in rows] != [
        (0, 1, 1),
        (0, 2, 2),
        (1, 1, 3),
        (1, 2, 4),
    ]:
        raise ValueError("Actual update boundaries differ")
    if (
        len(first["updates"]) != 1
        or numerical_updates(first) + numerical_updates(last) != rows
        or predictions[0] != predictions[1]
    ):
        raise ValueError("Update trajectory or exact MRI predictions differ")
    return digest(state_record(full))


def admit_inputs(control):
    configs = {
        name: TrainConfig(**json.loads((control / (name + "-pilot.json")).read_text()))
        for name in ("control", "candidate")
    }
    for name, c in configs.items():
        c.validate()
        if (
            c.epochs,
            c.diagnostic_study_limit,
            c.side,
            c.batch_size,
            c.accumulation_steps,
            c.warmup_updates,
            c.submission_every_epochs,
        ) != (2, 16, 384, 1, 8, 1, 0) or c.fitting_supervision["enabled"] != (name == "candidate"):
            raise ValueError("Fixed bounded matched engineering recipe required")
    a, b = (asdict(configs[n]) for n in ("control", "candidate"))
    b["fitting_supervision"] = a["fitting_supervision"]
    if a != b:
        raise ValueError("Only supervision-enabled flag may differ")
    root = Path("/home/sa305415/rsna-knee/code-stopfix-1f486765")
    studies = root / "initiative/releases/report-supervision-v1/studies.parquet"
    folds = root / "initiative/experiments/competitive-v1/folds"
    series_path = (
        root / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    )
    manifest, release = load_verified_release(studies, root)
    train, outer = select_development(manifest, load_folds(folds, manifest, release), 0)
    series = pd.read_csv(series_path)
    if file_sha256(series_path) != release["series_sha256"] or (len(train), len(outer)) != (
        2247,
        1156,
    ):
        raise ValueError("Original complete rosters and series required")
    arms, receipts = {}, {}
    for name, c in configs.items():
        arms[name], receipts[name] = apply_fitting_supervision(
            train,
            outer,
            control / "overlay.json",
            c.fitting_supervision,
            release_sha256=release["manifest_sha256"],
            folds_sha256=file_sha256(folds / "fold.csv"),
            fold=0,
        )
    changed = int(
        (arms["control"].head(16).Effusion__mask != arms["candidate"].head(16).Effusion__mask).sum()
    )
    if (
        changed != 1
        or receipts["candidate"]["applied_cells"] != 31
        or receipts["control"]["applied_cells"] != 0
    ):
        raise ValueError("Pilot must exercise an actual reviewed masked positive")
    cache = Path("/tmp/rsna-native384-cohort-866458-lp6rcfdw/restored/cache")
    kwargs = {
        k: getattr(configs["control"], k)
        for k in ("slices", "side", "max_series", "slice_sampling", "series_sampling")
    }
    datasets = {
        n: CompetitiveStudyDataset(f.head(16), series, cache, augment=False, **kwargs)
        for n, f in {**arms, "outer": outer}.items()
    }
    image_proofs, observed = [], {n: 0 for n in arms}
    view_keys = ("images", "features", "series_mask", "slice_mask", "positions")
    for i in range(16):
        rows = {n: d[i] for n, d in datasets.items()}
        proofs = {
            n: digest({k: _tensor_record(row[k]) for k in view_keys}) for n, row in rows.items()
        }
        if (
            proofs["control"] != proofs["candidate"]
            or rows["control"]["study_uid"] != rows["candidate"]["study_uid"]
        ):
            raise ValueError("Actual images or order differ across arms")
        for row in rows.values():
            if not torch.isfinite(row["images"]).all() or not row["slice_mask"].any():
                raise ValueError("Finite actual original MRI views required")
        for name in arms:
            observed[name] += int(rows[name]["label_mask"].sum())
        image_proofs.append(proofs)
    if observed["candidate"] - observed["control"] != 1:
        raise ValueError("Actual tensors did not expose the reviewed supervision change")
    return (
        {
            "status": "actual_native384_effusion_inputs_passed",
            "diagnostic_only": True,
            "clinical_accuracy_evidence": False,
            "full_training_studies": len(train),
            "full_outer_studies": len(outer),
            "engineering_training_studies": 16,
            "engineering_outer_studies": 16,
            "changed_engineering_cells": changed,
            "observed_engineering_cells": observed,
            "matched_input_tensor_sha256": digest(image_proofs),
            "training_uids_sha256": _uid_hash(train.head(16)),
            "outer_uids_sha256": _uid_hash(outer.head(16)),
            "supervision_receipts": receipts,
            "configs": {n: asdict(c) for n, c in configs.items()},
            "release_sha256": file_sha256(studies),
            "folds_sha256": file_sha256(folds / "fold.csv"),
            "source_manifest_sha256": file_sha256(control / "source.json"),
        },
        root,
        cache,
        series_path,
    )


def recover(control, output, admitted, root, cache, series):
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0)
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Actual assigned H100 BF16 required")
    cpu = json.loads((control / "inputs-accepted.json").read_text())
    if {
        k: v for k, v in cpu.items() if k not in ("job_id", "node", "original_assets_verified")
    } != admitted:
        raise ValueError("Actual GPU inputs differ from original CPU acceptance")
    source = json.loads((control / "source.json").read_text())
    expected_code = {
        n.removeprefix("rsna/"): r["sha256"]
        for n, r in source["files"].items()
        if n.startswith("rsna/") and n.endswith(".py")
    }
    initial = {}
    for arm in ("control", "candidate"):
        c = TrainConfig(**admitted["configs"][arm])
        random.seed(c.seed)
        np.random.seed(c.seed % 2**32)
        torch.manual_seed(c.seed)
        encoder, width, provenance = create_encoder(c.encoder)
        configure_encoder_training(encoder, c.freeze_encoder)
        model = CompetitiveStudyNet(
            encoder,
            width,
            pooling=c.pooling,
            hidden_dim=c.hidden_dim,
            encode_chunk_size=c.encode_chunk_size,
            slice_context=c.slice_context,
        ).to("cuda")
        initial[arm] = {
            "complete_model_sha256": digest(state_record(model.state_dict())),
            "all_RNG_sha256": digest(state_record(capture_rng(torch.device("cuda:0")))),
            "encoder_provenance": provenance,
        }
        del model, encoder
        torch.cuda.empty_cache()
    if initial["control"] != initial["candidate"]:
        raise ValueError("Actual official-weight initial model or RNG differs across arms")
    results = {}
    for arm in ("control", "candidate"):
        base = [
            sys.executable,
            "-B",
            str(Path.cwd() / "scripts/newton/enforce_deterministic_runtime.py"),
            "--module",
            "rsna.competitive.train",
            "--config",
            str(control / (arm + "-pilot.json")),
            "--studies",
            str(root / "initiative/releases/report-supervision-v1/studies.parquet"),
            "--series",
            str(series),
            "--cache",
            str(cache),
            "--folds",
            str(root / "initiative/experiments/competitive-v1/folds"),
            "--artifact-root",
            str(root),
            "--fitting-supervision",
            str(control / "overlay.json"),
            "--device",
            "cuda",
        ]
        full, replay = output / (arm + "-full"), output / (arm + "-replay")

        def execute(extra, name, code):
            with (
                (output / (name + ".stdout")).open("x") as out,
                (output / (name + ".stderr")).open("x") as err,
            ):
                r = subprocess.run(base + extra, stdout=out, stderr=err, timeout=1500)
            if r.returncode != code:
                raise ValueError("Actual trainer process returned unexpected exit")

        execute(["--output", str(full)], arm + "-full", 0)
        execute(["--output", str(replay), "--stop-after-updates", "1"], arm + "-stop", 75)
        partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        first = json.loads((replay / "invocation.json").read_text())
        execute(
            ["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], arm + "-resume", 0
        )
        completed = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        invocations = [
            json.loads((full / "invocation.json").read_text()),
            first,
            json.loads((replay / "invocation.json").read_text()),
        ]
        predictions = [(p / "outer_predictions.csv").read_bytes() for p in (full, replay)]
        state_sha = compare_recovery(completed, resumed, partial, invocations, predictions)
        binding = completed["binding"]
        if (
            binding["code_sha256"] != expected_code
            or binding["training_uids_sha256"] != admitted["training_uids_sha256"]
            or binding["outer_uids_sha256"] != admitted["outer_uids_sha256"]
            or binding.get("experimental_fitting_supervision")
            != admitted["supervision_receipts"][arm]
            or binding.get("device_type") != "cuda"
        ):
            raise ValueError("Actual trainer source/input/supervision bindings differ")
        frame = pd.read_csv(full / "outer_predictions.csv")
        values = frame[list(TARGETS)]
        if (
            values.shape != (16, 12)
            or frame.study_uid.duplicated().any()
            or not values.map(math.isfinite).all().all()
            or not ((values >= 0) & (values <= 1)).all().all()
        ):
            raise ValueError("Actual MRI-only engineering predictions invalid")
        results[arm] = {
            "decoded_checkpoint_sha256": state_sha,
            "checkpoint_sha256": file_sha256(full / "checkpoint.pt"),
            "predictions_sha256": file_sha256(full / "outer_predictions.csv"),
            "updates": numerical_updates(invocations[0]),
            "exact_complete_state_and_prediction_recovery": True,
        }
        del completed, resumed, partial
    return {
        "status": "actual_native384_effusion_trainer_recovery_passed",
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "input_acceptance_sha256": file_sha256(control / "inputs-accepted.json"),
        "results": results,
        "matched_initial_model_and_RNG": initial["control"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--phase", choices=("inputs", "recovery"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    admitted, root, cache, series = admit_inputs(args.control)
    if args.phase == "inputs":
        record = dict(
            admitted,
            job_id=os.environ["SLURM_JOB_ID"],
            node=os.uname().nodename,
            original_assets_verified=True,
        )
    else:
        args.output.mkdir(mode=0o700)
        record = recover(args.control, args.output, admitted, root, cache, series)
        record.update(job_id=os.environ["SLURM_JOB_ID"], node=os.uname().nodename)
    filename = "inputs-accepted.json" if args.phase == "inputs" else "recovery-accepted.json"
    with (args.control / filename).open("x") as out:
        json.dump(record, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            {k: record[k] for k in ("status", "diagnostic_only", "clinical_accuracy_evidence")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
