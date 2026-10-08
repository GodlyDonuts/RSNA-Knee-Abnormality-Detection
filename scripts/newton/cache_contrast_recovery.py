"""Prepared H100 recovery of the bound cache-contrast recipe; no GPU admission implied."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rsna.competitive.adapters import configure_encoder_training  # noqa: E402
from rsna.competitive.backbones import create_encoder  # noqa: E402
from rsna.competitive.model import CompetitiveStudyNet  # noqa: E402
from rsna.competitive.train import TrainConfig  # noqa: E402
from scripts.newton.probe_patch_witness_trainer_recovery import (  # noqa: E402
    compare_recovery,
    digest,
    numerical_updates,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def engineering_config(base, role, spec):
    if role not in {"control", "candidate"}:
        raise ValueError("Unknown input-contrast arm")
    for key in set(base["encoder"]) | set(spec):
        if key not in {"weights_path", "config_path"} and base["encoder"].get(key) != spec.get(key):
            raise ValueError("Mounted encoder policy differs from official CPU initialization")
    values = dict(
        base,
        encoder=spec,
        witness=None,
        plane_memory=None,
        patch_summary=None,
        fitting_supervision=None,
        cache_contrast=None if role == "control" else {"mode": "p01_p99"},
        epochs=2,
        diagnostic_study_limit=16,
        warmup_updates=2,
        submission_every_epochs=0,
        min_free_gib=3,
    )
    TrainConfig(**values).validate()
    if (
        values["precision"],
        values["amp_init_scale"],
        values["encode_chunk_size"],
        values["side"],
        values["batch_size"],
        values["accumulation_steps"],
    ) != ("fp16", 128, 8, 384, 1, 8):
        raise ValueError("Explicit admitted FP16 native384 effective8 recipe required")
    return values


def validate_cpu_receipt(cpu, proof):
    if (
        cpu.get("status")
        != "actual_cache_contrast_32_MRI_inputs_and_common_official_initialization_CPU_accepted"
        or cpu.get("training_engineering_studies") != 16
        or cpu.get("outer_engineering_studies") != 16
        or cpu.get("protected_labels_used_for_recipe_or_metrics") is not False
        or cpu["initial_learned_state"]["control"] != cpu["initial_learned_state"]["candidate"]
        or cpu["image_tensor_sha256"] != proof["contrast_image_tensor_sha256"]
        or cpu["other_tensor_sha256"] != proof["contrast_nonimage_tensor_sha256"]
        or cpu["other_tensor_sha256"]["control"] != cpu["other_tensor_sha256"]["candidate"]
    ):
        raise ValueError(
            "Actual32 original MRI input and common official initialization receipt required"
        )


def run(args):
    start = time.monotonic()
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("One assigned visible H100 required for sequential arms")
    if shutil.disk_usage(args.output.parent).free < 9 * 2**30:
        raise ValueError("Three GiB scratch reserve plus six GiB diagnostics required")
    args.output.mkdir(mode=0o700)
    cpu = json.loads(args.cpu_receipt.read_text())
    proof = json.loads((args.control / "original-input-proof.json").read_text())
    validate_cpu_receipt(cpu, proof)
    spec = json.loads(args.encoder_spec.read_text())
    base = json.loads(args.base_config.read_text())
    source = args.source.resolve()
    source_manifest = json.loads(args.source_manifest.read_text())
    for name, value in source_manifest.items():
        if sha(source / name) != value:
            raise ValueError("New immutable numerical source differs")
    artifact = args.control / "artifacts"
    results = {}
    for role in ("control", "candidate"):
        values = engineering_config(base, role, spec)
        torch.manual_seed(values["seed"])
        encoder, dim, ep = create_encoder(spec)
        configure_encoder_training(encoder, values["freeze_encoder"])
        model = CompetitiveStudyNet(
            encoder,
            dim,
            hidden_dim=values["hidden_dim"],
            encode_chunk_size=8,
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
        observed = {
            "learned_graph_values_sha256": digest(records),
            "post_factory_RNG_sha256": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            "learned_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        if observed != cpu["initial_learned_state"][role] or ep != cpu["encoder_provenance"]:
            raise ValueError("Actual official graph/RNG/provenance differs from CPU")
        del model, encoder, records
        gc.collect()
        root = args.output / role
        root.mkdir(mode=0o700)
        config = root / "config.json"
        config.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(source),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            WORLD_SIZE="1",
            RANK="0",
            LOCAL_RANK="0",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        command = [
            sys.executable,
            "-B",
            str(source / "scripts/newton/enforce_deterministic_runtime.py"),
            "--module",
            "rsna.competitive.train",
            "--config",
            str(config),
            "--device",
            "cuda",
            "--studies",
            str(artifact / "initiative/releases/report-supervision-v1/studies.parquet"),
            "--artifact-root",
            str(artifact),
            "--folds",
            str(artifact / "initiative/experiments/competitive-v1/folds"),
            "--series",
            str(
                artifact
                / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
            ),
            "--cache",
            str(args.cache),
        ]
        full, replay = root / "full", root / "replay"

        def execute(extra, phase, expected):
            if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 6 * 2**30:
                raise ValueError("Six GiB total recovery cap exceeded")
            remaining = args.remaining_seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError("Whole pipeline diagnostic deadline exhausted")
            print(
                json.dumps(
                    {"role": role, "phase": phase, "elapsed_seconds": time.monotonic() - start}
                ),
                flush=True,
            )
            with (
                (root / (phase + ".stdout")).open("x") as out,
                (root / (phase + ".stderr")).open("x") as err,
            ):
                result = subprocess.run(
                    command + extra, env=env, cwd=source, stdout=out, stderr=err, timeout=remaining
                )
            if result.returncode != expected:
                raise ValueError("Actual trainer failed; retain original phase logs")

        execute(["--output", str(full)], "full", 0)
        execute(["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
        partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        first = json.loads((replay / "invocation.json").read_text())
        execute(["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0)
        whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        history = json.loads((full / "invocation.json").read_text())
        last = json.loads((replay / "invocation.json").read_text())
        state = compare_recovery(
            whole,
            resumed,
            partial,
            history,
            first,
            last,
            (full / "outer_predictions.csv").read_bytes(),
            (replay / "outer_predictions.csv").read_bytes(),
        )
        binding = whole["binding"]
        if (
            binding["training_uids_sha256"] != proof["training_16_sha256"]
            or binding["outer_uids_sha256"] != proof["outer_16_sha256"]
            or binding["config"].get("cache_contrast") != values["cache_contrast"]
            or binding["config"].get("fitting_supervision") is not None
            or whole["amp_scaler"]["_growth_tracker"] != 4
            or binding["code_sha256"]
            != {
                n.removeprefix("rsna/"): h
                for n, h in source_manifest.items()
                if n.startswith("rsna/")
            }
        ):
            raise ValueError("Actual numerical source/recipe/roster/scaler differs")
        prediction = pd.read_csv(full / "outer_predictions.csv")
        from rsna.schema import TARGETS

        scores = prediction[list(TARGETS)].to_numpy()
        if (
            list(prediction.columns) != ["study_uid", "fold", "seed", *TARGETS]
            or len(prediction) != 16
            or digest(sorted(prediction.study_uid.tolist())) != proof["outer_16_sha256"]
            or not np.isfinite(scores).all()
            or (scores < 0).any()
            or (scores > 1).any()
        ):
            raise ValueError("Exact finite original16x12 predictions required")
        results[role] = {
            "complete_state_sha256": state,
            "checkpoint_sha256": sha(full / "checkpoint.pt"),
            "prediction_sha256": sha(full / "outer_predictions.csv"),
            "finite_updates": numerical_updates(history),
            "amp_scaler": whole["amp_scaler"],
            "exact_complete_state_scaler_RNG_and_CSV_recovery": True,
        }
        del whole, resumed, partial
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "status": "actual_single_H100_sequential_cache_contrast_recovery_accepted",
        "roles": results,
        "science_git": cpu["source_git"],
        "CPU_receipt_sha256": sha(args.cpu_receipt),
        "elapsed_seconds": time.monotonic() - start,
        "GPUs_used_concurrently": 1,
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "competition_submission": False,
    }
    (args.output / "recovery-accepted.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "control",
        "cache",
        "cpu-receipt",
        "base-config",
        "encoder-spec",
        "source",
        "source-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--remaining-seconds", type=float, required=True)
    run(parser.parse_args())
