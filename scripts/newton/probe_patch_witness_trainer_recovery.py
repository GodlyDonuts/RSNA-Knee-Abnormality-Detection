"""Run the unchanged canonical trainer in distinct full/stop/resume processes.

Diagnostic MRI-only recovery, never a clinical result or competition candidate.
Signed patch evidence zero/one arms; no diagnostic checkpoint is deployable.
Only an independently admitted owning-node control may invoke this wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.delta_checkpoint import _tensor_record
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def decoded(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint tensor")
        return {"tensor": _tensor_record(value)}
    if isinstance(value, dict):
        return {
            "dict": [
                [decoded(k), decoded(v)]
                for k, v in sorted(value.items(), key=lambda row: repr(row[0]))
            ]
        }
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [decoded(v) for v in value]}
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return {"float": value}
    raise ValueError("Unsupported decoded checkpoint value")


def numerical_updates(record):
    rows = record["updates"]
    result = []
    for row in rows:
        selected = {
            k: v for k, v in row.items() if k not in {"elapsed_seconds", "peak_gpu_allocated_bytes"}
        }
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in selected.values()):
            raise ValueError("Nonfinite numerical optimizer update")
        result.append(selected)
    return result


def compare_recovery(full, resumed, partial, whole, first, last, csv, replayed):
    if any(
        (c.get("epoch"), c.get("next_update"), c.get("step")) != (2, 0, 4) for c in (full, resumed)
    ):
        raise ValueError("Exact completed two-epoch four-update state required")
    for checkpoint in (full, resumed):
        if (
            checkpoint.get("schema_version") != 2
            or checkpoint.get("optimizer_boundary") is not True
            or not checkpoint.get("model")
            or not checkpoint.get("optimizer", {}).get("state")
            or len(checkpoint.get("rng_by_rank", [])) != 1
            or checkpoint.get("model_storage", {}).get("format") != "frozen_base_delta_v1"
        ):
            raise ValueError("Complete learned/base/AdamW/RNG delta state required")
    if (partial.get("epoch"), partial.get("next_update"), partial.get("step")) != (0, 1, 1):
        raise ValueError("Exact interrupted optimizer boundary required")
    if (
        full["binding"] != partial["binding"]
        or decoded(full) != decoded(resumed)
        or csv != replayed
    ):
        raise ValueError("Complete state or MRI-only CSV recovery differs")
    updates = numerical_updates(whole)
    if [(r["epoch"], r["update"], r["optimizer_steps"]) for r in updates] != [
        (0, 1, 1),
        (0, 2, 2),
        (1, 1, 3),
        (1, 2, 4),
    ]:
        raise ValueError("Actual contiguous four-update history required")
    if (
        len(numerical_updates(first)) != 1
        or numerical_updates(first) + numerical_updates(last) != updates
    ):
        raise ValueError("Numerical full/stop/resume trajectory differs")
    return digest(decoded(full))


def validate_recipe(config, mode, expected):
    if mode not in ("control", "candidate") or config != expected:
        raise ValueError("Exact independently accepted arm configuration required")
    if (
        config["epochs"],
        config["diagnostic_study_limit"],
        config["side"],
        config["precision"],
        config["batch_size"],
        config["accumulation_steps"],
        config["warmup_updates"],
        config["fit_scope"],
        config["seed"],
        config["pooling"],
        config["slice_context"],
        config["slice_sampling"],
        config["series_sampling"],
        config["slices"],
        config["max_series"],
    ) != (
        2,
        16,
        384,
        "bf16",
        1,
        8,
        2,
        "outer_fold",
        20260922,
        "attention",
        "none",
        "uniform",
        "selected",
        16,
        6,
    ):
        raise ValueError("Exact native384 diagnostic recipe required")
    if (
        config["witness"]
        != {"patches": 4, "slices": 2, "seed": 20261003, "weight": 0 if mode == "control" else 1}
        or config.get("neighbor_channels") is not None
        or config.get("relation") is not None
        or config.get("fitting_supervision") is not None
        or config["encoder"]["architecture"] != "dinov3_l16"
        or config["encoder"]["adapter"]["rank"] != 32
    ):
        raise ValueError("Witness arms must isolate signed local evidence")


def validate_placement(proof):
    node = proof.get("owning_node", "evc31")
    if (
        node not in {"evc31", "evc48"}
        or proof["node_future_bytes"] != (8 if node == "evc31" else 16) * 1024**3
    ):
        raise ValueError("Exact separately funded owning node required")
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    root = args.control
    proof = json.loads((root / "cpu-accepted.json").read_text())
    manifest = json.loads((root / "source.json").read_text())
    expected_node = validate_placement(proof)
    if (
        os.uname().nodename != expected_node
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or not os.environ.get("SLURM_JOB_ID")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Owning-node allocated one-H100 two-CPU BF16 context required")
    if (
        proof["status"]
        != "actual_patch_witness_32_study_inputs_and_official_initialization_accepted"
        or proof["source_sha256"] != manifest["files"]
        or proof["checkpoint_future_bytes"] != 6 * 1024**3
        or proof["shared_future_bytes"] != 0
        or proof["shared_future_files"] != 0
        or proof["initial_learned_state"]["control"] != proof["initial_learned_state"]["candidate"]
    ):
        raise ValueError("Exact independently funded CPU input admission required")
    for name, sha in manifest["files"].items():
        if file_sha256(root / "code" / name) != sha:
            raise ValueError("Accepted source changed")
    if shutil.disk_usage(root).free < 20 * 1024**3 + proof["node_future_bytes"]:
        raise ValueError("Twenty GiB reserve plus eight GiB total node obligation required")
    result_root = root / "recovery"
    result_root.mkdir(mode=0o700)
    results = {}
    for mode in ("control", "candidate"):
        if file_sha256(root / (mode + ".json")) != proof["config_sha256"][mode]:
            raise ValueError("CPU accepted numerical configuration changed")
        config = json.loads((root / (mode + ".json")).read_text())
        TrainConfig(**config).validate()
        validate_recipe(config, mode, proof["expected_config"][mode])
        full = result_root / (mode + "-full")
        replay = result_root / (mode + "-replay")
        base = [
            sys.executable,
            "-B",
            str(root / "code/scripts/newton/enforce_deterministic_runtime.py"),
            "--module",
            "rsna.competitive.train",
            "--config",
            str(root / (mode + ".json")),
            "--device",
            "cuda",
        ]
        for key, value in proof["trainer_inputs"].items():
            base += ["--" + key, value]

        def execute(command, phase, expected):
            if sum(p.stat().st_size for p in result_root.rglob("*") if p.is_file()) > 6 * 1024**3:
                raise ValueError("Diagnostic checkpoint cap exceeded")
            with (
                (result_root / (mode + "-" + phase + ".stdout")).open("x") as out,
                (result_root / (mode + "-" + phase + ".stderr")).open("x") as err,
            ):
                code = subprocess.run(command, stdout=out, stderr=err, timeout=1200).returncode
            if code != expected:
                raise ValueError("Actual trainer process failed; preserve original logs")

        execute(base + ["--output", str(full)], "full", 0)
        execute(base + ["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
        partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        first = json.loads((replay / "invocation.json").read_text())
        execute(
            base + ["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0
        )
        whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        invocation = json.loads((full / "invocation.json").read_text())
        last = json.loads((replay / "invocation.json").read_text())
        csv = (full / "outer_predictions.csv").read_bytes()
        state = compare_recovery(
            whole,
            resumed,
            partial,
            invocation,
            first,
            last,
            csv,
            (replay / "outer_predictions.csv").read_bytes(),
        )
        bound = whole["binding"]
        expected_code = {
            n.removeprefix("rsna/"): h
            for n, h in manifest["files"].items()
            if n.startswith("rsna/")
        }
        if (
            bound["code_sha256"] != expected_code
            or bound["training_uids_sha256"] != proof["training_16_sha256"]
            or bound["outer_uids_sha256"] != proof["outer_16_sha256"]
            or bound["config"]["witness"] != config["witness"]
            or "neighbor_channels" in bound["config"]
        ):
            raise ValueError("Actual source/input/numerical recipe binding differs")
        frame = pd.read_csv(full / "outer_predictions.csv")
        values = frame[list(TARGETS)].to_numpy()
        if (
            values.shape != (16, 12)
            or frame.study_uid.duplicated().any()
            or digest(sorted(frame.study_uid.tolist())) != proof["outer_16_sha256"]
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
        ):
            raise ValueError("Finite bounded sixteen-study MRI-only output required")
        results[mode] = {
            "complete_decoded_state_sha256": state,
            "checkpoint_sha256": file_sha256(full / "checkpoint.pt"),
            "predictions_sha256": hashlib.sha256(csv).hexdigest(),
            "optimizer_updates": 4,
            "updates": numerical_updates(invocation),
            "exact_complete_model_base_AdamW_LR_cursors_all_RNG_and_CSV_recovery": True,
        }
        del whole, resumed, partial
    record = {
        "status": "actual_native384_patch_witness_two_arm_trainer_recovery_accepted",
        "job_id": os.environ["SLURM_JOB_ID"],
        "source_git": manifest["source_git"],
        "source_sha256": manifest["files"],
        "CPU_acceptance_sha256": file_sha256(root / "cpu-accepted.json"),
        "results": results,
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "competition_candidate": False,
    }
    with (root / "recovery-verified.json").open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": record["status"],
                "job_id": record["job_id"],
                "receipt_sha256": file_sha256(root / "recovery-verified.json"),
                "clinical_accuracy_evidence": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
