"""Two-T4 source-neighbor/dense precision and timing diagnostic, never a submission.

The fixture contains only anonymous original fitting MRI windows. Diagnostic
four-update weights cannot become a competition model through this probe.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_dense_infer import POLICY
from rsna.competitive.compact_neighbor_infer import load_model
from scripts.kaggle.compact_t4_probe import compare

KIND = "compact_neighbor_dense30_diagnostic_component_v1"


def validate_fixture(value: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if set(value) != {"fixture_identity", "windows", "original_window_indices", "inference_policy"}:
        raise ValueError("Only anonymous image windows, indices and fixed policy allowed")
    if value["fixture_identity"] != "engineering-dense30-v1" or value["inference_policy"] != POLICY:
        raise ValueError("Exact original engineering fixture/policy required")
    windows, indices = value["windows"], value["original_window_indices"]
    if (
        not isinstance(windows, torch.Tensor)
        or windows.dtype != torch.float32
        or windows.ndim != 4
        or windows.shape[1:] != (3, 384, 384)
        or not 1 <= len(windows) <= 180
        or not torch.isfinite(windows).all()
        or windows.min() < -1
        or windows.max() > 1
        or not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int64
        or indices.ndim != 1
        or not 1 <= len(indices) <= 84
        or (indices < 0).any()
        or (indices >= len(windows)).any()
        or not torch.equal(indices, indices.unique(sorted=True))
    ):
        raise ValueError(
            "Finite bounded FP32 dense windows and exact ordered original subset required"
        )
    return windows, indices


def admit_component(root: Path, expected_sha: str):
    path = root / "component-manifest.json"
    if path.is_symlink() or file_sha256(path) != expected_sha:
        raise ValueError("Exact diagnostic component manifest required")
    value = json.loads(path.read_text())
    files = {
        "checkpoint.pt",
        "snapshot.json",
        "dense-fixture.pt",
        "reference.json",
        "provenance.json",
        "Apache-2.0-LICENSE",
        "initialization-model-card.md",
    }
    if (
        value.get("kind") != KIND
        or value.get("diagnostic_only") is not True
        or value.get("competition_eligible") is not False
        or value.get("inference_policy") != POLICY
        or set(value.get("files", {})) != files
    ):
        raise ValueError("Exact diagnostic-only dense component scope required")
    for name, item in value["files"].items():
        p = root / name
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(root.resolve())
            or p.stat().st_size != item["bytes"]
            or file_sha256(p) != item["sha256"]
        ):
            raise ValueError("Original diagnostic component bytes changed")
    reference = json.loads((root / "reference.json").read_text())
    snapshot = json.loads((root / "snapshot.json").read_text())
    if (
        reference.get("checkpoint_sha256") != value["files"]["checkpoint.pt"]["sha256"]
        or reference.get("fixture_sha256") != value["files"]["dense-fixture.pt"]["sha256"]
        or snapshot.get("checkpoint_sha256") != reference["checkpoint_sha256"]
        or snapshot.get("epoch") != 2
        or snapshot.get("optimizer_steps") != 4
        or snapshot.get("binding", {}).get("diagnostic_only") is not True
        or reference.get("model_training_source_git") != "3803b717b69db6c5e89e11724c4daee15490cbe2"
    ):
        raise ValueError("Original four-update diagnostic checkpoint/reference binding differs")
    return reference


def qualify(root: Path, manifest_sha: str, output: Path):
    if output.exists() or output.name == "submission.csv":
        raise ValueError("One diagnostic receipt only; never competition output")
    reference = admit_component(root, manifest_sha)
    fixture = torch.load(root / "dense-fixture.pt", map_location="cpu", weights_only=True)
    windows, original_indices = validate_fixture(fixture)
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in range(2)
    ):
        raise ValueError("Actual two assigned T4 devices required")
    torch.set_num_threads(2)
    devices = []
    for index in range(2):
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
        model, scope = load_model(
            root / "checkpoint.pt",
            reference["checkpoint_sha256"],
            precision="float32",
            device=device,
            allow_diagnostic=True,
        )
        if not scope["diagnostic_only"] or scope["competition_eligible"]:
            raise ValueError(
                "Only the original actual-recovery diagnostic weights qualify this probe"
            )
        dense = windows.to(device)
        original = dense[original_indices.to(device)]
        errors = {}
        torch.cuda.reset_peak_memory_stats(index)
        with torch.inference_mode():
            for precision, tolerance in (("float32", 0.0003), ("float16", 0.003)):
                model.precision = precision
                errors[precision] = {
                    role: compare(
                        model(w, chunk_size=12).sigmoid().cpu(), reference[role], tolerance
                    )
                    for role, w in (("original", original), ("dense", dense))
                }
            model.precision = "float16"
            timings = {}
            for count in (84, 180):
                # Repetition is a conservative shape benchmark, not extra clinical evidence.
                timing = dense.repeat(((count + len(dense) - 1) // len(dense), 1, 1, 1))[:count]
                durations = []
                for repeat in range(4):
                    torch.cuda.synchronize(index)
                    started = time.monotonic()
                    logits = model(timing, chunk_size=12)
                    torch.cuda.synchronize(index)
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite timing forward")
                    if repeat:
                        durations.append(time.monotonic() - started)
                timings[str(count)] = {
                    "seconds": durations,
                    "median_seconds": statistics.median(durations),
                }
        peak = torch.cuda.max_memory_allocated(index)
        if peak >= 15 * 2**30:
            raise ValueError("Two-T4 memory margin insufficient")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "precision_probability_max_errors_vs_CPU": errors,
                "FP16_encoder_bag_timings": timings,
                "peak_allocated_bytes": peak,
            }
        )
        del model, dense, original, timing, logits
        torch.cuda.empty_cache()
    result = {
        "status": "actual_neighbor_dense30_two_T4_precision_and_timing_accepted",
        "diagnostic_only": True,
        "competition_eligible": False,
        "clinical_accuracy_claim": False,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "fixture_sha256": reference["fixture_sha256"],
        "inference_policy": POLICY,
        "original_windows": len(original_indices),
        "dense_windows": len(windows),
        "devices": devices,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "hidden_scale_end_to_end_runtime_qualified": False,
        "competition_submission": False,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.component, args.manifest_sha256, args.output)
