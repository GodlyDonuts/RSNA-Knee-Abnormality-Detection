"""Admit and precision-check scheduled N3 weights separately from shared runtime.

The existing diagnostic Dataset supplies only locked dependencies and its
unlabelled fixture. Its diagnostic checkpoint is never used by this entrypoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_infer import load_model
from scripts.kaggle.compact_t4_probe import compare, validate_fixture

SOURCE = "dcfff319cb7bd7bbae679a20da2b1b9268e8d680"
KIND = "compact_N3_scheduled_epoch_component_v1"


def admit_component(root: Path, expected_manifest_sha256: str) -> dict:
    """Verify all component bytes before deserializing a checkpoint."""
    manifest = root / "component-manifest.json"
    if manifest.is_symlink() or file_sha256(manifest) != expected_manifest_sha256:
        raise ValueError("Exact production component manifest required")
    value = json.loads(manifest.read_text())
    epoch = value.get("epoch")
    if (
        value.get("kind") != KIND
        or value.get("diagnostic_only") is not False
        or value.get("competition_eligible") is not True
        or value.get("source_training_git") != SOURCE
        or type(epoch) is not int
        or epoch not in (4, 8, 12, 16)
        or value.get("optimizer_steps") != epoch * 281
        or set(value.get("files", {}))
        != {"checkpoint.pt", "epoch.json", "reference.json", "provenance.json"}
    ):
        raise ValueError("Scheduled production checkpoint scope differs")
    for name, record in value["files"].items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or path.stat().st_size != record["bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError("Production component bytes differ")
    checkpoint_sha = value["files"]["checkpoint.pt"]["sha256"]
    sidecar = json.loads((root / "epoch.json").read_text())
    reference = json.loads((root / "reference.json").read_text())
    if (
        sidecar.get("checkpoint_sha256") != checkpoint_sha
        or sidecar.get("checkpoint_bytes") != value["files"]["checkpoint.pt"]["bytes"]
        or sidecar.get("epoch") != epoch
        or sidecar.get("optimizer_steps") != epoch * 281
        or sidecar.get("binding", {}).get("diagnostic_only") is not False
        or reference.get("checkpoint_sha256") != checkpoint_sha
        or reference.get("epoch") != epoch
    ):
        raise ValueError("Production sidecar/reference binding differs")
    return value


def qualify(root: Path, manifest_sha: str, fixture_root: Path, output: Path) -> None:
    if output.exists() or output.name == "submission.csv":
        raise ValueError("Distinct precision receipt required")
    manifest = admit_component(root, manifest_sha)
    reference = json.loads((root / "reference.json").read_text())
    fixture_path = fixture_root / "fixture.pt"
    if file_sha256(fixture_path) != reference["fixture_sha256"]:
        raise ValueError("Original unlabelled fixture differs")
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    count = validate_fixture(fixture)
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in range(2)
    ):
        raise ValueError("Actual two T4 devices required")
    torch.set_num_threads(2)
    devices = []
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        model, policy = load_model(
            root / "checkpoint.pt",
            reference["checkpoint_sha256"],
            device=device,
            allow_diagnostic=False,
        )
        if policy["epoch"] != manifest["epoch"] or not policy["competition_eligible"]:
            raise ValueError("Actual loaded model differs from scheduled candidate")
        windows = fixture["windows"].to(device)
        errors = {}
        with torch.inference_mode():
            for precision, tolerance in (("float32", 0.0003), ("float16", 0.003)):
                model.precision = precision
                value = model(windows, chunk_size=12).sigmoid().cpu()
                errors[precision] = compare(value, reference["full"], tolerance)
        devices.append({"index": index, "maximum_probability_errors": errors})
        del model, windows, value
        torch.cuda.empty_cache()
    output.write_text(
        json.dumps(
            {
                "status": "scheduled_compact_two_T4_precision_accepted",
                "epoch": manifest["epoch"],
                "checkpoint_sha256": reference["checkpoint_sha256"],
                "fixture_windows": count,
                "devices": devices,
                "clinical_accuracy_claim": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--fixture-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.component, args.manifest_sha256, args.fixture_bundle, args.output)
