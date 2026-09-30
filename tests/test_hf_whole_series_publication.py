"""Private epoch preservation cannot confuse diagnostic, sampled or all-data runs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "whole_series_publisher", ROOT / "scripts/hf/publish_whole_series_epoch.py"
)
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


@pytest.mark.parametrize("variant", ["control", "context"])
@pytest.mark.parametrize("fault", [None, "diagnostic", "coverage", "pool", "base", "role", "code"])
def test_private_whole_series_epoch_requires_the_exact_matched_recipe(
    tmp_path, monkeypatch, variant, fault
):
    code_root = tmp_path / "code"
    source = code_root / "rsna" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic source fixture")
    monkeypatch.setattr(
        publisher.subprocess,
        "check_output",
        lambda *a, **kw: (publisher.TRAINING_GIT + "\n").encode(),
    )
    license_file = tmp_path / "LICENSE.md"
    license_file.write_bytes(b"synthetic license fixture")
    monkeypatch.setattr(publisher, "LICENSE_SHA", publisher.sha(license_file))
    config = json.loads(
        (
            ROOT
            / f"initiative/experiments/competitive-v1/whole-series-context-v1/full-coverage-{variant}-delta-config.json"
        ).read_text()
    )
    config["encoder"]["adapter"].update(
        method="rs_lora_v1",
        trainable_parameters=10223616,
        base_frozen=True,
        target_modules=[str(i) for i in range(120)],
    )
    binding = {
        "config": config,
        "world_size": 1,
        "code_sha256": {"model.py": publisher.sha(source)},
        "release": {"binding_sha256": "b" * 64},
    }
    if fault == "diagnostic":
        config["diagnostic_study_limit"] = 16
    elif fault == "coverage":
        config["slice_sampling"] = "uniform"
    elif fault == "pool":
        config["fit_scope"] = "all_development"
    elif fault == "base":
        config["encoder"]["weights_sha256"] = "0" * 64
    elif fault == "role":
        config["slice_context"] = "neighbor3" if variant == "control" else "none"
    elif fault == "code":
        binding["code_sha256"]["model.py"] = "0" * 64
    run = tmp_path / "run"
    folder = run / "epoch-checkpoints"
    folder.mkdir(parents=True)
    checkpoint = folder / "epoch-001.pt"
    checkpoint.write_bytes(b"synthetic payload fixture; not trained weights")
    (run / "run.json").write_text(
        json.dumps(
            {
                "binding": binding,
                "training_studies": 2247,
                "outer_studies": 1156,
                "encoder_trainable_parameters": 10223616,
            }
        )
    )
    snapshot = folder / "epoch-001.json"
    snapshot.write_text(
        json.dumps(
            {
                "status": "scheduled_epoch_snapshot",
                "epoch": 1,
                "optimizer_steps": 281,
                "checkpoint_file": checkpoint.name,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": publisher.sha(checkpoint),
                "fold": 0,
                "seed": 20260922,
                "diagnostic_only": False,
                "release_binding_sha256": "b" * 64,
                "binding_sha256": hashlib.sha256(
                    json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    )
    if fault is not None:
        with pytest.raises(ValueError):
            publisher.source_files(snapshot, run, license_file, code_root, variant)
    else:
        manifest, files = publisher.source_files(snapshot, run, license_file, code_root, variant)
        assert manifest["variant"] == variant and manifest["checkpoint_schema_version"] == 2
        assert manifest["requires_pinned_original_weights"] is True
        assert set(files) == {"checkpoint.pt", "snapshot.json", "DINOv3-LICENSE.md"}
