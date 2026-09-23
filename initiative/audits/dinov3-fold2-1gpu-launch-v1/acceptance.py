"""Check the matched effective batch and a real fold-2 odd snapshot capture."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import torch

from rsna.competitive.sampling import EffectiveBatchSampler
from rsna.competitive.train import TrainConfig

ROOT = Path(__file__).resolve().parent


def main() -> None:
    reference = json.loads((ROOT / "fold1-reference.json").read_text())
    pilot = json.loads((ROOT / "pilot-config.json").read_text())
    production = json.loads((ROOT / "production-config.json").read_text())
    TrainConfig(**pilot).validate()
    TrainConfig(**production).validate()
    assert {key for key in pilot if pilot[key] != reference[key]} == {
        "fold",
        "accumulation_steps",
        "epochs",
        "diagnostic_study_limit",
        "submission_every_epochs",
    }
    assert {key for key in production if production[key] != reference[key]} == {
        "fold",
        "accumulation_steps",
    }
    assert production["batch_size"] * production["accumulation_steps"] == 8
    for size in (100, 2247, 2300):
        dual = sorted(
            (index, valid)
            for rank in (0, 1)
            for batch in EffectiveBatchSampler(
                size, 1, 4, seed=20260922, epoch=0, rank=rank, world_size=2
            )
            for index, valid, _ in batch
        )
        single = sorted(
            (index, valid)
            for batch in EffectiveBatchSampler(size, 1, 8, seed=20260922, epoch=0)
            for index, valid, _ in batch
        )
        assert single == dual

    path = ROOT / "archive_odd_epochs.py"
    spec = importlib.util.spec_from_file_location("fold2_odd_epoch_archive", path)
    assert spec is not None and spec.loader is not None
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    binding = {
        "config": {
            "fold": 2,
            "seed": 20260922,
            "batch_size": 1,
            "accumulation_steps": 8,
            "submission_every_epochs": 2,
            "diagnostic_study_limit": None,
        },
        "world_size": 1,
        "release": {"binding_sha256": "a" * 64},
    }
    event = {"epoch": 0, "update": 2, "optimizer_steps": 2}
    with tempfile.TemporaryDirectory(prefix="rsna-fold2-odd-") as directory:
        run = Path(directory)
        (run / "run.json").write_text(json.dumps({"binding": binding, "training_studies": 16}))
        torch.save(
            {
                "schema_version": 1,
                "optimizer_boundary": True,
                "epoch": 1,
                "next_update": 0,
                "step": 2,
                "scheduled_epoch": False,
                "binding": binding,
            },
            run / "checkpoint.pt",
        )
        record = watcher.capture(run, 1, "synthetic-watcher", event)
        assert record is not None and record["fold"] == 2
        snapshot = run / "epoch-checkpoints/epoch-001.pt"
        assert snapshot.stat().st_ino == (run / "checkpoint.pt").stat().st_ino
        assert watcher.capture(run, 1, "synthetic-watcher", event) == record

    receipt = {
        "status": "passed",
        "pilot_config_valid": True,
        "production_config_valid": True,
        "effective_batch_studies": 8,
        "sampler_study_counts_checked": [100, 2247, 2300],
        "fold2_world1_odd_epoch_capture": "passed",
        "submission_created": False,
    }
    output = ROOT / "acceptance.json"
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
