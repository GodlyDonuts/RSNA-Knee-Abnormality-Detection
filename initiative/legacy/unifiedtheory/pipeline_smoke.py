#!/usr/bin/env python3
"""Exercise cache -> train -> checkpoint -> ensemble inference end to end."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

try:
    from .constants import CACHE_SCHEMA_VERSION, TARGETS
except ImportError:
    from constants import CACHE_SCHEMA_VERSION, TARGETS


def main() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cache_dir = root / "cache"
        run_dir = root / "run"
        cache_dir.mkdir()
        index_rows = []
        label_rows = []
        sample_rows = []
        generator = torch.Generator().manual_seed(2026)
        for study in range(18):
            uid = f"1.2.840.synthetic.{study}"
            n_series = 2 + study % 2
            features = torch.randn(n_series, 5, 8, generator=generator)
            patch_features = torch.randn(n_series, 5, 4, 4, generator=generator)
            # Install a simple image signal so the training path is not purely
            # random while retaining all 12 independent output heads.
            features[..., 0] += 1.5 * (study % 2)
            cache_path = cache_dir / f"{study}.pt"
            torch.save(
                {
                    "schema_version": torch.tensor(CACHE_SCHEMA_VERSION),
                    "features": features.half(),
                    "patch_features": patch_features.half(),
                    "patch_mask": torch.ones(n_series, 5, 4, dtype=torch.bool),
                    "positions_mm": torch.arange(5).float()[None, :].expand(n_series, -1),
                    "slice_mask": torch.ones(n_series, 5, dtype=torch.bool),
                    "series_mask": torch.ones(n_series, dtype=torch.bool),
                    "plane": torch.arange(1, n_series + 1).clamp_max(3),
                    "fluid": torch.full((n_series,), 1 + study % 2),
                    "fatsat": torch.full((n_series,), 1 + (study // 2) % 2),
                },
                cache_path,
            )
            index_rows.append(
                {
                    "StudyInstanceUID": uid,
                    "cache_file": str(cache_path),
                    "scanner_group": f"scanner_{study // 6}",
                }
            )
            label = {"StudyInstanceUID": uid}
            sample = {"StudyInstanceUID": uid}
            for target_index, target in enumerate(TARGETS):
                label[target] = (study + target_index) % 2
                label[f"{target}__conf"] = 1.0
                sample[target] = 0.5
            label_rows.append(label)
            sample_rows.append(sample)
        index_path = root / "cache_index.csv"
        labels_path = root / "labels.csv"
        sample_path = root / "sample_submission.csv"
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
        pd.DataFrame(label_rows).to_csv(labels_path, index=False)
        pd.DataFrame(sample_rows).to_csv(sample_path, index=False)

        train_command = [
            sys.executable,
            str(here / "train.py"),
            "--cache-index",
            str(index_path),
            "--labels-csv",
            str(labels_path),
            "--output",
            str(run_dir),
            "--aggregator",
            "physical_alibi",
            "--folds",
            "3",
            "--fold",
            "0",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--hidden-dim",
            "16",
            "--heads",
            "4",
            "--series-depth",
            "1",
            "--study-depth",
            "1",
            "--dropout",
            "0",
            "--batch-size",
            "6",
            "--workers",
            "0",
            "--rank-weight",
            "0",
            "--device",
            "cpu",
        ]
        subprocess.run(train_command, check=True, capture_output=True, text=True)
        checkpoint = run_dir / "physical_alibi_fold0.pt"
        if not checkpoint.exists():
            raise AssertionError("training did not produce a checkpoint")
        patch_train_command = [
            *train_command,
            "--model-type",
            "patch",
            "--series-dropout",
            "0",
            "--token-adapter-bottleneck",
            "4",
        ]
        subprocess.run(patch_train_command, check=True, capture_output=True, text=True)
        patch_checkpoint = run_dir / "patch_physical_alibi_fold0.pt"
        if not patch_checkpoint.exists():
            raise AssertionError("patch training did not produce a checkpoint")

        submission = root / "submission.csv"
        infer_command = [
            sys.executable,
            str(here / "infer.py"),
            "--cache-index",
            str(index_path),
            "--sample-submission",
            str(sample_path),
            "--checkpoints",
            str(checkpoint),
            str(patch_checkpoint),
            "--output",
            str(submission),
            "--batch-size",
            "6",
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
        subprocess.run(infer_command, check=True, capture_output=True, text=True)
        result = pd.read_csv(submission)
        if list(result.columns) != ["StudyInstanceUID", *TARGETS] or len(result) != 18:
            raise AssertionError("submission schema or row count is wrong")
        if result[TARGETS].isna().any().any():
            raise AssertionError("inference emitted missing predictions")
        blend_path = root / "blend.json"
        blend_path.write_text(
            json.dumps(
                {
                    "members": ["summary", "patch"],
                    "weights": {target: [0.5, 0.5] for target in TARGETS},
                }
            )
        )
        blended_submission = root / "blended_submission.csv"
        subprocess.run(
            [
                sys.executable,
                str(here / "ensemble_infer.py"),
                "--cache-index",
                str(index_path),
                "--sample-submission",
                str(sample_path),
                "--blend",
                str(blend_path),
                "--member",
                f"summary={checkpoint}",
                "--member",
                f"patch={patch_checkpoint}",
                "--output",
                str(blended_submission),
                "--batch-size",
                "6",
                "--workers",
                "0",
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        blended = pd.read_csv(blended_submission)
        if blended.shape != result.shape or blended[TARGETS].isna().any().any():
            raise AssertionError("OOF-fitted ensemble inference failed")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "studies": 18,
                    "targets": len(TARGETS),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "patch_checkpoint_bytes": patch_checkpoint.stat().st_size,
                    "submission_shape": list(result.shape),
                    "blended_submission_shape": list(blended.shape),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
