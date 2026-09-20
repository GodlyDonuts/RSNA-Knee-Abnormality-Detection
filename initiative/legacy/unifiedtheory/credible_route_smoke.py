#!/usr/bin/env python3
"""End-to-end smoke for nested subset selection and offline inference planning."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def main() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "raw"
        run.mkdir()
        rows = []
        for fold in range(4):
            for offset in range(8):
                index = fold * 8 + offset
                row = {"StudyInstanceUID": f"S{index:03d}", "fold": fold}
                for target in TARGETS:
                    row[target] = offset % 2
                    row[f"{target}__gold"] = 1
                rows.append(row)
        labels = pd.DataFrame(rows)
        labels_path = root / "labels.csv"
        labels.to_csv(labels_path, index=False)

        for fold in range(4):
            held = labels[labels["fold"] == fold]
            anchor = held[["StudyInstanceUID"]].copy()
            candidate = held[["StudyInstanceUID"]].copy()
            for target_index, target in enumerate(TARGETS):
                y = held[target].to_numpy(float)
                nuisance = ((np.arange(len(held)) * 5 + target_index) % 7) / 20
                anchor[target] = 0.35 + nuisance
                candidate[target] = 0.05 + 0.9 * y + nuisance / 100
            anchor.to_csv(root / f"anchor_fold{fold}_oof.csv", index=False)
            candidate.to_csv(run / f"candidate_fold{fold}_oof.csv", index=False)

        audit = root / "audit.json"
        registry = root / "registry.json"
        blend = root / "blend.json"
        audit.write_text(json.dumps({"promoted": ["candidate"]}))
        registry.write_text(
            json.dumps(
                {
                    "families": {
                        "candidate": {
                            "output": str(run),
                            "oof_pattern": "candidate_fold{fold}_oof.csv",
                            "checkpoint_pattern": "candidate_fold{fold}.pt",
                            "inference_slices": 8,
                        }
                    }
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(here / "fit_nested_subset_ensemble.py"),
                "--labels-csv", str(labels_path),
                "--audit", str(audit),
                "--registry", str(registry),
                "--anchor", f"anchor={root / 'anchor_fold*_oof.csv'}",
                "--output", str(blend),
                "--minimum-nested-gain", "0.001",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(blend.read_text())
        assert result["raw_extension_promoted"]
        assert result["members"] == ["anchor", "candidate"]
        assert result["nested_gain"] > 0

        sample = labels[["StudyInstanceUID", *TARGETS]].copy()
        sample_path = root / "sample.csv"
        anchor_submission = root / "anchor_submission.csv"
        anchor_runtime = root / "anchor_runtime.json"
        sample.to_csv(sample_path, index=False)
        sample.to_csv(anchor_submission, index=False)
        anchor_runtime.write_text(
            json.dumps(
                {
                    "studies": len(sample),
                    "checkpoints": [
                        str(root / f"anchor_fold{fold}.pt") for fold in range(4)
                    ],
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(here / "run_selected_raw_inference.py"),
                "--blend", str(blend),
                "--registry", str(registry),
                "--data-root", str(root),
                "--sample-submission", str(sample_path),
                "--existing-member", f"anchor={anchor_submission}",
                "--existing-runtime", f"anchor={anchor_runtime}",
                "--require-existing-runtime",
                "--work-dir", str(root / "inference"),
                "--output", str(root / "submission.csv"),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads((root / "submission.manifest.json").read_text())
        assert manifest["selected"] == ["anchor", "candidate"]
        assert len(manifest["inference_commands"]) == 1
        assert manifest["uploaded"] is False and manifest["submitted"] is False
    print("credible route smoke passed")


if __name__ == "__main__":
    main()
