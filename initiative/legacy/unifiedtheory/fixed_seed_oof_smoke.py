#!/usr/bin/env python3
"""Smoke-test the fixed equal-rank seed OOF builder."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def main() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for fold in range(2):
            for index in range(8):
                uid = f"F{fold}S{index}"
                row = {"StudyInstanceUID": uid, "fold": fold}
                for target in TARGETS:
                    row[target] = index % 2
                    row[f"{target}__gold"] = 1
                rows.append(row)
            for seed in (1, 2, 3):
                run = root / f"seed{seed}"
                run.mkdir(exist_ok=True)
                frame = pd.DataFrame({"StudyInstanceUID": [f"F{fold}S{i}" for i in range(8)]})
                for target_index, target in enumerate(TARGETS):
                    frame[target] = [
                        (i % 2) + seed * 0.01 + target_index * 0.0001 for i in range(8)
                    ]
                frame.to_csv(run / f"patch_mean_fold{fold}_oof.csv", index=False)
                (run / f"patch_mean_fold{fold}.pt").write_bytes(f"{seed}-{fold}".encode())
        labels = root / "labels.csv"
        pd.DataFrame(rows).to_csv(labels, index=False)
        output = root / "ensemble"
        subprocess.run(
            [
                sys.executable,
                str(here / "build_fixed_seed_oof.py"),
                "--runs", str(root),
                "--prefix", "patch_mean",
                "--name", "consensus_patch",
                "--seeds", "1", "2", "3",
                "--folds", "2",
                "--labels-csv", str(labels),
                "--output", str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["macro_auc"]["consensus_patch"] == 1.0
        assert len(manifest["checkpoint_source_files"]) == 6
        assert len(list(output.glob("consensus_patch_fold*_oof.csv"))) == 2
    print("fixed seed OOF smoke passed")


if __name__ == "__main__":
    main()
