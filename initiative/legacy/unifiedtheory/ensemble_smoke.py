#!/usr/bin/env python3
"""Check target-wise OOF blending without training a large image model."""

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
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        rng = np.random.default_rng(2026)
        count = 90
        labels = pd.DataFrame(
            {
                "StudyInstanceUID": [f"study-{index}" for index in range(count)],
                "fold": np.arange(count) % 3,
            }
        )
        member_a = labels[["StudyInstanceUID"]].copy()
        member_b = labels[["StudyInstanceUID"]].copy()
        for target_index, target in enumerate(TARGETS):
            y = ((np.arange(count) + target_index) % 3 == 0).astype(int)
            labels[target] = y
            # Alternate which model is informative so target-wise weights are
            # distinguishable from a single global blend.
            if target_index % 2 == 0:
                member_a[target] = y + rng.normal(0, 0.15, count)
                member_b[target] = rng.normal(0, 1, count)
            else:
                member_a[target] = rng.normal(0, 1, count)
                member_b[target] = y + rng.normal(0, 0.15, count)
        labels_path = root / "labels.csv"
        a_path = root / "a.csv"
        b_path = root / "b.csv"
        output = root / "blend.json"
        labels.to_csv(labels_path, index=False)
        member_a.to_csv(a_path, index=False)
        member_b.to_csv(b_path, index=False)
        subprocess.run(
            [
                sys.executable,
                str(here / "fit_oof_ensemble.py"),
                "--labels-csv",
                str(labels_path),
                "--member",
                f"a={a_path}",
                "--member",
                f"b={b_path}",
                "--output",
                str(output),
                "--samples",
                "1000",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        artifact = json.loads(output.read_text())
        if artifact["nested_macro_auc"] < 0.9:
            raise AssertionError("nested OOF blend did not recover the synthetic signal")
        for target_index, target in enumerate(TARGETS):
            weights = artifact["weights"][target]
            if not np.isclose(sum(weights), 1):
                raise AssertionError("blend weights are not normalized")
            expected = 0 if target_index % 2 == 0 else 1
            if int(np.argmax(weights)) != expected:
                raise AssertionError(f"wrong member selected for {target}")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "nested_macro_auc": artifact["nested_macro_auc"],
                    "targets": len(TARGETS),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
