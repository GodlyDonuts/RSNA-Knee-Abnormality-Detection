#!/usr/bin/env python3
"""Smoke-test heterogeneous submission blending and schema checks."""

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
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sample = pd.DataFrame({"StudyInstanceUID": ["a", "b", "c"]})
        for target in TARGETS:
            sample[target] = 0.5
        sample.to_csv(root / "sample.csv", index=False)
        for name, values in {"left": [0.1, 0.4, 0.9], "right": [0.8, 0.3, 0.2]}.items():
            frame = sample.copy()
            for target in TARGETS:
                frame[target] = values
            frame.to_csv(root / f"{name}.csv", index=False)
        artifact = {
            "members": ["left", "right"],
            "weights": {target: [0.75, 0.25] for target in TARGETS},
        }
        (root / "blend.json").write_text(json.dumps(artifact))
        subprocess.run(
            [
                sys.executable,
                str(here / "blend_submission_files.py"),
                "--blend",
                str(root / "blend.json"),
                "--sample-submission",
                str(root / "sample.csv"),
                "--member",
                f"left={root / 'left.csv'}",
                "--member",
                f"right={root / 'right.csv'}",
                "--output",
                str(root / "output.csv"),
            ],
            check=True,
        )
        output = pd.read_csv(root / "output.csv")
        assert output.shape == (3, 13)
        assert np.isfinite(output[TARGETS].to_numpy()).all()
    print("heterogeneous submission blend smoke test passed")


if __name__ == "__main__":
    main()
