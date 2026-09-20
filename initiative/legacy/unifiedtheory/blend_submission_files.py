#!/usr/bin/env python3
"""Apply a nested-OOF blend artifact to heterogeneous submission files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--member", action="append", required=True, help="NAME=CSV")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rank01(values: pd.Series) -> np.ndarray:
    return values.rank(method="average", pct=True).to_numpy(float)


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.blend.read_text())
    expected = list(artifact["members"])
    paths: dict[str, Path] = {}
    for specification in args.member:
        if "=" not in specification:
            raise ValueError("member must be NAME=CSV")
        name, value = specification.split("=", 1)
        if name in paths:
            raise ValueError(f"duplicate member {name}")
        paths[name] = Path(value)
    if set(paths) != set(expected):
        raise ValueError(f"blend expects {expected}, received {sorted(paths)}")

    sample = pd.read_csv(args.sample_submission, dtype={"StudyInstanceUID": str})
    required = {"StudyInstanceUID", *TARGETS}
    if missing := required.difference(sample.columns):
        raise ValueError(f"sample submission is missing {sorted(missing)}")
    frames = {}
    for name in expected:
        frame = pd.read_csv(paths[name], dtype={"StudyInstanceUID": str})
        if missing := required.difference(frame.columns):
            raise ValueError(f"member {name} is missing {sorted(missing)}")
        if frame["StudyInstanceUID"].tolist() != sample["StudyInstanceUID"].tolist():
            raise ValueError(f"member {name} does not match sample study order")
        frames[name] = frame

    output = sample.copy()
    for target in TARGETS:
        weights = np.asarray(artifact["weights"][target], dtype=float)
        if weights.shape != (len(expected),) or np.any(weights < 0):
            raise ValueError(f"invalid weights for {target}: {weights}")
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"weights for {target} do not sum to one")
        ranked = np.column_stack([rank01(frames[name][target]) for name in expected])
        output[target] = ranked @ weights
    if not np.isfinite(output[TARGETS].to_numpy(float)).all():
        raise ValueError("blend produced non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output} from {len(expected)} heterogeneous members")


if __name__ == "__main__":
    main()
