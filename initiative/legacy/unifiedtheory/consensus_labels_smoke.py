#!/usr/bin/env python3
"""Smoke-test fixed report-label consensus and gold overrides."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .consensus_labels import UID, build_consensus
    from .constants import TARGETS
except ImportError:
    from consensus_labels import UID, build_consensus
    from constants import TARGETS


def main() -> None:
    ids = ["a", "b", "c", "d"]
    train = pd.DataFrame({UID: ids, **{target: [1.0, 0.0, np.nan, np.nan] for target in TARGETS}})
    folds = pd.DataFrame({UID: ids, "fold": [0, 1, 0, 1], "scanner_group": ["x", "y", "x", "y"]})
    first = pd.DataFrame(0.2, index=ids, columns=TARGETS)
    second = pd.DataFrame(0.8, index=ids, columns=TARGETS)
    first.index.name = UID
    second.index.name = UID
    output, audit = build_consensus(train, folds, [first, second])
    if not np.allclose(output.loc[2:, TARGETS].to_numpy(float), 0.5):
        raise AssertionError("equal-source consensus is not the arithmetic mean")
    if not np.allclose(output.loc[:1, TARGETS].to_numpy(float), [[1] * 12, [0] * 12]):
        raise AssertionError("official labels did not override soft targets")
    if not np.allclose(output.loc[:1, [f"{target}__conf" for target in TARGETS]], 1.0):
        raise AssertionError("gold overrides did not receive confidence one")
    if output["fold"].tolist() != [0, 1, 0, 1]:
        raise AssertionError("fold metadata was not preserved")
    if audit["sources"] != 2 or audit["missing_consensus_cells"] != 0:
        raise AssertionError("consensus audit is inconsistent")
    print("consensus label smoke test passed")


if __name__ == "__main__":
    main()
