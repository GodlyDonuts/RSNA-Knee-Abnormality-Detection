"""Per-target rank fusion for competition member predictions.

The public ensembles that score ~0.94 combine member predictions by
percentile-ranking each target column within every member, then averaging the
ranks. For a per-target ROC-AUC metric a rank transform preserves a single
member's AUC while decorrelated member errors average out.

Fused values are cohort-relative ranks, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.infer import write_submission
from rsna.schema import TARGETS

ID_COLUMN = "StudyInstanceUID"


def load_member(path: Path, template: pd.DataFrame) -> pd.DataFrame:
    member = pd.read_csv(path)
    if list(member.columns) != [ID_COLUMN, *TARGETS]:
        raise ValueError(f"Member columns differ from the competition template: {path}")
    if member[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate studies in member predictions: {path}")
    if set(member[ID_COLUMN]) != set(template[ID_COLUMN]):
        raise ValueError(f"Member studies differ from the template: {path}")
    values = member[list(TARGETS)]
    if (
        not np.isfinite(values.to_numpy(dtype=float)).all()
        or not values.ge(0).all().all()
        or not values.le(1).all().all()
    ):
        raise ValueError(f"Member predictions must be finite values in [0,1]: {path}")
    return member.set_index(ID_COLUMN).loc[template[ID_COLUMN]].reset_index()


def fuse_members(members: list[pd.DataFrame], template: pd.DataFrame) -> pd.DataFrame:
    if not members:
        raise ValueError("Rank fusion requires at least one member")
    fused = template[[ID_COLUMN]].copy()
    for target in TARGETS:
        ranks = [
            member.set_index(ID_COLUMN).loc[template[ID_COLUMN], target].rank(pct=True)
            for member in members
        ]
        fused[target] = pd.concat(ranks, axis=1).mean(axis=1).to_numpy()
    return fused


def run(args: argparse.Namespace) -> None:
    template = pd.read_csv(args.template)
    members = [load_member(path, template) for path in args.members]
    write_submission(fuse_members(members, template), template, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
