#!/usr/bin/env python3
"""Launch all folds of a raw-DICOM MIL specialist."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args, extra = parser.parse_known_args()
    here = Path(__file__).resolve().parent
    for fold in range(args.folds):
        existing = sorted(args.output.glob(f"*_fold{fold}.pt"))
        if args.skip_existing and existing:
            print(f"skip fold {fold}: {existing}", flush=True)
            continue
        command = [
            sys.executable,
            str(here / "train_raw_mil.py"),
            "--output",
            str(args.output),
            "--fold",
            str(fold),
            "--folds",
            str(args.folds),
            *extra,
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
