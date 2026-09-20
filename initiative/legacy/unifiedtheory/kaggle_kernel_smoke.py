#!/usr/bin/env python3
"""Check the generated Kaggle patch kernel contract without publishing it."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "kernel"
        subprocess.run(
            [sys.executable, str(here / "build_kaggle_patch_kernel.py"), "--output", str(output)],
            check=True, capture_output=True, text=True,
        )
        metadata = json.loads((output / "kernel-metadata.json").read_text())
        notebook = json.loads((output / metadata["code_file"]).read_text())
        code = "".join(notebook["cells"][1]["source"])
        if metadata["enable_internet"] or not metadata["enable_gpu"]:
            raise AssertionError("kernel is not offline GPU inference")
        if metadata["is_private"]:
            raise AssertionError("competition code must be public")
        if len(metadata["dataset_sources"]) != 2:
            raise AssertionError("kernel must attach both code and checkpoint datasets")
        if "len(checkpoints) != 15" not in code or "submission.csv" not in code:
            raise AssertionError("kernel does not enforce checkpoint and output contracts")
    print("Kaggle kernel smoke test passed")


if __name__ == "__main__":
    main()
