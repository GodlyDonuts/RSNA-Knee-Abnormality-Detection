#!/usr/bin/env python3
"""Smoke-test the fail-closed promoted-family CV driver."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audit = root / "audit.json"
        registry = root / "registry.json"
        status = root / "status.json"
        audit.write_text(json.dumps({"promoted": ["candidate"]}))
        registry.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "families": {
                        "candidate": {
                            "output": str(root / "run"),
                            "train_args": ["--data-root", "/data", "--backbone", "dinov2"],
                            "checkpoint_pattern": "dinov2_topk_336_fold{fold}.pt",
                            "oof_pattern": "dinov2_topk_336_fold{fold}_oof.csv",
                        }
                    },
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(here / "run_promoted_raw_cv.py"),
                "--audit",
                str(audit),
                "--registry",
                str(registry),
                "--status",
                str(status),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(status.read_text())
        command = result["families"]["candidate"]["command"]
        assert result["state"] == "dry_run"
        assert result["promoted"] == ["candidate"]
        assert "--skip-existing" in command
        assert command[-4:] == ["--data-root", "/data", "--backbone", "dinov2"]

        bad = json.loads(registry.read_text())
        bad["families"]["candidate"]["train_args"] += ["--fold", "4"]
        registry.write_text(json.dumps(bad))
        failed = subprocess.run(
            [
                sys.executable,
                str(here / "run_promoted_raw_cv.py"),
                "--audit",
                str(audit),
                "--registry",
                str(registry),
                "--status",
                str(status),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "overrides driver-owned arguments" in failed.stderr
    print("promoted raw CV smoke passed")


if __name__ == "__main__":
    main()
