#!/usr/bin/env python3
"""Run complete raw-image CV only for families that passed the pilot gate.

The promotion audit and family registry are immutable inputs.  This driver is
deliberately fail-closed: it never trains an unregistered family, never accepts
fold/output overrides inside a family command, and verifies the complete
checkpoint/OOF artifact matrix before marking a family complete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


FORBIDDEN_FAMILY_ARGS = {"--fold", "--folds", "--output", "--skip-existing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_registry(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if int(registry.get("schema_version", -1)) != 1:
        raise ValueError("unsupported family-registry schema")
    families = registry.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError("registry must define at least one family")
    for name, family in families.items():
        if not isinstance(name, str) or not name or not isinstance(family, dict):
            raise ValueError("family names and definitions must be nonempty objects")
        output = family.get("output")
        arguments = family.get("train_args")
        checkpoint = family.get("checkpoint_pattern")
        oof = family.get("oof_pattern")
        if not isinstance(output, str) or not output:
            raise ValueError(f"family {name!r} has no output directory")
        if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
            raise ValueError(f"family {name!r} train_args must be a string list")
        if FORBIDDEN_FAMILY_ARGS.intersection(arguments):
            raise ValueError(f"family {name!r} overrides driver-owned arguments")
        if not isinstance(checkpoint, str) or "{fold}" not in checkpoint:
            raise ValueError(f"family {name!r} needs a checkpoint_pattern containing {{fold}}")
        if not isinstance(oof, str) or "{fold}" not in oof:
            raise ValueError(f"family {name!r} needs an oof_pattern containing {{fold}}")
    return families


def family_command(
    here: Path, family: dict[str, Any], folds: int
) -> list[str]:
    return [
        sys.executable,
        str(here / "run_raw_mil_cv.py"),
        "--folds",
        str(folds),
        "--output",
        str(family["output"]),
        "--skip-existing",
        *family["train_args"],
    ]


def verify_family(family: dict[str, Any], folds: int) -> dict[str, list[str]]:
    output = Path(family["output"])
    checkpoints = [output / family["checkpoint_pattern"].format(fold=fold) for fold in range(folds)]
    oof = [output / family["oof_pattern"].format(fold=fold) for fold in range(folds)]
    missing = [str(path) for path in [*checkpoints, *oof] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete promoted family; missing {missing}")
    return {
        "checkpoints": [str(path) for path in checkpoints],
        "oof": [str(path) for path in oof],
    }


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    audit = read_json(args.audit)
    registry = read_json(args.registry)
    families = validate_registry(registry)
    promoted = audit.get("promoted")
    if not isinstance(promoted, list) or not all(isinstance(name, str) for name in promoted):
        raise ValueError("audit has no valid promoted-family list")
    unknown = sorted(set(promoted).difference(families))
    if unknown:
        raise ValueError(f"audit promotes unregistered families: {unknown}")

    here = Path(__file__).resolve().parent
    status: dict[str, Any] = {
        "schema_version": 1,
        "audit": str(args.audit),
        "registry": str(args.registry),
        "folds": args.folds,
        "promoted": promoted,
        "state": "dry_run" if args.dry_run else "running",
        "families": {},
    }
    write_status(args.status, status)
    for name in promoted:
        family = families[name]
        command = family_command(here, family, args.folds)
        record: dict[str, Any] = {"command": command, "state": "dry_run" if args.dry_run else "running"}
        status["families"][name] = record
        write_status(args.status, status)
        if args.dry_run:
            continue
        started = time.monotonic()
        try:
            subprocess.run(command, check=True)
            record["artifacts"] = verify_family(family, args.folds)
            record["state"] = "complete"
            record["elapsed_seconds"] = time.monotonic() - started
        except Exception as error:
            record["state"] = "failed"
            record["error"] = repr(error)
            record["elapsed_seconds"] = time.monotonic() - started
            status["state"] = "failed"
            write_status(args.status, status)
            raise
        write_status(args.status, status)
    status["state"] = "dry_run" if args.dry_run else "complete"
    write_status(args.status, status)
    print(json.dumps(status, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
