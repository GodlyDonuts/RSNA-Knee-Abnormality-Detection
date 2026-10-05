"""Finite persistent worker protocol; no network, training or precision change."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

if __package__:
    from .resident_infer import ResidentInference
else:
    from resident_infer import ResidentInference


def parse_command(command):
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("checkpoint", "series", "template", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encoder-config", type=Path)
    group.add_argument("--encoder-spec", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32",), required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-revision", default="kaggle-runtime-test")
    parser.add_argument("--scheduled-epoch", action="store_true")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--fitting-admission", type=Path)
    args = parser.parse_args(command[command.index("--checkpoint") :])
    args.dicom_root = None
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    options = parser.parse_args()
    runner = None
    for line in sys.stdin:
        try:
            if len(line.encode()) > 1024**2:
                raise ValueError("Oversized resident command")
            request = json.loads(line)
            if request == {"action": "close"}:
                return
            if set(request) != {"command", "request_id"}:
                raise ValueError("Invalid resident request")
            args = parse_command(request["command"])
            with contextlib.redirect_stdout(sys.stderr):
                if runner is None:
                    runner = ResidentInference(args, device_index=options.device_index)
                result = runner.run(args)
            response = {"request_id": request["request_id"], "status": "passed", "result": result}
        except Exception:
            response = {
                "request_id": locals().get("request", {}).get("request_id"),
                "status": "failed",
                "stderr": traceback.format_exc(),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
