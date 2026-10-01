"""Run a pinned experiment with strict deterministic CUDA algorithms.

This is an explicit runtime-policy experiment, not a claim that earlier
checkpoint differences have a proven cause. Keep this wrapper's hash in the
launch manifest alongside the unchanged scientific source and configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def configure(torch_module) -> dict:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    if not torch_module.are_deterministic_algorithms_enabled() or (
        torch_module.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError("Strict deterministic algorithms were not enabled")
    return {
        "status": "strict_deterministic_runtime_enabled",
        "torch_version": str(torch_module.__version__),
        "deterministic_algorithms": True,
        "warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def main() -> None:
    # Forward --help to the target too: its actual imports must run during the
    # CPU acceptance check, rather than returning this wrapper's help early.
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module")
    target.add_argument("--script", type=Path)
    args, remaining = parser.parse_known_args()
    # An external script normally puts its own directory on sys.path. The
    # experiment's verified checkout must supply rsna and helper modules.
    checkout = Path.cwd().resolve()
    sys.path.insert(0, str(checkout))
    if args.module:
        expected = checkout.joinpath(*args.module.split(".")).with_suffix(".py")
        spec = importlib.util.find_spec(args.module)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
            raise ValueError("Target module must come from the verified working checkout")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Deterministic CUDA workspace must be set before importing torch")
    import torch

    print(json.dumps(configure(torch), sort_keys=True), flush=True)
    sys.argv = [args.module or str(args.script), *remaining]
    if args.module:
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    else:
        assert args.script is not None
        runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
