"""Convert two actual browser-observed Colab JSON outputs into final evidence.

Input is {terminal, completion}, captured through the authorized browser UI.
This tool never browses, trains, uploads, downloads or calls an account API.
"""

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-output", type=Path, required=True)
    args = parser.parse_args()
    h = load("original_Colab_handoff", HERE / "handoff.py")
    c = load(
        "original_Colab_contract", HERE.parent / "colab-C100-final-serving-20261021-v1/contract.py"
    )
    observed = json.loads(h.read_small(args.observed_output))
    h.require(
        set(observed) == {"terminal", "completion"}, "Only original two JSON outputs required"
    )
    own = {}
    for name, digest in c.MODULE_SHA256.items():
        # Both package placeholders were deliberately empty in the actual
        # Colab numerical archive; their complete bytes are fixed by SHA(empty).
        raw = (
            b""
            if digest == h.sha(b"")
            else subprocess.check_output(["git", "show", c.SCIENCE + ":" + name], cwd=ROOT)
        )
        h.require(h.sha(raw) == digest, "Actual own numerical source differs")
        own[name] = raw.decode()
    bundle = h.from_saved_output(observed["terminal"], observed["completion"], own, c)
    destination = ROOT / "artifacts" / HERE.name / h.FILENAME
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    # Atomic visibility to the waiting owner; never replace or repeat delivery.
    temporary = destination.with_suffix(".writing")
    with temporary.open("xb") as f:
        f.write(h.canonical(bundle))
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    print(
        json.dumps(
            {
                "status": "actual_original_saved_Colab_output_bound",
                "evidence_sha256": bundle["evidence_sha256"],
                "new_model_transfers_or_account_calls": 0,
                "path": str(destination),
            }
        )
    )


if __name__ == "__main__":
    os.umask(0o077)
    main()
