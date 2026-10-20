"""Distinct finite HTTP-cache owner; reuse the exact prior operational capsule."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import contract as c

ROOT, HERE = Path(__file__).resolve().parents[3], Path(__file__).resolve().parent
_base, _builder, _prior = None, None, None


def git_file(pin, relative):
    c.require(len(pin) == 40 and all(ch in "0123456789abcdef" for ch in pin), "Exact source pin")
    return subprocess.check_output(["git", "show", pin + ":" + relative], cwd=ROOT)


def base():
    """Local source reads only; importing this module performs no operation."""
    global _base, _builder, _prior
    if _base is not None:
        return _base
    _prior = {name: git_file(c.PRIOR, c.PRIOR_DIR + "/" + name).decode() for name in c.PRIOR_SHA}
    c.require({name: c.sha(text.encode()) for name, text in _prior.items()} == c.PRIOR_SHA,
              "Frozen88 operational source changed")
    c.bind(_prior["contract.py"])
    source = c.replace_once(_prior["launch.py"], '"accepted_v4_cache_owner_handoff"',
                            '"accepted_http_cache_owner_handoff"')
    source = c.replace_once(source, '"original_owner_source_git": c.PARENT, "original_owner_pid": 80995,',
        '"original_owner_source_git": c.PRIOR, "original_owner_pid": 87673,\n'
        '        "prior_operational_sources_sha256": c.PRIOR_SHA,\n'
        '        "observer_handoff_sha256": p.file_sha(args.observer_handoff),')
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("frozen_http_owner_base", loader=None))
    mod.__file__ = str(HERE / "launch.py")
    previous = sys.modules.get("contract")
    sys.modules["contract"] = c
    try:
        exec(compile(source, "frozen88_http_operational_adapter", "exec"), mod.__dict__)
    finally:
        if previous is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = previous
    _base, _builder = mod, mod.build_program
    mod.build_program = build_program
    return mod


def build_program(own, originals, *, readonly=False):
    """Pure capsule construction: original builder plus exact prior-contract bind."""
    base()
    bound = dict(own, **{"contract.py": own["contract.py"] +
                          "\nbind(" + repr(_prior["contract.py"]) + ")\n"})
    return _builder(bound, originals, readonly=readonly)


def freeze(args):
    old = base()
    observer = json.loads(args.observer_handoff.read_text())
    c.observer_handoff(observer, ROOT, old.process_command)
    frozen = old.freeze(args)
    # Original fitting reader has already applied chain(). Bind all new HTTP proofs
    # before admitting the combined final cache, not merely its status string.
    original = frozen[4]
    reader = original.cache_receipts

    def receipts():
        result = reader()
        if result is not None:
            c.proofs(*(result[role]["original_actual"] for role in ("raw", "context")))
        return result

    original.cache_receipts = receipts
    frozen[-1]["prior_operational_sources_sha256"] = c.PRIOR_SHA
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fitting", "deployment"))
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--deployment-decision", type=Path)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--deployment-handoff", type=Path, required=True)
    parser.add_argument("--observer-handoff", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--deadline-utc", default="2026-10-22T09:00:00+00:00")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    os.umask(0o077)
    c.require(args.mode != "deployment" or args.deployment_decision is not None,
              "Explicit deployment decision required")
    frozen = freeze(args)
    (base().fitting if args.mode == "fitting" else base().deployment)(args, frozen)


if __name__ == "__main__":
    main()
