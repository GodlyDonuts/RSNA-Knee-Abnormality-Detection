"""Exclusive HTTP-successor observer; exact prior read-only loop and telemetry."""
import argparse
import json
import sys
from pathlib import Path

import launch

PRIOR_OBSERVER = "initiative/research/predictive-training-observer-20261020-v1/observe.py"
PRIOR_OBSERVER_SHA = "0463797c8e106302a2d42f677686fc26ae999aaf84ea059227c48d194f858944"


def observer():
    base = launch.base()
    c = launch.c
    source = launch.git_file(c.PRIOR, PRIOR_OBSERVER).decode()
    c.require(c.sha(source.encode()) == PRIOR_OBSERVER_SHA, "Original observer changed")
    new = launch.HERE.relative_to(launch.ROOT).as_posix()
    source = source.replace(c.PRIOR_DIR, new)
    source = c.replace_once(source, "artifacts/" + Path(c.PRIOR_DIR).name + "/finite-owner",
                            "artifacts/" + launch.HERE.name + "/finite-owner")
    source = c.replace_once(source, 'owner["original_owner_source_git"] == PIN',
                            'owner["original_owner_source_git"] == ' + repr(c.PRIOR))
    source = c.replace_once(source, 'owner["original_owner_pid"] == 80995',
                            'owner["original_owner_pid"] == 87673')
    source = c.replace_once(source, '"accepted_v4_cache_owner_handoff"',
                            '"accepted_http_cache_owner_handoff"')
    source = c.replace_once(source, '    contract.handoff(handoff, "fitting")',
        '    prior_contract = git_file(' + repr(c.PRIOR) + ', ' + repr(c.PRIOR_DIR + '/contract.py') + ').decode()\n'
        '    contract.bind(prior_contract)\n'
        '    contract.handoff(handoff, "fitting")')
    return base.module(source, "http_read_only_observer", launch.HERE / "observe.py")


def main():
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--observer-handoff", type=Path, required=True)
    gate.add_argument("--source-git", required=True)
    args, remaining = gate.parse_known_args()
    for name in ("observe.py", "launch.py", "contract.py", "remote_adapter.py"):
        path = launch.HERE / name
        launch.c.require(launch.git_file(args.source_git, path.relative_to(launch.ROOT).as_posix())
                         == path.read_bytes(), "Exact committed observer dependencies required")
    base = launch.base()
    launch.c.observer_handoff(json.loads(args.observer_handoff.read_text()),
                              launch.ROOT, base.process_command)
    prior = sys.argv
    try:
        sys.argv = [str(launch.HERE / "observe.py"), "--source-git", args.source_git, *remaining]
        observer().main()
    finally:
        sys.argv = prior


if __name__ == "__main__":
    main()
