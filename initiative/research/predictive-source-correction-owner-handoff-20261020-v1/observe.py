"""New allocation-free observer for the distinct corrected-source owner."""

import argparse
import json
import sys
from pathlib import Path

import launch

ORIGINAL = "initiative/research/predictive-training-observer-20261020-v1/observe.py"
ORIGINAL_SHA = "0463797c8e106302a2d42f677686fc26ae999aaf84ea059227c48d194f858944"


def observer():
    base, c = launch.base(), launch.c
    source = launch.git_file(c.BUILDER, ORIGINAL).decode()
    c.require(c.sha(source.encode()) == ORIGINAL_SHA, "Exact read-only observer required")
    new = launch.HERE.relative_to(launch.ROOT).as_posix()
    source = source.replace(c.BUILDER_DIR, new)
    source = c.replace_once(source, "artifacts/" + Path(c.BUILDER_DIR).name + "/finite-owner",
                            "artifacts/" + launch.HERE.name + "/finite-owner")
    source = c.replace_once(source, 'owner["original_owner_source_git"] == PIN',
                            'owner["original_owner_source_git"] == ' + repr(c.RESILIENT))
    source = c.replace_once(source, 'owner["original_owner_pid"] == 80995',
                            'owner["original_owner_pid"] == 96074')
    source = c.replace_once(source, '"accepted_v4_cache_owner_handoff"',
                            '"accepted_source_correction_owner_handoff"')
    source = c.replace_once(source, '    contract.handoff(handoff, "fitting")',
        "    contract.bind(git_file(" + repr(c.PRIOR) + ", " +
        repr(c.PRIOR_DIR + "/contract.py") + ").decode(), git_file(" +
        repr(c.BUILDER) + ", " + repr(c.BUILDER_DIR + "/contract.py") + ").decode())\n"
        '    contract.handoff(handoff, "fitting")')
    source = c.replace_once(source, 'DEADLINE = "2026-10-22T09:00:00+00:00"',
                            'DEADLINE = "2026-10-22T12:00:00+00:00"')
    return base.module(source, "source_corrected_readonly_observer", launch.HERE / "observe.py")


def main():
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--observer-handoff", type=Path, required=True)
    gate.add_argument("--source-git", required=True)
    args, remaining = gate.parse_known_args()
    for name in ("observe.py", "launch.py", "contract.py", "remote_adapter.py"):
        path = launch.HERE / name
        launch.c.require(launch.git_file(args.source_git, path.relative_to(launch.ROOT).as_posix()) ==
                         path.read_bytes(), "Exact committed observer dependencies required")
    base = launch.base()
    launch.c.observer_handoff(json.loads(args.observer_handoff.read_text()), launch.ROOT,
                              base.process_command)
    previous = sys.argv
    try:
        sys.argv = [str(launch.HERE / "observe.py"), "--source-git", args.source_git, *remaining]
        observer().main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
