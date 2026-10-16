"""Export only the actual fixed-final head, retaining licensed parent provenance."""

import argparse
import json
from pathlib import Path

from rsna.competitive.released_context_infer import export_component


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "terminal", "recovery", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(export_component(args.checkpoint, args.terminal, args.recovery, args.output)),
        flush=True,
    )


if __name__ == "__main__":
    main()
