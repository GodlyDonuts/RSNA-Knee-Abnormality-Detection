"""Export only the actual fixed-final confidence head with its real control proof."""

import argparse
import json
from pathlib import Path

from rsna.competitive.confidence_context_infer import export_component


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "terminal", "recovery", "parity", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(export_component(a.checkpoint, a.terminal, a.recovery, a.parity, a.output)),
        flush=True,
    )


if __name__ == "__main__":
    main()
