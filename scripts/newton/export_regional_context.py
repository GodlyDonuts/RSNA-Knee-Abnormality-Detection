"""Export only an accepted full3887 fixed-final matched spatial role."""

import argparse
import json
from pathlib import Path

from rsna.competitive.regional_context_infer import export_component
from rsna.competitive.regional_context_train import ROLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "terminal", "recovery", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_component(args.checkpoint, args.terminal, args.recovery, args.output, args.role)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
