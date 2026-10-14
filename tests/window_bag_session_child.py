"""One fresh-process trainer session for the session-boundary test (not collected).

``argv[1]`` is a JSON spec. Optional trailing trainer flags (``--output``, ``--resume``,
``--stop-after-updates``), as appended by the segment runners, override it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_window_bag import _diagnostic_run  # noqa: E402

if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-updates", type=int)
    flags = parser.parse_args(sys.argv[2:])
    if flags.output is not None:
        spec |= {
            "output": flags.output,
            "resume": flags.resume,
            "stop": flags.stop_after_updates,
        }
    resume = Path(spec["resume"]) if spec.get("resume") else None
    code = _diagnostic_run(
        Path(spec["root"]),
        spec["arm"],
        Path(spec["output"]),
        stop=spec.get("stop"),
        resume=resume,
        every=spec.get("every", 0),
        precision=spec.get("precision", "fp16"),
    )
    sys.exit(code)
