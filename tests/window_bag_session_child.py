"""One fresh-process trainer session for the session-boundary test (not collected)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_window_bag import _diagnostic_run  # noqa: E402

if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
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
