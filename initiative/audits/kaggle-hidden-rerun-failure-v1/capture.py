"""Capture only safe fields from the five failed Kaggle submissions, read-only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "scripts/kaggle/workflow.py"
OUTPUT = Path(__file__).with_name("submissions.json")
SUBMISSION_IDS = (56482890, 56483451, 56483731, 56483949, 56485356)


def main() -> None:
    spec = importlib.util.spec_from_file_location("kaggle_workflow", WORKFLOW)
    if spec is None or spec.loader is None:
        raise ValueError("Kaggle workflow source unavailable")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    remote = workflow.Remote(ROOT / ".env")
    account = remote.account()
    rows = []
    for reference in SUBMISSION_IDS:
        found = remote.competition("GetSubmission", {"ref": reference})
        if found.get("ref") != reference:
            raise ValueError("Kaggle submission identity differs")
        rows.append(
            {
                key: found.get(key)
                for key in (
                    "ref",
                    "date",
                    "description",
                    "status",
                    "errorDescription",
                    "totalBytes",
                    "publicScore",
                )
            }
        )
    result = {
        "schema_version": 1,
        "status": "kaggle_hidden_rerun_failures_observed",
        "source": "authenticated GetSubmission for exact IDs; GetSubmissionLimits",
        "checked_at_utc": account["checked_at"],
        "competition": account["competition"],
        "num_today": account["num_today"],
        "num_allowed_now": account["num_allowed_now"],
        "submissions": rows,
    }
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "submissions": len(rows)}))


if __name__ == "__main__":
    main()
