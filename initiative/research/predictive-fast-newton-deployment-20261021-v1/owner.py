"""Distinct finite faster-serving handoff; no training or automatic retries."""

import os
from pathlib import Path

from run_once import BASE, git, load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = HERE.parent / "predictive-C100-deployment-owner-20261021-v1"


def implementation():
    body = git(BASE, ORIGINAL / "owner.py")
    start = body.index("                require(\n                    subprocess.run(")
    end = body.index("                self.sleep()", start)
    body = body[:start] + body[end:]
    m = load(body, ORIGINAL / "owner.py", "original_final_owner")
    m.HERE = HERE
    m.DEPLOY = HERE
    m.JOURNAL = ROOT / "artifacts" / HERE.name / "owner"
    m.FINAL = (
        ROOT
        / "artifacts/predictive-C100-commit-reconciliation-20261021-v1/continuation/owner-result.json"
    )
    return m


if __name__ == "__main__":
    os.umask(0o077)
    implementation().main()
