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
    body = body.replace(
        "other_node_unpaid_bytes=512 * 2**30,", "other_node_unpaid_bytes=518 * 2**30,"
    )
    m = load(body, ORIGINAL / "owner.py", "original_final_owner")
    m.HERE = HERE
    m.DEPLOY = HERE
    m.JOURNAL = ROOT / "artifacts" / HERE.name / "owner"
    m.FINAL = ROOT / "artifacts/predictive-C100-private-backup-20261021-v1/owner-result.json"
    return m


if __name__ == "__main__":
    os.umask(0o077)
    implementation().main()
