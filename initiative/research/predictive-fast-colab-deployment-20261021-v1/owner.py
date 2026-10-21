"""Distinct finite faster-serving handoff; no training or automatic retries."""

import os
from pathlib import Path

from run_once import BASE, git, load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = HERE.parent / "colab-C100-deployment-owner-20261021-v1"


def implementation():
    body = git(BASE, ORIGINAL / "owner.py")
    body = body.replace(
        "/tmp/rsna-predictive-final-deployment-20261020-v1-C100-deadline",
        "/tmp/rsna-predictive-final-deployment-20261020-v1-C100-selected-20261021-v1",
    )
    body = body.replace(
        "/tmp/rsna-predictive-final-deployment-20261020-v1-Colab-C100-deadline",
        "/tmp/rsna-predictive-final-deployment-20261020-v1-Colab-C100-selected-20261021-v1",
    )
    m = load(body, ORIGINAL / "owner.py", "original_final_owner")
    original_configuration = m.configuration

    def configuration(base, owner, observed, mode):
        value = original_configuration(base, owner, observed, mode)
        value["funding"]["other_node_unpaid_bytes"] = 518 * 2**30
        owner.policy.storage(
            value["funding"],
            node_free=observed["node_free_bytes"],
            shared_free=observed["shared_free_bytes"],
            shared_files=observed["shared_free_files"],
        )
        return value

    m.configuration = configuration
    m.HERE = HERE
    m.DEPLOY = HERE
    m.JOURNAL = ROOT / "artifacts" / HERE.name / "owner"
    return m


if __name__ == "__main__":
    os.umask(0o077)
    implementation().main()
