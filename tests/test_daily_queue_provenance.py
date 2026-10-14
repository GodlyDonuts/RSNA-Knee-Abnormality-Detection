"""Committed-receipt provenance for the daily queue hashes; no remote access."""

import json
from pathlib import Path

from scripts.kaggle import reviewed_infer as reviewed

AUDITS = Path("initiative/audits")
QUEUE = AUDITS / "daily-kaggle-comparison-queue-20261014-v1"


def test_queue_code_and_binding_hashes_match_committed_production_receipts():
    cpu = json.loads(
        (AUDITS / "public-label-final-inputs-20261013-v1/actual-CPU-accepted.json").read_text()
    )
    code = {
        k.removeprefix("code/rsna/"): v
        for k, v in cpu["source_modules"].items()
        if k.startswith("code/rsna/")
    }
    production = (
        AUDITS / "final-public-source-fit-production-start-20261013-v1/production-admission.json"
    ).read_text()
    queue = json.loads((QUEUE / "queue.json").read_text())
    assert cpu["source_git"] == queue["scientific_source_git"] == reviewed.ALL_DEVELOPMENT_SOURCE
    assert reviewed.ALL_DEVELOPMENT_BINDING == queue["training_binding_sha256"]
    assert reviewed.ALL_DEVELOPMENT_BINDING in production
    assert [x["epoch"] for x in queue["entries"]] == [1, 2, 3, 4, 5]
    for entry in queue["entries"]:
        admission = json.loads((QUEUE / f"admissions/epoch-{entry['epoch']:03d}.json").read_text())
        assert entry["training_code_sha256"] == reviewed.object_sha(code)
        assert admission["training_code_sha256"] == reviewed.object_sha(code)
        assert admission["checkpoint_sha256"] == entry["checkpoint_sha256"]
