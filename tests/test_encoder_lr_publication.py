import copy
import json
from pathlib import Path

import pytest

from rsna.competitive.encoder_lr_publication import (
    ORIGINAL_STATUS,
    TREATMENT_STATUS,
    sum_checkpoint_receipts,
)

ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    "initiative/audits/research-progress-20261017-0550-v1/actual-influence-E1-private-verified.json",
    "initiative/audits/iid16-session-status-recovery-20261017-v1/actual-influence-E2-private-verified.json",
    "initiative/audits/097-label-campaign-continuation-20261017-v1/influence-final-private-verified.json",
    "initiative/audits/research-progress-20261017-0448-v1/actual-uniform-E1-private-verified.json",
    "initiative/audits/research-progress-20261017-0735-v1/actual-uniform-E2-private-verified.json",
    "initiative/audits/lockin-097-execution-20261017-v2/uniform-final-private-verified.json",
)


@pytest.fixture
def old():
    return [json.loads((ROOT / name).read_text()) for name in PATHS]


def new_format(old):
    # Actual old receipt shape plus synthetic new-class fields: not a real new epoch.
    row = copy.deepcopy(old[0])
    row["status"] = TREATMENT_STATUS
    row["independent_CPU_state_verification"].update(encoder_lr=3e-5, head_lr=3e-5)
    return row


def test_real_old_receipts_and_synthetic_new_format_keep_independent_credit(old):
    assert {x["status"] for x in old} == {ORIGINAL_STATUS}
    assert sum_checkpoint_receipts(old, collection="original_iid16") == sum(
        x["retained_original_bytes"] for x in old
    )
    row = new_format(old)
    assert sum_checkpoint_receipts([row], collection="encoder_lr") == row["retained_original_bytes"]
    assert sum_checkpoint_receipts([], collection="encoder_lr") == 0
    with pytest.raises(ValueError):
        sum_checkpoint_receipts(old, collection="encoder_lr")
    with pytest.raises(ValueError):
        sum_checkpoint_receipts([row, *old[1:]], collection="original_iid16")


@pytest.mark.parametrize("damage", ["status", "missing", "duplicate", "bytes", "step", "model"])
def test_corrupt_or_missing_original_funding_proofs_fail(old, damage):
    if damage == "status":
        old[0]["status"] = TREATMENT_STATUS
    elif damage == "missing":
        old.pop()
    elif damage == "duplicate":
        old[-1] = copy.deepcopy(old[0])
    elif damage == "bytes":
        old[0]["retained_original_bytes"] = -1
    elif damage == "step":
        old[0]["completed"] = 1
    else:
        old[0]["independent_CPU_state_verification"]["actual_AdamW_parameters"] = 1
    with pytest.raises(ValueError):
        sum_checkpoint_receipts(old, collection="original_iid16")


def test_new_collection_cannot_skip_epochs_or_claim_old_receipts(old):
    row = new_format(old)
    row.update(epoch=2, completed=972)
    row["independent_CPU_state_verification"]["finite_contiguous_updates"] = 972
    with pytest.raises(ValueError):
        sum_checkpoint_receipts([row], collection="encoder_lr")
