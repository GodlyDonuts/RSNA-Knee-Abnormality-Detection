"""Synthetic complete-state checks, never actual epoch or accuracy evidence."""

import copy
import hashlib
import json
import runpy
from dataclasses import asdict
from pathlib import Path

import torch
from epoch_verify import verify

from rsna.competitive.licensed_importance import VERSION, Recipe
from rsna.competitive.licensed_importance_epoch import PROPOSAL_SHA

D = Path(__file__).parent
source = runpy.run_path(str(D.parent / "multisource-supervision-v1/sampled_overlay.py"))
state, snapshot, receipt = runpy.run_path("tests/test_licensed_parent_epoch.py")[
    "completed"
].__wrapped__()
labels = {
    "version": source["VERSION"],
    "role": "safe_family_mean",
    "diagnostic_only": False,
    "population": 3887,
    "population_sha256": source["POPULATION"],
    "parent_checkpoint_sha256": source["PARENT_SHA"],
    "original_target_file_sha256": source["ORIGINAL_FILE"],
    "original_labels_sha256": source["PRODUCTION_ORIGINAL_LABELS"],
    "labels_sha256": source["PRODUCTION_LABELS"],
    "mask_sha256": source["PRODUCTION_MASK"],
    "teacher_sha256": source["PRODUCTION_TEACHER"],
    "target_file_sha256": source["PRODUCTION_TARGET"],
    "annotation_file_sha256": source["PRODUCTION_ANNOTATIONS"],
    "eligible_sha256": "f" * 64,
}
state["binding"].update(
    role="influence",
    version=VERSION,
    recipe=asdict(Recipe(role="influence")),
    population_sha256=source["POPULATION"],
    proposal={"proposal_table": {"bytes": 2923152, "sha256": PROPOSAL_SHA}},
    teacher_proposal_sha256="a" * 64,
    actual_recovery={"kernel_id": 139130771},
    production_plan="licensed_MRI_parent_family_mean_influence16_full3887_fixed3_v1",
    source_label_role="safe_family_mean",
    source_label_target_sha256=source["PRODUCTION_TARGET"],
    annotation_file_sha256=source["PRODUCTION_ANNOTATIONS"],
    multisource_supervision=labels,
    actual_family_recovery={
        "kernel_id": 139336728,
        "version": 1,
        "source_sha256": "3fd77316770abec5b05a6297e6f3a38c73458b86f84cb2d6b6763ecd37c6f825",
        "worker_sha256": "acfa946919bdf1d9fbff3b34c196d4e5c4545117bdadb9ff68da419afbfdd5d5",
        "receipt_canonical_JSON_sha256": "f" * 64,
    },
)
receipt.update(
    status="actual_family_influence16_full_parent_epoch_session_complete", role="influence"
)
assert verify(state, snapshot, receipt, expected_parameters=3)["actual_AdamW_parameters"] == 3
refusals = []
for damage in (
    "labels",
    "mask",
    "teacher",
    "annotation",
    "scope",
    "recovery",
    "optimizer",
    "model",
    "last_batch",
    "status",
    "history",
    "rng",
):
    s, p, r = copy.deepcopy((state, snapshot, receipt))
    if damage in ("labels", "mask", "teacher"):
        s["binding"]["multisource_supervision"][damage + "_sha256"] = "0" * 64
    elif damage == "annotation":
        s["binding"]["multisource_supervision"]["annotation_file_sha256"] = "0" * 64
    elif damage == "scope":
        s["binding"]["multisource_supervision"]["diagnostic_only"] = True
    elif damage == "recovery":
        s["binding"]["actual_family_recovery"]["kernel_id"] = 139334242
    elif damage == "optimizer":
        s["optimizer"]["state"][0]["step"] = torch.tensor(485.0)
    elif damage == "model":
        p["model"]["backbone.weight"][0] = 2.0
    elif damage == "last_batch":
        s["history"][-1]["studies"] = 8
    elif damage == "status":
        r["status"] = "actual_matched_IID16_full_parent_epoch_session_complete"
    elif damage == "history":
        s["history"].pop()
    else:
        s["rng_by_rank"].pop()
    try:
        verify(s, p, r, expected_parameters=3)
    except ValueError:
        refusals.append(damage)
    else:
        raise AssertionError(damage)
result = {
    "status": "synthetic_family_epoch_complete_state_and_twelve_refusals_passed",
    "validator_sha256": hashlib.sha256((D / "epoch_verify.py").read_bytes()).hexdigest(),
    "refusals": refusals,
    "actual_epoch_or_accuracy": False,
}
(D / "epoch-verifier-checked.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
