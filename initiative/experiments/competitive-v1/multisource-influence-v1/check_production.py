"""Synthetic receipt contract checks; never evidence of an actual GPU pass."""

import copy
import hashlib
import json
from pathlib import Path

import production as p

value = {
    "finite_updates": 4,
    "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal": True,
    "all16_full94_two_view_prediction_bytes_equal": True,
}
receipt = {
    "status": "actual_family_label_only_influence16_twoT4_MRI_recovery_accepted",
    "worker_source_sha256": p.RECOVERY_WORKER,
    "family_target_sha256": p.TARGET,
    "source_module_sha256": {},
    "original_control_complete_state_and_predictions_exact_except_label_binding": True,
    "same_initial_complete_model_and_allrank_RNG_between_roles_and_original": True,
    "clinical_or_competition_accuracy_claim": False,
    "roles": {
        role: {**value, "label_binding": {"role": role, "diagnostic_only": True}}
        for role in ("unchanged_reference", "safe_family_mean")
    },
}
actual = {
    "status": "COMPLETE",
    "source_sha256": p.RECOVERY,
    "kernel_id": 139336728,
    "version": 1,
    "receipt": receipt,
}
assert p.accept_recovery(actual, {}) == receipt
refusals = []


def refuse(name, change):
    broken = copy.deepcopy(actual)
    change(broken)
    try:
        p.accept_recovery(broken, {})
    except ValueError:
        refusals.append(name)
    else:
        raise AssertionError(name)


refuse("pending is not accepted", lambda x: x.update(status="RUNNING"))
refuse("different source", lambda x: x.update(source_sha256="0" * 64))
refuse("old uniform kernel", lambda x: x.update(kernel_id=139334242))
refuse("new unreviewed version", lambda x: x.update(version=2))
refuse("wrong worker", lambda x: x["receipt"].update(worker_source_sha256="0" * 64))
refuse("wrong targets", lambda x: x["receipt"].update(family_target_sha256="0" * 64))
refuse("missing module", lambda x: x["receipt"].update(source_module_sha256={"x": "0" * 64}))
refuse(
    "lost control equality",
    lambda x: x["receipt"].update(
        original_control_complete_state_and_predictions_exact_except_label_binding=False
    ),
)
refuse(
    "lost prediction equality",
    lambda x: x["receipt"]["roles"]["safe_family_mean"].update(
        all16_full94_two_view_prediction_bytes_equal=False
    ),
)
refuse("missing source arm", lambda x: x["receipt"]["roles"].pop("safe_family_mean"))
refuse(
    "zero update",
    lambda x: x["receipt"]["roles"]["unchanged_reference"].update(finite_updates=0),
)
assert p.assignments("A=1\nB=2\n", {"A": 3}) == "A=3\nB=2\n"
for text in ("A=1\nA=2\n", "B=2\n"):
    try:
        p.assignments(text, {"A": 3})
    except ValueError:
        pass
    else:
        raise AssertionError("Unique complete source assignment required")
compile(p.HELPER, "production_family_array_loader", "exec")
result = {
    "status": "synthetic_production_recovery_contract_and_helper_compile_passed",
    "compiler_sha256": hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
    "refusals": refusals,
    "actual_GPU_or_full_fit_or_accuracy": False,
}
Path(__file__).with_name("production-checked.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
