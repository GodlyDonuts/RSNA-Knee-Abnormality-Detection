"""One encoder-LR treatment over the unchanged accepted influence trainer."""

from __future__ import annotations

import ast
import base64

from scripts.kaggle import licensed_importance_production as original

PLAN = "licensed_influence_encoderLR3e5_fixed3887_3epoch_v1"
STATUS = "actual_encoderLR3e5_influence_full_parent_epoch_complete"
LR = 3e-5


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Exact single source substitution required")
    return text.replace(before, after)


def literals(source):
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"WORKER", "BINDING", "SEGMENT", "PREVIOUS"}
    }


def build(*args, segment, previous=None):
    """Reuse exact source/input admission; alter encoder LR only in the worker.

    Previous recovery remains evidence for the unchanged engine, not an actual
    recovery or accuracy result for this new LR. Old serving/epoch validators
    deliberately remain unchanged and cannot admit these treatment weights.
    """
    if type(segment) is not int or not 1 <= segment <= 3:
        raise ValueError("Fixed three epochs required")
    source, meta = original.build(*args, "influence", 1)
    values = literals(source)
    old = values["BINDING"]
    binding = {
        **old,
        "production_plan": PLAN,
        "single_factor": {"encoder_lr_control": 3e-6, "encoder_lr_treatment": LR},
        "actual_new_LR_recovery_claim": False,
    }
    worker = base64.b64decode(values["WORKER"], validate=True).decode()
    worker = once(worker, "_BINDING=" + repr(old), "_BINDING=" + repr(binding))
    worker = once(
        worker,
        "Recipe(epochs=3,role=_ROLE)",
        "Recipe(epochs=3,role=_ROLE,encoder_lr=3e-5)",
    )
    worker_sha = original.digest(worker)
    if segment == 1:
        if previous is not None:
            raise ValueError("Treatment starts from the original parent")
    else:
        if (
            not isinstance(previous, dict)
            or previous.get("status") != STATUS
            or previous.get("role") != "influence"
            or previous.get("segment") != segment - 1
            or previous.get("completed") != 486 * (segment - 1)
            or previous.get("worker_sha256") != worker_sha
            or previous.get("binding") != binding
        ):
            raise ValueError("Only exact previous treatment optimizer boundary accepted")
        for key in ("checkpoint", "run"):
            item = previous.get(key, {})
            h = item.get("sha256")
            if (
                type(item.get("bytes")) is not int
                or not 0 < item["bytes"] <= 2 * 2**30
                or not isinstance(h, str)
                or len(h) != 64
                or any(c not in "0123456789abcdef" for c in h)
            ):
                raise ValueError("Original treatment checkpoint/run bytes and SHA required")
    for before, after in (
        (
            "WORKER=" + repr(values["WORKER"]),
            "WORKER=" + repr(base64.b64encode(worker.encode()).decode()),
        ),
        ("BINDING=" + repr(old), "BINDING=" + repr(binding)),
        ("SEGMENT=1", "SEGMENT=" + repr(segment)),
        ("PREVIOUS=None", "PREVIOUS=" + repr(previous)),
        ("actual_matched_IID16_full_parent_epoch_session_complete", STATUS),
        (
            "state['binding']['recipe']['role']==ROLE",
            "state['binding']['recipe']['encoder_lr']==3e-5 and state['binding']['recipe']['head_lr']==3e-5 and state['binding']['recipe']['role']==ROLE",
        ),
    ):
        source = once(source, before, after)
    compile(worker, "single_encoder_LR_treatment_worker", "exec")
    compile(source, "single_encoder_LR_treatment_bootstrap", "exec")
    return source, {
        **meta,
        "source_sha256": original.digest(source),
        "worker_sha256": worker_sha,
        "binding": binding,
        "segment": segment,
        "completed_target": 486 * segment,
    }
