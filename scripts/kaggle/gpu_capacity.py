"""Read-only admission for the campaign's two shared Kaggle GPU sessions.

Keep one lane for the unfinished original whole94 continuation. Serving and the
matched sampler share the other lane; a short, live serving lease has priority.
The caller must hold the canonical account claim through admission and SaveKernel.
This covers registered campaign families, not unrelated/manual account sessions.
"""

from __future__ import annotations

import json
import time

FAMILIES = (
    "licensed-parent-whole94-full-fit:",
    "licensed-importance-full-fit:",
    "licensed-parent-full-epoch-serving:",
)


def admission(db, remote, *, caller, now=None):
    if caller not in ("serving", "sampling"):
        raise ValueError("Known auxiliary campaign caller required")
    now = time.time() if now is None else now
    refs: dict[tuple[str, int], dict] = {}
    final_main_complete = False
    priority = []

    def visit(value, main):
        if isinstance(value, dict):
            if "slug" in value and type(value.get("kernel_id")) is int and value["kernel_id"] > 0:
                slug, version = value.get("slug"), value.get("version", 1)
                if not isinstance(slug, str) or not slug.startswith(remote.username + "/"):
                    raise ValueError("Registered GPU identity must belong to the actual account")
                if type(version) is not int or version < 1:
                    raise ValueError("Exact registered GPU version required")
                key = (slug, version)
                entry = {
                    "slug": slug,
                    "version": version,
                    "kernel_id": value["kernel_id"],
                    "main": main,
                }
                if key in refs and refs[key] != entry:
                    raise ValueError("Conflicting canonical GPU identity")
                refs[key] = entry
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child, main)
        elif isinstance(value, list):
            for child in value:
                visit(child, main)

    for key, raw in db.execute("SELECT key,value FROM meta"):
        if not key.startswith(FAMILIES):
            continue
        value = json.loads(raw)
        main = key.startswith(FAMILIES[0])
        state = value.get("state", "")
        if main and state == "whole94_full_fit_actual_complete" and value.get("segment") == 3:
            if value.get("receipt", {}).get("completed") != 1458:
                raise ValueError("Actual fixed-final main completion required to release its lane")
            final_main_complete = True
        if (
            key.startswith(FAMILIES[2])
            and state == "full_parent_epoch_qualification_accepted"
            and type(value.get("GPU_wait_until")) in (int, float)
            and now < value["GPU_wait_until"] <= now + 300
        ):
            priority.append(key)
        if state.endswith(("_pushed", "_push_uncertain", "_push_intent")):
            visit(value, main)

    active = []
    for value in refs.values():
        metadata = remote.kernel("GetKernel", value["slug"], value["version"])["metadata"]
        if metadata.get("enableGpu") is not True:
            continue
        status = remote.kernel("GetKernelSessionStatus", value["slug"], value["version"]).get(
            "status"
        )
        if status not in ("COMPLETE", "ERROR", "CANCELLED"):
            active.append({**value, "status": status})
    main_active = sum(v["main"] for v in active)
    auxiliary_active = len(active) - main_active
    main_lane = max(main_active, int(not final_main_complete))
    occupied = main_lane + auxiliary_active
    return {
        "available": occupied < 2 and not (caller == "sampling" and priority),
        "maximum_batch_GPU_sessions": 2,
        "actual_active_registered_sessions": active,
        "reserved_main_lanes": main_lane,
        "effective_occupied_lanes": occupied,
        "serving_priority_waiters": priority,
        "new_GPU_mutations": 0,
    }
