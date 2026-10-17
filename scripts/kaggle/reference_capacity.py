"""Reconcile one proven cancellation omitted by the old admission helper."""

from copy import deepcopy

OLD_SLUG = "godlydonuts/rsna-public0958-native-300b34417bc1"
OLD_SHA = "e13e0e84e701c61ac400f9cc39254c7690f66961ffd47ef7dae9ebb2772cabb6"


def acknowledged_reference_capacity(raw, proof):
    """Keep every unknown/live session; retire only the exact actual terminal."""
    if not (
        proof.get("kernel_id") == 139199164
        and proof.get("version") == 2
        and proof.get("actual_session", {}).get("status") == "CANCEL_ACKNOWLEDGED"
        and proof.get("fresh_source_sha256") == OLD_SHA
        and proof.get("original_source_preserved") is True
        and raw.get("maximum_batch_GPU_sessions") == 2
    ):
        raise ValueError("Exact original source and actual cancellation acknowledgement required")
    out = deepcopy(raw)
    active, retired = [], []
    for row in raw["actual_active_registered_sessions"]:
        if (row["slug"], row["version"], row["kernel_id"]) == (OLD_SLUG, 2, 139199164):
            if row.get("status") != "CANCEL_ACKNOWLEDGED" or row.get("main") is not False:
                raise ValueError("Do not retire a live, unknown, or differently owned reference")
            retired.append(row)
        else:
            active.append(row)
    if len(retired) != 1:
        raise ValueError("One exact stale terminal reference must be present")
    if raw["effective_occupied_lanes"] != raw["reserved_main_lanes"] + sum(
        not row["main"] for row in raw["actual_active_registered_sessions"]
    ):
        raise ValueError("Original capacity accounting must agree before reconciliation")
    occupied = raw["reserved_main_lanes"] + sum(not row["main"] for row in active)
    out.update(
        actual_active_registered_sessions=active,
        effective_occupied_lanes=occupied,
        available=occupied < 2,
        actual_terminal_reference_retired=retired,
        original_raw_admission=deepcopy(raw),
        no_platform_state_or_live_helper_changes=True,
    )
    return out
