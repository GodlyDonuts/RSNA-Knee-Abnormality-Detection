"""Appended to the isolated C100 policy; no account calls or state changes."""

import json

PRIMARY_FAMILY = "licensed-parent-full-epoch-serving:predictive-final-v1:"
PRIMARY_MODELS = {
    "B": "5a1ad987c046a7783c692c95b9753dbae9db198c5caebfa5bedce206562d6d0e",
    "C": "8231ea03a4109509ec162e2a19b0a477de1affe31ac4ac53ab22bf287723bfbd",
}


def protect_primary_budget(plan, db):
    """Release 3360 seconds only after the same original B/C/C0 bundle submitted.

    Inspect the canonical ledger under its existing claim. A caller-provided
    zero reservation or a later dated status summary cannot retire this budget.
    Hidden scoring may continue; these public GPU executions are then finished.
    """
    bundles = {}
    for key, raw in db.execute(
        "SELECT key, value FROM meta WHERE key LIKE ?", (PRIMARY_FAMILY + "%",)
    ):
        row = json.loads(raw)
        identity, separator, mode = key[len(PRIMARY_FAMILY) :].partition(":")
        if (
            not separator
            or mode not in ("B", "C", "C0")
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
            or row.get("bundle_identity") != identity
            or row.get("mode") != mode
            or row.get("state") != "predictive_final_submitted"
            or type(row.get("submission_id")) is not int
            or row["submission_id"] <= 0
            or row.get("version") != 1
        ):
            continue
        files = row.get("model_files", {})
        expected = "predictive-" + mode + "-epoch-006.pt"
        if set(files) != {expected}:
            continue
        digest = files[expected].get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            continue
        if mode in PRIMARY_MODELS and digest != PRIMARY_MODELS[mode]:
            continue
        bundles.setdefault(identity, {})[mode] = row["submission_id"]
    complete = any(
        set(entries) == {"B", "C", "C0"} and len(set(entries.values())) == 3
        for entries in bundles.values()
    )
    return {
        **plan,
        "other_reserved_VM_seconds": max(
            plan["other_reserved_VM_seconds"], 0 if complete else 3360
        ),
    }
