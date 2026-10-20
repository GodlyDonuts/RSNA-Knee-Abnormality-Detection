"""Append to frozen v4 reconciliation; preserve all numerical/source checks."""

import types

# Preserve the exact original globals before extending predecessor ownership.
_capacity_guard = types.FunctionType(predecessor_guard.__code__, dict(globals()))  # noqa: F821
PREDECESSOR_SOURCE = "370f2c598f4a25d92fc4764c899a1632ff50b675"
PREDECESSORS = {"raw": "980743", "context": "980744"}
INTENTS = {
    "raw": "2c301386f28e3a5de96886dbbfc031317e73c324c3d58218e3e5360c9debc656",
    "context": "211870c05293866180e5c8e82f08f48a23fe6310e012362f5ef769b647247036",
}
HTTP_ROOTS = {
    role: f"/tmp/rsna-predictive-full-{role}-capacity-recovery-20261020-v1"
    for role in PREDECESSORS
}


def predecessor_guard(value, recon, check_output, roots):
    ancestors = _capacity_guard(value, recon, check_output, roots)
    raw = value["http_recovery_decision_body"].encode()
    require(sha(raw) == value["http_recovery_decision_sha256"], "HTTP decision changed")  # noqa: F821
    decision = json.loads(raw)  # noqa: F821
    require(  # noqa: F821
        decision["status"] == "accepted_original_v4_HTTP_failure_reconciled"
        and decision["original_jobs"] == PREDECESSORS
        and decision["verified_failure_site"] == "HTTP_response_status_before_body"
        and decision["all_originals_preserved"] is True
        and decision["source_http_sha256"] == value["source_http_sha256"],
        "Exact accepted HTTP failure and source transport required",
    )
    accounting = check_output(
        ["sacct", "-X", "-n", "-P", "-j", "980743,980744", "--format=JobIDRaw,State,ExitCode"],
        text=True,
    )
    queue = check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"], text=True)
    result = {}
    for role, job in PREDECESSORS.items():
        terminal = recon.terminal_receipt(accounting, queue, job_id=job, allowed_states=("FAILED",))
        require(terminal["exit_code"] == "1:0", "Original HTTP failure terminal changed")  # noqa: F821
        root = Path(HTTP_ROOTS[role])  # noqa: F821
        recon.private_directory(root)
        require(not (root / "actual.json").exists(), "Completed original must not restart")  # noqa: F821
        recon.regular(root / "intent.json")
        body = (root / "intent.json").read_bytes()
        intent = json.loads(body)  # noqa: F821
        require(  # noqa: F821
            sha(body) == INTENTS[role] and intent["CPU_job"] == job  # noqa: F821
            and intent["source_git"] == PREDECESSOR_SOURCE
            and intent["continuation_chain"] == ancestors,
            "Original HTTP generation intent/ancestry changed",
        )
        result[role] = {
            "job": job, "root": str(root), "intent_sha256": sha(body),  # noqa: F821
            "source_git": PREDECESSOR_SOURCE, "actual_terminal": terminal,
            "ancestor_chain": ancestors,
        }
    return result


def admission_receipt(value):
    body = value["mixed_prefix_admission"].encode()
    require(sha(body) == value["mixed_prefix_admission_sha256"], "Mixed prefix SHA differs")  # noqa: F821
    receipt = json.loads(body)  # noqa: F821
    require(  # noqa: F821
        receipt["status"] == "actual_mixed_v4_http_prefix_read_only_admitted"
        and receipt["original_jobs"] == PREDECESSORS and receipt["no_original_writes"] is True
        and receipt["suffix_reconcile_sha256"] == value["suffix_reconcile_sha256"]
        and receipt["http_recovery_decision_sha256"] == value["http_recovery_decision_sha256"]
        and receipt["source_http_sha256"] == value["source_http_sha256"],
        "Actual exact HTTP-successor prefix admission required",
    )
    return receipt
