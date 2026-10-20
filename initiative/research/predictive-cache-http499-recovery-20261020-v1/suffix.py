# ruff: noqa: F821
"""One exact terminal HTTP499 generation; sealed corrected bytes stay unchanged."""

_data_guard = types.FunctionType(predecessor_guard.__code__, dict(globals()))
_data_overlay = overlay
HTTP499_PREDECESSORS = {"raw": "981958", "context": "981960"}
HTTP499_SOURCE = "e9bdd1249fa7cf75f792b3762c2028bd7beb4337"
HTTP499_INTENTS = {
    "raw": "7ad45b9e78c7fc987bd913356b3f91c615f9ab446f92dc626b411da154c9d27d",
    "context": "e32a42c84d2b99a8185d19d18178a77fa972665e6008940e983f15e6aac7425c",
}
HTTP499_ROOTS = {
    r: f"/tmp/rsna-predictive-full-{r}-data-recovery-20261020-v1" for r in HTTP499_PREDECESSORS
}
HTTP499_DATA_ROOT = Path(HTTP499_ROOTS["raw"])
HTTP499_STATUS = "actual_mixed_corrected_five_generation_HTTP499_prefix_read_only_admitted"
PREVIOUS_HTTP_SHA = "ea7bf0f44af60ec4b0876e85af5cc1536d15cad671ea45d4ac548adccca0e28c"


def http499_metadata(value):
    return {key: value[key] for key in ("http499_recovery_decision_sha256",
                                      "previous_source_http_sha256", "source_http_sha256")}


def http499_contract(value):
    body = value["http499_recovery_decision_body"].encode()
    require(sha(body) == value["http499_recovery_decision_sha256"], "Exact HTTP499 decision required")
    decision = json.loads(body)
    require(decision["status"] == "accepted_original_HTTP499_missing_only_recovery"
            and decision["original_jobs"] == HTTP499_PREDECESSORS
            and decision["predecessor_source_git"] == HTTP499_SOURCE
            and decision["verified_failure_site"] == "HTTP_response_status_before_body"
            and decision["actual_HTTP_status"] == 499
            and decision["all_originals_preserved"] is True
            and decision["original_science_preserved"] is True
            and decision["authoritative_corrections_unchanged"] is True
            and decision["source_http_sha256"] == value["source_http_sha256"]
            and decision["previous_source_http_sha256"] == value["previous_source_http_sha256"] == PREVIOUS_HTTP_SHA
            and decision["lane_count"] == 4 and decision["maximum_HTTP_requests"] == 8
            and decision["scratch_cap_bytes"] == 6 * 2**30
            and decision["funding"] == value["funding"], "Exact diagnosed499 correction/funding required")
    require(sha(value["http499_failure_receipt_body"].encode())
            == decision["actual_failure_receipt_sha256"], "Actual failure receipt differs")
    return decision


def predecessor_guard(value, recon, check_output, roots):
    # All older decision checks retain their exact old transport identity.
    old_value = {**value, "source_http_sha256": PREVIOUS_HTTP_SHA}
    ancestors = _data_guard(old_value, recon, check_output, roots)
    http499_contract(value)
    accounting = check_output(["sacct", "-X", "-n", "-P", "-j", "981958,981960",
                               "--format=JobIDRaw,State,ExitCode"], text=True)
    queue = check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"], text=True)
    result = {}
    for role, job in HTTP499_PREDECESSORS.items():
        terminal = recon.terminal_receipt(accounting, queue, job_id=job, allowed_states=("FAILED",))
        require(terminal["exit_code"] == "1:0", "Actual499 predecessor terminal differs")
        root = Path(HTTP499_ROOTS[role])
        recon.private_directory(root)
        require(not (root / "actual.json").exists(), "Completed original cannot be continued")
        intent_body = (root / "intent.json").read_bytes()
        intent = recon.bound_json(root / "intent.json")
        require(sha(intent_body) == HTTP499_INTENTS[role] and intent["CPU_job"] == job
                and intent["source_git"] == HTTP499_SOURCE
                and intent["source_http_sha256"] == PREVIOUS_HTTP_SHA
                and intent["continuation_chain"] == ancestors, "Exact source-corrected predecessor required")
        result[role] = {"job": job, "root": str(root), "intent_sha256": sha(intent_body),
                        "source_git": HTTP499_SOURCE, "actual_terminal": terminal,
                        "ancestor_chain": ancestors}
    return result


def configure(value, records, original_index_sha256):
    global _effective_records, _source_corrections
    _effective_records, _source_corrections, metadata, resident = _data_overlay(
        value, records, original_index_sha256)
    return _effective_records, _source_corrections, metadata, resident


def require_unsealed(inventory, corrections):
    """The two authoritative cases are already complete; reuse their exact rows."""
    require(set(corrections) == {1740, 1850}, "Exact historical correction set required")
    for i, correction in corrections.items():
        require(i in inventory["complete"] and i not in inventory["orphan_caches"]
                and inventory["complete"][i].get("source_correction") == correction,
                "Both authoritative corrected raw cases must remain sealed and unchanged")


def require_corrected_contexts(inventory, corrections):
    for i, correction in corrections.items():
        require(i in inventory["complete"] and i not in inventory["orphan_caches"]
                and inventory["complete"][i]["provenance"].get("source_correction") == correction,
                "Both authoritative corrected contexts must remain sealed and unchanged")


def admission_receipt(value):
    body = value["mixed_prefix_admission"].encode()
    require(sha(body) == value["mixed_prefix_admission_sha256"], "Actual mixed-prefix SHA differs")
    result = json.loads(body)
    require(result["status"] == HTTP499_STATUS and result["original_jobs"] == HTTP499_PREDECESSORS
            and result["no_original_writes"] is True
            and result["suffix_reconcile_sha256"] == value["suffix_reconcile_sha256"]
            and result["source_correction_decision_sha256"] == value["source_correction_decision_sha256"]
            and all(result.get(k) == x for k, x in http499_metadata(value).items()),
            "Actual five-generation corrected-prefix admission required")
    return result
