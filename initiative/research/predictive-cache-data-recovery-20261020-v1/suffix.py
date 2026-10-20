# ruff: noqa: F821
"""Append-only exact failed-HTTP-generation and two-source correction contract."""

import types

_http_guard = types.FunctionType(predecessor_guard.__code__, dict(globals()))
PREDECESSOR_SOURCE = "d7326951eeb986f26eb21183d61c2b52476a7986"
PREDECESSORS = {"raw": "980960", "context": "980969"}
INTENTS = {
    "raw": "530c1068bd680cab3c0a9d7835b3d6644a51706ac2c19f2327ea8bfb23dd9e58",
    "context": "af7aa20d269f9d04463bfb46629268be5390c47040729d31c3cecaf3fbc9ef3d",
}
DATA_PREDECESSOR_ROOTS = {
    role: f"/tmp/rsna-predictive-full-{role}-http-recovery-20261020-v1" for role in PREDECESSORS
}
POLICY = "baseline_plus_exact_authoritative_two_file_patch"
ADMISSION_STATUS = "actual_mixed_HTTP_prefix_and_authoritative_overlay_read_only_admitted"


def correction_contract(value):
    body = value["source_correction_decision_body"].encode()
    require(sha(body) == value["source_correction_decision_sha256"], "Correction decision changed")
    decision = json.loads(body)
    require(decision["status"] == "accepted_original_source_correction_missing_only_recovery"
            and decision["original_jobs"] == PREDECESSORS
            and decision["all_originals_preserved"] is True
            and decision["original_science_preserved"] is True
            and decision["scratch_cap_bytes"] == 6 * 2**30,
            "Accepted exact source correction required")
    for key in ("plan", "receipt", "helper", "regression"):
        name = "source_correction_" + key
        require(decision[name + "_sha256"] == value[name + "_sha256"], "Correction evidence differs")
        require(sha(value[name + ("_source" if key == "helper" else "_body")].encode())
                == value[name + "_sha256"], "Correction evidence bytes differ")
    require(decision["corrected_sources"] == value["corrected_sources"], "Resident source refs differ")
    proof = json.loads(value["source_correction_regression_body"])
    require(proof["status"] == "actual_existing_authoritative_overlay_readonly_reproduction"
            and proof["CPU_job"] == "981840"
            and proof["failed_source_matches_accepted_before"] is True
            and proof["frames"] == 188 and proof["groups"] == 3
            and proof["stages"] == [{"stage": s, "passed": True} for s in ("decoder", "conversion", "assembly")]
            and proof["source_HTTP_requests"] == 0 and proof["remote_writes"] == 0
            and proof["labels_changed"] is False, "Actual original strict decoder proof required")
    return decision


def predecessor_guard(value, recon, check_output, roots):
    ancestors = _http_guard(value, recon, check_output, roots)
    correction_contract(value)
    accounting = check_output(["sacct", "-X", "-n", "-P", "-j", "980960,980969",
                               "--format=JobIDRaw,State,ExitCode"], text=True)
    queue = check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"], text=True)
    result = {}
    for role, job in PREDECESSORS.items():
        terminal = recon.terminal_receipt(accounting, queue, job_id=job, allowed_states=("FAILED",))
        require(terminal["exit_code"] == "1:0", "Original source failure terminal differs")
        root = Path(DATA_PREDECESSOR_ROOTS[role])
        recon.private_directory(root)
        require(not (root / "actual.json").exists(), "Never repeat completed producer")
        recon.regular(root / "intent.json")
        body = (root / "intent.json").read_bytes()
        intent = json.loads(body)
        require(sha(body) == INTENTS[role] and intent["CPU_job"] == job
                and intent["source_git"] == PREDECESSOR_SOURCE
                and intent["continuation_chain"] == ancestors, "Exact failed source ancestry differs")
        result[role] = {"job": job, "root": str(root), "intent_sha256": sha(body),
                        "source_git": PREDECESSOR_SOURCE, "actual_terminal": terminal,
                        "ancestor_chain": ancestors}
    return result


def buffer_terminal(chain, root, original_terminal):
    path = str(root)
    if path == "/tmp/rsna-predictive-full-raw-cache-20261020-v1":
        return original_terminal
    current = chain
    while current:
        item = current["raw"]
        if item["root"] == path:
            return item["actual_terminal"]
        current = item.get("ancestor_chain")
    raise ValueError("Retained source buffer has no exact terminal owner")


def overlay(value, records, original_index_sha256):
    correction_contract(value)
    namespace = {}
    exec(compile(value["source_correction_helper_source"], "exact_source_table_correction", "exec"), namespace)
    result = namespace["apply"](records, value["source_correction_plan_body"].encode(),
                                value["source_correction_receipt_body"].encode())
    lineage = result["lineage"]
    require(lineage["studies"] == 3887 and lineage["source_members"] == 720851
            and lineage["original_source_bytes"] == 500897467290
            and lineage["effective_source_bytes"] == 500898049946
            and lineage["byte_delta"] == 582656, "Only the complete exact two-file overlay is admitted")
    require({c["source_ordinal"] for c in lineage["changes"]} == {1740, 1850}, "Affected study set differs")
    corrections, resident_paths = {}, {}
    for change in lineage["changes"]:
        before, after = change["before"], change["after"]
        corrections[change["source_ordinal"]] = {
            "effective_source_policy": POLICY,
            "original_revision": lineage["original_revision"],
            "corrected_revision": lineage["corrected_revision"],
            "plan_sha256": value["source_correction_plan_sha256"],
            "receipt_sha256": value["source_correction_receipt_sha256"],
            "before_sha256": before["sha256"], "before_bytes": before["bytes"],
            "after_sha256": after["sha256"], "after_bytes": after["bytes"],
        }
        ref = value["corrected_sources"][after["sha256"]]
        require(ref["sha256"] == after["sha256"] and ref["bytes"] == after["bytes"], "Resident exact after ref differs")
        resident_paths[after["path"]] = Path(ref["path"])
    require(len(value["corrected_sources"]) == 2, "Exactly two resident corrected sources required")
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(result["records"]):
        digest.update(part.encode())
    metadata = {"effective_source_policy": POLICY,
                "original_source_index_sha256": original_index_sha256,
                "effective_source_index_sha256": digest.hexdigest(),
                "source_correction_revision": lineage["corrected_revision"],
                **{key: value[key] for key in (
                    "source_correction_plan_sha256", "source_correction_receipt_sha256",
                    "source_correction_helper_sha256", "source_correction_regression_sha256",
                    "source_correction_decision_sha256", "source_overlay_resume_sha256")}}
    if value.get("mixed_prefix_admission") is not None:
        admitted = admission_receipt(value)
        require(all(admitted.get(k) == x for k, x in metadata.items()), "Admitted effective source index differs")
    return result["records"], corrections, metadata, resident_paths


def require_unsealed(inventory, corrections):
    require(not (set(corrections) & (set(inventory["complete"]) | set(inventory["orphan_caches"]))),
            "Corrected source must not replace already accepted or orphan cache bytes")


def overlay_module(value):
    import sys
    module = types.ModuleType("source_correction")
    exec(compile(value["source_correction_helper_source"], "exact_source_correction", "exec"), module.__dict__)
    sys.modules["source_correction"] = module
    require(sha(value["source_overlay_resume_source"].encode()) == value["source_overlay_resume_sha256"],
            "Exact source resume helper differs")
    overlay = types.ModuleType("exact_resident_overlay_resume")
    exec(compile(value["source_overlay_resume_source"], "exact_resident_overlay_resume", "exec"), overlay.__dict__)
    return overlay


def extend_one(text):
    anchors = {
        '    p = output_path(i)': '    if i in corrections:\n        provenance["source_correction"] = corrections[i]\n    p = output_path(i)',
        '    compressed = gzip.compress(json.dumps(source_proof).encode())':
            '    if i in corrections:\n        source_proof["source_correction"] = corrections[i]\n    compressed = gzip.compress(json.dumps(source_proof).encode())',
        '    encoded = json.dumps(record) + "\\n"':
            '    if i in corrections:\n        record["source_correction"] = corrections[i]\n    encoded = json.dumps(record) + "\\n"',
    }
    for old, new in anchors.items():
        require(text.count(old) == 1, "Original one-case provenance anchor differs")
        text = text.replace(old, new)
    return text


def admission_receipt(value):
    body = value["mixed_prefix_admission"].encode()
    require(sha(body) == value["mixed_prefix_admission_sha256"], "Mixed original source admission SHA differs")
    receipt = json.loads(body)
    require(receipt["status"] == ADMISSION_STATUS and receipt["original_jobs"] == PREDECESSORS
            and receipt["no_original_writes"] is True
            and receipt["suffix_reconcile_sha256"] == value["suffix_reconcile_sha256"]
            and receipt["source_correction_decision_sha256"] == value["source_correction_decision_sha256"],
            "Actual exact source-corrected prefix admission required")
    return receipt
