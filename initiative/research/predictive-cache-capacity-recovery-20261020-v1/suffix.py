"""Read-only mixed-generation reconciliation for one explicitly admitted successor."""

import gzip
import hashlib
import json
import time
from pathlib import Path

PREDECESSOR_SOURCE = "91224c242c5758fcb25b71c27d0a0b920458ee6b"
PREDECESSORS = {"raw": "979664", "context": "979665"}
INTENTS = {
    "raw": "aa9e9876f21a5bc1c1754c9c7050c30e4a481ce725b5ce30c804f10a775e1e58",
    "context": "810ccba69551b436c3397c4bdcbbf756523e190f7f459b2416e74c1c92d2376a",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def predecessor_guard(value, recon, check_output, roots):
    """Fresh exact terminals plus explicit diagnosed-capacity admission; no allocation."""
    decision_body = value["capacity_decision_body"].encode()
    require(
        sha(decision_body) == value["capacity_decision_sha256"], "Capacity decision bytes differ"
    )
    decision = json.loads(decision_body)
    require(
        decision["status"] == "accepted_original_v3_capacity_failure_reconciled"
        and decision["original_jobs"] == PREDECESSORS
        and decision["verified_failure_site"] == "funding_gate"
        and decision["all_originals_preserved"] is True,
        "Explicit actual capacity-cause/funding decision required",
    )
    accounting = check_output(
        ["sacct", "-X", "-n", "-P", "-j", "979664,979665", "--format=JobIDRaw,State,ExitCode"],
        text=True,
    )
    queue = check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"], text=True)
    chain = {}
    for role, job in PREDECESSORS.items():
        terminal = recon.terminal_receipt(accounting, queue, job_id=job, allowed_states=("FAILED",))
        require(terminal["exit_code"] == "1:0", "Original failure changed")
        root = Path(roots[role])
        require(
            not (root / "actual.json").exists(),
            "Complete predecessor needs collection, not continuation",
        )
        recon.private_directory(root)
        recon.regular(root / "intent.json")
        raw = (root / "intent.json").read_bytes()
        intent = json.loads(raw)
        require(
            sha(raw) == INTENTS[role]
            and intent["CPU_job"] == job
            and intent["source_git"] == PREDECESSOR_SOURCE,
            "Original producer intent changed",
        )
        chain[role] = {
            "job": job,
            "root": str(root),
            "intent_sha256": sha(raw),
            "source_git": PREDECESSOR_SOURCE,
            "actual_terminal": terminal,
        }
    return chain


def raw_inventory(recon, roots, records, prior, *, maximum_buffers=4):
    """Proof owner may differ from NPZ owner; original rows are never rewritten."""
    roots = [Path(root) for root in roots]
    require(
        len(roots) == len(set(roots)) and 1 <= len(roots) <= 3,
        "Finite distinct raw generations required",
    )
    for root in roots:
        recon.private_directory(root)
    mapping = {record["uid"]: i for i, record in enumerate(records)}
    require(len(mapping) == len(records), "Duplicate source roster")
    complete, proof_roots, all_intents, buffers, owners = {}, {}, {}, {}, {}
    partial_verified, partial_source, orphans = {}, {}, {}
    allowed_paths = {
        i: {root / "cache" / f"case-{i:04d}.private.npz" for root in roots}
        for i in range(len(records))
    }
    for row in prior:
        i = mapping[row["uid"]]
        require(
            i not in complete and recon.signature(row["path"]) == row["proof"]["stat"],
            "Initial16 original proof changed",
        )
        complete[i] = {key: row[key] for key in ("uid", "path", "proof")} | {
            "original16_reused": True
        }
    for root in roots:
        intents = {}
        expected = set()
        for i, record in enumerate(records):
            stem = f"case-{i:04d}"
            ip = root / "proofs" / (stem + ".intent.json")
            sp = root / "proofs" / (stem + ".source-proof.private.json.gz")
            vp = root / "proofs" / (stem + ".verified.private.json")
            expected.update((ip.name, sp.name, vp.name))
            if ip.exists():
                intent = recon.bound_json(ip)
                require(
                    intent["source_ordinal"] == i
                    and intent["source_bytes"]
                    == sum(row["bytes"] for row in record["source_files"])
                    and intent["original_source_table_sha256"]
                    == recon.digest(recon.json_bytes(record["source_files"])),
                    "Original source intent changed",
                )
                intents[i] = intent
                all_intents.setdefault(i, []).append({"root": str(root), "intent": intent})
            if not vp.exists():
                continue
            require(i in intents, "Unbound completed source proof")
            try:
                row = recon.bound_json(vp)
            except json.JSONDecodeError:
                partial_verified.setdefault(i, []).append(vp)
                continue
            require(
                row["uid"] == record["uid"]
                and Path(row["path"]) in allowed_paths[i]
                and row["original16_reused"] is False
                and recon.signature(row["path"]) == row["proof"]["stat"],
                "Mixed-generation original cache binding differs",
            )
            recon.regular(sp)
            body = sp.read_bytes()
            require(sha(body) == row["source_proof_sha256"], "Original source proof hash differs")
            recon.source_proof_matches(json.loads(gzip.decompress(body)), record)
            if i in complete:
                require(
                    all(complete[i][key] == row[key] for key in ("uid", "path", "proof")),
                    "Conflicting duplicate complete cache references",
                )
            else:
                complete[i], proof_roots[i] = row, root
        for path in (root / "proofs").iterdir():
            require(path.name in expected, "Unexpected original proof member")
            recon.regular(path)
        for owned in (root / "scratch").iterdir():
            recon.private_directory(owned)
            require(
                owned.name.startswith(("fitting-study-", "fitting-resume-")),
                "Unexpected retained source root",
            )
            data = owned / "data"
            recon.private_directory(data)
            recon.regular(data / "train.csv")
            lines = (data / "train.csv").read_bytes().decode().splitlines()
            require(
                len(lines) == 2 and lines[0] == "StudyInstanceUID" and lines[1] in mapping,
                "Retained buffer source identity is ambiguous",
            )
            i = mapping[lines[1]]
            require(i in intents, "Retained buffer lacks its original source intent")
            recon.regular(data / "train_series.csv")
            require(
                (data / "train_series.csv").read_bytes() == records[i]["series_csv"].encode(),
                "Retained original series CSV bytes changed",
            )
            require(
                all(
                    path.name
                    in {"data", "source-proof.private.json", "source-failure.private.json"}
                    and not path.is_symlink()
                    for path in owned.iterdir()
                ),
                "Unexpected retained buffer member",
            )
            sp = owned / "source-proof.private.json"
            if sp.exists():
                try:
                    proof = recon.bound_json(sp)
                except json.JSONDecodeError:
                    partial_source.setdefault(i, []).append(sp)
                else:
                    recon.source_proof_matches(proof, records[i])
            buffers.setdefault(i, []).append(data)
            owners[str(data)] = root
        cache_names = {f"case-{i:04d}.private.npz": i for i in range(len(records))}
        for path in (root / "cache").iterdir():
            recon.regular(path)
            require(path.name in cache_names, "Unexpected original raw cache file")
            i = cache_names[path.name]
            # A v3 verified row may honestly point at the old intact orphan.
            if i not in complete:
                require(i in all_intents, "Unverified cache lacks original intent")
                orphans.setdefault(i, []).append(path)
    require(
        sum(map(len, buffers.values())) <= maximum_buffers,
        "Retained source buffers exceed explicit bound",
    )
    selected = {}
    for i, options in buffers.items():
        if i in complete:
            continue
        require(
            len(options) == 1,
            "Multiple unfinished buffers for one case require explicit merged-source reconciliation",
        )
        selected[i] = options[0]
    orphans = {i: options for i, options in orphans.items() if i not in complete}
    for i, options in orphans.items():
        require(i in selected, "Unverified original cache lacks retained original pixels")
        require(
            len(options) == 1,
            "Multiple uncertain orphan archives require original-pixel reconciliation",
        )
    require(
        len(set(all_intents) - set(complete)) <= maximum_buffers,
        "Unfinished source intents exceed bounded predecessor lanes",
    )
    return {
        "complete": complete,
        "pending": [i for i in range(len(records)) if i not in complete],
        "proof_roots": proof_roots,
        "intents": all_intents,
        "buffers": selected,
        "all_buffers": buffers,
        "buffer_owners": owners,
        "orphan_caches": {i: paths[0] for i, paths in orphans.items()},
        "partial_verified_records": partial_verified,
        "partial_source_records": partial_source,
    }


def context_inventory(recon, roots, initial, raw_complete, uids, provenance):
    """Reuse complete old/new-renderer rows; report orphan locations without touching them."""
    roots = [Path(root) for root in roots]
    require(
        1 <= len(roots) <= 3 and len(roots) == len(set(roots)),
        "Finite distinct context generations",
    )
    mapping = {uid: i for i, uid in enumerate(uids)}
    complete, partial, orphans = {}, {}, {}
    for row in initial:
        i = mapping[row["uid"]]
        require(
            i not in complete and recon.signature(row["path"]) == row["proof"]["stat"],
            "Initial context proof changed",
        )
        require(
            row["provenance"] == provenance(raw_complete[i], original=True),
            "Initial renderer provenance changed",
        )
        complete[i] = row
    for root in roots:
        recon.private_directory(root)
        for i, uid in enumerate(uids):
            record = root / f"context-{i:04d}.verified.private.json"
            if not record.exists():
                continue
            try:
                row = recon.bound_json(record)
            except json.JSONDecodeError:
                partial.setdefault(i, []).append(record)
                continue
            allowed = {prior / f"context-{i:04d}.private.npz" for prior in roots}
            require(
                i in raw_complete
                and row["uid"] == uid
                and Path(row["path"]) in allowed
                and recon.signature(row["path"]) == row["proof"]["stat"],
                "Original context binding changed",
            )
            require(
                row["provenance"]
                in (provenance(raw_complete[i]), provenance(raw_complete[i], original=True)),
                "Original or corrected renderer lineage required",
            )
            if i in complete:
                require(row == complete[i], "Conflicting duplicate complete context")
            else:
                complete[i] = row
        names = {f"context-{i:04d}.private.npz": i for i in range(len(uids))}
        for path in root.glob("context-*.npz"):
            recon.regular(path)
            require(path.name in names, "Unexpected context archive")
            i = names[path.name]
            if i not in complete:
                orphans.setdefault(i, []).append(path)
    for i, paths in list(orphans.items()):
        if i in complete:
            del orphans[i]
        else:
            require(len(paths) == 1, "Multiple unverified context orphans require explicit review")
    return {
        "complete": complete,
        "orphan_caches": {i: v[0] for i, v in orphans.items()},
        "partial_verified_records": partial,
    }


def inventory_digest(inv, recon):
    # The index stays private. Hash only unchanged proof/row identity, not ctime
    # of source files whose hardlinks are explicitly allowed during continuation.
    body = {
        "complete": {str(i): row for i, row in inv["complete"].items()},
        "orphans": {str(i): str(path) for i, path in inv["orphan_caches"].items()},
    }
    return recon.digest(recon.json_bytes(body))


def admission_receipt(value):
    body = value["mixed_prefix_admission"].encode()
    require(
        sha(body) == value["mixed_prefix_admission_sha256"], "Mixed prefix receipt hash changed"
    )
    receipt = json.loads(body)
    require(
        receipt["status"] == "actual_mixed_v3_prefix_read_only_admitted"
        and receipt["original_jobs"] == PREDECESSORS
        and receipt["no_original_writes"] is True,
        "Actual mixed-prefix admission required",
    )
    require(
        receipt["suffix_reconcile_sha256"] == value["suffix_reconcile_sha256"],
        "Actual admission must run this exact reconciler",
    )
    return receipt


def check_admission(value, inventory, recon):
    receipt = admission_receipt(value)
    require(
        inventory_digest(inventory, recon) == receipt["raw_inventory_sha256"],
        "Retained raw prefix changed after original admission",
    )
    require(
        len(inventory["complete"]) == receipt["raw_complete"]
        and len(inventory["pending"]) == receipt["raw_pending"],
        "Mixed raw count changed",
    )


def check_context_admission(value, inventory, recon):
    receipt = admission_receipt(value)
    require(
        inventory_digest(inventory, recon) == receipt["context_inventory_sha256"],
        "Retained context prefix changed after original admission",
    )
    require(
        len(inventory["complete"]) == receipt["context_complete"], "Mixed context count changed"
    )


def capacity_snapshot(raw_paid, context_paid, free, other, *, scratch_cap):
    require(
        all(type(n) is int and n >= 0 for n in (raw_paid, context_paid, free, other)),
        "Actual integer capacity readings required",
    )
    require(scratch_cap == 6 * 2**30, "Four-lane source scratch requires funded six GiB")
    require(raw_paid < 320 * 2**30 and context_paid < 96 * 2**30, "Original cache bound exhausted")
    require(other >= 62 * 2**30 + 594 * 2**20, "Existing node liabilities must remain funded")
    required = 20 * 2**30 + other + scratch_cap + 320 * 2**30 - raw_paid + 96 * 2**30 - context_paid
    return {
        "owner_event": "capacity_observation",
        "free_bytes": free,
        "required_bytes": required,
        "margin_bytes": free - required,
        "raw_paid_bytes": raw_paid,
        "context_paid_bytes": context_paid,
        "scratch_cap_bytes": scratch_cap,
        "other_node_future_bytes": other,
        "accepted": free > required,
        "no_deletion_credit": True,
    }


def wait_capacity(snapshot, *, emit, wait_seconds, monotonic=time.monotonic, sleep=time.sleep):
    require(
        type(wait_seconds) is int and 0 < wait_seconds <= 1800, "Bounded capacity wait required"
    )
    started = monotonic()
    while True:
        result = snapshot()
        if result["accepted"]:
            return result
        result = {**result, "wait_elapsed_seconds": monotonic() - started}
        emit(result)  # Numeric diagnostics only; no source IDs or private paths.
        if result["wait_elapsed_seconds"] >= wait_seconds:
            raise RuntimeError("Bounded capacity wait expired; preserve every original")
        sleep(min(30, wait_seconds - result["wait_elapsed_seconds"]))
