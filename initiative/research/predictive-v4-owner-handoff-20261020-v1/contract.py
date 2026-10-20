"""Bound operational successor; unchanged scientific modules and checkpoint schema."""
import hashlib
import json
import re
from pathlib import Path

PARENT = "926b657770564c79b23509592e2d6a95761abd70"
SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
OLD = "initiative/research/predictive-c0-portable-placement-20261020-v1"
FIT = "initiative/research/predictive-full-fitting-20261020-v2"
CACHE = "initiative/research/predictive-cache-capacity-recovery-20261020-v1"
ADMIT_SHA = "2ed7b4c7ffd8db0469a0bd35bfe4cc1e6e6beaf6d34db33d94ecfd22c536b17c"
REMOTE_SHA = "dc68d3d32c9cdc98ef06e305bc53216286d22760c20171e3276983c3d69df71d"
OLD_SOURCE = "91224c242c5758fcb25b71c27d0a0b920458ee6b"
ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-capacity-recovery-20261020-v1" for r in ("raw", "context")}
OLD_ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-alias-recovery-20261020-v3" for r in ROOTS}
PREDECESSORS = {"raw": "979664", "context": "979665"}
INTENTS = {
    "raw": "aa9e9876f21a5bc1c1754c9c7050c30e4a481ce725b5ce30c804f10a775e1e58",
    "context": "810ccba69551b436c3397c4bdcbbf756523e190f7f459b2416e74c1c92d2376a",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def replace_once(source, old, new):
    require(source.count(old) == 1, "Frozen operational source anchor changed")
    return source.replace(old, new)


def handoff(value, role):
    pid = {"fitting": 80995, "deployment": 80996}[role]
    require(
        value["status"] == "actual_idle_v3_cache_dependent_owner_stopped_before_phase"
        and value["role"] == role and value["pid"] == pid
        and value["source_git"] == PARENT
        and value["phase_intents"] == [] and value["children"] == []
        and value["owner_process_absent_after_stop"] is True
        and value["remote_admission_or_dispatch_exists"] is False
        and value["remote_jobs_cancelled"] == []
        and re.fullmatch("[0-9a-f]{64}", value["remote_read_only_snapshot_sha256"]),
        "Actual exclusive idle-owner handoff required",
    )
    return value


def chain(value):
    require(set(value) == set(PREDECESSORS), "Both exact predecessors required")
    for role, expected in PREDECESSORS.items():
        item = value[role]
        terminal = item["actual_terminal"]
        require(
            str(item["job"]) == expected and item["root"] == OLD_ROOTS[role]
            and item["source_git"] == OLD_SOURCE
            and item["intent_sha256"] == INTENTS[role]
            and str(terminal["job_id"]) == expected
            and terminal["state"] == "FAILED" and terminal["exit_code"] == "1:0"
            and terminal["live_queue_empty"] is True
            and re.fullmatch("[0-9a-f]{64}", terminal["sha256"]),
            "Exact original capacity-failure lineage required",
        )
    return sha(canonical(value))


def derived_admission(source):
    require(sha(source.encode()) == ADMIT_SHA, "Original prepared CPU helper changed")
    for role in ROOTS:
        require(OLD_ROOTS[role] in source, "Missing exact original cache path")
        source = source.replace(OLD_ROOTS[role], ROOTS[role])
    source = replace_once(source, '    rows = json.loads(index_path.read_text())',
        '    _cache_successor = validate_successor_chain(terminal, intent, raw, terminal_job)\n'
        '    rows = json.loads(index_path.read_text())')
    source = replace_once(source, '        write_once(out / f"input-{arm}.json", admission)',
        '        admission["cache_successor_control"] = _cache_successor\n'
        '        write_once(out / f"input-{arm}.json", admission)')
    source = replace_once(source, '        write_once(out / f"binding-{arm}.private.json", binding)',
        '        binding["cache_successor_control"] = _cache_successor\n'
        '        write_once(out / f"binding-{arm}.private.json", binding)')
    compile(source, "explicit_v4_operational_admission", "exec")
    return source


def derived_remote(source):
    require(sha(source.encode()) == REMOTE_SHA, "Original frozen owner operations changed")
    for role in ROOTS:
        require(OLD_ROOTS[role] in source, "Missing exact original cache path")
        source = source.replace(OLD_ROOTS[role], ROOTS[role])
    # Original base_other includes4GiB scratch; successor funds6GiB total.
    source = replace_once(source, 'return 30 * GIB + 594 * MIB + remaining_cache(v)',
                          'return 32 * GIB + 594 * MIB + remaining_cache(v)')
    return source


def derived_placement(source):
    start = source.index('def handoff(receipt):')
    end = source.index('\n\ndef capacity(', start)
    return source[:start] + 'def handoff(receipt):\n    successor_handoff(receipt, "fitting")\n' + source[end:]


def derived_deployment_owner(source):
    """Change only the receipt location used to account for already-paid caches."""
    return replace_once(
        source,
        '        cache = ROOT / "initiative/research/predictive-full-cache-alias-recovery-20261020-v3"',
        f'        cache = ROOT / "{CACHE}"',
    )


def deployment_funding(value):
    base = value["funding"]["other_node_base_bytes"]
    require(type(base) is int and base >= 68 * 2**30 + 594 * 2**20,
            "Deployment must retain the complete six-GiB source scratch scope")


def local_handoff(value, role, repo, process_command):
    handoff(value, role)
    require(not process_command(value["pid"]).strip(), "Predecessor PID still exists")
    suffix = "finite-owner" if role == "fitting" else None
    journal = repo / "artifacts" / (Path(OLD).name + ("-deployment" if suffix is None else ""))
    if suffix:
        journal /= suffix
    allowed = {"owner-intent.json", "launch-intent.json", "launcher.json", "stdout", "stderr",
               "frozen_phase_launcher.py", "frozen-phase-source.json"}
    require(journal.is_dir() and {p.name for p in journal.iterdir()} <= allowed,
            "Old owner has phase evidence; reconcile instead of overriding")
