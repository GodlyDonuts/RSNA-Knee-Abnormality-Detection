"""Bounded CPU inspection/admission; reuse original inputs, runtime and parent."""

import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

GIB, MIB = 2**30, 2**20
PORTABLE = Path("/tmp/rsna-predictive-portable-recovery-20261020-v1")
PILOT = Path("/tmp/rsna-predictive-density100-pilot-20261020-v1")
PRIMARY_SOURCE = "2a224ae05fc7c2fc78d4d508b6407344ac523a1c"
PRIMARY_OWNER_PID = 32377
PRIMARY_AUDIT_REL = "initiative/research/predictive-evc27-deadline-placement-20261020-v1"
CONTINUATION_REL = "artifacts/predictive-evc27-startup-continuation-20261020-v1/continuation-intent.json"
PRIMARY_INTENT_SHA = "b310a8cf6267ea4ce1b40b4efef27ad375b9bc869dfaee435b9c86ed0f40ebd0"
CONTINUATION_SOURCE = "7dcd87e3749998ee7d8f84c08f8f7f118072bf9d"
CONTINUATION_PID = 33107
CONTINUATION_INTENT_SHA = "360bc552517cdc0a886ae36e4efa55100327eebf0a4204b233aa20159c0b7a71"
ENGINEERING_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
# Preserve all old obligations conservatively, including prepared full C100.
# The existing descriptor artifact remains reserved, with no speculative paid credit.
DESCRIPTOR_RESERVED_BYTES = MIB
CONTRAST_RESERVED_BYTES = 7 * GIB + 4 * MIB
MINIMUM_OTHER_BYTES = (205676085248 + 96 * GIB + 36 * GIB + 128 * MIB
                       + 6 * GIB + 4 * MIB + 12 * GIB + 16 * MIB + DESCRIPTOR_RESERVED_BYTES
                       + CONTRAST_RESERVED_BYTES)
PURE_DRIVER_FUNCTIONS = {
    "require", "sha", "digest", "recipe", "once", "pilot_completed",
    "validate_prerequisites", "topology", "paid_log_count", "shared_capacity",
}


def primary_control():
    """Bind the unchanged original capsule and its separately recorded continuation."""
    return {"source_git": PRIMARY_SOURCE, "original_owner_pid": PRIMARY_OWNER_PID,
            "audit_directory": PRIMARY_AUDIT_REL,
            "original_intent_sha256": PRIMARY_INTENT_SHA,
            "continuation_source_git": CONTINUATION_SOURCE,
            "continuation_pid": CONTINUATION_PID,
            "continuation_intent_sha256": CONTINUATION_INTENT_SHA}


def validate_primary_ownership(original_raw, continuation_raw):
    assert hashlib.sha256(original_raw).hexdigest() == PRIMARY_INTENT_SHA, "Original primary intent changed"
    assert hashlib.sha256(continuation_raw).hexdigest() == CONTINUATION_INTENT_SHA, "Actual continuation intent changed"
    owner, resumed = json.loads(original_raw), json.loads(continuation_raw)
    assert owner["source_git"] == PRIMARY_SOURCE and owner["pid"] == PRIMARY_OWNER_PID
    assert resumed["source_git"] == CONTINUATION_SOURCE and resumed["pid"] == CONTINUATION_PID
    assert resumed["original_source_git"] == PRIMARY_SOURCE and resumed["original_owner_pid"] == PRIMARY_OWNER_PID
    assert resumed["original_startup_sha256"]["owner-intent.json"] == PRIMARY_INTENT_SHA
    assert resumed["original_owner_intent_retained"] is True and resumed["only_startup_marker_renamed"] is True
    assert resumed["automatic_retry"] is False
    return primary_control()


def validate_primary_control(value):
    assert value == primary_control(), "Exact current primary shared-log control required"
    return value


def controls(files, hashes):
    """Execute only stdlib control functions; management Python needs no Torch."""
    assert set(files) == set(hashes)
    assert all(hashlib.sha256(body.encode()).hexdigest() == hashes[name]
               for name, body in files.items())
    builder = {"__name__": "frozen_routed_worker_builder"}
    exec(compile(files["worker_builder.py"], "frozen_routed_worker_builder", "exec"), builder)
    tree = ast.parse(files["driver.py"])
    selected = [node for node in tree.body if isinstance(node, ast.Assign)
                or isinstance(node, ast.FunctionDef) and node.name in PURE_DRIVER_FUNCTIONS]
    assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == PURE_DRIVER_FUNCTIONS
    policy = dict(hashlib=hashlib, json=json, math=math, os=os, re=re, Path=Path)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "driver_control_only", "exec"), policy)
    return builder, policy


def validate_other_funding(funding):
    assert funding["other_unpaid_node_liabilities_bytes"] >= MINIMUM_OTHER_BYTES
    assert funding["prior_C0_and_full_C100_scopes_retained"] is True


def stamp(value):
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0
    return parsed


def decision_time(decision, funding, now):
    assert now.tzinfo is not None
    assert 0 <= (now - stamp(funding["observed_at_UTC"])).total_seconds() <= 1800
    assert 0 <= (now - stamp(decision["accepted_at_UTC"])).total_seconds()
    assert now.timestamp() + 5400 <= stamp(decision["finish_by_UTC"]).timestamp()


def signature(path, *, private=True):
    st = path.stat()
    assert path.is_file() and not path.is_symlink()
    if private:
        assert st.st_uid == os.getuid() and not st.st_mode & 0o077
    else:
        assert st.st_mode & 0o111 and not st.st_mode & 0o022
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def private_json(path, expected=None):
    before = signature(path)
    assert before[1] <= 4 * MIB
    raw = path.read_bytes()
    assert signature(path) == before
    if expected is not None:
        assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw), raw


def stage(root, files, binding, decision, funding, result, once):
    """Exclusive new root; never repair or overwrite an uncertain admission."""
    assert not root.exists() and not root.is_symlink()
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in files)
    assert sum(len(s.encode()) for s in files.values()) < 2 * MIB
    root.mkdir(mode=0o700)
    for name, body in files.items():
        path = root / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("x") as stream:
            stream.write(body)
    once(root / "binding.private.json", binding)
    once(root / "engineering-decision.json", decision)
    once(root / "funding.json", funding)
    result["binding_sha256"] = hashlib.sha256((root / "binding.private.json").read_bytes()).hexdigest()
    once(root / "prepared.json", result)
    assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) < 4 * MIB


def main(v):
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and sys.dont_write_bytecode
    assert v["mode"] in {"inspect", "admit"}
    os.umask(0o077)
    primary = validate_primary_control(v["primary_control"])
    b, p = controls(v["files"], v["sha256"])
    root, old, sha = b["ROOT"], b["ORIGINAL"], p["sha"]
    assert re.fullmatch("[0-9a-f]{40}", v["source_git"])
    assert not root.exists() and not root.is_symlink(), "Existing admission is not retry authority"
    decision, funding = (json.loads(v[k]) for k in ("decision_body", "funding_body"))
    assert decision["control_sha256"] == v["control_sha256"]
    decision_time(decision, funding, datetime.now(timezone.utc))
    validate_other_funding(funding)
    p["topology"](subprocess.check_output(["scontrol", "show", "node", "evc27", "--oneliner"], text=True))
    p["pilot_completed"](subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", "981620", "--format=JobIDRaw,State,ExitCode,NodeList"], text=True))
    queue = subprocess.check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i"], text=True)
    assert "981620" not in queue.split()
    pilot, pilot_raw = private_json(PILOT / "GPU-accepted.json", v["pilot_acceptance_sha256"])
    previous, _ = private_json(PILOT / "binding.private.json", p["PILOT_BINDING"])
    binding, old_raw = private_json(old / "binding.private.json", previous["original_binding_sha256"])
    private_json(old / "GPU-accepted.json", ENGINEERING_SHA)
    assert binding["source_git"] == b["SCIENCE"]
    assert v["files"]["worker.py"] == b["build"]((old / "code/worker.py").read_text())
    original_modules = {n: h for n, h in binding["code_sha256"].items() if n.startswith("rsna/")}
    routing = "rsna/competitive/predictive/generative_routing.py"
    assert {n for n in v["files"] if n.startswith("rsna/")} == set(original_modules) | {routing}
    assert v["sha256"][routing] == b["ROUTING_SHA"]
    for name, expected in original_modules.items():
        assert v["sha256"][name] == expected
        assert (old / "code" / name).read_bytes() == v["files"][name].encode()
    for name, expected in b["NUMERICAL_SHA"].items():
        assert original_modules[name] == expected
    for key in ("targets16", "context_index"):
        assert sha(Path(binding[key + "_private_path"]).read_bytes()) == binding[key + "_sha256"]
    index = json.loads(Path(binding["context_index_private_path"]).read_bytes())
    assert len(index) == 16
    for row in index:
        assert signature(Path(row["path"])) == row["proof"]["stat"]
    parent_stat = signature(Path(binding["parent_weight_private_path"]))
    assert {"inode": parent_stat[0], "bytes": parent_stat[1], "mtime_ns": parent_stat[2]} \
        == binding["parent_weight_stat"]
    assert sha((PORTABLE / "assets/parent_source.private.py").read_bytes()) == binding["parent_source_sha256"]
    runtime_files = {}
    for name in ("runtime-import.private.json", "replica-accepted.json"):
        raw = (PORTABLE / name).read_bytes()
        runtime_files[name] = {"sha256": sha(raw), "stat": signature(PORTABLE / name)}
    runtime = json.loads((PORTABLE / "runtime-import.private.json").read_bytes())
    versions = json.loads(runtime["stdout"])
    assert runtime["returncode"] == 0 and versions["torch"] == "2.14.0+cu130"
    assert versions["new_predictive_graph_imported"] is True
    python = (PORTABLE / "runtime/bin/python").resolve()
    st = python.stat()
    runtime_files["resolved_python"] = dict(path=str(python), stat=signature(python, private=False),
                                           owner_uid=st.st_uid, mode=st.st_mode & 0o777)
    assert runtime_files == previous["runtime_files"] and versions == previous["runtime_versions"], (
        "Original accepted runtime/interpreter binding changed"
    )
    predicted_bytes = 4 * signature(old / "engineering/C/full/final.pt")[1] + 64 * MIB
    assert predicted_bytes < 6 * GIB
    binding.update(
        source_git=v["source_git"], original_numerical_source_git=b["SCIENCE"],
        primary_shared_log_control=primary,
        original_binding_sha256=sha(old_raw), original_engineering_receipt_sha256=ENGINEERING_SHA,
        branch=p["BRANCH"], roles=["C100", "G100"], recipe=p["recipe"](),
        code_sha256=v["sha256"], original_module_sha256=original_modules,
        runtime_versions=versions, runtime_files=runtime_files,
        density100_pilot={"path": str(PILOT / "GPU-accepted.json"), "sha256": sha(pilot_raw)},
        engineering_decision={"path": str(root / "engineering-decision.json"),
                              "sha256": sha((json.dumps(decision, sort_keys=True, allow_nan=False) + "\n").encode())},
        funding={"path": str(root / "funding.json"),
                 "sha256": sha((json.dumps(funding, sort_keys=True, allow_nan=False) + "\n").encode())},
    )
    p["validate_prerequisites"](binding, pilot, decision, funding)
    paid = p["paid_log_count"](funding.get("original_dispatched_jobs", {}))
    quota = p["shared_capacity"](subprocess.check_output(
        ["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True), funding, paid)
    free = shutil.disk_usage("/tmp").free
    required = 20 * GIB + funding["other_unpaid_node_liabilities_bytes"] + 6 * GIB + 4 * MIB
    assert free > required
    now = datetime.now(timezone.utc)
    decision_time(decision, funding, now)
    out = dict(status="actual_matched_generative_routing_CPU_admitted" if v["mode"] == "admit"
               else "actual_matched_generative_routing_readonly_preflight_passed",
               CPU_job=os.environ["SLURM_JOB_ID"], source_git=v["source_git"], observed_at_UTC=now.isoformat(),
               density100_pilot_acceptance_sha256=sha(pilot_raw),
               primary_shared_log_control=primary,
               context_files_reused_without_copy=16, training_studies=12, engineering_studies=4,
               node_capacity=dict(free_bytes=free, required_bytes=required, remaining_bytes=free-required),
               shared_capacity=quota, four_checkpoint_size_bound_bytes=predicted_bytes,
               new_shared_files=0, new_source_downloads=0, GPU_dispatched=False)
    if v["mode"] == "admit":
        pre = v["preflight"]
        assert pre["status"] == "actual_matched_generative_routing_readonly_preflight_passed"
        assert pre["source_git"] == v["source_git"]
        assert 0 <= (now - stamp(pre["observed_at_UTC"])).total_seconds() <= 600
        stage(root, v["files"], binding, decision, funding, out, p["once"])
    print(json.dumps(out, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
