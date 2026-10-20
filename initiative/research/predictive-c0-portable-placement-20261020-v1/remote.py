"""Distinct placement operations; parent supplies frozen source through stdin."""

import base64
import copy
import gzip
import json
import os
import runpy
import secrets
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import placement as p


def namespace(v, node):
    source = p.derived_remote(v["original_remote"], node, v["shared_bytes"], v["shared_files"])
    paid = p.paid_shared_logs(v.get("all_dispatched_jobs", {}))
    source = source.replace(
        f'"shared_unpaid_files": {v["shared_files"]}',
        f'"shared_unpaid_files": {v["shared_files"] - paid["files"]}',
    )
    ns = {"__name__": "placement_original_owner_control"}
    exec(compile(source, "frozen_44cc_owner_with_explicit_placement", "exec"), ns)
    if node == "evc33":
        original = ns["base_other"]
        # Retain the unused original33 C0 scope for its final export return.
        ns["base_other"] = lambda value: original(value) + p.FIT_CAP
    else:

        def other(value):
            paid = sum(
                file.stat().st_size
                for file in p.ROOT27.rglob("*")
                if file.is_file() and "contexts" not in file.relative_to(p.ROOT27).parts
            )
            p.require(paid < p.CONTROL_CAP, "Portable control scope exceeded")
            return (
                p.OTHER27
                + p.CONTEXT_CAP
                - value["portable"]["context_bytes"]
                + p.CONTROL_CAP
                - paid
            )

        ns["base_other"] = other
    return ns


def source_capsule(v):
    ns = namespace(v, "evc33")
    ns["SCIENCE"] = p.SCIENCE
    ns["preparation"](v)
    root = p.ROOT33
    admitted = json.loads((root / "full-admission/actual.json").read_text())
    p.require(admitted["arms"] == ["B", "C"], "Original C0 admission must never exist")
    p.require(not (root / "full-admission/binding-C0.private.json").exists(), "C0 already admitted")
    p.require(
        not (root / "dispatch/C0-intent.json").exists(), "Original C0 dispatch already exists"
    )
    binding_path = root / "full-admission/binding-B.private.json"
    p.require(
        p.file_sha(binding_path) == admitted["binding_sha256"]["B"], "Source admission changed"
    )
    binding = json.loads(binding_path.read_text())
    rows = json.loads(p.read_ref(binding["context_index"]))
    terminal = json.loads(p.read_ref(binding["context_terminal"]))
    total = p.check_rows(rows, terminal)
    # Each existing producer proof is reused by stat; destination independently hashes
    # while receiving. No second source-side bulk read/hash is needed.
    for row in rows:
        p.require(p.signature(row["path"]) == row["proof"]["stat"], "Source context changed")
    small = {}
    for name in (
        "alias_regression",
        "original_preparation",
        "license",
        "targets",
        "context_terminal",
        "context_index",
        "parent_source",
    ):
        small[name + ".private"] = base64.b64encode(p.read_ref(binding[name])).decode()
    for name, digest in binding["code_sha256"].items():
        path = root / "code" / name
        raw = path.read_bytes()
        p.require(p.sha(raw) == digest, "Frozen full numerical source changed")
        small["code/" + name] = base64.b64encode(raw).decode()
    master = (root / "engineering-decision.json").read_bytes()
    p.require(
        p.sha(master) == v["prepared"]["engineering_decision_sha256"], "Master decision changed"
    )
    small["master.json"] = base64.b64encode(master).decode()
    p.require(
        sum(len(base64.b64decode(x)) for x in small.values()) < 24 * p.MIB,
        "Small input/source capsule exceeds controls allowance",
    )
    return {
        "binding": binding,
        "prepared": v["prepared"],
        "files": small,
        "context_bytes": total,
        "source_index_sha256": binding["context_index"]["sha256"],
        "handoff_sha256": v["handoff_sha256"],
        "placement_source_git": v["placement_source_git"],
        "original_B_binding_sha256": p.file_sha(binding_path),
    }


def sender(v):
    p.require(os.uname().nodename == "evc33", "Source owning node required")
    capsule = source_capsule(v)
    endpoint = v["endpoint"]
    p.require(endpoint["host"] == "evc27" and len(endpoint["token"]) == 64, "Exact receiver")
    with socket.create_connection(("evc27", endpoint["port"]), timeout=60) as connection:
        connection.settimeout(600)
        outgoing, incoming = connection.makefile("wb"), connection.makefile("rb")
        outgoing.write(endpoint["token"].encode() + b"\n")
        p.write_json_frame(outgoing, capsule)
        missing = p.read_json_frame(incoming)
        rows = json.loads(base64.b64decode(capsule["files"]["context_index.private"]))
        p.send_contexts(outgoing, rows, missing)
        connection.shutdown(socket.SHUT_WR)
        accepted = p.read_json_frame(incoming)
        p.require(
            accepted["status"] == "actual_C0_context_transfer_and_CPU_admission_accepted",
            "Destination admission did not complete",
        )
    return accepted


def write_capsule(root, capsule):
    for name, encoded in capsule["files"].items():
        path = root / name
        p.require(
            not Path(name).is_absolute() and ".." not in Path(name).parts, "Capsule path escaped"
        )
        raw = base64.b64decode(encoded, validate=True)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)


def portable_admit(v, capsule, rows, transferred):
    root = p.ROOT27
    source = capsule["binding"]
    prepared = capsule["prepared"]
    p.require(
        prepared == v["prepared"]
        and source["source_git"] == p.SCIENCE
        and source["expected_initial_shared_tensor_sha256"] == p.INITIAL,
        "Unchanged actual scientific preparation required",
    )
    for name, digest in source["code_sha256"].items():
        p.require(p.file_sha(root / "code" / name) == digest, "Copied numerical source differs")
    old = json.loads((p.OLD27 / "binding.private.json").read_text())
    accepted_replica = json.loads((p.OLD27 / "replica-accepted.json").read_text())
    p.require(
        accepted_replica["node"] == "evc27"
        and accepted_replica["CPU_job"] == "978186"
        and accepted_replica["files"] == 55685
        and accepted_replica["bytes"] == 12434309116
        and accepted_replica["payload_sha256"]
        == "d5a56a8a48bce368e4380a7ed197bb01330c6698c1dc7544109306737bd8f183",
        "Exact accepted original27 runtime/assets replica required",
    )
    with gzip.open(p.OLD27 / "original-transfer-proof.private.json.gz", "rt") as handle:
        replica = json.load(handle)
    parent = Path(old["parent_weight_private_path"])
    stats = p.signature(parent)
    p.require(
        {"bytes": stats[1], "mtime_ns": stats[2], "inode": stats[0]} == old["parent_weight_stat"]
        and replica["footer"]["files"]["assets/parent.pt"] == source["parent_weight"]["sha256"],
        "Existing independently verified27 parent changed",
    )
    refs = {
        name: p.reference(root / (name + ".private"))
        for name in (
            "parent_source",
            "license",
            "targets",
            "alias_regression",
            "original_preparation",
            "context_terminal",
        )
    }
    for name, ref in refs.items():
        p.require(ref["sha256"] == source[name]["sha256"], "Copied original reference changed")
    refs["parent_weight"] = {
        "path": str(parent),
        "sha256": source["parent_weight"]["sha256"],
        "stat": stats,
    }
    p.require(refs["targets"]["sha256"] == p.TARGETS, "Original target table required")
    p.once(root / "context-index.private.json", rows)
    refs["context_index"] = p.reference(root / "context-index.private.json")
    original = runpy.run_path(str(root / "code/admit_cpu.py"), run_name="original_initialization")
    # Exact original constructor/seed/schema function; only the admitted import path
    # points to the already verified27 runtime replica.
    original["initialize"].__globals__["OVERLAY"] = p.OLD27 / "overlay"
    models, versions = original["initialize"](root, refs)
    p.require(
        models == prepared["models"] and versions == prepared["runtime_versions"],
        "Destination CPU initialization or runtime differs",
    )
    sys.path[:0] = [str(root / "code"), str(p.OLD27 / "overlay")]
    import numpy as np
    from alias_provenance import renderer_counts, validate_terminal

    with np.load(refs["targets"]["path"], allow_pickle=False) as arrays:
        uids = arrays["uids"].tolist()
        p.require(
            uids == [row["uid"] for row in rows] and p.sha(p.canonical(uids)) == p.ROSTER,
            "Exact original labels/context join changed",
        )
        p.require(arrays["labels"].shape == arrays["masks"].shape == (3887, 12), "Target shape")
    accepted = source["accepted_renderer_sha256"]
    validate_terminal(
        json.loads(p.read_ref(refs["context_terminal"])),
        accepted,
        refs["alias_regression"]["sha256"],
    )
    counts = renderer_counts(rows, accepted)
    p.require(counts == source["renderer_provenance_counts"], "Renderer lineage changed")
    lineage = {
        "status": "actual_C0_portable_CPU_runtime_initialization_accepted",
        "node": "evc27",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "scientific_source_git": p.SCIENCE,
        "placement_control_source_git": v["placement_source_git"],
        "placement_control_sha256": v["placement_code_sha256"],
        "handoff_sha256": v["handoff_sha256"],
        "original_prepared_receipt": prepared,
        "runtime_versions": versions,
        "models": models,
        "original_27_replica_receipt_sha256": p.file_sha(p.OLD27 / "replica-accepted.json"),
        "initialization_function_sha256": source["code_sha256"]["admit_cpu.py"],
        "numerical_worker_and_modules_unchanged": True,
    }
    p.once(root / "runtime-receipt.json", lineage)
    refs["runtime_receipt"] = p.reference(root / "runtime-receipt.json")
    master = json.loads((root / "master.json").read_text())
    decision = {
        **master,
        "status": "accepted_full_predictive_fitting_recipe",
        "source_git": p.SCIENCE,
        "code_sha256": source["code_sha256"],
        "arm": "C0",
        "recipe": master["recipes"]["C0"],
        "master_decision_sha256": p.file_sha(root / "master.json"),
        "placement_control_source_git": v["placement_source_git"],
    }
    p.require(
        decision["recipe"]["density_weight"] == 0
        and decision["recipe"]["epochs"] == 6
        and decision["recipe"]["schedule_updates"] == 2916,
        "Fixed C0 recipe required",
    )
    p.once(root / "engineering-decision.json", master)
    out = root / "full-admission"
    out.mkdir(mode=0o700)
    p.once(out / "engineering-C0.json", decision)
    refs["engineering_decision"] = p.reference(out / "engineering-C0.json")
    controls_paid = sum(
        f.stat().st_size
        for f in root.rglob("*")
        if f.is_file() and "contexts" not in f.relative_to(root).parts
    )
    p.require(controls_paid < p.CONTROL_CAP - 4 * p.MIB, "Bounded portable controls required")
    fresh = p.capacity(
        shutil.disk_usage("/tmp").free,
        contexts_paid=capsule["context_bytes"],
        controls_paid=controls_paid,
    )
    shared = fresh_shared(v)
    run_root = "/tmp/rsna-predictive-full-C0-20261020-v2"
    p.require(not Path(run_root).exists(), "Existing C0 output requires reconciliation")
    fund = {
        "status": "actual_predictive_full_arm_storage_funded",
        "arm": "C0",
        "node": "evc27",
        "run_root": run_root,
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "arm_cap_bytes": p.FIT_CAP,
        "standing_free_reserve_bytes": 20 * p.GIB,
        "actual_node_free_bytes": fresh["free_bytes"],
        "other_unpaid_node_liabilities_bytes": p.OTHER27
        + p.CONTEXT_CAP
        - capsule["context_bytes"]
        + p.CONTROL_CAP
        - controls_paid,
        "shared_free_after_all_liabilities_bytes": shared["free_bytes"] - v["shared_bytes"],
        "shared_files_after_all_liabilities": shared["free_files"] - shared["unpaid_shared_files"],
        "all_other_obligations_retained": True,
        "no_deletion_credit": True,
        "placement_control_source_git": v["placement_source_git"],
        "handoff_sha256": v["handoff_sha256"],
    }
    p.once(out / "funding-C0.json", fund)
    refs["funding"] = p.reference(out / "funding-C0.json")
    admission = {
        "status": "actual_complete3887_predictive_fitting_inputs_accepted",
        "node": "evc27",
        "physical_gpu_count": 2,
        "source_git": p.SCIENCE,
        "code_sha256": source["code_sha256"],
        "expected_initial_shared_tensor_sha256": p.INITIAL,
        **{
            name + "_sha256": refs[name]["sha256"]
            for name in (
                "context_index",
                "targets",
                "parent_source",
                "parent_weight",
                "runtime_receipt",
            )
        },
        "all_context_original_hashes_and_stats_verified": True,
        "original_context_bytes": capsule["context_bytes"],
        "original_context_count": 3887,
        "accepted_renderer_sha256": accepted,
        "renderer_provenance_counts": counts,
        "alias_regression_receipt_sha256": refs["alias_regression"]["sha256"],
        "fitting_order_sha256": p.ROSTER,
        "placement_control_source_git": v["placement_source_git"],
        "placement_control_sha256": v["placement_code_sha256"],
        "handoff_sha256": v["handoff_sha256"],
        "source_B_binding_sha256": capsule["original_B_binding_sha256"],
        "source_context_index_sha256": capsule["source_index_sha256"],
        "transfer_only_missing_context_bytes": transferred,
    }
    p.once(out / "input-C0.json", admission)
    refs["input_admission"] = p.reference(out / "input-C0.json")
    binding = {
        **copy.deepcopy(source),
        **refs,
        "arm": "C0",
        "node": "evc27",
        "code_root": str(root / "code"),
        "runtime_python": p.PY27,
        "python_paths": [str(p.OLD27 / "overlay")],
        "runtime_versions": versions,
        "run_root": run_root,
        "placement_control_source_git": v["placement_source_git"],
        "placement_control_sha256": v["placement_code_sha256"],
        "placement_handoff_sha256": v["handoff_sha256"],
    }
    p.once(out / "binding-C0.private.json", binding)
    result = {
        "status": "actual_C0_context_transfer_and_CPU_admission_accepted",
        "node": "evc27",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "science_git": p.SCIENCE,
        "placement_control_source_git": v["placement_source_git"],
        "handoff_sha256": v["handoff_sha256"],
        "context_bytes": capsule["context_bytes"],
        "transferred_bytes": transferred,
        "source_context_index_sha256": capsule["source_index_sha256"],
        "destination_context_index_sha256": refs["context_index"]["sha256"],
        "initial_shared_tensor_sha256": p.INITIAL,
        "new_GPU_jobs": 0,
        "binding_sha256": {"C0": p.file_sha(out / "binding-C0.private.json")},
    }
    p.once(root / "portable-accepted.json", result)
    return result


def fresh_shared(v):
    text = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    row = next(
        line.split() for line in text.splitlines() if line.strip().startswith("/home/sa305415")
    )
    paid = p.paid_shared_logs(v.get("all_dispatched_jobs", {}))
    value = {
        "paid_original_GPU_logs": paid,
        "unpaid_shared_files": v["shared_files"] - paid["files"],
        "free_bytes": (int(row[2]) - int(row[1])) * 1024,
        "free_files": int(row[6]) - int(row[5]),
    }
    p.require(
        value["free_bytes"] > 20 * p.GIB + v["shared_bytes"]
        and value["free_files"] > 5000 + value["unpaid_shared_files"],
        "Fresh shared reserve required",
    )
    return value


def receiver(v):
    p.require(
        os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS"),
        "Owning27 CPU-only receiver required",
    )
    p.require(
        Path(sys.executable).resolve() == Path(p.PY27).resolve(), "Original27 runtime required"
    )
    p.handoff(v["handoff"])
    p.require(
        p.sha(v["handoff_text"].encode()) == v["handoff_sha256"]
        and json.loads(v["handoff_text"]) == v["handoff"],
        "Exact original handoff file bytes required",
    )
    p.capacity(shutil.disk_usage("/tmp").free)
    fresh_shared(v)
    root = p.ROOT27
    p.require(not root.exists(), "Existing transfer requires read-only reconciliation")
    root.mkdir(mode=0o700)
    (root / "contexts").mkdir(mode=0o700)
    p.once(
        root / "transfer-intent.json",
        {
            "CPU_job": os.environ["SLURM_JOB_ID"],
            "placement_source_git": v["placement_source_git"],
            "handoff_sha256": v["handoff_sha256"],
            "automatic_retry": False,
            "new_GPU_jobs": 0,
            "context_cap_bytes": p.CONTEXT_CAP,
        },
    )
    listener = socket.socket()
    listener.bind(("0.0.0.0", 0))
    listener.listen(1)
    listener.settimeout(1800)
    token = secrets.token_hex(32)
    print(
        json.dumps({"host": "evc27", "port": listener.getsockname()[1], "token": token}), flush=True
    )
    connection, _ = listener.accept()
    listener.close()
    with connection:
        connection.settimeout(600)
        incoming, outgoing = connection.makefile("rb"), connection.makefile("wb")
        p.require(incoming.readline(80) == token.encode() + b"\n", "Private endpoint token differs")
        capsule = p.read_json_frame(incoming)
        p.require(
            capsule["handoff_sha256"] == v["handoff_sha256"]
            and capsule["placement_source_git"] == v["placement_source_git"],
            "Wrong source owner",
        )
        raw_index = base64.b64decode(capsule["files"]["context_index.private"], validate=True)
        rows = json.loads(raw_index)
        terminal = json.loads(base64.b64decode(capsule["files"]["context_terminal.private"]))
        p.require(
            p.sha(raw_index) == capsule["source_index_sha256"]
            and p.check_rows(rows, terminal) == capsule["context_bytes"],
            "Source manifest changed",
        )
        write_capsule(root, capsule)
        missing = p.missing_contexts(root, rows)
        p.write_json_frame(outgoing, missing)
        rebound, transferred = p.receive_contexts(incoming, root, rows, missing)
        result = portable_admit(v, capsule, rebound, transferred)
        p.write_json_frame(outgoing, result)
    return result


def return_sender(v):
    p.require(os.uname().nodename == "evc27", "Actual C0 owning node required")
    binding = json.loads((p.ROOT27 / "full-admission/binding-C0.private.json").read_text())
    run = Path(binding["run_root"])
    terminal = json.loads((run / "terminal.json").read_text())
    p.require(
        terminal["updates"] == 2916
        and terminal["epochs"] == 6
        and terminal["source_git"] == p.SCIENCE
        and terminal["arm"] == "C0",
        "Fixed-final C0 only",
    )
    state = namespace(v, "evc27")["accounting"]([terminal["GPU_job"]])[terminal["GPU_job"]]
    p.require(
        state["state"] == "COMPLETED"
        and state["exit_code"] == "0:0"
        and state["queue_empty"]
        and state["node"] == "evc27"
        and terminal["binding_sha256"]
        == p.file_sha(p.ROOT27 / "full-admission/binding-C0.private.json"),
        "Actual original C0 terminal and exact binding required before return",
    )
    receipts = [
        json.loads((run / f"publication/epoch-{e:03d}/verified.json").read_text())
        for e in range(1, 7)
    ]
    p.require(
        all(
            r["status"] == v["verified_status"]
            and r["objects"] == 5
            and r["source_git"] == p.SCIENCE
            and r["arm"] == "C0"
            and r["epoch"] == e
            for e, r in enumerate(receipts, 1)
        ),
        "All six original C0 epochs must be privately verified",
    )
    paths = {
        "terminal.json": run / "terminal.json",
        "publication/epoch-006/verified.json": run / "publication/epoch-006/verified.json",
    }
    for name in receipts[-1]["files"]:
        paths["publication/epoch-006/" + name] = (
            Path(binding["license"]["path"])
            if name == "source-permission.txt"
            else run / "publication/epoch-006" / name
        )
    files = {}
    for name, path in paths.items():
        st = p.signature(path)
        expected = (
            receipts[-1]["files"][Path(name).name]
            if Path(name).name in receipts[-1]["files"]
            else {"bytes": st[1], "sha256": p.file_sha(path)}
        )
        p.require(st[1] == expected["bytes"], "Final source object size changed")
        files[name] = {**expected, "stat": st}
    p.require(sum(r["bytes"] for r in files.values()) < 512 * p.MIB, "Bounded final return")
    endpoint = v["endpoint"]
    p.require(endpoint["host"] == "evc33", "Only owning33 final mirror")
    with socket.create_connection(("evc33", endpoint["port"]), timeout=60) as connection:
        connection.settimeout(600)
        out, inc = connection.makefile("wb"), connection.makefile("rb")
        out.write(endpoint["token"].encode() + b"\n")
        p.write_json_frame(
            out,
            {
                "files": files,
                "terminal": terminal,
                "publications": receipts,
                "handoff_sha256": v["handoff_sha256"],
                "binding_sha256": p.file_sha(p.ROOT27 / "full-admission/binding-C0.private.json"),
            },
        )
        for name, expected in files.items():
            path = paths[name]
            p.require(p.signature(path) == expected["stat"], "Source export stat changed")
            with path.open("rb") as source:
                shutil.copyfileobj(source, out, p.MIB)
            p.require(p.signature(path) == expected["stat"], "Source export changed in transit")
        out.write(p.EOF)
        out.flush()
        connection.shutdown(socket.SHUT_WR)
        return p.read_json_frame(inc)


def return_receiver(v):
    p.require(
        os.uname().nodename == "evc33" and not os.environ.get("SLURM_JOB_GPUS"),
        "Return uses owning33 CPU only",
    )
    root = p.RETURN33
    p.require(not root.exists(), "Existing final return must be reconciled")
    ns = namespace(v, "evc33")
    # Original unused33 C0 12GiB is retained by base_other, containing the <=512MiB
    # mirror. This reuses an unspent obligation and grants no deletion credit.
    minimum = (
        20 * p.GIB
        + ns["base_other"](v)
        + sum(max(0, p.FIT_CAP - ns["used_bytes"](ns["run_root"](arm))) for arm in ("B", "C"))
    )
    p.require(shutil.disk_usage("/tmp").free > minimum, "Fresh33 return capacity required")
    fresh_shared(v)
    root.mkdir(mode=0o700)
    p.once(
        root / "return-intent.json",
        {
            "handoff_sha256": v["handoff_sha256"],
            "CPU_job": os.environ["SLURM_JOB_ID"],
            "automatic_retry": False,
            "source_node": "evc27",
            "cap_bytes": 512 * p.MIB,
        },
    )
    listener = socket.socket()
    listener.bind(("0.0.0.0", 0))
    listener.listen(1)
    listener.settimeout(1800)
    token = secrets.token_hex(32)
    print(
        json.dumps({"host": "evc33", "port": listener.getsockname()[1], "token": token}), flush=True
    )
    connection, _ = listener.accept()
    listener.close()
    with connection:
        connection.settimeout(600)
        inc, out = connection.makefile("rb"), connection.makefile("wb")
        p.require(inc.readline(80) == token.encode() + b"\n", "Return endpoint token")
        header = p.read_json_frame(inc)
        p.require(header["handoff_sha256"] == v["handoff_sha256"], "Wrong return owner")
        names = {"terminal.json", "publication/epoch-006/verified.json"} | {
            "publication/epoch-006/" + name
            for name in (
                "model.pt",
                "provenance.json",
                "own-source.json",
                "manifest.json",
                "source-permission.txt",
            )
        }
        p.require(
            set(header["files"]) == names
            and sum(r["bytes"] for r in header["files"].values()) < 512 * p.MIB,
            "Exact seven final-return objects required",
        )
        p.require(
            all(
                type(r["bytes"]) is int and r["bytes"] > 0 and len(r["sha256"]) == 64
                for r in header["files"].values()
            ),
            "Positive exact original return-object lengths required",
        )
        for name, expected in header["files"].items():
            path = root / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            h, left = __import__("hashlib").sha256(), expected["bytes"]
            with path.open("xb") as target:
                while left:
                    block = p.frame_read(inc, min(left, p.MIB))
                    target.write(block)
                    h.update(block)
                    left -= len(block)
                target.flush()
                os.fsync(target.fileno())
            p.require(h.hexdigest() == expected["sha256"], "Final return SHA mismatch")
        p.require(
            p.frame_read(inc, len(p.EOF)) == p.EOF and inc.read(1) == b"", "Return EOF mismatch"
        )
        p.require(
            json.loads((root / "terminal.json").read_text()) == header["terminal"],
            "Original terminal body changed",
        )
        p.require(
            json.loads((root / "publication/epoch-006/verified.json").read_text())
            == header["publications"][-1],
            "Original publication proof changed",
        )
        result = {
            "status": "actual_C0_final_export_returned_to33_all_bytes_SHA_verified",
            "CPU_job": os.environ["SLURM_JOB_ID"],
            "source_node": "evc27",
            "destination_node": "evc33",
            "original_training_GPU_job": header["terminal"]["GPU_job"],
            "science_git": p.SCIENCE,
            "handoff_sha256": v["handoff_sha256"],
            "binding_sha256": header["binding_sha256"],
            "objects": {
                name: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for name, row in header["files"].items()
            },
            "original_six_private_revisions": [r["revision"] for r in header["publications"]],
            "mirror_run_root": str(root),
            "originals_deleted": False,
            "new_publication": False,
        }
        p.once(root / "return-verified.json", result)
        p.write_json_frame(out, result)
    return result


def main(v):
    os.umask(0o077)
    p.handoff(v["handoff"])
    p.require(
        p.sha(v["handoff_text"].encode()) == v["handoff_sha256"]
        and json.loads(v["handoff_text"]) == v["handoff"],
        "Exact original handoff file bytes required",
    )
    mode = v["mode"]
    if mode in ("receive", "send", "return_receive", "return_send"):
        return {
            "receive": receiver,
            "send": sender,
            "return_receive": return_receiver,
            "return_send": return_sender,
        }[mode](v)
    node = v["node"]
    ns = namespace(v, node)
    ns["SCIENCE"] = p.SCIENCE
    if node == "evc27":

        def preparation(value):
            actual = json.loads((p.ROOT27 / "portable-accepted.json").read_text())
            p.require(
                actual == value["portable"] and actual["handoff_sha256"] == value["handoff_sha256"],
                "Exact original portable admission required",
            )
            p.require(
                os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS"),
                "Owning27 CPU operation required",
            )
            p.require(
                Path(sys.executable).resolve() == Path(p.PY27).resolve(),
                "Actual accepted27 runtime required",
            )
            binding_path = p.ROOT27 / "full-admission/binding-C0.private.json"
            p.require(
                p.file_sha(binding_path) == actual["binding_sha256"]["C0"],
                "Portable C0 binding changed",
            )
            binding = json.loads(binding_path.read_text())
            for name, digest in binding["code_sha256"].items():
                p.require(
                    p.file_sha(p.ROOT27 / "code" / name) == digest,
                    "Exact original worker/modules changed",
                )
            return actual

        ns["preparation"] = preparation
        if mode == "dispatch":
            # Reuse original one-arm exact-once submission function with its required
            # admission envelope; the sole portable owner writes it once.
            envelope = {
                **v["portable"],
                "arms": ["C0"],
                "status": "actual_full3887_predictive_CPU_admission_complete_no_GPU_dispatched",
            }
            p.once(p.ROOT27 / "full-admission/actual.json", envelope)
            v["admitted"] = envelope
    return ns["main"](v)
