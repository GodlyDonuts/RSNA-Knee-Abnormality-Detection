"""Operational placement only: one verified context copy, three original fits on27.

Installed after the frozen HTTP499 adapter. No worker, optimizer, model, target,
renderer or publisher source is changed. Importing this module performs no work.
"""

import ast
import copy
import json
import os
import re
import sys
from pathlib import Path

GIB, MIB = 2**30, 2**20
ARMS = ("C", "B", "C0")
SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
PRIOR_OWNER = "291283545c930b730bb257abb10c2a0814b055eb"
ROOT27 = Path("/tmp/rsna-predictive-evc27-deadline-placement-20261020-v1")
ORIGINAL_OTHER = 191 * GIB + 564 * MIB
OPTIONAL = 24 * GIB + 25 * MIB
EXTRA = 24 * GIB + 64 * MIB
FUNDING = {
    "original_other27_bytes": ORIGINAL_OTHER,
    "context_cap_bytes": 96 * GIB,
    "original_control_cap_bytes": 64 * MIB,
    "additional_control_bytes": 64 * MIB,
    "primary_arm_count": 3,
    "primary_arm_cap_bytes": 12 * GIB,
    "c100_pilot_bytes": 6 * GIB + 4 * MIB,
    "full_c100_bytes": 12 * GIB + 16 * MIB,
    "generative_pilot_bytes": 6 * GIB + 4 * MIB,
    "descriptor_bytes": MIB,
    "standing_free_reserve_bytes": 20 * GIB,
    "shared_future_bytes": 123 * MIB,
    "shared_future_files": 376,
    "no_paid_optional_credit": True,
    "original33_obligations_retained": True,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_decision(value):
    require(value["node"] == "evc27"
            and value["dispatch_order"] == list(ARMS)
            and value["scientific_source_git"] == SCIENCE
            and str(value["original_preparation_CPU_job"]) == "979667",
            "Exact original science and all27 placement required")
    require(value["funding"] == FUNDING, "Complete independently reserved node27 scope required")


def return_root(arm):
    require(arm in ARMS, "Unknown fixed-final return arm")
    return Path(f"/tmp/rsna-predictive-{arm}-evc27-return-20261020-v1")


def reconcile_dispatch(p, value):
    """Read original sealed per-arm receipts after ambiguity; never run sbatch."""
    root = p.ROOT27 / "dispatch"
    require(root.is_dir() and not root.is_symlink(), "Original dispatch intent required")

    def read(path):
        return json.loads(p.read_ref(p.reference(path)))

    intent = read(root / "intent.json")
    require(intent["source_git"] == SCIENCE and intent["owner_git"] == value["owner_git"]
            and intent["arms"] == list(ARMS) and intent["automatic_retry"] is False,
            "Exact original all27 dispatch intent required")
    jobs = {}
    for arm in ARMS:
        receipt = root / (arm + "-submitted.json")
        if not receipt.exists():
            require(not receipt.is_symlink(), "Uncertain dispatch receipt is not reusable")
            continue
        job = read(receipt)
        single = read(root / (arm + "-intent.json"))
        returned = read(root / (arm + "-sbatch.json"))
        identity = job["GPU_job"]
        binding = p.ROOT27 / f"full-admission/binding-{arm}.private.json"
        script_sha = p.file_sha(root / (arm + ".sh"))
        name = "rsna-predictive-full2-" + arm
        require(re.fullmatch(r"\d+", identity)
                and returned["returncode"] == 0
                and re.fullmatch(r"\d+(?:;[^\s]+)?\s*", returned["stdout"])
                and returned["stdout"].strip().split(";")[0] == identity
                and job["arm"] == single["arm"] == arm
                and job["source_git"] == single["source_git"] == SCIENCE
                and job["node"] == "evc27" and job["job_name"] == name
                and job["run_root"] == f"/tmp/rsna-predictive-full-{arm}-20261020-v2"
                and job["binding_sha256"] == single["binding_sha256"]
                == value["portable"]["binding_sha256"][arm] == p.file_sha(binding)
                and job["script_sha256"] == single["script_sha256"] == script_sha
                and single["automatic_retry"] is False,
                "Sealed dispatch differs from original returned job/script/binding")
        for field, suffix in (("stdout", "out"), ("stderr", "err")):
            require(job[field] == f"/home/sa305415/rsna-knee/kaggle/{name}-{identity}.{suffix}",
                    "Original shared log path changed")
        jobs[arm] = job
    complete = (root / "submitted.json").exists()
    if complete:
        final = read(root / "submitted.json")
        require(final["status"] == "actual_predictive_placement_jobs_submitted_once"
                and final["source_git"] == SCIENCE and final["submitted_arm_count"] == 3
                and set(jobs) == set(ARMS) and final["jobs"] == jobs,
                "Original completed dispatch receipt changed")
    return {"status": "actual_all27_original_dispatch_reconciled_readonly", "jobs": jobs,
            "submission_completeness_known": complete,
            "unsealed_arms": [a for a in ARMS if a not in jobs], "new_GPU_jobs": 0}


def specialize_returns(source, arm, namespace):
    """Select the original transport functions; change their fixed arm literal only."""
    require(arm in ARMS, "Unknown final arm")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("return_sender", "return_receiver")]
    require(len(nodes) == 2, "Exact original final transport functions required")

    class Arm(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                if node.value == "actual_C0_final_export_returned_to33_all_bytes_SHA_verified":
                    return ast.copy_location(ast.Constant(
                        "actual_predictive_final_export_returned_to33_all_bytes_SHA_verified"), node)
                return ast.copy_location(ast.Constant(node.value.replace("C0", arm)), node)
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            if any(isinstance(key, ast.Constant) and key.value == "mirror_run_root"
                   for key in node.keys):
                node.keys.append(ast.Constant("arm"))
                node.values.append(ast.Constant(arm))
            return node

    derived = ast.fix_missing_locations(Arm().visit(ast.Module(body=nodes, type_ignores=[])))
    local = dict(namespace)
    exec(compile(derived, "original_final_return_with_explicit_arm", "exec"), local)
    return local["return_sender"], local["return_receiver"]


def admit_remaining(remote, value, original, capsule, rows, transferred):
    """Original CPU initialization validates all three models once, then seal all refs."""
    p = remote["p"]
    root, deferred = p.ROOT27, []
    write = p.once

    def defer_final(path, body):
        if Path(path) == root / "portable-accepted.json":
            require(not deferred and not Path(path).exists(), "Portable final already exists")
            deferred.append(copy.deepcopy(body))
        else:
            write(path, body)

    # This CPU process has one receiver and performs no other publication or writer.
    p.once = defer_final
    try:
        actual = original(value, capsule, rows, transferred)
    finally:
        p.once = write
    require(deferred == [actual] and set(actual["binding_sha256"]) == {"C0"},
            "Original portable acceptance must be sealed only after all arm bindings")
    out = root / "full-admission"
    base = json.loads((out / "binding-C0.private.json").read_text())
    master = json.loads((root / "master.json").read_text())
    input_body = json.loads(p.read_ref(base["input_admission"]))
    funding = json.loads(p.read_ref(base["funding"]))
    require(funding["other_unpaid_node_liabilities_bytes"] >= ORIGINAL_OTHER + OPTIONAL + EXTRA,
            "Other two primary fits and all optional obligations must remain funded")
    require(base["node"] == "evc27" and base["source_git"] == SCIENCE
            and base["expected_initial_shared_tensor_sha256"] == p.INITIAL,
            "Original full model initialization changed")
    input_hashes = {"C0": base["input_admission"]["sha256"]}
    for arm in ("B", "C"):
        require(master["recipes"][arm]["arm"] == arm
                and master["recipes"][arm]["density_weight"] == 0.1
                and master["recipes"][arm]["epochs"] == 6
                and master["recipes"][arm]["schedule_updates"] == 2916,
                "Unchanged original per-arm recipe required")
        run_root = f"/tmp/rsna-predictive-full-{arm}-20261020-v2"
        require(not Path(run_root).exists(), "Existing arm output requires reconciliation")
        decision = json.loads(p.read_ref(base["engineering_decision"]))
        decision.update(arm=arm, recipe=copy.deepcopy(master["recipes"][arm]))
        write(out / f"engineering-{arm}.json", decision)
        arm_funding = dict(funding, arm=arm, run_root=run_root)
        write(out / f"funding-{arm}.json", arm_funding)
        # The input contract is arm-independent. Preserve the complete corrected-source
        # and HTTP499 lineage verbatim, including the one destination index SHA.
        write(out / f"input-{arm}.json", copy.deepcopy(input_body))
        binding = copy.deepcopy(base)
        binding.update(arm=arm, run_root=run_root,
                       engineering_decision=p.reference(out / f"engineering-{arm}.json"),
                       funding=p.reference(out / f"funding-{arm}.json"),
                       input_admission=p.reference(out / f"input-{arm}.json"))
        write(out / f"binding-{arm}.private.json", binding)
        actual["binding_sha256"][arm] = p.file_sha(out / f"binding-{arm}.private.json")
        input_hashes[arm] = binding["input_admission"]["sha256"]
    actual.update(admitted_arms=list(ARMS), input_admission_sha256=input_hashes,
                  placement="all_three_original_arms_on_one_ordinary_two_H100_node",
                  deadline_decision_sha256=p.sha(p.canonical(value["deadline_decision"])),
                  deadline_control_sha256=value["deadline_code_sha256"],
                  original_scientific_initialization_reused=True,
                  additional_context_copies=0)
    write(root / "portable-accepted.json", actual)
    return actual


def install(remote, payload):
    """Retain frozen transport/admission and replace only placement operations."""
    validate_decision(payload["deadline_decision"])
    p = remote["p"]
    require(p.SCIENCE == SCIENCE and p.OTHER27 == ORIGINAL_OTHER,
            "Expected original portable funding implementation required")
    require(payload["shared_bytes"] == 123 * MIB and payload["shared_files"] == 376,
            "Existing six-log scope is reassigned, never enlarged")
    # Original capacity() counts one12GiB fit; OTHER now retains the other two,
    # the added64MiB controls, and all optional scopes without speculative credit.
    p.ROOT27 = ROOT27
    p.OTHER27 = ORIGINAL_OTHER + OPTIONAL + EXTRA
    original_logs = p.paid_shared_logs

    def paid_logs(jobs, *args, **kwargs):
        require(set(jobs) <= set(ARMS)
                and all(j["node"] == "evc27" for j in jobs.values()),
                "Paid log claims must come from actual all27 dispatch receipts")
        # Reuse all existing exact job/path/inode/private-file checks. This local
        # validation projection is never stored or returned as a dispatch receipt.
        compatible = copy.deepcopy(jobs)
        for arm, job in compatible.items():
            job["node"] = "evc27" if arm == "C0" else "evc33"
        return original_logs(compatible, *args, **kwargs)

    p.paid_shared_logs = paid_logs
    derive = p.derived_remote

    def derived(source, node, shared_bytes, shared_files):
        result = derive(source, node, shared_bytes, shared_files)
        if node == "evc27":
            require(result.count('ARMS = ("C0",)') == 1, "Original arm-selection anchor")
            result = result.replace('ARMS = ("C0",)', 'ARMS = ("C", "B", "C0")')
            result = result.replace('"placement_total_GPU_limit": 3',
                                    '"placement_total_GPU_limit": 2')
        return result

    p.derived_remote = derived
    namespace = remote["namespace"]

    def all_namespace(value, node):
        ns = namespace(value, node)
        if node == "evc27":
            other = ns["base_other"]
            # dispatch/publish explicitly add the all-arm fit allowance themselves.
            ns["base_other"] = lambda v: other(v) - 24 * GIB
        return ns

    remote["namespace"] = all_namespace
    source_capsule = remote["source_capsule"]

    def source_only(value):
        require(not (p.ROOT33 / "dispatch").exists(),
                "Source33 has an existing dispatch: reconcile instead of placing duplicate arms")
        return source_capsule(value)

    remote["source_capsule"] = source_only
    original_admit = remote["portable_admit"]
    remote["portable_admit"] = lambda v, c, r, t: admit_remaining(
        remote, v, original_admit, c, r, t)
    original_main = remote["main"]

    def main(value):
        os.umask(0o077)  # Preserve the original private-control-file creation policy.
        require(value["mode"] != "dispatch" or value["node"] == "evc27",
                "Source33 is CPU validation only; no GPU dispatch permitted")
        require(value["mode"] != "admit" or value["node"] == "evc33",
                "Destination admission occurs only inside the single verified receiver")
        if value["mode"] in ("return_send", "return_receive"):
            arm = value["return_arm"]
            p.RETURN33 = return_root(arm)
            send, receive = specialize_returns(remote["__portable_source__"], arm, remote)
            read_frame = p.read_json_frame

            def checked_frame(stream):
                header = read_frame(stream)
                if value["mode"] == "return_receive":
                    job = value["dispatched"]["jobs"][arm]
                    require(header["terminal"]["arm"] == arm
                            and header["terminal"]["GPU_job"] == job["GPU_job"]
                            and header["terminal"]["source_git"] == SCIENCE
                            and header["terminal"]["binding_sha256"]
                            == header["binding_sha256"] == job["binding_sha256"],
                            "Final return header differs from the actual arm dispatch")
                return header

            p.read_json_frame = checked_frame
            # Retain the original entry-point handoff checks before invoking these
            # parameterized functions through its original routing table.
            remote["return_sender"], remote["return_receiver"] = send, receive
            try:
                result = original_main(value)
            finally:
                p.read_json_frame = read_frame
            require(result["original_training_GPU_job"]
                    == value["dispatched"]["jobs"][arm]["GPU_job"], "Returned job differs")
            return dict(result, arm=arm)
        if value["node"] != "evc27" or value["mode"] in ("receive", "send"):
            return original_main(value)
        p.handoff(value["handoff"])
        require(p.sha(value["handoff_text"].encode()) == value["handoff_sha256"]
                and json.loads(value["handoff_text"]) == value["handoff"],
                "Original handoff byte binding changed")
        ns = all_namespace(value, "evc27")
        ns["SCIENCE"] = SCIENCE

        def preparation(v):
            actual = json.loads((p.ROOT27 / "portable-accepted.json").read_text())
            require(actual == v["portable"] and actual["admitted_arms"] == list(ARMS)
                    and actual["handoff_sha256"] == v["handoff_sha256"],
                    "Exact all-arm portable admission required")
            require(os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
                    and Path(sys.executable).resolve() == Path(p.PY27).resolve(),
                    "Original owning27 CPU runtime required")
            for arm in ARMS:
                path = p.ROOT27 / f"full-admission/binding-{arm}.private.json"
                require(p.file_sha(path) == actual["binding_sha256"][arm], "Arm binding changed")
                binding = json.loads(path.read_text())
                for name, digest in binding["code_sha256"].items():
                    require(p.file_sha(p.ROOT27 / "code" / name) == digest,
                            "Original numerical code changed")
            return actual

        ns["preparation"] = preparation
        if value["mode"] == "dispatch_reconcile":
            preparation(value)
            return reconcile_dispatch(p, value)
        if value["mode"] == "dispatch":
            preparation(value)
            envelope = dict(value["portable"], arms=list(ARMS),
                            status="actual_full3887_predictive_CPU_admission_complete_no_GPU_dispatched")
            p.once(p.ROOT27 / "full-admission/actual.json", envelope)
            value = dict(value, admitted=envelope)
        return ns["main"](value)

    remote["main"] = main
