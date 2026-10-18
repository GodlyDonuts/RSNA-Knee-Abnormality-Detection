"""One sampling-only fixed-final comparison using the accepted original engine."""

import ast
import base64
import hashlib
import json

CONTROL_SOURCE = "9f863df7849ba4a120b690ffc4246943fba10b50cbef436cdacf233fce55c15d"
CONTROL_WORKER = "932c1bd956da6673271275b8652c0d25b38e2840400629026a7b237d90d2c7ea"
SAMPLER = "71d7e41228bf06c0358f619d90f85cbf9db2b81acbdf5e4c83221fb4b1304ebf"
FIT = "2bbb6e0cfd327a14ef1a96017dae3e8193519a8bd1f1c83a5f913ad64cdfddb0"
RECOVERY = "7a4e7db6aa40c70e3a64f5faf2a5f11f3e20423725c76acecfa12a59861e74d2"
RECOVERY_WORKER = "ff8c3991f9b4f4382251dcaec2b52e9726d5a95f04d3393deba0788dfa7e0e55"
TENSOR = "3581baf78d76d75e1e0514817ca3fba024b0d8febd8ab05e90ba21878478628e"
ROSTER = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
PLAN = "finding_stratified12plus4_original_density_full3887_fixed3_v1"
STATUS = "actual_finding_strata_full_parent_epoch_session_complete"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def literals(text, names):
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                if target.id in result:
                    raise ValueError("Duplicate original literal")
                result[target.id] = ast.literal_eval(node.value)
    if set(result) != set(names):
        raise ValueError("Complete original closure required")
    return result


def assignments(text, values):
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in values:
                nodes.append((node.lineno, node.end_lineno, target.id))
    if len(nodes) != len(values) or {n for _, _, n in nodes} != set(values):
        raise ValueError("Unique complete original assignments required")
    lines = text.splitlines(keepends=True)
    for first, last, name in sorted(nodes, reverse=True):
        lines[first - 1 : last] = [name + "=" + repr(values[name]) + "\n"]
    return "".join(lines)


def one(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Ambiguous original substitution")
    return text.replace(old, new)


HELPER = r"""
def _full_components(data):
    roots=[Path('/kaggle/input/rsna-finding-strata-3581baf78d76'),Path('/kaggle/input/datasets/godlydonuts/rsna-finding-strata-3581baf78d76')]
    roots=[r for r in roots if (r/'components.private.npy').is_file()];assert len(roots)==1
    root=roots[0];path=root/'components.private.npy';uids=(root/'uids.private.json').read_bytes()
    assert path.stat().st_size==35076416 and hashlib.sha256(path.read_bytes()).hexdigest()=='3581baf78d76d75e1e0514817ca3fba024b0d8febd8ab05e90ba21878478628e'
    assert len(uids)==260430 and hashlib.sha256(uids).hexdigest()=='0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42'
    assert json.loads(uids)==data.uids and len(data)==3887
    components=np.load(path,allow_pickle=False)
    assert components.dtype==np.float64 and components.shape==(3887,94,12)
    assert hashlib.sha256(components.tobytes()).hexdigest()==_BINDING['component_raw_sha256']
    return components

"""


def prepare(original, sampler, fit):
    if sha(original) != CONTROL_SOURCE or sha(sampler) != SAMPLER or sha(fit) != FIT:
        raise ValueError(
            "Exact completed original influence source and GPU-accepted adapters required"
        )
    values = literals(
        original, ("FILES", "WORKER", "BINDING", "SEGMENT", "ROLE", "PREVIOUS", "PERMISSION")
    )
    worker = base64.b64decode(values["WORKER"], validate=True).decode()
    if (
        values["ROLE"] != "influence"
        or values["SEGMENT"] != 1
        or values["PREVIOUS"] is not None
        or values["BINDING"].get("role") != "influence"
        or values["BINDING"].get("diagnostic_only") is not False
        or len(values["FILES"]) != 13
        or {n: sha(s) for n, s in values["FILES"].items()}
        != values["BINDING"]["source_module_sha256"]
        or sha(worker) != CONTROL_WORKER
        or hashlib.sha256(base64.b64decode(values["PERMISSION"], validate=True)).hexdigest()
        != "34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79"
    ):
        raise ValueError(
            "Actual control modules, original source labels, masks and rights required"
        )
    worker = one(
        worker,
        "from rsna.competitive.licensed_importance_train import fit",
        "from finding_fit import fit",
    )
    worker = one(worker, "def main():\n", HELPER + "def main():\n")
    worker = one(
        worker,
        "data=ProposalCases(original,torch.from_numpy(q));del q",
        "data=ProposalCases(original,torch.from_numpy(q));del q\n    finding_components=_full_components(data)",
    )
    worker = one(
        worker,
        "device=device,resume=a.resume,stop_after=a.stop_after)",
        "device=device,resume=a.resume,stop_after=a.stop_after,components=finding_components,allocation='finding_strata')",
    )
    compile(worker, "finding_strata_full_worker", "exec")
    return values, worker, {**values["FILES"], "sampling.py": sampler, "finding_fit.py": fit}


def accept_recovery(actual, files):
    receipt = actual.get("receipt", {})
    numerical = {
        n: sha(s)
        for n, s in files.items()
        if n
        not in (
            "rsna/competitive/licensed_pixel_cache.py",
            "rsna/competitive/licensed_pixel_bags.py",
            "rsna/competitive/licensed_parent_population.py",
        )
    }
    if (
        actual.get("status") != "COMPLETE"
        or actual.get("kernel_id") != 139349540
        or actual.get("version") != 1
        or actual.get("source_sha256") != RECOVERY
        or receipt.get("status")
        != "actual_finding_strata_bothT4_original_control_and_recovery_accepted"
        or receipt.get("worker_source_sha256") != RECOVERY_WORKER
        or receipt.get("source_module_sha256") != numerical
        or receipt.get("original_influence_complete_control_state_and_prediction_bytes_equal")
        is not True
        or receipt.get("clinical_or_competition_accuracy_claim") is not False
        or set(receipt.get("roles", {})) != {"iid", "finding_strata"}
    ):
        raise ValueError("Actual exact both-T4 original-control and sampling recovery required")
    for row in receipt["roles"].values():
        if (
            row.get("finite_updates") != 4
            or row.get(
                "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal"
            )
            is not True
            or row.get("all16_full94_two_view_prediction_bytes_equal") is not True
        ):
            raise ValueError("Both allocations must pass all complete-state and prediction checks")
    return receipt


def build(original, sampler, fit, actual, source_git, segment, raw_sha, *, previous=None):
    values, worker, files = prepare(original, sampler, fit)
    recovery = accept_recovery(actual, files)
    if (
        type(segment) is not int
        or not 1 <= segment <= 3
        or not isinstance(source_git, str)
        or len(source_git) != 40
        or any(c not in "0123456789abcdef" for c in source_git)
        or not isinstance(raw_sha, str)
        or len(raw_sha) != 64
        or any(c not in "0123456789abcdef" for c in raw_sha)
    ):
        raise ValueError("Pinned source, full tensor and fixed three-epoch boundary required")
    binding = {
        **values["BINDING"],
        "production_plan": PLAN,
        "source_git": source_git,
        "source_module_sha256": {n: sha(s) for n, s in files.items()},
        "component_file_sha256": TENSOR,
        "component_raw_sha256": raw_sha,
        "component_roster_sha256": ROSTER,
        "actual_finding_recovery": {
            "kernel_id": 139349540,
            "version": 1,
            "source_sha256": RECOVERY,
            "worker_sha256": RECOVERY_WORKER,
            "receipt_canonical_JSON_sha256": sha(
                json.dumps(recovery, sort_keys=True, separators=(",", ":"))
            ),
        },
    }
    worker = assignments(worker, {"_BINDING": binding})
    if segment == 1:
        if previous is not None:
            raise ValueError("First epoch starts from the original parent")
    elif (
        not isinstance(previous, dict)
        or previous.get("status") != STATUS
        or previous.get("role") != "influence"
        or previous.get("segment") != segment - 1
        or previous.get("completed") != (segment - 1) * 486
        or previous.get("binding") != binding
        or previous.get("worker_sha256") != sha(worker)
    ):
        raise ValueError("Exact previous same-allocation optimizer/RNG/source boundary required")
    if previous is not None:
        for n in ("checkpoint", "run"):
            row = previous.get(n, {})
            if (
                type(row.get("bytes")) is not int
                or not 0 < row["bytes"] <= 2 * 2**30
                or len(row.get("sha256", "")) != 64
            ):
                raise ValueError("Whole-byte original resume proof required")
    source = assignments(
        original,
        {
            "FILES": files,
            "WORKER": base64.b64encode(worker.encode()).decode(),
            "BINDING": binding,
            "SEGMENT": segment,
            "ROLE": "influence",
            "PREVIOUS": previous,
        },
    )
    source = one(
        source,
        "'status':'actual_matched_IID16_full_parent_epoch_session_complete'",
        "'status':" + repr(STATUS),
    )
    source = one(
        source,
        "state['binding']['total_steps']==1458;assert",
        "state['binding']['total_steps']==1458 and state['binding']['finding_allocation']=={'version':'finding_stratified_12plus4_original_density_v1','allocation':'finding_strata','components_sha256':BINDING['component_raw_sha256'],'mixture_density_unchanged':True,'images_per_study':16};assert",
    )
    compile(source, "accepted_finding_strata_fixed3_full_fit", "exec")
    return source, {
        "source_sha256": sha(source),
        "worker_sha256": sha(worker),
        "binding": binding,
        "segment": segment,
        "completed_target": 486 * segment,
        "maximum_GPU_seconds": 6000,
        "competition_entries": 0,
        "full_production_fit": True,
    }
