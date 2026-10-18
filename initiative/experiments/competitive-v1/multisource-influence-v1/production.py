"""Compile one label-only candidate around the actual completed influence fit.

This compiler cannot acquire data, dispatch a job or submit to competition.
Preparation is separate from building: building requires actual MRI recovery.
"""

import ast
import base64
import hashlib
import json

CONTROL_SOURCE = "9f863df7849ba4a120b690ffc4246943fba10b50cbef436cdacf233fce55c15d"
CONTROL_WORKER = "932c1bd956da6673271275b8652c0d25b38e2840400629026a7b237d90d2c7ea"
SOURCE_CLASS = "3b10769c0f75aef474fcb587daf6e3e2725f2b1cfcccfa9845c08904a6f903a0"
FIT = "107c24cccee6a16bb4fc4913b2c9193161cb38060610297ef3c3f58d8d29d27e"
RECOVERY = "3fd77316770abec5b05a6297e6f3a38c73458b86f84cb2d6b6763ecd37c6f825"
RECOVERY_WORKER = "acfa946919bdf1d9fbff3b34c196d4e5c4545117bdadb9ff68da419afbfdd5d5"
TARGET = "65d3730702a098c8c819396dba6122902a4f3720dde9b2cb3f9ffd7ac5eca3b5"
ANNOTATIONS = "54f1c94943bd7049b3e3da79fdcffb1f7083ecb373a64b9e2642aa2fe9230df9"
STATUS = "actual_family_influence16_full_parent_epoch_session_complete"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def literals(text, names):
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                if target.id in result:
                    raise ValueError("Duplicate original assignment")
                result[target.id] = ast.literal_eval(node.value)
    if set(result) != set(names):
        raise ValueError("Complete original literal closure required")
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


def one(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Ambiguous original worker substitution")
    return text.replace(before, after)


HELPER = r"""
import importlib.util
_ms=importlib.util.spec_from_file_location('production_family_source',Path(__file__).parent/'multisource/sampled_overlay.py')
_sf=importlib.util.module_from_spec(_ms);_ms.loader.exec_module(_sf)
_fi_spec=importlib.util.spec_from_file_location('production_family_influence',Path(__file__).parent/'multisource/influence_fit.py')
_fi=importlib.util.module_from_spec(_fi_spec);_fi_spec.loader.exec_module(_fi);fit=_fi.fit_for(_sf)

def _full_source_labels(original):
    assert len(original)==3887
    roots=[Path('/kaggle/input/rsna-family-fit-65d3730702a0'),Path('/kaggle/input/datasets/godlydonuts/rsna-family-fit-65d3730702a0')]
    roots=[p for p in roots if (p/'targets.private.npz').is_file()];assert len(roots)==1
    root=roots[0];raw=(root/'targets.private.npz').read_bytes()
    assert len(raw)==493839 and hashlib.sha256(raw).hexdigest()==_sf.PRODUCTION_TARGET
    annotation=(root/'annotation-tensor.private.npz').read_bytes()
    assert len(annotation)==343927 and hashlib.sha256(annotation).hexdigest()==_sf.PRODUCTION_ANNOTATIONS
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:values={k:a[k].copy() for k in a.files}
    with np.load(io.BytesIO(annotation),allow_pickle=False) as a:eligible=torch.from_numpy(a['eligible'].copy())
    assert set(values)=={'uids','labels','masks','parent_logits'} and values['uids'].tolist()==original.uids
    labels=torch.from_numpy(values['labels']);mask=torch.from_numpy(values['masks']);teacher=torch.from_numpy(values['parent_logits'])
    assert torch.equal(mask,original.mask) and torch.equal(teacher,original.teacher)
    policy={'version':_sf.VERSION,'role':'safe_family_mean','diagnostic_only':False,'parent_checkpoint_sha256':PARENT_SHA,'original_target_file_sha256':_sf.ORIGINAL_FILE,'population':len(original),'population_sha256':_sf.POPULATION,'original_labels_sha256':_sf.sha(original.labels),'labels_sha256':_sf.sha(labels),'mask_sha256':_sf.sha(mask),'teacher_sha256':_sf.sha(teacher),'eligible_sha256':_sf.sha(eligible),'target_file_sha256':_sf.PRODUCTION_TARGET,'annotation_file_sha256':_sf.PRODUCTION_ANNOTATIONS}
    return _sf.ConsensusLabels(original,labels,eligible,policy,diagnostic_only=False)

"""


def prepare(original, source_class, influence_fit):
    if (
        sha(original) != CONTROL_SOURCE
        or sha(source_class) != SOURCE_CLASS
        or sha(influence_fit) != FIT
    ):
        raise ValueError("Exact completed influence control and accepted source adapters required")
    values = literals(
        original, ("FILES", "WORKER", "BINDING", "SEGMENT", "ROLE", "PREVIOUS", "PERMISSION")
    )
    binding = values["BINDING"]
    files = values["FILES"]
    worker = base64.b64decode(values["WORKER"], validate=True).decode()
    if (
        values["ROLE"] != "influence"
        or values["SEGMENT"] != 1
        or values["PREVIOUS"] is not None
        or binding.get("role") != "influence"
        or binding.get("diagnostic_only") is not False
        or len(files) != 13
        or {n: sha(s) for n, s in files.items()} != binding["source_module_sha256"]
        or sha(worker) != CONTROL_WORKER
        or hashlib.sha256(base64.b64decode(values["PERMISSION"], validate=True)).hexdigest()
        != "34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79"
    ):
        raise ValueError("Actual original source/worker/parent/mask/rights closure required")
    worker = one(
        worker,
        "from rsna.competitive.licensed_importance_train import fit",
        "from rsna.competitive.licensed_importance_train import fit as _unchanged_fit",
    )
    worker = one(worker, "def main():\n", HELPER + "def main():\n")
    worker = one(
        worker,
        "data=ProposalCases(original,torch.from_numpy(q));del q",
        "data=ProposalCases(_full_source_labels(original),torch.from_numpy(q));del q",
    )
    compile(worker, "family_influence_full_fit_worker_template", "exec")
    return (
        values,
        worker,
        {
            **files,
            "multisource/sampled_overlay.py": source_class,
            "multisource/influence_fit.py": influence_fit,
        },
    )


def accept_recovery(actual, files):
    if (
        not isinstance(actual, dict)
        or actual.get("status") != "COMPLETE"
        or actual.get("source_sha256") != RECOVERY
        or actual.get("kernel_id") != 139336728
        or actual.get("version") != 1
    ):
        raise ValueError("Actual exact coupled influence-family MRI recovery required")
    r = actual.get("receipt", {})
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
        r.get("status") != "actual_family_label_only_influence16_twoT4_MRI_recovery_accepted"
        or r.get("worker_source_sha256") != RECOVERY_WORKER
        or r.get("family_target_sha256") != TARGET
        or r.get("source_module_sha256") != numerical
        or r.get("original_control_complete_state_and_predictions_exact_except_label_binding")
        is not True
        or r.get("same_initial_complete_model_and_allrank_RNG_between_roles_and_original")
        is not True
        or r.get("clinical_or_competition_accuracy_claim") is not False
        or set(r.get("roles", {})) != {"unchanged_reference", "safe_family_mean"}
    ):
        raise ValueError("Full numerical/array/control/recovery identity required")
    for role in ("unchanged_reference", "safe_family_mean"):
        value = r["roles"][role]
        if (
            value.get("finite_updates") != 4
            or value.get(
                "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal"
            )
            is not True
            or value.get("all16_full94_two_view_prediction_bytes_equal") is not True
            or value.get("label_binding", {}).get("role") != role
            or value["label_binding"].get("diagnostic_only") is not True
        ):
            raise ValueError("Both original source roles must pass complete actual recovery")
    return r


def build(
    original, source_class, influence_fit, actual_recovery, source_git, segment, *, previous=None
):
    values, worker, files = prepare(original, source_class, influence_fit)
    receipt = accept_recovery(actual_recovery, files)
    if (
        type(segment) is not int
        or not 1 <= segment <= 3
        or not isinstance(source_git, str)
        or len(source_git) != 40
        or any(c not in "0123456789abcdef" for c in source_git)
    ):
        raise ValueError("One pinned fixed-final three-epoch source required")
    binding = {
        **values["BINDING"],
        "production_plan": "licensed_MRI_parent_family_mean_influence16_full3887_fixed3_v1",
        "source_git": source_git,
        "source_module_sha256": {n: sha(s) for n, s in files.items()},
        "source_label_role": "safe_family_mean",
        "source_label_target_sha256": TARGET,
        "annotation_file_sha256": ANNOTATIONS,
        "actual_family_recovery": {
            "kernel_id": 139336728,
            "version": 1,
            "source_sha256": RECOVERY,
            "worker_sha256": RECOVERY_WORKER,
            "receipt_canonical_JSON_sha256": sha(
                json.dumps(receipt, sort_keys=True, separators=(",", ":"))
            ),
        },
    }
    worker = assignments(worker, {"_BINDING": binding})
    if segment == 1:
        if previous is not None:
            raise ValueError("First epoch starts from original parent only")
    elif (
        not isinstance(previous, dict)
        or previous.get("status") != STATUS
        or previous.get("role") != "influence"
        or previous.get("segment") != segment - 1
        or previous.get("completed") != (segment - 1) * 486
        or previous.get("worker_sha256") != sha(worker)
        or previous.get("binding") != binding
    ):
        raise ValueError("Exact previous same-candidate optimizer/RNG/source boundary required")
    if previous is not None:
        for name in ("checkpoint", "run"):
            row = previous.get(name, {})
            if (
                type(row.get("bytes")) is not int
                or not 0 < row["bytes"] <= 2 * 2**30
                or not isinstance(row.get("sha256"), str)
                or len(row["sha256"]) != 64
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
        "state['binding']['total_steps']==1458 and state['binding']['multisource_supervision']['role']=='safe_family_mean' and state['binding']['multisource_supervision']['diagnostic_only'] is False and state['binding']['multisource_supervision']['target_file_sha256']==BINDING['source_label_target_sha256'];assert",
    )
    compile(source, "actual_MRI_recovery_gated_family_influence_fixed3_epoch", "exec")
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
