"""Prepare the matched corrected-logit fit; compilation never admits execution.

Reuse the accepted original full-population compiler and change only the explicit
paired readout, source-bound correction loader and minimal optimizer fork.
Actual noise/runtime decision, funding and one-time dispatch remain separate.
"""

from __future__ import annotations

import ast
import base64
import json
from dataclasses import asdict

from rsna.competitive.licensed_importance import Recipe
from scripts.kaggle import licensed_importance_production as original

RECOVERY_SOURCE = "4b129e1c5c76bcc226eff30f30e9e59b25df0ba91358771b795f7d65f68ad6b5"
RECOVERY_WORKER = "f888b4cc327bf1d09adb05376a4ef3c3fa804169a601e7dc7974fb5bb3e24a2e"
EXTRA = {
    "rsna/competitive/paired_logit_cases.py": "6912ee44c6e3b2891de526471bbab2e32356577264382f15289fc5cfa99b19e3",
    "rsna/competitive/paired_logit_control.py": "c95b35b672654cabeac6a8d46432ef878c2e084d2d8ea7dbcb38b0f1fa0ce896",
    "rsna/competitive/paired_logit_train.py": "6e971e68c20a923d95a1f6115717aca97bb0761576ec1ca8a79c3ae34e3b9350",
}
TABLE = {
    "bytes": 1629919,
    "sha256": "bfd9d6d67afc7e7f3c9cd091bfccae8c8fc741b23ea1679deb3da122cd0fb619",
}
STATUS = "actual_paired_parent_logit_full3887_epoch_session_complete"


def _literal(source, name):
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    ]
    if len(nodes) != 1:
        raise ValueError("One explicit compiler literal required: " + name)
    return nodes[0], ast.literal_eval(nodes[0].value)


def _set_literal(source, name, value):
    node, _ = _literal(source, name)
    lines = source.encode().splitlines(keepends=True)
    start = sum(map(len, lines[: node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[: node.end_lineno - 1])) + node.end_col_offset
    return (
        source.encode()[:start] + (name + "=" + repr(value)).encode() + source.encode()[end:]
    ).decode()


def _paired_inputs(actual, correction, files):
    r = actual.get("receipt", {})
    role = r.get("roles", {}).get("influence", {})
    expected_modules = {**original.ACCEPTED_MODULES, **EXTRA}
    if (
        actual.get("status") != "COMPLETE"
        or actual.get("kernel_id") != 139417155
        or actual.get("version") != 1
        or actual.get("source_sha256") != RECOVERY_SOURCE
        or r.get("status") != "actual_paired_logit_encoder_FP16_dropout_recovery_accepted"
        or r.get("worker_source_sha256") != RECOVERY_WORKER
        or r.get("source_module_sha256") != expected_modules
        or r.get("fit_studies") != 16
        or r.get("disjoint_engineering_studies") != 16
        or r.get("epochs") != 3
        or r.get("updates_per_role") != 6
        or set(r.get("roles", {})) != {"influence"}
        or role.get(
            "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal"
        )
        is not True
        or role.get("all16_full94_two_view_prediction_bytes_equal") is not True
        or len(role.get("devices", [])) != 2
        or any(
            d.get("GPU") != "Tesla T4"
            or d.get("trainable_parameters") != 73136096
            or d.get("all_BN_buffers_equal_parent") is not True
            for d in role.get("devices", [])
        )
        or set(files) != {*original.ACCEPTED_MODULES, *original.POPULATION_MODULES, *EXTRA}
        or any(original.digest(files[n]) != pin for n, pin in EXTRA.items())
    ):
        raise ValueError(
            "Actual exact paired-encoder recovery and immutable numerical closure required"
        )
    if (
        correction.get("status")
        != "actual_full3887_fixed3_IID16_parent_logit_correction_schedule_CPU_constructed"
        or correction.get("population") != 3887
        or correction.get("epochs") != 3
        or correction.get("same_original_sampler_replay_matches") != 11661
        or correction.get("fitting_order_sha256") != original.POPULATION_SHA
        or correction.get("original_proposal_NPY_sha256") != original.PROPOSAL_SHA
        or {k: correction.get("table", {}).get(k) for k in TABLE} != TABLE
        or correction.get("control_sha256") != EXTRA["rsna/competitive/paired_logit_control.py"]
        or correction.get("recipe") != asdict(Recipe(role="influence"))
        or correction.get("Torch_CPU_global_RNG_unchanged") is not True
        or correction.get("accepted_masks_labels_and_live_training_changed") is not False
    ):
        raise ValueError("Exact privately preserved full fitting correction schedule required")


def build(
    accepted_source,
    baseline_recovery,
    files,
    proofs,
    proposal,
    source_git,
    segment,
    *,
    paired_recovery,
    correction,
    previous=None,
):
    _paired_inputs(paired_recovery, correction, files)
    if type(segment) is not int or not 1 <= segment <= 3:
        raise ValueError("One fixed original epoch boundary required")
    source, base_meta = original.build(
        accepted_source,
        baseline_recovery,
        {n: t for n, t in files.items() if n not in EXTRA},
        proofs,
        proposal,
        source_git,
        "influence",
        1,
    )
    _, encoded = _literal(source, "WORKER")
    worker = base64.b64decode(encoded, validate=True).decode()
    binding = {
        **base_meta["binding"],
        "production_plan": "paired_parent_logit_IID16_full3887_fixed3_continuous_v1",
        "training_bag": "same_IID16_inverse_proposal_with_explicit_detached_parent_logit_correction",
        "source_module_sha256": {n: original.digest(t) for n, t in sorted(files.items())},
        "paired_recovery": {
            "kernel_id": 139417155,
            "source_sha256": RECOVERY_SOURCE,
            "worker_sha256": RECOVERY_WORKER,
            "receipt_sha256": original.digest(
                json.dumps(paired_recovery["receipt"], sort_keys=True, separators=(",", ":"))
            ),
        },
        "correction_table": TABLE,
        "correction_receipt_sha256": original.digest(
            json.dumps(correction, sort_keys=True, separators=(",", ":"))
        ),
    }

    def change(old, new):
        nonlocal worker
        if worker.count(old) != 1:
            raise ValueError("Original full-population worker changed; explicit review required")
        worker = worker.replace(old, new, 1)

    change(
        "class WholeStudyRaptor(ImportanceReadout,CheckpointedEncode,RaptorClassifier):pass",
        "from rsna.competitive.paired_logit_control import PairedLogitReadout\nclass WholeStudyRaptor(PairedLogitReadout,ImportanceReadout,CheckpointedEncode,RaptorClassifier):pass",
    )
    change(
        "from rsna.competitive.licensed_importance_train import fit",
        "from rsna.competitive.paired_logit_train import fit\nfrom rsna.competitive.paired_logit_cases import PairedCases",
    )
    change(
        "data=ProposalCases(original,torch.from_numpy(q));del q",
        "raw=ProposalCases(original,torch.from_numpy(q));del q;table=locate('/kaggle/input','corrections.private.npz',1629919,_BINDING['correction_table']['sha256']);data=PairedCases(raw,table,_BINDING['correction_table']['sha256'],Recipe(role=_ROLE))",
    )
    worker = _set_literal(worker, "_BINDING", binding).replace("importance-full", "paired-full")
    compile(worker, "new_paired_full3887_worker", "exec")
    worker_sha = original.digest(worker)
    if segment == 1:
        if previous is not None:
            raise ValueError(
                "First paired epoch starts from original parent, never another candidate"
            )
    elif (
        not isinstance(previous, dict)
        or previous.get("status") != STATUS
        or previous.get("role") != "influence"
        or previous.get("segment") != segment - 1
        or previous.get("completed") != 486 * (segment - 1)
        or previous.get("binding") != binding
        or previous.get("worker_sha256") != worker_sha
    ):
        raise ValueError(
            "Same paired source, correction table and complete previous optimizer required"
        )
    if previous is not None:
        for name in ("checkpoint", "run"):
            item = previous.get(name, {})
            if (
                type(item.get("bytes")) is not int
                or not 0 < item["bytes"] < 2 * 2**30
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in item["sha256"])
            ):
                raise ValueError("Complete previous checkpoint/run byte identities required")
    for name, value in {
        "FILES": files,
        "WORKER": base64.b64encode(worker.encode()).decode(),
        "BINDING": binding,
        "SEGMENT": segment,
        "PREVIOUS": previous,
    }.items():
        source = _set_literal(source, name, value)
    source = (
        source.replace(
            "licensed-importance-full-session-accepted.json",
            "paired-logit-full-session-accepted.json",
        )
        .replace("actual_matched_IID16_full_parent_epoch_session_complete", STATUS)
        .replace("importance-full", "paired-full")
    )
    guard = ";del state\n"
    if source.count(guard) != 1:
        raise ValueError("One complete checkpoint acceptance boundary required")
    source = source.replace(
        guard,
        ";assert state['binding']['version']=='paired_parent_logit_IID16_training_v1' and state['binding']['parent_correction_sha256']==BINDING['correction_table']['sha256'];del state\n",
        1,
    )
    if _literal(source, "FILES")[1] != files:
        raise ValueError("Template edits must not change any numerical module byte")
    compile(source, "new_paired_full3887_epoch", "exec")
    return source, {
        **base_meta,
        "source_sha256": original.digest(source),
        "worker_sha256": worker_sha,
        "binding": binding,
        "segment": segment,
        "completed_target": 486 * segment,
        "prepared_template_only": True,
        "full_production_fit": False,
        "intended_full_fit_recipe": True,
        "actual_noise_runtime_funding_and_dispatch_decision_still_required": True,
        "production_execution_admitted_by_compilation": False,
        "new_dataset_or_GPU_mutation": False,
    }
