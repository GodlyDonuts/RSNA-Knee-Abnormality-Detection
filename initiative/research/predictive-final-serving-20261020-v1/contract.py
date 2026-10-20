"""Strict model-only final export contract; no account, network or data access."""

import hashlib
import json
import math
import re

PARENT = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
PARENT_SOURCE = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
LICENSE = "34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79"
TARGETS = "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
ROSTER = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
SOURCE_REVISION = "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
PUBLISHER = "6215f45748fdc4685565a94f31da7fc51f7ad0fa325192d004b362645bbf8d87"
PUBLISHER_CORE = "4e327e271816e4a4055b4f3ad28a8dab59fc650a1367fd13bbd7fe06d9a5efde"
PARAMETERS = {"B": 73092652, "C": 73271124, "C0": 73271124}
SCHEMAS = {
    "B": "8372ccea6c397149dd221b79353ab51e563e3627342a1cb1926dd0eb68fad725",
    "C": "14f6daa9697ac5c50499abb3a981aff776de1048159aefd249888e9b1d1740a7",
    "C0": "14f6daa9697ac5c50499abb3a981aff776de1048159aefd249888e9b1d1740a7",
}
MODULES = tuple(
    "rsna/competitive/predictive/" + n + ".py"
    for n in (
        "__init__",
        "inputs",
        "cases",
        "supports",
        "density",
        "model",
        "infer",
        "dicom",
    )
) + ("rsna/__init__.py", "rsna/competitive/__init__.py")
COLUMNS = [
    "StudyInstanceUID",
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def recipe(arm):
    return dict(
        arm=arm,
        epochs=6,
        effective_batch=8,
        encoder_lr=3e-5,
        head_lr=3e-5,
        weight_decay=0.02,
        warmup_fraction=0.05,
        end_lr_fraction=0.1,
        density_weight=0 if arm == "C0" else 0.1,
        head_only_updates=3,
        clip=1.0,
        seed=20261020,
        schedule_updates=2916,
    )


def architecture(arm):
    return dict(
        arm=arm,
        encoder="CoAtNet_RMLP2",
        resolution=320,
        hidden=128,
        classes=12,
        descriptor_width=64,
        context_cap=24,
        halo=8,
    )


def candidate(publication, provenance_text, own_source_text, license_bytes):
    """Bind exact independently downloaded bytes; never accept a pilot/old teacher."""
    p = json.loads(provenance_text)
    sources = json.loads(own_source_text)
    arm = p["arm"]
    require(arm in PARAMETERS, "Known trained predictive arm required")
    require(
        publication["status"]
        == "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
        and publication["objects"] == 5
        and publication["canonical_shared_publication_lock_used"] is True,
        "Actual independent private five-object publication required",
    )
    files = publication["files"]
    require(
        set(files)
        == {
            "model.pt",
            "provenance.json",
            "own-source.json",
            "source-permission.txt",
            "manifest.json",
        },
        "Exact original five publication objects required",
    )
    for name, contents in (
        ("provenance.json", provenance_text.encode()),
        ("own-source.json", own_source_text.encode()),
        ("source-permission.txt", license_bytes),
    ):
        require(
            files[name] == {"bytes": len(contents), "sha256": digest(contents)},
            "Original independently downloaded small object differs",
        )
    require(digest(license_bytes) == LICENSE, "Exact inherited MRI license required")
    require(
        re.fullmatch("[0-9a-f]{40}", publication["revision"]) is not None
        and re.fullmatch("[0-9a-f]{40}", p["source_git"]) is not None,
        "Immutable source and private publication revisions required",
    )
    require(
        (
            publication["arm"],
            publication["epoch"],
            publication["GPU_job"],
            publication["source_git"],
        )
        == (arm, 6, p["GPU_job"], p["source_git"]),
        "Exact final production publication required",
    )
    require(
        publication["prefix"]
        == f"checkpoints/predictive-full3887-{p['source_git'][:12]}/{arm}/epoch-006",
        "Original final publication path required",
    )
    expected = dict(
        epoch=6,
        completed_updates=2916,
        cohort_size=3887,
        recipe=recipe(arm),
        parent_weight_sha256=PARENT,
        parent_source_sha256=PARENT_SOURCE,
        source_revision=SOURCE_REVISION,
        targets_sha256=TARGETS,
        fitting_order_sha256=ROSTER,
        license_sha256=LICENSE,
        parameter_count=PARAMETERS[arm],
        model_schema_sha256=SCHEMAS[arm],
        publisher_sha256=PUBLISHER,
        original_publisher_core_sha256=PUBLISHER_CORE,
        model_source="public_licensed_MRI_parent_and_unchanged_RSNA_fitting_only",
        additional_Stanford_or_KneeCoT_or_teacher_probabilities_used=False,
        optimizer_RNG_private_paths_labels_IDs_exported=False,
    )
    require(
        all(p.get(k) == v for k, v in expected.items()),
        "Final trained lineage/recipe/schema differs",
    )
    for key in (
        "binding_sha256",
        "context_index_sha256",
        "source_context_index_sha256",
        "runtime_receipt_sha256",
        "engineering_decision_sha256",
        "input_admission_sha256",
        "funding_sha256",
        "initial_shared_tensor_sha256",
        "checkpoint_sha256",
        "history_sha256",
        "sealed_sidecar_sha256",
    ):
        require(
            re.fullmatch("[0-9a-f]{64}", p[key]) is not None,
            "Complete original production provenance required",
        )
    require(
        all(
            name in sources and digest(sources[name]) == p["code_sha256"].get(name)
            for name in MODULES
        ),
        "Every serving numerical module must equal the admitted production source",
    )
    require(
        p["renderer_sha256"]
        == {
            n: p["code_sha256"][n]
            for n in MODULES
            if n.rsplit("/", 1)[-1] in ("inputs.py", "cases.py", "supports.py")
        },
        "Exact source-masking/rendering proof required",
    )
    model = files["model.pt"]
    require(
        type(model["bytes"]) is int
        and 280_000_000 < model["bytes"] < 340_000_000
        and re.fullmatch("[0-9a-f]{64}", model["sha256"]) is not None,
        "Complete model-only bytes required",
    )
    return {
        "arm": arm,
        "architecture": architecture(arm),
        "provenance": p,
        "publication_revision": publication["revision"],
        "publication_prefix": publication["prefix"],
        "model": model,
        "model_filename": f"predictive-{arm}-epoch-006.pt",
        "module_sha256": {n: p["code_sha256"][n] for n in MODULES},
    }


def validate_export(export, expected, initialized):
    import torch

    require(
        set(export) == {"model", "architecture", "provenance"},
        "Model-only predictive schema required",
    )
    require(
        export["architecture"] == expected["architecture"]
        and export["provenance"] == expected["provenance"],
        "Exact trained export provenance required",
    )
    state = export["model"]
    template = initialized.state_dict()
    require(
        set(state) == set(template), "Missing or extra learned tensor; old classifier is forbidden"
    )
    for name, target in template.items():
        value = state[name]
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.shape == target.shape
            and value.dtype == target.dtype
            and bool(torch.isfinite(value).all()),
            "Learned tensor schema or finite values differ",
        )
    schema = {n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in state.items()}
    require(
        digest(canonical(schema)) == expected["provenance"]["model_schema_sha256"],
        "Complete model schema differs",
    )
    require(
        sum(p.numel() for p in initialized.parameters())
        == expected["provenance"]["parameter_count"],
        "Actual complete parameter count differs",
    )
    initialized.load_state_dict(state, strict=True)
    return initialized.eval()


def validate_qualification(receipt, chosen):
    require(
        receipt["status"] == "actual_trained_predictive_public_T4_qualification_accepted"
        and receipt["public_studies"] == 3
        and len(receipt["GPU_names"]) == 2
        and all("T4" in name for name in receipt["GPU_names"])
        and receipt["pinned_Docker"] == DOCKER
        and receipt["submission_csv_absent"] is True
        and receipt["hidden_runtime_acceptance"] is False,
        "Actual trained two-T4 public qualification required",
    )
    rows = [r for r in receipt["models"] if r["arm"] == chosen["arm"]]
    require(len(rows) == 1, "One qualified exact arm required")
    row = rows[0]
    require(
        row["candidate_sha256"] == digest(canonical(chosen))
        and row["trained_export_sha256"] == chosen["model"]["sha256"]
        and row["precision_accepted"] is True
        and row["memory_accepted"] is True,
        "Qualified actual trained export differs",
    )
    measures = row["measurements"]
    require(
        len(measures) == 4
        and {(m["device"], m["study_index"]) for m in measures} == {(0, 0), (0, 2), (1, 1), (1, 0)}
        and all(
            math.isfinite(m["maximum_probability_gap"])
            and 0 <= m["maximum_probability_gap"] <= 0.005
            and all(math.isfinite(m[k]) and m[k] > 0 for k in ("FP16_seconds", "FP32_seconds"))
            and 0 < m["windows"] <= 48
            and m["regions"] > 0
            for m in measures
        )
        and row["FP16_FP32_tolerance"] == 0.005
        and row["cross_device_FP16_tolerance"] == 0.0001
        and math.isfinite(row["cross_device_FP16_gap"])
        and 0 <= row["cross_device_FP16_gap"] <= 0.0001
        and len(row["device_peak_bytes"]) == 2
        and all(0 < n < 13 * 2**30 for n in row["device_peak_bytes"])
        and 0 < row["host_peak_bytes"] < 22 * 2**30,
        "Actual trained public-study precision/memory measurements do not qualify",
    )
    return row


def check_predictions(uids, rows):
    require(
        bool(uids) and len(uids) == len(set(uids)) == len(rows),
        "Exact unique complete test roster required",
    )
    require(
        all(isinstance(uid, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", uid) for uid in uids),
        "Original study UIDs required",
    )
    require(
        all(
            len(row) == 12 and all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in row)
            for row in rows
        ),
        "Exactly twelve finite probabilities per test study required",
    )
