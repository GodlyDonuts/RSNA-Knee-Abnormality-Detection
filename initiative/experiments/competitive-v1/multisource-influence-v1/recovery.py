"""Change only sampler/control identity around the actual accepted family test."""

import ast
import base64
import hashlib

ACCEPTED_SOURCE = "1e106435c9bbd2bb13462a09f21e55c94d140830b5716094c8b68a0160e3fa4a"
ACCEPTED_CLASS = "3b10769c0f75aef474fcb587daf6e3e2725f2b1cfcccfa9845c08904a6f903a0"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def one(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Missing or ambiguous original substitution")
    return text.replace(before, after)


def assignments(text, values):
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0]
            if isinstance(name, ast.Name) and name.id in values:
                nodes.append((node.lineno, node.end_lineno, name.id))
    if len(nodes) != len(values) or {n for _, _, n in nodes} != set(values):
        raise ValueError("Unique complete literal assignments required")
    lines = text.splitlines(keepends=True)
    for start, end, name in sorted(nodes, reverse=True):
        lines[start - 1 : end] = [name + "=" + repr(values[name]) + "\n"]
    return "".join(lines)


def build(original, fit_source, old_receipt):
    if sha(original) != ACCEPTED_SOURCE:
        raise ValueError("Actual accepted uniform family diagnostic required")
    if not old_receipt.get("same_initial_complete_model_and_allrank_RNG_between_roles"):
        raise ValueError("Original matched uniform/influence recovery required")
    prediction = old_receipt["roles"]["influence"]["prediction_SHA"]
    if not isinstance(prediction, str) or len(prediction) != 64:
        raise ValueError("Actual original influence prediction digest required")
    wanted = {"FILES", "WORKER"}
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(original).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in wanted
    }
    if (
        set(values) != wanted
        or sha(values["FILES"]["multisource/sampled_overlay.py"]) != ACCEPTED_CLASS
    ):
        raise ValueError("Complete actual accepted source class required")
    worker = base64.b64decode(values["WORKER"], validate=True).decode()
    extra = """_fi_spec=importlib.util.spec_from_file_location('family_influence_fit',Path(__file__).parent/'multisource/influence_fit.py')
_fi=importlib.util.module_from_spec(_fi_spec);_fi_spec.loader.exec_module(_fi);fit=_fi.fit_for(_sf)
"""
    worker = one(worker, "fit=_sf.fit\n", extra)
    worker = one(worker, "Recipe(epochs=2,role='uniform')", "Recipe(epochs=2,role='influence')")
    worker = worker.replace(
        "/kaggle/working/family-source-recovery", "/kaggle/working/family-influence-recovery"
    )
    compile(fit_source, "unchanged_engine_influence_fit_adapter", "exec")
    files = {**values["FILES"], "multisource/influence_fit.py": fit_source}
    source = assignments(
        original,
        {
            "FILES": files,
            "WORKER": base64.b64encode(worker.encode()).decode(),
            "CONTROL_PREDICTION_SHA": prediction,
        },
    )
    for before, after in (
        ("/kaggle/working/family-source-recovery", "/kaggle/working/family-influence-recovery"),
        ("old_receipt['roles']['uniform']", "old_receipt['roles']['influence']"),
        ("old/'uniform/full/", "old/'influence/full/"),
        (
            "full['binding']['recipe']['role']=='uniform'",
            "full['binding']['recipe']['role']=='influence'",
        ),
        (
            "actual_family_label_only_uniform16_twoT4_MRI_recovery_accepted",
            "actual_family_label_only_influence16_twoT4_MRI_recovery_accepted",
        ),
        (
            "family-source-real-MRI-recovery-accepted.json",
            "family-influence-real-MRI-recovery-accepted.json",
        ),
    ):
        if before not in source:
            raise ValueError("Expected immutable control marker missing")
        source = source.replace(before, after)
    compile(source, "private_family_influence_MRI_recovery", "exec")
    return source, {
        "source_sha256": sha(source),
        "worker_sha256": sha(worker),
        "accepted_class_sha256": ACCEPTED_CLASS,
        "influence_fit_sha256": sha(fit_source),
        "original_control_prediction_sha256": prediction,
        "sampler": "influence",
        "same_private_target_and_annotation_dataset": 12668561,
        "actual_GPU_or_production_admission": False,
    }
