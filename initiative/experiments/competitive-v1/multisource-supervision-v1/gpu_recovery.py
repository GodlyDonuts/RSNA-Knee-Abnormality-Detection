"""Compose the accepted real-MRI diagnostic with one new label-only adapter.

The licensed original payload stays on its owning private storage. This compiler
cannot acquire data, dispatch a GPU, create a Dataset or submit to competition.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path

OLD_COMPILER = "404d8cc75cdbe8f19a5bd12292b7423b9dc6f1cc60e13e308f06e19691bb1afc"
TARGET = "65d3730702a098c8c819396dba6122902a4f3720dde9b2cb3f9ffd7ac5eca3b5"
ANNOTATIONS = "54f1c94943bd7049b3e3da79fdcffb1f7083ecb373a64b9e2642aa2fe9230df9"
DATASET = "rsna-family-fit-65d3730702a0"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def original_compiler():
    path = Path(__file__).parent.parent / "s6-source-label-contrast-v1/gpu_recovery.py.txt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != OLD_COMPILER:
        raise ValueError("Exact accepted original source-label compiler required")
    loader = importlib.machinery.SourceFileLoader("accepted_label_recovery_compiler", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def literals(text):
    wanted = {"FILES", "WORKER", "TARGETS", "PERMISSION"}
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0]
            if isinstance(name, ast.Name) and name.id in wanted:
                if name.id in result:
                    raise ValueError("Duplicate payload literal")
                result[name.id] = ast.literal_eval(node.value)
    if result.keys() != wanted:
        raise ValueError("Complete original literal closure required")
    return result


HELPER = r"""
import importlib.util
_ms=importlib.util.spec_from_file_location('fixed_family_sampled_fit',Path(__file__).parent/'multisource/sampled_overlay.py')
_sf=importlib.util.module_from_spec(_ms);_ms.loader.exec_module(_sf);fit=_sf.fit

FACADE_LITERAL

def _source_labels(original,role,root):
    facade=_SourceArrayFacade(original)
    roots=[Path('/kaggle/input/rsna-family-fit-65d3730702a0'),Path('/kaggle/input/datasets/godlydonuts/rsna-family-fit-65d3730702a0')]
    paths=[p for p in roots if (p/'targets.private.npz').is_file()];assert len(paths)==1
    source_root=paths[0];raw=(source_root/'targets.private.npz').read_bytes()
    assert len(raw)==493839 and hashlib.sha256(raw).hexdigest()==_sf.PRODUCTION_TARGET
    annotation=(source_root/'annotation-tensor.private.npz').read_bytes()
    assert len(annotation)==343927 and hashlib.sha256(annotation).hexdigest()==_sf.PRODUCTION_ANNOTATIONS
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:values={k:a[k].copy() for k in a.files}
    with np.load(io.BytesIO(annotation),allow_pickle=False) as a:eligible=a['eligible'].copy()
    assert set(values)=={'uids','labels','masks','parent_logits'} and values['labels'].shape==eligible.shape==(3887,12)
    assert hashlib.sha256(json.dumps(values['uids'].tolist(),separators=(',',':')).encode()).hexdigest()==_sf.POPULATION
    lookup={u:i for i,u in enumerate(values['uids'].tolist())};assert len(lookup)==3887 and all(u in lookup for u in original.uids)
    positions=[lookup[u] for u in original.uids]
    mask=torch.from_numpy(values['masks'][positions]);teacher=torch.from_numpy(values['parent_logits'][positions])
    assert torch.equal(mask,facade.mask) and torch.equal(teacher,facade.teacher)
    labels=facade.labels if role=='unchanged_reference' else torch.from_numpy(values['labels'][positions])
    safe=torch.from_numpy(eligible[positions])
    policy={'version':_sf.VERSION,'role':role,'diagnostic_only':True,'parent_checkpoint_sha256':PARENT_SHA,'original_target_file_sha256':_sf.ORIGINAL_FILE,'population':len(facade),'population_sha256':hashlib.sha256(json.dumps(facade.uids,separators=(',',':')).encode()).hexdigest(),'original_labels_sha256':_sf.sha(facade.labels),'labels_sha256':_sf.sha(labels),'mask_sha256':_sf.sha(mask),'teacher_sha256':_sf.sha(teacher),'eligible_sha256':_sf.sha(safe),'target_file_sha256':_sf.ORIGINAL_FILE if role=='unchanged_reference' else _sf.PRODUCTION_TARGET,'annotation_file_sha256':_sf.PRODUCTION_ANNOTATIONS}
    return _sf.ConsensusLabels(facade,labels,safe,policy,diagnostic_only=True)

"""


def build(original, adapter, old_overlay, old_sampled):
    old = original_compiler()
    source, accepted = old.build(original, old_overlay, old_sampled)
    parts = literals(source)
    # Retain the exact previously MRI/GPU-admitted array facade, including the
    # original teacher/image relation. No new image or target extractor here.
    start = old.HELPER.index("class _SourceArrayFacade:")
    end = old.HELPER.index("def _source_labels(", start)
    helper = HELPER.replace("FACADE_LITERAL", old.HELPER[start:end])
    worker = base64.b64decode(parts["WORKER"], validate=True).decode()
    worker = old.change_once(worker, old.HELPER, helper)
    worker = old.change_once(
        worker,
        "parser.add_argument('role',choices=('original_source','released_S6_source'))",
        "parser.add_argument('role',choices=('unchanged_reference','safe_family_mean'))",
    )
    worker = worker.replace(
        "/kaggle/working/s6-source-label-recovery", "/kaggle/working/family-source-recovery"
    )
    compile(worker, "family_label_only_original_MRI_worker", "exec")
    files = {k: v for k, v in parts["FILES"].items() if not k.startswith("s6_labels/")}
    if {k: digest(v) for k, v in files.items()} != accepted["original_module_sha256"]:
        raise ValueError("Original numerical modules differ")
    compile(adapter, "family_sampled_adapter", "exec")
    files["multisource/sampled_overlay.py"] = adapter
    bootstrap = old.BOOTSTRAP
    for before, after in (
        ("s6-source-label-recovery", "family-source-recovery"),
        ("original_source", "unchanged_reference"),
        ("released_S6_source", "safe_family_mean"),
        ("source_label_contrast", "multisource_supervision"),
        ("new_label_array_sha256", "labels_sha256"),
        (
            "actual_S6_label_only_uniform16_twoT4_MRI_recovery_accepted",
            "actual_family_label_only_uniform16_twoT4_MRI_recovery_accepted",
        ),
        (
            "S6-label-only-real-MRI-recovery-accepted.json",
            "family-source-real-MRI-recovery-accepted.json",
        ),
        ("released_S6_target_sha256", "family_target_sha256"),
        (old.TARGET_SHA, TARGET),
    ):
        bootstrap = bootstrap.replace(before, after)
    for key, value in {
        "FILES_VALUE": files,
        "WORKER_VALUE": base64.b64encode(worker.encode()).decode(),
        "TARGETS_VALUE": parts["TARGETS"],
        "PERMISSION_VALUE": parts["PERMISSION"],
        "ORIGINAL_SOURCE_SHA_VALUE": old.ORIGINAL_SHA,
        "CONTROL_PREDICTION_SHA_VALUE": old.CONTROL_PREDICTION_SHA,
    }.items():
        if bootstrap.count(key) != 1:
            raise ValueError("Ambiguous original literal replacement")
        bootstrap = bootstrap.replace(key, repr(value))
    compile(bootstrap, "private_family_source_GPU_recovery", "exec")
    return bootstrap, {
        "source_sha256": digest(bootstrap),
        "worker_sha256": digest(worker),
        "adapter_sha256": digest(adapter),
        "original_module_sha256": accepted["original_module_sha256"],
        "original_recovery_source_sha256": old.ORIGINAL_SHA,
        "original_control_prediction_sha256": old.CONTROL_PREDICTION_SHA,
        "target_sha256": TARGET,
        "annotation_sha256": ANNOTATIONS,
        "fit_studies": 16,
        "prediction_only_studies": 16,
        "epochs": 2,
        "updates_per_full": 4,
        "actual_GPU_training_admitted": False,
    }
