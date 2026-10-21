"""Build an exact preparation-only successor of the frozen two-T4 worker.

No account calls, model loading or owner mutation. This builder is not a
deployment admission. A successor must retain the original final-model
contracts and obtain its own actual two-T4 qualification within funded quota.
"""

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "predictive-final-serving-20261020-v1"
BATCH = HERE.parent / "predictive-descriptor-batch-20261021-v1"
WORKER_SHA = "0b1a218482cddb9fd0ec16ac9ae18c0bf7cd7fa9e03b1e0a5c4e998f4c447525"
REFERENCE_SHA = "a40a3e313922d2577706eb430429675bac8cfe0c52df768fcd8eac0435078560"
BATCH_SHA = "3c2a7785f1aac85929371852731431e0a3940a359abdf510a867b61163d6abd9"
SELECTED_SHA = "d61e89a8bde5d252e0464ef25c149011ba7ab9d92dfeeeedfde9931ff230152b"


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Exact single original serving anchor required")
    return text.replace(old, new)


def runtime_source(reference, batched, selected):
    if (sha(reference), sha(batched), sha(selected)) != (REFERENCE_SHA, BATCH_SHA, SELECTED_SHA):
        raise ValueError("Exact measured preparation sources required")
    header = (
        '"""Exact CPU preparation with original-context comparison during qualification."""\n'
        "import hashlib,time,types\nimport torch\n"
        "from rsna.competitive.predictive.cases import assemble as original_assemble\n"
        f"SOURCES = {dict(reference=reference, batched=batched, selected=selected)!r}\n"
        f"HASHES = {dict(reference=REFERENCE_SHA, batched=BATCH_SHA, selected=SELECTED_SHA)!r}\n"
    )
    body = """
def load(name):
    text = SOURCES[name]
    assert hashlib.sha256(text.encode()).hexdigest() == HASHES[name]
    module = types.ModuleType("accepted_preparation_" + name)
    exec(compile(text, module.__name__, "exec"), module.__dict__)
    return module

reference = load("reference")
batched = load("batched")
selected = load("selected")

def equal_contexts(expected, actual):
    assert len(expected) == len(actual) == 2
    for e, a in zip(expected, actual, strict=True):
        assert set(e) == set(a) == {"images", "token_metadata", "region_metadata", "targets", "observed", "supports"}
        for key in ("images", "token_metadata", "region_metadata", "targets", "observed"):
            x, y = e[key], a[key]
            assert x.dtype == y.dtype and x.shape == y.shape and x.device == y.device == torch.device("cpu")
            assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)), key
        es, ass = e["supports"], a["supports"]
        assert type(es) is type(ass) and es.region_count == ass.region_count
        for key in ("indptr", "regions", "log_prior"):
            x, y = getattr(es, key), getattr(ass, key)
            assert x.dtype == y.dtype and x.shape == y.shape and x.device == y.device == torch.device("cpu")
            assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)), key

def prepare_contexts(groups, *, verify_original):
    assert type(verify_original) is bool
    reference_seconds = None
    if verify_original:
        start = time.monotonic()
        expected = original_assemble(reference.cache_descriptors_once(groups), side=320, halo=8, context_cap=24, classes=12)
        reference_seconds = time.monotonic() - start
    start = time.monotonic()
    actual = selected.assemble(batched.cache_descriptors_once(groups), side=320, halo=8, context_cap=24, classes=12)
    optimized_seconds = time.monotonic() - start
    if verify_original:
        equal_contexts(expected, actual)
        del expected
    return actual, {
        "render_seconds": optimized_seconds,
        "original_render_comparison_seconds": reference_seconds,
        "original_context_bitwise_equal": True if verify_original else None,
        "context_equivalence_is_public_only": verify_original,
        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",
    }
"""
    result = header + body
    ast.parse(result)
    return result


def upgrade(original):
    if (
        sha(original["worker.py.txt"]) != WORKER_SHA
        or sha(original["descriptor_cache.py"]) != REFERENCE_SHA
    ):
        raise ValueError("Only the original accepted worker can be upgraded")
    helper = runtime_source(
        original["descriptor_cache.py"],
        (BATCH / "descriptor_batch.py").read_text(),
        (HERE / "selected_frames.py").read_text(),
    )
    worker = original["worker.py.txt"]
    worker = replace(
        worker,
        "from _predictive_descriptor_cache import cache_descriptors_once",
        "from _predictive_descriptor_cache import prepare_contexts",
    )
    worker = replace(worker, "    from rsna.competitive.predictive.cases import assemble\n", "")
    worker = replace(
        worker,
        "        groups = cache_descriptors_once(groups)\n        contexts = assemble(groups, side=320, halo=8, context_cap=24, classes=12)",
        '        contexts, preparation = prepare_contexts(groups, verify_original=BINDING["mode"] == "qualification")',
    )
    worker = replace(
        worker,
        '            "render_seconds": time.monotonic() - decoded,',
        "            **preparation,",
    )
    worker = replace(
        worker,
        '        "clinical_score": None,',
        '        "clinical_score": None,\n        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",',
    )
    ast.parse(worker)
    return {**original, "descriptor_cache.py": helper, "worker.py.txt": worker}


def require_qualification(receipt):
    """Additional to, never instead of, the original final-model contract."""
    if receipt.get("preparation_version") != "exact_selected_frames_and_batched_descriptors_v1":
        raise ValueError("Actual selected-frame qualification required")
    records = receipt.get("models", [])
    if not records:
        raise ValueError("Actual trained measurements required")
    for row in records:
        measures = row.get("measurements", [])
        if not measures or {m.get("study_index") for m in measures} != {0, 1, 2}:
            raise ValueError("Every original public study must be compared")
        for m in measures:
            if not (
                m.get("original_context_bitwise_equal") is True
                and m.get("context_equivalence_is_public_only") is True
                and m.get("preparation_version") == receipt["preparation_version"]
                and 0 < m.get("original_render_comparison_seconds", 0) < 1200
                and 0 < m.get("render_seconds", 0) < 1200
            ):
                raise ValueError("Complete exact original-context equality and timing required")


def capsule(files):
    """Keep every original admission/account file; transform only compiler facade."""
    original = dict(files)
    # Admit exact frozen worker/helpers before adding a compiler transform.
    upgrade({n: files['serving/' + n] for n in ('worker.py.txt', 'descriptor_cache.py')})
    portable = replace(Path(__file__).read_text(),
                       'BATCH = HERE.parent / "predictive-descriptor-batch-20261021-v1"',
                       'BATCH = HERE')
    files = {**files,
             'serving/preparation_upgrade.py': portable,
             'serving/descriptor_batch.py': (BATCH / 'descriptor_batch.py').read_text(),
             'serving/selected_frames.py': (HERE / 'selected_frames.py').read_text()}
    facade = files['serving/compile.py']
    anchor = 'spec.loader.exec_module(implementation)'
    addition = '''
prep_spec = importlib.util.spec_from_file_location("exact_preparation_upgrade", HERE / "preparation_upgrade.py")
preparation = importlib.util.module_from_spec(prep_spec)
prep_spec.loader.exec_module(preparation)
_originals = implementation.originals
def exact_preparation_originals():
    return preparation.upgrade(_originals())
implementation.originals = exact_preparation_originals
'''
    facade = replace(facade, anchor, anchor + '\n' + addition)
    # Preserve original final-model compiler's complete provenance and actual
    # trained qualification validation. Add the independent preparation proof.
    facade = replace(facade, 'def build(parent,', 'def _original_build(parent,')
    facade += '''
def build(parent, entries, **kwargs):
    if kwargs.get("mode") == "competition":
        preparation.require_qualification(kwargs.get("qualification") or {})
    source, receipt = _original_build(parent, entries, **kwargs)
    receipt["preparation_version"] = "exact_selected_frames_and_batched_descriptors_v1"
    receipt["actual_preparation_CPU_kernel"] = 140041310
    return source, receipt
'''
    ast.parse(facade)
    files['serving/compile.py'] = facade
    assert {n for n in original if original[n] != files[n]} == {'serving/compile.py'}
    return files
