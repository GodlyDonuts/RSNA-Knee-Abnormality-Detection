import ast
import copy
import importlib.util
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


u = load("selected_serving_upgrade", HERE / "serving_upgrade.py")
f = load("original_context_fixtures", HERE.parents[2] / "tests/test_predictive_descriptor_cache.py")


def originals():
    return {
        n: (u.PARENT / n).read_text()
        for n in ("worker.py.txt", "descriptor_cache.py", "contract.py", "mounts.py")
    }


def test_only_preparation_and_receipt_are_changed_not_model_gpu_submission_logic():
    old = originals()
    new = u.upgrade(old)
    assert set(new) == set(old)
    assert new["contract.py"] == old["contract.py"]
    assert new["mounts.py"] == old["mounts.py"]
    before, after = ast.parse(old["worker.py.txt"]), ast.parse(new["worker.py.txt"])

    def definitions(tree):
        return {
            n.name: ast.dump(n)
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != "main"
        }

    assert definitions(before) == definitions(after)
    # Original per-GPU complete model loading, precision checks, scheduling and
    # submission logic after preparation remain exactly the same source.
    marker = "    model_records = []\n"
    assert (
        old["worker.py.txt"]
        .split(marker)[1]
        .replace(
            '        "clinical_score": None,',
            '        "clinical_score": None,\n        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",',
        )
        == new["worker.py.txt"].split(marker)[1]
    )
    bad = dict(old, **{"worker.py.txt": old["worker.py.txt"] + "\n"})
    with pytest.raises(ValueError):
        u.upgrade(bad)


def test_actual_generated_helper_compares_all_original_input_bytes_and_records_only_fast_time():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        text = u.upgrade(originals())["descriptor_cache.py"]
        module = types.ModuleType("actual_compiled_selected_helper")
        exec(compile(text, module.__name__, "exec"), module.__dict__)
        groups = f.synthetic_groups(frames_per_group=9)
        original = f.assemble(f.cache.cache_descriptors_once(groups), side=320, context_cap=24)
        qualified, q = module.prepare_contexts(groups, verify_original=True)
        production, p = module.prepare_contexts(groups, verify_original=False)
        f.assert_contexts_equal(original, qualified)
        f.assert_contexts_equal(qualified, production)
        assert (
            q["original_context_bitwise_equal"] is True
            and q["original_render_comparison_seconds"] > 0
        )
        assert (
            p["original_context_bitwise_equal"] is None
            and p["original_render_comparison_seconds"] is None
        )
        assert all(x["render_seconds"] > 0 for x in (q, p))
        changed = copy.deepcopy(qualified)
        changed[0]["targets"][0, 0] = -0.0
        qualified[0]["targets"][0, 0] = 0.0
        with pytest.raises(AssertionError):
            module.equal_contexts(qualified, changed)
    finally:
        torch.set_num_threads(before)


def receipt():
    return {
        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",
        "models": [
            {
                "measurements": [
                    {
                        "study_index": i,
                        "original_context_bitwise_equal": True,
                        "context_equivalence_is_public_only": True,
                        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",
                        "original_render_comparison_seconds": 50.0,
                        "render_seconds": 25.0,
                    }
                    for i in range(3)
                ]
            }
        ],
    }


@pytest.mark.parametrize(
    "kind",
    [
        "version",
        "empty_models",
        "empty_measures",
        "missing_study",
        "missing_equality",
        "missing_reference_timing",
        "missing_optimized_timing",
    ],
)
def test_actual_extra_qualification_required(kind):
    actual = receipt()
    u.require_qualification(actual)
    if kind == "version":
        actual["preparation_version"] = "old"
    elif kind == "empty_models":
        actual["models"] = []
    elif kind == "empty_measures":
        actual["models"][0]["measurements"] = []
    elif kind == "missing_study":
        actual["models"][0]["measurements"].pop()
    else:
        key = {
            "missing_equality": "original_context_bitwise_equal",
            "missing_reference_timing": "original_render_comparison_seconds",
            "missing_optimized_timing": "render_seconds",
        }[kind]
        actual["models"][0]["measurements"][0].pop(key)
    with pytest.raises(ValueError):
        u.require_qualification(actual)
