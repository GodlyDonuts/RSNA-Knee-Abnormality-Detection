"""Keep the qualified model/input math while exporting the original ensemble CSVs."""

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from rsna.competitive.reference_member import (
    HEADER,
    RAW_EXPORT,
    export_as_member,
    inspect_member,
    tree_sha,
)

PARENT = """calls=[]
class RaptorClassifier: pass
SLOT_SIZES=[26,22,18,12,18]
def main():
    nproc=64
    calls.append('inference')
    out='/kaggle/working/submission.csv'
    return out
"""


def external():
    return (
        HEADER
        + "\n"
        + PARENT.replace(
            "    out=", "".join("    " + x + "\n" for x in RAW_EXPORT.splitlines()) + "    out="
        ).replace("out='/kaggle/working/submission.csv'", "out='/kaggle/working/raptor_sub.csv'")
    )


def paired():
    return (
        PARENT.replace("nproc=64", "nproc=2")
        + """\nfrom pathlib import Path
def _adapted_competition():
    main()
    path=Path('/kaggle/working/submission.csv')
    return str(path)
"""
    )


def test_complete_closure_equal_only_after_exact_output_delta():
    proof = inspect_member(PARENT, external())
    assert proof["whole_module_equal_except_CSV_exports"] is True
    assert proof["parent_module_AST_sha256"] == proof["normalized_R6_member_module_AST_sha256"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("SLOT_SIZES=[26,22,18,12,18]", "SLOT_SIZES=[20,20,20,20,16]"),
        ("calls.append('inference')", "calls.append('different inference')"),
        ("class RaptorClassifier: pass", "class RaptorClassifier: different=True"),
    ],
)
def test_unrelated_numerical_or_class_changes_rejected(old, new):
    with pytest.raises(ValueError, match="Whole original"):
        inspect_member(PARENT, external().replace(old, new))


def test_wrong_script_or_raw_output_contract_rejected():
    with pytest.raises(ValueError):
        inspect_member(PARENT, external().replace("/tmp/raptor_run.py", "/tmp/other.py", 1))
    with pytest.raises(ValueError):
        inspect_member(PARENT, external().replace("index=False", "index=True"))


def test_export_uses_one_inference_keeps_bounded_workers_and_raw_values():
    writes = []

    class Frame:
        def __init__(self, values, columns):
            self.values = values.copy()
            self.columns = columns

        def insert(self, *args):
            assert args == (0, "StudyInstanceUID", ["study"])

        def to_csv(self, path, index):
            writes.append((path, index, self.values.copy()))

    source, proof = export_as_member(PARENT, external(), paired())
    namespace = {
        "pd": SimpleNamespace(DataFrame=Frame),
        "np": np,
        "arm_probs": [np.array([[0.2, 0.8]], np.float32)],
        "LAB": ["a", "b"],
        "test_ids": ["study"],
    }
    exec(compile(source.split("\n", 1)[1], "one_replacement_member", "exec"), namespace)
    assert namespace["_adapted_competition"]() == "/kaggle/working/raptor_sub.csv"
    assert namespace["calls"] == ["inference"]
    assert writes[0][0:2] == ("/kaggle/working/raptor_raw.csv", False)
    np.testing.assert_array_equal(writes[0][2], namespace["arm_probs"][0])
    tree = ast.parse(source.split("\n", 1)[1])
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.literal_eval(main.body[0].value) == 2
    assert proof["new_inference_or_model_calls"] == 0
    old = next(n for n in ast.parse(paired()).body if isinstance(n, ast.ClassDef))
    new = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert tree_sha(old) == tree_sha(new)


def test_reordered_or_extra_qualified_main_statements_refused():
    with pytest.raises(ValueError):
        export_as_member(
            PARENT,
            external(),
            paired().replace(
                "    calls.append('inference')",
                "    calls.append('inference')\n    calls.append('extra')",
            ),
        )


def test_reference_compiler_loads_with_original_schema_in_isolated_package(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    stage = root / "initiative/research/R6-paired-member-replacement-20261018-v1/stage_v2.py.txt"
    node = next(
        n
        for n in ast.parse(stage.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "load"
    )
    loader = ast.get_source_segment(stage.read_text(), node)
    code = (
        """import importlib.util,sys,types
from pathlib import Path
rsna=types.ModuleType('rsna');rsna.__path__=[];sys.modules['rsna']=rsna
"""
        + loader
        + "\n"
        + f"load('rsna.schema',Path({str(root / 'rsna/schema.py')!r}))\nbase=load('own_source_reference',Path({str(root / 'scripts/kaggle/public_reference_script.py')!r}))\nassert len(base.TARGETS)==12 and base.TARGETS[2]=='Medial Meniscus'\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
