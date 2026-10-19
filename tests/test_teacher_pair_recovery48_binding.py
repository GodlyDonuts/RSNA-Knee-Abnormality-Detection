"""Check host binding and preserve the actual previously tested numerical engines."""

import ast
from pathlib import Path

OLD = Path("initiative/research/teacher-pair-real-recovery-20261019-v1")
NEW = Path("initiative/research/teacher-pair-real-recovery48-20261019-v2")


def functions(directory, name):
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse((directory / (name + ".py.txt")).read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_actual_recovery_numerical_functions_unchanged():
    for name in ("worker", "driver"):
        before = functions(OLD, name)
        after = functions(NEW, name)
        assert before.keys() == after.keys()
        for key in before.keys() - {"main"}:
            assert before[key] == after[key], (name, key)


def test_all_recovery_wrappers_bind_same_actual_scout_host_and_source():
    for name in ("worker", "driver", "stage_once", "follow_once"):
        s = (NEW / (name + ".py.txt")).read_text()
        compile(s, name, "exec")
        assert "evc26" not in s and "972928" not in s
        assert "evc48" in s
        if name in ("stage_once", "follow_once"):
            assert "973601" in s and "--nodelist=evc48" in s
        if name in ("worker", "stage_once"):
            assert "/tmp/rsna-teacher-pair-gradient-admitted48-20261019-v3" in s
        assert "teacher-pair-optimizer-recovery-20261019-v1" not in s
        if name == "follow_once":
            assert "hf-teacher-pair-recovery48-source-20261019-v1" in s
            assert "teacher-pair-admitted48-fallback-20261019-v1/actual-gradient-terminal.json" in s
