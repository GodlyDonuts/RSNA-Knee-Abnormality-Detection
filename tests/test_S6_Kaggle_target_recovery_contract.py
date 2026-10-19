"""Control-flow acceptance for the bounded alternative GPU route."""

import ast
from pathlib import Path

D = Path("initiative/research/S6-Kaggle-target-recovery-20261019-v1")


def test_sources_compile_and_create_is_only_in_explicit_create_arm():
    for p in D.glob("*.py.txt"):
        compile(p.read_text(), str(p), "exec")
    tree = ast.parse((D / "program.py.txt").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "create_once"
    ]
    assert len(calls) == 1
    arm = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "phase == 'create'"
    )
    assert calls[0] in [n for child in arm.body for n in ast.walk(child)]
    assert calls[0] not in [n for child in arm.orelse for n in ast.walk(child)]


def test_no_competition_mutation_and_exact_source_is_reused():
    source = (D / "program.py.txt").read_text()
    tree = ast.parse(source)
    rpc = [
        n.args[1].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "call"
        and len(n.args) > 1
        and isinstance(n.args[1], ast.Constant)
    ]
    assert set(rpc) == {
        "IntrospectToken",
        "GetDatasetStatus",
        "GetAcceleratorQuotaStatistics",
        "SaveKernel",
    }
    assert "6747a9deaff18122565bd0a6b9b58382be781d42dd9ab8c4f5ca18d066dd1ff5" in source
    assert 'p.reconcile(r, DS, row["files"])' in source
    assert "1800 + 600 + 30000" in source
    assert '"enableInternet": False' in source
    assert '"sessionTimeoutSeconds": 1800' in source
