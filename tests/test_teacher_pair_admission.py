"""Exercise actual corrected wrapper programs without a network/GPU mutation."""

import ast
import hashlib
from pathlib import Path

import pytest

D = Path("initiative/research/teacher-pair-full-encoder-20261019-v1")


def test_actual_v2_stage_transformation_preserves_original_and_checks_failed_preintent():
    text = (D / "coordinate_v2_once.py.txt").read_text()
    tree = ast.parse(text)
    start = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "stage" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "payload" for t in n.targets)
    )
    ns = {"D": D, "hashlib": hashlib}
    original = (D / "stage_once.py.txt").read_bytes()
    exec(
        compile(
            ast.Module(body=tree.body[start:end], type_ignores=[]),
            "actual_v2_stage_construction",
            "exec",
        ),
        ns,
    )
    stage = ns["stage"]
    compile(stage, "actual_corrected_stage", "exec")
    assert (D / "stage_once.py.txt").read_bytes() == original
    assert "972885|FAILED|1:0" in stage
    assert "assert not Path('/tmp/rsna-teacher-pair-gradient-20261019-v1').exists()" in stage
    assert 'F = Path("/tmp/rsna-teacher-pair-gradient-20261019-v2")' in stage
    assert stage.index("972885|FAILED|1:0") < stage.index("F.mkdir(") < stage.index('"sbatch"')


@pytest.mark.parametrize(
    "marker",
    [
        '"""Only a package docstring."""\n',
        '"""Only a package docstring."""\nraise RuntimeError()\n',
    ],
)
def test_actual_original_marker_compatibility_guard_accepts_only_docstring(marker, tmp_path):
    tree = ast.parse((D / "coordinate_v2_once.py.txt").read_text())
    start = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "expected" for t in n.targets)
    )
    block = ast.Module(body=tree.body[start : start + 2], type_ignores=[])
    names = ast.literal_eval(block.body[1].iter)
    for name in names:
        p = tmp_path / "rsna/competitive" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(marker if name == "__init__.py" else "unchanged numerical dependency")
    ns = {"R": tmp_path, "hashlib": hashlib}
    if "raise RuntimeError" in marker:
        with pytest.raises(AssertionError):
            exec(compile(block, "actual_dependency_marker_guard", "exec"), ns)
    else:
        exec(compile(block, "actual_dependency_marker_guard", "exec"), ns)
        assert ns["expected"]["rsna/competitive/__init__.py"] == hashlib.sha256(b"").hexdigest()
        assert all(
            sha == hashlib.sha256(b"unchanged numerical dependency").hexdigest()
            for name, sha in ns["expected"].items()
            if not name.endswith("__init__.py")
        )
