import ast
import importlib.machinery
import importlib.util
from pathlib import Path

D = Path("initiative/research/selected-window-rendering-20261020-v1")


def module(name):
    loader = importlib.machinery.SourceFileLoader(name, str(D / (name + ".py.txt")))
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(m)
    return m


def test_finalization_runs_zero_images_and_binds_actual_receipt_failure():
    original = (D / "manifest_stage.py.txt").read_text()
    failed = module("continue_manifest").build(original)
    assert failed.splitlines()[161].strip() == '"original_source_git": old["source_git"],'
    source = module("finalize_manifest").build(original, module("continue_manifest").build)
    tree = ast.parse(source)
    loop = next(
        n
        for n in tree.body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"
    )
    assert ast.unparse(loop.iter) == "range(prefix, 3887)"
    assert "prefix = 3887" in source
    assert '"original_source_git": old_intent["source_git"]' in source
    assert '"975384|FAILED|1:0"' in source
    assert 'mmap_mode="r+"' in source and 'mode="w+"' not in source
    assert "assert not np.any(np.all(windows[:prefix] == 0" in source
    assert '"original_full94_tensor_hash_checks": prefix * 2 + len(data.verified_inputs)' in source
    assert '"new_MRI_decode_or_render_calls": 0' in source


def test_finalizer_keeps_original_decode_loop_unchanged():
    original = (D / "manifest_stage.py.txt").read_text()
    source = module("finalize_manifest").build(original, module("continue_manifest").build)

    def loop(s):
        return next(
            n
            for n in ast.parse(s).body
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"
        )

    assert ast.dump(ast.Module(body=[loop(original).body[0]], type_ignores=[])) == ast.dump(
        ast.Module(body=[loop(source).body[0]], type_ignores=[])
    )
