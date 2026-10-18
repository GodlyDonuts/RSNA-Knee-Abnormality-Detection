"""Regression: cache configuration must precede helper imports and be explicit."""

import ast
from pathlib import Path

import pytest

D = Path("initiative/research/paired-full-fitting-20261018-v1")


def check(source):
    tree = ast.parse(source)
    early = []
    imports = []
    downloads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) == "os.environ.update":
            keys = {k.arg for k in node.keywords}
            if {"HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_HUB_DISABLE_XET"} <= keys:
                early.append(node.lineno)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "exec_module":
            imports.append(node.lineno)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "hf_hub_download":
            downloads.append(node)
    assert early and min(early) < min(imports)
    assert downloads and all("cache_dir" in {k.arg for k in n.keywords} for n in downloads)


def test_new_publication_routes_cache_before_import_and_explicitly():
    check((D / "publish_v2.py.txt").read_text())


def test_original_order_is_rejected():
    with pytest.raises(AssertionError):
        check((D / "publish.py.txt").read_text())
