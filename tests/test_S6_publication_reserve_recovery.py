"""Model publication preserves the original funded reserve without phantom files."""

import ast
from pathlib import Path

import pytest

D = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")
source = (D / "publish_epoch.py.txt").read_text()
fn = next(
    n
    for n in ast.parse(source).body
    if isinstance(n, ast.FunctionDef) and n.name == "shared_metadata_budget"
)
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "actual_shared_reserve", "exec"), ns)


def row(free_files, free_kib=24 * 2**20):
    return [
        "/home/sa305415",
        str(1048576000 - free_kib),
        "1048576000",
        "1059061760",
        "-",
        str(1000000 - free_files),
        "1000000",
        "1010000",
        "-",
    ]


def test_actual5392_free_preserves_funded128_and_four_transients():
    result = ns["shared_metadata_budget"](row(5392))
    assert result["shared_reserve_after_liabilities"] == 5260
    assert result["funded_metadata_liability"] == 128 and result["transient_inode_bound"] == 4


@pytest.mark.parametrize("files", [0, 5000, 5128, 5132])
def test_unfunded_or_bare_reserve_is_refused(files):
    with pytest.raises(ValueError):
        ns["shared_metadata_budget"](row(files))


def test_soft_quota_byte_reserve_retained():
    with pytest.raises(ValueError):
        ns["shared_metadata_budget"](row(6000, 20 * 2**20))


def test_recovery_preserves_failed_intent_and_requires_read_only_reconciliation():
    text = (D / "publish_after_preflight_once.py.txt").read_text()
    assert "publish-preflight-recovery-" in text and "974818" in text and "epoch == 1" in text
    assert "publish-1-intent.json" in text and "original_bulk_model_or_HF_mutations" in text
    assert "unlink" not in text and "rmtree" not in text
    compile(text, "explicit_reconciled_recovery", "exec")
