"""Resource-only bridge must round trip to exact frozen original sources."""

from pathlib import Path

import pytest
from funding_bridge import adapt

ROOT = Path.cwd() / "initiative/research/finding-stratified-bags-20261018-v1"


@pytest.mark.parametrize(
    "kind,name", [("full", "full_program.py.txt"), ("publication", "publish_epoch_program.py.txt")]
)
def test_only_one_admission_statement_changes(kind, name):
    original = (ROOT / name).read_text()
    changed = adapt(original, kind)
    assert changed != original
    assert "f210549eb1d2aa76a62140ce8bbc9ae03164f1d68542f191357710ee0a00c54a" in changed
    assert "cap['new_family_cap_bytes']-family_paid" in changed
    assert "source_git" in changed
    assert len(changed.splitlines()) == len(original.splitlines()) + 1


@pytest.mark.parametrize(
    "kind,name", [("full", "full_program.py.txt"), ("publication", "publish_epoch_program.py.txt")]
)
def test_modified_original_or_reapplied_bridge_refused(kind, name):
    source = (ROOT / name).read_text()
    for bad in (source + "# change\n", adapt(source, kind)):
        with pytest.raises(ValueError):
            adapt(bad, kind)
