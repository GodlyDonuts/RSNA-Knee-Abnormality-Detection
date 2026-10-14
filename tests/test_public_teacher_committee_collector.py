import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsna.competitive import committee_evidence as ce
from scripts.newton import collect_public_teacher_committee as collector


def acceptance():
    source = {"source_sha256": "a" * 64, "teacher_node_text_sha256": "b" * 64}
    value = {
        "status": "actual_twenty_teacher_shared_prefix_eight_MRI_bitwise_accepted",
        "studies": 8,
        "members": 20,
        "targets": 12,
        "windows": 10,
        "all_full_vs_shared_fingerprint_window_logits_and_probabilities_bit_exact": True,
        "original_source_sha256": source["source_sha256"],
        "teacher_node_text_sha256": source["teacher_node_text_sha256"],
        "prefix": {"original_inspection_sha256": collector.committee.INSPECTED_PREFIX_DIGEST},
        "labels_reports_or_evaluation_pixels_read": False,
    }
    return value, source


@pytest.mark.parametrize(
    "field,value",
    [
        ("studies", 3),
        ("members", 19),
        ("windows", 9),
        ("all_full_vs_shared_fingerprint_window_logits_and_probabilities_bit_exact", False),
        ("original_source_sha256", "c" * 64),
        ("labels_reports_or_evaluation_pixels_read", True),
    ],
)
def test_collector_refuses_nonaccepted_or_leaking_execution(field, value):
    original, source = acceptance()
    collector.validate_acceptance(original, source)
    changed = copy.deepcopy(original)
    changed[field] = value
    with pytest.raises(ce.CommitteeEvidenceError):
        collector.validate_acceptance(changed, source)


def test_collector_requires_original_prefix_receipt():
    value, source = acceptance()
    value["prefix"]["original_inspection_sha256"] = "c" * 64
    with pytest.raises(ce.CommitteeEvidenceError):
        collector.validate_acceptance(value, source)


def test_funding_preserves_existing_liabilities_and_floor(monkeypatch):
    out = Path("/tmp/private-committee-test/evidence")
    funding = {
        "status": "root_admitted_private_committee_evidence_storage",
        "cap_bytes": ce.MAX_EVIDENCE_BYTES,
        "output": str(out),
        "shared_growth_bytes": 0,
        "retained_obligations_bytes": 100 * 2**30,
        "floor_bytes": 20 * 2**30,
    }
    monkeypatch.setattr(collector.shutil, "disk_usage", lambda p: SimpleNamespace(free=121 * 2**30))
    collector.validate_funding(funding, out)
    monkeypatch.setattr(collector.shutil, "disk_usage", lambda p: SimpleNamespace(free=120 * 2**30))
    with pytest.raises(ce.CommitteeEvidenceError):
        collector.validate_funding(funding, out)
    funding["shared_growth_bytes"] = 1
    with pytest.raises(ce.CommitteeEvidenceError):
        collector.validate_funding(funding, out)
