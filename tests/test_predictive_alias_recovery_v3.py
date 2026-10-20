"""Preserved CRLF metadata is compared byte-for-byte, without normalizing it."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-full-cache-alias-recovery-20261020-v3"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


recon = load("alias_v3_reconcile", CONTROL / "reconcile.py")
contract = load("alias_v3_contract", CONTROL / "contract.py")
fixture = load("original_inventory_fixture", ROOT / "tests/test_predictive_full_cache_continuation.py")


@pytest.mark.parametrize("ending", ["\r\n", "\n"])
def test_original_series_csv_exact_bytes_reuse_without_rewrite(tmp_path, ending):
    original, records, prior, data = fixture.case.__wrapped__(tmp_path)
    records[2]["series_csv"] = records[2]["series_csv"].replace("\n", ending)
    path = data / "train_series.csv"
    expected = records[2]["series_csv"].encode()
    path.write_bytes(expected)
    before = path.stat()
    result = recon.inventory(original, records, prior)
    assert len(result["complete"]) == 2 and result["pending"] == [2]
    assert path.read_bytes() == expected and path.stat() == before
    path.write_bytes(expected + b" ")
    with pytest.raises(ValueError, match="series metadata differs"):
        recon.inventory(original, records, prior)


def test_actual_corrected_inventory_receipt_binds_exact_reconciliation_source():
    p = ROOT / "initiative/research/predictive-alias-failure-20261020-v1/actual-prefix-byte-admission.json"
    body = p.read_bytes()
    actual = contract.prefix_admission(
        body, "6c7cb68e72d0b3822ebc6894c38815e79988a256613770a5b1a5c19b617258ad",
        (CONTROL / "reconcile.py").read_text(),
    )
    assert actual["verified_cases"] == 522 and actual["pending_cases"] == 3365
    with pytest.raises(AssertionError):
        contract.prefix_admission(
            body, contract.digest(body), (CONTROL / "reconcile.py").read_text() + "\n"
        )
    changed = json.loads(body)
    changed["raw_v2_root_exists"] = True
    encoded = json.dumps(changed).encode()
    with pytest.raises(AssertionError):
        contract.prefix_admission(
            encoded, contract.digest(encoded), (CONTROL / "reconcile.py").read_text()
        )
