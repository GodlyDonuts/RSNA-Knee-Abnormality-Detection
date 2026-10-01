import hashlib
import json
from pathlib import Path

import pytest

from scripts.newton.launch_mrnet_restored_source_training import admitted_restoration


def original(tmp_path):
    original = (
        Path(__file__).parents[1]
        / "initiative/audits/mrnet-full-source-preparation-v1/restoration/restoration-verified.json"
    )
    receipt = tmp_path / "restoration.json"
    receipt.write_bytes(original.read_bytes())
    row = json.loads(receipt.read_text())
    source = {
        "restoration_path": str(receipt),
        "restoration_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "acquisition_receipt_sha256": row["original_acquisition_receipt_sha256"],
        "input_admission_sha256": row["original_input_sha256"],
        "restored_input_sha256": row["restored_input_sha256"],
    }
    return receipt, source


def test_actual_original_restoration_admits_only_its_owning_node(tmp_path):
    _, source = original(tmp_path)
    result = admitted_restoration(source, "evc24")
    assert result["path"] == "/tmp/rsna-mrnet-restored-0nri0agx"
    with pytest.raises(ValueError, match="owning-node"):
        admitted_restoration(source, "evc32")


@pytest.mark.parametrize(
    "field",
    [
        "original_input_sha256",
        "restored_input_sha256",
        "original_eight_image_label_tensor_records_equal",
    ],
)
def test_a_changed_restoration_cannot_accept_the_frozen_numerical_engine(tmp_path, field):
    receipt, source = original(tmp_path)
    row = json.loads(receipt.read_text())
    row[field] = False if field.endswith("equal") else "0" * 64
    receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="receipt"):
        admitted_restoration(source, "evc24")
    source["restoration_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="tensor-equal"):
        admitted_restoration(source, "evc24")
