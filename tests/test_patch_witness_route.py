import copy
import hashlib
import json

import pytest

from scripts.newton.patch_witness_route import HEADER_CAP, INPUT_CAP, parse_header


def original():
    raw = '{"original": true}\n'
    return {
        "source_node": "evc31",
        "destination_node": "evc48",
        "members": {
            f"cache/train_series/1/{i}.npz": {"bytes": 1, "sha256": "a" * 64} for i in range(370)
        },
        "original_CPU_raw": raw,
        "seed": {"original_CPU_receipt_sha256": hashlib.sha256(raw.encode()).hexdigest()},
    }


def encoded(value):
    return (json.dumps(value) + "\n").encode()


def test_exact_original_header_is_admitted():
    value = original()
    assert parse_header(encoded(value)) == value


@pytest.mark.parametrize(
    "mutation", ["destination", "missing", "traversal", "size", "oversize", "hash", "proof"]
)
def test_changed_or_unfunded_transfer_is_rejected_before_copy(mutation):
    value = copy.deepcopy(original())
    if mutation == "destination":
        value["destination_node"] = "evc34"
    elif mutation == "missing":
        value["members"].pop(next(iter(value["members"])))
    elif mutation == "traversal":
        value["members"]["cache/../outside.npz"] = value["members"].pop(
            next(iter(value["members"]))
        )
    elif mutation == "size":
        next(iter(value["members"].values()))["bytes"] = True
    elif mutation == "oversize":
        next(iter(value["members"].values()))["bytes"] = INPUT_CAP
    elif mutation == "hash":
        next(iter(value["members"].values()))["sha256"] = "bad"
    else:
        value["original_CPU_raw"] += " "
    with pytest.raises(ValueError):
        parse_header(encoded(value))


def test_unterminated_or_oversized_header_rejected():
    for raw in (encoded(original()).rstrip(), b"x" * (HEADER_CAP + 1) + b"\n"):
        with pytest.raises(ValueError):
            parse_header(raw)
