import copy
import hashlib
import io
import itertools
import json
from contextlib import closing
from pathlib import Path

import pytest
import torch

from rsna.competitive.predictive import cache, cases, context_cache, visible_contrast
from rsna.competitive.predictive import visible_contrast_cache as preparation
from rsna.competitive.predictive import visible_contrast_stream as transport

RECIPE = dict(side=16, halo=8, context_cap=2, classes=2)
CAP = 16 * 2**20


@pytest.fixture
def inputs(tmp_path):
    raw_rows, source_rows, destination_rows = [], [], []
    for i in range(2):
        uid = f"synthetic-{i}"
        raw = torch.randint(0, 2000, (2, 96, 96), generator=torch.Generator().manual_seed(40+i), dtype=torch.int32)
        group = cache.SourceGroup(raw, torch.tensor([12, 12]), torch.zeros(2, dtype=torch.bool),
                                  torch.zeros(2, dtype=torch.bool), torch.tensor([0, 1]),
                                  torch.tensor([0, 1]), torch.tensor([0, 4]))
        path = tmp_path / f"raw-{i}.npz"
        proof = cache.write(path, [group], {"synthetic": True}, cap_bytes=CAP)
        raw_rows.append(dict(uid=uid, path=str(path), proof=proof))
        groups, _ = cache.load(path, proof)
        source = tmp_path / f"original-context-{i}.npz"
        provenance = {"original_raw_cache_sha256": proof["sha256"], **RECIPE}
        proof = context_cache.write(source, cases.assemble(groups, **RECIPE), provenance, cap_bytes=CAP)
        source_rows.append(dict(uid=uid, path=str(source), proof=proof, provenance=provenance))
        # Simulate an independently verified original context copy on the receiver.
        target = tmp_path / f"receiver-original-{i}.npz"
        target.write_bytes(source.read_bytes())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == proof["sha256"]
        destination_rows.append(dict(uid=uid, path=str(target),
            proof={**proof, "stat": cache.signature(target)}, provenance=provenance))
    pins = preparation.source_pins("d" * 40)
    manifest = {"version": transport.VERSION, **pins, "studies": 2, **RECIPE,
        "raw_terminal_sha256": "a" * 64, "context_terminal_sha256": "b" * 64,
        "raw_index_sha256": "c" * 64, "source_context_index_sha256": "d" * 64,
        "roster_sha256": hashlib.sha256(transport.canonical([r["uid"] for r in raw_rows])).hexdigest(),
        "preparer_sha256": hashlib.sha256(Path(preparation.__file__).read_bytes()).hexdigest()}
    return raw_rows, source_rows, destination_rows, manifest, pins


def units(inputs, tmp_path):
    raw, source, _, manifest, pins = inputs
    return preparation.iter_units(raw, source, expected_count=2, roster_sha256=manifest["roster_sha256"],
        scratch_parent=tmp_path, source=pins, per_study_cap_bytes=CAP, **RECIPE)


def serialized(inputs, tmp_path):
    out = io.BytesIO()
    with closing(units(inputs, tmp_path)) as prepared:
        terminal = transport.send(out, prepared, inputs[1], inputs[3],
                                  per_study_cap_bytes=CAP, total_cap_bytes=2*CAP)
    assert not list(tmp_path.glob("contrast-stream-unit-*"))
    return out.getvalue(), terminal


def receive(body, inputs, destination, **overrides):
    caps = dict(per_study_cap_bytes=CAP, total_cap_bytes=2*CAP, index_cap_bytes=CAP)
    caps.update(overrides)
    return transport.receive(io.BytesIO(body), destination, inputs[2], inputs[3], **caps)


def test_streamed_inputs_have_independent_destination_proofs_and_exact_both_roles(inputs, tmp_path):
    body, terminal = serialized(inputs, tmp_path)
    destination = tmp_path / "receiver"
    result = receive(body, inputs, destination)
    index_path = destination / "context-index.private.json"
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == result["private_index_sha256"]
    rows = json.loads(index_path.read_bytes())
    assert result["studies"] == 2 and result["bytes"] == terminal["bytes"]
    assert result["wire_index_sha256"] == terminal["wire_index_sha256"]
    assert result["all_destination_sizes_SHA_EOF_verified"]
    assert not result["training_or_clinical_acceptance"]
    assert destination.stat().st_mode & 0o777 == 0o700
    for i, row in enumerate(rows):
        assert row["original_control"] == inputs[2][i]
        path = Path(row["path"])
        assert cache.signature(path) == row["proof"]["stat"]
        assert path.stat().st_mode & 0o777 == 0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["proof"]["sha256"]
        groups, _ = cache.load(inputs[0][i]["path"], inputs[0][i]["proof"])
        for role, normalize in (("control", False), ("normalized", True)):
            expected, _ = visible_contrast.assemble(groups, normalize_images=normalize, **RECIPE)
            actual = preparation.load_role(row, role)
            for a, b in zip(actual, expected, strict=True):
                for key in context_cache.KEYS:
                    assert torch.equal(a[key], b[key])
                for key in ("indptr", "regions", "log_prior"):
                    assert torch.equal(getattr(a["supports"], key), getattr(b["supports"], key))


def body_offset(body):
    stream = io.BytesIO(body)
    transport.read_header(stream)
    transport.read_header(stream)
    return stream.tell()


@pytest.mark.parametrize("failure", ["corruption", "truncation", "trailing"])
def test_bad_stream_never_accepts_and_preserves_partial_original_output(inputs, tmp_path, failure):
    body, _ = serialized(inputs, tmp_path)
    offset = body_offset(body)
    if failure == "corruption":
        body = body[:offset+31] + bytes([body[offset+31] ^ 1]) + body[offset+32:]
    elif failure == "truncation":
        body = body[:offset+100]
    else:
        body += b"unexpected"
    destination = tmp_path / "receiver"
    with pytest.raises(ValueError):
        receive(body, inputs, destination)
    assert (destination / "intent.private.json").is_file()
    assert list(destination.glob("normalized-*.npz"))
    assert not (destination / "transfer-accepted.json").exists()
    assert not (destination / "context-index.private.json").exists()
    with pytest.raises(FileExistsError, match="reconciliation"):
        receive(body, inputs, destination)


@pytest.mark.parametrize("cap", ["total_cap_bytes", "index_cap_bytes", "per_study_cap_bytes"])
def test_receiver_refuses_unfunded_body_before_pixel_write(inputs, tmp_path, cap):
    body, _ = serialized(inputs, tmp_path)
    destination = tmp_path / "receiver"
    with pytest.raises(ValueError):
        receive(body, inputs, destination, **{cap: 1})
    assert (destination / "intent.private.json").is_file()
    assert not list(destination.glob("*.npz"))


@pytest.mark.parametrize("failure", ["order", "uid", "raw_source", "candidate"])
def test_receiver_rejects_unmatched_input_header_before_pixels(inputs, tmp_path, failure):
    body, _ = serialized(inputs, tmp_path)
    src, out = io.BytesIO(body), io.BytesIO()
    transport.write_header(out, transport.read_header(src))
    header = transport.read_header(src)
    if failure == "order":
        header["ordinal"] = False  # bool equals0 in Python but is not an ordinal.
    elif failure == "uid":
        header["row"]["uid"] = inputs[2][1]["uid"]
    else:
        key = "original_raw_cache_sha256" if failure == "raw_source" else "candidate_sha256"
        header["row"]["provenance"][key] = "0" * 64
    transport.write_header(out, header)
    out.write(src.read())
    destination = tmp_path / "receiver"
    with pytest.raises(ValueError):
        receive(out.getvalue(), inputs, destination)
    assert not list(destination.glob("*.npz"))


def test_receiver_rejects_wrong_manifest_before_any_destination_intent(inputs, tmp_path):
    body, _ = serialized(inputs, tmp_path)
    other = copy.deepcopy(inputs)
    other[3]["raw_terminal_sha256"] = "0" * 64
    destination = tmp_path / "receiver"
    with pytest.raises(ValueError, match="sender manifest"):
        receive(body, other, destination)
    assert not destination.exists()


def test_incomplete_sender_cannot_emit_complete_terminal_and_scratch_is_closed(inputs, tmp_path):
    with pytest.raises(ValueError, match="Incomplete source cohort"):
        with closing(units(inputs, tmp_path)) as prepared:
            transport.send(io.BytesIO(), itertools.islice(prepared, 1), inputs[1], inputs[3],
                           per_study_cap_bytes=CAP, total_cap_bytes=2*CAP)
    assert not list(tmp_path.glob("contrast-stream-unit-*"))


def test_accepted_destination_is_never_overwritten(inputs, tmp_path):
    body, _ = serialized(inputs, tmp_path)
    destination = tmp_path / "receiver"
    result = receive(body, inputs, destination)
    with pytest.raises(FileExistsError):
        receive(body, inputs, destination)
    assert hashlib.sha256((destination / "context-index.private.json").read_bytes()).hexdigest() == result["private_index_sha256"]
