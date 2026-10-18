"""New exact private input route: reconstruction and corruption rejection."""

import hashlib
import io
import subprocess
import sys
import zipfile

import pytest

from scripts.kaggle import paired_input_chain as m


def payload():
    values = {n: (n + ":original-synthetic-payload").encode() for n in m.NAMES}
    files = {
        n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in values.items()
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as z:
        for n in sorted(values):
            z.writestr(n, values[n])
    return stream.getvalue(), files, values


def execute(source, input_path, working):
    working.mkdir()
    source = source.replace("Path('/kaggle/working')", f"Path({str(working)!r})")
    source = source.replace("Path('/kaggle/input')", f"Path({str(input_path)!r})")
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)


def test_exact_two_process_reconstruction(tmp_path):
    raw, files, values = payload()
    (first, second), info = m.build(raw, files)
    a, b = tmp_path / "first", tmp_path / "second"
    assert execute(first, tmp_path / "unused", a).returncode == 0
    out = execute(second, a, b)
    assert out.returncode == 0, out.stderr
    assert {p.name: p.read_bytes() for p in (b / "paired-correction-input").iterdir()} == values
    assert max(info["source_bytes"]) < m.SOURCE_LIMIT


def test_corrupted_first_piece_never_writes_input(tmp_path):
    raw, files, _ = payload()
    (first, second), _ = m.build(raw, files)
    a, b = tmp_path / "first", tmp_path / "second"
    assert execute(first, tmp_path / "unused", a).returncode == 0
    p = a / "paired-part-000.bin"
    original = p.read_bytes()
    p.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert execute(second, a, b).returncode != 0
    assert not (b / "paired-correction-input").exists()


@pytest.mark.parametrize("kind", ["wrong_hash", "missing_file", "oversized", "duplicate"])
def test_original_contract_rejections(kind):
    raw, files, _ = payload()
    if kind == "wrong_hash":
        files["source-permission.txt"]["sha256"] = "0" * 64
    elif kind == "missing_file":
        files.pop("provenance.json")
    elif kind == "oversized":
        raw = b"x" * m.LIMIT
    else:
        stream = io.BytesIO(raw)
        with zipfile.ZipFile(stream, "a") as z:
            with pytest.warns(UserWarning, match="Duplicate name"):
                z.writestr("provenance.json", b"duplicate")
        raw = stream.getvalue()
    with pytest.raises(ValueError):
        m.build(raw, files)
