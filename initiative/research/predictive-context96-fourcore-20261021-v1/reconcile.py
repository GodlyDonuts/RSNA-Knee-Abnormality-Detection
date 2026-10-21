"""Dependency-free reader for existing four-core outputs; no new GPU execution."""

import ast
import inspect


def read_paired(raw):
    import io
    import math
    import struct
    import zipfile

    names = {"candidate_FP16", "reference_FP16", "candidate_FP32", "reference_FP32"}
    assert len(raw) < 16384
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert set(z.namelist()) == {k + ".npy" for k in names} and len(z.infolist()) == 4
        assert all(x.file_size < 4096 for x in z.infolist())
        for key in names:
            b = z.read(key + ".npy")
            assert b[:6] == b"\x93NUMPY" and b[6:8] in (b"\x01\x00", b"\x02\x00")
            size = 2 if b[6] == 1 else 4
            n = int.from_bytes(b[8 : 8 + size], "little")
            assert 0 < n < 1024
            offset = 8 + size + n
            header = ast.literal_eval(b[8 + size : offset].decode("latin1").strip())
            assert header == {"descr": "<f4", "fortran_order": False, "shape": (12,)}
            assert len(b) == offset + 48
            values = struct.unpack("<12f", b[offset:])
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            arrays[key] = values
    return arrays


def probability_gap(left, right):
    import struct

    return max(
        abs(struct.unpack("<f", struct.pack("<f", x - y))[0])
        for x, y in zip(left, right, strict=True)
    )


def accept_source(original_function):
    source = inspect.getsource(original_function)
    assert source.count("    import numpy as np\n") == 1
    source = source.replace("    import numpy as np\n", "")
    source = source.replace("    import io\n", "")
    start = source.index("        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:")
    end = source.index("        verified.append(", start)
    source = (
        source[:start]
        + """        arrays = read_paired(raw)
        gap32 = probability_gap(arrays["candidate_FP32"], arrays["reference_FP32"])
        gap16 = probability_gap(arrays["candidate_FP16"], arrays["reference_FP16"])
        assert gap32 <= 1e-5 and gap16 <= 1e-4
        assert gap32 == x["thread_FP32_probability_gap"] and gap16 == x["thread_FP16_probability_gap"]
"""
        + source[end:]
    )
    assert source.count("float(np.mean(list(per_study.values())))") == 1
    source = source.replace(
        "float(np.mean(list(per_study.values())))", "(sum(per_study.values()) / len(per_study))"
    )
    ast.parse(source)
    return source


def program(original_control, original_account):
    body = original_control.derive(original_account)
    body += "\nimport ast\n" + inspect.getsource(read_paired)
    body += "\n" + inspect.getsource(probability_gap)
    body += "\n" + accept_source(original_control.accept_public)
    ast.parse(body)
    return body
