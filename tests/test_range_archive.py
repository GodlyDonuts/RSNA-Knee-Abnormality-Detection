import hashlib

import pytest

from rsna.competitive.range_archive import complete


class Response:
    def __init__(self, raw, first, last, *, status=206, change=False, incomplete=False):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {first}-{last}/{len(raw)}"}
        self.raw = raw[first : last + 1]
        if change:
            self.raw = b"X" + self.raw[1:]
        if incomplete:
            self.raw = self.raw[:-1]
        self.closed = False

    def iter_content(self, size):
        for i in range(0, len(self.raw), 3):
            yield self.raw[i : i + 3]

    def close(self):
        self.closed = True


def test_suffix_retains_prefix_and_bounds_every_request(tmp_path):
    raw = b"some original immutable archive"
    path = tmp_path / "archive"
    path.write_bytes(raw[:9])
    calls = []
    responses = []

    def request(first, last):
        calls.append((first, last))
        r = Response(raw, first, last)
        responses.append(r)
        return r

    result = complete(
        path,
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        request,
        tmp_path / "journal",
        chunk_bytes=7,
    )
    assert path.read_bytes() == raw and result["new_suffix_bytes"] == len(raw) - 9
    assert all(last - first + 1 <= 7 + 65536 for first, last in calls) and all(
        r.closed for r in responses
    )


@pytest.mark.parametrize("bad", ["ignored", "overlap", "incomplete"])
def test_wrong_server_or_truncated_suffix_preserves_existing_bytes(tmp_path, bad):
    raw = b"abcdefghijklmno"
    path = tmp_path / "archive"
    path.write_bytes(raw[:5])

    def request(first, last):
        return Response(
            raw,
            first,
            last,
            status=200 if bad == "ignored" else 206,
            change=bad == "overlap",
            incomplete=bad == "incomplete",
        )

    with pytest.raises(ValueError):
        complete(
            path,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            request,
            tmp_path / "journal",
            chunk_bytes=7,
        )
    assert path.read_bytes().startswith(raw[:5])
    if bad != "incomplete":
        assert path.read_bytes() == raw[:5]


def test_complete_original_archive_needs_no_network(tmp_path):
    raw = b"verified original"
    p = tmp_path / "archive"
    p.write_bytes(raw)

    def no_network(*args):
        raise AssertionError("no duplicate download")

    result = complete(
        p, len(raw), hashlib.sha256(raw).hexdigest(), no_network, tmp_path / "journal"
    )
    assert result["new_suffix_bytes"] == 0
