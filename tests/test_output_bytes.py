"""Exact streaming bounds and redirect isolation for the new CPU input route."""

import hashlib
from types import SimpleNamespace

import pytest
import requests

from scripts.kaggle import output_bytes as m


class Response:
    def __init__(self, body=b"", status=200, location=""):
        self.body, self.status_code = body, status
        self.url = m.API + "download"
        self.headers = {"Location": location}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass


def remote(response):
    return SimpleNamespace(
        session=SimpleNamespace(
            cookies=SimpleNamespace(clear=lambda: None), post=lambda *a, **k: response
        )
    )


def contract(body=b"original"):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def test_complete_original_stream():
    out = m.verify(remote(Response(b"original")), "owner/part0", 1, "part.bin", contract())
    assert out == {**contract(), "full_stream_verified": True}


@pytest.mark.parametrize("body", [b"changed!", b"too large to accept", b"short"])
def test_rejects_any_changed_or_unbounded_bytes(body):
    with pytest.raises(ValueError):
        m.verify(remote(Response(body)), "owner/part0", 1, "part.bin", contract())


def test_rejects_untrusted_redirect_before_get(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Untrusted download"))
    with pytest.raises(ValueError, match="redirect host"):
        m.verify(
            remote(Response(status=302, location="https://evil.test/private")),
            "owner/part0",
            1,
            "part.bin",
            contract(),
        )


def test_external_download_does_not_use_bearer_session(monkeypatch):
    r = remote(Response(status=302, location="https://storage.googleapis.com/bucket/private"))
    monkeypatch.setattr(requests, "get", lambda *a, **k: Response(b"original"))
    assert m.verify(r, "owner/part0", 1, "part.bin", contract())["full_stream_verified"]


def test_network_error_has_no_signed_url(monkeypatch):
    def fail(*a, **k):
        raise requests.ConnectionError("https://storage.googleapis.com/file?secret=private")

    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(ValueError) as e:
        m.verify(
            remote(Response(status=302, location="https://storage.googleapis.com/file")),
            "owner/part0",
            1,
            "part.bin",
            contract(),
        )
    assert str(e.value) == "Original output transport failed"
