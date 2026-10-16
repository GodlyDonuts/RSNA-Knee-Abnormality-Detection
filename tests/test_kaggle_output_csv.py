import hashlib
import json

import pytest

from rsna.schema import TARGETS
from scripts.kaggle.output_csv import verify_csv

COLS = ["StudyInstanceUID", *TARGETS]
ROSTER = hashlib.sha256(json.dumps(["b", "a"], separators=(",", ":")).encode()).hexdigest()


def content(a="0.25", b="1.0", ids=("b", "a")):
    return (
        ",".join(COLS)
        + "\n"
        + "\n".join(
            uid + "," + ",".join([value] * 12) for uid, value in zip(ids, [a, b], strict=True)
        )
        + "\n"
    ).encode()


def test_exact_header_roster_hash_and_all_probabilities_without_record_export():
    raw = content()
    result = verify_csv(raw, COLS, ROSTER)
    assert result["studies"] == 2 and result["columns"] == 13
    assert result["CSV_sha256"] == hashlib.sha256(raw).hexdigest()
    assert not result["individual_records_exported"]
    assert "b" not in result.values() and "a" not in result.values()


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1", ""])
def test_invalid_numeric_cells_fail(value):
    with pytest.raises(ValueError):
        verify_csv(content(a=value), COLS, ROSTER)


def test_wrong_roster_header_or_duplicate_identity_fail():
    for raw in [
        content(ids=("a", "b")),
        content(ids=("a", "a")),
        content().replace(b"StudyInstanceUID", b"id"),
        content() + b"extra,0.5\n",
    ]:
        with pytest.raises(ValueError):
            verify_csv(raw, COLS, ROSTER)


class Response:
    def __init__(self, status, data=b"", target=None):
        self.status_code = status
        self.data = data
        self.url = "https://api.kaggle.com/v1/kernels.KernelsApiService/DownloadKernelOutput"
        self.headers = {} if target is None else {"Location": target}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield self.data

    def close(self):
        pass


def session(response):
    from types import SimpleNamespace

    def post(url, **options):
        assert url.startswith("https://api.kaggle.com/")
        assert options["json"] == {
            "ownerSlug": "owner",
            "kernelSlug": "exact-slug",
            "versionNumber": 1,
            "filePath": "submission.csv",
        }
        assert options["stream"] and not options["allow_redirects"]
        return response

    return SimpleNamespace(cookies=SimpleNamespace(clear=lambda: None), post=post)


def test_signed_download_uses_no_authenticated_session(monkeypatch):
    from types import SimpleNamespace

    from scripts.kaggle import output_csv as module

    def get(url, **options):
        assert url == "https://storage.googleapis.com/test-public-output.csv"
        assert "headers" not in options and not options["allow_redirects"]
        return Response(200, content())

    monkeypatch.setattr(module.requests, "get", get)
    remote = SimpleNamespace(
        session=session(
            Response(302, target="https://storage.googleapis.com/test-public-output.csv")
        )
    )
    assert module.download_csv(remote, "owner/exact-slug", 1) == content()


@pytest.mark.parametrize(
    "response",
    [Response(302, target="https://untrusted.example/output"), Response(200, b"x" * (1024**2 + 1))],
)
def test_unrecognized_redirect_or_unbounded_output_is_rejected(response):
    from types import SimpleNamespace

    from scripts.kaggle.output_csv import download_csv

    with pytest.raises(ValueError, match="verification failed"):
        download_csv(SimpleNamespace(session=session(response)), "owner/exact-slug", 1)
