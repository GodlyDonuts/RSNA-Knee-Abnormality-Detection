import httpx
import pytest

from scripts.hf import httpx_stream


def test_complete_chunked_bytes_and_close(monkeypatch):
    payload = bytes(range(251)) * 100
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payload))
    ) as c:
        monkeypatch.setattr(httpx_stream.httpx, "stream", c.stream)
        with httpx_stream.get(
            "https://huggingface.co/file", headers={}, stream=True, timeout=(20, 180)
        ) as response:
            response.raise_for_status()
            chunks = list(response.iter_content(1024))
            assert b"".join(chunks) == payload
            assert max(map(len, chunks)) <= 1024
        assert response.response.is_closed


def test_signed_redirect_never_receives_hub_authorization(monkeypatch):
    seen = []

    def serve(request):
        seen.append((request.url.host, request.headers.get("Authorization")))
        if request.url.host == "huggingface.co":
            return httpx.Response(
                302, headers={"Location": "https://cdn.example/private?signature=fixture"}
            )
        return httpx.Response(200, content=b"exact original bytes")

    with httpx.Client(transport=httpx.MockTransport(serve)) as c:
        monkeypatch.setattr(httpx_stream.httpx, "stream", c.stream)
        with httpx_stream.get(
            "https://huggingface.co/file",
            headers={"Authorization": "Bearer fake-test-only"},
            stream=True,
            timeout=(20, 180),
        ) as response:
            response.raise_for_status()
            assert b"".join(response.iter_content(3)) == b"exact original bytes"
    assert seen == [("huggingface.co", "Bearer fake-test-only"), ("cdn.example", None)]


def test_http_failure_propagates_and_response_closes(monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403))) as c:
        monkeypatch.setattr(httpx_stream.httpx, "stream", c.stream)
        with pytest.raises(httpx.HTTPStatusError):
            with httpx_stream.get(
                "https://huggingface.co/file", headers={}, stream=True, timeout=(20, 180)
            ) as response:
                response.raise_for_status()
        assert response.response.is_closed


@pytest.mark.parametrize("stream,timeout", [(False, (20, 180)), (True, (0, 180)), (True, (20,))])
def test_unbounded_or_nonstreaming_call_refused(stream, timeout):
    with pytest.raises(ValueError):
        with httpx_stream.get(
            "https://huggingface.co/file", headers={}, stream=stream, timeout=timeout
        ):
            pytest.fail("must not contact network")
