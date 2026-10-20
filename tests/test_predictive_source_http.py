from __future__ import annotations

import hashlib
import io
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
from urllib.parse import unquote

import pytest
import requests

from rsna.competitive.predictive import source_resume as sr
from rsna.competitive.predictive.source_http import SourceHTTP, SourceHTTPError

REVISION = "a" * 40
TOKEN = "private-test-credential"
ROW = {"path": "train_series/1.2.3/1.2.4/1.2.5.dcm", "bytes": 6}


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Body(io.BytesIO):
    def __init__(self, value=b"pixels"):
        super().__init__(value)
        self.reads = 0
        self.decode_content = True

    def read(self, *args):
        self.reads += 1
        return super().read(*args)


class Response:
    def __init__(self, status=200, *, headers=None, body=b"pixels"):
        self.status_code = status
        self.headers = {} if headers is None else headers
        self.raw = Body(body)
        self.closed = 0

    def close(self):
        self.closed += 1


def transport(outcomes, *, clock=None, events=None, **options):
    clock = Clock() if clock is None else clock
    events = [] if events is None else events
    pending, calls = iter(outcomes), []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    http = SourceHTTP(
        revision=REVISION, token=TOKEN, request=request, clock=clock,
        sleep=clock.sleep, wall_clock=lambda: 1_800_000_000,
        emit=events.append, **options,
    )
    return http, calls, clock, events


def test_transient_before_body_retries_preserve_identity_and_close_responses():
    responses = [
        Response(429, headers={"Retry-After": "10"}),
        Response(503),
        Response(200, headers={"Content-Length": "6"}),
    ]
    http, calls, clock, events = transport(responses)
    with http.open_response(ROW) as received:
        assert received.revision == REVISION and received.path == ROW["path"]
        assert received.status == 200
        assert received.stream.decode_content is False
        assert received.stream.read() == b"pixels"
        assert responses[-1].closed == 0
    assert [r.closed for r in responses] == [1, 1, 1]
    assert [r.raw.reads for r in responses] == [0, 0, 1]
    assert clock.sleeps == [10.0, 2.0]
    assert events == [
        {"status": 429, "retry_delay_seconds": 10.0, "attempt": 1},
        {"status": 503, "retry_delay_seconds": 2.0, "attempt": 2},
    ]
    assert len(calls) == 3
    assert all(c[1]["headers"]["Accept-Encoding"] == "identity" for c in calls)
    assert all(c[1]["headers"]["Authorization"] == "Bearer " + TOKEN for c in calls)
    assert all(c[1]["stream"] is True for c in calls)
    assert TOKEN not in repr(events) and ROW["path"] not in repr(events)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_only_designated_statuses_are_transient(status):
    rejected, accepted = Response(status), Response()
    http, calls, _, _ = transport([rejected, accepted])
    with http.open_response(ROW):
        pass
    assert len(calls) == 2
    assert rejected.raw.reads == 0 and rejected.closed == accepted.closed == 1


@pytest.mark.parametrize("status", [401, 403, 404, 400, 301, 416, 206])
def test_permanent_or_unexpected_status_never_reads_or_retries(status):
    response = Response(status)
    http, calls, clock, events = transport([response])
    with pytest.raises(SourceHTTPError, match="response rejected"):
        with http.open_response(ROW):
            pytest.fail("Rejected status exposed a body")
    assert len(calls) == response.closed == 1
    assert response.raw.reads == 0 and not clock.sleeps
    assert events == [{"status": status, "retry_delay_seconds": 0.0, "attempt": 1}]


@pytest.mark.parametrize("error", [
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
])
def test_connection_failures_retry_before_response_only(error):
    accepted = Response()
    http, calls, clock, events = transport([error("secret URL " + TOKEN), accepted])
    with http.open_response(ROW) as response:
        assert response.stream.read() == b"pixels"
    assert len(calls) == 2 and clock.sleeps == [1]
    assert events == [{"status": None, "retry_delay_seconds": 1.0, "attempt": 1}]
    assert accepted.closed == 1


@pytest.mark.parametrize("error", [
    requests.exceptions.SSLError,
    requests.exceptions.InvalidURL,
    RuntimeError,
])
def test_nonretryable_request_errors_are_sanitized(error):
    http, calls, _, _ = transport([error(TOKEN + " " + ROW["path"])])
    try:
        with http.open_response(ROW):
            pytest.fail("Request should fail")
    except SourceHTTPError:
        text = traceback.format_exc()
    else:
        pytest.fail("Expected sanitized error")
    assert len(calls) == 1
    assert TOKEN not in text and ROW["path"] not in text


def test_attempt_exhaustion_closes_every_response_without_body():
    responses = [Response(503) for _ in range(4)]
    http, calls, clock, events = transport(responses)
    with pytest.raises(SourceHTTPError, match="retry budget exhausted"):
        with http.open_response(ROW):
            pass
    assert len(calls) == 4 and clock.sleeps == [1, 2, 4]
    assert all(r.closed == 1 and r.raw.reads == 0 for r in responses)
    assert events[-1] == {"status": 503, "retry_delay_seconds": 0.0, "attempt": 4}


def test_timeout_exhaustion_suppresses_original_secret_traceback():
    http, calls, _, _ = transport(
        [requests.exceptions.ReadTimeout(TOKEN + ROW["path"])], max_attempts=1
    )
    try:
        with http.open_response(ROW):
            pass
    except SourceHTTPError:
        text = traceback.format_exc()
    assert len(calls) == 1 and TOKEN not in text and ROW["path"] not in text


@pytest.mark.parametrize("header,expected_delay", [
    ("15", 15),
    (formatdate(1_800_000_012, usegmt=True), 12),
    (formatdate(1_799_999_990, usegmt=True), 1),
])
def test_retry_after_seconds_or_http_date(header, expected_delay):
    response = Response(429, headers={"rEtRy-AfTeR": header})
    http, _, clock, _ = transport([response, Response()])
    with http.open_response(ROW):
        pass
    assert sum(clock.sleeps) == expected_delay and response.closed == 1


@pytest.mark.parametrize("header", ["180", "99999999", "-1", "nonsense", "x" * 129])
def test_invalid_or_excessive_retry_after_closes_without_waiting(header):
    response = Response(429, headers={"Retry-After": header})
    http, calls, clock, _ = transport([response])
    with pytest.raises(SourceHTTPError, match="Retry-After"):
        with http.open_response(ROW):
            pass
    assert len(calls) == response.closed == 1 and not clock.sleeps
    assert response.raw.reads == 0


def test_response_arriving_after_deadline_is_closed_before_body_yield():
    clock, response = Clock(), Response()

    def request(*_args, **kwargs):
        assert kwargs["timeout"] == (2.5, 2.5)
        clock.now = 6
        return response

    http = SourceHTTP(
        revision=REVISION, token=TOKEN, request=request, clock=clock,
        sleep=clock.sleep, max_elapsed_seconds=5,
    )
    with pytest.raises(SourceHTTPError, match="deadline"):
        with http.open_response(ROW):
            pytest.fail("Late response exposed bytes")
    assert response.closed == 1 and response.raw.reads == 0


@pytest.mark.parametrize("failure", ["body", "consumer"])
def test_no_retry_after_exposing_stream_and_always_close(failure):
    response = Response()
    http, calls, _, _ = transport([response])
    if failure == "body":
        def bad_read(*_args):
            raise requests.exceptions.ReadTimeout(TOKEN + ROW["path"])
        response.raw.read = bad_read
    try:
        with http.open_response(ROW) as received:
            if failure == "body":
                received.stream.read()
            else:
                raise ValueError("bad source metadata " + ROW["path"])
    except SourceHTTPError:
        text = traceback.format_exc()
    assert "no retry" in text
    assert TOKEN not in text and ROW["path"] not in text
    assert len(calls) == 1 and response.closed == 1


def test_suffix_uses_exact_range_and_returns_original_metadata_to_strict_validator():
    headers = {"Content-Range": "bytes 2-5/6", "Content-Length": "4"}
    response = Response(206, headers=headers, body=b"xels")
    http, calls, _, _ = transport([response])
    with http.open_response(ROW, 2) as received:
        assert received.status == 206 and received.headers is headers
        assert received.stream.read() == b"xels"
    assert calls[0][1]["headers"]["Range"] == "bytes=2-"
    assert response.closed == 1
    bad = Response(200)
    http, calls, _, _ = transport([bad])
    with pytest.raises(SourceHTTPError):
        with http.open_response(ROW, 2):
            pass
    assert bad.raw.reads == 0 and len(calls) == bad.closed == 1


def test_global_eight_stream_limit_across_instances_releases_on_context_exit():
    release, eight_started = threading.Event(), threading.Event()
    lock = threading.Lock()
    counts = {"active": 0, "maximum": 0, "requests": 0, "closed": 0}

    def request(*_args, **_kwargs):
        response = Response()
        with lock:
            counts["active"] += 1
            counts["requests"] += 1
            counts["maximum"] = max(counts["maximum"], counts["active"])
            if counts["active"] == 8:
                eight_started.set()
        def close():
            with lock:
                counts["active"] -= 1
                counts["closed"] += 1
        response.close = close
        return response

    clients = [SourceHTTP(revision=REVISION, token=TOKEN, request=request) for _ in range(2)]

    def consume(index):
        with clients[index % 2].open_response(ROW) as response:
            assert release.wait(5)
            return response.stream.read()

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(consume, index) for index in range(20)]
        try:
            assert eight_started.wait(3)
            with lock:
                assert counts["requests"] == counts["active"] == 8
        finally:
            release.set()
        assert [future.result(timeout=3) for future in futures] == [b"pixels"] * 20
    assert counts == {"active": 0, "maximum": 8, "requests": 20, "closed": 20}


def test_default_sessions_reuse_per_thread_without_cross_thread_sharing_and_close(monkeypatch):
    sessions, lock = [], threading.Lock()
    ready = threading.Barrier(2)

    class Session:
        def __init__(self):
            self.calls, self.closed, self.creator = 0, 0, threading.get_ident()
            with lock:
                sessions.append(self)

        def mount(self, prefix, adapter):
            assert prefix == "https://" and adapter.max_retries.total == 0
            adapter.close()

        def get(self, _url, **_kwargs):
            assert threading.get_ident() == self.creator
            self.calls += 1
            return Response()

        def close(self):
            self.closed += 1

    monkeypatch.setattr(requests, "Session", Session)
    http = SourceHTTP(revision=REVISION, token=TOKEN)
    for _ in range(2):
        with http.open_response(ROW):
            pass
    assert len(sessions) == 1 and sessions[0].calls == 2 and sessions[0].closed == 0

    def work():
        with http.open_response(ROW):
            ready.wait(timeout=3)
        with http.open_response(ROW):
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert len(sessions) == 3 and all(session.calls == 2 for session in sessions)
    assert all(session.closed == 1 for session in sessions[1:])
    http.close()
    http.close()
    assert sessions[0].closed == 1


@pytest.mark.parametrize("changes", [
    {"revision": "main"}, {"token": ""}, {"max_attempts": 0}, {"max_attempts": 9},
    {"max_elapsed_seconds": 181}, {"max_elapsed_seconds": float("nan")},
    {"connect_timeout": 0}, {"read_timeout": float("inf")},
])
def test_finite_bounded_configuration(changes):
    args = {"revision": REVISION, "token": TOKEN}
    args.update(changes)
    with pytest.raises(SourceHTTPError):
        SourceHTTP(**args)


def original_case(tmp_path):
    original, destination = tmp_path / "original", tmp_path / "destination"
    original.mkdir(mode=0o700)
    destination.mkdir(mode=0o700)
    uid, series = "1.2.3", "1.2.4"
    csv = ",".join(sr.SERIES_COLUMNS) + f"\n{uid},{series},1,0,sagittal\n"
    (original / "train.csv").write_text("StudyInstanceUID\n" + uid + "\n")
    (original / "train_series.csv").write_text(csv)
    bodies = [b"complete original DICOM", b"partial original DICOM", b"missing original DICOM"]
    rows = []
    for index, body in enumerate(bodies):
        sop = f"1.2.5.{index}"
        row = {
            "study_uid": uid, "series_uid": series, "sop_uid": sop,
            "split": "train_series", "path": f"train_series/{uid}/{series}/{sop}.dcm",
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
        }
        rows.append(row)
        if index < 2:
            path = original / row["path"]
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.write_bytes(body if index == 0 else body[:7])
    plan = sr.reconcile(original, uid=uid, rows=rows, series_csv=csv, revision=REVISION, cap=1 << 20)
    return original, destination, rows, bodies, plan


@pytest.mark.parametrize("bad_range", [False, True])
def test_real_resume_keeps_full_prefix_and_strict_range_hash_checks(tmp_path, bad_range):
    original, destination, rows, bodies, plan = original_case(tmp_path)
    calls, responses = [], []
    clock = Clock()

    def request(url, *, headers, **_kwargs):
        path = unquote(url.split("/resolve/" + REVISION + "/", 1)[1])
        index = next(i for i, row in enumerate(rows) if row["path"] == path)
        offset = int(headers.get("Range", "bytes=0-")[6:-1])
        calls.append((index, offset))
        if len(calls) == 1:
            response = Response(503)
        else:
            body = bodies[index]
            response_headers = {"Content-Length": str(len(body) - offset)}
            if offset:
                start = 0 if bad_range else offset
                response_headers["Content-Range"] = f"bytes {start}-{len(body) - 1}/{len(body)}"
            response = Response(
                206 if offset else 200, headers=response_headers, body=body[offset:]
            )
        responses.append(response)
        return response

    http = SourceHTTP(
        revision=REVISION, token=TOKEN, request=request, clock=clock, sleep=clock.sleep,
    )
    args = dict(
        opener=http.open_response,
        terminal_receipt={"job_id": "123", "state": "TIMEOUT", "sha256": "b" * 64},
        workers=1,
    )
    if bad_range:
        with pytest.raises(SourceHTTPError, match="consumer failed; no retry"):
            sr.resume_missing(destination, plan, **args)
        assert calls == [(1, 7), (1, 7)]
        assert responses[-1].raw.reads == 0
    else:
        result, proof = sr.resume_missing(destination, plan, **args)
        assert calls == [(1, 7), (1, 7), (2, 0)]
        for row, body in zip(rows, bodies, strict=True):
            assert (result / row["path"]).read_bytes() == body
        assert proof["fetched_bytes"] == len(bodies[1]) - 7 + len(bodies[2])
        assert (result / rows[0]["path"]).stat().st_ino == (original / rows[0]["path"]).stat().st_ino
    assert (original / rows[1]["path"]).read_bytes() == bodies[1][:7]
    assert not (original / rows[2]["path"]).exists()
    assert all(response.closed == 1 for response in responses)
