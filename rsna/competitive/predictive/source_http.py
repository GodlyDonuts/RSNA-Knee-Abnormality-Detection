"""Bounded pre-body transport for the already admitted immutable MRI source.

Create one SourceHTTP instance for all study lanes. All instances also share an
8-slot process-wide semaphore, held until the response context exits. Only
transient status/connect failures before yielding a body can be retried. No
source selection, cache writes, decoding, or mid-body retries occur here.
"""

from __future__ import annotations

import math
import re
import threading
import time
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests

from .source_resume import SourceResponse

_TRANSIENT = frozenset({429, 500, 502, 503, 504})
_SLOTS = threading.BoundedSemaphore(8)


class SourceHTTPError(RuntimeError):
    """Fixed sanitized failure text; request URLs/credentials are never included."""


class _ThreadSession:
    """Own a connection pool until this file-worker thread exits."""

    def __init__(self):
        self.session = requests.Session()
        self.session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))

    def close(self):
        session, self.session = self.session, None
        if session is not None:
            session.close()

    def __del__(self):
        # threading.local drops its thread dictionary when the worker exits.
        # Cleanup must not expose request/session internals from __del__.
        try:
            self.close()
        except Exception:
            pass


def _finite_positive(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


class SourceHTTP:
    def __init__(
        self,
        *,
        revision,
        token,
        request=None,
        sleep=time.sleep,
        clock=time.monotonic,
        wall_clock=time.time,
        emit=None,
        max_attempts=4,
        max_elapsed_seconds=180,
        connect_timeout=10,
        read_timeout=90,
    ):
        if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
            raise SourceHTTPError("Original immutable source revision is required")
        if not isinstance(token, str) or not token:
            raise SourceHTTPError("Original source credential is required")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 8:
            raise SourceHTTPError("Bounded pre-body attempt count is required")
        if not _finite_positive(max_elapsed_seconds) or max_elapsed_seconds > 180:
            raise SourceHTTPError("Pre-body elapsed budget must be at most 180 seconds")
        if not all(_finite_positive(x) for x in (connect_timeout, read_timeout)):
            raise SourceHTTPError("Finite request timeouts are required")
        self._revision, self._token = revision, token
        self._request = request
        self._thread = threading.local()
        self._sleep, self._clock, self._wall_clock = sleep, clock, wall_clock
        self._emit = (lambda _event: None) if emit is None else emit
        self._max_attempts, self._max_elapsed = max_attempts, float(max_elapsed_seconds)
        self._connect_timeout, self._read_timeout = float(connect_timeout), float(read_timeout)
        self._cooldown_lock = threading.Lock()
        self._cooldown_until = 0.0

    def _default_request(self, url, **kwargs):
        holder = getattr(self._thread, "holder", None)
        if holder is None:
            holder = _ThreadSession()
            self._thread.holder = holder
        return holder.session.get(url, **kwargs)

    def close(self):
        """Close this calling thread's pool; exiting workers close their own."""
        holder = getattr(self._thread, "holder", None)
        if holder is not None:
            try:
                holder.close()
            except Exception:
                raise SourceHTTPError("Source connection cleanup failed") from None
            finally:
                del self._thread.holder

    def _diagnostic(self, status, delay, attempt):
        self._emit({"status": status, "retry_delay_seconds": float(delay), "attempt": attempt})

    def _retry_after(self, headers):
        value = next((v for k, v in headers.items() if str(k).lower() == "retry-after"), None)
        if value is None:
            return None
        text = str(value).strip()
        if len(text) > 128:
            raise SourceHTTPError("Invalid Retry-After header")
        if re.fullmatch(r"[0-9]+", text):
            return float(int(text))
        try:
            parsed = parsedate_to_datetime(text)
            if parsed.tzinfo is None:
                raise ValueError
            delay = parsed.timestamp() - self._wall_clock()
        except (ValueError, OverflowError, TypeError):
            raise SourceHTTPError("Invalid Retry-After header") from None
        if not math.isfinite(delay):
            raise SourceHTTPError("Invalid Retry-After header")
        return max(0.0, delay)

    def _acquire(self, deadline):
        while True:
            now = self._clock()
            remaining = deadline - now
            if remaining <= 0:
                raise SourceHTTPError("Pre-body elapsed budget exhausted")
            with self._cooldown_lock:
                delay = max(0.0, self._cooldown_until - now)
            if delay:
                if delay >= remaining:
                    raise SourceHTTPError("Retry-After exceeds remaining pre-body budget")
                self._sleep(min(delay, 30.0))
                continue
            if not _SLOTS.acquire(timeout=remaining):
                raise SourceHTTPError("Pre-body concurrency wait exhausted")
            with self._cooldown_lock:
                ready = self._clock() >= self._cooldown_until
            if ready and self._clock() < deadline:
                return
            _SLOTS.release()  # A peer established cooldown while we waited.

    @staticmethod
    def _close(response):
        try:
            response.close()
        except Exception:
            raise SourceHTTPError("Source response cleanup failed") from None

    def _schedule(self, *, status, headers, attempt, deadline):
        remaining = deadline - self._clock()
        if attempt >= self._max_attempts or remaining <= 0:
            self._diagnostic(status, 0, attempt)
            raise SourceHTTPError("Pre-body retry budget exhausted")
        try:
            server_delay = self._retry_after(headers)
        except SourceHTTPError:
            self._diagnostic(status, 0, attempt)
            raise
        # A shared deadline avoids 32 per-file workers independently hammering
        # the same source during a server cooldown. It does not alter Torch RNG.
        delay = max(min(2 ** (attempt - 1), 30), server_delay or 0)
        self._diagnostic(status, delay, attempt)
        if delay >= remaining:
            raise SourceHTTPError("Retry-After exceeds remaining pre-body budget")
        with self._cooldown_lock:
            self._cooldown_until = max(self._cooldown_until, self._clock() + delay)

    @contextmanager
    def open_response(self, row, offset=0):
        """Yield SourceResponse for offset0/200 or suffix/206, preserving headers.

        Existing source_resume performs exact Content-Range/length/identity/SHA
        admission; fresh-source consumers retain their original size/SHA checks.
        Request timeouts are finite and clamped to the remaining admission budget.
        The elapsed budget is rechecked after request return and before body yield;
        it is not a deadline for consuming an admitted MRI body.
        """
        path, size = row.get("path"), row.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or any(part in ("", ".", "..") for part in path.split("/"))
            or type(size) is not int
            or size <= 0
            or type(offset) is not int
            or not 0 <= offset < size
        ):
            raise SourceHTTPError("Invalid original source request metadata")
        url = (
            "https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/resolve/"
            + self._revision
            + "/"
            + quote(path, safe="/")
        )
        headers = {"Authorization": "Bearer " + self._token, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        expected = 206 if offset else 200
        deadline = self._clock() + self._max_elapsed
        for attempt in range(1, self._max_attempts + 1):
            self._acquire(deadline)
            response = None
            retry_status, retry_headers = None, {}
            accepted = False
            try:
                remaining = deadline - self._clock()
                timeout = (
                    min(self._connect_timeout, remaining / 2),
                    min(self._read_timeout, remaining / 2),
                )
                try:
                    request = self._default_request if self._request is None else self._request
                    response = request(
                        url, headers=dict(headers), timeout=timeout, stream=True
                    )
                except (requests.exceptions.SSLError, requests.exceptions.InvalidURL):
                    raise SourceHTTPError("Source connection configuration rejected") from None
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    pass  # No response or body escaped; bounded reattempt may be safe.
                except Exception:
                    raise SourceHTTPError("Source request failed before response") from None
                if response is not None:
                    status = response.status_code
                    if type(status) is not int or not 100 <= status <= 599:
                        raise SourceHTTPError("Invalid source HTTP status")
                    if status in _TRANSIENT:
                        retry_status, retry_headers = status, dict(response.headers)
                    elif status != expected:
                        self._diagnostic(status, 0, attempt)
                        raise SourceHTTPError("Source HTTP response rejected")
                    elif self._clock() >= deadline:
                        self._diagnostic(status, 0, attempt)
                        raise SourceHTTPError("Source response arrived after pre-body deadline")
                    else:
                        response.raw.decode_content = False
                        accepted = True
                        try:
                            yield SourceResponse(
                                response.raw, status, response.headers, self._revision, path
                            )
                        except Exception:
                            # Never restart after exposing bytes. Original partials
                            # remain available to the separate reconciliation path.
                            raise SourceHTTPError(
                                "Source body or consumer failed; no retry"
                            ) from None
                        return
            finally:
                try:
                    if response is not None:
                        self._close(response)
                finally:
                    _SLOTS.release()
            if accepted:
                raise SourceHTTPError("Accepted source cannot be retried")
            self._schedule(
                status=retry_status, headers=retry_headers, attempt=attempt, deadline=deadline
            )
        raise SourceHTTPError("Pre-body retry budget exhausted")
