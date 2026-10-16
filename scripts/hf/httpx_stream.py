"""Stream private Hub bytes through the publisher's existing get interface.

The current Hub runtime already uses httpx. This adapter avoids modifying that
shared runtime merely to supply the legacy requests-style iter_content method.
"""

from __future__ import annotations

from contextlib import contextmanager

import httpx


class StreamingResponse:
    def __init__(self, response: httpx.Response):
        self.response = response

    def raise_for_status(self):
        self.response.raise_for_status()

    def iter_content(self, chunk_size: int):
        return self.response.iter_bytes(chunk_size)


@contextmanager
def get(url: str, *, headers: dict, stream: bool, timeout: tuple[float, float]):
    if stream is not True or len(timeout) != 2 or min(timeout) <= 0:
        raise ValueError("Bounded streaming connect/read timeouts required")
    connect, read = timeout
    limits = httpx.Timeout(connect=connect, read=read, write=read, pool=connect)
    # httpx removes authorization on cross-origin redirects. Do not manually
    # reattach Hub credentials to signed CDN locations.
    with httpx.stream(
        "GET", url, headers=headers, follow_redirects=True, timeout=limits
    ) as response:
        yield StreamingResponse(response)
