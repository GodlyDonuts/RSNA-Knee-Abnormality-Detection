"""Backpressure for CPU-decoded MRI producers feeding slower GPU consumers."""

from collections import deque
from typing import Any


def bounded_results(pool, function, items, *, limit=4):
    """Keep at most `limit` submitted decoded results, including pending tasks.

    Unlike Pool.imap_unordered's unbounded result-handler buffer, callers can
    block on their own bounded GPU queue without accumulating the entire cohort.
    Submission order changes only preprocessing delivery, not per-study pixels.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Positive pending-result bound required")
    iterator = iter(items)
    pending: deque[Any] = deque()
    for _ in range(limit):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.append(pool.apply_async(function, (item,)))
    while pending:
        value = pending.popleft().get()
        yield value
        try:
            item = next(iterator)
        except StopIteration:
            continue
        pending.append(pool.apply_async(function, (item,)))
