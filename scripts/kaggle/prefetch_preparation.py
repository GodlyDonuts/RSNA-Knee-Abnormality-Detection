"""One-chunk-ahead CPU preparation; original inference order/bytes and failures stay fixed."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

if __package__:
    from .parallel_preparation import ParallelPreparation
else:
    from parallel_preparation import ParallelPreparation


class PrefetchPreparation:
    def __init__(
        self, frame, identifiers, spool, *, chunk_size=8, preparation_factory=ParallelPreparation
    ):
        if (
            not identifiers
            or len(set(identifiers)) != len(identifiers)
            or type(chunk_size) is not int
            or chunk_size < 1
        ):
            raise ValueError("Original distinct complete roster and fixed chunk size required")
        self.frame = frame.copy()
        self.identifiers = list(identifiers)
        self.spool = Path(spool)
        if self.spool.exists():
            raise ValueError("Prefetch scratch must be a new private path")
        self.spool.mkdir(mode=0o700)
        self.chunks = [
            self.identifiers[i : i + chunk_size] for i in range(0, len(identifiers), chunk_size)
        ]
        self.preparation = preparation_factory()
        self.thread = ThreadPoolExecutor(max_workers=1)
        self.next = 0
        self.future = None
        self.options = None
        self.completed = 0

    def _frame(self, identifiers):
        return self.frame.loc[self.frame.StudyInstanceUID.isin(identifiers)].copy()

    @staticmethod
    def _cache_bytes(path):
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    def _stage(self, index, options):
        # The current materialized chunk is already charged to free space.
        if shutil.disk_usage(self.spool).free < 6 * 2**30:
            raise OSError("Future3GiB chunk plus3GiB scratch reserve required")
        root = self.spool / f"prefetch-{index:04d}"
        root.mkdir(mode=0o700)
        cache = root / "cache"
        ids = self.chunks[index]
        frame = self._frame(ids)
        upper = sum(
            len(
                list(
                    (
                        Path(options["dicom_root"])
                        / "test_series"
                        / row.StudyInstanceUID
                        / row.SeriesInstanceUID
                    ).glob("*.dcm")
                )
            )
            * (options["cache_side"] ** 2 * 2 + 4096)
            + 65536
            for row in frame.itertuples(index=False)
        )
        if upper > 3 * 2**30:
            raise ValueError("Prefetch chunk upper byte bound exceeds funded3GiB cap")
        selected, receipt = self.preparation.prepare(frame, ids, cache=cache, **options)
        if self._cache_bytes(cache) > 3 * 2**30:
            raise ValueError("Actual prefetched cache exceeds3GiB cap")
        return selected, receipt, cache

    def prepare(self, frame, identifiers, **options):
        if self.next >= len(self.chunks) or identifiers != self.chunks[self.next]:
            raise ValueError("Original inference chunk order cannot change")
        pd.testing.assert_frame_equal(frame, self._frame(identifiers))
        if "cache" not in options:
            raise ValueError("Original destination cache required")
        destination = Path(options["cache"])
        policy = {k: v for k, v in options.items() if k != "cache"}
        if self.options is None:
            self.options = policy
        elif policy != self.options:
            raise ValueError("MRI preparation policy cannot change during prefetch")
        if self.future is None:
            if shutil.disk_usage(self.spool).free < 9 * 2**30:
                raise OSError("Two3GiB chunks plus3GiB scratch reserve required")
            selected, receipt = self.preparation.prepare(frame, identifiers, **options)
            if self._cache_bytes(destination) > 3 * 2**30:
                raise ValueError("Original current chunk exceeds funded3GiB cap")
        else:
            selected, receipt, cache = self.future.result()
            if destination.exists():
                raise ValueError("Original chunk cache destination must be unused")
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            cache.rename(destination)
            cache.parent.rmdir()
            self.completed += 1
        self.next += 1
        self.future = None
        if self.next < len(self.chunks):
            self.future = self.thread.submit(self._stage, self.next, dict(self.options))
        return selected, receipt

    def close(self):
        # No automatic retry or removal of a failed/pending cache; retain evidence.
        self.thread.shutdown(wait=True, cancel_futures=True)
        self.preparation.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
