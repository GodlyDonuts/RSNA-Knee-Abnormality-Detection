"""Bounded study-level CPU preparation using the unchanged serial MRI recovery code."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

if __package__:
    from .chunked_infer import prepare_usable_series
else:
    from chunked_infer import prepare_usable_series


def initialize_cpu():
    import os

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import torch

    torch.set_num_threads(1)


def prepare_partition(frame, identifiers, options):
    return prepare_usable_series(frame, identifiers, **options)


class ParallelPreparation:
    def __init__(self, *, workers=2, executor_factory=ProcessPoolExecutor):
        if type(workers) is not int or workers != 2:
            raise ValueError("Exactly two one-thread CPU preparation workers required")
        self.executor = executor_factory(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_cpu,
        )
        self.requests = 0

    def prepare(self, frame, study_ids, **options):
        if len(set(study_ids)) != len(study_ids) or not study_ids:
            raise ValueError("Nonempty distinct original study chunk required")
        # Ambiguous metadata follows the exact original serial path; it is never
        # repaired or split in a way that could change original recovery policy.
        if (
            not frame.index.is_unique
            or not set(frame.StudyInstanceUID) <= set(study_ids)
            or (frame.groupby("SeriesInstanceUID").StudyInstanceUID.nunique() > 1).any()
            or len(study_ids) < 2
        ):
            return prepare_usable_series(frame, study_ids, **options)
        self.requests += 1
        groups = [study_ids[i::2] for i in range(2)]
        futures = [
            self.executor.submit(
                prepare_partition, frame.loc[frame.StudyInstanceUID.isin(ids)].copy(), ids, options
            )
            for ids in groups
        ]
        results = [future.result() for future in futures]
        combined = pd.concat([selected for selected, _ in results])
        selected = combined.loc[frame.index[frame.index.isin(combined.index)]].copy()
        receipts = [receipt for _, receipt in results]
        expected = set(receipts[0])
        if any(set(row) != expected for row in receipts):
            raise ValueError("Original preparation receipt policies differ")
        failures = set(receipts[0]["skipped_series"])
        if any(set(row["skipped_series"]) != failures for row in receipts):
            raise ValueError("Original recovery counters differ")
        receipt = {
            "selected_series": len(selected),
            "skipped_series": {
                key: sum(row["skipped_series"][key] for row in receipts) for key in failures
            },
        }
        if "series_sampling" in expected:
            if any(row["series_sampling"] != "all" for row in receipts):
                raise ValueError("Unknown series policy")
            receipt["series_sampling"] = "all"
        if "unusable_study_ids" in expected:
            unavailable = {uid for row in receipts for uid in row["unusable_study_ids"]}
            receipt["unusable_study_ids"] = [uid for uid in study_ids if uid in unavailable]
        if "source_recovery" in expected:
            order = {uid: index for index, uid in enumerate(study_ids)}
            receipt["source_recovery"] = sorted(
                [item for row in receipts for item in row["source_recovery"]],
                key=lambda item: order[item["study_uid"]],
            )
        if expected != set(receipt):
            raise ValueError("Parallel aggregation lacks an original receipt field")
        return selected, receipt

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
