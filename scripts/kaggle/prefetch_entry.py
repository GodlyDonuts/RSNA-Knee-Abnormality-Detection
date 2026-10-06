"""Production entry for the measured exact-FP32 two-GPU prefetch pipeline."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

if __package__:
    from . import chunked_infer
    from .prefetch_preparation import PrefetchPreparation
    from .process_memory import LinuxProcess, linux_peak_rss
    from .resident_pool import ResidentPool
else:
    import chunked_infer
    from prefetch_preparation import PrefetchPreparation
    from process_memory import LinuxProcess, linux_peak_rss
    from resident_pool import ResidentPool


def live_memory(pool, preparation):
    """Cover notebook, pipeline, model workers and any preparation/tracker processes."""
    current = LinuxProcess(os.getpid(), Path("/proc"))
    children = current.children(recursive=True)
    residents = {p.pid for p in pool.children}
    preparers = set(preparation.preparation.worker_pids())
    if len(residents) != 2 or len(preparers) not in (0, 2) or residents & preparers:
        raise ValueError("Unexpected production worker topology")
    pids = {child.pid for child in children}
    if not residents | preparers <= pids:
        raise ValueError("Workers disappeared before lifetime memory accounting")
    trackers = []
    for child in children:
        if child.pid not in residents | preparers:
            if "multiprocessing.resource_tracker" not in " ".join(child.cmdline()):
                raise ValueError("Unclassified production descendant")
            trackers.append(child.pid)
    if len(trackers) > 1:
        raise ValueError("Unexpected resource trackers")
    peaks = {pid: linux_peak_rss(pid) for pid in [current.pid, *sorted(pids), os.getppid()]}
    total = sum(peaks.values())
    if total > 29 * 2**30:
        raise ValueError("Notebook and complete workload exceed the RAM envelope")
    return {"process_count": len(peaks), "summed_lifetime_RSS_bytes": total}


def run(
    args,
    *,
    remaining_seconds=28800,
    pool_factory=ResidentPool,
    preparation_factory=PrefetchPreparation,
    memory_reader=live_memory,
):
    if (
        args.precision != "fp32"
        or args.workers != 0
        or args.device != "cuda"
        or args.chunk_size != 8
        or args.baseline_epoch_one
        or args.allow_diagnostic
        or not args.resilient_series
        or not args.recover_slices
        or not args.isolate_study_errors
        or args.test_studies is None
        or not 0 < remaining_seconds <= 28800
    ):
        raise ValueError("Exact FP32/eight-study production pipeline and test.csv roster required")
    started = time.monotonic()
    # Match the original chunk consumer's CSV parsing and index boundary exactly.
    frame = pd.read_csv(args.series)
    identifiers = pd.read_csv(args.test_studies).StudyInstanceUID.tolist()
    args.single_decode = True
    pool = pool_factory(workers=2, timeout_seconds=remaining_seconds)
    preparation = None
    try:
        preparation = preparation_factory(
            frame,
            identifiers,
            args.cache_root.with_name(args.cache_root.name + "-prefetch-spool"),
            chunk_size=8,
        )
        result = chunked_infer.run(args, invoke=pool.invoke, prepare_invoke=preparation.prepare)
        memory = memory_reader(pool, preparation)
        completed = preparation.completed
    finally:
        # A failed future must not strand the model processes.
        try:
            if preparation is not None:
                preparation.close()
        finally:
            pool.close()
    result.update(
        resident_engine="two_device_exact_FP32_prefetch_v1",
        resident_engine_elapsed_seconds=time.monotonic() - started,
        resident_total_budget_seconds=remaining_seconds,
        CPU_preparation_workers=2,
        CPU_threads_per_preparation_worker=1,
        maximum_materialized_chunks=2,
        completed_prefetch_chunks=completed,
        production_memory=memory,
        hidden_test_runtime_validated=False,
    )
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


def main():
    parser = chunked_infer.argument_parser()
    parser.add_argument("--resident-remaining-seconds", type=float, default=28800)
    args = parser.parse_args()
    print(json.dumps(run(args, remaining_seconds=args.resident_remaining_seconds)), flush=True)


if __name__ == "__main__":
    main()
