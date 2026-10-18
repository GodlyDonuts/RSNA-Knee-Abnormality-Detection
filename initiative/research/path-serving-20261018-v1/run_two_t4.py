"""Executable two-T4 path inference, not a notebook push or submission tool."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import resource
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
HERE = Path(__file__).resolve().parent


def balanced_lanes(costs):
    """Deterministic longest-estimated-study-first bins; no image selection."""
    if not costs or any(
        type(c) not in (int, float) or not math.isfinite(c) or c <= 0 for c in costs
    ):
        raise ValueError("Positive finite per-study work estimates required")
    lanes = [[] for _ in range(min(2, len(costs)))]
    loads = [0.0] * len(lanes)
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        rank = min(range(len(lanes)), key=lambda r: (loads[r], r))
        lanes[rank].append(index)
        loads[rank] += costs[index]
    assert sorted(i for lane in lanes for i in lane) == list(range(len(costs)))
    return lanes


def rss_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def read_test_plan(root):
    import pandas as pd
    from competition_input import UID

    ids = pd.read_csv(
        root / "test.csv", usecols=["StudyInstanceUID"], dtype=str
    ).StudyInstanceUID.tolist()
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(not isinstance(u, str) or not UID.fullmatch(u) for u in ids)
    ):
        raise ValueError("Original complete safe test.csv identities required")
    frame = pd.read_csv(
        root / "test_series.csv", usecols=["StudyInstanceUID", "SeriesInstanceUID"], dtype=str
    )
    frame = frame.loc[frame.StudyInstanceUID.isin(ids)]
    if (
        frame.SeriesInstanceUID.duplicated().any()
        or set(frame.StudyInstanceUID) != set(ids)
        or any(not isinstance(s, str) or not UID.fullmatch(s) for s in frame.SeriesInstanceUID)
    ):
        raise ValueError("Exact test MRI series identities required")
    groups = {
        uid: group.SeriesInstanceUID.tolist() for uid, group in frame.groupby("StudyInstanceUID")
    }
    costs = []
    for uid in ids:
        # Count paths only. Costs guide hardware placement, never filtering or
        # decoding: rejected/recovery alternatives are conservatively included.
        counts = [len(list((root / "test_series" / uid / s).glob("*.dcm"))) for s in groups[uid]]
        costs.append(max(1, sum(counts) + 16 * min(6, len(counts))))
    return ids, costs


class IndexedProvider:
    def __init__(self, provider, indices):
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(provider) for i in indices)
        ):
            raise ValueError("Nonempty unique original study indices required")
        self.provider, self.indices, self.binding = provider, tuple(indices), provider.binding

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.provider[self.indices[i]]


def lane_worker(options, rank, indices, deadline):
    import torch
    from competition_input import CompetitionPathProvider
    from serving import PARENT, PathInference

    from rsna.competitive.compact_infer import load_model

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    if torch.cuda.device_count() != 2 or "T4" not in torch.cuda.get_device_name(rank):
        raise ValueError("Actual two-T4 target hardware required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(f"cuda:{rank}")
    parent, parent_policy = load_model(
        Path(options["parent"]), options["parent_sha256"], precision="float32", device=device
    )
    if options["parent_sha256"] != PARENT:
        raise ValueError("Same exact N3E16 parent as actual path training required")
    head = PathInference(
        Path(options["head"]),
        options["head_sha256"],
        parent_sha256=PARENT,
        device=device,
        vectorized_scoring=options["vectorized_scoring"],
    )
    provider = CompetitionPathProvider(
        Path(options["competition"]),
        Path(options["work"]) / f"cache-{rank}",
        parent,
        parent_sha256=PARENT,
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Shared kernel inference budget expired during setup")
    result = head.predict_csv(
        IndexedProvider(provider, indices),
        Path(options["work"]) / f"lane-{rank}.csv",
        deadline_seconds=remaining,
    )
    result.update(
        rank=rank,
        GPU=torch.cuda.get_device_name(rank),
        GPU_total_bytes=torch.cuda.get_device_properties(rank).total_memory,
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
        GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),
        CPU_peak_RSS_bytes=rss_peak_bytes(),
        input_statistics=provider.statistics(),
        parent_policy=parent_policy,
    )
    with (Path(options["work"]) / f"lane-{rank}.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


def supervise(children, deadline):
    """Stop only this launcher's owned children on failure or a shared timeout."""
    try:
        while any(p.is_alive() for p in children):
            if time.monotonic() >= deadline:
                raise TimeoutError("Shared inference deadline exceeded")
            if any(p.exitcode not in (None, 0) for p in children):
                raise RuntimeError("One GPU lane failed; no partial submission is published")
            for p in children:
                p.join(timeout=0.25)
        if any(p.exitcode != 0 for p in children):
            raise RuntimeError("A GPU lane did not complete successfully")
    except BaseException:
        for p in children:
            if p.is_alive():
                p.terminate()
        for p in children:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("competition", "work", "output", "parent", "head"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--head-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, default=8 * 3600)
    parser.add_argument("--vectorized-scoring", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 8.5 * 3600:
        raise ValueError("Finite shared inference budget of at most8.5hours required")
    if args.work.exists() or args.output.exists():
        raise FileExistsError("Fresh owned work and final output paths required")
    if args.work.resolve().is_relative_to(
        args.competition.resolve()
    ) or args.output.resolve().is_relative_to(args.competition.resolve()):
        raise ValueError("Owned inference writes must stay outside original competition assets")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONNOUSERSITE"] = "1"
    sys.path[:0] = [
        str(HERE),
        str(HERE.parent / "connected-witness-paths-20261017-v1"),
        str(HERE.parent / "path-serving-vectorization-20261018-v1"),
        str(HERE.parents[2]),
    ]
    # Spawned workers inherit explicit scientific/core module search paths.
    from serving import merge_shards

    ids, costs = read_test_plan(args.competition)
    lanes = balanced_lanes(costs)
    deadline = STARTED + args.max_seconds
    args.work.mkdir(mode=0o700, parents=True)
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with (args.work / "launch.json").open("x") as f:
        json.dump(
            {
                "studies": len(ids),
                "lanes": len(lanes),
                "estimated_lane_costs": [sum(costs[i] for i in lane) for lane in lanes],
                "parent_sha256": args.parent_sha256,
                "head_sha256": args.head_sha256,
                "shared_seconds": args.max_seconds,
                "vectorized_scoring": args.vectorized_scoring,
                "no_training_or_competition_entry": True,
            },
            f,
            indent=2,
        )
    context = multiprocessing.get_context("spawn")
    children = [
        context.Process(target=lane_worker, args=(options, rank, lane, deadline))
        for rank, lane in enumerate(lanes)
    ]
    try:
        for p in children:
            p.start()
        supervise(children, deadline)
    except BaseException:
        for p in children:
            if p.pid is not None and p.is_alive():
                p.terminate()
                p.join(timeout=5)
        raise
    receipts = [
        json.loads((args.work / f"lane-{rank}.json").read_text()) for rank in range(len(lanes))
    ]
    if sum(r["CPU_peak_RSS_bytes"] for r in receipts) + rss_peak_bytes() + 2 * 2**30 > 29 * 2**30:
        raise MemoryError("Owned process RSS upper bound does not preserve2GiB system RAM headroom")
    if time.monotonic() >= deadline:
        raise TimeoutError("No final CSV publication after shared deadline")
    merged = merge_shards(
        [args.work / f"lane-{rank}.csv" for rank in range(len(lanes))], ids, args.output
    )
    result = {
        "status": "actual_new_path_twoT4_inference_complete",
        **merged,
        "elapsed_launcher_seconds": time.monotonic() - STARTED,
        "GPU_lanes": receipts,
        "parent_CPU_peak_RSS_bytes": rss_peak_bytes(),
        "full_notebook_wall_clock_qualification_claim": False,
        "clinical_or_Kaggle_score_claim": False,
    }
    with (args.work / "complete.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        json.dumps({k: v for k, v in result.items() if k != "GPU_lanes"}, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
