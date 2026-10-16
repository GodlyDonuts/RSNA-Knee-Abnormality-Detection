"""Separate matched-source-neighbor raw MRI endpoint; no N3 policy substitution.

Reuses the accepted bounded decoder, test.csv roster and persistent-worker
protocol. Actual target-device precision/runtime qualification is still required.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

from rsna.competitive.compact_neighbor_infer import inference_dataset, load_model, prediction_rows
from rsna.competitive.compact_source_neighbors import ENGINE, POLICY

if __package__:
    from .compact_runtime import run as original_run
    from .prefetch_entry import live_memory
    from .prefetch_preparation import PrefetchPreparation
    from .resident_pool import ResidentPool
else:
    from compact_runtime import run as original_run  # type: ignore[no-redef]
    from prefetch_entry import live_memory  # type: ignore[no-redef]
    from prefetch_preparation import PrefetchPreparation  # type: ignore[no-redef]
    from resident_pool import ResidentPool  # type: ignore[no-redef]


class NeighborPool(ResidentPool):
    def __init__(self, *, timeout_seconds):
        self.legacy_edge_fallback_windows = 0

        def start(command, **options):
            command = list(command)
            if Path(command[2]).name != "resident_worker.py":
                raise ValueError("Original persistent-worker launch protocol changed")
            command[2] = str(Path(__file__).resolve())
            command.insert(3, "--worker")
            return subprocess.Popen(command, **options)

        super().__init__(workers=2, timeout_seconds=timeout_seconds, process_factory=start)

    def exchange(self, commands):
        replies = super().exchange(commands)
        for reply in replies:
            if reply.get("status") == "passed":
                result = reply["result"]
                if result.get("engine") != ENGINE or result.get("input_policy") != POLICY:
                    raise ValueError("Wrong source-neighbor worker input/model policy")
                count = result["legacy_edge_fallback_windows"]
                if type(count) is not int or count < 0 or count > result["encoded_windows"]:
                    raise ValueError("Invalid original-edge fallback count")
                self.legacy_edge_fallback_windows += count
        return replies


class NeighborWorker:
    def __init__(self, request: dict, index: int):
        if request["aggregation"] != "full":
            raise ValueError("Matched-source-neighbor evaluator is fixed full-bag")
        self.identity = {
            k: request[k]
            for k in [
                "checkpoint",
                "checkpoint_sha256",
                "precision",
                "aggregation",
                "allow_diagnostic",
            ]
        }
        self.device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(index)
        torch.set_num_threads(1)
        if "T4" not in torch.cuda.get_device_name(index):
            raise ValueError("Actual target T4 required")
        self.model, self.policy = load_model(
            Path(request["checkpoint"]),
            request["checkpoint_sha256"],
            precision=request["precision"],
            device=self.device,
            allow_diagnostic=request["allow_diagnostic"],
        )
        self.index = index

    def run(self, request: dict):
        if {k: request[k] for k in self.identity} != self.identity:
            raise ValueError("One immutable source-neighbor model/policy per worker")
        ids = request["identifiers"]
        data = inference_dataset(ids, pd.read_csv(request["series"]), Path(request["cache"]))
        result = prediction_rows(self.model, data, ids, device=self.device)
        return {
            **result,
            "model_load_count": 1,
            "device_index": self.index,
            "checkpoint_sha256": self.identity["checkpoint_sha256"],
            "peak_GPU_bytes": torch.cuda.max_memory_allocated(self.index),
            "engine": ENGINE,
            "input_policy": dict(POLICY),
        }


def worker(index: int):
    runner = None
    for line in sys.stdin:
        if len(line.encode()) > 1024**2:
            raise ValueError("Oversized worker request")
        message = json.loads(line)
        if message == {"action": "close"}:
            return
        try:
            request = message["command"]
            with contextlib.redirect_stdout(sys.stderr):
                if runner is None:
                    runner = NeighborWorker(request, index)
                result = runner.run(request)
            response = {"request_id": message["request_id"], "status": "passed", "result": result}
        except Exception:
            response = {
                "request_id": message.get("request_id"),
                "status": "failed",
                "stderr": traceback.format_exc(),
            }
        print(json.dumps(response), flush=True)


def run(
    args,
    *,
    pool_factory=NeighborPool,
    preparation_factory=PrefetchPreparation,
    memory_reader=live_memory,
):
    if args.aggregation != "full":
        raise ValueError("Matched-source-neighbor evaluator is fixed full-bag")
    pools = []

    def start_pool(**options):
        pool = pool_factory(**options)
        pools.append(pool)
        return pool

    result = original_run(
        args,
        pool_factory=start_pool,
        preparation_factory=preparation_factory,
        memory_reader=memory_reader,
    )
    if len(pools) != 1:
        raise ValueError("One original persistent pool required")
    result.update(
        status="compact_matched_source_neighbor_test_roster_complete",
        engine=ENGINE,
        input_policy=dict(POLICY),
        legacy_edge_fallback_windows=pools[0].legacy_edge_fallback_windows,
    )
    args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device-index", type=int)
    options, unknown = parser.parse_known_args()
    if options.worker:
        if unknown or options.device_index not in (0, 1):
            parser.error("Known worker device required")
        worker(options.device_index)
        return
    for name in ["checkpoint", "series", "test-studies", "dicom-root", "scratch", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--precision", choices=["float32", "float16"], required=True)
    parser.add_argument("--aggregation", choices=["full"], required=True)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=28800)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
