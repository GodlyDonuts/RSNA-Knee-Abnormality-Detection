"""Bounded native-MRI compact inference with two persistent GPU workers.

Reuses the accepted native decoder/recovery and two-CPU prefetch machinery.
Models load once per GPU; only paths/rosters cross the worker protocol. Full or
preregistered MC128 pooling; neither fitting labels nor reports enter prediction.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from rsna.competitive.compact_bag import candidate_windows, window_tensor
from rsna.competitive.compact_infer import load_model
from rsna.competitive.compact_mc import predict_both
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.infer import write_submission
from rsna.schema import TARGETS

if __package__:
    from .prefetch_entry import live_memory
    from .prefetch_preparation import PrefetchPreparation
    from .resident_pool import ResidentPool
else:
    from prefetch_entry import live_memory  # type: ignore[no-redef]
    from prefetch_preparation import PrefetchPreparation  # type: ignore[no-redef]
    from resident_pool import ResidentPool  # type: ignore[no-redef]


class CompactPool(ResidentPool):
    """Use the existing bounded request/reap protocol with the compact worker."""

    def __init__(self, *, timeout_seconds):
        def start(command, **options):
            command = list(command)
            if Path(command[2]).name != "resident_worker.py":
                raise ValueError("Original persistent-worker launch protocol changed")
            command[2] = str(Path(__file__).resolve())
            command.insert(3, "--worker")
            return subprocess.Popen(command, **options)

        super().__init__(workers=2, timeout_seconds=timeout_seconds, process_factory=start)


def frame_from_rows(rows: dict, expected: list[str]) -> pd.DataFrame:
    if len(expected) != len(set(expected)) or set(rows) != set(expected):
        raise ValueError("Complete distinct test.csv roster required")
    frame = pd.DataFrame(
        [[uid, *rows[uid]] for uid in expected], columns=["StudyInstanceUID", *TARGETS]
    )
    values = torch.tensor(frame[list(TARGETS)].to_numpy(), dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Finite twelve-finding probabilities required")
    return frame


class CompactWorker:
    def __init__(self, request: dict, index: int):
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

    def run(self, request: dict) -> dict:
        if {k: request[k] for k in self.identity} != self.identity:
            raise ValueError("One immutable model/precision/aggregation per worker")
        ids = request["identifiers"]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Distinct worker study roster required")
        series = pd.read_csv(request["series"])
        data = CompetitiveStudyDataset(
            pd.DataFrame({"study_uid": ids}),
            series,
            cache=Path(request["cache"]),
            slices=16,
            side=384,
            max_series=6,
            augment=False,
            raw_split="test_series",
            slice_sampling="uniform",
            series_sampling="selected",
        )
        rows = {}
        insufficient = 0
        encoded = 0
        with torch.inference_mode():
            for sample_index in range(len(data)):
                sample = data[sample_index]
                uid = sample["study_uid"]
                images = sample["images"]
                series_mask = sample["series_mask"]
                slice_mask = sample["slice_mask"]
                if (
                    not isinstance(uid, str)
                    or not isinstance(images, torch.Tensor)
                    or not isinstance(series_mask, torch.Tensor)
                    or not isinstance(slice_mask, torch.Tensor)
                ):
                    raise ValueError("Original native image tensor/identity contract required")
                if not any(
                    bool(active) and int(mask.sum()) >= 3
                    for active, mask in zip(series_mask, slice_mask)
                ):
                    rows[uid] = [0.5] * 12
                    insufficient += 1
                    continue
                candidates = candidate_windows(series_mask, slice_mask)
                windows = window_tensor(images, candidates).to(self.device)
                if request["aggregation"] == "full":
                    probabilities = self.model(windows, chunk_size=12).sigmoid()
                else:
                    probabilities = predict_both(
                        self.model,
                        windows,
                        series_mask,
                        slice_mask,
                        study_uid=uid,
                    )["mc128"]
                rows[uid] = probabilities.float().cpu().tolist()
                encoded += len(candidates)
        frame_from_rows(rows, ids)
        return {
            "rows": rows,
            "model_load_count": 1,
            "device_index": self.index,
            "checkpoint_sha256": self.identity["checkpoint_sha256"],
            "insufficient_window_studies": insufficient,
            "encoded_windows": encoded,
            "peak_GPU_bytes": torch.cuda.max_memory_allocated(self.index),
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
                    runner = CompactWorker(request, index)
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
    pool_factory=CompactPool,
    preparation_factory=PrefetchPreparation,
    memory_reader=live_memory,
):
    if args.allow_diagnostic and args.output.name == "submission.csv":
        raise ValueError("Diagnostic weights cannot write submission.csv")
    if args.output.exists() or args.scratch.exists() or args.scratch.is_symlink():
        raise ValueError("Distinct output and bounded scratch required")
    if args.aggregation not in {"full", "mc128"} or args.precision not in {"float32", "float16"}:
        raise ValueError("Declared compact evaluator policy required")
    if not 0 < args.max_seconds <= 28800:
        raise ValueError("Eight-hour inference budget required")
    roster = pd.read_csv(args.test_studies, dtype={"StudyInstanceUID": str})
    ids = roster.StudyInstanceUID.tolist()
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not isinstance(uid, str) or not uid for uid in ids)
    ):
        raise ValueError("Original complete test.csv identities required")
    series = pd.read_csv(args.series)
    args.scratch.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    rows = {}
    receipts = []
    max_memory = 0
    preparation = preparation_factory(series, ids, args.scratch / "spool", chunk_size=4)
    pool = None
    try:
        pool = pool_factory(timeout_seconds=args.max_seconds)
        for index, offset in enumerate(range(0, len(ids), 4)):
            if time.monotonic() - started >= args.max_seconds:
                raise TimeoutError("Total compact inference budget exhausted")
            chunk_started = time.monotonic()
            chosen = ids[offset : offset + 4]
            root = args.scratch / f"chunk-{index:04d}"
            root.mkdir(mode=0o700)
            cache = root / "cache"
            preparation_started = time.monotonic()
            selected, receipt = preparation.prepare(
                series.loc[series.StudyInstanceUID.isin(chosen)].copy(),
                chosen,
                dicom_root=args.dicom_root,
                cache=cache,
                cache_side=384,
                max_series=6,
                source_revision="kaggle-runtime-test",
                recover_slices=True,
                single_decode=True,
                validate_geometry=True,
                allow_unusable_studies=True,
                series_sampling="selected",
            )
            preparation_wait = time.monotonic() - preparation_started
            unavailable = set(receipt["unusable_study_ids"])
            if not unavailable <= set(chosen):
                raise ValueError("Invalid MRI recovery roster")
            rows.update({uid: [0.5] * 12 for uid in unavailable})
            available = [uid for uid in chosen if uid not in unavailable]
            series_file = root / "series.csv"
            selected.to_csv(series_file, index=False)
            commands = []
            for device in range(min(2, len(available))):
                commands.append(
                    {
                        "identifiers": available[device::2],
                        "series": str(series_file),
                        "cache": str(cache),
                        "checkpoint": str(args.checkpoint),
                        "checkpoint_sha256": args.checkpoint_sha256,
                        "precision": args.precision,
                        "aggregation": args.aggregation,
                        "allow_diagnostic": args.allow_diagnostic,
                    }
                )
            inference_started = time.monotonic()
            worker_receipts = []
            for reply in pool.exchange(commands) if commands else []:
                if reply.get("status") != "passed":
                    raise RuntimeError(reply.get("stderr", "Compact worker failed"))
                result = reply["result"]
                if (
                    result["model_load_count"] != 1
                    or result["checkpoint_sha256"] != args.checkpoint_sha256
                ):
                    raise ValueError("Worker model identity/load count changed")
                if set(result["rows"]) & set(rows):
                    raise ValueError("Duplicate worker study prediction")
                rows.update(result["rows"])
                worker_receipts.append(
                    {
                        key: result[key]
                        for key in [
                            "device_index",
                            "model_load_count",
                            "encoded_windows",
                            "insufficient_window_studies",
                            "peak_GPU_bytes",
                        ]
                    }
                )
            if set(rows) != set(ids[: offset + len(chosen)]):
                raise ValueError("Incomplete chunk prediction roster")
            max_memory = max(
                max_memory, memory_reader(pool, preparation)["summed_lifetime_RSS_bytes"]
            )
            receipts.append(
                {
                    "studies": len(chosen),
                    "unusable_MRI_studies": len(unavailable),
                    "selected_series": receipt["selected_series"],
                    "preparation_wait_seconds": preparation_wait,
                    "inference_wait_seconds": time.monotonic() - inference_started,
                    "total_chunk_seconds": time.monotonic() - chunk_started,
                    "workers": worker_receipts,
                }
            )
            # This is this invocation's decoded cache only; original inputs and
            # checkpoint are read-only and remain outside scratch.
            shutil.rmtree(root)
        frame = frame_from_rows(rows, ids)
        template = pd.DataFrame({"StudyInstanceUID": ids, **{target: 0.5 for target in TARGETS}})
        write_submission(frame, template, args.output)
        result = {
            "status": "compact_native_test_roster_complete",
            "studies": len(ids),
            "columns": 13,
            "checkpoint_sha256": args.checkpoint_sha256,
            "aggregation": args.aggregation,
            "precision": args.precision,
            "diagnostic_only": args.allow_diagnostic,
            "elapsed_seconds": time.monotonic() - started,
            "maximum_summed_lifetime_RSS_bytes": max_memory,
            "chunks": receipts,
            "hidden_scale_runtime_qualified": False,
            "CSV_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "test_roster_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        try:
            preparation.close()
        finally:
            if pool is not None:
                pool.close()


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
    parser.add_argument("--aggregation", choices=["full", "mc128"], required=True)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=28800)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
