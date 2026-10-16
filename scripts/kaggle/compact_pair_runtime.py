"""Evaluate fixed late-average and original E16 on one shared raw-MRI decode.

This research runner cannot write submission.csv. It reuses the original bounded
two-T4 pipeline, qualifies both exact weights against private CPU references, and
retains two separate predictions without reading clinical targets.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_average_infer import load_model as load_average
from rsna.competitive.compact_bag import candidate_windows, window_tensor
from rsna.competitive.compact_infer import load_model as load_epoch
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.infer import write_submission
from rsna.schema import TARGETS

if __package__:
    from . import compact_runtime as original
    from .compact_t4_probe import compare, validate_fixture
    from .resident_pool import ResidentPool
else:
    import compact_runtime as original
    from compact_t4_probe import compare, validate_fixture
    from resident_pool import ResidentPool


class PairWorker:
    def __init__(self, request: dict, index: int):
        self.identity = {
            k: v for k, v in request.items() if k not in {"identifiers", "series", "cache"}
        }
        if (
            request["precision"] != "float16"
            or request["aggregation"] != "full"
            or request["allow_diagnostic"] is not False
            or request["epoch_sha256"] != request["source_shas"][-1]
        ):
            raise ValueError("Fixed average versus its original final epoch required")
        torch.set_num_threads(1)
        torch.cuda.set_device(index)
        self.device = torch.device(f"cuda:{index}")
        if "T4" not in torch.cuda.get_device_name(index):
            raise ValueError("Actual target T4 required")
        self.average, average_scope = load_average(
            Path(request["checkpoint"]),
            request["checkpoint_sha256"],
            request["source_shas"],
            precision="float32",
            device=self.device,
        )
        self.epoch, epoch_scope = load_epoch(
            Path(request["epoch_checkpoint"]),
            request["epoch_sha256"],
            precision="float32",
            device=self.device,
        )
        if not average_scope["competition_eligible"] or epoch_scope["epoch"] != 16:
            raise ValueError("Only complete original fitted scope may be compared")
        fixture_path = Path(request["fixture"])
        if file_sha256(fixture_path) != request["fixture_sha256"]:
            raise ValueError("Original anonymous MRI fixture changed")
        fixture = torch.load(fixture_path, weights_only=True, map_location="cpu")
        count = validate_fixture(fixture)
        windows = fixture["windows"].to(self.device)
        qualification = {}
        with torch.inference_mode():
            for role, model, reference_name, digest in [
                ("average", self.average, "average_reference", request["checkpoint_sha256"]),
                ("epoch16", self.epoch, "epoch_reference", request["epoch_sha256"]),
            ]:
                path = Path(request[reference_name])
                if file_sha256(path) != request[reference_name + "_sha256"]:
                    raise ValueError("Original CPU reference changed")
                reference = json.loads(path.read_text())
                if (
                    reference["checkpoint_sha256"] != digest
                    or reference["fixture_sha256"] != request["fixture_sha256"]
                ):
                    raise ValueError("CPU reference and full weight/fixture identities differ")
                errors = {}
                for precision, tolerance in [("float32", 0.0003), ("float16", 0.003)]:
                    model.precision = precision
                    values = model(windows, chunk_size=12).sigmoid()
                    torch.cuda.synchronize(index)
                    errors[precision] = compare(values, reference["full"], tolerance)
                qualification[role] = {
                    "checkpoint_sha256": digest,
                    "maximum_probability_errors": errors,
                }
        del windows, fixture
        torch.cuda.reset_peak_memory_stats(index)
        self.index = index
        self.qualification = {
            "device_index": index,
            "device": torch.cuda.get_device_name(index),
            "fixture_windows": count,
            "models": qualification,
        }

    def run(self, request: dict) -> dict:
        if {k: request[k] for k in self.identity} != self.identity:
            raise ValueError("One immutable pair and precision policy per worker required")
        ids = request["identifiers"]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Distinct paired study roster required")
        data = CompetitiveStudyDataset(
            pd.DataFrame({"study_uid": ids}),
            pd.read_csv(request["series"]),
            cache=Path(request["cache"]),
            slices=16,
            side=384,
            max_series=6,
            augment=False,
            raw_split="test_series",
            slice_sampling="uniform",
            series_sampling="selected",
        )
        primary, secondary = {}, {}
        insufficient = encoded = 0
        with torch.inference_mode():
            for index in range(len(data)):
                sample = data[index]
                uid = sample["study_uid"]
                if uid not in ids or uid in primary:
                    raise ValueError("Unexpected or repeated paired MRI study")
                if not any(
                    bool(active) and int(mask.sum()) >= 3
                    for active, mask in zip(sample["series_mask"], sample["slice_mask"])
                ):
                    primary[uid] = secondary[uid] = [0.5] * 12
                    insufficient += 1
                    continue
                candidates = candidate_windows(sample["series_mask"], sample["slice_mask"])
                windows = window_tensor(sample["images"], candidates).to(self.device)
                primary[uid] = self.average(windows, chunk_size=12).sigmoid().float().cpu().tolist()
                secondary[uid] = self.epoch(windows, chunk_size=12).sigmoid().float().cpu().tolist()
                encoded += len(candidates)
        original.frame_from_rows(primary, ids)
        original.frame_from_rows(secondary, ids)
        return {
            "rows": primary,
            "secondary_rows": secondary,
            "model_load_count": 1,
            "device_index": self.index,
            "checkpoint_sha256": request["checkpoint_sha256"],
            "epoch_sha256": request["epoch_sha256"],
            "insufficient_window_studies": insufficient,
            "encoded_windows": encoded,
            "peak_GPU_bytes": torch.cuda.max_memory_allocated(self.index),
            "qualification": self.qualification,
        }


class PairPool(ResidentPool):
    def __init__(self, *, timeout_seconds, pair):
        self.pair = dict(pair)
        self.secondary = {}
        self.qualifications = {}

        def start(command, **options):
            command = list(command)
            if Path(command[2]).name != "resident_worker.py":
                raise ValueError("Original persistent worker protocol changed")
            command[2] = str(Path(__file__).resolve())
            command.insert(3, "--worker")
            return subprocess.Popen(command, **options)

        super().__init__(workers=2, timeout_seconds=timeout_seconds, process_factory=start)

    def exchange(self, commands):
        replies = super().exchange([{**c, **self.pair} for c in commands])
        for command, reply in zip(commands, replies):
            if reply.get("status") != "passed":
                continue
            result = reply["result"]
            if result["epoch_sha256"] != self.pair["epoch_sha256"] or set(result["rows"]) != set(
                command["identifiers"]
            ):
                raise ValueError("Paired worker checkpoint or exact roster differs")
            secondary = result["secondary_rows"]
            original.frame_from_rows(secondary, command["identifiers"])
            if set(secondary) & set(self.secondary):
                raise ValueError("Duplicate secondary prediction")
            self.secondary.update(secondary)
            index = result["device_index"]
            proof = result["qualification"]
            if index in self.qualifications and self.qualifications[index] != proof:
                raise ValueError("Target qualification changed during evaluation")
            self.qualifications[index] = proof
        print(
            json.dumps({"paired_decoded_studies_complete": len(self.secondary)}),
            file=sys.stderr,
            flush=True,
        )
        return replies


def worker(index):
    runner = None
    for line in sys.stdin:
        if len(line.encode()) > 1024**2:
            raise ValueError("Oversized paired request")
        message = json.loads(line)
        if message == {"action": "close"}:
            return
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if runner is None:
                    runner = PairWorker(message["command"], index)
                result = runner.run(message["command"])
            response = {"request_id": message["request_id"], "status": "passed", "result": result}
        except Exception:
            response = {
                "request_id": message.get("request_id"),
                "status": "failed",
                "stderr": traceback.format_exc(),
            }
        print(json.dumps(response), flush=True)


def run(args, *, pool_factory=PairPool, **options):
    outputs = [args.output, args.epoch_output]
    if len(set(outputs)) != 2 or any(p.name == "submission.csv" or p.exists() for p in outputs):
        raise ValueError("Two distinct unused research outputs; no submission.csv")
    if args.allow_diagnostic or args.precision != "float16" or args.aggregation != "full":
        raise ValueError("Complete fitted pair with fixed full FP16 evaluation required")
    if len(args.source_shas) != 3 or args.source_shas[-1] != args.epoch_sha256:
        raise ValueError("Average's third source must be the paired E16")
    pair = {
        name: str(getattr(args, name))
        for name in [
            "epoch_checkpoint",
            "epoch_sha256",
            "fixture",
            "fixture_sha256",
            "average_reference",
            "average_reference_sha256",
            "epoch_reference",
            "epoch_reference_sha256",
        ]
    }
    pair["source_shas"] = args.source_shas
    pools = []

    def factory(**kwargs):
        pool = pool_factory(pair=pair, **kwargs)
        pools.append(pool)
        return pool

    started = time.monotonic()
    result = original.run(args, pool_factory=factory, **options)
    if len(pools) != 1:
        raise ValueError("One resident paired pool required")
    pool = pools[0]
    primary = pd.read_csv(args.output, dtype={"StudyInstanceUID": str})
    ids = primary.StudyInstanceUID.tolist()
    missing = set(ids) - set(pool.secondary)
    if len(missing) != sum(c["unusable_MRI_studies"] for c in result["chunks"]):
        raise ValueError("Secondary missingness must equal original image-unavailable cases")
    if not primary.loc[primary.StudyInstanceUID.isin(missing), list(TARGETS)].eq(0.5).all().all():
        raise ValueError("Only original neutral image fallback may be shared")
    pool.secondary.update({uid: [0.5] * 12 for uid in missing})
    secondary = original.frame_from_rows(pool.secondary, ids)
    template = pd.DataFrame({"StudyInstanceUID": ids, **{t: 0.5 for t in TARGETS}})
    write_submission(secondary, template, args.epoch_output)
    if set(pool.qualifications) != {0, 1}:
        raise ValueError("Actual both-T4 per-model qualification required")
    result.update(
        status="paired_late_average_E16_raw_MRI_complete",
        epoch_sha256=args.epoch_sha256,
        epoch_CSV_sha256=hashlib.sha256(args.epoch_output.read_bytes()).hexdigest(),
        average_source_checkpoint_sha256=args.source_shas,
        qualifications=pool.qualifications,
        elapsed_seconds=time.monotonic() - started,
        one_image_decode_for_both_models=True,
        clinical_targets_or_reports_read=False,
        competition_submission=False,
    )
    args.output.with_suffix(".paired.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device-index", type=int)
    early, unknown = parser.parse_known_args()
    if early.worker:
        if unknown or early.device_index not in (0, 1):
            parser.error("Known paired worker device required")
        worker(early.device_index)
        return
    for name in [
        "checkpoint",
        "epoch-checkpoint",
        "fixture",
        "average-reference",
        "epoch-reference",
        "series",
        "test-studies",
        "dicom-root",
        "scratch",
        "output",
        "epoch-output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in [
        "checkpoint-sha256",
        "epoch-sha256",
        "fixture-sha256",
        "average-reference-sha256",
        "epoch-reference-sha256",
    ]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source-shas", nargs=3, required=True)
    parser.add_argument("--max-seconds", type=int, default=14400)
    args = parser.parse_args()
    args.allow_diagnostic, args.precision, args.aggregation = False, "float16", "full"
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
