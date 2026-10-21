"""Label-free large-input engineering on accessible training MRI, CPU only.

Source-size selection is not clinical selection or hidden-data access. Each
child handles one original study/variant and retains no pixels outside RAM.
Parent survives a bounded child failure and records it, never inventing success.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LIMIT = 2400
CHILD_LIMIT = 360
MEMORY_LIMIT = 24 * 2**30


def source_inventory(train, deadline):
    """Rank by count and sampled matrix size, never diagnosis or report."""
    import pydicom

    rows = []
    available = sorted(p for p in train.iterdir() if p.is_dir())
    for study in available:
        if time.monotonic() >= deadline:
            break
        count, estimated, inspected = 0, 0, 0
        for series in sorted(p for p in study.iterdir() if p.is_dir()):
            files = sorted(p for p in series.iterdir() if p.suffix.lower() == ".dcm")
            count += len(files)
            if files:
                try:
                    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
                    estimated += len(files) * int(ds.Rows) * int(ds.Columns) * 4
                    inspected += 1
                except Exception:
                    pass
        rows.append(
            {
                "path": str(study),
                "frames": count,
                "estimated_FP32_pixel_bytes": estimated,
                "sampled_series": inspected,
            }
        )
    return rows, len(available)


def select(rows):
    picked = {}
    for key in ("frames", "estimated_FP32_pixel_bytes"):
        for row in sorted(rows, key=lambda r: (-r[key], r["path"]))[:4]:
            picked[row["path"]] = row
    return sorted(picked.values(), key=lambda r: (-r["estimated_FP32_pixel_bytes"], r["path"]))


def context_digest(value):
    """Complete exact context bytes; digest is retained in RAM, not reported."""
    import torch

    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            assert v.device.type == "cpu"
            h.update(json.dumps([str(v.dtype), list(v.shape)]).encode())
            h.update(v.detach().contiguous().numpy().tobytes())
        elif dataclasses.is_dataclass(v):
            h.update(type(v).__qualname__.encode())
            for f in dataclasses.fields(v):
                h.update(f.name.encode())
                visit(getattr(v, f.name))
        elif isinstance(v, dict):
            for key in sorted(v):
                h.update(key.encode())
                visit(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(json.dumps([type(v).__name__, len(v)]).encode())
            for item in v:
                visit(item)
        else:
            h.update(json.dumps(v, allow_nan=False).encode())

    visit(value)
    return h.hexdigest()


def child(task):
    import resource
    import traceback

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.device_count() == 0
    if task["variant"] == "original":
        from _predictive_descriptor_cache import cache_descriptors_once

        from rsna.competitive.predictive.cases import assemble
        from rsna.competitive.predictive.dicom import load_study
    elif task["variant"] == "successor":
        from descriptor_batch import cache_descriptors_once
        from resilient_decoder import load_study
        from selected_frames import assemble
    else:
        raise ValueError("Two explicit serving variants only")
    start = time.monotonic()
    try:
        path = Path(task["path"])
        groups, stats = load_study(path, path.name)
        decoded = time.monotonic()
        groups = cache_descriptors_once(groups)
        described = time.monotonic()
        contexts = assemble(groups, side=320, halo=8, context_cap=24, classes=12)
        prepared = time.monotonic()
        digest = context_digest(contexts)
        return dict(
            status="complete",
            context_digest_private=digest,
            stats=stats,
            decode_seconds=decoded - start,
            descriptor_seconds=described - decoded,
            assembly_seconds=prepared - described,
            total_seconds=time.monotonic() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
    except Exception as error:
        return dict(
            status="exception",
            error_type=type(error).__name__,
            frames=[
                dict(module=Path(f.filename).name, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)[-5:]
            ],
            total_seconds=time.monotonic() - start,
        )


def bounded_child(task, seconds):
    import psutil

    command = "import benchmark,json,sys;print(json.dumps(benchmark.child(json.load(sys.stdin))))"
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent)},
    )
    process.stdin.write(json.dumps(task))
    process.stdin.close()
    process.stdin = None
    start, peak = time.monotonic(), 0
    failed = None
    while process.poll() is None:
        try:
            peak = max(peak, psutil.Process(process.pid).memory_info().rss)
        except psutil.NoSuchProcess:
            break
        if peak > MEMORY_LIMIT:
            failed = "bounded_child_memory_limit"
        elif time.monotonic() - start > seconds:
            failed = "bounded_child_time_limit"
        if failed:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break
        time.sleep(0.1)
    stdout, _ = process.communicate()
    if failed or process.returncode:
        return {
            "status": failed or "child_failed",
            "exit_code": process.returncode,
            "observed_peak_RSS_bytes": peak,
            "private_stderr_suppressed": True,
        }
    result = json.loads(stdout)
    result["observed_peak_RSS_bytes"] = peak
    return result


def main(binding):
    import _predictive_serving_mounts as mounts
    import pydicom
    import torch

    assert torch.cuda.device_count() == 0
    assert str(torch.__version__) == binding["torch"]
    assert str(pydicom.__version__) == binding["pydicom"]
    start = time.monotonic()
    train = mounts.resolve_competition_roster("/kaggle/input").parent / "train_series"
    inventory, available = source_inventory(train, start + 300)
    candidates = select(inventory)
    result = dict(
        status="actual_bounded_large_study_preparation_observed",
        binding=binding,
        available_training_studies=available,
        inventoried_studies=len(inventory),
        complete_inventory=len(inventory) == available,
        selected_studies=len(candidates),
        selection="top4_frame_count_union_top4_sampled_matrix_pixel_estimate",
        matrix_estimate_is_not_exact=True,
        studies=[],
        GPU_count=0,
        model_or_label_reads=False,
        clinical_score=None,
        competition_entry=False,
        hidden_error_cause_proven=False,
    )
    output = Path("/kaggle/working/predictive-large-study-CPU.json")
    for rank, row in enumerate(candidates):
        if time.monotonic() - start > LIMIT - 30:
            break
        record = {"selection_rank": rank, **{k: v for k, v in row.items() if k != "path"}}
        results = {}
        # Original then successor on the same real study; page-cache ordering
        # limits causal speed comparisons, but not full tensor equality.
        for variant in ("original", "successor"):
            remaining = min(CHILD_LIMIT, LIMIT - (time.monotonic() - start))
            if remaining < 10:
                results[variant] = {"status": "total_time_bound"}
                continue
            results[variant] = bounded_child({"variant": variant, "path": row["path"]}, remaining)
        hashes = [results[v].pop("context_digest_private", None) for v in ("original", "successor")]
        record.update(
            variants=results,
            original_context_bitwise_equal=(hashes[0] == hashes[1]) if all(hashes) else None,
        )
        result["studies"].append(record)
        result["elapsed_seconds"] = time.monotonic() - start
        output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"phase": "large_study_prepared", "record": record}), flush=True)
    result["elapsed_seconds"] = time.monotonic() - start
    output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
