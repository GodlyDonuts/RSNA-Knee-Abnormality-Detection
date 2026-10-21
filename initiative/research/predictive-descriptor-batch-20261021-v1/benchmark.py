"""Exact original-versus-batched descriptor preparation on three public MRI studies."""

import concurrent.futures as cf
import csv
import dataclasses
import gc
import hashlib
import io
import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path

CAP = 24 * 2**30
LIMIT = 900


def initialize():
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def pack(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Only original CPU tensors")
        return value.detach().contiguous().numpy().copy()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__module__ + "." + type(value).__qualname__,
            "fields": {f.name: pack(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(pack(v) for v in value)
    if isinstance(value, (np.ndarray, str, int, float, bool, type(None))):
        return value
    raise TypeError("Unexpected original context value")


def context_hash(value):
    """Complete structured values, dtypes, shapes and exact tensor bytes."""
    import numpy as np

    digest = hashlib.sha256()

    def visit(v):
        if isinstance(v, np.ndarray):
            digest.update(json.dumps(["array", v.dtype.str, list(v.shape)]).encode())
            digest.update(v.tobytes(order="C"))
        elif isinstance(v, dict):
            digest.update(b"dict")
            for k in sorted(v):
                digest.update(json.dumps(k).encode())
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            digest.update(json.dumps([type(v).__name__, len(v)]).encode())
            for item in v:
                visit(item)
        else:
            digest.update(json.dumps([type(v).__name__, v], allow_nan=False).encode())

    visit(value)
    return digest.hexdigest()


def prepare(task):
    from _predictive_descriptor_cache import cache_descriptors_once as reference
    from descriptor_batch import cache_descriptors_once as batched

    from rsna.competitive.predictive.cases import assemble
    from rsna.competitive.predictive.dicom import load_study

    variant, index, root, uid = task
    assert variant in ("reference", "batched")
    cache_descriptors_once = reference if variant == "reference" else batched
    begin = time.monotonic()
    groups, stats = load_study(Path(root) / "test_series" / uid, uid)
    decoded = time.monotonic()
    groups = cache_descriptors_once(groups)
    described = time.monotonic()
    contexts = assemble(groups, side=320, halo=8, context_cap=24, classes=12)
    assert len(contexts) == 2 and sum(len(c["images"]) for c in contexts) <= 48
    rendered = time.monotonic()
    # Bytes avoid torch's implicit shared-memory reducer (/dev/shm may be small).
    payload = pickle.dumps(pack(contexts), protocol=5)
    if len(payload) > 512 * 2**20:
        raise ValueError("Complete original context exceeds transport bound")
    return {
        "index": index,
        "decode_seconds": decoded - begin,
        "render_seconds": rendered - decoded,
        "descriptor_seconds": described - decoded,
        "assembly_seconds": rendered - described,
        "pack_seconds": time.monotonic() - rendered,
        "stats": stats,
        "payload": payload,
        "payload_bytes": len(payload),
        "worker_pid": os.getpid(),
    }


def trial(kind, workers, tasks, remaining):
    begin = time.monotonic()
    assert kind in ("reference", "batched") and workers == 2
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    rows = []
    try:
        futures = [pool.submit(prepare, (kind, *task)) for task in tasks]
        for future in cf.as_completed(futures, timeout=remaining()):
            row = future.result()
            start = time.monotonic()
            value = pickle.loads(row.pop("payload"))  # Only our local child process bytes.
            row["context_sha256"] = context_hash(value)
            row["unpack_and_hash_seconds"] = time.monotonic() - start
            del value
            rows.append(row)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return {
        "kind": kind,
        "workers": workers,
        "wall_seconds": time.monotonic() - begin,
        "studies": sorted(rows, key=lambda r: r["index"]),
    }


def main(binding):
    import psutil

    initialize()
    import pydicom
    import torch

    assert torch.cuda.device_count() == 0
    assert str(torch.__version__) == binding["torch"]
    assert str(pydicom.__version__) == binding["pydicom"]
    roots = [
        Path("/kaggle/input") / suffix
        for suffix in (
            "rsna-knee-abnormality-detection",
            "competitions/rsna-knee-abnormality-detection",
        )
    ]
    roots = [p for p in roots if (p / "test.csv").is_file()]
    assert len(roots) == 1
    raw = (roots[0] / "test.csv").read_bytes()
    ids = [r["StudyInstanceUID"] for r in csv.DictReader(io.StringIO(raw.decode()))]
    roster = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    assert len(ids) == 3 and len(set(ids)) == 3 and roster == binding["public_roster_sha256"]
    tasks = [(i, str(roots[0]), uid) for i, uid in enumerate(ids)]
    started = time.monotonic()
    done = threading.Event()
    usage = {"peak_process_tree_RSS_bytes": 0}

    def remaining():
        seconds = LIMIT - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError("CPU preparation comparison exhausted its bound")
        return seconds

    def monitor():
        parent = psutil.Process()
        while not done.wait(0.2):
            children = parent.children(recursive=True)
            rss = 0
            for process in [parent, *children]:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            usage["peak_process_tree_RSS_bytes"] = max(usage["peak_process_tree_RSS_bytes"], rss)
            if rss > CAP or time.monotonic() - started > LIMIT:
                for process in reversed(children):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                print(json.dumps({"status": "CPU_limit_exceeded", **usage}), flush=True)
                os._exit(124)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results = []
    try:
        for kind, workers in (
            ("reference", 2),
            ("batched", 2),
            ("reference", 2),
            ("batched", 2),
            ("reference", 2),
        ):
            row = trial(kind, workers, tasks, remaining)
            reference = results[0]["studies"] if results else row["studies"]
            assert len(row["studies"]) == len(reference) == 3
            for actual, original in zip(row["studies"], reference, strict=True):
                assert actual["index"] == original["index"]
                assert actual["context_sha256"] == original["context_sha256"]
                assert actual["stats"] == original["stats"]
            row["complete_original_input_equality"] = True
            results.append(row)
            print(json.dumps({"phase": "CPU_trial_complete", **row}), flush=True)
            gc.collect()
        result = {
            "status": "actual_bitwise_descriptor_batch_preparation_comparison_complete",
            "binding": binding,
            "roster_sha256": roster,
            "studies": 3,
            "trials": results,
            "elapsed_seconds": time.monotonic() - started,
            **usage,
            "memory_cap_bytes": CAP,
            "cpu_count": os.cpu_count(),
            "CPU_affinity": sorted(os.sched_getaffinity(0)),
            "torch": str(torch.__version__),
            "pydicom": str(pydicom.__version__),
            "GPU_count": 0,
            "model_or_label_reads": False,
            "production_pipeline_accepted": False,
            "hidden_runtime_proven": False,
            "submission_csv_created": False,
            "clinical_score": None,
            "limitations": "Three public studies; alternating sequential trials still have cache/host effects. No encoder inference or GPU pipeline overlap was measured. Every full context must be byte identical.",
        }
        Path("/kaggle/working/predictive-descriptor-batch.json").write_text(
            json.dumps(result, indent=2)
        )
        print(
            json.dumps(
                {"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], **usage}
            ),
            flush=True,
        )
    finally:
        done.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
